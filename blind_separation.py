# %% Imports
import pandas as pd
import numpy as np
import argparse
from tqdm import trange
from lcnmf import LCNMF
from utils import normalize_curves, functional_norm
from utils import month_to_season
from scipy.linalg import block_diag
import pickle
import logging
import locale

 # HOTFIX for parsing IMCEI file
locale.setlocale(locale.LC_ALL, 'it_IT')

# Logging paramneters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

consumption_file = 'data/1_input/consumption/consumption.xlsx'
imser_file = 'data/1_input/indicators/IMSER.xlsx'
imcei_file = 'data/1_input/indicators/IMCEI_mensile.xlsx'
enel_files = [
    'data/1_input/indicators/Domestico 2020 e 2021.xlsx', 
    'data/1_input/indicators/Domestico 2022 e 2023.xlsx'
]


# %%
def get_curves_info(curves_df):
    unit_info = curves_df.index.str.extract(r'^(?P<region>[\w.]+)_(?P<year>\d{4})-(?P<month>\d{2})-\d{2}_(?P<daytype>[\w ]+)$').set_index(curves_df.index)
    h = 24 / (curves_df.shape[1]-1)
    unit_info['cons'] = curves_df.apply(lambda row: functional_norm(row.values, h), axis=1)
    unit_info['season'] = unit_info.month.astype(int).map(month_to_season)
    return unit_info

# %% Functions to process consumption and indicators
def process_sector_consumption(infile, year_total):
    """Read and process annual sector consumption data"""
    cons_df = pd.read_excel(infile, index_col=0)
    cons_df['Industria'] = cons_df['Industria'] + cons_df['Agricoltura'] # We put agricoltura sector in the industrial one
    cons_df.drop('Agricoltura', axis=1, inplace=True)
    cons_df.rename({'Servizi': 'Services', 'Industria': 'Industry', 'Domestico': 'Domestic'}, axis=1, inplace=True)
    # Align with the consumption computed from the load data
    cons_df = cons_df[cons_df.index.isin(year_total.index.astype(int))] # Get only years in the years we have for the totals
    uncorrected_total = cons_df[['Services', 'Industry', 'Domestic']].sum(axis=1).values
    corrected_total = year_total.values
    cons_df = cons_df.mul(corrected_total / uncorrected_total, axis=0)
    return cons_df

def get_indicators(imser_file, imcei_file, enel_files):
    """Read indicator files and merge them in the same monthly dataframe"""
    imser = pd.read_excel(imser_file)
    imser = imser.set_index('Mese').rename({'IMSER (GWh)': 'IMSER'}, axis=1) # We don't have data for this year

    imcei = pd.read_excel(imcei_file)
    imcei['Mese'] = pd.to_datetime(imcei['Anno'].astype(str) + '-' + imcei['Mese'], format='%Y-%b')
    imcei = imcei.set_index('Mese').drop('Anno', axis=1).rename({'IMCEI Mensile': 'IMCEI'}, axis=1)

    enel = pd.concat([pd.read_excel(file, skiprows=12, usecols=range(3)) for file in enel_files], ignore_index=True)
    enel = enel.set_index('Mese').drop('Domestico kWh', axis=1)

    indics = imcei.copy()
    indics['IMSER'] = imser['IMSER']
    indics['Enel'] = enel['Domestico GWh']
    indics = indics[indics.index >= "2020-01-01"]
    
    return indics

# %% Function to get matrix Y in the LCNMF
def get_Y(indic_df, cons_df, year_month_totals):
    indics = indic_df.copy()
    indics.rename(columns={'IMSER': 'Services', 'IMCEI': 'Industry', 'Enel': 'Domestic'}, inplace=True)
    indics['year'] = indics.index.year
    # Get the years of cons_df
    indics = indics[indics.year.isin(cons_df.index.unique())]
    indic_totals = indics[['year']].merge(indics.groupby('year').sum(), left_on='year', right_index=True, how='left').drop('year', axis=1)
    cons_rescaling = indics[['year']].merge(cons_df, left_on='year', right_index=True, how='left').drop('year', axis=1)
    indics.drop('year', axis=1, inplace=True)
    Y = indics.div(indic_totals).mul(cons_rescaling)
    # Realign with load data at month level (since it is not guaranteed anymore at the month level after the breakdown along months)
    uncorrected_total = Y[['Domestic', 'Industry', 'Services']].sum(axis=1).values
    corrected_total = year_month_totals.values
    Y = Y.mul(corrected_total / uncorrected_total, axis=0)
    return Y

# %% Function to get matrix B in the LCNMF
def get_B(curves_df):
    h = 24 / (curves_df.shape[1]-1)
    norm_X = curves_df.apply(lambda row: functional_norm(row, h), axis=1, raw=True)
    norm_X = pd.DataFrame(norm_X, index=norm_X.index, columns=['norm_X'])
    norm_X['month'] = norm_X.index.str[6:13]
    months = norm_X['month'].unique()
    m = len(months) # n_months
    n = len(curves_df) # n_days
    blocks = [norm_X.loc[norm_X.month == month, 'norm_X'].values[np.newaxis, :] for month in months]
    B = block_diag(*blocks)
    B = pd.DataFrame(B, index=months, columns=norm_X.index)
    return B

# %% Function to get matrix D in the LCNMF
# Creating matrix D is a little more complicated as the multiplication of each row of S with the column matrix D should give the functional norm
# The lines below essentially consist in writing the functional_norm as a vectorized function
def get_D(p):
    D = np.ones((p, 1))
    D[0, 0] = 0.5
    D[-1, 0] = 0.5
    h = 24 / (p-1)
    D = h * D
    return D

# %%
# Columns correspond respectively to Domestic, Industry, Services
A = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])


# %% Parameters of solver
tol = 1e-5
max_iter = 10000
verbose = 0

def initialize_C(X, n_components, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)  # Set the random seed for reproducibility
    C = pd.DataFrame(
        np.random.rand(len(X), n_components), 
        index=X.index, 
        columns=[f"Component {k+1}" for k in range(n_components)]
    )
    C = C.div(C.sum(axis=1), axis=0)
    return C

def main(n_components, alpha, beta, n_runs, infile, outfile, random_state):
    logging.info("Reading input files and creating matrices")
    curves_df = pd.read_csv(infile, index_col=0)
    curves_info = get_curves_info(curves_df)
    year_totals = curves_info.groupby('year')['cons'].sum()
    year_month_totals = curves_info.groupby(['year', 'month'])['cons'].sum()

    cons_df = process_sector_consumption(consumption_file, year_totals)
    indics_df = get_indicators(imser_file, imcei_file, enel_files)

    X = normalize_curves(curves_df)
    
    n = len(X)
    p = len(X.columns)

    B = get_B(curves_df)
    Y = get_Y(indics_df, cons_df, year_month_totals)

    E = np.eye(n_components)
    Z = np.ones((n_components, 1))

    # Creating matrix D is a little more complicated as the multiplication of each row of S with the column matrix D should give the functional norm
    # The lines below essentially consist in writing the functional_norm as a vectorized function
    D = get_D(p)

    S_results = np.zeros((n_components, p, n_runs))
    C_results = np.zeros((n, n_components, n_runs))
    iterations = np.zeros(n_runs)
    loss_nmf = np.zeros(n_runs)
    loss_constraint_c = np.zeros(n_runs)
    loss_constraint_s = np.zeros(n_runs)
    estimators = []

    logging.info("Start running the {} monte-carlo simulations of the blind separation...".format(n_runs))
    for i in trange(n_runs):
        # Initialize C matrix with rows uniformely sampled on the simplex(n_components)
        # The complex random_state specification is to avoid to have replicate draws of C_init with two different values of random_state
        # (the one defined in the main() function) e.g. if main(..., n_runs=1000, random_state=42) is called, 1000 calls
        # initialize_C(..., random_state=42000), initialize_C(..., random_state=42001), ..., initialize_C(..., random_state=42999)
        # will be performed and none of them will be overlap with the calls initialize_C(..., random_state=43000),
        # initialize_C(..., random_state=43001), ..., initialize_C(..., random_state=43999) if main(..., n_runs=1000, random_state=43) is called
        C_init = initialize_C(X, n_components, random_state=n_runs * random_state + i)
        # Initialise S matrix with equal values
        S_init = normalize_curves(np.ones((n_components, p)))

        # Specify LCNMF model
        model = LCNMF(
            n_components=n_components,
            alpha=alpha,
            beta=beta,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose
        )

        # Run the solving algorithm
        C = model.fit_transform(
            X.values,
            C_init.values,
            S_init,
            Y.values,
            A,
            B.values,
            Z,
            D,
            E
        )

        S = model.components_

        # Store solution in the results tensors
        S_results[...,i] = S
        C_results[...,i] = C
        iterations[i] = model.n_iter_
        loss_nmf[i] = model.losses_nmf_[-1]
        loss_constraint_c[i] = model.losses_constraint_c_[-1]
        loss_constraint_s[i] = model.losses_constraint_s_[-1]
        estimators.append(model)

    # Save the results tensor
    np.savez(outfile, S_results=S_results, C_results=C_results, iterations=iterations, loss_nmf=loss_nmf, loss_constraint_c=loss_constraint_c, loss_constraint_s=loss_constraint_s)
    logging.info("Saved results at {}".format(outfile))
    # And save the estimators
    estimators_outfile = outfile.replace('.npz', '.pkl')
    with open(estimators_outfile, 'wb') as file:
        pickle.dump(estimators, file)
    logging.info("Saved estimators at {}".format(estimators_outfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run LCNMF for daily load curves")
    parser.add_argument("--n_comp", dest="n_components", type=int, help="Number of components")
    parser.add_argument("--alpha", dest="alpha", type=float, help="alpha parameter")
    parser.add_argument("--beta", dest="beta", type=float, help="beta parameter")
    parser.add_argument("--n_runs", dest="n_runs", type=int, help="Number of runs")
    parser.add_argument("--infile", dest="infile", type=str, help="Input file for the daily load curves")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the LCNMF results")
    parser.add_argument("--random_state", dest="random_state", type=int, default=None, help="Seed for random number generator (default: None)")
    args = parser.parse_args()
    main(args.n_components, args.alpha, args.beta, args.n_runs, args.infile, args.outfile, args.random_state)

# %%
