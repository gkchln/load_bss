import pandas as pd
import numpy as np
import argparse
from tqdm import trange
from PenNMF import PenNMF
from NMF_iter import normalize_curves, initialize_W
from utils import month_to_season
import pickle

# Parameters of solver
tol = 1e-5
max_iter = 10000
verbose = 0

# Parameters for data subset
seasons = ['Spring', 'Fall', 'Winter', 'Summer']


# Matrices for the PenNMF
A = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])

# A = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# with open('data/2_processed/PenNMF/C_init_mean.pkl', 'rb') as file:
#     C_init_mean = pickle.load(file)

def initialize_C_gaussian_prior(C_init_mean, sd=0.02):
    noise = np.random.normal(loc=0, scale=sd, size=C_init_mean.shape)
    C_init = C_init_mean + noise # Add the noise
    return C_init.div(C_init.sum(axis=1), axis=0) # normalize the concentrations

def main(n_components, alpha, train_years, n_runs, init, infile, outfile):
    # Load matrices for BSS
    with open('data/2_processed/PenNMF/B_{}.pkl'.format('_'.join(train_years)), 'rb') as file:
        B = pickle.load(file)
    B = B.loc[B.index.month.map(month_to_season).isin(seasons), B.columns.str[11:13].astype(int).map(month_to_season).isin(seasons)] # HOTFIX: Select only seaons of interest

    Y = pd.read_pickle('data/2_processed/PenNMF/Y_{}.pkl'.format('_'.join(train_years)))
    Y = Y[Y.index.month.map(month_to_season).isin(seasons)] # HOTFIX: Select only period of interest

    # Load matrix X
    input_df = pd.read_csv(infile, index_col=0)
    unit_info = input_df.index.str.extract(r'^(?P<region>[\w.]+)_(?P<year>\d{4})-(?P<month>\d{2})-\d{2}_(?P<daytype>[\w ]+)$').set_index(input_df.index)
    df = pd.concat([input_df, unit_info], axis=1)
    df = df[df.month.astype(int).map(month_to_season).isin(seasons)] # HOTFIX: Select only period of interest
    df = df[df.year.isin(train_years)]
    X = df.drop(unit_info.columns, axis=1)
    X = normalize_curves(X)

    n = len(X)
    p = len(X.columns)

    H_results = np.zeros((n_components, p, n_runs))
    W_results = np.zeros((n, n_components, n_runs))
    iterations = np.zeros(n_runs)
    loss_nmf = np.zeros(n_runs)
    loss_constraint = np.zeros(n_runs)
    estimators = []

    # # HOTFIX: Select subset for C_init_mean:
    # C_init_mean = C_init_mean.loc[df.index, :]

    for i in trange(n_runs):
        # Initialize W matrix with rows uniformely sampled on the simplex(n_components)
        if init == 'uniform':
            C_init = initialize_W(X, n_components)
        # elif init == 'gaussian':
        #     C_init = initialize_C_gaussian_prior(C_init_mean)
        S_init = normalize_curves(np.ones((n_components, p)))

            # Specify NMF model
        model = PenNMF(
            n_components=n_components,
            alpha=alpha,
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
            B.values
        )

        S = model.components_

        # Store solution in the results tensors
        H_results[...,i] = S
        W_results[...,i] = C
        iterations[i] = model.n_iter_
        loss_nmf[i] = model.losses_nmf_[-1]
        loss_constraint[i] = model.losses_constraint_[-1]
        estimators.append(model)

    np.savez(outfile, H_results=H_results, W_results=W_results, iterations=iterations, loss_nmf=loss_nmf, loss_constraint=loss_constraint)
    print("Saved results at", outfile)
    estimators_outfile = outfile.replace('.npz', '.pkl')
    with open( estimators_outfile, 'wb') as file:
        pickle.dump(estimators, file)
    print("Saved estimators at",  estimators_outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMF for daily load curves")
    parser.add_argument("--n_comp", dest="n_components", type=int, help="Number of components")
    parser.add_argument("--alpha", dest="alpha", type=float, help="alpha parameter")
    parser.add_argument("--train_years", dest="train_years", type=str, nargs='+', help="Output file for the NMF results")
    parser.add_argument("--n_runs", dest="n_runs", type=int, help="Number of runs")
    parser.add_argument("--init", dest="init", type=str, help="Initialization mode, 'uniform' or 'gaussian'")
    parser.add_argument("--infile", dest="infile", type=str, help="Input file for the daily load curves")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the NMF results")
    args = parser.parse_args()
    main(args.n_components, args.alpha, args.train_years, args.n_runs, args.init,  args.infile, args.outfile)
