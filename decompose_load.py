# %%%
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import pickle
import argparse
from utils import functional_norm, normalize_curves

# Logging paramneters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

random_state = 42

# %%
def non_blind_separation(X, model_ens, random_state=42):
    K = model_ens[0].n_components
    N = len(model_ens)
    n, p = X.shape
    C_ens = np.zeros((n, K, N))
    for run, model in enumerate(tqdm(model_ens)):
        C_ens[..., run] = model.transform(X, random_state=random_state)
    return C_ens


# %%%
def decompose_load(C_ens, S_ens, E):
    n, K, N = C_ens.shape
    p = S_ens.shape[1]
    norms = E.values.reshape(n, 1, 1, 1)
    load = norms * C_ens[..., np.newaxis, :] * np.stack([S_ens] * n, axis=0)

    # Here we reshape load so that we have run changing first, then, hour and finally unit
    load = pd.DataFrame(load.transpose((0, 3, 2, 1)).reshape((n * N * p, K)), columns=[f'$S_{k+1}$' for k in range(K)])
    load['model'] = np.tile(np.repeat(np.arange(N), p), n)
    load['hour'] = np.tile(np.linspace(0, 24, p), N * n)
    # Add the datetime and the zone
    parsed_idx = E.index.str.split('_')
    zone = parsed_idx.str[0]
    date = parsed_idx.str[1]
    load['obs'] = np.repeat(E.index, N * p)
    load['datetime'] = pd.to_datetime(np.repeat(date, N * p)) + pd.to_timedelta(load['hour'], unit='h')
    load['zone'] = np.repeat(zone, N * p)

    return load

# %%
def decompose_daily_consumption(C_ens, E):
    n, K, N = C_ens.shape
    df_list = []

    for b in range(N):
        C = C_ens[..., b]

        consumptions = (E.to_numpy()[:, np.newaxis] * C)
        consumptions = pd.DataFrame(consumptions, columns=[f'$S_{k+1}$' for k in range(K)])
        consumptions['model'] = b

        parsed_idx = E.index.str.split('_')
        consumptions['date'] = pd.to_datetime(parsed_idx.str[1], format='%Y-%m-%d')
        consumptions['zone'] = parsed_idx.str[0]

        df_list.append(consumptions)

    return pd.concat(df_list, ignore_index=True).sort_values(['date', 'model']).reset_index(drop=True)

# %%
def get_monthly_sectors_consumption(input_df, model_ens):
    K = model_ens[0].n_components
    N = len(model_ens)
    n, p = input_df.shape
    h = 24 / (p - 1)
    X = normalize_curves(input_df)
    E = input_df.apply(lambda row: functional_norm(row, h), axis=1, raw=True)

    logging.info(f"Estimating the sources concentrations according to the {N} models...")
    C_ens = non_blind_separation(X, model_ens)

    logging.info("Using the concentrations to disaggregate daily consumptions...")
    E_k = decompose_daily_consumption(C_ens, E)
    logging.info("Done.")

    E_k['month'] = pd.to_datetime(E_k['date']).dt.to_period('M')

    monthly = E_k.groupby(['month', 'zone', 'model'], as_index=False, sort=False)[[f'$S_{k+1}$' for k in range(K)]].sum()
    
    monthly['Domestic'] = monthly['$S_1$'] + monthly['$S_2$']
    monthly['Industry'] = monthly['$S_3$']
    monthly['Services'] = monthly['$S_4$'] + monthly['$S_5$']

    monthly.drop(columns=[f'$S_{k+1}$' for k in range(K)], inplace=True)
    
    return monthly

# %%
def main(infile_load, infile_models, outfile):
    input_df = pd.read_csv(infile_load, index_col=0)
    with open(infile_models, 'rb') as file:
        model_ens = pickle.load(file)
    out_df = get_monthly_sectors_consumption(input_df, model_ens)
    out_df.to_csv(outfile, index=False)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate sectors monthly consumption from daily load curves and ensemble of LCNMF models")
    parser.add_argument("--infile_load", dest="infile_load", type=str, help="Input file for the daily load curves")
    parser.add_argument("--infile_models", dest="infile_models", type=str, help="Input pickle file containing the LCNMF models")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the ensemble monthly sectors consumption")
    args = parser.parse_args()
    main(args.infile_load, args.infile_models, args.outfile)

