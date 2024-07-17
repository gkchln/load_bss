import pandas as pd
import numpy as np
import argparse
from tqdm import trange
from PenNMF import PenNMF
from NMF_iter import normalize_curves, initialize_W
import pickle

# Parameters of solver
tol = 1e-5
max_iter = 10000
verbose = 0
alpha = 5e-10

# Parameters for training period
start_year = 2020
end_year = 2022

# Matrices for the PenNMF
# A = np.array([
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1]
# ])

A = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

with open('data/2_processed/PenNMF/B.pkl', 'rb') as file:
    B = pickle.load(file)

Y = pd.read_pickle('data/2_processed/PenNMF/Y_train.pkl')

def main(n_components, n_runs, infile, outfile):
    # Load matrix X
    input_df = pd.read_csv(infile, index_col=0)
    unit_info = input_df.index.str.extract(r'^(?P<year>\d{4})-(?P<month>\d{2})-\d{2}_(?P<daytype>[\w ]+)$').set_index(input_df.index)
    df = pd.concat([input_df, unit_info], axis=1)
    df = df[(df.year.astype(int) >= start_year) & (df.year.astype(int) <= end_year)]
    X = df.drop(unit_info.columns, axis=1)
    X = normalize_curves(X)

    # Specify NMF model
    model = PenNMF(
        n_components=n_components,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    n = len(X)
    p = len(X.columns)

    H_results = np.zeros((n_components, p, n_runs))
    W_results = np.zeros((n, n_components, n_runs))
    iterations = np.zeros(n_runs)
    loss_nmf = np.zeros(n_runs)
    loss_constraint = np.zeros(n_runs)

    for i in trange(n_runs):
        # Initialize W matrix with rows uniformely sampled on the simplex(n_components)
        C_init = initialize_W(X, n_components)
        S_init = normalize_curves(np.ones((n_components, p)))

        # Run the solving algorithm
        C = model.fit_transform(
            X.values,
            C_init.values,
            S_init,
            Y.values,
            A,
            B
        )

        S = model.components_

        # Store solution in the results tensors
        H_results[...,i] = S
        W_results[...,i] = C
        iterations[i] = model.n_iter_
        loss_nmf[i] = model.losses_nmf_[-1]
        loss_constraint[i] = alpha * model.losses_constraint_[-1]

    np.savez(outfile, H_results=H_results, W_results=W_results, iterations=iterations, loss_nmf=loss_nmf, loss_constraint=loss_constraint)
    print("Saved results at", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMF for daily load curves")
    parser.add_argument("--n_comp", dest="n_components", type=int, help="Number of components")
    parser.add_argument("--n_runs", dest="n_runs", type=int, help="Number of runs")
    parser.add_argument("--infile", dest="infile", type=str, help="Input file for the daily load curves")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the NMF results")
    args = parser.parse_args()
    main(args.n_components, args.n_runs, args.infile, args.outfile)
