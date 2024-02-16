import pandas as pd
import numpy as np
import argparse
from tqdm import trange
from sklearn.decomposition import NMF

# Parameters of solver
random_state = None
tol = 1e-5
init = 'custom'
max_iter = 10000
solver = 'mu'
verbose = 0

def functional_norm(y, h):
    n = len(y) - 1
    res = 0
    for i in range(n):
        res += y[i] + y[i+1]
    return res * h / 2

def normalize_curves(data):
    """Normalize curves in a dataframe or 2-dimensional array."""
    if isinstance(data, pd.DataFrame):
        h = 24 / (data.shape[1]-1)
        norm_data = data.apply(lambda row: functional_norm(row, h), axis=1, raw=True)
        return data.div(norm_data, axis=0)
    elif isinstance(data, np.ndarray):
        h = 24 / (data.shape[1]-1)
        norm_data = np.apply_along_axis(lambda row: functional_norm(row, h), axis=1, arr=data)
        return data / norm_data[:, np.newaxis]
    else:
        raise ValueError("Input must be either a DataFrame or a 2-dimensional numpy array.")

def initialize_W(X, n_components):
    W = pd.DataFrame(np.random.rand(len(X), n_components), index=X.index, columns=[f"Component {k+1}" for k in range(n_components)])
    W = W.div(W.sum(axis=1), axis=0)
    return W

def main(n_components, n_runs, infile, outfile):
    # Load matrix X
    input_df = pd.read_csv(infile, index_col=0)
    unit_info = input_df.index.str.extract(r'^(?P<region>[\w.]+)_(?P<year>\d{4})-(?P<month>\d{2})-\d{2}_(?P<daytype>[\w ]+)$').set_index(input_df.index)
    df = pd.concat([input_df, unit_info], axis=1)
    X = df.drop(unit_info.columns, axis=1)
    X = normalize_curves(X)

    # Specify NMF model
    model = NMF(
        n_components=n_components,
        random_state=random_state,
        solver=solver,
        tol=tol,
        init=init,
        max_iter=max_iter,
        verbose=verbose
    )

    n = len(X)
    p = len(X.columns)

    H_results = np.zeros((n_components, p, n_runs))
    W_results = np.zeros((n, n_components, n_runs))
    iterations = np.zeros(n_runs)
    errors = np.zeros(n_runs)

    for i in trange(n_runs):
        # Initialize W matrix with rows uniformely sampled on the simplex(n_components)
        W_init = initialize_W(X, n_components)
        H_init = normalize_curves(np.ones((n_components, p)))

        # Run the solving algorithm
        W = model.fit_transform(
            X.values,
            W=W_init.values.copy(order='C'),
            H=H_init
        )
        H = model.components_

        # Store solution in the results tensors
        H_results[...,i] = H
        W_results[...,i] = W
        iterations[i] = model.n_iter_
        errors[i] = model.reconstruction_err_

    np.savez(outfile, H_results=H_results, W_results=W_results, iterations=iterations, errors=errors)
    print("Saved results at", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMF for daily load curves")
    parser.add_argument("--n_comp", dest="n_components", type=int, help="Number of components")
    parser.add_argument("--n_runs", dest="n_runs", type=int, help="Number of runs")
    parser.add_argument("--infile", dest="infile", type=str, help="Input file for the daily load curves")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the NMF results")
    args = parser.parse_args()
    main(args.n_components, args.n_runs, args.infile, args.outfile)
