import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import cdist, euclidean
import warnings
import matplotlib.pyplot as plt

### Calendar utils ###
weekday_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

month_to_season = {
    1: "Winter",
    2: "Winter",
    3: "Winter",
    4: "Spring",
    5: "Spring",
    6: "Spring",
    7: "Summer",
    8: "Summer",
    9: "Summer",
    10: "Fall",
    11: "Fall",
    12: "Fall"
}

### Functions ###
def get_string_color(color, opacity=1):
    r, g, b = tuple(int(value * 255) for value in color)
    return f'rgba({r}, {g}, {b}, {opacity})'

def geometric_median(X, eps=1e-5):
    """
    Compute the geometric median of a set of points using an iterative algorithm.
    Taken from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points (accepted answer)

    Parameters:
    - points: An array-like object containing the input points.
    - epsilon: Tolerance for convergence.

    Returns:
    - The geometric median point.
    """
    # Initialize the current estimate of the geometric median as the mean of the input points
    y = np.mean(X, 0)

    # Iteratively refine the estimate
    while True:
        # Compute the distances from each point to the current estimate
        D = cdist(X, [y])

        # Identify non-zero distances to avoid division by zero
        nonzeros = (D != 0)[:, 0]

        # Compute the inverse distances for non-zero elements
        Dinv = 1 / D[nonzeros]

        # Compute the sum of inverse distances
        Dinvs = np.sum(Dinv)

        # Compute weights based on inverse distances
        W = Dinv / Dinvs

        # Compute the weighted sum of points
        T = np.sum(W * X[nonzeros], 0)

        # Count the number of zero and non-zero distances
        num_zeros = len(X) - np.sum(nonzeros)

        # Update the estimate of the geometric median based on the number of zero distances
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            # If all distances are zero, the geometric median is found
            return y
        else:
            # Update the estimate using a weighted combination of the current estimate and the sum of points
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        # Check for convergence based on Euclidean distance between current and updated estimates
        if euclidean(y, y1) < eps:
            return y1

        # Update the current estimate for the next iteration
        y = y1


def geometric_medoid(X, *kwargs):
    """
    Compute the geometric medoid of a set of points using geometric median.
    
    Parameters:
    - X: An array-like object containing the input points.

    Returns:
    - The geometric medoid point.
    """
    geomedian = geometric_median(X)
    argmin = pairwise_distances_argmin(np.array([geomedian]), X)[0]
    return X[argmin,:]

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array-like
        The input array which could be either be a vector or a 2 dimensional array.

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order="K")
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn(
            (
                "Array type is integer, np.dot may overflow. "
                "Data should be float type to avoid this issue"
            ),
            UserWarning,
        )
    return np.dot(x, x)


# Exceptions  used in PenNMF
class NotFittedError(ValueError, AttributeError):
    pass

class ConvergenceWarning(UserWarning):
    pass

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

def initialize_C(X, n_components):
    C = pd.DataFrame(np.random.rand(len(X), n_components), index=X.index, columns=[f"Component {k+1}" for k in range(n_components)])
    C = C.div(C.sum(axis=1), axis=0)
    return C

def plot_components(H, ax=None, figsize=(10, 6), labels=None, emphasize_comp=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_components = H.shape[0]
    abscissa = np.linspace(0, 24, H.shape[1])
    for k in range(n_components):
        if labels:
            label = labels[k]
        else:
            label = f'Component {k + 1}'
        if emphasize_comp and emphasize_comp != k + 1:
            alpha = 0.2
        else:
            alpha = 1
        ax.plot(abscissa, H[k, :], linestyle='-', label=label, alpha=alpha, **kwargs)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Normalized Load')
    # ax.set_title(title)
    ax.legend(loc='upper left')
    
    return fig, ax