import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import cdist, euclidean

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