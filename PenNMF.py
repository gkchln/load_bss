###########################################################
####### Penalized Non-negative Matrix Factorisation #######
###########################################################
import warnings
import time
import numpy as np
from numpy.linalg import multi_dot
from utils import ConvergenceWarning, NotFittedError, squared_norm

EPSILON = np.finfo(np.float32).eps

def _multiplicative_update_s(X, C, S):
    """Performs the multiplicative update of matrix S

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p

    Returns:
        2D array: Updated matrix S
    """
    numerator = np.dot(C.T, X)
    denominator = np.linalg.multi_dot([C.T, C, S])
    denominator[denominator == 0] = EPSILON
    update_factor = numerator / denominator
    S *= update_factor
    
    return S

def _multiplicative_update_c(X, C, S, Y, A, B, alpha):
    """Performs the multiplicative update of matrix S

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p
        Y (2D array): Matrix Y of shape m x g
        A (2D array): Matrix A of shape K x g
        B (2D array): Matrix B of shape m x n
        alpha (float): regularisation parameter

    Returns:
        2D array: Updated matrix C
    """
    numerator = np.dot(X, S.T) + alpha * np.linalg.multi_dot([B.T, Y, A.T])
    denominator = np.linalg.multi_dot([C, S, S.T]) + alpha * np.linalg.multi_dot([B.T, B, C, A, A.T])
    denominator[denominator == 0] = EPSILON
    update_factor = numerator / denominator
    C *= update_factor
    
    return C

def _loss_nmf(X, C, S):
    """Compute the first part of the loss corresponding to the non-negative matrix factorization

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p

    Returns:
        float: loss value ||X - CS||^2
    """
    return squared_norm(X - np.dot(C, S))

def _loss_constraint(Y, B, C, A):
    """Compute the second part of the loss corresponding to the constraint on C

    Args:
        Y (2D array): Matrix Y of shape m x g
        B (2D array): Matrix B of shape m x n
        C (2D array): Matrix C of shape n x K
        A (2D array): Matrix A of shape K x g

    Returns:
        float: loss value ||Y - BCA||^2
    """
    return squared_norm(Y - multi_dot([B, C, A]))

def _loss(X, C, S, Y, A, B, alpha):
    """Compute the global loss as the sum of the loss related to the NMF approximation and the loss related to the constraint on C

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p
        Y (2D array): Matrix Y of shape m x g
        A (2D array): Matrix A of shape K x g
        B (2D array): Matrix B of shape m x n
        alpha (float): regularisation parameter
    
    Returns:
        float: loss value ||X - CS||^2 + alpha * ||Y - BCA||^2
    """
    return _loss_nmf(X, C, S) + alpha * _loss_constraint(Y, B, C, A)

def _fit_transform(X, C_init, S_init, Y, A, B, alpha, max_iter, tol, fit=True, return_loss_sequence=True, verbose=0):
        """Fit the penalized multiplicative update procedure to provided matrices or
        compute the concentrations of fitted sources S for a new input matrix X (in this
        case we take the loss without the constraint on C)

        Args:
            X (2D array): Matrix X of shape n x p
            C (2D array): Matrix C of shape n x K
            S (2D array): Matrix S of shape K x p
            Y (2D array): Matrix Y of shape m x g
            A (2D array): Matrix A of shape K x g
            B (2D array): Matrix B of shape m x n
            alpha (float): Regularisation parameter
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            fit (bool): Whether to fit or to transform only (in case of tranform, S
                        will not be updated and the constraint term in the loss will
                        not be considered). Default it True.
            verbose (int): Verbose level

        Returns:
            C (2D array): Matrix C of shape n x K. Fitted concentrations.
            S (2D array): Matrix S of shape K x p. Fitted profiles.
            n_iter (int): Actual number of iterations.
        """
        
        start_time = time.time()

        loss_nmf = _loss_nmf(X, C_init, S_init)
        loss_constraint = _loss_constraint(Y, B, C_init, A)
        # We store the losses in case we want to return them for diagnostic
        losses_nmf = [loss_nmf]
        losses_constraint  = [loss_constraint]

        error_at_init = loss_nmf + alpha * loss_constraint
        previous_error = error_at_init

        C, S = C_init.copy(), S_init.copy()

        for n_iter in range(1, max_iter + 1):
            C = _multiplicative_update_c(X, C, S, Y, A, B, alpha)
            if fit:
                S = _multiplicative_update_s(X, C, S)

            if tol > 0 and n_iter % 10 == 0:
                loss_nmf = _loss_nmf(X, C, S)
                loss_constraint = _loss_constraint(Y, B, C, A)
                losses_nmf.append(loss_nmf)
                losses_constraint.append(loss_constraint)

                error = loss_nmf + alpha * loss_constraint

                if verbose:
                    iter_time = time.time()
                    print(
                        "Epoch %02d reached after %.3f seconds, error: %f"
                        % (n_iter, iter_time - start_time, error)
                    )

                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error

        # do not print if we have already printed in the convergence test
        if verbose and (tol == 0 or n_iter % 10 != 0):
            end_time = time.time()
            print(
                "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
            )
        
        if n_iter == max_iter and tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % max_iter,
                ConvergenceWarning,
            )
        if return_loss_sequence:
            return C, S, n_iter, losses_nmf, losses_constraint
        else:
            return C, S, n_iter

class PenNMF:
    """
    Penalized Non-negative Matrix Factorization:

        C,S = argmin ||X-CS||^2 + alpha * ||BCA - Y||^2

    """
    def __init__(self, n_components, alpha, tol=1e-4, max_iter=200, verbose=0):
        self.n_components = n_components
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit_transform(self, X, C_init, S_init, Y, A, B):
        """Fit the penalized multiplicative update procedure to provided matrices

        Args:
            X (2D array): Matrix X of shape n x p
            C (2D array): Matrix C of shape n x K
            S (2D array): Matrix S of shape K x p
            Y (2D array): Matrix Y of shape m x g
            A (2D array): Matrix A of shape K x g
            B (2D array): Matrix B of shape m x n

        Returns:
            C (2D array): Matrix C of shape n x K. Fitted concentrations.
        """
        C, S, n_iter, losses_nmf, losses_constraint = _fit_transform(X, C_init, S_init, Y, A, B, self.alpha, self.max_iter, self.tol, verbose=self.verbose)

        self.n_components_ = S.shape[0]
        self.components_ = S
        self.n_iter_ = n_iter
        self.losses_nmf_ = losses_nmf
        self.losses_constraint_ = losses_constraint

        return C
    
    
    def transform(self, X, C_init):
        """Compute the concentrations associated to a new data matrix X, with learnt profiles S.

        Args:
            X (2D array): The new data matrix for which the concentrations must be estimated
        """
        if not hasattr(self, "components_"):
            raise NotFittedError("The PenNMF instance is not fitted yet. Call 'fit_transform' with "
                                 "appropriate arguments before using transform.")
        
        # In this case we just want to estimate the concentrations
        # we do not consider the contraint on C
        Y, A, B, alpha = 0, 0, 0, 0
        
        C, *_ = _fit_transform(X, C_init, self.components_, Y, A, B, alpha, self.max_iter, self.tol, fit=False, verbose=self.verbose)

        return C