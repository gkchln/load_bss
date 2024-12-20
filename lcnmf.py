##############################################################################
####### Linearly-Constrained Non-negative Matrix Factorisation (LCNMF) #######
##############################################################################

import warnings
import time
import numpy as np
from numpy.linalg import multi_dot
from utils import squared_norm

class NotFittedError(ValueError, AttributeError):
    pass

class ConvergenceWarning(UserWarning):
    pass

EPSILON = np.finfo(np.float32).eps

def _multiplicative_update_s(X, C, S, Z, D, E, beta):
    """Performs the multiplicative update of matrix S

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p
        Z (2D array): Matrix Z of shape l x q
        D (2D array): Matrix D of shape p x q
        E (2D array): Matrix E of shape l x K
        beta (float): regularisation parameter

    Returns:
        2D array: Updated matrix S
    """
    numerator = np.dot(C.T, X) + beta * np.linalg.multi_dot([E.T, Z, D.T])
    denominator = np.linalg.multi_dot([C.T, C, S]) + beta * np.linalg.multi_dot([E.T, E, S, D, D.T])
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

def _loss_constraint_c(Y, B, C, A):
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

def _loss_constraint_s(Z, E, S, D):
    """Compute the third part of the loss corresponding to the constraint on S

    Args:
        Z (2D array): Matrix Z of shape l x q
        E (2D array): Matrix B of shape l x K
        S (2D array): Matrix C of shape K x p
        D (2D array): Matrix A of shape p x q

    Returns:
        float: loss value ||Z - ESD||^2
    """
    return squared_norm(Z - multi_dot([E, S, D]))

def _loss(X, C, S, Y, A, B, Z, D, E, alpha, beta):
    """Compute the global loss as the sum of the loss related to the NMF approximation and the loss related to the constraints on C and S

    Args:
        X (2D array): Matrix X of shape n x p
        C (2D array): Matrix C of shape n x K
        S (2D array): Matrix S of shape K x p
        Y (2D array): Matrix Y of shape m x g
        A (2D array): Matrix A of shape K x g
        B (2D array): Matrix B of shape m x n
        Z (2D array): Matrix Z of shape l x q
        D (2D array): Matrix D of shape p x q
        E (2D array): Matrix E of shape l x K
        alpha (float): regularisation parameter for C
        beta (float): regularisation parameter for S
    
    Returns:
        float: loss value ||X - CS||^2 + alpha * ||Y - BCA||^2
    """
    return _loss_nmf(X, C, S) + alpha * _loss_constraint_c(Y, B, C, A) + beta * _loss_constraint_s(Z, E, S, D)

def _fit_transform(X, C_init, S_init, Y, A, B, Z, D, E, alpha, beta, max_iter, tol, fit=True, return_loss_sequence=True, verbose=0):
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
            Z (2D array): Matrix Z of shape l x q
            D (2D array): Matrix D of shape p x q
            E (2D array): Matrix E of shape l x K
            alpha (float): Regularisation parameter for C
            beta (float): regularisation parameter for S
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            fit (bool): Whether to fit or to transform only (in case of tranform, S
                        will not be updated and the constraint terms in the loss will
                        not be considered). Default it True.
            verbose (int): Verbose level

        Returns:
            C (2D array): Matrix C of shape n x K. Fitted concentrations.
            S (2D array): Matrix S of shape K x p. Fitted profiles.
            n_iter (int): Actual number of iterations.
        """
        
        start_time = time.time()

        loss_nmf = _loss_nmf(X, C_init, S_init)
        loss_constraint_c = _loss_constraint_c(Y, B, C_init, A)
        loss_constraint_s = _loss_constraint_s(Z, E, S_init, D)
        # We store the losses in case we want to return them for diagnostic
        losses_nmf = [loss_nmf]
        losses_constraint_c  = [loss_constraint_c]
        losses_constraint_s  = [loss_constraint_s]

        error_at_init = loss_nmf + alpha * loss_constraint_c + beta * loss_constraint_s
        previous_error = error_at_init

        C, S = C_init.copy(), S_init.copy()

        for n_iter in range(1, max_iter + 1):
            C = _multiplicative_update_c(X, C, S, Y, A, B, alpha)
            if fit:
                S = _multiplicative_update_s(X, C, S, Z, D, E, beta)

            if tol > 0 and n_iter % 10 == 0:
                loss_nmf = _loss_nmf(X, C, S)
                loss_constraint_c = _loss_constraint_c(Y, B, C, A)
                loss_constraint_s = _loss_constraint_s(Z, E, S, D)
                losses_nmf.append(loss_nmf)
                losses_constraint_c.append(loss_constraint_c)
                losses_constraint_s.append(loss_constraint_s)

                error = loss_nmf + alpha * loss_constraint_c + beta * loss_constraint_s

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
            return C, S, n_iter, losses_nmf, losses_constraint_c, losses_constraint_s
        else:
            return C, S, n_iter

class LCNMF:
    """
    Penalized Non-negative Matrix Factorization:

        C,S = argmin ||X-CS||^2 + alpha * ||BCA - Y||^2 + beta * ||ESD - Z||^2

    """
    def __init__(self, n_components, alpha, beta, tol=1e-5, max_iter=10000, verbose=0):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit_transform(self, X, C_init, S_init, Y, A, B, Z, D, E):
        """Fit the penalized multiplicative update procedure to provided matrices

        Args:
            X (2D array): Matrix X of shape n x p
            C (2D array): Matrix C of shape n x K
            S (2D array): Matrix S of shape K x p
            Y (2D array): Matrix Y of shape m x g
            A (2D array): Matrix A of shape K x g
            B (2D array): Matrix B of shape m x n
            Z (2D array): Matrix Z of shape l x q
            D (2D array): Matrix D of shape p x q
            E (2D array): Matrix E of shape l x K

        Returns:
            C (2D array): Matrix C of shape n x K. Fitted concentrations.
        """
        C, S, n_iter, losses_nmf, losses_constraint_c, losses_constraint_s = _fit_transform(X, C_init, S_init, Y, A, B, Z, D, E, self.alpha, self.beta, self.max_iter, self.tol,
                                                                                            verbose=self.verbose)

        self.n_components_ = S.shape[0]
        self.components_ = S
        self.n_iter_ = n_iter
        self.losses_nmf_ = losses_nmf
        self.losses_constraint_c_ = losses_constraint_c
        self.losses_constraint_s_ = losses_constraint_s

        return C
    
    
    def transform(self, X, C_init=None, random_state=None):
        """Compute the concentrations associated to a new data matrix X, with learnt profiles S.

        Args:
            X (2D array): The new data matrix for which the concentrations must be estimated
            C_init (2D array, optional): A possible initial value for the concentrations. Default is None, where C is initialised at random.
        """
        if not hasattr(self, "components_"):
            raise NotFittedError("The PenNMF instance is not fitted yet. Call 'fit_transform' with "
                                 "appropriate arguments before using transform.")
        
        # In this case we just want to estimate the concentrations
        # we do not consider the contraint on C and S
        alpha = 0
        beta = 0
        # We just set a null value of the matrices appearing in the constraint terms for compatibility in the code
        Y = np.zeros((1,1))
        A = np.zeros((self.n_components_, 1))
        B = np.zeros((1, X.shape[0]))
        Z = np.zeros((1,1))
        D = np.zeros((X.shape[1], 1))
        E = np.zeros((1, self.n_components_))

        # Initialize random generator
        rng = np.random.RandomState(random_state)

        if C_init is None:
            C_init = rng.rand(X.shape[0], self.n_components_)
            C_init = C_init / C_init.sum(axis=1, keepdims=True) # Normalize to "project on simplex"
        
        C, *_ = _fit_transform(X, C_init, self.components_, Y, A, B, Z, D, E, alpha, beta, self.max_iter, self.tol, fit=False, verbose=self.verbose)

        return C