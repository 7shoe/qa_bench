import math
import numpy as np
from typing import Tuple, Optional
from itertools import combinations


def get_alpha(d:int, k:int, p_zero:float=0, dist:str='gaussian', sigma:float=1, rnd_val:int=5967):
    """
    Returns a (sparse) random coefficient vector for data of dimension d and order (of interactions) of up to k sampled IID.
    
    Args:
        d (int)          :  Dimension of input 
        k (int)          :  Highest order of coefficients included in the model
        p_zeros (float)  :  Probability of an element being zero in the coefficient vector if sparse=True; controls degree of sparsity
        dist (str)       :  Distribution from which coefficients are sampled
        sigma (float)    :  Standard deviation of the distribution from which entries are sampled, i.e. empirical standard deviation before sparsification is applied

    Returns:
        NumPy array with the randomly sampled coefficients sorted by order (0 to incl. k)

    Raises:
        - AssertionError : If order of interaction is larger than the dimension
    """
    
    assert dist in ['gaussian', 'cauchy', 'uniform'], "Distribution `dist` must be either `gaussian`, `cauchy` or `uniform`."
    assert p_zero>=0 and p_zero<1., "Probability for zero entry must be in unit interval [0, 1)."
    
    # length of vector alpha
    p = sum([math.comb(d,i) for i in range(k+1)])
    
    # sample
    np.random.seed(rnd_val)
    if(dist=='gaussian'):
        alpha = np.random.normal(scale=sigma, size=p)
    elif(dist=='cauchy'):
        alpha = np.random.standard_cauchy(size=p)
        std_hat  = min(50.0, alpha.std())         # Cauchy: E[X^{k}]=inf
        alpha *= sigma/std_hat
    else:
        b = np.sqrt(12) * sigma / 2.                 # symmetric uniform, E[U]=0, Var[U]=sigma
        alpha = np.random.uniform(low=-b, high=b, size=p)
        
    # sparsify
    if(p_zero>0):
        alpha[np.random.choice(a=range(p), size=round(p*p_zero), replace=False).tolist()] = 0
        
    # round
    alpha = alpha.round(10)
    
    return alpha



def expand(X:np.array, k:int=2, intercept:bool=True):
    '''
    Expands an (nxd) matrix to a (n x p) matrix where p=p(k) is a function of the highest order of interactions `k`.
    
    Args:
        X (np.array)       :         NumPy array of dimension d (length d for 1-d array)
        k (int)            :         Highest order up to which interaction terms are computed
                                       0 : constant; no terms but intercept
                                       1 : linear; no interactions
                                       2 : interactions (x_{i}*x_{j} etc.)
                                       3 : x_{i}*x_{j}*x_{k} etc)
                                       4 :   ...
        intercept (bool)    :       If a leading (column of) 1 is to be added so that subsequent operations can happen as X @ alpha
    
    Returns:
        NumPy array augmented with interactions (sorted by (i) order of interaction and (ii) lexicographical order of index tuple)
        
    Raises:
        NotImplementedError: If array X is of dim >=3 instead of being a vector (1d) or matrix (2d).
        AssertionError: If order k is larger than dimension f of data. Output would be well-defined but cannot contain interaction terms 
    '''

    # infer length
    if(X.ndim==1):
        d = len(X)
    elif(X.ndim==2):
        d = X.shape[1]
    else:
        raise NotImplemtedError(f"Input `X` must be 1d or 2d array not {X.ndim}-dimensional as provided.")
    
    assert d>1,  "Expansion of 1-variable (d=1) is not viable."
    assert k<=d, f"Interaction terms of order up to k={k} not available if input dimension is d={d}."
    
    # compute interactions of <= k-th order
    if(k==0):
        return np.ones(X.shape[0], dtype=int).reshape(X.shape[0],-1)
    elif(k==1):
        pass
    else:
        # indices
        col_Idx = []
        for l in range(2,k+1):
            col_Idx += sorted(list(combinations(range(d),l)))

        # generate interaction columns
        if(X.ndim==2):
            for idx_tuple in col_Idx:
                colProd = np.prod(X[:, idx_tuple], axis=1, keepdims=True)
                X = np.hstack((X, colProd))
        else:
            col_prod_list = []
            for idx_tuple in col_Idx:
                col_prod_list.append(np.prod(X[list(idx_tuple)]))
            X = np.concatenate((X, col_prod_list))

    # if intercept: leading 1's in front
    if(intercept):
        if(X.ndim==2):
            X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        else:
            X = np.concatenate(([1], X))

    return X

def trafo_for_mvg(X:np.array, y:float, sigma:float, Sigma_star: np.array) -> Tuple[np.array, float]:
    '''
    Transforms the Bayesian regression's data and parameters into the format that allow accelerated generation of multivariate Gaussian random numbers.
    (Source: Bhattacharya, Chakraborty, Mallick (2016), https://arxiv.org/pdf/1506.04778.pdf p. 4)
    
    Args:
        X (np.array)          : (Standardized) design matrix of the regression (n x p)
        y (np.array)          : Vector of observed repsonse (n x 1)
        sigma (float)         : Estimate of the error standard deviation
        Sigma_star (np.array) : Precision matrix; is diagonal with (tau^2 * diag(lambda_{1}^{2}, ... lambda_{p}^{2}) for the Horshoe Bayes Reg. (p x p)

    Returns:
        Tuple of matrices (Phi, D, alpha) that can be directly imputed to the accelerated multivariate Normal random number generator.

    Raises:
        AssertionError : Mismatch of dimensions of provided matrices
            
    '''
    
    assert sigma>0, "Standard deviation must be positive."
    assert X.ndim==2 and Sigma_star.ndim==2, "Both `Sigma_star` and `X` must be matrices."
    assert X.shape[1]==Sigma_star.shape[0], "Column size of X (p) must coincide with row/column size of `Sigma_star`."
    assert Sigma_star.shape[0]==Sigma_star.shape[1], "Precision matrix `Sigma_star` must be quadratic."
    
    Phi   = X / sigma
    alpha = y / sigma
    D     = sigma *  Sigma_star
    
    return Phi, D, alpha
    
    
def mvg_generation(Phi:np.array, D:np.array, alpha:np.array, n:int=1):
    '''
    Accelerates sampling of multivariate Gaussian random numbers for conditional Gaussian data in the Bayesian regression.
    (Source: Bhattacharya, Chakraborty, Mallick (2016), https://arxiv.org/pdf/1506.04778.pdf p. 4)
    
    Args:
        Phi (np.array)        : (Standardized) design matrix of the regression (n x p)
        D (np.array)          : Precision matrix multiplied with standard deviation (sigma)
        alpha (np.array)      : Rescaled (by sigma) vector of observed repsonse(n x 1)
        Sigma_star (np.array) : Precision matrix; is diagonal with (tau^2 * diag(lambda_{1}^{2}, ... lambda_{p}^{2}) for the Horshoe Bayes Reg. (p x p)

    Returns:
        Tuple of matrices (Phi, D, alpha) that can be directly imputed to the accelerated multivariate Normal random number generator.

    Raises:
        AssertionError : Mismatch of dimensions of provided matrices
            
    '''
    # read-out dimension
    n, d = Phi.shape[0], D.shape[0]
    
    # (i) sample Standard gaussian
    u     = np.random.multivariate_normal(mean=np.zeros(d),cov=D)
    delta = np.random.normal(size=n)
    
    # (ii) set v
    v = Phi @ u + delta
    
    # (iii) solve linear system
    w = np.linalg.solve(a=Phi@D@Phi.T+np.eye(n), b=alpha-v)
    
    # (iv) reconstruct multivariate gaussian
    theta = u + D @ Phi.T @ w
    
    return theta
    
def fast_multivariate_normal(X:np.array, y:np.array, sigma:float, Sigma_star:np.array) -> np.array:
    '''
    Wraps the transformation and generation of fast sampling of multivariate normal for conditional sampling of coefficient vector in Bayes Regression.
    Accelerates sampling only if n<p and p>200.
    (Source: Bhattacharya, Chakraborty, Mallick (2016), https://arxiv.org/pdf/1506.04778.pdf p. 4)
    
    Args:
        X (np.array)          : (Standardized) design matrix of the regression (n x p)
        y (np.array)          : Vector of observed repsonse (n x 1)
        sigma (float)         : Estimate of the error's standard deviation
        Sigma_star (np.array) : Precision matrix; is diagonal with (tau^2 * diag(lambda_{1}^{2}, ... lambda_{p}^{2}) for the Horshoe Bayes Reg. (p x p)

    Returns:
        Tuple of matrices (Phi, D, alpha) that can be directly imputed to the accelerated multivariate Normal random number generator.

    Raises:
        AssertionError : Mismatch of dimensions of provided matrices
    '''
    Phi, D, alpha  = trafo_for_mvg(X, y, sigma, Sigma_star)
    coef = mvg_generation(Phi, D, alpha)

    return coef

def cholesky_multivariate_normal(X:np.array, y:np.array, sigma:float, Sigma_star:np.array) -> np.array:
    '''
    Runs plain-vanilla multivariate normal sampling by first Cholesky-decomposing the covariance matrix
    '''
    
    # Sigma
    Sigma_star_inv = np.linalg.inv(Sigma_star)
    A = X.T @ X + Sigma_star_inv
    
    # Solve the linear system A @ alpha_T = X^T @ y using Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    # conditional mean
    mu = np.linalg.solve(L.T, np.linalg.solve(L, X.T @ y))
    
    # conditional covariance matrix
    cov = sigma**2 * np.linalg.inv(L.T) @ np.linalg.inv(L)
    coef = np.random.multivariate_normal(mean=mu, cov=cov, size=1).T[:,0]
    
    return coef