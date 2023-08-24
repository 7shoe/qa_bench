from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import math

from utils import expand

class BayesReg(ABC):
    def __init__(self, n_sim:int, seed:int, burnin:int, thinning:int, standardizeX:bool, translateY:bool):
        
        assert isinstance(n_sim, int) and n_sim > 0, "Number of simulations must be positive."
        assert isinstance(seed, int), "`seed` must be an integer."
        assert isinstance(burnin, int) and burnin>=0, "`burnin` must be a non-negative integer."
        assert isinstance(thinning, int) and thinning>0, "`thinning` must be an positive integer."
        
        self.n_sim = n_sim
        self.seed = seed
        self.burnin = burnin
        self.thinning = thinning
        self.standardizeX = standardizeX
        self.translateY = translateY
        self.Xy_is_set = False
        
    def setXy(self, X:np.array, y:np.array, k:int=2, intercept:bool=True, standardizeX:bool=True, translateY:bool=True) -> None:
        '''
        Sets up design matrix X (standardized, incl. leading 1-column and up to k-th order interaction terms) and target vector y (transalted to E[y]=0)
        '''
        
        assert X.ndim==2, "Design matrix `X` must be a 2d numpy array."
        assert y.ndim==1, "Vector of reponses `y` must be a 1d numpy array."
        assert X.shape[1]>= k, "The dimension of (raw) input `d` must be at least as big as the order of interactions `k`."
        assert X.shape[1]<=self.d_MAX, f"The inferred dimension `d`={X.shape[1]} should be smaller than {self.d_MAX}"
        
        self.d = X.shape[1]
        self.k = k
        self.p = sum([math.comb(self.d,i) for i in range(1,self.k+1)]) + (1 if intercept else 0)
        
        self.intercept = intercept
        self.X = expand(np.array(X), k=self.k, intercept=self.intercept)
        
        # standardize
        if(self.standardizeX):
            self.__standardizeX__(intercept=self.intercept)
        else:
            self.i_start = 1
            self.X_mu    = np.zeros(self.p-1)
            self.X_sigma = np.ones(self.p-1)
        
        # translate y
        if(self.translateY):
            self.__translateY__(y)
        else:
            self.y_mu = 0
            self.y = y

    def __standardizeX__(self, intercept:bool=True) -> None:
        '''
        Standardizes (translates & rescales) the columns of the design matrix (input matrix) 
        '''
        
        X = np.array(self.X)
        
        # 1st, 2nd moment
        X_mu, X_sigma = X.mean(axis=0), X.std(axis=0)
        # - numerically stabilzed standard deviation
        X_sigma[np.abs(X_sigma) < self.NUM_EPS] = 1    # TODO: Check functionality
        
        # standardize
        X_new = (X - X_mu) /  X_sigma
        
        # recover intercept column
        if(intercept):
            X_new[:,0] = 1
            self.i_start = 1
        
        # store moments
        self.X_mu = X_mu[self.i_start:]
        self.X_sigma = X_sigma[self.i_start:]
        self.X = X_new

    def __translateY__(self, y:np.array) -> None:
        '''
        Translation of the target vector y such that priori condition E[y]=0 is satisfied.
        (No rescaling to unit variance is applied, though.)
        
        UNSURE IF THIS MAKES SENSE: Deems intercept alpha_0 useless, right?
        '''
        X = np.array(self.X)

        assert len(X) == len(y), "Length of target vector y does not coincide with design matrix X"

        # translate y's
        y_mu = np.mean(y)
        y_new = y - y_mu
        
        self.y_mu = y_mu
        self.y = y_new
        
    def add(self, x_new:np.array, y_new:float, refitFlag:bool=True) -> None:
        '''
        Appends new datapoint to X,y and optionally re-runs fitting
        '''

        assert x_new.ndim==1 and len(x_new)==self.d, f"Input has dimension {x_new.shape} but ({self.d},) was expected."

        # expand exogenous vector
        x_new_exp = expand(x_new, k=self.k, intercept=self.intercept) # expand(np.array(X), k=self.k, intercept=self.intercept)
        
        # standardize X
        x_new_exp[self.i_start:] = x_new_exp[self.i_start:] - self.X_mu 
        x_new_exp[self.i_start:] = x_new_exp[self.i_start:] / self.X_sigma
        # - append X
        self.X = np.vstack((self.X, x_new_exp))
        
        # translate, append y
        self.y = np.append(arr=self.y, values=(y_new - self.y_mu))
        
        # re-fit
        if(refitFlag):
            self.__fit__()

    @abstractmethod
    def fit(self, method:Optional[str]=None):
        """
        Runs the training (estimation/fitting procedure) of the statistical model given the registered data `X` and `y`.
        
        Args:
            method (str, optional): The statatistical procedure by which parameter estimation is done.

        Raises:
            NotImplementedError: If an unkwon method is called.
        """
        
        pass
    
    @abstractmethod
    def model_estimate(self,):
        """
        Wraps the function that returns the estimate of the statistical model (e.g. coefficient vector)
        
        Args:
            -

        Raises:
            -
        """
        
        pass
    
    def get_alphas(self) -> np.array:
        '''
        Returns current array of alpha posterior samples
        
        Args:
            - rescaled (bool) : Indicates if the estimate alpha is with respect to the design matrix X or standardized version (X / X_sigma)
        
        Returns:
            NumPy array (MAP) of the parameter coefficients
        
        Raises:
            - 
        '''
        
        # array of `alpha` samples
        if(self.alphas.ndim==2):
            alpha_sample = np.array(self.alphas[:,-1])
        else:
            alpha_sample = np.array(self.alphas)
        
        alpha_sample[1:] = alpha_sample[1:] / self.X_sigma
        
        return alpha_sample