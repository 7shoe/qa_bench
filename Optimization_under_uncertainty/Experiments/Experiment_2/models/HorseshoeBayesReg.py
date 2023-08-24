import math
import numpy as np
import scipy
from typing import Tuple, Optional
from scipy.stats import halfcauchy, invgamma

from BayesReg import BayesReg
from utils import expand, cholesky_multivariate_normal, fast_multivariate_normal, mvg_generation, trafo_for_mvg


class HorseshoeBayesReg(BayesReg):

    d_MAX   = 50
    NUM_EPS = 0.01 # for X_sigma
    
    def __init__(self, n_sim:int=100, seed:int=3553, burnin:int=0, thinning:int=1, k:int=2, intercept:bool=True, standardizeX:bool=False, translateY:bool=False):
        super().__init__(n_sim, seed, burnin, thinning, standardizeX, translateY)
        
        self.k            = k
        self.intercept    = intercept
        self.standardizeX = standardizeX
        self.translateY   = translateY
        self.Xy_is_set    = False
    
    def __mvg__(self, Phi, alpha, D):
        '''
        LEGACY
        
        Sample multivariate Gaussian (independent of d) from NumPy
        Not used Rue (2001) or et. al. (2015) approaches on fast sampling mvg
        N(mean = S@Phi.T@y, cov = inv(Phi'Phi + inv(D))
        '''
        
        assert len(D.shape)==2 and D.shape[0]==D.shape[1], "`D` must be a quadratic matrix."
        
        S = np.linalg.inv(Phi.T @ Phi + np.linalg.inv(D))
        x = np.random.multivariate_normal(mean=((S @ Phi.T) @ y), cov=S, size=1)
        
        return x
    
    def sample_alpha(self) -> np.array:
        '''
        Samples posterior  ~ P( |X,y) from (most current) posterior distribution
        '''
        # coef vector
        if(self.p > self.n and self.p > 200):
            return fast_multivariate_normal(self.X, self.y, self.sigma, self.Sigma_star)
        
        return cholesky_multivariate_normal(self.X, self.y, self.sigma, self.Sigma_star)
    
    def model_estimate(self,):
        """
        Function that returns the estimate of the statistical model for Horseshoe Bayesian regression is just the coefficient vector.
        
        Args:
            -

        Raises:
            -
        """
        
        return self.sample_alpha()
    
    def __fit__(self) -> None:
        '''
        Core of fitting procedure (on self.X, self.y)
        '''
        
        # D0
        self.n = len(self.X)
        
        # setup values
        alphas_out = np.zeros((self.p, 1))
        s2_out     = np.zeros((1, 1))
        t2_out     = np.zeros((1, 1))
        l2_out     = np.zeros((self.p, 1))

        # sample priors
        betas   = halfcauchy.rvs(size=self.p)
        tau_2   = halfcauchy.rvs(size=1)                            
        nu      = np.ones(self.p) # ?
 
        sigma_2, xi = 1.0, 1.0
        
        # Gibbs sampler
        for k in range(self.n_sim):
            self.sigma = np.sqrt(sigma_2)

            # alphas
            # - Sigma_star
            self.Sigma_star     = tau_2 * np.diag(betas**2) # Sigma_star
            Sigma_star_inv = np.diag(1. / betas**2) * (1. / tau_2)
            
            # - sample alpha
            alphas = self.sample_alpha()
            
            # sigma_2
            sigma_2 = invgamma.rvs(0.5*(self.n+self.p), scale=0.5*(np.linalg.norm((self.y - self.X @ alphas), 2)**2 + (alphas.T @ Sigma_star_inv @ alphas)))

            # - betas
            betas = np.sqrt(invgamma.rvs(np.ones(self.p), scale=(1. / nu) + (alphas**2)/(2 * sigma_2 * tau_2)))

            # - tau_2
            tau_2 = invgamma.rvs(0.5*(self.p+1), scale=1.0 / xi + (1. / (2. * sigma_2)) * sum(alphas**2 / betas**2), size=1)

            # - nu
            nu = invgamma.rvs(np.ones(self.p), scale=1.0 + betas**(-2), size=self.p)

            # - xi
            xi = invgamma.rvs(1.0, scale=1.0 + 1.0 / tau_2, size=1)
            
            # store samples
            if k > self.burnin:
                # - append
                if(k%self.thinning==0):
                    alphas_out = np.append(arr=alphas_out, values=alphas.reshape(-1,1), axis=1)
                    s2_out = np.append(s2_out, sigma_2)
                    t2_out = np.append(t2_out, tau_2)
                    l2_out = np.append(arr=l2_out, values=betas.reshape(-1,1), axis=1)

        # Clip 1st value
        self.alphas = alphas_out[:,1:]
        self.s2 = s2_out[1:]
        self.t2 = t2_out[1:]
        self.l2 = l2_out[1:]
    
    def fit(self, X:np.array, y:np.array) -> None:
        '''
        Fitting the (initial) model on the data D0={X0,y0}
        '''
        
        if(not(self.Xy_is_set)):
            self.setXy(X=X, y=y, k=self.k, intercept=self.intercept, standardizeX=self.standardizeX, translateY=self.translateY)
            self.Xy_is_set = True
        
        assert len(X.shape)==2 and len(y.shape)==1, "Design matrix X and target vector y."
        assert X.shape[0]==len(y), f"Dimension of design matrix X and target vector y do not coincide: X.shape[1]={X.shape[1]}!={len(y)}=len(y)"
        
        # fitting
        self.__fit__()