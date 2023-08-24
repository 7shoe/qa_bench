import sys
import numpy as np
from typing import Callable
from utils import *
import inspect
import re

class Oracle:
    def __init__(self, fun:Callable, x_dim:int, alpha:np.array, sigma:float=0.0, N_total:int=100_000, seed:int=6784, max_eval:int=10**4):
        self.fun = fun
        self.alpha = alpha
        self.x_dim = x_dim
        self.sigma = sigma
        self.seed = seed
        self.N_total = N_total
        self.N_current = 0
        
        self.compute_optimum(max_eval)
        
    def compute_optimum(self, max_eval:int) -> None:
        """
        Computes the global minimum and maximum of a function f. Approximates binary search space if enumeration would exceed `max_eval`.
        """
        # enumerate/approximate search space
        if(self.x_dim < int(np.log2(max_eval))):
            bin_grid = np.array(np.meshgrid(*[[0, 1]] * self.x_dim)).T.reshape(-1, self.x_dim)
        else:
            bin_grid = np.random.randint(2, size=(max_eval, self.x_dim))

        # alpha
        if(self.alpha is not None):
            #print('Compute optimum...`alpha` was provided. Ignore `f` & compute function value via 2nd order interactions and `alpha`.')
            f_vals = expand(bin_grid, 2, True) @ self.alpha
        else:
            f_vals = np.apply_along_axis(self.fun, axis=1, arr=bin_grid)
        
        self.f_max = max(f_vals)
        self.f_min = min(f_vals)
    
    def f(self, x:np.array) -> np.array:
        '''
        Returns vector of (noisy) function values for groundtruth funtion f
        '''
        
        if(x.ndim==1):
            x = x.reshape(1,-1)
        
        assert self.N_current + len(x) <= self.N_total, f"Limit of `N_total`={self.N_total} exceeded as current={self.N_current} + additional {len(x)} calls."
        
        self.N_current += len(x)
        
        if(self.alpha is not None):
            #print('`alpha` was provided. Ignore `f` & compute function value via 2nd order interactions and `alpha`.')
            return np.random.normal(expand(x, 2, True) @ self.alpha, scale=self.sigma)
            
        else:
            return np.random.normal(self.fun(x), scale=self.sigma)
        
    def regret(self, x_loc:np.array, mode:str) -> float:
        assert mode in ['max', 'min'], "`mode` must be either `min` or `max`"
        
        # no noise function evaluation
        f_exact = self.fun(x_loc)
        
        if(mode=='min'):
            delta = f_exact - self.f_min
        else:
            delta = self.f_max - f_exact
            
        if(delta < 0):
            print('Regret is negative... likely an artefact from the approximation of binary mesh grid')
            
        return delta
        
        
        