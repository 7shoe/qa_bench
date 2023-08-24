import os
from typing import Tuple, List, Iterable, Callable

import math
import time
import numpy as np
import pandas as pd
from scipy.stats import halfcauchy, invgamma
import cvxpy as cp
from matplotlib import pyplot as plt
from scipy import stats

from scipy.stats import multivariate_normal
from numpy.random import default_rng
from itertools import chain, combinations, product

class SDP2:
    penOrdList = [1, 2]
    optModes = ['min', 'max']
    d_MAX = 20
    
    def __init__(self, alpha:np.array, lambd:float=0.1, pen_ord:int=2, mode:str='min') -> List[np.array]:
        
        assert isinstance(mode, str), "Input `mode` must be a str either `min` or `max`."
        mode = mode.lower()
        assert mode in self.optModes, "Input `mode` is str. In addition, it must be a str either `min` or `max`."
        assert isinstance(lambd, float) and lambd >=0, "lambda (regularization parameter) must be non-negative scalar."
        assert pen_ord in self.penOrdList, "Only l=1 or l=2 supported for `pen_ord`"
        
        self.mode = mode
        self.lambd = lambd
        self.pen_ord = pen_ord
        self.alpha = alpha
        
        # infer d(imension)
        self.p = len(self.alpha)
        dDict = {1+dLoc+math.comb(dLoc,2) : dLoc for dLoc in range(1,self.d_MAX+1)}
        if(self.p in dDict.keys()):
            self.d = dDict[self.p]
        else:
            assert False, f'Length of `alpha` is not a 1+d+binom(d,2) for any 1,2,...,{self.d_MAX}'
        assert isinstance(self.d, int), "Dimension `d` must be non-negative integer."
        
        #print('self...d: ', self.d)
        
        # extract 1st/2nd order terms
        b = self.alpha[1:1 + self.d]  # 1st
        a = self.alpha[1 + self.d:]   # 2nd

        # get indices for quadratic terms
        idx_prod = np.array(list(combinations(np.arange(self.d), 2)))
        d_idx = idx_prod.shape[0]

        # check number of coefficients
        if len(a)!=d_idx:
            assert False, 'Number of Coefficients does not match indices!'

        # xAx-term
        A = np.zeros((self.d, self.d))
        for i in range(d_idx):
            A[idx_prod[i,0], idx_prod[i,1]] = 0.5 * a[i]
        A += A.T

        # bx-term
        bt = 0.5 * (b + A @ np.ones(self.d)).reshape((-1, 1))
        bt = bt.reshape((self.d, 1))
        At = np.vstack((np.append(0.25*A, 0.25*bt, axis=1), np.append(bt.T, 2.)))
        
        self.A  = A
        self.b  = b
        self.At = At
        self.bt = bt
        
    def run(self) -> np.array:
        '''
        Runs the BQP-relaxation, SDP-optimization, and extracts candidate x via geometric rounding
        '''
        self.solve()
        self.decompose()
        return self.geometricRounding()
        
        
    def solve(self, ) -> None:
        '''Solve SDP'''
        # SDP relaxation
        Xvar = cp.Variable((self.d+1, self.d+1), PSD=True)
        
        # - objective function
        if(self.mode=='min'):
            f0 = cp.Minimize(cp.trace((self.At @ Xvar)))
        else:
            f0 = cp.Maximize(cp.trace((self.At @ Xvar)))
        
        # - constraints
        constraints = [cp.diag(Xvar) == np.ones(self.d+1)]
        prob = cp.Problem(f0, constraints)
        prob.solve()
        
        self.Xvar = Xvar
        
        #print('Xvar.shape: ', Xvar.shape)
        
    def decompose(self) -> None:
        '''
        Wrapper for stable Cholesky decomposition
        '''
        self.L = self.__stableCholesky__(eTol=1E-12)
    
    def __stableCholesky__(self, eTol:float=1E-10) -> np.array:
        '''
        Performs numerically stable Cholesky decomposition (by adding regularity to the matrix until PSD). 
        '''
        try:
            return np.linalg.cholesky(self.Xvar.value + eTol*np.eye(self.Xvar.value.shape[0]))
        except Exception as e:
            if(isinstance(e, np.linalg.LinAlgError)):
                return self.__stableCholesky__(10*eTol)
            else:
                pass
    
    def geometricRounding(self, k_rounds:int=100) -> np.array:
        '''
        Random geometric round and conversion to original space
        - k_rounds: number of iterations
        '''
        x_cand  = np.zeros((self.d, k_rounds))
        f_star  = np.zeros(k_rounds)

        for j in range(k_rounds):
            # rnd cutting plane vector (U on Sn) 
            r = np.random.randn(self.d+1)
            r /= np.linalg.norm(r, ord=2)
            
            # rnd hyperplane
            y_star = np.sign(self.L.T @ r)

            # convert solution to original domain and assign to output vector
            x_cand[:,j] = 0.5 * (1.0 + y_star[:self.d])
            f_star[j] = (x_cand[:,j].T @ self.A @ x_cand[:,j]) + (self.b @  x_cand[:,j])

            # Find optimal rounded solution
            if(self.mode=='min'):
                f_argopt = np.argmin(f_star)
            else:
                f_argopt = np.argmax(f_star)
            x_0      = x_cand[:,f_argopt]
            f_0      = f_star[f_argopt]

        return (x_0, f_0)