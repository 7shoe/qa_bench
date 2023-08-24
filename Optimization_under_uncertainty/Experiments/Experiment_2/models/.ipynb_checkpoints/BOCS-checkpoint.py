from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import math

from BO import BayesianOptimization
from Oracle import Oracle
from SDP import SDP
from HorseshoeBayesReg import HorseshoeBayesReg
from utils import expand

class BOCS(BayesianOptimization):
    def __init__(self, oracle:Oracle, d:int, k:int, afo:str, mode:str='max', n_sim:int=100,  burnin:int=50, thinning:int=1, reg_lambda:float=0.1, standardizeX:bool=True,  seed:int=9973):
        super().__init__(oracle, mode, n_sim, burnin, thinning, seed, standardizeX)
        
        assert afo in ['SDP', 'SA'], "Acquisition function optimization is either `SDP` (semidefinite relaxation) or `SA` (simulated annealing)."
        assert d>0, "The dimension of the (raw) input binary data must be positive."
        assert k>=0, "The maximal order of interactions `k` considered must be non-negative."
        assert reg_lambda>0, "Penalty/regularization parameter `reg_lambda` should be positive."
        if(k==2):
            assert afo=='SDP', "SDP is only supported for order `k=2`."
        
        self.d = d
        self.k = k
        self.afo = afo
        self.reg_lambda = reg_lambda
        self.isFit = False
        
        self._statistical_model = HorseshoeBayesReg(n_sim=self.n_sim, seed=self.seed, burnin=self.burnin, thinning=self.thinning)
        
        if(self.afo=='SDP'):
            self._optimizer = SDP(d=self.d, mode=self.mode, n_sim=self.n_sim, reg_lambda=self.reg_lambda)
        else:
            raise NotImplementedError("Only `SDP` is implemented as the `afo` for BOCS.")
        
        pass
    
    def fit(self, X:np.array, y:np.array) -> None:
        super().fit(X=X,y=y)
        
        self.statistical_model.fit(X=self.X, y=self.y)
        self.isFit = True
    
    def run(self, N:int, refitFreq:int=1) -> np.array:
        assert self.isFit, "No statistical model estimate yet. `.fit(X, y)` for initial estimation before running the iterative optimizer."
        super().run(N=N, refitFreq=refitFreq)
    
    @property
    def statistical_model(self):
        return self._statistical_model
    
    @property
    def optimizer(self):
        return self._optimizer