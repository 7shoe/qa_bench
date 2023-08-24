from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import math
import time

from Oracle import Oracle
from utils import expand

class BayesianOptimization(ABC):
    def __init__(self, oracle:Oracle, mode:str='min', n_sim:int=100, burnin:int=50, thinning:int=1, seed:int=9973, standardizeX:bool=True):
        self.mode = mode
        self.n_sim = n_sim
        self.burnin = burnin
        self.thinning = thinning
        self.seed = seed
        self.oracle = oracle
        self.standardizeX = standardizeX
        
    def fit(self, X:np.array, y:np.array):
        assert X.ndim==2, "Design matrix must be a 2d array."
        assert y.ndim==1, "Vector of observations must be a 1d array."
        assert len(X)==len(y), "Lengths of `X` and `y` must coincide."
        
        self.X = X
        self.y = y
        
    def run(self, N:int, refitFreq:int):
        assert N>0, "Number of simulations `n_sim` must be positive."
        
        self.N = N
        self.refitFreq = refitFreq
        
        self.f_opt  = np.inf if self.mode=='min' else -np.inf
        self.x_opt  = np.nan
        self.regret = np.inf
        
        self.timestamps  = []
        self.f_obs_list  = []
        self.f_hat_list  = []
        self.x_loc_list  = []
        self.regret_list = []
        
        # - optimize sequence
        for i in range(self.N):
            # - sample new x
            stat_model_estimate = self.statistical_model.model_estimate()
            
            # run
            f_hat_loc, x_loc = self.optimizer.run(stat_model_estimate)
            
            # oracle
            f_loc = self.oracle.f(x_loc)
            
            # regret
            regret_loc = self.oracle.regret(x_loc=x_loc, mode=self.mode)
            
            # tracking
            self.track(f_hat=f_hat_loc, f_obs=f_loc, x_loc=x_loc, regret=regret_loc)
            
            # - add x to X etc.
            self.statistical_model.add(x_new=x_loc, y_new=f_loc, refitFlag=i%self.refitFreq==0)
        
        pass
    
    def track(self, f_hat:float, f_obs:float, x_loc:np.array, regret:float) -> None:
        """
        Keeps track of rolling information.
        """
        
        # rolling
        self.timestamps.append(round(time.time()))
        self.f_hat_list.append(f_hat)
        self.f_obs_list.append(f_obs[0])
        self.x_loc_list.append(x_loc)
        self.regret_list.append(regret)
        
        # update best
        f_obs = f_obs[0]
        if(self.mode=='max'):
            if(f_obs > self.f_opt):
                self.f_opt  = f_obs
                self.x_opt  = x_loc
                self.regret = regret
        else:
            if(f_obs < self.f_opt):
                self.f_opt = f_obs
                self.x_opt = x_loc
                self.regret = regret
    
    def get_result(self, variable:str) -> List[float]:
        """
        Shows evolution of tracked variable during run iteration of Bayesian optimization.
        
        Args:
            variable (str): Either `f_opt`, `x_opt`, or `regret`
        Returns:
            List of monitored variable; one observation per optimization step.
        Raises:
            - 
        """
        if(variable=='f_opt'):
            variable_list = self.f_obs_list
        elif(variable=='x_opt'):
            return self.x_loc_list
        elif(variable=='regret'):
            variable_list = self.regret_list
        elif(variable=='normalized_regret'):
            variable_list = [(regret - self.oracle.f_min) / (self.oracle.f_max - self.oracle.f_min) for regret in self.regret_list]
        else:
            raise NotImplementedError(f"Unknown `variable`={variable}")
            
        if('regret' in variable or self.mode=='min'):
            return [min(variable_list[:i]) for i in range(1,len(variable_list)+1)]
        else:
            return [max(variable_list[:i]) for i in range(1,len(variable_list)+1)]
    
    @property
    @abstractmethod
    def statistical_model(self):
        pass
    
    
    @property
    @abstractmethod
    def statistical_model(self):
        pass
    
    @property
    @abstractmethod
    def optimizer(self):
        pass