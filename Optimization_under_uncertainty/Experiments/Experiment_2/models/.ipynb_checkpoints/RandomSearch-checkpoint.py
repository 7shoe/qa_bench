import time
import numpy as np
from typing import Tuple
from Oracle import Oracle
from utils import get_alpha, expand

class RandomSearch:
    def __init__(self, d:int, n_total:int, oracle:Oracle, mode:str='min'):
        self.d = d
        self.n_total = n_total
        self.oracle  = oracle
        self.mode = mode
        
    def run(self,) -> None:
        bin_grid = np.random.randint(2, size=(self.n_total, self.d))
        t0       = time.time()
        f_vals   = np.apply_along_axis(self.oracle.fun, axis=1, arr=bin_grid)
        t1       = time.time()
        
        # best values
        if(self.mode=='min'):
            self.f_opt  = min(f_vals)
            self.x_opt  = bin_grid[np.argmin(f_vals),]
            self.regret = min(f_vals)-self.oracle.f_min
        else:
            self.f_opt  = max(f_vals)
            self.x_opt  = bin_grid[np.argmax(f_vals),]
            self.regret = min(f_vals)-self.oracle.f_max
            
        # tracking
        self.timestamps  = np.linspace(t0, t1, len(f_vals)).tolist()
        self.f_hat_list  = [np.nan] *  len(f_vals) # 
        self.f_obs_list  = f_vals.tolist() # 
        self.x_loc_list  = bin_grid.tolist()
        if(self.mode=='min'):
            self.regret_list = (f_vals - self.oracle.f_min).tolist()
        else:
            self.regret_list = (self.oracle.f_max - f_vals).tolist()
        
        return