# Auxiliary Script
import time
import numpy as np
from typing import Tuple
from Oracle import Oracle
from utils import get_alpha, expand

class GeneticAlgo:
    def __init__(self, n_init:int, n_total:int, oracle:Oracle, mode:str='min', p_mut:float=0, p_sel:float=0, rndVal:int=5734):
        
        assert 0<=p_sel<=1, "Selection probability `p_select` must be in (0,1]."
        assert 0<p_mut<=1,    "Mutation probability `p_select` must be in (0,1]."
        
        self.n_init     = n_init
        self.n_total    = n_total
        self.oracle     = oracle
        self.mode       = mode
        self.p_mut      = p_mut
        self.p_sel      = p_sel
        self.rndVal     = rndVal
        self.n_current  = 0
        self.f_opt      = np.inf if mode=='min' else -np.inf
        self.x_opt      = np.nan
        self.regret     = np.inf if mode=='min' else -np.inf
        self.terminated = False
        
        self.timestamps  = []
        self.f_hat_list  = []
        self.f_obs_list  = []
        self.x_loc_list  = []
        self.regret_list = []
        
        np.random.seed(self.rndVal)
        X_init, y_init = self.initial_dset()
        self.X_current = X_init
        self.y_current = y_init
        
        pass
    
    def run(self, ) -> None:
        k_iter = 1 + round((self.n_total - self.n_current) / round(0.5*self.n_init))
        while(not(self.terminated)):
            self.optimize()
    
    def optimize(self,) -> None:
        """
        Run optimization
        """
        
        # read-out current population
        X_current = np.array(self.X_current)
        y_current = np.array(self.y_current)
        
        # select
        X_, y_ = self.tournament_selection(X_current, y_current)
        
        # cross-over (recombination)
        X_, y_ = self.cross_over(X_, y_)
        
        # mutation
        X_, y_ = self.mutation(X_, y_, p_mut=self.p_mut)
        
        # evaluate
        X_, y_ = self.evaluate(X_, y_, oracle=self.oracle)
        
        # update sets
        self.X_current = X_current
        self.y_current = y_current
        
        # update best
        if(self.mode=='min'):
            if(self.f_opt > np.min(self.y_current)):
                self.f_opt = np.min(self.y_current)
                self.x_opt = self.X_current[np.argmin(self.y_current),:]
        else:
            if(self.f_opt < np.max(self.y_current)):
                self.f_opt = np.max(self.y_current)
                self.x_opt = self.X_current[np.argmax(self.y_current),:]
        
        pass
    
    
    def initial_dset(self,) -> Tuple[np.array, np.array]:
        """
        Generates initial population
        """
        
        X  = np.random.binomial(n=1, p=0.5, size=self.n_init * self.oracle.x_dim).reshape(self.n_init,-1)
        y  = self.oracle.f(X)
        
        self.n_current += len(y)
        
        return X, y
    
    
    def tournament_selection(self,X:np.array, y:np.array) -> Tuple[np.array, np.array]:
        """
        Select the better half of the population
        """
        

        # selection
        indices = np.argsort(y)
        subset  = indices[:round(len(y) * 0.5)]

        X_sub = np.array(X[subset,:])
        y_sub = np.array(y[subset])

        return X_sub, y_sub

    def cross_over(self, X:np.array, y:np.array) -> np.array:
        """
        Takes the better (in terms of fittness) half of observations and crosses them over (random pairing, random binary mask)

        Args:
           - X (np.array) : Binary design matrix sorted by fitness (best=first)
           - y (np.array) : Vector of response observations ("fitness values") sorted according to X

        Returns:
           - X_new (np.array) : updated design matrix where ~half of observations have no corresponding fitness (yet)
           - y_new (np.array) : 
        """

        # copy better `half`
        X_sub       = np.array(X)
        y_sub       = np.array(y)

        # random permutation of order
        rnd_indices = np.random.permutation(len(X_sub))

        # sort subset randomly (X & y)
        X_sub       = X_sub[rnd_indices,:] 
        y_sub       = y_sub[rnd_indices]

        # new subset (of children)
        X_new       = np.zeros_like(X_sub)

        d = X_sub.shape[1]
        # pairwise cross-combine
        for i in range(0,2*(len(X_sub)//2),2):
            # read out current pair
            x1  = np.array(X_sub[i,:])
            x2  = np.array(X_sub[i+1,:])

            # random recombination
            mask = np.random.choice([True, False], size=d)

            # cross-over
            x_h      = np.array(x1)
            x1[mask] = x2[mask]
            x2[mask] = x_h[mask]

            # store
            X_new[i,:]  =x1
            X_new[i+1,:]=x2

        # merge
        X_merge = np.vstack((X_sub, X_new))
        y_merge = np.concatenate((y_sub,  np.array([np.nan]*len(X_new))))

        return X_merge, y_merge


    def mutation(self, X:np.array, y:np.array, p_mut:float) -> Tuple[np.array, np.array]:
        """
        Apply random mutation to the -as of now- un-evaluated observations `p_mut` 
        """

        # design sub-matrix for which there are no observations taken
        X_new = np.array(X[np.isnan(y),:])

        # restore
        X_new_flat  = np.array(X_new.flatten())
        rnd_index   = np.random.choice(range(np.prod(X_new.shape)), size=round(len(X_new)*p_mut))
        rnd_entries = 1 - np.array(X_new_flat[rnd_index])
        X_new_flat[rnd_index] = rnd_entries

        # output
        X_out = X_new_flat.reshape(len(X_new),-1)

        # re-assign
        X[np.isnan(y),:] = X_out

        return X, y

    def evaluate(self, X:np.array, y:np.array, oracle:Oracle) -> Tuple[np.array, np.array]:
        """
        Query oracle for observations for which y_{i} is np.nan
        """

        if(self.n_total <= self.n_current):
            self.terminated = True
            return
        elif(self.n_total > self.n_current + len(y[np.isnan(y)])):
            n_remaining     = len(y[np.isnan(y)])
            X_in            = X[np.isnan(y),:]
            f_obs           = oracle.f(X_in)
            y[np.isnan(y)]  = f_obs
            self.n_current += n_remaining
        else:
            n_remaining = self.n_total - self.n_current
            X_in        = X[np.isnan(y),:][:n_remaining,]
            f_obs       = oracle.f(X_in)
            y[np.isnan(y)][:n_remaining,] = f_obs
            self.terminated = True
            self.n_current += n_remaining
        
        # update best
        if(self.mode=='max'):
            if(max(f_obs) > self.f_opt):
                self.f_opt  = max(f_obs)
                self.x_opt  = X_in[np.argmax(f_obs),]
                self.regret = self.oracle.f_max - self.f_opt
        else:
            if(min(f_obs) < self.f_opt):
                self.f_opt  = min(f_obs)
                self.x_opt  = X_in[np.argmin(f_obs),]
                self.regret = self.f_opt - self.oracle.f_min
                
        # rolling
        self.timestamps.append(time.time())
        self.f_hat_list += [np.nan] *  len(f_obs) # 
        self.f_obs_list += f_obs.tolist() # 
        self.x_loc_list += X_in.tolist()
        self.regret_list.append(self.regret)
        
        return X,y