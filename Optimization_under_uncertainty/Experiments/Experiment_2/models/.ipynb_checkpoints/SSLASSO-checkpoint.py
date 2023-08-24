import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from typing import Union, Any, Optional

from BayesReg import BayesReg

class SSLASSO(BayesReg):

    d_MAX   = 50
    NUM_EPS = 0.01 # for X_sigma
    
    def __init__(self, n_sim:int=100, seed:int=1234, burnin:int=0, thinning:int=1,  k:int=2, intercept:bool=True, standardizeX:bool=False, translateY:bool=False):
        super().__init__(n_sim, seed, burnin, thinning, standardizeX, translateY)
        self.k = k
        self.intercept = intercept
    
    def __parse_R_alpha__(self, output:str):
        """
        Parses the str-output of the SSLASSO method into the coefficient vector estimate `alpha` of length 1+p (incl. intercept)

        Args:
            output (str)    :     String output from the R script running the SSLASSO routine representing the numeric vector. 
        Returns:
            A 1-d NumPy array corresponding to the parameter estimate.
        Raises:
            AssertionError  :     If resulting 
        """

        # remove square bracket integers and newlines
        cleaned_str = re.sub(r'\[\d+\]|\n', '', output)
        
        # Handle the special case of all zeros
        if(set(cleaned_str)=={' ', '0'}):
            alpha_hat = np.zeros(len(re.findall(r'0', cleaned_str)))
        else:
            alpha_hat = np.array([float(val) for val in re.findall(r'-?\d+\.\d+', cleaned_str)])

        assert alpha_hat.dtype==float, "Parsed numpy array should be float vector."
        
        return alpha_hat
    
    def __fit__(self) -> None:
        '''
        Core of fitting procedure (on self.X, self.y)
        '''
        self.fit(self.X, self.y, method='MAP')
        
        
    
    def fit(self, X:np.array, y:np.array, method:Optional[str]=None, verbose:bool=False):
        """
        Runs the training (estimation/fitting procedure) of the statistical model given the registered data `X` and `y`.
        
        Args:
            method (str, optional): The statatistical procedure by which parameter estimation is done.
                                    - MAP      :  provides maximum a posteriori
                                    - MCMC     :  runs (expensive) Markov Chain Monte Carlo simulation of the aposteriori distribution
                                    - debaised :  provides estimate of aposteriori by leveraging asymptotic results, circumventing expensive MCMC

        Raises:
            NotImplementedError: If an unkwon method is called.
        """
        
        if(not(self.Xy_is_set)):
            self.setXy(X=X, y=y, k=self.k, intercept=self.intercept, standardizeX=self.standardizeX, translateY=self.translateY)
            self.Xy_is_set = True
        
        if(method=='MAP'):
            self.alphas = self.get_alpha_MAP(verbose=verbose)
        else:
            raise NotImplementedError("Only `MAP` estimation is available for SS-LASSO at this point.")

    def get_alpha_MAP(self,
                      verbose:bool              = False, 
                      tmp_dir:Union[Path, str]  = '/home/siebenschuh/Projects/Optimization_under_uncertainty/Experiments/Tmp',
                      script_name:str           = '/home/siebenschuh/Projects/Optimization_under_uncertainty/Experiments/Experiment_2/models/sslasso_map.r',
                      conda_env:str             = 'R',
                      conda_path:str            = '/soft/datascience/conda/2022-09-08/mconda3/condabin/conda'):
        """
        Wraps the MAP SSLASSO routine in R. Runs max a-posterior estimation for Spike & Slab LASSO returning an estimate for the coefficient vector alpha.
        (Source: https://cran.r-project.org/web/packages/SSLASSO/SSLASSO.pdf)

        Args:
            - X (np.array)       :    Design matrix (exogenous)
            - y (np.array)       :    Vector of observations corresponding to X
            - verbose (bool)     :    Indicates if more elaborate output is desired.
            - tmp_dir (Path)     :    Directory in which `X` and `y` are stored so the R script can pick them up.
            - script_name (str)  :    File name of the R script that runs SSLASSO.

        Returns:
            - 1d NumPy array representing the MAP of the coefficinet vector alpha.

        Raises:
            - AssertionError
        """

        # check input format
        assert isinstance(self.X, np.ndarray) and isinstance(self.y, np.ndarray), "Attributes `self.X` and `self.y` must be defined."
        assert self.X.ndim==2, "Design matrix `X` should be a 2d numpy array."
        assert self.y.ndim==1, "Vector of responses `y` should be a 1d numpy array."
        assert len(self.X)==len(self.y), "Lengths of design matrix `X` and observation vector `y` must coincide."
        assert os.path.isdir(tmp_dir), "Directory to store `X`, `y` temporarily does not exist."
        assert os.path.isfile(script_name), f"The R script `{script_name}` does not exist."
        assert conda_env in ['R'], "Only valid conda env containing R utilities is `R`."
        assert os.path.isfile(conda_path), f"`conda_path` is invalid. {conda_path} does not exist"

        # store data temporarily
        # - paths
        X_path = Path(tmp_dir) / 'X_tmp.txt'
        y_path = Path(tmp_dir) / 'y_tmp.txt'

        # - store
        np.savetxt(X_path, self.X)
        np.savetxt(y_path, self.y)

        # run R script (incl. conda environment activation)
        command = [
            conda_path, 'run', '-n', conda_env, 'Rscript',
            script_name, X_path, y_path
        ]

        # Run the command and capture the output
        output = None
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout
            error_output = result.stderr
        except subprocess.CalledProcessError as e:
            error_output = e.stderr

        # delete tmp files
        for file_path in [X_path, y_path]:
            if(os.path.isfile(file_path)):
                os.remove(file_path)

        # Display the captured output
        if(output):
            alpha = self.__parse_R_alpha__(output)
            if(verbose):
                print("Output ok.")
            
            return alpha

        if(verbose):
            print("Error Output:")
            print(error_output)

        return None

    def model_estimate(self,):
        """
        Runs get_alpha_MAP
        
        Args:
            -

        Raises:
            -
        """
        
        return self.get_alpha_MAP()