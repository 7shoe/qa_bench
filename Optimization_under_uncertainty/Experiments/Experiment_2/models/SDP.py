import scipy
import numpy as np
import cvxpy as cvx
from typing import Tuple, Optional
from itertools import combinations

from utils import expand


class SDP():
    def __init__(self, d:int, mode:str='max', n_sim:int=100, reg_lambda:float=0) -> None:
        """
        Instantiates a semidefinite program (SDP) with `d` dimensions and regularization term `reg_lambda` that runs `mode`! objective.

        Args:
            d (int)              :  Dimension of the (raw) input
            mode (str)           :  Type of optimization. Either `max` or `min`.
            n_sim (int)          :  Number of random projections computed in the geometric rounding procedure.
            reg_lambda (float)   :  Standard deviation of the distribution from which entries are sampled, i.e. empirical standard deviation before sparsification is applied

        Returns:
            NumPy array with the randomly sampled coefficients sorted by order (0 to incl. k)

        Raises:
            - AssertionError : If order of interaction is larger than the dimension
        """
        
        assert mode in ['max', 'min'], "Mode of optimization must be either `min` or `max`."
        assert d>0, "Input size must be positive."
        
        self.d = d
        self.mode = mode
        self.n_sim = n_sim
        self.reg_lambda = reg_lambda
        
        if(self.mode=='max'):
            self.f_opt = -np.inf
        else:
            self.f_opt = np.inf
        self.x_opt = None
        
    def compile_B_matrix(self, reg_lambda:float) -> None:
        """
        Compiles the quadratic matrix from the regression coefficient vector alpha
        (Source: Baptista, Poloczek (2018) : https://arxiv.org/pdf/1806.08838.pdf, p.4)

        Args:
            reg_lambda (float)   :  Positive regularization/penalty parameter with which lp (l1)-norm enters objective.

        Returns:
            -

        Raises:
            - AssertionError : If order of interaction is larger than the dimension
        """
        
        assert reg_lambda >= 0, "Regularization parameter `reg_lambda` should be non-negative float."
        
        # - b: linear term
        b = self.alpha[1:self.d+1]
        b -= reg_lambda
        
        # DEBUG
        # - add penalty
        
        # - a: quadratic terms
        a = self.alpha[1+self.d:] 
        # - indices for quadratic terms
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
        
        # LEGACY
        # A, c
        #A -=reg_lambda * np.eye(self.d)
        #c = 0.25*(b + A.T @ np.ones(self.d))

        # B
        #B = np.zeros((self.d+1, self.d+1))
        #B[0:-1,0:-1] = 0.25*A
        #B[-1,:-1] = c
        #B[:-1,-1] = c
        
        #self.B = B
        
    def setup_program(self,):
        '''
        Defines the optimization program (SDP) via objective and constraints that is to be solved in terms of a CVPXY instance.
        
        Args:
            -

        Returns:
            -

        Raises:
            - 
        '''
        
        # objective, constraints
        self.Xvar = cvx.Variable((self.d+1, self.d+1), PSD=True)
        self.obj = cvx.trace(self.At @ self.Xvar)
        self.cons = [cvx.diag(self.Xvar) == np.ones(self.d+1)]
        
        # problem
        if(self.mode=='max'):
            self.prob = cvx.Problem(cvx.Maximize(self.obj), self.cons)
        else:
            self.prob = cvx.Problem(cvx.Minimize(self.obj), self.cons)
        
    def optimize(self,) -> np.array:
        '''
        Solve semidefinite program SDP of opt! tr(B*X) s.t. diag(X)=[1,...,1], X PSD.

        Args:
            self  :    -

        Returns:
            Lower triangular matrix L (2d NumPy array) from Cholesky decomposition of X = L@L^T. Columns V[:,i]**2 have l2-norm of 1.0. 

        Raises:
            ValueError : Optimization failed. Check objective and constraints.
        '''
        
        # run optimization
        result = self.prob.solve()
        
        # check
        if(self.prob.status!=cvx.OPTIMAL):
            self.status = -1
            raise ValueError("Optimization failed.")
        
        # successful optim
        self.status = 0
        
        self.X_star = self.Xvar.value
        
        return
    
    def stable_cholesky_decompose(self, eps:float=1E-6, lower:bool=True) -> None:
        '''
        Cholesky decomposition of solution to SDP. Returns lower (or upper, respectively) triangular matrix of X = L@L^T.

        Args:
            eps (float)  :      Small, positive number added to the matrix to ensure it is PSD prior to Cholesky-decomposition
            lower (bool) :      Return lower (rather than upper) triangular matrix from the decomposition

        Returns:
            Lower (or upper) triangular matrix L. All columns V[:,i]**2 have l2-norm of 1.0. 

        Raises:
            LinAlgError : If matrix X_star is not PSD (despite transformation to ensure so). Increase eps. 
        '''
            
        assert self.status==0, ""
        assert eps>0, "Small numerical constant `eps` must be positive."
        
        # check numerical robustness
        eigvals, eigvecs = np.linalg.eigh(self.X_star)
        min_eigval = np.min(eigvals)
        if min_eigval < 0:
            perturbation = np.identity(self.d + 1) * (-min_eigval + eps)  # Apply a small perturbation
            X_star = np.array(self.X_star) + perturbation
        else:
            X_star = self.X_star

        # Cholesky: Translate back
        self.V_star = scipy.linalg.cholesky(X_star, lower=lower)
        
        return 
    
    def geometric_rounding(self, n_sim:int) -> tuple:
        '''
        Samples {0,1}-valued solution approximation via geometric random rounding 
        Source: Charikar,Wirth (2004) : Maximizing quadratic programs: extending Grothendieck’s inequality
        
        Args:
            eps (float)  :      Small, positive number added to the matrix to ensure it is PSD prior to Cholesky-decomposition
            lower (bool) :      Return lower (rather than upper) triangular matrix from the decomposition

        Returns:
            Lower (or upper) triangular matrix L. All columns V[:,i]**2 have l2-norm of 1.0. 

        Raises:
            LinAlgError : If matrix X_star is not PSD (despite transformation to ensure so). Increase eps. 
        
        '''
        
        # generate a single (very) high-dimensional Standard Normal vector
        r = np.random.randn((self.d+1)*n_sim)

        # - rescale
        mask = np.where(np.abs(r)>1)
        r[mask] = r[mask] / np.abs(r[mask])
        r = r.reshape(-1, self.d+1)
        
        # 
        x_cand = ((np.sign(np.dot(r,self.V_star))+1) / 2).round()

        # x_cand @ alpha
        X_cand = expand(x_cand[:,:-1], 2, True)

        # DEBUG
        #print('X_cand.shape: ', X_cand.shape)
        
        # optimal candidate
        if(self.mode=='max'):
            i_star = np.argmax(X_cand @ self.alpha)
        else:
            i_star = np.argmin(X_cand @ self.alpha)
        
        # (estimated) optimal function value
        f_current = X_cand[i_star,:] @ self.alpha

        # register best value
        if(self.mode=='max'):
            if(f_current > self.f_opt):
                self.f_opt = f_current
                self.x_opt = X_cand[i_star,:]
        else:
            if(f_current < self.f_opt):
                self.f_opt = f_current
                self.x_opt = X_cand[i_star,:]
        
        # DEBUG (only return non-expanded bin vector)
        return f_current, X_cand[i_star,1:1+self.d]
    
    def run(self, alpha:np.array,) -> Tuple[float, np.array]:
        '''
        Runs a single optimization stepfor the given coefficient vector `alpha` (correspond to the AFO).
        
        Args:
            alpha (np.array)                         :     Vector of regression coefficient estimates; implicitely assumed that it arose of order k=2.

        Returns:
            f_hat, x_opt (Tuple[float, np.array])    :     Estimated optimal next vector x (x_opt) and associated, estimated function value f_hat = \hat{f(x_opt)}

        Raises:
            LinAlgError : If matrix X_star is not PSD (despite transformation to ensure so). Increase eps. 
        
        '''
        
        # opt. problem parametrization
        self.alpha = alpha
        self.compile_B_matrix(self.reg_lambda)
        
        # optimization
        self.setup_program()
        self.optimize()
        
        # back-translation
        self.stable_cholesky_decompose()
        return self.geometric_rounding(self.n_sim)