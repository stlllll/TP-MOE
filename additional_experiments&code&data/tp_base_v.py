# -*- coding: utf-8 -*-
"""
A TP base

@author: Taole Sha, Michael Zhang
"""

import autograd.numpy as np 
from GPy.util.linalg import dtrtrs, tdot, jitchol, dpotrs
from utils import _unscaled_dist
from autograd import jacobian
from scipy.optimize import minimize
from autograd.scipy.stats import norm
from autograd.scipy.linalg import solve_triangular
import pdb
from scipy.special import gammaln
from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import lognorm
from scipy.optimize import minimize
from mvt import multivariate_t

def logexp(x): # reals -> positive reals
    _log_lim_val = np.log(np.finfo(np.float64).max)
    _lim_val = 36.0
    return np.where(x>_lim_val, x, np.log1p(np.exp(np.clip(x, -_log_lim_val, _lim_val)))) #+ epsilon

def inv_logexp(f): # positive reals -> reals
    _lim_val = 36.0
    return np.where(f>_lim_val, f, np.log(np.expm1(f)))

def jitchol_ag(A, maxtries=5): # autograd friendly version of jitchol
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise np.linalg.LinAlgError("not pd: non-positive diagonal elements")
    jitter = diagA.mean() * 1e-6
    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise np.linalg.LinAlgError("not positive definite, even with jitter.")    


class TPBase(object):
    def __init__(self, rng, hyp, amp, mb_weight=1, full_cov=False, LL=None, 
                 hyp_var = 10., df = 1, noise_sd = 1):
        self.rng = rng
        self.mb_weight = mb_weight
        self.hyp = np.array(hyp)
        self.amp = amp
        self.N_hyp = len(self.hyp)
        self.LL = LL
        self.full_cov = full_cov
        self.hyp_var = hyp_var
        self.df = df
        self.noise_sd = noise_sd


    def ess(self, norm_k, Y_k):   #fix the overall scale, sample the length-scale and the local scale of noise
        nu = self.rng.normal(scale=[np.sqrt(self.hyp_var), self.noise_sd],size = 2)
        
        log_u = np.log(self.rng.uniform())
        hyp1 = self.hyp[[0,2]].copy()
        
        hyp1[0] = inv_logexp(hyp1[0])
        
        LL = self.marginal_likelihood(hyp1, norm_k, Y_k) + log_u
        theta = self.rng.uniform(0.,2.*np.pi)
        theta_min = theta - 2.*np.pi
        theta_max = float(theta)
        hyp_proposal = hyp1 * np.cos(theta) + nu * np.sin(theta)
        proposal_LL = self.marginal_likelihood(hyp_proposal, norm_k, Y_k)
        while proposal_LL < LL:
            if theta < 0:
                theta_min = float(theta)
            else:
                theta_max = float(theta)
            theta = self.rng.uniform(theta_min,theta_max)
            hyp_proposal = hyp1 * np.cos(theta) + nu * np.sin(theta)
            proposal_LL = self.marginal_likelihood(hyp_proposal, norm_k, Y_k)
            
        hyp_proposal[0] = logexp(hyp_proposal[0])
        
        hyp_proposal = np.concatenate((hyp_proposal[[0]], np.abs(hyp_proposal[[1]]), hyp_proposal[[1]]))
        
        self.hyp = hyp_proposal
        self.LL = proposal_LL
        return(self.hyp, self.LL)
    
    def ss2(self, norm_k, Y_k):  #for the overall scale
        N_k = Y_k.size
        ls = self.hyp[0]
        noise = np.abs(self.hyp[1])
        if N_k > 1:
            kernel_k  = np.exp(  -.5 * (norm_k/ ls )**2)+((noise/self.mb_weight)  + 1e-6) * np.eye(N_k) #RBF kernel
            try:
                LW = jitchol_ag(kernel_k)  #cholesky factorization
            except np.linalg.LinAlgError: # if cholesky fails, give up
                return(-np.inf)
            alpha = solve_triangular(LW.T, solve_triangular(LW, Y_k, lower=1)) #dpotrs y^T K^-1 y
            sum1 = np.sum(alpha * Y_k)
            return 1/self.rng.gamma((self.df+N_k)/2, 2/(self.df + sum1) )
        else:
            kernel_k = 1  + (noise/self.mb_weight)
            return 1/self.rng.gamma((self.df+N_k)/2, 2/(self.df + (Y_k[0][0]**2 / kernel_k)) )

    
    def predict(self, X, Xnew, Y_k, norm_k):
        N_star = Xnew.shape[0]
        N_k = Y_k.size
        kernel_k = np.exp(  -.5 * (norm_k/ self.hyp[0] )**2)
        kernel_k += (self.hyp[1]/(self.mb_weight)  + 1e-6)*np.eye(N_k)
        woodbury_chol = jitchol(kernel_k)
        woodbury_vector, _ = dpotrs(woodbury_chol, 
                                    Y_k, 
                                    lower=1)

        Kx = np.exp(  -.5 * (_unscaled_dist(X,Xnew)/ self.hyp[0] )**2)
        mu = np.dot(Kx.T, woodbury_vector) 
            
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        
        
        if self.full_cov:
            norm_X_star = _unscaled_dist(Xnew)
            Kxx = np.exp(  -.5 * (norm_X_star/ self.hyp[0] )**2)
            Kxx += (self.hyp[1]/(self.mb_weight)  + 1e-6)*np.eye(N_star)
            
            beta1 = np.sum(Y_k.T * woodbury_vector.T)
            
            tmp = dtrtrs(woodbury_chol, Kx)[0]
            var = (Kxx - tdot(tmp.T))*(self.df+beta1)/(self.df+N_k)
            var = var
        else:
            Kxx = 1+(self.hyp[1])/(self.mb_weight)
            tmp = dtrtrs(woodbury_chol, Kx)[0]
            var = (Kxx - np.square(tmp).sum(0))[:, None]
            beta1 = np.sum(Y_k.T * woodbury_vector.T)
            var = (self.df+beta1)/(self.df+N_k)*var
            var = var
        
        df = self.df+N_k
        return df,mu, var

        
    def marginal_likelihood(self, hyp, norm_k, Y_k): 
        """
        Calculates the marginal likelihood of the TP model
        Parameters:
            hyp: 1 x 1 array, hyperparameters of the kernel, length scale, noise
            norm_k: N x N numpy array, pairwise distances of data
            Y_k: N_k x 1 numpy array, output data
        """
        ls = logexp(hyp[0])
        noise = np.abs(hyp[1])
        N_k  = Y_k.size #number of observations
        if N_k > 1:
            kernel_k  = np.exp(  -.5 * (norm_k/ ls )**2)+((noise/self.mb_weight)  + 1e-6) * np.eye(N_k) #RBF kernel
            try:
                LW = jitchol_ag(kernel_k)  #cholesky factorization
            except np.linalg.LinAlgError: # if cholesky fails, give up
                return(-np.inf)
            W_logdet = 2.*np.sum(np.log(np.diag(LW)))  #log|K|
            alpha = solve_triangular(LW.T, solve_triangular(LW, Y_k, lower=1)) #dpotrs y^T K^-1 y
            LL =  0.5*(-N_k *self.mb_weight * np.log(np.pi*2) - N_k*np.log(self.df/2) -  W_logdet + 2*gammaln((self.df+N_k)/2) - 2*gammaln((self.df)/2) - (self.df+N_k)*np.log(1+(np.sum(alpha * Y_k)/(self.df)))   )
        else:
            kernel_k = 1  + (noise/self.mb_weight)
            LL = 0.5*(self.mb_weight*-N_k* np.log(np.pi*2) - N_k*np.log(self.df/2) - np.log(kernel_k) + 2*gammaln((self.df+N_k)/2) - 2*gammaln((self.df)/2)- (self.df+N_k)*np.log(1+((Y_k[0][0]**2 / kernel_k)/self.df))  )

        return(LL)
