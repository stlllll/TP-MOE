"""
Created on Wed Feb 23 22:13:42 2022

SMC GP-MOE Code

@author: Michael Zhang, Taole Sha

Modified for TP-MOE on 07/06/2023
"""
from tp_base_v import TPBase
from mvt import multivariate_t, MultivariateTGenerator 
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.stats import lognorm
from utils import _unscaled_dist, pad_kernel_matrix
import numpy as np
import pdb
import pymp
from scipy.stats import invgamma
from scipy.stats import gamma
from scipy.stats import norm
import CRPS.CRPS as pscore
from scipy.stats import t
import matplotlib.pyplot as plt

class ParticleTPMOE(object):

    def __init__(self, rng, num_threads, X, Y, J, alpha, X_mean, prior_obs, 
                 nu, psi, alpha_a, alpha_b, mb_size):
        """
        Initialize the TP-MOE object.
        Parameters:
            rng: Numpy RandomState object, used to set random seed
            num_threads: int, number of cores to use for OpenMP process
            X: 1 x D numpy array, initial input data
            Y: 1 x 1 numpy array, initial output data
            J: int, number of particles to use
            alpha: positive float, initial concentration parameter value
            X_mean: 1 x D numpy array, prior mean of X
            prior_obs: positive int, prior number of observations for Normal-I.W. mixture. lambda 0
            nu: positive int > D - 1, prior degrees of freedom for I.W. distribution
            psi: D x D positive definite numpy array, prior covariance matrix for I.W. distribution
            alpha_a: positive float, prior shape of alpha
            alpha_b: positive float, prior scale of alpha
            mb_size: positive int, minibatch size; set to None to not use minibatching
        """


        # global settings
        self.rng = rng   #random state
        self.num_threads = num_threads   #number of cores used
        self.X = X  #inputs
        self.Y = Y  #outputs
        self.N, self.D = self.X.shape  #shape of inputs
        self.X_star = np.zeros((1,self.D))  #empty new input array
        self.J = J  #number of particles
        self.W = np.ones(self.J)/self.J #particle weights
        self.r = np.ones(self.J)   #unique r for each particle

        # dpmm hyper parameters
        self.K = 1    
        self.alpha = alpha*np.ones(self.J)   #alpha array for J particles
        self.X_mean = X_mean   #prior mean of x for all clusters, particles
        self.prior_obs = prior_obs  #number of prior observations, lambda_0
        self.nu = nu    #prior degree of freedom for student t likelihood
        self.psi = psi  #prior covariance matrix
        self.alpha_a = alpha_a  #prior a0 for alpha
        self.alpha_b = alpha_b  #prior b0 for alpha

        # dpmm parameters
        self.Z = np.zeros((self.J,1)).astype(int) #mixture indicator matrix
        self.max_Z = self.Z.max()  #number of different clusters
        self.K = np.ones(self.J,dtype=int)   #number of existing clusters for each particle
        self.Z_count = np.array([np.bincount(self.Z[j],   #count of observations in each mixture
                                    minlength=self.Z.max()) for j in range(self.J)])
        self.alpha = self.parallel_alpha()  #sample an alpha for each particle, 
                                            #alpha|a0,b0,z1
        self.dpmm_marg_LL = self.parallel_init_marg_LL() #dpmm marginal likelihood for each particle

        # gp parameters
        if mb_size is None:
            self.mb_size = np.inf #minibatch size
        else:
            self.mb_size = mb_size

        self.full_cov = False
        self.kernels = {j:None for j in range(self.J)} #a dictionary for j kernels
        out = self.parallel_model_init()
        self.kernels = {j: dict(out[j][2]) for j in range(self.J)}
        self.tp_marg_LL_k = {j:dict(out[j][1]) for j in range(self.J)}  #marginal likelihood for k th cluster
        self.tp_marg_LL = np.array([sum(self.tp_marg_LL_k[j].values()) for j in range(self.J)]) #marginal likelihood
        self.models = {j:dict(out[j][0]) for j in range(self.J)}  #hyperparameters
        self.df = {j:dict(out[j][4]) for j in range(self.J)}  #individual dof for each mixture
        self.amp = {j:dict(out[j][3]) for j in range(self.J)}   #overall scale
        
    def parallel_model_init(self):
        out = pymp.shared.dict() #a shared dictionary
        with pymp.Parallel(self.num_threads) as p: 
            for j in p.range(self.J):
                out[j] = self.model_init(j)
        return(out)
        
    def model_init(self,j):  #intialize for the j th particle
        """
        Initializes TPbase objects and optimizes hyperparameters
        Parameters:
            j: int, index for particle j
        """
        tp_model = {} 
        tp_amp = {}
        tp_marg_LL_k = {}
        tp_df = {}
        kernel_j_k = {0:_unscaled_dist(self.X)} #X: the intial 1xD data
        Y_k = np.copy(self.Y).reshape(-1,1) #change to ?x1 data
        X_k = np.copy(self.X)
        init_hyp = self.rng.gamma(1,1,size=1)  #initialize hyp
        r = self.r[j]
        init_noise = self.rng.normal(0,r**.5,size=1)
        init_hyp = np.concatenate([init_hyp,np.abs(init_noise),init_noise],axis=0)
        init_amp = 1

        df = 1
        
        tpb = TPBase(rng=self.rng, hyp=init_hyp, amp = init_amp, mb_weight=1.,
                     full_cov=self.full_cov,df=df,noise_sd = r**.5)  #an tpbase object

        tp_model[0], tp_marg_LL_k[0] = tpb.ess(kernel_j_k, Y_k)  #elliptical slice sampling
        tp_amp[0] = tpb.ss2(kernel_j_k, Y_k)
        
        tp_df[0] = self.sample_df(0, 30, df, tp_amp[0])

        return(tp_model, tp_marg_LL_k, kernel_j_k, tp_amp, tp_df)
    def parallel_init_marg_LL(self):  #for j particle
        out = pymp.shared.array((self.J,), dtype='float')
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):
                out[j] = self.init_marg_LL(j)
        return out
    
    def init_marg_LL(self, j):  #for a certain particle
        Z_count = (self.Z_count[j])
        Z = (self.Z[j])
        alpha = self.alpha[j]
        log_prob  = np.log(np.hstack((Z_count,alpha)) / (alpha + self.N))  #CRP probability of existing clusters and new
        log_prob += np.hstack([self.posterior_mvn_t(self.X[Z==k,:],self.X[0]) for k in range(log_prob.size)]) #multiply with the t likelihood
        marg_LL   = logsumexp(log_prob)
        return(marg_LL)

    def particle_update(self, X_star, Y_star):
        """
        Update the TP-MOE object with a new observation.
        Parameters:
            X_star: 1 x D numpy array, new input data
            Y_star: 1 x 1 numpy array, new output data
        """
        self.X_star = X_star
        self.X_star.shape[1] == self.D  #make sure the size is equal
        self.Y_star = Y_star
        self.alpha = self.parallel_alpha() #sample \alpha|z_{1:i-1}
        out = self.parallel_update()      #sample zi|\alpha^j, z1:i-1,x1:i-1
                                          #sample theta|X1:i,Y1:i
            #some updating after sampling
        self.X  = np.vstack((self.X,self.X_star))
        self.Y  = np.vstack((self.Y,self.Y_star))
        self.N += 1
        self.X_star = np.zeros((1,self.D))
        self.Z = np.array([out[j][0] for j in range(self.J)])
        self.max_Z = self.Z.max()
        self.K = np.array([np.unique(self.Z[j]).size for j in range(self.J)],dtype=int)
        self.Z_count = np.array([np.bincount(self.Z[j], minlength=self.max_Z+1) for j in range(self.J)])
        self.dpmm_marg_LL = np.array([out[j][1] for j in range(self.J)])
        self.W = np.log(self.W) + self.dpmm_marg_LL - self.tp_marg_LL
        self.models = {j:dict(out[j][2]) for j in range(self.J)}
        self.amp = {j:dict(out[j][5]) for j in range(self.J)}    #the overall scale
        self.df = {j:dict(out[j][6]) for j in range(self.J)}
        self.tp_marg_LL_k = {j:dict(out[j][3]) for j in range(self.J)}
        self.tp_marg_LL = np.array([sum(self.tp_marg_LL_k[j].values()) for j in range(self.J)])
        self.kernels = {j:dict(out[j][4]) for j in range(self.J)}        
        self.W = self.W + self.tp_marg_LL
        self.W = np.exp(self.W  - logsumexp(self.W))
        self.W = self.W / self.W.sum()
        ESS = 1./((self.W**2).sum())  #effective sample size
        print("ESS: %.2f" % ESS)
        if ESS < .5*self.J:  #resample the particles w.r.t their weights
            print("Resampling.")
            resample_idx = self.rng.choice(self.J,
                                            p=self.W,
                                            size=self.J)
            self.Z = self.Z[resample_idx]
            self.max_Z = self.Z.max()
            self.Z_count = np.array([np.bincount(self.Z[j], minlength=self.max_Z+1) for j in range(self.J)])
            self.dpmm_marg_LL = self.dpmm_marg_LL[resample_idx]
            self.tp_marg_LL_k = {idx:dict(self.tp_marg_LL_k[j]) for idx,j in enumerate(resample_idx)}
            self.models = {idx:dict(self.models[j]) for idx, j in enumerate(resample_idx)}
            self.amp = {idx:dict(self.amp[j]) for idx, j in enumerate(resample_idx)}
            self.df = {idx:dict(self.df[j]) for idx, j in enumerate(resample_idx)}
            self.kernels = {idx:dict(self.kernels[j]) for idx, j in enumerate(resample_idx)}
            self.tp_marg_LL = np.array([sum(self.tp_marg_LL_k[j].values()) for j in range(self.J)])
            self.alpha = self.alpha[resample_idx]
            self.W = (1./self.J)*np.ones(self.J)
            self.K = self.K[resample_idx]
            self.r = self.r[resample_idx]
        
        r = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                sum1 = 1
                for k in range(self.K[j]):
                    sum1 += self.models[j][k][2]**2
                r[j] = 1/self.rng.gamma((self.K[j]+1)/2,2/sum1)  #randomstate has no inversegamma, so use 1/gamma
        self.r = np.array([r[j] for j in range(self.J)])
        
    def sample_df(self,l_d, r_d, df, amp):  #slice sampler
        a = np.log(self.rng.uniform())
        y = self.log_posterior(df, amp)+a
        for i in range(50):
            U = self.rng.uniform(l_d, r_d)
            if y < self.log_posterior(U, amp):
                break
        return(U)
    
    def log_posterior(self, df, amp):
        marginal_likelihood = invgamma.logpdf(amp, a = df/2,loc=0, scale=df/2)
        return(gamma.logpdf(df, a = 2,loc=0, scale=10) + marginal_likelihood)

    def parallel_update(self):
        out = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                Z, log_norm = self.crp_predict(j)
                out[j] = self.tp_update(j, Z, log_norm)
        return(out)

    def crp_predict(self, j):  #CRP  zi|\alpha^j, z1:i-1,x1:i
        """
        Assigns new data sequentially to according to CRP and multivariate-t
        marginal likelihood
        """
        Z_count = self.Z_count[j][self.Z_count[j].nonzero()]
        Z = np.copy(self.Z[j])
        alpha = float(self.alpha[j])
        K = int(self.K[j])        
        log_prob  = np.log(np.hstack((Z_count,alpha)) / (alpha + self.N))        
        log_prob += [self.posterior_mvn_t(self.X[np.where(Z==k)],self.X_star[0]) for k in range(log_prob.size)]
        log_norm  = logsumexp(log_prob)  #log normalizing constant
        log_prob  = np.exp(log_prob-log_norm)
        Z_i = self.rng.multinomial(1,log_prob).argmax()
        #Z_i = 0
        Z = np.append(Z,Z_i)
        return(Z,log_norm)
    
    def tp_update(self, j, Z, log_norm):
        k = Z[-1]    #cluster indicator    
        K_mask = Z[:-1]==k    #same cluster observations' indicators
        if k > Z[:-1].max():  #if is a new cluster
            kernel_j_k = _unscaled_dist(self.X_star) #only one data, =0  
            init_hyp = self.rng.gamma(1,1,size=1)  #initialize hyp
            r = self.r[j]
            init_noise = self.rng.normal(0,r**.5,size=1)
            init_hyp = np.concatenate([init_hyp,np.abs(init_noise),init_noise],axis=0)
            init_amp = 1
            
            df = 1
            
            tpb = TPBase(rng=self.rng, hyp=init_hyp, amp = init_amp, mb_weight=1.,
                     full_cov=self.full_cov,df=df,noise_sd = r**.5)  
            models_j_k, tp_marg_LL_k = tpb.ess(kernel_j_k, self.Y_star) #First
            amp_j_k = tpb.ss2(kernel_j_k, self.Y_star)
            df_j_k = self.sample_df(0, 30, df, amp_j_k)
            
        else:
            X_k = np.copy(self.X[K_mask]) #previous same cluster inputs
            Y_k = np.copy(self.Y[K_mask]).reshape(-1,1)  #previous same cluster outputs      
            N_k = int(Y_k.size)       #number of previous same cluster observations     
            if N_k <= self.mb_size:  #no need to minibatch
                new_Y = np.vstack((Y_k,self.Y_star))  #new X, Y, kernel
                new_X = np.vstack((X_k,self.X_star))
                kernel_j_k = pad_kernel_matrix(self.kernels[j][k], X_k, self.X_star)    
                r = self.r[j]
                hyp = self.models[j][k]
                amp = self.amp[j][k]
                df=self.df[j][k]
                tpb = TPBase(rng=self.rng, hyp=hyp, amp = amp,
                             mb_weight=1.,
                             full_cov=self.full_cov,df=df,noise_sd = r**.5)            
            else:
                U = self.rng.choice(a=N_k, size=self.mb_size, replace=False) #use minibatch
                new_X = np.vstack((X_k[U], self.X_star)) #new X, Y, kernel
                new_Y = np.vstack((Y_k[U], self.Y_star))
                kernel_j_k = _unscaled_dist(new_X)
                mb_weight = float(N_k) / float(self.mb_size) 
                
                r = self.r[j]
                hyp = self.models[j][k]
                amp = self.amp[j][k]
                
                df=self.df[j][k]
                tpb = TPBase(rng=self.rng, hyp=hyp, amp = amp,
                             mb_weight=mb_weight,
                             full_cov=self.full_cov,df=df,noise_sd = r**.5)            
            models_j_k, tp_marg_LL_k = tpb.ess(kernel_j_k, new_Y) #elliptical slice sampling
            amp_j_k = tpb.ss2(kernel_j_k, new_Y)
            
            df_j_k = self.sample_df(0, 30, df, amp_j_k)
            
        model_j = dict(self.models[j])
        model_j[k] = models_j_k
        amp_j = dict(self.amp[j])
        amp_j[k] = amp_j_k
        df_j = dict(self.df[j])
        df_j[k] = df_j_k
        new_tp_marg_j = dict(self.tp_marg_LL_k[j])
        new_tp_marg_j[k] = tp_marg_LL_k
        kernel_j = dict(self.kernels[j])
        kernel_j[k] = kernel_j_k
        
        return(Z, log_norm, model_j, new_tp_marg_j, kernel_j, amp_j, df_j)

    def posterior_mvn_t(self,X_k,X_star_i):  
        """
        Calculates the multivariate-t distributed marginal likelihood of a
        for a NIW mixture model.
        Parameters:
            X_k: N_k x D numpy array, data assigned to cluster k
            X_star_i : 1 x D numpy array, likelihood calculated for this
                       observation
        """
        if X_k.shape[0] > 0:
            X_bar = X_k.mean(axis=0)  #some summary statistics
            diff = X_k - X_bar
            SSE = np.dot(diff.T,diff)
            N_k = X_k.shape[0]
            prior_diff = X_bar - self.X_mean
            SSE_prior = np.outer(prior_diff.T, prior_diff)
        else:
            X_bar = 0.
            SSE = 0.
            N_k = 0.
            SSE_prior = 0.

        mu_posterior = (self.prior_obs * self.X_mean) + (N_k * X_bar)
        mu_posterior /= (N_k + self.prior_obs)  #posterior mean mu
        nu_posterior = self.nu + N_k  
        lambda_posterior = self.prior_obs + N_k  #posterior lambda
        psi_posterior = self.psi + SSE  #posterior covariance psi
        psi_posterior += ((self.prior_obs * N_k) / (
                    self.prior_obs + N_k)) * SSE_prior
        psi_posterior *= (lambda_posterior + 1.) / (
                    lambda_posterior * (nu_posterior - self.D + 1.))
        df_posterior = (nu_posterior - self.D + 1.) #posterior degree of freedom nu
        return multivariate_t.logpdf(X_star_i, mu_posterior, psi_posterior,
                                     df_posterior)     

    def parallel_alpha(self): #sample alpha for each j
        alpha_array = pymp.shared.array((self.J,), dtype='float')
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):
                alpha_array[j] = self.alpha_resample(j)
        return(alpha_array)
    
    def alpha_resample(self,j):  #by a variable augmentation scheme
        """code to gibbs sample alpha"""
        K_j = self.K[j]  #number of existing clusters in j th particle
        eta = self.rng.beta(self.alpha[j] + 1,  #eta=rho|alpha
                              self.N)        
        ak1 = self.alpha_a + K_j - 1  
        pi = ak1 / (ak1 + self.N * (self.alpha_b - np.log(eta))) #pi_alpha
        a = self.alpha_a + K_j
        b = self.alpha_b - np.log(eta)
        gamma1 = self.rng.gamma(a, 1./ b) 
        gamma2 = self.rng.gamma(a - 1, 1./ b) 
        return(pi * gamma1 + (1 - pi) * gamma2)

    def predict(self, X_star, Y_star = False):     #weighted combination of prediction from different particles
        """
        Predict at test point X_star.
        Parameters:
            X_star: N_star x D numpy array, test data
        """
        pred_j = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                if Y_star:
                    pred_j[j] = self.predict_j(j, X_star, Y_star)
                else:
                    pred_j[j] = self.predict_j(j,X_star)
        mean = np.vstack([pred_j[j][0] for j in range(self.J)])
        var = np.vstack([pred_j[j][1] for j in range(self.J)]) 
        df = np.vstack([pred_j[j][2] for j in range(self.J)]) 
        mean = self.W.dot(mean) #weighted combination
        var = self.W.dot(var)
        df = self.W.dot(df)
        return(mean,var,df)
                      
    def predict_j(self, j, X_star):
        N_star, _ = X_star.shape  
        nnz       = self.Z_count[j].nonzero() #existing clusters of j th particle
        log_prob_N = np.zeros((N_star,nnz[0].size)) #log probability of test belong to which cluster
        for i,X_i in enumerate(X_star):
            log_prob  = np.log(self.Z_count[j][nnz])   
            log_prob += [self.posterior_mvn_t(self.X[np.where(self.Z[j]==k)],X_i) for k in nnz[0]]
            log_prob -= logsumexp(log_prob) #normalizing
            log_prob  = np.exp(log_prob) #probability
            log_prob[log_prob < .01] = 0
            log_prob /= log_prob.sum()
            log_prob_N[i] = log_prob  

        out_mean  = np.zeros((N_star,nnz[0].size)) #matrix
        out_var   = np.zeros((N_star,nnz[0].size))
        out_df   = np.zeros((N_star,nnz[0].size))

        for k in np.where(log_prob > 0)[0]:        
            K_mask = self.Z[j] == k  
            X_k = np.copy(self.X[K_mask])
            Y_k = np.copy(self.Y[K_mask]).reshape(-1,1)
            N_k = Y_k.size
            if N_k <= self.mb_size:  #no minibatch
                df=self.df[j][k]
                tpb = TPBase(rng=self.rng, 
                             hyp=self.models[j][k],amp = self.amp[j][k],
                             mb_weight=1.,
                             full_cov=self.full_cov,df=df)
                pred_TP_v, pred_TP_mean, pred_TP_cov = tpb.predict(X_k, 
                                                        X_star, 
                                                        Y_k, 
                                                        self.kernels[j][k])
                
            else:  #minibatch
                U = self.rng.choice(a=N_k, size=self.mb_size, replace=False)
                X_k_mb = X_k[U]
                Y_k_mb = Y_k[U]
                kernel_mb = _unscaled_dist(X_k_mb)
                mb_weight = float(N_k) / float(self.mb_size)
                df=self.df[j][k]
                tpb = TPBase(rng=self.rng, 
                             hyp=self.models[j][k],amp = self.amp[j][k],
                             mb_weight=mb_weight,
                             full_cov=self.full_cov,df=df)
                pred_TP_v, pred_TP_mean, pred_TP_cov = tpb.predict(X_k_mb, 
                                                        X_star, 
                                                        Y_k_mb, 
                                                        kernel_mb)
            pred_TP_v = np.array([pred_TP_v])
            out_mean[:,k] = pred_TP_mean.flatten()
            out_var[:,k]  = pred_TP_cov.flatten()
            out_df[:,k]  = pred_TP_v.flatten()
            
        pred_mean = np.sum(log_prob*out_mean, axis=1)
        pred_var  = np.sum(log_prob*out_var, axis=1)
        pred_df  = np.sum(log_prob*out_df, axis=1)
        return(pred_mean, pred_var, pred_df)


###################################Sample code for experiments on the motorcycle dataset############################
if __name__ == '__main__':
    from   numpy.random import RandomState
    import time
    rng  = RandomState(0)
    mvt = MultivariateTGenerator(seed=rng)
    motorcycle = np.loadtxt("motorcycle.txt")
    motorcycle[:,0] -= motorcycle[:,0].mean()
    motorcycle[:,0] /= np.sqrt(motorcycle[:,0].var())
    motorcycle[:,1] -= motorcycle[:,1].mean()
    motorcycle[:,1] /= np.sqrt(motorcycle[:,1].var())
    X = motorcycle[:,0][:,None]
    Y = motorcycle[:,1][:,None]
    Y -= Y.mean()
    Y /= Y.std()
    N = Y.size
    X = np.linspace(-1,1,N)[:,None]
    X -= X.mean()
    X /= X.std()
    tpmoe  = ParticleTPMOE(rng=rng,
                           num_threads=16,
                           X=X[0,None],
                           Y=Y[0,None],
                           J=100,
                           alpha=1, 
                           X_mean=np.zeros(1), 
                           prior_obs=1, 
                           nu=3, 
                           psi=.5*np.eye(1),
                           alpha_a=10,
                           alpha_b=1,
                           mb_size=50)

    pred_m = np.empty((0,1))
    pred_v = np.empty((0,1))
    pred_df = np.empty((0,1))
    pred_ll = 0
    runtime = []

    for i in range(1,N,1):
        m, v, df = tpmoe.predict(X[i,None])
        pred_m = np.vstack((pred_m,m))
        pred_v = np.vstack((pred_v,v))
        pred_df = np.vstack((pred_df,df))
        start=time.time()
        tpmoe.particle_update(X[i,None], Y[i,None])
        end_time=time.time()-start
        runtime.append(end_time)
        MSE = np.mean((m - Y[i])**2)
        print("Obs: %i\tPredict Time: %.2f\tMSE: %.2f" % (i, end_time, MSE))
        pred_ll += mvt.logpdf(Y[i, None], mean = m, shape = np.array([v]), df=df)
    for i in range(tpmoe.max_Z+1):
        plt.scatter(X[tpmoe.Z[np.argmax(tpmoe.W)]==i],Y[tpmoe.Z[np.argmax(tpmoe.W)]==i])
    plt.plot(np.array(X[1:]),pred_m, 'r-')
    plt.plot(np.array(X[1:]),pred_m+1.96*np.sqrt(pred_v),'k--')
    plt.plot(np.array(X[1:]),pred_m-1.96*np.sqrt(pred_v),'k--')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(-3.5, 2.5)
    #plt.savefig("output/motor_tp.png",format="png")
    plt.cla()
    print(np.mean((pred_m-Y[1:])**2))
    print(pred_ll)
    print(sum(runtime)/len(runtime))

"""
#############################For load datasets#########################
#1. single GP
from   numpy.random import RandomState
X = np.linspace(-1,1,100)[:, None]
norm_X = _unscaled_dist(X)
Kxx = np.exp(  -.5 * (norm_X/ 0.5 )**2)
Kxx += (0.5  + 1e-6)*np.eye(100)
rng1 = RandomState(1)
Y = rng1.multivariate_normal(np.zeros(100), Kxx)[:,None]
Y -= Y.mean()
Y /= Y.std()
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()
x = X[:,0][:, None]

#2. TP mixture
X = np.linspace(-1,1,100)[:, None]
X1 = X[:50]
X2 = X[50:]
norm_X1 = _unscaled_dist(X1)
Kxx1 = np.exp(  -.5 * (norm_X1/ 0.5 )**2)
Kxx1 += (0.5  + 1e-6)*np.eye(50)
norm_X2 = _unscaled_dist(X2)
Kxx2 = np.exp(  -.5 * (norm_X2/ 0.2 )**2)
Kxx2 += (0.2  + 1e-6)*np.eye(50)
Y1 = multivariate_t.rvs(mean=np.zeros(50), shape=Kxx1, df=5, size=1,random_state=1)[0][:, None]
Y2 = multivariate_t.rvs(mean=np.zeros(50), shape=Kxx2, df=2, size=1,random_state=1)[0][:, None]
Y = np.vstack((Y1,Y2))
plt.plot(X, Y)
Y -= Y.mean()
Y /= Y.std()
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()
x = X[:,0][:, None]

#3. motorcycle
motorcycle = np.loadtxt("data/motorcycle.txt")
motorcycle[:,0] -= motorcycle[:,0].mean()
motorcycle[:,0] /= np.sqrt(motorcycle[:,0].var())
motorcycle[:,1] -= motorcycle[:,1].mean()
motorcycle[:,1] /= np.sqrt(motorcycle[:,1].var())
X = motorcycle[:,0][:,None]
Y = motorcycle[:,1][:,None]
Y -= Y.mean()
Y /= Y.std()
Y=10*Y
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()

#4. Brent
Brent = np.loadtxt("data/Europe_Brent_Spot_Price_FOB_Daily.csv")
Brent = Brent[range(0, Brent.size, 8)]
Y = Brent[:,None]
Y -= Y.mean()
Y /= Y.std()
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()

#5. DJI
import pandas as pd
data = pd.read_csv('data/Processed_DJI.csv').iloc[np.array(range(0,1114,10))]
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,2][:,None]
X = np.array(data)[:,1][:,None]
x = X[:,0][:, None]

#6. Canada
import pandas as pd
data = pd.read_csv('data/co2_canada_diff.csv')
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,1][:,None]    #10
X = np.array(data)[:,0][:,None]
x = X[:,0][:, None]

#7. heart rate
data      = np.loadtxt("data/heart_rate.csv",skiprows=1,
                           delimiter=",")
N = data.shape[0]
sub = np.linspace(0,N-1,10000).astype(int)
data = data[sub]
data_mean = data.mean(axis=0)
data_std  = data.std(axis=0)
data     -= data.mean(axis=0)
data     /= data.std(axis=0)
Y         = data[:,-1][:,None]

X         = data[:,:-1]
N,D       = X.shape

#8. nile
Nile = np.array([1120.0, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140, 1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969, 831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759, 781, 865, 845, 944, 984, 897, 822, 1010, 771, 676, 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890, 744, 749, 838, 1050, 918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919, 718, 714,740])
Y = Nile[:,None]
Y -= Y.mean()
Y /= Y.std()
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()

#9. EUR-USD
eu = np.loadtxt("data/usd_exchange_rate.txt")
Y = eu[:,None]
Y -= Y.mean()
Y /= Y.std()
N = Y.size
X = np.linspace(-1,1,N)[:,None]
X -= X.mean()
X /= X.std()

#10. Dow
import pandas as pd
data = pd.read_csv('data/dow_jones_index.csv', sep=",")
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,2][:,None]
X =  np.hstack((np.array(data)[:,1][:,None], np.array(data)[:, 3:]))
x = X[:,0][:, None]

#11. Istanbul
import pandas as pd
data = pd.read_csv('data/data_akbilgic.csv')
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,1][:,None]
X =  np.hstack((np.array(data)[:,0][:,None], np.array(data)[:, 3:]))
x = X[:,0][:, None]

#12. Exchange
import pandas as pd
data = pd.read_csv('data/Foreign_Exchange_Rates.csv').iloc[np.array(range(0,5015,50))]
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,10][:,None]
X =  np.hstack((np.array(data)[:,2][:,None], np.array(data)[:, 3:10],np.array(data)[:, 11:]))
x = X[:,0][:, None]

#13. Wind
import pandas as pd
data = pd.read_csv('data/wind.csv', sep=",").iloc[np.array(range(0,43800,4))]
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,9][:,None]
X =  np.array(data)[:,:9]
x = X[:,0][:, None]

#14. Power
import pandas as pd
data = pd.read_csv('data/power.csv', sep=",").iloc[np.array(range(0,52416,5))]
data = (data - data.mean(axis=0))/data.std(axis=0)
Y = np.array(data)[:,6][:,None]
X =  np.array(data)[:,:6]
x = X[:,0][:, None]
"""
