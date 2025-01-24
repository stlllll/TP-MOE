# TP-MOE

Codes for implementing the TP-MOE model intrduced in the paper titled "Online Student-$t$ Processes with an Overall-local Scale Structure for Modelling Non-stationary Data" which will appear in AISTATS 2025.

Kindly note that due to the usage of package pymp, you need to have a Linux system to achieve the parallel computation. You could download Ubuntu to deal with this if your computer is Windows, which is also adopted by the author.

To run the code, check the file pymp_TPMOE.py, at the bottom there are example experiment codes to help you load the datasets in the paper and run the experiments with TP-MOE.

Also note that when running it on multivariate datasets, change the paramters to:
tpmoe  = ParticleTPMOE(rng=rng,
                           num_threads=16,
                           X=X[0,None],
                           Y=Y[0,None],
                           J=100,
                           alpha=1, 
                           X_mean=np.zeros(######### the dimension of the input), 
                           prior_obs=1, 
                           nu=######### a number larger than the dimension of the input, 
                           psi=.5*np.eye(######### the dimension of the input),
                           alpha_a=10,
                           alpha_b=1,
                           mb_size=50)

If you want to use other kernal functions for the TP, modify the code of functions ss2, predict, marginal_likelihood in the file tp_base_v.py by yourself. The kernel adopted in this paper is the squared exponential kernel.

Abstract of the paper:

Mixture-of-expert (MOE) models are popular methods in machine learning, since they can model heterogeneous behaviour across the space of the data using an ensemble collection of learners. These models are especially useful for modelling dynamic data as time-dependent data often exhibit non-stationarity and heavy-tailed errors, which may be inappropriate to model with a typical single expert model. We propose a mixture of Student-$t$ processes with an adaptive structure for the covariance and noise behaviour for each mixture. Moreover, we use a sequential Monte Carlo (SMC) sampler to perform online inference as data arrive in real time. We demonstrate the superiority of our proposed approach over other models on synthetic and real-world datasets to prove the necessity of the novel method.
