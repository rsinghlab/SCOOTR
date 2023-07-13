import numpy as np
import ot
from scipy import stats
from scipy.sparse import random
from utils import * 
from optim import *

def agw_scootr(X1, X2, D1, D2, w1 = None, w2 = None, v1 = None, v2 = None, 
				alpha=1.0, niter=10, algo='cg', reg=0, algo2='emd', reg2=0, verbose=True, log=False,
				scaleCost=False, random_init=False, C_lin=[None,None], supervised=False):
	""" 
	Parameters
	----------
	X1 : numpy array, shape (n, d)
		 Source dataset
	X2 : numpy array, shape (n', d')
		 Target dataset
	D1: numpy array, shape (n, n)
		 Intra-domain sample distances for X1 (structural info)
	D2: numpy array, shape (n', n')
		 Intra-domain sample distances for X2 (structural info)
	w1 : numpy array, shape (n,)
		Weight (histogram) on the samples of X1. If None uniform distribution is considered.
	w2 : numpy array, shape (n',)
		Weight (histogram) on the samples of X2. If None uniform distribution is considered.
	v1 : numpy array, shape (d,)
		Weight (histogram) on the features of X1. If None uniform distribution is considered.
	v2 : numpy array, shape (d',)
		Weight (histogram) on the features of X2. If None uniform distribution is considered.
	niter : integer
			Number max of iterations of the block coordinate descent for joint optimization.
	alpha: scalar, between 0 and 1.
			Strength of Gromovization. The sample cost will be calculated based on alpha*GW + (1-alpha)* COOT
	algo : string
			Choice of algorithm for solving OT problems on samples each iteration. Choice ['cg','sinkhorn'].
			If 'cg' returns sparse solution (via conditional gradient descent, similar to FGW)
			If 'sinkhorn' returns regularized solution
	algo2 : string
			Choice of algorithm for solving OT problems on features each iteration. Choice ['emd','sinkhorn'].
			If 'emd' returns sparse solution
			If 'sinkhorn' returns regularized solution
	reg : float
			Regularization parameter for samples coupling matrix. Ignored if algo='cg'
	reg2 : float
			Regularization parameter for features coupling matrix. Ignored if algo='emd'
	eps : float
		Threshold for the convergence
	random_init : bool
			Wether to use random initialization for the coupling matrices. If false identity couplings are considered.
	log : bool, optional
		 record log if True
	C_lin: numpy array, shape (n, n')
			Linear prior on the sample correspondences. Added to the cost for the samples transport
	C_mult: numpy array, shape (n, n')
			Multiplicative prior on the sample correspondences. Multiplied with the cost for the samples transport
	
	Returns
	-------
	Ts : numpy array, shape (n,n')
		   Optimal Transport coupling between the samples
	Tv : numpy array, shape (d,d')
		   Optimal Transport coupling between the features
	cost : float
			Optimization value after convergence
	log : dict
		convergence information and coupling marices

	Example
	----------
	import numpy as np
	from fgcoot import fgcoot_numpy
	
	n_samples=300
	Xs=np.random.rand(n_samples,2)
	Xt=np.random.rand(n_samples,1)
	Ts, Tv, cost = fgcoot_numpy(Xs,Xt, log=False)
	"""  
	if v1 is None:
	   v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
	if v2 is None:
	   v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
	if w1 is None:
	   w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
	if w2 is None:
	   w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

	if random_init:
		Ts=random_gamma_init(w1,w2) 
		Tv=random_gamma_init(v1,v2)

	else:
		Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
		Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
	
	constC_gw,hC1_gw,hC2_gw=init_matrix_GW(D1,D2,w1,w2)
	constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)
	constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
	cost = np.inf

	log_out ={}
	log_out['cost'] = []
	
	for i in range(niter):
		Tsold = Ts
		Tvold = Tv
		costold = cost

		M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
		#Adding supervision to the sample alignments:
		if supervised:
			M=M+C_lin[0]
		#Solving for the sample coupling:
		if algo=="cg":
			if scaleCost:
				M=(M-np.amin(M))/(np.amax(M)-np.amin(M))
			#note, the interpolation with gw is computed within cg(), and scaling of gw is also handled there beforehand if scaleCost=True.
			Ts=cg(w1,w2,M,alpha,G0=Ts,amijo=True,C1=D1,C2=D2, constC=constC_gw, hC1=hC1_gw, hC2=hC2_gw, scaleCost=scaleCost)
		elif algo=="sinkhorn":
			gw=ot.gromov.gwggrad(constC_gw,hC1_gw,hC2_gw,Ts)
			if scaleCost:
				M=(M-np.amin(M))/(np.amax(M)-np.amin(M))
				gw=(gw-np.amin(gw))/(np.amax(gw)-np.amin(gw))
			Ms=(alpha*gw)+((1-alpha)*M)
			Ts = ot.sinkhorn(w1,w2, Ms, reg)
		elif algo=="emd":
			gw=ot.gromov.gwggrad(constC_gw,hC1_gw,hC2_gw,Ts)
			if scaleCost:
				M=(M-np.amin(M))/(np.amax(M)-np.amin(M))
				gw=(gw-np.amin(gw))/(np.amax(gw)-np.amin(gw))
			Ms=(alpha*gw)+((1-alpha)*M)
			Ts = ot.emd(w1,w2, Ms, numItermax=1e7)
			
		else:
			raise Exception("The 'algo' parameter has to be one of 'cg' or 'sinkhorn' or 'emd'.")

		M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)
		#Adding supervision to the feature alignments:
		# if C_mult[1]!=None:
		# 	M=np.multiply(M,C_mult[1])
		#Solving for the sample coupling:
		if algo2 == 'emd':
			Tv = ot.emd(v1, v2, M, numItermax=1e7)
		elif algo2 == 'sinkhorn':
			Tv = ot.sinkhorn(v1,v2, M, reg2)
		else:
			raise Exception("The 'algo2' parameter has to be one of 'emd' or 'sinkhorn'.")
		delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
		cost = np.sum(M * Tv)
		
		#Documentation
		if log:
			log_out['cost'].append(cost)    
		if verbose:
			print('Delta: {0}  Loss: {1}'.format(delta, cost))

		#Check for convergence:
		if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
			if verbose:
				print('converged at iter ', i)
			break
	if log:
		return Ts, Tv, cost, log_out
	else:
		return Ts, Tv, cost


