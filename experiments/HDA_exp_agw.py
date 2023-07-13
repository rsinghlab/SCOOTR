#%% Reproduce HDA experiment of the paper
import os
import time
from random import *
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import sys
from agw_scootr import *
from functools import reduce
import matplotlib.pylab as pl
from sklearn import svm

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-d1", "--dataset1", type=str, default=None, required=False) 
parser.add_argument("-d2", "--dataset2", type=str, default=None, required=False)
parser.add_argument("-m", "--metric", type=str, default=None, required=False)
parser.add_argument("-s", "--scale", type=str, default=None, required=False)
parser.add_argument("-r1", "--reg1",  nargs='+',type=float, required=True)
parser.add_argument("-r2", "--reg2",  nargs='+',type=float, required=True)
parser.add_argument("-a", "--alpha",  nargs='+',type=float, required=True)
parser.add_argument("-nt", "--nTabs",  nargs='+',type=float, required=True)
################# READ IN THE INPUTS #########################
args = parser.parse_args()
dataset1=args.dataset1
dataset2=args.dataset2
metricFx=args.metric
alphas=args.alpha
reg1=args.reg1
reg2=args.reg2
alph=alphas[0]
n_samples_tab = args.nTabs
scale=args.scale
if scale=="True":
	scaleCost=True
elif scale=="False":
	scaleCost=False

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #
 
featuresToUse = ["CaffeNet4096", "GoogleNet1024"]# surf, CaffeNet4096, GoogleNet1024
numRepetition = 4
# n_samples_tab = [0] # nombre de samples par classe [0,1,3,5]

if dataset1=="caltech10":
	domainSourceNames = ['caltech10'] #['caltech10','amazon','webcam']
elif dataset1=="amazon":
	domainSourceNames = ['amazon'] #['caltech10','amazon','webcam']
elif dataset1=="webcam":
	domainSourceNames = ['webcam'] #['caltech10','amazon','webcam']

if dataset2=="caltech10":
	domainTargetNames = ['caltech10'] #['caltech10','amazon','webcam']
elif dataset2=="amazon":
	domainTargetNames = ['amazon'] #['caltech10','amazon','webcam']
elif dataset2=="webcam":
	domainTargetNames = ['webcam'] #['caltech10','amazon','webcam']

# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################
def generateSubset(X, Y, nPerClass):
	idx = []
	for c in np.unique(Y):
		idxClass = np.argwhere(Y == c).ravel()
		shuffle(idxClass)
		idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
	return (X[idx, :], Y[idx])


# ---------------------------- DATA Loading Part ------------------------------
tests = []
data_source = {}
data_target = {}

min_max_scaler = preprocessing.MinMaxScaler()

for sourceDomain in domainSourceNames:
	possible_data = loadmat(os.path.join("data/", "features", featuresToUse[0],
										 sourceDomain + '.mat'))
	if featuresToUse == "surf":
		# Normalize the surf histograms
		feat = (possible_data['fts'].astype(float) /
				np.tile(np.sum(possible_data['fts'], 1),
						(np.shape(possible_data['fts'])[1], 1)).T)
	else:
		feat = possible_data['fts'].astype(float)

	# Z-score
	#feat = preprocessing.scale(feat)
	#feat = min_max_scaler.fit_transform(feat)

	labels = possible_data['labels'].ravel()
	data_source[sourceDomain] = [feat, labels]

	for targetDomain in domainTargetNames:
		#if targetDomain is sourceDomain:
			possible_data = loadmat(os.path.join("data/", "features", featuresToUse[1],
												 targetDomain + '.mat'))
			if featuresToUse == "surf":
				# Normalize the surf histograms
				feat = (possible_data['fts'].astype(float) /
						np.tile(np.sum(possible_data['fts'], 1),
								(np.shape(possible_data['fts'])[1], 1)).T)
			else:
				feat = possible_data['fts'].astype(float)
	
			# Z-score
			#feat = preprocessing.scale(feat)
			#feat = min_max_scaler.fit_transform(feat)

			#feat=np.dot(np.diag(1./np.sum(feat,axis=1)),feat)
	
			labels = possible_data['labels'].ravel()
			data_target[targetDomain] = [feat, labels]
	
			perClassSource = 20
			if sourceDomain == 'dslr':
				perClassSource = 8
			tests.append([sourceDomain, targetDomain, perClassSource])

meansAcc = {}
stdsAcc = {}
totalTime = {}

print("Feature used for source: ", featuresToUse[0])
print("Feature used for target: ", featuresToUse[1])

#%%
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.neighbors import KNeighborsClassifier
enc = onehot(handle_unknown='ignore',sparse=False)


def comp_(v=1e6):
	def comp(x,y):
		if x==y or y==-1:
			return 0
		else:
			return v
	return comp


# def euclidean_distances(X, Y, squared=False):

# 	nx = ot.get_backend(X, Y)

# 	a2 = nx.einsum('ij,ij->i', X, X)
# 	b2 = nx.einsum('ij,ij->i', Y, Y)

# 	c = -2 * nx.dot(X, Y.T)
# 	c += a2[:, None]
# 	c += b2[None, :]

# 	c = nx.maximum(c, 0)

# 	if not squared:
# 		c = nx.sqrt(c)

# 	if X is Y:
# 		c = c * (1 - nx.eye(X.shape[0], type_as=c))

# 	return c

def dist(x1, x2=None, metric='euclidean', w=None):
	if x2 is None:
		x2 = x1
	if w is not None:
		return cdist(x1, x2, metric=metric, w=w)
	if metric=="sqeuclidean":
		C=cdist(x1, x2, metric="euclidean")
		return C**2
	return cdist(x1, x2, metric=metric)

def compute_cost_matrix(ys,yt,v=np.inf):
	M=dist(ys.reshape(-1,1),yt.reshape(-1,1),metric=comp_(v))
	return M
#%%
	
import ot 

# -------------------- Main testing loop --------------------------------------
# seeds=[0,1,5,10,55,101,123,999,1000,1234]

all_res_list=[]
mean_perfs_COOT=[]
r1s=[]
r2s=[]

for r1 in reg1:
	for r2 in reg2:
		all_results={}
		for n_samples in n_samples_tab:
			dict_tmp={}
			
			for test in tests:
				Sname = test[0]
				Tname = test[1]
				perClassSource = test[2]
				testName = Sname.upper()[:1] + '->' + Tname.upper()[:1]
				print(testName, end=" ")

				dict_tmp[testName] = {} 
				
				perf_baseline= []
				perf_COT = []
				time_COT = []

				
				# --------------------II. prepare data-------------------------------------
				Sx_tot = data_source[Sname][0]
				Sy_tot = data_source[Sname][1]
				Tx_tot = data_target[Tname][0]
				Ty_tot = data_target[Tname][1]
				
				
				for repe in range(numRepetition):
					# seed=seeds[repe]
					Sx, Sy = generateSubset(Sx_tot, Sy_tot, perClassSource)
					Tx, Ty = generateSubset(Tx_tot, Ty_tot, perClassSource)
					
					idx = np.random.permutation(Tx.shape[0])
					for i in range(Tx.shape[0]):
						Tx=Tx[idx,:]
						Ty=Ty[idx]
					
					#semi supervision
					nb_perclass = n_samples
					Sy_ss =-1*np.ones_like(Sy)    
					
					for c in np.unique(Sy):
						idx=np.where(Sy==c)[0]
						Sy_ss[idx[:nb_perclass]]=c
					
					M_lin = compute_cost_matrix(Ty,Sy_ss,v=1e2)
					# --- compuet baseline score by 1NN
					
					idx=np.where(Sy_ss!=-1)[0]
					idx_inv=np.where(Sy_ss==-1)[0]
					
					if nb_perclass!=0:
						neigh = KNeighborsClassifier(n_neighbors=3).fit(Sx[idx,:],Sy[idx])
						ys_estimated = neigh.predict(Sx[idx_inv,:])
						perf_baseline.append(100*np.mean(Sy[idx_inv]==ys_estimated))
						print('Accuracy 3NN on source (baseline): {:.2f}'.format(100*np.mean(Sy[idx_inv]==ys_estimated)))
					
					#print('mean perf',np.mean(r))
				
					# --------------------III. run experiments---------------------------------
				
		 
					# ------------------- COT -----------------------------------------------
					ot.tic()
					algo="sinkhorn"
					algo2="sinkhorn"
					if r1==0:
						algo="cg"
					if r1==10:
						algo="emd"
					if r2==0:
						algo2="emd"

					D1=dist(Sx,Sx)
					D2=dist(Tx,Tx)
					Tv, Tc, cost = agw_scootr(Sx,Tx, D1, D2,alpha=alph, niter=100, algo=algo, reg=r1, algo2=algo2, reg2=r2, 
						verbose=False, log=False, scaleCost=scaleCost, C_lin=[M_lin.T,None])
	

					time_COT.append(ot.toc())

					yt_onehot = enc.fit_transform(Ty.reshape(-1,1))
					ys_onehot_estimated = Tv.shape[0]*np.dot(Tv,yt_onehot)
					ys_estimated=enc.inverse_transform(ys_onehot_estimated).reshape(-1)
					
					perf=100*np.mean(Sy[idx_inv]==ys_estimated[idx_inv])
					perf_COT.append(perf)
					print('Accuracy AGW labelprop: {:.2f}'.format(perf))
				
		  
				if n_samples!=0:
					print('mean perf baseline= {:.2f} ({:.2f})'.format(np.mean(perf_baseline),np.std(perf_baseline)))
				print('mean perf AGW= {:.2f} ({:.2f})'.format(np.mean(perf_COT),np.std(perf_COT)))

				mean_perfs_COOT.append(np.mean(perf_COT))
				r1s.append(r1)
				r2s.append(r2)

				dict_tmp[testName]['baseline']=perf_baseline
				dict_tmp[testName]['AGW']=perf_COT
				dict_tmp[testName]['time_AGW']=time_COT

			all_results[n_samples] = dict_tmp
			all_res_list.append(all_results)

print("Best performance:",np.amax(mean_perfs_COOT))
idx_best=np.argmax(mean_perfs_COOT)
print("With hyperparameters:", alph, r1s[idx_best], r2s[idx_best])

# fname="res/AGW"+str(alph)+"_"str(metricFx)+"_"+str(scaleCost)+"_"+dataset1+"_"+dataset2+"_t"+str(n_samples_tab[0])+".npy"
# np.save('res/resAGW0sq_C_to_W_t0.npy',all_res_list)



