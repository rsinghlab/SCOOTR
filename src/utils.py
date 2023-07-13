import numpy as np
import ot
import pickle
import random 
# For computing distances:
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

class StopError(Exception):
	pass
class NonConvergenceError(Exception):
	pass
class StopError(Exception):
	pass

def sinkhorn_scaling(a,b,K,numItermax=1000, stopThr=1e-9, verbose=False,log=False,always_raise=False, **kwargs):
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	K = np.asarray(K, dtype=np.float64)

	# init data
	Nini = len(a)
	Nfin = len(b)

	if len(b.shape) > 1:
		nbb = b.shape[1]
	else:
		nbb = 0

	if log:
		log = {'err': []}

	# we assume that no distances are null except those of the diagonal of
	# distances
	if nbb:
		u = np.ones((Nini, nbb)) / Nini
		v = np.ones((Nfin, nbb)) / Nfin
	else:
		u = np.ones(Nini) / Nini
		v = np.ones(Nfin) / Nfin

	# print(reg)
	# print(np.min(K))

	Kp = (1 / a).reshape(-1, 1) * K
	cpt = 0
	err = 1
	while (err > stopThr and cpt < numItermax):
		uprev = u
		vprev = v
		KtransposeU = np.dot(K.T, u)
		v = np.divide(b, KtransposeU)
		u = 1. / np.dot(Kp, v)

		zero_in_transp=np.any(KtransposeU == 0)
		nan_in_dual= np.any(np.isnan(u)) or np.any(np.isnan(v))
		inf_in_dual=np.any(np.isinf(u)) or np.any(np.isinf(v))
		if zero_in_transp or nan_in_dual or inf_in_dual:
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print('Warning: numerical errors at iteration in sinkhorn_scaling', cpt)
			#if zero_in_transp:
				#print('Zero in transp : ',KtransposeU)
			#if nan_in_dual:
				#print('Nan in dual')
				#print('u : ',u)
				#print('v : ',v)
				#print('KtransposeU ',KtransposeU)
				#print('K ',K)
				#print('M ',M)

			#    if always_raise:
			#        raise NanInDualError
			#if inf_in_dual:
			#    print('Inf in dual')
			u = uprev
			v = vprev

			break
		if cpt % 10 == 0:
			# we can speed up the process by checking for the error only all
			# the 10th iterations
			if nbb:
				err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
					np.sum((v - vprev)**2) / np.sum((v)**2)
			else:
				transp = u.reshape(-1, 1) * (K * v)
				err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
			if log:
				log['err'].append(err)

			if verbose:
				if cpt % 200 == 0:
					print(
						'{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))
		cpt = cpt + 1
	if log:
		log['u'] = u
		log['v'] = v

	if nbb:  # return only loss
		res = np.zeros((nbb))
		for i in range(nbb):
			res[i] = np.sum(
				u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
		if log:
			return res, log
		else:
			return res

	else:  # return OT matrix

		if log:
			return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
		else:
			return u.reshape((-1, 1)) * K * v.reshape((1, -1))

def barycentric_projection(source, target, couplingMatrix):
	"""
	Given: data in the target space, data in the source space, a coupling matrix learned via Gromow-Wasserstein OT
	Returns: source (target) matrix transported onto the target (source)
	"""
	P = (couplingMatrix.T/couplingMatrix.sum(1)).T
	transported_data= np.matmul(P, target)
	return transported_data

def compute_graph_distances(data, n_neighbors=5, mode="distance", metric="correlation"):
	"""
	
	"""
	graph=kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode, metric=metric, include_self=True)
	shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
	max_dist=np.nanmax(shortestPath[shortestPath != np.inf])
	shortestPath[shortestPath > max_dist] = max_dist

	return np.asarray(shortestPath)

def random_gamma_init(p,q, **kwargs):
	rvs=stats.beta(1e-1,1e-1).rvs
	S=random(len(p), len(q), density=1, data_rvs=rvs)
	return sinkhorn_scaling(p,q,S.A, **kwargs)

def init_matrix_np(X1, X2, v1, v2):
	def f1(a):
		return (a ** 2)

	def f2(b):
		return (b ** 2)

	def h1(a):
		return a

	def h2(b):
		return 2 * b

	constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
					 np.ones(f1(X2).shape[0]).reshape(1, -1))
	constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
					 np.dot(v2.reshape(1, -1), f2(X2).T))

	constC = constC1 + constC2
	hX1 = h1(X1)
	hX2 = h2(X2)

	return constC, hX1, hX2

def init_matrix_GW(C1,C2,p,q,loss_fun='square_loss'):
	""" 
	"""        
	if loss_fun == 'square_loss':
		def f1(a):
			return a**2 

		def f2(b):
			return b**2

		def h1(a):
			return a

		def h2(b):
			return 2*b

	constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
					 np.ones(len(q)).reshape(1, -1))
	constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
					 np.dot(q.reshape(1, -1), f2(C2).T))
	constC=constC1+constC2
	hC1 = h1(C1)
	hC2 = h2(C2)

	return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):
	""" 
	"""
	A=-np.dot(hC1, T).dot(hC2.T)
	tens = constC+A

	return tens

def dist(x1, x2=None, metric='sqeuclidean'):
	"""
	Compute distances between pairs of samples across x1 and x2 using scipy.spatial.distance.cdist
	If x2=None, x2=x1, then we compute intra-domain sample-sample distances

	Parameters
	----------
	x1 : np.array (n1,d)-- A matrix with n1 samples of size d
	x2 : np.array (n2,d)-- optional. Matrix with n2 samples of size d (if None, then x2=x1)
	metric : str or function -- distance metric, optional
		If a string, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
		'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
		 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'. Full list in the doc of scipy
	Returns
	-------
	M : np.array (n1,n2) -- Distance matrix computed with the given metric
	"""
	if x2 is None:
		x2 = x1

	return cdist(x1, x2, metric=metric)

def split_train_test(dataset,ratio=0.9, seed=None):
	idx_train = []
	X_train = []
	X_test = []
	random.seed(seed)
	for idx, val in random.sample(list(enumerate(dataset)),int(ratio*len(dataset))):
		idx_train.append(idx)
		X_train.append(val)
	idx_test=list(set(range(len(dataset))).difference(set(idx_train)))
	for idx in idx_test:
		X_test.append(dataset[idx])
	x_train,y_train=zip(*X_train)
	x_test,y_test=zip(*X_test)
	return np.array(x_train),np.array(y_train),np.array(idx_train),np.array(x_test),np.array(y_test),np.array(idx_test)    

