import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

def barycentric_projection(X,y,coupling):
	y_aligned=y
	weights=np.sum(coupling, axis = 0)
	X_aligned=np.matmul(coupling, y) / weights[:, None]
		
	return X_aligned, y_aligned

def compute_graphDist(X,k):
	graph=kneighbors_graph(X,k, mode="connectivity", metric="correlation", include_self=True)
	X_shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
	# Deal with unconnected stuff (infinities):
	X_max=np.nanmax(X_shortestPath[X_shortestPath != np.inf])
	X_shortestPath[X_shortestPath > X_max] = X_max
	Cx=X_shortestPath/X_shortestPath.max()
		
	return Cx