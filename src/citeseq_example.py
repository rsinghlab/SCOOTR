from scootrv2 import *

adt=np.genfromtxt("../data/citeseq_adt_normalized_1000cells.csv", delimiter=",")
rna=unit_normalize(np.genfromtxt("../data/citeseq_rna_normalizedFC_1000cells.csv", delimiter=","))
#these datasets are of form [n_feat x n_samples] but the algorithm expects [n_samples x n_feat]
#So we transpose them below. Also, we found normalization useful for all OT applications on single-cell datasets (including SCOT and SCOTv2):
adt=unit_normalize(np.transpose(adt))
rna=unit_normalize(np.transpose(rna))

# Compute alignments:
Ts, Tv, cost, log_out= scootr(adt,rna, alpha=0.5, k=25, reg=0.001,reg2=0.1,verbose=True)
# Ts=sample correspondence matrix
# Tv= feature correspondence matrix

####Some notes on the parameters of scootr() function:
# alpha= interpolation coefficient between the graph distances and euclidean distances between samples. Default=0.5
# k= number of neighbors to use in kNN graph when computing graph distances for samples
# reg= epsilon / coefficient of entropic regularization for SAMPLE alignments
# reg2= epsilon / coefficient of entropic regularization for FEATURE alignments


### EVALUATE THE ALIGNMENT:
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from utils import *
from evals import *
#Perform barycentric projection based on sammple alignment matrix, Ts:
adt_aligned,rna_aligned=barycentric_projection(adt,rna,Ts)
# Compute mean FOSCTTM (lower the better, ranges between 0-1)
print(np.mean(calc_domainAveraged_FOSCTTM(adt_aligned,rna_aligned)))

# Visualize feature alignments (alignments should roughly lie along the diagonal):
plt.imshow(Tv,cmap="Reds")
plt.show()

