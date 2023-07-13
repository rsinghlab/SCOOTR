from scootrv2 import *
from agw_scootr import *
adt=np.load("../data/cite_adt_1000cells.npy")
rna=np.load("../data/cite_rna_1000cells.npy")

#these datasets are of form [n_feat x n_samples] but the algorithm expects [n_samples x n_feat]
#So we transpose them below. Also, we found normalization useful for all OT applications on single-cell datasets (including SCOT and SCOTv2):
adt=normalize(np.transpose(adt))
rna=normalize(np.transpose(rna))

Ms=np.ones((adt.shape[0],rna.shape[0]))
np.fill_diagonal(Ms, 0.95)
# Compute alignments:
Ts, Tv, cost, log_out= agw_scootr(adt,rna, alpha=0.1, k=10, reg=0.001,reg2=0.005,verbose=True)

np.savetxt("cite_fgcoot_pi_feat.txt",Tv,delimiter="\t")
# Ts=sample correspondence matrix
# Tv= feature correspondence matrix

####Some notes on the parameters of agw_scootr() function:
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

cite_Tv=np.load("cite_coot_pi_feat.npy")
np.savetxt("cite_coot_pi_feat.txt",cite_Tv,delimiter="\t")
plt.clf()
plt.imshow(cite_Tv,cmap="Reds")
plt.show()








