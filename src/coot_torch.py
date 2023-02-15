import torch
import numpy as np
from functools import partial
from ot.lp import emd
from sklearn.preprocessing import normalize

def compute_kl(p, log_q):
    kl = (p * torch.log(p + 1.0 * (p==0))).sum() - (p * log_q).sum()
    return kl

def emd_solver(cost, p_np, device, dtype):
    p1, p2 = p_np
    cost_np = cost.detach().cpu().numpy()

    pi_np, log = emd(p1, p2, cost_np, log=True)
    f1 = torch.from_numpy(log["u"]).to(device).to(dtype)
    f2 = torch.from_numpy(log["v"]).to(device).to(dtype)
    pi = torch.from_numpy(pi_np).to(device).to(dtype)

    return (f1, f2), pi

def sinkhorn(cost, init_duals, tuple_log_p, eps, n_iters, tol, eval_freq):
    log_a, log_b, ab = tuple_log_p
    f, g = init_duals

    for idx in range(n_iters):
        f_prev = f.detach().clone()

        g = - ((f + log_a)[:, None] - cost / eps).logsumexp(dim=0)
        f = - ((g + log_b)[None, :] - cost / eps).logsumexp(dim=1)

        if (idx % eval_freq == 0) and (f - f_prev).abs().max().item() < tol:
            break

    pi = ab * (f[:, None] + g[None, :] - cost / eps).exp()

    return (f, g), pi

def get_cost(ot_cost, pi_samp, pi_feat, data, eps):
    eps_samp, eps_feat = eps
    log_pxy_samp, log_pxy_feat, D_samp, alpha_samp = data

    # UGW part
    cost = (ot_cost * pi_feat).sum()
    if alpha_samp != 0:
        cost = cost + alpha_samp * (D_samp * pi_samp).sum()

    # Entropic part
    ent_cost = cost
    if eps_samp != 0:
        ent_cost = ent_cost + eps_samp * compute_kl(pi_samp, log_pxy_samp)
    if eps_feat != 0:
        ent_cost = ent_cost + eps_feat * compute_kl(pi_feat, log_pxy_feat)

    return cost, ent_cost

def coot_solver(X, Y, px=(None, None), py=(None, None), eps=(1e-2, 1e-2), alpha=(1, 1), \
                D=(None, None), init_duals=(None, None), init_pi=(None, None), log=False, \
                verbose=False, early_stopping_tol=1e-6, eval_bcd=10, eval_sinkhorn=1, \
                tol_bcd=1e-7, nits_bcd=100, tol_sinkhorn=1e-7, nits_sinkhorn=500):
    """
    Parameters
    ----------
    X: matrix of size nx x dx. First input data.
    Y: matrix of size ny x dy. Second input data.
    D: matrix of size nx x ny. Sample matrix, in case of fused GW
    px: tuple of 2 vectors of length (nx, dx). Histograms assigned on rows and columns of X.
        Uniform distributions by default.
    py: tuple of 2 vectors of length (ny, dy). Histograms assigned on rows and columns of Y.
        Uniform distributions by default.
    eps: tuple of scalars. Regularisation parameters for entropic approximation of sample and feature couplings.
    alpha: tuple of scalars. Interpolation parameter for fused UGW w.r.t the sample and feature couplings.
    D: tuple of matrices of size (nx x ny) and (dx x dy). The linear terms in sinkhorn. 
        By default, set to None.
    init_duals: tuple of tuple of vectors of size (nx,ny) and (dx, dy) if not None.
        Initialisation of sample and feature dual vectors if using Sinkhorn algorithm.
    init_pi: tuple of tuple of matrices of size (nx,ny) and (dx, dy) if not None.
        Initialisation of sample and feature couplings.
    log: True if the cost is recorded, False otherwise.
    verbose: if True then print the recorded cost.
    early_stopping_tol: threshold for the early stopping.
    eval_bcd: multiplier of iteration at which the cost is calculated. For example, 
        if eval_bcd = 10, then the cost is calculated at iteration 10, 20, 30, etc...
    eval_bcd: multiplier of iteration at which the old and new duals are compared in the Sinkhorn 
        algorithm.
    tol_bcd: tolerance of BCD scheme.
    nits_bcd: number of BCD iterations.
    tol_sinkhorn: tolerance of Sinkhorn.
    nits_sinkhorn: number of Sinkhorn.

    Returns
    ----------
    pi_samp: matrix of size nx x ny. Sample matrix.
    pi_feat: matrix of size dx x dy. Feature matrix.

    if log is True, then return additionally a dictionary whose keys are:
        dual_samp: tuple of vectors of size (nx, ny). Pair of dual vectors 
            when solving OT problem w.r.t the sample coupling. 
        dual_feat: tuple of vectors of size (dx, dy). Pair of dual vectors 
            when solving OT problem w.r.t the feature coupling. 
        log_cost: list of costs (without taking into account the KL terms).
        log_ent_cost: list of entropic costs.
    """

    # constant data variables
    alpha_samp, alpha_feat = alpha
    D_samp, D_feat = D
    if D_samp is None or alpha_samp == 0:
        D_samp, alpha_samp = 0, 0
    if D_feat is None or alpha_feat == 0:
        D_feat, alpha_feat = 0, 0

    nx, dx = X.shape
    ny, dy = Y.shape
    device, dtype = X.device, X.dtype

    # histograms on rows and columns
    px_samp, px_feat = px
    py_samp, py_feat = py

    if px_samp is None:
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
        px_samp_np = np.ones(nx) / nx
    else:
        px_samp_np = px_samp.cpu().numpy()

    if px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
        px_feat_np = np.ones(dx) / dx
    else:
        px_feat_np = px_feat.cpu().numpy()

    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
        py_samp_np = np.ones(ny) / ny
    else:
        py_samp_np = py_samp.cpu().numpy()

    if py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy
        py_feat_np = np.ones(dy) / dy
    else:
        py_feat_np = py_feat.cpu().numpy()

    pxy_samp = px_samp[:, None] * py_samp[None, :]
    pxy_feat = px_feat[:, None] * py_feat[None, :]

    eps_samp, eps_feat = eps

    if eps_samp == 0:
        tuple_pxy_np_samp = (px_samp_np, py_samp_np)
    else:
        tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)

    if eps_feat == 0:
        tuple_pxy_np_feat = (px_feat_np, py_feat_np)
    else:
        tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    data = (pxy_samp.log(), pxy_feat.log(), D_samp, alpha_samp)

    X_sqr = X ** 2
    Y_sqr = Y ** 2
    XY_sqr = (X_sqr @ px_feat)[:,None] + (Y_sqr @ py_feat)[None,:] + alpha_samp * D_samp
    XY_sqr_T = (X_sqr.T @ px_samp)[:,None] + (Y_sqr.T @ py_samp)[None,:] + alpha_feat * D_feat

    # initialise coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy

    duals_samp, duals_feat = init_duals
    if eps_samp != 0 and duals_samp is None:
        duals_samp = (torch.zeros_like(px_samp), torch.zeros_like(py_samp))  # shape nx, ny
    if eps_feat != 0 and duals_feat is None:
        duals_feat = (torch.zeros_like(px_feat), torch.zeros_like(py_feat))  # shape dx, dy

    self_sinkhorn = partial(sinkhorn, n_iters=nits_sinkhorn, tol=tol_sinkhorn, eval_freq=eval_sinkhorn)
    self_emd_solver = partial(emd_solver, device=device, dtype=dtype)
    self_get_cost = partial(get_cost, data=data, eps=eps)

    # initialise log
    log_cost = []
    log_ent_cost = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_samp_prev = pi_samp.detach().clone()

        # Update pi (sample coupling)
        ot_cost = XY_sqr - 2 * X @ pi_feat @ Y.T # size nx x ny
        if eps_samp > 0:
            duals_samp, pi_samp = self_sinkhorn(ot_cost, duals_samp, tuple_log_pxy_samp, eps_samp)
        elif eps_samp == 0:
            duals_samp, pi_samp = self_emd_solver(ot_cost, tuple_pxy_np_samp)

        # Update pi_feat (feature coupling)
        ot_cost = XY_sqr_T - 2 * X.T @ pi_samp @ Y # size dx x dy
        if eps_feat > 0:
            duals_feat, pi_feat = self_sinkhorn(ot_cost, duals_feat, tuple_log_pxy_feat, eps_feat)
        elif eps_feat == 0:
            duals_feat, pi_feat = self_emd_solver(ot_cost, tuple_pxy_np_feat)

        if idx % eval_bcd == 0:
            # Update error
            err = (pi_samp - pi_samp_prev).abs().sum().item()
            cost, ent_cost = self_get_cost(ot_cost, pi_samp, pi_feat)
            log_cost.append(cost.item())
            log_ent_cost.append(ent_cost.item())

            if err < tol_bcd or abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_tol:
                break

            if verbose:
                print("Cost at iteration {}: {}".format(idx+1, cost.item()))

    if pi_samp.isnan().any() or pi_feat.isnan().any():
        print("There is NaN in coupling")
    
    if log:
        dict_log = {"duals_samp": duals_samp, \
                    "duals_feat": duals_feat, \
                    "log_cost": log_cost, \
                    "log_ent_cost": log_ent_cost[1:]}
        return pi_samp, pi_feat, dict_log
    else:
        return pi_samp, pi_feat

#Example alignment (1e-3 is the best hyperparam for sample alignment, I'd need to redo tuning for the feature eps):
methyl=torch.Tensor(normalize(np.genfromtxt("../data/scGEM_methyl.txt", delimiter=" ")))
rna=torch.Tensor(normalize(np.genfromtxt("../data/scGEM_rna.txt", delimiter=" ")))
pi_samp, pi_feat=coot_solver(rna,methyl,eps=(1e-3, 0.11))

import matplotlib.pyplot as plt
plt.imshow(pi_samp, cmap="Reds")
plt.show()
plt.clf()
plt.imshow(pi_feat, cmap="seismic")
plt.show()
