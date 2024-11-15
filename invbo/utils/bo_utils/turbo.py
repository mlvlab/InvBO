import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from .approximate_gp import *
from botorch.generation import MaxPosteriorSampling 
import numpy as np

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32 
    success_counter: int = 0
    success_tolerance: int = 10 
    best_value: float = -float("inf")
    restart_triggered: bool = False

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling 
    acqf="ts",  # "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
    cand_max = 5000,
    tr_cand=100,
):
    assert acqf in ("ts")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(cand_max, max(2000, 200 * X.shape[-1]))
    
    X = X.clone().cuda()
    dim = X.shape[-1] 
    sobol = SobolEngine(dim, scramble=True)
    prob_perturb = min(20.0 / dim, 1.0)
    
    with torch.no_grad():
        acq_values = []
        for x_cand in X:
            perturbation = sobol.draw(tr_cand).to(dtype=dtype).cuda()
            mask = (torch.rand(tr_cand, dim, dtype=dtype, device=device)<= prob_perturb)
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
            
            weights = torch.ones_like(x_cand)*8
            tr_lb = x_cand - weights * state.length / 2.0
            tr_ub = x_cand + weights * state.length / 2.0 
            pert = tr_lb + (tr_ub - tr_lb) * perturbation 
            
            local_X_cand = x_cand.expand(tr_cand, dim).clone()
            local_X_cand[mask] = pert[mask]
        
            top_acq_values, _ = TS(local_X_cand, model, 1)

            acq_values.append(top_acq_values)
        torch.cuda.empty_cache()

    max_acq_values_batch = torch.tensor(acq_values)
    scaled_max_acq_value_batch = (max_acq_values_batch - max_acq_values_batch.min()) / (max_acq_values_batch.max() - max_acq_values_batch.min())
    
    scaled_max_acq_value_batch = scaled_max_acq_value_batch * (Y.max() - Y.min())
    score = Y + scaled_max_acq_value_batch
    top_scores, tr_index = torch.topk(score, 1)

    X_next = torch.zeros([0])

    with torch.no_grad():
        for idx in tr_index:
            x_center = X[idx, :].clone()
            weights = torch.ones_like(x_center)*8 
            tr_lb = x_center - weights * state.length / 2.0
            tr_ub = x_center + weights * state.length / 2.0 

            tr_lb = tr_lb.cuda()
            tr_ub = tr_ub.cuda() 
            pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
            pert = tr_lb + (tr_ub - tr_lb) * pert 

            mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]
            
            top_acq_values, next_X  = TS(X_cand, model, batch_size)
            torch.cuda.empty_cache()

            X_next = torch.cat((X_next, next_X.cpu()))

    return X_next

def TS(X_cand, model, batch_size):
    posterior = model.posterior(X_cand)
    covar = posterior.lazy_covariance_matrix
    res = covar.zero_mean_mvn_samples(batch_size) + posterior.loc.unsqueeze(0)
    samples = res.view(torch.Size([batch_size]) + posterior.loc.shape)

    obj = samples.squeeze(-1) 
    _, idcs_full = torch.topk(obj, batch_size, dim=-1)
    ridx, cindx = torch.tril_indices(batch_size, batch_size)
    sub_idcs = idcs_full[ridx, ..., cindx]
    idcs = torch.zeros([0]).cuda()
    cnt = 1
    prev = 0
    for i in range(batch_size):
        next_idx = sub_idcs[prev:prev+cnt]
        prev += cnt
        cnt += 1
        idcs = torch.cat((idcs, next_idx[torch.where(torch.isin(next_idx, idcs) == False)[0][0]].unsqueeze(0)))
    idcs = idcs.long()
    acq_score = torch.gather(obj, dim=1, index=idcs.unsqueeze(-1))
    idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X_cand.size(-1))
    Xe = X_cand.expand(*obj.shape[1:], X_cand.size(-1))
    next_X = torch.gather(Xe, -2, idcs)

    return acq_score, next_X