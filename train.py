"""
Minimal TopK SAE trainer.

Trains ``sae.BatchTopKSAE`` so checkpoints round-trip through ``sae.load_sae``.
Adds only a training loop: per-sample TopK (matches EleutherAI/sparsify and the
inference SAE), unit-norm decoder atoms, and dead-feature resampling so no atoms
are wasted.

``seed`` controls weight INIT; ``data_seed`` (defaults to ``seed``) controls
minibatch ORDER + resampling. Separating them lets us build a same-init /
different-data ensemble to isolate genuine init diversity from data-order
diversity (see mechanism.py).
"""
import os

import numpy as np
import torch

from sae import BatchTopKSAE


def train_topk_sae(X, d_sae, k, seed=0, data_seed=None, steps=3000, batch=4096,
                   lr=4e-4, resample_every=500, resample_until=0.6, device="cpu",
                   verbose=False):
    """Train a TopK SAE on activations X [N, d_in]. Returns an eval-mode SAE."""
    if data_seed is None:
        data_seed = seed
    init_gen = torch.Generator(device=device).manual_seed(int(seed))
    data_gen = torch.Generator(device=device).manual_seed(int(data_seed))
    d_in = X.shape[1]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    data_mean = Xt.mean(0)
    n = Xt.shape[0]

    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device)
    with torch.no_grad():
        W = torch.randn(d_in, d_sae, generator=init_gen, device=device)  # atoms=cols
        W /= W.norm(dim=0, keepdim=True).clamp_min(1e-8)
        sae.decoder.weight.copy_(W)
        sae.decoder.bias.copy_(data_mean)
        sae.encoder.weight.copy_(W.t().contiguous())         # tied init
        sae.encoder.bias.zero_()

    make_opt = lambda: torch.optim.Adam(sae.parameters(), lr=lr)
    opt = make_opt()
    denom = ((Xt - data_mean) ** 2).sum(1).mean().clamp_min(1e-8)  # for FVU log
    fired = torch.zeros(d_sae, dtype=torch.bool, device=device)

    for step in range(steps):
        x = Xt[torch.randint(0, n, (batch,), generator=data_gen, device=device)]
        z = sae.encode(x)
        xhat = sae.decode(z)
        loss = ((x - xhat) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w = sae.decoder.weight
            w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)   # unit-norm atoms
            fired |= (z > 0).any(0)

        resampling = resample_every and step < steps * resample_until
        if resampling and (step + 1) % resample_every == 0:
            with torch.no_grad():
                dead = (~fired).nonzero(as_tuple=True)[0]
                if len(dead):
                    xb = Xt[torch.randint(0, n, (batch,), generator=data_gen, device=device)]
                    err = ((xb - sae.decode(sae.encode(xb))) ** 2).sum(1)
                    pick = err.topk(min(len(dead), batch)).indices
                    dirs = xb[pick] - sae.decoder.bias
                    dirs /= dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    m = min(len(dead), len(pick))
                    di = dead[:m]
                    sae.decoder.weight[:, di] = dirs[:m].t()
                    sae.encoder.weight[di] = dirs[:m] * 0.1
                    sae.encoder.bias[di] = 0.0
                opt = make_opt()     # reset optimizer moments
                fired.zero_()

        if verbose and (step + 1) % max(1, steps // 5) == 0:
            fvu = (loss / denom).item()
            print(f"    seed{seed} step {step+1}/{steps} "
                  f"fvu={fvu:.3f} alive={int(fired.sum())}/{d_sae}")

    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


def save_sae(sae, path, k):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": sae.state_dict(),
                "model_config": {"d_in": sae.d_in, "d_sae": sae.d_sae,
                                 "k": int(k)}},
               path)


@torch.no_grad()
def alive_fraction(sae, X, n_sample=8192, device="cpu"):
    """Fraction of atoms that fire at least once over a sample of X."""
    x = torch.as_tensor(X[:n_sample], dtype=torch.float32, device=device)
    return float((sae.encode(x) > 0).any(0).float().mean())
