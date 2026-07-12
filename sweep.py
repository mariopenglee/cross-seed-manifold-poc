"""
Sparsity x capacity sweep: is there ANY regime where cross-seed diversity beats
a width-matched single SAE?

For each (k_sae, expansion) cell we train N seeds + one N-x-width control on a
fixed harder benchmark (c=256), then summarize statistical manifold capture by
the AUC of the code-reconstruction VE curve over the first B atoms. Reports:

  U-S = union_auc - single_auc     (naive "cross-seed helps" — expected > 0)
  U-C = union_auc - control_auc     (the effect we care about: does seed diversity
                                     beat just-more-atoms?)

A '*' marks cells where U-C > 0.01. See results/logs/sweep.log for a saved run.

Usage: python sweep.py
"""
import time

import numpy as np

from synthetic import (build_dictionary, generate_dataset, probe_in_context,
                       first_of_shape, K_ACTIVE)
from train import train_topk_sae
from sae import get_decoder, encode_sae
from metrics import greedy_codes, auc


def eval_cell(saes, ctrl, decs, dec_union, dec_ctrl, manifolds, shapes, B, n_probe):
    res = {}
    for shape in shapes:
        m = first_of_shape(manifolds, shape)
        probe, _ = probe_in_context(m, manifolds, k_active=K_ACTIVE,
                                    n_probe=n_probe, noise_sigma=0.0)
        codes = [encode_sae(s, probe) for s in saes]
        single = np.mean([auc(greedy_codes(probe, decs[i], codes[i], B), B)
                          for i in range(len(saes))])
        union = auc(greedy_codes(probe, dec_union, np.hstack(codes), B), B)
        control = auc(greedy_codes(probe, dec_ctrl, encode_sae(ctrl, probe), B), B)
        res[shape] = (float(single), union, control)
    return res


def run():
    d, c, n_seeds = 128, 256, 3
    steps, n_samples, noise, batch = 1000, 20000, 0.05, 4096
    B, n_probe = 8, 512
    shapes = ["circle", "swiss_roll"]
    expansions = [2, 4, 8]
    ks = [4, 8, 16, 32]

    t0 = time.time()
    manifolds = build_dictionary(c=c, d=d, seed=0)
    total_dims = sum(m.e for m in manifolds)
    X = generate_dataset(manifolds, n_samples, k_active=K_ACTIVE,
                         noise_sigma=noise, seed=1)
    print(f"benchmark: d={d} c={c} total manifold dims={total_dims}  data {X.shape}\n")
    print(f"{'exp':>4}{'d_sae':>7}{'atoms/dim':>10}{'k':>4}{'shape':>11}"
          f"{'single':>8}{'union':>8}{'ctrl':>8}{'U-S':>8}{'U-C':>8}")

    for exp in expansions:
        d_sae = exp * d
        for k in ks:
            if k >= d_sae:
                continue
            saes = [train_topk_sae(X, d_sae, k, seed=s, steps=steps, batch=batch)
                    for s in range(n_seeds)]
            ctrl = train_topk_sae(X, d_sae * n_seeds, k, seed=0, steps=steps, batch=batch)
            decs = [get_decoder(s) for s in saes]
            dec_union, dec_ctrl = np.vstack(decs), get_decoder(ctrl)
            res = eval_cell(saes, ctrl, decs, dec_union, dec_ctrl,
                            manifolds, shapes, B, n_probe)
            for shape in shapes:
                sg, un, ct = res[shape]
                flag = " *" if (un - ct) > 0.01 else ""
                print(f"{exp:>4}{d_sae:>7}{d_sae/total_dims:>10.2f}{k:>4}"
                      f"{shape:>11}{sg:>8.3f}{un:>8.3f}{ct:>8.3f}"
                      f"{un-sg:>+8.3f}{un-ct:>+8.3f}{flag}")
        print()
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
