"""
Why does the cross-seed union beat the width-matched control? Two hypotheses:

  (a) genuine init diversity — different weight inits settle in different optima
      that cover different manifold regions;
  (b) wide-TopK underutilization — one very wide SAE leaves many atoms DEAD
      under a fixed small k, so N narrower SAEs simply utilize atoms better.

Decisive comparisons per (expansion, k) cell:

  single              1 SAE, width w
  union (diff-init)   N SAEs, width w, vary INIT seed, FIXED data order  <- paper's cross-seed
  union (same-init)   N SAEs, width w, FIXED init, vary DATA order only
  control             1 SAE, width N*w (same total atoms & compute)

plus alive-atom fraction of seeds vs control.

Reads:
  - control alive << seed alive             => hypothesis (b) is active
  - union(diff-init) ~= union(same-init)    => diversity source doesn't matter (b)
  - union(diff-init) >  union(same-init)    => init diversity is genuinely special (a)

See results/logs/mechanism.log for a saved run. Usage: python mechanism.py
"""
import time

import numpy as np

from synthetic import (build_dictionary, generate_dataset, probe_in_context,
                       first_of_shape, K_ACTIVE)
from train import train_topk_sae, alive_fraction
from sae import get_decoder, encode_sae
from metrics import greedy_codes, auc


def run():
    d, c, n_seeds = 128, 48, 3   # c: paper-matched crowding (was 256 — over-crowded)
    steps, n_samples, noise, batch = 1000, 20000, 0.05, 4096
    B, n_probe = 8, 512
    shapes = ["circle", "swiss_roll"]
    cells = [(4, 8), (4, 16), (8, 8), (8, 16)]
    DATA_FIXED, INIT_FIXED = 100, 0

    t0 = time.time()
    manifolds = build_dictionary(c=c, d=d, seed=0)
    X = generate_dataset(manifolds, n_samples, k_active=K_ACTIVE,
                         noise_sigma=noise, seed=1)
    tot = sum(m.e for m in manifolds)
    print(f"benchmark d={d} c={c} total dims={tot}  data {X.shape}\n")

    for exp, k in cells:
        d_sae = exp * d
        diff = [train_topk_sae(X, d_sae, k, seed=s, data_seed=DATA_FIXED,
                               steps=steps, batch=batch) for s in range(n_seeds)]
        same = [train_topk_sae(X, d_sae, k, seed=INIT_FIXED, data_seed=200 + s,
                               steps=steps, batch=batch) for s in range(n_seeds)]
        ctrl = train_topk_sae(X, d_sae * n_seeds, k, seed=0, data_seed=0,
                              steps=steps, batch=batch)

        a_seed = float(np.mean([alive_fraction(s, X) for s in diff]))
        a_ctrl = alive_fraction(ctrl, X)
        dec_d = [get_decoder(s) for s in diff]
        dec_s = [get_decoder(s) for s in same]
        ud, us, dc = np.vstack(dec_d), np.vstack(dec_s), get_decoder(ctrl)

        print(f"cell exp{exp} (d_sae {d_sae}, atoms/dim {d_sae/tot:.2f}) k{k}:  "
              f"alive seed={a_seed:.2f} ctrl={a_ctrl:.2f}  "
              f"-> eff atoms ctrl={a_ctrl*d_sae*n_seeds:.0f} "
              f"union={a_seed*d_sae*n_seeds:.0f}  (of {d_sae*n_seeds})")
        print(f"    {'shape':>11}{'single':>8}{'U_diff':>8}{'U_same':>8}"
              f"{'ctrl':>8}{'Ud-C':>8}{'Us-C':>8}{'Ud-Us':>8}")
        for shape in shapes:
            m = first_of_shape(manifolds, shape)
            probe, _ = probe_in_context(m, manifolds, k_active=K_ACTIVE,
                                        n_probe=n_probe, noise_sigma=0.0)
            cd = [encode_sae(s, probe) for s in diff]
            cs = [encode_sae(s, probe) for s in same]
            single = np.mean([auc(greedy_codes(probe, dec_d[i], cd[i], B), B)
                              for i in range(n_seeds)])
            u_diff = auc(greedy_codes(probe, ud, np.hstack(cd), B), B)
            u_same = auc(greedy_codes(probe, us, np.hstack(cs), B), B)
            c_auc = auc(greedy_codes(probe, dc, encode_sae(ctrl, probe), B), B)
            print(f"    {shape:>11}{single:>8.3f}{u_diff:>8.3f}{u_same:>8.3f}"
                  f"{c_auc:>8.3f}{u_diff-c_auc:>+8.3f}{u_same-c_auc:>+8.3f}"
                  f"{u_diff-u_same:>+8.3f}")
        print()
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
