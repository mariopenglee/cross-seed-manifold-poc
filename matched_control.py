"""
Final confound: the union fires N*k active atoms per sample (each of N seeds
contributes its own k), while the standard control fires only k. So part of the
union's win could be pure effective-sparsity, not structure.

Clean test: compare the diff-init union against a single wide SAE (dictionary
N*w) with k' = N*k active atoms — matched active-atom budget AND dictionary size.

  union            N SAEs, width w, k active each  -> up to N*k active / sample
  ctrl_k           1 SAE, width N*w, k active       (previous control)
  ctrl_Nk          1 SAE, width N*w, k'=N*k active  (matched active budget)

Read:
  union > ctrl_Nk  => genuine structural gain from independent SAEs / seeds
  union ~ ctrl_Nk  => the win was mostly effective sparsity (more active atoms)

See results/logs/matched.log for a saved run. Usage: python matched_control.py
"""
import time

import numpy as np

from synthetic import (build_dictionary, generate_dataset, probe_in_context,
                       first_of_shape, K_ACTIVE)
from train import train_topk_sae, alive_fraction
from sae import get_decoder, encode_sae
from metrics import greedy_codes, auc


def cap(probe, decoder, codes, B):
    return auc(greedy_codes(probe, decoder, codes, B), B)


def run():
    d, c, n_seeds = 128, 48, 3   # c: paper-matched crowding (was 256 — over-crowded)
    steps, n_samples, noise, batch = 1000, 20000, 0.05, 4096
    B, n_probe = 8, 512
    shapes = ["circle", "swiss_roll"]
    cells = [(4, 16), (8, 8), (8, 16)]
    DATA_FIXED = 100

    t0 = time.time()
    manifolds = build_dictionary(c=c, d=d, seed=0)
    X = generate_dataset(manifolds, n_samples, k_active=K_ACTIVE,
                         noise_sigma=noise, seed=1)
    tot = sum(m.e for m in manifolds)
    print(f"benchmark d={d} c={c} total dims={tot}  data {X.shape}\n")

    for exp, k in cells:
        d_sae = exp * d
        wide, kNk = d_sae * n_seeds, k * n_seeds
        union = [train_topk_sae(X, d_sae, k, seed=s, data_seed=DATA_FIXED,
                                steps=steps, batch=batch) for s in range(n_seeds)]
        ctrl_k = train_topk_sae(X, wide, k, seed=0, data_seed=0, steps=steps, batch=batch)
        ctrl_Nk = train_topk_sae(X, wide, kNk, seed=0, data_seed=0, steps=steps, batch=batch)

        dec_u = [get_decoder(s) for s in union]
        du, d_k, d_Nk = np.vstack(dec_u), get_decoder(ctrl_k), get_decoder(ctrl_Nk)

        print(f"cell exp{exp} (atoms/dim {d_sae/tot:.2f}) k{k}:  "
              f"union N*k active={kNk}  ctrl_Nk k'={kNk}  "
              f"alive ctrl_k={alive_fraction(ctrl_k, X):.2f} "
              f"ctrl_Nk={alive_fraction(ctrl_Nk, X):.2f}")
        print(f"    {'shape':>11}{'single':>8}{'union':>8}{'ctrl_k':>8}"
              f"{'ctrl_Nk':>9}{'U-ck':>8}{'U-cNk':>8}")
        for shape in shapes:
            m = first_of_shape(manifolds, shape)
            probe, _ = probe_in_context(m, manifolds, k_active=K_ACTIVE,
                                        n_probe=n_probe, noise_sigma=0.0)
            cu = [encode_sae(s, probe) for s in union]
            single = np.mean([cap(probe, dec_u[i], cu[i], B) for i in range(n_seeds)])
            u = cap(probe, du, np.hstack(cu), B)
            ck = cap(probe, d_k, encode_sae(ctrl_k, probe), B)
            cNk = cap(probe, d_Nk, encode_sae(ctrl_Nk, probe), B)
            print(f"    {shape:>11}{single:>8.3f}{u:>8.3f}{ck:>8.3f}"
                  f"{cNk:>9.3f}{u-ck:>+8.3f}{u-cNk:>+8.3f}")
        print()
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
