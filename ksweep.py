"""
Sparsity sweep: where (in the paper's regime axis) does pooling seeds help?

Bhalla et al. show a manifold's fate is set by the TopK sparsity k: too low and the
manifold shatters, near k = manifold dimension a compact atom group captures it, too
high and it dilutes across many redundant atoms. This sweeps k on their synthetic
benchmark (48 manifolds in d=128, SAE width 512) and, at each k, reports the RAW
reconstruction scores (restricted-R^2 AUC over 8 atoms, averaged over all shapes) so
nothing is hidden inside a difference:

  single   : one seed's SAE
  union    : the pooled N seeds
  control  : one SAE of the same total width AND compute as the union
  rho      : R^2(e atoms)/R^2(plateau), single-seed capture compactness
             (~1 = compact capture; -> 0 = diluted slow ramp to a low ceiling)

Read it two honest ways:
  union - single   = the pooling benefit against a CONSISTENT baseline. It grows as k
                     pushes a single seed into dilution (rho falls) -- pooling helps
                     precisely where one SAE stops capturing compactly.
  union vs control = the resource-matched question. Note the control dips BELOW single
                     at high k: extra capacity at high sparsity fragments the manifold
                     instead of spanning it, so a wider SAE captures worse -- the paper's
                     dilution thesis in one line. (So union-minus-control mixes "union
                     wins" with "control fails"; we plot all three raw instead.)

Usage:  uv run ksweep.py            # full sweep, writes results/figures/ksweep_crossseed.png
        uv run ksweep.py --quick    # fewer k / replicates, for a smoke test
"""
import argparse
import time
from pathlib import Path

import numpy as np

from synthetic import (build_dictionary, generate_dataset, probe_in_context,
                       first_of_shape, K_ACTIVE)
from train import train_topk_sae
from sae import get_decoder, encode_sae
from metrics import greedy_codes, auc


def rho(curve, e):
    c = np.asarray(curve, float)
    if len(c) == 0 or c[-1] <= 1e-9:
        return np.nan
    return float(c[min(e, len(c)) - 1] / c[-1])


def one_replicate(k, r, manifolds, shapes, n_seeds, d_sae, steps, n_samples, B, maxk):
    X = generate_dataset(manifolds, n_samples, k_active=K_ACTIVE,
                         noise_sigma=0.05, seed=1000 + r)
    b = 1000 + r * 37
    seeds = [train_topk_sae(X, d_sae, k, seed=b + s, data_seed=b + s,
                            steps=steps, batch=4096) for s in range(n_seeds)]
    ctrl = train_topk_sae(X, d_sae * n_seeds, k, seed=b + 900, data_seed=b + 900,
                          steps=steps, batch=4096)
    decs = [get_decoder(s) for s in seeds]
    du, dc = np.vstack(decs), get_decoder(ctrl)
    sng, uni, ctl, rhos = [], [], [], []
    for shape in shapes:
        tgt = first_of_shape(manifolds, shape)
        probe, _ = probe_in_context(tgt, manifolds, k_active=K_ACTIVE,
                                    n_probe=512, noise_sigma=0.0, seed=7 + r)
        cs = [encode_sae(s, probe) for s in seeds]
        sng.append(np.mean([auc(greedy_codes(probe, decs[i], cs[i], B), B)
                            for i in range(n_seeds)]))
        uni.append(auc(greedy_codes(probe, du, np.hstack(cs), B), B))
        ctl.append(auc(greedy_codes(probe, dc, encode_sae(ctrl, probe), B), B))
        rhos.append(rho(greedy_codes(probe, decs[0], cs[0], maxk), tgt.e))
    return (float(np.mean(sng)), float(np.mean(uni)),
            float(np.mean(ctl)), float(np.nanmean(rhos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    d, n_mani, d_sae, n_seeds = 128, 48, 512, 3
    steps, n_samples, B, maxk = 800, 20000, 8, 48
    k_values = [4, 8, 12, 16, 20, 24, 28, 32]
    R = 3
    if args.quick:
        k_values, R, steps = [4, 16, 32], 2, 300

    manifolds = build_dictionary(c=n_mani, d=d, seed=0)
    shapes = sorted({m.shape for m in manifolds})
    print(f"benchmark: {n_mani} manifolds in d={d}, SAE width {d_sae}, "
          f"{n_seeds} seeds, R={R} replicates")
    print(f"{'k':>4}{'single':>9}{'union':>8}{'control':>9}{'U-single':>10}"
          f"{'rho':>7}  regime")

    rows = []
    for k in k_values:
        t0 = time.time()
        vals = np.array([one_replicate(k, r, manifolds, shapes, n_seeds, d_sae, steps,
                                       n_samples, B, maxk) for r in range(R)])
        sng, uni, ctl, rh = vals.mean(0)
        regime = ("capture" if rh > 0.6 else "dilution" if rh < 0.45 else "transition")
        rows.append((k, sng, uni, ctl, rh))
        print(f"{k:>4}{sng:>9.3f}{uni:>8.3f}{ctl:>9.3f}{uni-sng:>+10.3f}{rh:>7.2f}"
              f"  {regime}   ({time.time()-t0:.0f}s)")

    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = np.array([r[0] for r in rows])
    sng = np.array([r[1] for r in rows])
    uni = np.array([r[2] for r in rows])
    ctl = np.array([r[3] for r in rows])
    rh = np.array([r[4] for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    lu, = ax.plot(ks, uni, "o-", color="tab:red", lw=2, label="union (pooled seeds)")
    ls, = ax.plot(ks, sng, "^-", color="tab:green", label="single seed")
    lc, = ax.plot(ks, ctl, "v--", color="tab:orange", label="width-matched control")
    ax.set_xlabel("k  (TopK sparsity — the paper's regime axis)")
    ax.set_ylabel("restricted-R² capture (AUC over 8 atoms)")

    ax2 = ax.twinx()
    lr, = ax2.plot(ks, rh, "s:", color="tab:blue", alpha=0.5, label="rho (capture)")
    ax2.set_ylabel("rho  (1 = compact capture, -> 0 = dilution)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.05)

    ax.set_title("Pooling beats a single seed as k drives it into dilution;\n"
                 "the wide control dips below single (extra capacity fragments)")
    ax.legend([lu, ls, lc, lr],
              ["union (pooled seeds)", "single seed", "width-matched control",
               "rho (capture)"], fontsize=8, loc="center left")

    out = Path(__file__).resolve().parent / "results" / "figures" / "ksweep_crossseed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
