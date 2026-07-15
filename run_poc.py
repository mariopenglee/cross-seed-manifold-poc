"""
Cross-seed subspace-capture demo (single config).

Trains N TopK SAEs that differ ONLY in seed, plus one width-matched control SAE
(1 seed, N x width = same total atom count as the union). For each target
manifold it measures how fully the atoms capture the manifold subspace via two
greedy curves (geometric span + statistical code reconstruction), comparing:

    single seed   vs   cross-seed union   vs   width-matched control   vs   PCA

and writes one figure per manifold to results/figures/.

Usage:
  python run_poc.py                      # defaults (c=512)
  python run_poc.py --c 64 --expansion 8 --k 16 --n-seeds 5 --steps 2000 \
                    --shapes circle swiss_roll line
  python run_poc.py --quick              # tiny fast smoke run
"""
import argparse
import time
from pathlib import Path

import numpy as np

from synthetic import (build_dictionary, generate_dataset, probe_in_context,
                       first_of_shape, K_ACTIVE)
from train import train_topk_sae, save_sae
from sae import get_decoder, encode_sae
from metrics import find_support_greedy, greedy_codes

HERE = Path(__file__).resolve().parent


def _geo(probe, decoder, max_k):
    _, curve, _ = find_support_greedy(probe, decoder, max_k=max_k, var_threshold=1.0)
    return curve


def _pad(curve, L):
    curve = np.asarray(curve, float)
    if len(curve) == 0:
        return np.zeros(L)
    if len(curve) >= L:
        return curve[:L]
    return np.concatenate([curve, np.full(L - len(curve), curve[-1])])


def _atoms_to(curve, thresh):
    hit = np.where(np.asarray(curve) >= thresh)[0]
    return int(hit[0] + 1) if len(hit) else float("inf")


def analyze_shape(shape, manifolds, saes, ctrl, decs, dec_union, dec_ctrl, args, fig_dir):
    m = first_of_shape(manifolds, shape)
    probe, _ = probe_in_context(m, manifolds, k_active=K_ACTIVE,
                                n_probe=args.n_probe, noise_sigma=0.0)
    e, maxk = m.e, args.max_atoms

    g_single = [_geo(probe, d, maxk) for d in decs]
    g_union, g_ctrl = _geo(probe, dec_union, maxk), _geo(probe, dec_ctrl, maxk)
    Xc = probe - probe.mean(0)
    from sklearn.decomposition import PCA
    g_pca = np.cumsum(PCA(n_components=min(maxk, *Xc.shape)).fit(Xc).explained_variance_ratio_)

    codes_single = [encode_sae(s, probe) for s in saes]
    s_single = [greedy_codes(probe, decs[i], codes_single[i], maxk) for i in range(len(saes))]
    s_union = greedy_codes(probe, dec_union, np.hstack(codes_single), maxk)
    s_ctrl = greedy_codes(probe, dec_ctrl, encode_sae(ctrl, probe), maxk)

    L = maxk
    gs = np.stack([_pad(c, L) for c in g_single])
    ss = np.stack([_pad(c, L) for c in s_single])
    g_union, g_ctrl, g_pca = _pad(g_union, L), _pad(g_ctrl, L), _pad(g_pca, L)
    s_union, s_ctrl = _pad(s_union, L), _pad(s_ctrl, L)

    budgets = [b for b in sorted({e, 2 * e, 8, 16, 32}) if b <= L]
    hdr = "".join(f"{'VE@'+str(b):>8}" for b in budgets) + f"{'->0.9':>8}"

    def prow(name, curve):
        cells = "".join(f"{curve[b-1]:>8.3f}" for b in budgets)
        print(f"    {name:<16}{cells}{str(_atoms_to(curve, 0.9)):>8}")

    print(f"\n=== {shape}  (true subspace dim e={e}) ===")
    print(f"  GEOMETRIC (decoder span)\n    {'method':<16}{hdr}")
    prow("single(mean)", gs.mean(0)); prow("union", g_union)
    prow("control", g_ctrl); prow("PCA(optimal)", g_pca)
    print(f"  STATISTICAL (code reconstruction)\n    {'method':<16}{hdr}")
    prow("single(mean)", ss.mean(0)); prow("union", s_union); prow("control", s_ctrl)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, L + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (title, single, union, ctrl, pca) in zip(
        axes,
        [("Geometric (decoder span)", gs, g_union, g_ctrl, g_pca),
         ("Statistical (code reconstruction)", ss, s_union, s_ctrl, None)],
    ):
        ax.fill_between(x, single.min(0), single.max(0), color="tab:blue", alpha=0.15)
        ax.plot(x, single.mean(0), color="tab:blue", marker="o", ms=3, label="single seed (mean)")
        ax.plot(x, union, color="tab:red", marker="s", ms=3, lw=2, label=f"cross-seed union (N={len(saes)})")
        ax.plot(x, ctrl, color="tab:green", marker="^", ms=3, ls="--", label=f"control: 1 seed x{len(saes)} width")
        if pca is not None:
            ax.plot(x, pca, color="gray", ls=":", label="PCA (optimal linear)")
        ax.axvline(e, color="k", lw=0.7, alpha=0.4)
        ax.set(xlabel="# atoms selected", ylabel="variance explained",
               title=title, xscale="log")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{shape} — cross-seed manifold capture (d={args.d}, k_sae={args.k})")
    fig.tight_layout()
    out = fig_dir / f"{shape}_crossseed.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def run(args):
    device = "cpu"
    fig_dir = HERE / "results" / "figures"
    ckpt_dir = HERE / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    manifolds = build_dictionary(c=args.c, d=args.d, seed=0)
    X = generate_dataset(manifolds, args.n_samples, k_active=K_ACTIVE,
                         noise_sigma=args.noise, seed=1)
    print(f"data {X.shape}  signal RMS {np.sqrt((X**2).mean()):.2f}  ({time.time()-t0:.1f}s)")

    d_sae = args.expansion * args.d
    saes = []
    for s in range(args.n_seeds):
        ts = time.time()
        sae = train_topk_sae(X, d_sae, args.k, seed=s, steps=args.steps,
                             batch=args.batch, device=device, verbose=args.verbose)
        save_sae(sae, str(ckpt_dir / f"seed{s}.pt"), args.k)
        saes.append(sae)
        print(f"  seed{s}: width {d_sae} trained ({time.time()-ts:.1f}s)")

    ts = time.time()
    ctrl = train_topk_sae(X, d_sae * args.n_seeds, args.k, seed=0, steps=args.steps,
                          batch=args.batch, device=device, verbose=args.verbose)
    save_sae(ctrl, str(ckpt_dir / f"control_x{args.n_seeds}.pt"), args.k)
    print(f"  control: width {d_sae*args.n_seeds} trained ({time.time()-ts:.1f}s)")

    decs = [get_decoder(s) for s in saes]
    dec_union, dec_ctrl = np.vstack(decs), get_decoder(ctrl)
    for shape in args.shapes:
        analyze_shape(shape, manifolds, saes, ctrl, decs, dec_union, dec_ctrl, args, fig_dir)
    print(f"\ntotal {time.time()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--c", type=int, default=48)   # paper-matched crowding (was 512)
    p.add_argument("--n-samples", type=int, default=30000)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--expansion", type=int, default=4, help="d_sae = expansion * d")
    p.add_argument("--k", type=int, default=8, help="SAE TopK sparsity")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--shapes", nargs="*", default=["circle", "swiss_roll", "line"])
    p.add_argument("--n-probe", type=int, default=512)
    p.add_argument("--max-atoms", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny fast smoke run")
    a = p.parse_args()
    if a.quick:
        a.n_samples, a.steps, a.n_seeds, a.expansion = 4000, 300, 3, 4
        a.shapes = ["circle", "line"]
    run(a)


if __name__ == "__main__":
    main()
