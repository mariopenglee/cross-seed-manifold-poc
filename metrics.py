"""
Manifold subspace-capture metrics ("restricted R^2").

Given a manifold's activations and a set of SAE decoder atoms, how few atoms are
needed to reconstruct the manifold? Two greedy curves:

  find_support_greedy  -- GEOMETRIC: greedily add decoder DIRECTIONS that explain
                          the most remaining variance of the centered manifold
                          activations. Depends only on decoder atoms.
  greedy_codes         -- STATISTICAL: greedily add FEATURES whose actual centered
                          code contributions (z_i - <z_i>) d_i most reduce the
                          residual. Depends on the SAE codes.

Both are adapted from ``subspace_capture.py`` in goodfire-ai/sae-manifold (MIT;
Bhalla et al. 2026). ``greedy_codes`` is refactored to take a raw decoder [A, d_in]
and codes [N, A] so atoms can be POOLED across several SAEs (the cross-seed union).
``_detect_elbow`` carries a numpy>=2 fix (np.cross on 2-vectors was removed).
"""
import numpy as np


def _detect_elbow(curve, min_k=1):
    """Max-distance-to-chord elbow index on a monotone curve."""
    n = len(curve)
    if n <= min_k + 1:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(curve, dtype=float)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len
    diff = p0 - np.column_stack([x, y])           # [n, 2]
    dists = np.abs(line_unit[0] * diff[:, 1] - line_unit[1] * diff[:, 0])
    dists[:min_k] = -1
    return int(np.argmax(dists))


def find_support_greedy(activations, decoder, max_k=100, var_threshold=0.95):
    """GEOMETRIC greedy: pick decoder directions spanning the manifold variance.

    Returns (selected_indices, variance_explained_curve, elbow_k).
    """
    X = np.asarray(activations, dtype=np.float32)
    X = X - X.mean(0)
    total_ss = (X ** 2).sum()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    D_cand = np.asarray(decoder, dtype=np.float32)
    d_norms_sq = (D_cand ** 2).sum(1)
    alive = d_norms_sq > 1e-10

    selected, var_curve, residual = [], [], X.copy()
    for _ in range(max_k):
        projections = residual @ D_cand.T
        scores = (projections ** 2).sum(0) / d_norms_sq.clip(1e-10)
        scores[~alive] = -np.inf
        for i in selected:
            scores[i] = -np.inf
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected.append(best)
        _, s, Vt = np.linalg.svd(D_cand[selected], full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = X - (X @ basis.T) @ basis
        explained = 1.0 - (residual ** 2).sum() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected), var_curve, elbow_k


def greedy_codes(activations, decoder, codes, max_k=64, var_threshold=1.0):
    """STATISTICAL greedy: pick features whose centered codes reconstruct the
    manifold. ``decoder`` [A, d_in], ``codes`` [N, A] may pool several SAEs.

    Returns the variance-explained curve (one entry per selected feature).
    """
    X_c = np.asarray(activations, np.float32)
    X_c = X_c - X_c.mean(0)
    total = float((X_c ** 2).sum())
    if total < 1e-10:
        return np.array([])
    Z = np.asarray(codes, np.float32)
    cand = np.where((Z > 0).any(0))[0]
    if len(cand) == 0:
        return np.array([])
    Z_c = Z[:, cand] - Z[:, cand].mean(0)
    D_cand = decoder[cand]
    contrib_ss = (Z_c ** 2).sum(0) * (D_cand ** 2).sum(1)

    sel_local, curve = [], []
    residual = X_c.copy()
    alive = np.ones(len(cand), bool)
    for _ in range(max_k):
        cross = (residual @ D_cand.T) * Z_c
        scores = 2 * cross.sum(0) - contrib_ss
        scores[~alive] = -np.inf
        b = int(np.argmax(scores))
        if scores[b] <= 0:
            break
        sel_local.append(b)
        alive[b] = False
        S = np.array(sel_local)
        residual = X_c - Z_c[:, S] @ D_cand[S]
        curve.append(1.0 - float((residual ** 2).sum()) / total)
        if curve[-1] >= var_threshold:
            break
    return np.array(curve)


def greedy_codes_reconstruct(activations, decoder, codes, B):
    """Greedy code selection of up to B features; returns (reconstruction, R2, indices).

    Same selection rule as ``greedy_codes`` but also builds the reconstruction
    ``mean + sum_i (z_i - <z_i>) d_i`` over the chosen features (for visualization),
    and its restricted R2 = 1 - ||residual||^2 / ||centered||^2.
    """
    X = np.asarray(activations, np.float32)
    mean = X.mean(0)
    X_c = X - mean
    tot = float((X_c ** 2).sum())
    Z = np.asarray(codes, np.float32)
    cand = np.where((Z > 0).any(0))[0]
    if len(cand) == 0 or tot < 1e-12:
        return np.repeat(mean[None], len(X), 0), 0.0, []
    Z_c = Z[:, cand] - Z[:, cand].mean(0)
    D_cand = decoder[cand]
    contrib_ss = (Z_c ** 2).sum(0) * (D_cand ** 2).sum(1)
    sel, residual, alive = [], X_c.copy(), np.ones(len(cand), bool)
    for _ in range(int(B)):
        scores = 2 * ((residual @ D_cand.T) * Z_c).sum(0) - contrib_ss
        scores[~alive] = -np.inf
        b = int(np.argmax(scores))
        if scores[b] <= 0:
            break
        sel.append(b)
        alive[b] = False
        residual = X_c - Z_c[:, sel] @ D_cand[sel]
    if sel:
        recon = mean + Z_c[:, sel] @ D_cand[sel]
    else:
        recon = np.repeat(mean[None], len(X), 0)
    r2 = 1.0 - float((residual ** 2).sum()) / tot
    return recon, r2, [int(cand[i]) for i in sel]


def auc(curve, B):
    """Mean variance-explained over the first B atoms (a scalar capture score)."""
    c = np.asarray(curve, float)
    if len(c) == 0:
        return 0.0
    c = c[:B] if len(c) >= B else np.concatenate([c, np.full(B - len(c), c[-1])])
    return float(c.mean())
