"""
Synthetic manifold-superposition benchmark.

Reproduces the controlled setup from Section 4.2 / Appendix E of
"Do Sparse Autoencoders Capture Concept Manifolds?":

    x = sum_{i in S} (p_i @ V_i) * scale_i  +  eps

A ground-truth dictionary of ``C`` manifolds, each a low-dimensional shape whose
natural (canonical) embedding is mapped into R^D by a random *orthonormal* matrix
``V_i`` (rows orthonormal, so the manifold occupies a random e_i-dimensional
linear subspace of R^D), then isotropically rescaled to unit RMS norm. Every
sample activates ``K_ACTIVE`` manifolds chosen at random (sparse superposition),
each at a random point on its manifold, plus Gaussian observation noise.

Paper-confirmed values (Fig. 4): D=128, C=512, K_ACTIVE=4, eight shapes of
intrinsic dim 1-2. Everything the paper leaves unspecified (noise scale, sample
counts, canonical coordinate distributions, per-shape variant params) is OUR
choice and flagged as such.

Note on ``V_i``'s row count: we use each shape's *linear span* dimension e_i
(the number of coordinates in its canonical embedding: 2 for a circle, 3 for a
helix/sphere/torus, ...), NOT its topological intrinsic dim, because subspace
capture is about the linear subspace the manifold occupies.
"""
from dataclasses import dataclass, field

import numpy as np

# ── Paper-confirmed dimensions ───────────────────────────────────────────────
D = 128          # ambient dimension
C = 512          # dictionary width (number of ground-truth manifolds)
K_ACTIVE = 4     # active manifolds per sample

# shape name -> linear span (embedding) dimension e_i
SHAPE_EMBED_DIM = {
    "line": 1, "circle": 2, "helix": 3, "disk": 2,
    "sphere": 3, "torus": 3, "mobius": 3, "swiss_roll": 3,
}
SHAPE_TYPES = list(SHAPE_EMBED_DIM)


# ── Canonical shape samplers (return [n, e_i] points in natural coords) ───────

def _sample_canonical(shape, n, rng, params):
    if shape == "line":
        return rng.uniform(-1, 1, n)[:, None]
    if shape == "circle":
        th = rng.uniform(0, 2 * np.pi, n)
        return np.stack([np.cos(th), np.sin(th)], 1)
    if shape == "helix":
        turns, pitch = params["turns"], params["pitch"]
        th = rng.uniform(0, 2 * np.pi * turns, n)
        z = pitch * th / (2 * np.pi * turns)          # linear axis 0..pitch
        return np.stack([np.cos(th), np.sin(th), z], 1)
    if shape == "disk":
        r = np.sqrt(rng.uniform(0, 1, n))             # sqrt -> uniform on disk
        th = rng.uniform(0, 2 * np.pi, n)
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)
    if shape == "sphere":
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    if shape == "torus":
        R, r = params["R"], params["r"]
        th, ph = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        return np.stack([(R + r * np.cos(ph)) * np.cos(th),
                         (R + r * np.cos(ph)) * np.sin(th),
                         r * np.sin(ph)], 1)
    if shape == "mobius":
        u = rng.uniform(0, 2 * np.pi, n)
        v = rng.uniform(-0.5, 0.5, n)
        return np.stack([(1 + v * np.cos(u / 2)) * np.cos(u),
                         (1 + v * np.cos(u / 2)) * np.sin(u),
                         v * np.sin(u / 2)], 1)
    if shape == "swiss_roll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
        h = rng.uniform(0, 1, n) * params["height"]
        return np.stack([t * np.cos(t), h, t * np.sin(t)], 1)
    raise ValueError(f"unknown shape {shape!r}")


def _random_params(shape, rng):
    """Per-instance variant parameters (the paper uses 6 variants per type)."""
    if shape == "helix":
        return dict(turns=float(rng.uniform(1.5, 3.5)),
                    pitch=float(rng.uniform(0.5, 2.0)))
    if shape == "torus":
        return dict(R=float(rng.uniform(1.5, 3.0)),
                    r=float(rng.uniform(0.4, 1.0)))
    if shape == "swiss_roll":
        return dict(height=float(rng.uniform(1.0, 3.0)))
    return {}


def _grid_params(shape, v, n=6):
    """Deterministic 6-variant parameter grid per shape (``v`` in 0..n-1), matching the
    paper's *systematic* 'six parameter variants per type' rather than random draws. The
    grid spans the same ranges as ``_random_params``. Parameter-free shapes (line, circle,
    disk, sphere, mobius) have no variants — they differ only by their random embedding."""
    t = v / (n - 1) if n > 1 else 0.0            # 0..1 across the variants
    if shape == "helix":
        return dict(turns=1.5 + 2.0 * t, pitch=0.5 + 1.5 * t)
    if shape == "torus":
        return dict(R=1.5 + 1.5 * t, r=0.4 + 0.6 * t)
    if shape == "swiss_roll":
        return dict(height=1.0 + 2.0 * t)
    return {}


def _random_orthonormal(e, d, rng):
    """Random matrix [e, d] with orthonormal rows (e <= d)."""
    g = rng.normal(size=(d, e))
    q, _ = np.linalg.qr(g)          # q: [d, e], orthonormal columns
    return q[:, :e].T.astype(np.float32)   # [e, d], orthonormal rows


# ── Manifold instance ────────────────────────────────────────────────────────

@dataclass
class Manifold:
    name: str
    shape: str
    e: int                      # linear span dimension
    V: np.ndarray               # [e, D] orthonormal rows
    scale: float                # multiplier for unit-RMS contribution
    params: dict = field(default_factory=dict)

    def contribution(self, n, rng):
        """Embedded contribution [n, D] for n fresh points on this manifold."""
        p = _sample_canonical(self.shape, n, rng, self.params)
        return (p @ self.V).astype(np.float32) * self.scale


def build_dictionary(c=C, d=D, seed=0):
    """Build ``c`` manifolds cycling through the 8 shape types. Shape parameters follow
    the paper's deterministic 6-variant grid (variant = which pass through the shape
    cycle); position (origin), rotation (random ``V_i``) and unit-RMS size match the
    paper's ``x = Σ_i z_i V_i`` construction."""
    rng = np.random.default_rng(seed)
    manifolds = []
    n_shapes = len(SHAPE_TYPES)
    for i in range(c):
        shape = SHAPE_TYPES[i % n_shapes]
        e = SHAPE_EMBED_DIM[shape]
        params = _grid_params(shape, (i // n_shapes) % 6)
        V = _random_orthonormal(e, d, rng)
        # rescale to unit RMS: scale = 1 / rms(embedded canonical sample)
        emb = (_sample_canonical(shape, 4096, rng, params) @ V)
        scale = float(1.0 / np.sqrt(np.mean(emb ** 2)))
        manifolds.append(Manifold(f"{shape}_{i}", shape, e, V, scale, params))
    return manifolds


# ── Dataset + probe generation ───────────────────────────────────────────────

def generate_dataset(manifolds, n_samples, k_active=K_ACTIVE,
                     noise_sigma=0.05, seed=1):
    """Sparse-superposition activations [n_samples, D] (float32).

    noise_sigma is per-coordinate Gaussian std; each active manifold contributes
    unit-RMS signal, so total signal RMS ~ sqrt(k_active).
    """
    rng = np.random.default_rng(seed)
    c, d = len(manifolds), manifolds[0].V.shape[1]
    # exactly k_active distinct manifolds per sample, via per-row argsort
    active = np.argsort(rng.random((n_samples, c)), axis=1)[:, :k_active]  # [N, k]
    X = np.zeros((n_samples, d), dtype=np.float32)
    for i, m in enumerate(manifolds):
        rows = np.where((active == i).any(axis=1))[0]
        if len(rows):
            X[rows] += m.contribution(len(rows), rng)
    if noise_sigma > 0:
        X += rng.normal(0, noise_sigma, X.shape).astype(np.float32)
    return X


def probe_manifold(manifold, n_probe=512, noise_sigma=0.0, seed=7):
    """Clean activations for a single manifold [n_probe, D] plus its subspace.

    Analogous to the repo's per-manifold cached activations: dense sweep of one
    manifold with no other manifolds active. Returns (acts, V) where the rows of
    V span the manifold's true e-dim subspace.
    """
    rng = np.random.default_rng(seed)
    acts = manifold.contribution(n_probe, rng)
    if noise_sigma > 0:
        acts = acts + rng.normal(0, noise_sigma, acts.shape).astype(np.float32)
    return acts, manifold.V


def probe_in_context(target, manifolds, k_active=K_ACTIVE, n_probe=512,
                     noise_sigma=0.0, seed=7):
    """In-distribution probe: the target manifold varies across its full range
    while ``k_active-1`` other manifolds are held FIXED at a single random point
    each (a constant background offset, removed by the centering inside the
    greedy metrics). This matches how the SAE saw data during training (k_active
    atoms co-active) yet isolates the target manifold's variance — analogous to
    the paper's prompt probes that vary only the concept and fix the template.
    """
    rng = np.random.default_rng(seed)
    acts = target.contribution(n_probe, rng)                     # varying target
    others = [m for m in manifolds if m is not target]
    for j in rng.choice(len(others), size=min(k_active - 1, len(others)),
                        replace=False):
        acts = acts + others[j].contribution(1, rng)             # fixed offset
    if noise_sigma > 0:
        acts = acts + rng.normal(0, noise_sigma, acts.shape).astype(np.float32)
    return acts, target.V


def first_of_shape(manifolds, shape):
    for m in manifolds:
        if m.shape == shape:
            return m
    raise ValueError(f"no manifold of shape {shape!r} in dictionary")
