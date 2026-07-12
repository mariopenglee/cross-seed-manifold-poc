# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy>=1.24",
#     "torch>=2.2",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
# ]
# ///
"""
Interactive cross-seed manifold explorer.

Run it self-contained (uv fetches everything into an ephemeral env):

    uvx marimo edit --sandbox notebook.py     # edit / interact
    uvx marimo run  --sandbox notebook.py     # read-only app

or, inside the project venv:  marimo edit notebook.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)
    from sklearn.decomposition import PCA

    import synthetic
    import sae as sae_mod
    import train as train_mod
    import metrics

    return PCA, metrics, np, plt, sae_mod, synthetic, train_mod


@app.cell
def _(mo):
    mo.md("""
    # Cross-seed SAE manifold explorer

    Do several SAEs that differ **only in random seed**, pooled together, capture a
    concept **manifold** more fully than one SAE? This notebook lets you (1) look at
    the synthetic manifold geometries, (2) train a seed sweep + a width-matched
    control, and (3) *see* how few atoms each needs to reconstruct the manifold —
    against the PCA (optimal-linear) baseline.

    Full method + results: `docs/TECHNICAL.md`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Explore the manifold geometries
    """)
    return


@app.cell
def _(mo, synthetic):
    shape = mo.ui.dropdown(
        options=synthetic.SHAPE_TYPES, value="helix", label="Manifold shape"
    )
    shape
    return (shape,)


@app.cell
def _(PCA, np, plt, shape, synthetic):
    _rng = np.random.default_rng(0)
    _name = shape.value
    _e = synthetic.SHAPE_EMBED_DIM[_name]
    _params = synthetic._random_params(_name, _rng)
    _V = synthetic._random_orthonormal(_e, 128, _rng)
    _pts = synthetic._sample_canonical(_name, 1500, _rng, _params)     # [n, e]
    _ambient = _pts @ _V                                               # [n, 128]

    _fig = plt.figure(figsize=(11, 4.4))
    if _e == 3:
        _ax = _fig.add_subplot(1, 2, 1, projection="3d")
        _ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2], s=4, alpha=0.5)
    elif _e == 2:
        _ax = _fig.add_subplot(1, 2, 1)
        _ax.scatter(_pts[:, 0], _pts[:, 1], s=4, alpha=0.5)
        _ax.set_aspect("equal")
    else:
        _ax = _fig.add_subplot(1, 2, 1)
        _ax.scatter(_pts[:, 0], np.zeros(len(_pts)), s=4, alpha=0.5)
    _ax.set_title(f"{_name}: natural coords (span dim e={_e})")

    _P = PCA(n_components=3).fit_transform(_ambient - _ambient.mean(0))
    _ax2 = _fig.add_subplot(1, 2, 2, projection="3d")
    _ax2.scatter(_P[:, 0], _P[:, 1], _P[:, 2], s=4, alpha=0.5, color="tab:purple")
    _ax2.set_title("embedded in 128-d → PCA-3D")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Train the seed sweep + width-matched control

    Pick a regime and click **Train**. The union pools `N` seeds; the control is a
    *single* SAE with `N×` the width (same total atoms **and** compute). The first
    click trains `N+1` small SAEs on CPU (**~20–60s** — you'll see a spinner). Training
    is cached, so the sliders below won't retrain unless you change a training knob.
    """)
    return


@app.cell
def _(mo):
    c = mo.ui.slider(32, 384, value=128, step=32, label="ground-truth manifolds (c)")
    expansion = mo.ui.slider(2, 12, value=8, step=1, label="SAE expansion (×128)")
    ksae = mo.ui.dropdown(["4", "8", "16", "32"], value="16", label="SAE TopK k")
    nseeds = mo.ui.slider(2, 5, value=3, step=1, label="# seeds (union size)")
    steps = mo.ui.slider(200, 1500, value=800, step=100, label="training steps")
    run = mo.ui.run_button(label="▶ Train SAEs")
    mo.vstack([mo.hstack([c, expansion, ksae]), mo.hstack([nseeds, steps]), run])
    return c, expansion, ksae, nseeds, run, steps


@app.cell
def _(np, sae_mod, synthetic, train_mod):
    _CACHE = {}

    def _train_all(shape_name, c, expansion, k, nseeds, steps):
        d = 128
        manifolds = synthetic.build_dictionary(c=c, d=d, seed=0)
        X = synthetic.generate_dataset(
            manifolds, 20000, k_active=4, noise_sigma=0.05, seed=1
        )
        d_sae = expansion * d
        seeds = [
            train_mod.train_topk_sae(X, d_sae, k, seed=s, steps=steps, batch=4096)
            for s in range(nseeds)
        ]
        ctrl = train_mod.train_topk_sae(
            X, d_sae * nseeds, k, seed=0, steps=steps, batch=4096
        )
        decs = [sae_mod.get_decoder(s) for s in seeds]
        target = synthetic.first_of_shape(manifolds, shape_name)
        probe, _ = synthetic.probe_in_context(
            target, manifolds, k_active=4, n_probe=512, noise_sigma=0.0
        )
        return dict(
            seeds=seeds, ctrl=ctrl, decs=decs, dec_union=np.vstack(decs),
            dec_ctrl=sae_mod.get_decoder(ctrl), target=target, probe=probe,
            k=k, d_sae=d_sae,
        )

    def train_or_cached(key, force):
        if key in _CACHE:
            return _CACHE[key]
        if not force:
            return None
        _CACHE[key] = _train_all(*key)
        return _CACHE[key]

    return (train_or_cached,)


@app.cell
def _(c, expansion, ksae, mo, nseeds, run, shape, steps, train_or_cached):
    key = (shape.value, c.value, expansion.value, int(ksae.value),
           nseeds.value, steps.value)
    with mo.status.spinner(title="Training SAEs on CPU… (~20–60s)"):
        art = train_or_cached(key, run.value)
    mo.stop(
        art is None,
        mo.md("### 👆 Set parameters and click **▶ Train SAEs**"),
    )
    mo.md(
        f"Trained **{nseeds.value} seeds** (width {art['d_sae']}) + a width-"
        f"{art['d_sae'] * nseeds.value} control on shape **{shape.value}** "
        f"(k={art['k']})."
    )
    return (art,)


@app.cell
def _(mo):
    mo.md("""
    ## 3 · How few atoms capture the manifold? (restricted R²)
    """)
    return


@app.cell
def _(art, metrics, np, plt, sae_mod):
    _probe, _target = art["probe"], art["target"]
    _L = 48
    _cs = [sae_mod.encode_sae(s, _probe) for s in art["seeds"]]

    def _pad(cv, L):
        cv = np.asarray(cv, float)
        if len(cv) >= L:
            return cv[:L]
        return np.concatenate([cv, np.full(L - len(cv), cv[-1] if len(cv) else 0.0)])

    def _stat(dec, codes):
        return _pad(metrics.greedy_codes(_probe, dec, codes, max_k=_L), _L)

    _ss = np.stack([_stat(art["decs"][i], _cs[i]) for i in range(len(_cs))])
    _union = _stat(art["dec_union"], np.hstack(_cs))
    _ctrl = _stat(art["dec_ctrl"], sae_mod.encode_sae(art["ctrl"], _probe))

    _x = np.arange(1, _L + 1)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.fill_between(_x, _ss.min(0), _ss.max(0), color="tab:blue", alpha=0.15)
    _ax.plot(_x, _ss.mean(0), "o-", ms=3, color="tab:blue", label="single seed (mean)")
    _ax.plot(_x, _union, "s-", ms=3, lw=2, color="tab:red", label="cross-seed union")
    _ax.plot(_x, _ctrl, "^--", ms=3, color="tab:green", label="width-matched control")
    _ax.axvline(_target.e, color="k", lw=0.7, alpha=0.4)
    _ax.set(xlabel="# atoms selected", ylabel="variance explained",
            title=f"{_target.shape}: manifold reconstruction", xscale="log")
    _ax.set_ylim(-0.02, 1.02)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · See the reconstruction (vs. PCA)

    Reconstruct the manifold from the **top-B features' actual codes** (greedily chosen
    to explain the most variance; true manifold in gray). With only a few atoms the
    reconstruction is a partial *shadow* of the true shape. Slide **B** down and watch
    the cross-seed **union** hold more of the manifold than a single seed or the
    width-matched control at the same budget; **PCA (top-e)** is the optimal-linear
    ceiling.

    The **R²** under each panel is the *reconstruction similarity* — the fraction of the
    manifold's variance the shown reconstruction captures,
    R² = 1 − ‖true − reconstruction‖² ⁄ ‖true − mean‖²  (1.0 = perfect). It is the same
    **code-based restricted R²** as the §3 curves (their value at this atom budget), so
    the picture, its number, and the curves all agree.
    """)
    return


@app.cell
def _(mo):
    budget = mo.ui.slider(1, 24, value=3, step=1, label="# atoms for reconstruction (B)")
    budget
    return (budget,)


@app.cell
def _(PCA, art, budget, metrics, np, plt, sae_mod):
    _probe, _target = art["probe"], art["target"]
    _mean = _probe.mean(0)
    _Xc = _probe - _mean
    _tot = float((_Xc ** 2).sum())
    _B = budget.value

    _cs = [sae_mod.encode_sae(s, _probe) for s in art["seeds"]]
    _cc = sae_mod.encode_sae(art["ctrl"], _probe)

    _p = PCA(_target.e).fit(_Xc)                       # optimal-linear ceiling
    _pca = _mean + _p.inverse_transform(_p.transform(_Xc))
    _pca_r2 = 1.0 - float(((_probe - _pca) ** 2).sum()) / max(_tot, 1e-12)
    _single, _sr2, _ = metrics.greedy_codes_reconstruct(_probe, art["decs"][0], _cs[0], _B)
    _union, _ur2, _ = metrics.greedy_codes_reconstruct(_probe, art["dec_union"], np.hstack(_cs), _B)
    _control, _cr2, _ = metrics.greedy_codes_reconstruct(_probe, art["dec_ctrl"], _cc, _B)

    _panels = [
        (f"PCA (top {_target.e})", _pca, _pca_r2),
        (f"single seed ({_B} atoms)", _single, _sr2),
        (f"cross-seed union ({_B} atoms)", _union, _ur2),
        (f"width-matched control ({_B} atoms)", _control, _cr2),
    ]

    _view = PCA(3).fit(_Xc)
    _to3d = lambda Y: _view.transform(Y - _mean)
    _T = _to3d(_probe)
    _proj = [_to3d(_R) for _, _R, _ in _panels]
    # common cubic scale so low-variance PCA axes aren't visually exaggerated
    _L = float(np.abs(np.vstack([_T] + _proj)).max())

    _fig = plt.figure(figsize=(11, 8))
    for _i, ((_name, _R, _r), _Rp) in enumerate(zip(_panels, _proj)):
        _ax = _fig.add_subplot(2, 2, _i + 1, projection="3d")
        _ax.scatter(_T[:, 0], _T[:, 1], _T[:, 2], s=6, alpha=0.2, color="gray")
        _ax.scatter(_Rp[:, 0], _Rp[:, 1], _Rp[:, 2], s=6, alpha=0.7, color="tab:red")
        _ax.set_xlim(-_L, _L); _ax.set_ylim(-_L, _L); _ax.set_zlim(-_L, _L)
        _ax.set_box_aspect((1, 1, 1))
        _ax.set_title(f"{_name}\nR² = {_r:.2f}", fontsize=9)
        _ax.set_xticklabels([]); _ax.set_yticklabels([]); _ax.set_zticklabels([])
    _fig.suptitle(f"{_target.shape}: reconstruction from few atoms (true manifold in gray)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · Per-seed contributions — toggle each seed's SAE

    **Left:** each active seed reconstructs the manifold on its own (its own top-B
    atoms, one color per seed). **Right:** the pooled reconstruction from just the seeds
    left on. Turn seeds off and on to watch how independent seeds — each capturing a
    partial, differently-oriented shadow — combine into a fuller picture than any one
    alone. (Uses the same **B** slider as Section 4.)
    """)
    return


@app.cell
def _(mo, nseeds):
    seed_toggles = mo.ui.array(
        [mo.ui.checkbox(value=True, label=f"seed {i}") for i in range(nseeds.value)]
    )
    mo.vstack([mo.md("**Toggle each seed's SAE on/off:**"), seed_toggles])
    return (seed_toggles,)


@app.cell
def _(PCA, art, budget, metrics, np, plt, sae_mod, seed_toggles):
    _probe, _target, _decs = art["probe"], art["target"], art["decs"]
    _mean = _probe.mean(0)
    _Xc = _probe - _mean
    _B = budget.value
    _cs = [sae_mod.encode_sae(s, _probe) for s in art["seeds"]]

    _tog = seed_toggles.value
    _active = [i for i in range(len(_decs)) if i < len(_tog) and _tog[i]]

    _view = PCA(3).fit(_Xc)
    _to3d = lambda Y: _view.transform(Y - _mean)
    _T = _to3d(_probe)
    _colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    _seed_proj = {}
    for _i in _active:
        _rec, _r, _ = metrics.greedy_codes_reconstruct(_probe, _decs[_i], _cs[_i], _B)
        _seed_proj[_i] = (_to3d(_rec), _r)
    _pr2, _Uproj = 0.0, None
    if _active:
        _pool_dec = np.vstack([_decs[_i] for _i in _active])
        _pool_codes = np.hstack([_cs[_i] for _i in _active])
        _prec, _pr2, _ = metrics.greedy_codes_reconstruct(_probe, _pool_dec, _pool_codes, _B)
        _Uproj = _to3d(_prec)
    _stack = [_T] + [p for p, _ in _seed_proj.values()] + ([_Uproj] if _Uproj is not None else [])
    _L = float(np.abs(np.vstack(_stack)).max())

    _fig = plt.figure(figsize=(12, 5.5))
    _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax1.scatter(_T[:, 0], _T[:, 1], _T[:, 2], s=5, alpha=0.12, color="gray")
    for _i in _active:
        _Ri, _r = _seed_proj[_i]
        _ax1.scatter(_Ri[:, 0], _Ri[:, 1], _Ri[:, 2], s=5, alpha=0.5,
                     color=_colors[_i % 10], label=f"seed {_i}  R²={_r:.2f}")
    _ax1.set_xlim(-_L, _L); _ax1.set_ylim(-_L, _L); _ax1.set_zlim(-_L, _L)
    _ax1.set_box_aspect((1, 1, 1))
    _ax1.set_title(f"each seed alone ({_B} atoms)", fontsize=10)
    if _active:
        _ax1.legend(fontsize=7, loc="upper left")
    _ax1.set_xticklabels([]); _ax1.set_yticklabels([]); _ax1.set_zticklabels([])

    _ax2 = _fig.add_subplot(1, 2, 2, projection="3d")
    _ax2.scatter(_T[:, 0], _T[:, 1], _T[:, 2], s=5, alpha=0.12, color="gray")
    if _Uproj is not None:
        _ax2.scatter(_Uproj[:, 0], _Uproj[:, 1], _Uproj[:, 2], s=6, alpha=0.7, color="tab:red")
    _ax2.set_xlim(-_L, _L); _ax2.set_ylim(-_L, _L); _ax2.set_zlim(-_L, _L)
    _ax2.set_box_aspect((1, 1, 1))
    _ax2.set_title(f"pooled union of {len(_active)} seed(s)  R²={_pr2:.2f}", fontsize=10)
    _ax2.set_xticklabels([]); _ax2.set_yticklabels([]); _ax2.set_zticklabels([])

    _fig.suptitle(f"{_target.shape}: per-seed vs. pooled reconstruction (true = gray)")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
