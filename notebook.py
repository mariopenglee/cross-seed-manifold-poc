# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy>=1.24",
#     "torch>=2.2",
#     "scikit-learn>=1.3",
#     "scipy>=1.10",
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
    concept **manifold** more fully than one SAE? This notebook lets you (1) build a
    synthetic activation dataset and train a seed sweep + a width-matched control,
    (2) pick a target concept and see its geometry, and (3) *see* how few atoms each
    needs to reconstruct it — against the PCA (optimal-linear) baseline.

    Section 6 then asks *why* pooling helps — complementary **tiling** or plain
    **variance reduction**. See `README.md` for method + how to run.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Set up the activations and train

    Set the **dataset** knobs (what geometry the SAEs see) and the **SAE** knobs, then
    click **Train**. The union pools `N` seeds; the control is a *single* SAE with `N×`
    the width (same total atoms **and** compute). The first click trains `N+1` small SAEs
    on CPU (**~20–60s**). The SAEs train on the full mixture of *all* manifolds, so they
    are **shared across shapes** — only the dataset/SAE knobs below trigger a retrain;
    choosing which shape/instance to score (Section 2, *after* training) is instant.

    **Defaults reproduce Bhalla et al.'s synthetic benchmark** (c=48 manifolds in
    d=128, SAE width 512 = ×4, k=4, 4 co-active per sample). The legend below says which
    lever does what — and which one quietly misled us.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | lever | what it controls | guidance |
    |---|---|---|
    | **k (TopK)** | **the regime axis** | The paper's whole story lives here: low k **shatters**, k≈manifold dim **captures** (compact atom group spans it — their k=4 sweet-spot), high k **dilutes** (many redundant atoms per point). Sweep k to move between regimes. |
    | **c (# manifolds)** | crowding *(handle with care)* | Each manifold spans 2–3 of the 128 dims; pack in too many (c ≫ ~50) and the spans collide, capping capture and faking *"dilution everywhere"* — the knob that misled our first pass. Crank it to *watch* capture strangle, but know that's crowding, not a k effect. |
    | **width (×128)** | atoms to go around | Secondary. Paper uses ×4 (width 512). |
    | **# seeds** | union size | **Our lever, not the paper's** — the cross-seed question it doesn't ask. Union pools this many independent-seed SAEs vs. one `N×`-wide control (matched atoms + compute). |

    Nothing is locked — every knob is live. These are just the ones the paper pins vs.
    the ones that mislead. At the k=4 default a single SAE already *captures*, so the
    union barely helps; push **k toward dilution (≥24)** to see where pooling seeds pays off.
    """)
    return


@app.cell
def _(mo):
    # --- Dataset knobs: what activations the SAE is trained on ---
    d_ambient = mo.ui.slider(64, 256, value=128, step=32, label="ambient dim d")
    c = mo.ui.slider(16, 256, value=48, step=16, label="# manifolds c")
    k_active = mo.ui.slider(1, 8, value=4, step=1, label="co-active / sample")
    noise = mo.ui.slider(0.0, 0.20, value=0.05, step=0.01, label="noise σ (train data)")
    # --- SAE knobs: the model we train to recover them ---
    expansion = mo.ui.slider(2, 12, value=4, step=1, label="SAE width (×d)")
    ksae = mo.ui.dropdown(
        ["1", "2", "4", "8", "16", "24", "32"], value="4", label="TopK k (regime axis)"
    )
    nseeds = mo.ui.slider(2, 5, value=3, step=1, label="# seeds (union)")
    steps = mo.ui.slider(200, 1500, value=800, step=100, label="training steps")
    run = mo.ui.run_button(label="Train SAEs")
    mo.vstack([
        mo.md("**Dataset — the activations** &nbsp; *(these define the manifold "
              "geometry the SAE sees; changing any retrains)*"),
        mo.hstack([d_ambient, c, k_active, noise]),
        mo.md("**SAE — the model** &nbsp; *(how we try to recover the manifolds)*"),
        mo.hstack([expansion, ksae, nseeds, steps]),
        run,
    ])
    return c, d_ambient, expansion, k_active, ksae, noise, nseeds, run, steps


@app.cell
def _(c, d_ambient, k_active, mo, synthetic):
    _types = synthetic.SHAPE_TYPES
    _spans = [synthetic.SHAPE_EMBED_DIM[_types[i % len(_types)]] for i in range(c.value)]
    _total = sum(_spans)
    _rate = _total / d_ambient.value
    _load = k_active.value * _total / c.value
    _tag = ("**over-complete** — true superposition"
            if _rate > 1 else "under-full — room to spare")
    mo.md(
        f"**Superposition readout.** {c.value} manifolds span **{_total}** dims total "
        f"inside a **{d_ambient.value}-dim** space -> **span / d = {_rate:.2f}** ({_tag}). "
        f"Each sample superposes **{k_active.value}** of them "
        f"(~{_load:.0f} signal dims of {d_ambient.value}). "
        f"The paper's regime is span/d > 1; the c=48 / d=128 default sits at 0.94."
    )
    return


@app.cell
def _(np, sae_mod, synthetic, train_mod):
    _CACHE = {}

    def _train_all(c, d, expansion, k, nseeds, steps, k_active, noise):
        # SAEs train on the full mixture (all manifolds), independent of which target
        # shape/instance you later probe — the cache key omits shape + instance.
        manifolds = synthetic.build_dictionary(c=c, d=d, seed=0)
        X = synthetic.generate_dataset(
            manifolds, 20000, k_active=k_active, noise_sigma=noise, seed=1
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
        return dict(
            seeds=seeds, ctrl=ctrl, decs=decs, dec_union=np.vstack(decs),
            dec_ctrl=sae_mod.get_decoder(ctrl), manifolds=manifolds,
            k=k, d_sae=d_sae, k_active=k_active,
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
def _(mo):
    mo.md("""
    ## 2 · Choose what to score (after training)

    The SAEs above see **all** manifolds, so these are pure *probing* choices — changing
    them re-measures instantly, no retrain. Pick a **shape**, and either a specific
    **instance** or the mean over all instances of that shape (robust to a single unlucky
    embedding — e.g. the mobius fluke). The preview shows the target's geometry;
    Sections 3+ score it.
    """)
    return


@app.cell
def _(mo, synthetic):
    shape = mo.ui.dropdown(
        options=synthetic.SHAPE_TYPES, value="helix", label="target shape"
    )
    shape
    return (shape,)


@app.cell
def _(c, mo, shape, synthetic):
    _types = synthetic.SHAPE_TYPES
    _count = sum(1 for i in range(c.value) if _types[i % len(_types)] == shape.value)
    instance_sel = mo.ui.dropdown(
        ["average over all"] + [str(i) for i in range(max(_count, 1))],
        value="average over all",
        label=f"target instance  ({_count} {shape.value}(s) in the dictionary)",
    )
    mo.vstack([mo.md("**Which instance** — a specific one, or the mean over all "
                     "instances of this shape:"), instance_sel])
    return (instance_sel,)


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
    _ax2.set_title("embedded in 128-d -> PCA-3D")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(c, d_ambient, expansion, instance_sel, k_active, ksae, mo, noise, nseeds,
      run, shape, steps, synthetic, train_or_cached):
    key = (c.value, d_ambient.value, expansion.value, int(ksae.value), nseeds.value,
           steps.value, k_active.value, round(noise.value, 3))
    with mo.status.spinner(title="Training SAEs on CPU… (~20–60s)"):
        _trained = train_or_cached(key, run.value)
    mo.stop(
        _trained is None,
        mo.md("### Set parameters and click **Train SAEs**"),
    )
    # Switching shape/instance does NOT retrain — the SAEs are shared; we just re-probe.
    _mans = _trained["manifolds"]
    _idx = [i for i, m in enumerate(_mans) if m.shape == shape.value] or [0]
    if instance_sel.value == "average over all":
        _targets = [_mans[i] for i in _idx]
    else:
        _j = int(instance_sel.value)
        _targets = [_mans[_idx[_j if _j < len(_idx) else 0]]]
    _ka = _trained["k_active"]
    _probes = [synthetic.probe_in_context(t, _mans, k_active=_ka, n_probe=512,
                                          noise_sigma=0.0)[0] for t in _targets]
    art = {**_trained, "targets": _targets, "probes": _probes,
           "target": _targets[0], "probe": _probes[0]}
    _e = art["target"].e
    _regime = ("sparse / shattering-prone" if art["k"] < _e else
               "capture regime (paper sweet-spot, k≈dim)" if art["k"] <= 8 * _e else
               "dilution regime")
    _scope = (f"mean over {len(_targets)} {shape.value} instances"
              if len(_targets) > 1 else f"{shape.value} instance {instance_sel.value}")
    mo.md(
        f"Trained **{nseeds.value} seeds** (width {art['d_sae']}) + a width-"
        f"{art['d_sae'] * nseeds.value} control; scoring **{_scope}** "
        f"(span dim e={_e}, k={art['k']}) -> **{_regime}**."
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
    _probes, _target = art["probes"], art["target"]
    _L = 48

    def _pad(cv, L):
        cv = np.asarray(cv, float)
        if len(cv) >= L:
            return cv[:L]
        return np.concatenate([cv, np.full(L - len(cv), cv[-1] if len(cv) else 0.0)])

    # single / union / control capture curves, averaged over the selected instance(s)
    _sing, _uni, _ctl = [], [], []
    for _p in _probes:
        _cs = [sae_mod.encode_sae(s, _p) for s in art["seeds"]]
        _sing.append(np.mean(
            [_pad(metrics.greedy_codes(_p, art["decs"][i], _cs[i], max_k=_L), _L)
             for i in range(len(_cs))], axis=0))
        _uni.append(_pad(metrics.greedy_codes(_p, art["dec_union"], np.hstack(_cs), max_k=_L), _L))
        _ctl.append(_pad(metrics.greedy_codes(
            _p, art["dec_ctrl"], sae_mod.encode_sae(art["ctrl"], _p), max_k=_L), _L))
    _single, _union, _ctrl = np.mean(_sing, 0), np.mean(_uni, 0), np.mean(_ctl, 0)

    _x = np.arange(1, _L + 1)
    _n = len(_probes)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    if _n > 1:
        _ax.fill_between(_x, np.min(_uni, 0), np.max(_uni, 0), color="tab:red", alpha=0.12)
    _ax.plot(_x, _single, "o-", ms=3, color="tab:blue", label="single seed")
    _ax.plot(_x, _union, "s-", ms=3, lw=2, color="tab:red", label="cross-seed union")
    _ax.plot(_x, _ctrl, "^--", ms=3, color="tab:green", label="width-matched control")
    _ax.axvline(_target.e, color="k", lw=0.7, alpha=0.4)
    _scope = f"mean of {_n} instances (band = union spread)" if _n > 1 else "one instance"
    _ax.set(xlabel="# atoms selected", ylabel="variance explained",
            title=f"{_target.shape}: manifold reconstruction ({_scope})", xscale="log")
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Tiling or variance reduction? — why pooling helps

    Pooling seeds could help for two very different reasons:

    - **Variance reduction** — every seed learns *the same* atoms, just noisy copies of
      each other. The union simply gets to pick the cleaner version. No new territory is
      covered; you'd get the same benefit by *averaging* aligned seeds.
    - **Tiling** — seeds learn *complementary* atoms, each covering a different slice of
      the manifold. The union stitches the slices together. Averaging can't reproduce
      this — you can't average two atoms into a third one that sits somewhere else.

    Four measures decide which it is (computed on the trained seeds from Section 2, at
    the current **B**). The **aligned-average** test is the cleanest: align every seed's
    features to seed 0 (Hungarian match on decoder cosine), average the matched atoms,
    and see how much of the union's gain that recovers. **~100% -> variance reduction;
    little -> tiling.**
    """)
    return


@app.cell
def _(art, budget, metrics, mo, np, sae_mod):
    from scipy.optimize import linear_sum_assignment

    def _align_and_average(decs, codes, ref=0):
        # Align each seed's atoms to a reference (max decoder cosine), then average
        # matched atoms + their codes. Denoises copies; cannot invent a new-region atom.
        n, (m, _) = len(decs), decs[0].shape
        unit = lambda D: D / np.linalg.norm(D, axis=1, keepdims=True).clip(1e-8)
        dref = unit(decs[ref])
        dec_acc = decs[ref].astype(np.float64).copy()
        code_acc = codes[ref].astype(np.float64).copy()
        for j in range(n):
            if j == ref:
                continue
            row, col = linear_sum_assignment(-(dref @ unit(decs[j]).T))
            perm = np.empty(m, int)
            perm[row] = col
            dec_acc += decs[j][perm]
            code_acc += codes[j][:, perm]
        return (dec_acc / n).astype(np.float32), (code_acc / n).astype(np.float32)

    def _engaged(codes, top):
        sd = codes.std(0)
        idx = np.argsort(-sd)[:top]
        return idx[sd[idx] > 1e-3]

    _probe, _decs, _seeds = art["probe"], art["decs"], art["seeds"]
    _n, _m, _B = len(_seeds), _decs[0].shape[0], budget.value
    _codes = [sae_mod.encode_sae(_s, _probe) for _s in _seeds]
    _du = np.vstack(_decs)

    # single seeds vs union: greedy top-B reconstructions + per-point residuals
    _singles = [metrics.greedy_codes_reconstruct(_probe, _decs[_i], _codes[_i], _B)
                for _i in range(_n)]
    _best = max(range(_n), key=lambda _i: _singles[_i][1])
    _s_recon, _s_r2, _ = _singles[_best]
    _u_recon, _u_r2, _u_sel = metrics.greedy_codes_reconstruct(
        _probe, _du, np.hstack(_codes), _B)

    _s_res = ((_probe - _s_recon) ** 2).sum(1)
    _u_res = ((_probe - _u_recon) ** 2).sum(1)
    _imp = _s_res - _u_res
    _gap = _u_r2 - _s_r2  # how much the union beats the best single seed

    # (1) gap-fill: share of improvement landing on the worst-25% single-seed points
    _order = np.argsort(-_s_res)
    _qn = max(len(_order) // 4, 1)
    _frac_worst = float(_imp[_order[:_qn]].sum() / max(_imp.sum(), 1e-9))
    _corr = float(np.corrcoef(_s_res, _imp)[0, 1])

    # (2) composition: how many distinct seeds feed the union's greedy top-B
    _sel_seed = np.array([_idx // _m for _idx in _u_sel])
    _counts = np.bincount(_sel_seed, minlength=_n)
    _n_distinct = int((_counts > 0).sum())

    # (3) atom overlap: cross-seed nearest-neighbour |cos| vs within-seed baseline
    _dirs = []
    for _i in range(_n):
        _A = _decs[_i][_engaged(_codes[_i], _B)]
        _dirs.append(_A / np.linalg.norm(_A, axis=1, keepdims=True).clip(1e-8))
    _cross, _within = [], []
    for _i in range(_n):
        for _j in range(_n):
            _C = np.abs(_dirs[_i] @ _dirs[_j].T)
            if _i == _j:
                np.fill_diagonal(_C, 0.0)
                _within.append(_C.max(1))
            else:
                _cross.append(_C.max(1))
    _cross_ov = float(np.concatenate(_cross).mean()) if _cross else float("nan")
    _within_ov = float(np.concatenate(_within).mean())

    # (4) aligned-average: averaging denoises but cannot add coverage
    _dec_avg, _code_avg = _align_and_average(_decs, _codes)
    _, _a_r2, _ = metrics.greedy_codes_reconstruct(_probe, _dec_avg, _code_avg, _B)
    _recovered = float((_a_r2 - _s_r2) / _gap) if _gap > 1e-6 else float("nan")

    if _gap <= 1e-3:
        _out = mo.md(
            f"""
            **Union ≈ best single seed here** (union R²={_u_r2:.2f}, best single
            R²={_s_r2:.2f}, gap {_gap:+.2f}). There's essentially no pooling benefit to
            explain at this config — the SAE is roomy enough that one seed already
            captures the manifold. Lower the capacity (raise **c**, or drop expansion /
            **k**) so the union beats the single seed, then this section becomes
            informative.
            """
        )
    else:
        # simple verdict: tiling signatures vs variance-reduction signatures
        _tiling_votes = (
            (_frac_worst > 0.45)          # improvement concentrated on worst points
            + (_n_distinct >= 2)          # union draws atoms from ≥2 seeds
            + (_cross_ov < _within_ov + 0.02)  # cross-seed no more aligned than within
            + (_recovered < 0.6)          # averaging recovers little of the gain
        )
        _verdict = ("**TILING** — the extra atoms cover distinct territory"
                    if _tiling_votes >= 3 else
                    "**VARIANCE REDUCTION** — seeds mostly relearn the same atoms"
                    if _tiling_votes <= 1 else "**MIXED**")
        _out = mo.md(
            f"""
            Best single seed R²={_s_r2:.2f} · union R²={_u_r2:.2f} · **U−S={_gap:+.2f}**
            &nbsp;->&nbsp; verdict: {_verdict} &nbsp;({_tiling_votes}/4 tiling signatures)

            | measure | value | tiling looks like | variance-reduction looks like |
            |---|---|---|---|
            | **1. gap-fill** — share of improvement on the worst-25% single-seed points | **{_frac_worst*100:.0f}%** (null 25%); corr(res, improvement) **{_corr:+.2f}** | ≫ 25%, corr -> +1 | ≈ 25%, corr ≈ 0 |
            | **2. composition** — distinct seeds feeding the union's top-{_B} | **{_n_distinct}/{_n}** &nbsp; counts {_counts.tolist()} | several seeds | one seed dominates |
            | **3. atom overlap** — nearest-neighbour \\|cos\\| | cross **{_cross_ov:.2f}** vs within **{_within_ov:.2f}** | cross ≲ within | cross ≫ within (copies) |
            | **4. aligned-average** — gain recovered by averaging | R²={_a_r2:.2f} -> **{_recovered*100:.0f}%** | little (can't add coverage) | ~100% (denoising) |

            The map below colors each manifold point by how much the union improves it — a
            **tiling** union concentrates its help on the regions the best single seed
            reconstructs *worst* (the bright points); pure variance reduction would spread
            evenly.
            """
        )
    _out
    return


@app.cell
def _(PCA, art, budget, metrics, np, plt, sae_mod):
    _probe, _decs, _seeds = art["probe"], art["decs"], art["seeds"]
    _target, _B = art["target"], budget.value
    _n = len(_seeds)
    _codes = [sae_mod.encode_sae(_s, _probe) for _s in _seeds]
    _du = np.vstack(_decs)

    _singles = [metrics.greedy_codes_reconstruct(_probe, _decs[_i], _codes[_i], _B)
                for _i in range(_n)]
    _best = max(range(_n), key=lambda _i: _singles[_i][1])
    _s_recon = _singles[_best][0]
    _u_recon, _u_r2, _ = metrics.greedy_codes_reconstruct(
        _probe, _du, np.hstack(_codes), _B)
    _imp = ((_probe - _s_recon) ** 2).sum(1) - ((_probe - _u_recon) ** 2).sum(1)

    _mean = _probe.mean(0)
    _view = PCA(3).fit(_probe - _mean)
    _T = _view.transform(_probe - _mean)
    _L = float(np.abs(_T).max())
    _clip = np.clip(_imp, 0, None)
    _sz = 8 + 60 * (_clip / (_clip.max() + 1e-9))

    _fig = plt.figure(figsize=(7, 6))
    _ax = _fig.add_subplot(111, projection="3d")
    _sc = _ax.scatter(_T[:, 0], _T[:, 1], _T[:, 2], c=_clip, s=_sz,
                      cmap="magma", alpha=0.85)
    _ax.set_xlim(-_L, _L); _ax.set_ylim(-_L, _L); _ax.set_zlim(-_L, _L)
    _ax.set_box_aspect((1, 1, 1))
    _ax.set_xticklabels([]); _ax.set_yticklabels([]); _ax.set_zticklabels([])
    _fig.colorbar(_sc, ax=_ax, shrink=0.6, label="union improvement per point")
    _ax.set_title(f"{_target.shape}: where the union helps (bright = "
                  f"best single seed's worst regions)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · The headline — where does pooling help across the sparsity regime?

    Sections 2–6 explored **one** value of `k`. This sweeps `k` (the paper's regime
    axis) at the current **c / width / # seeds** and plots the **raw** capture scores
    (over all eight shapes), so nothing is hidden inside a difference:

    - **single** — one seed's SAE.
    - **union** — the pooled `N` seeds.
    - **control** — one SAE of the same total width **and** compute as the union.
    - **ρ** — single-seed capture compactness (1 = compact, → 0 = diluted).

    Read it two honest ways: **union − single** is the pooling benefit against a
    *consistent* baseline; **union vs control** is the resource-matched question. Watch
    the **control dip below single at high `k`** — that's the paper's dilution thesis in
    one line: extra atoms at high sparsity go into *fragmenting* the manifold, not
    spanning it, so a wider SAE captures *worse*. Pooling's edge over a single seed grows
    as `k` drives it into dilution (ρ falls). ~20 small SAEs → **~2–4 min** first run
    (cached afterwards).
    """)
    return


@app.cell
def _(mo):
    run_ksweep = mo.ui.run_button(label="Run the sparsity sweep")
    run_ksweep
    return (run_ksweep,)


@app.cell
def _(metrics, np, sae_mod, synthetic, train_mod):
    _KSWEEP_CACHE = {}
    _KLIST = (4, 8, 12, 16, 24, 32)

    def _rho(curve, e):
        cv = np.asarray(curve, float)
        if len(cv) == 0 or cv[-1] <= 1e-9:
            return np.nan
        return float(cv[min(e, len(cv)) - 1] / cv[-1])

    def _sweep(c, d, expansion, nseeds, steps, k_active, noise):
        manifolds = synthetic.build_dictionary(c=c, d=d, seed=0)
        X = synthetic.generate_dataset(manifolds, 20000, k_active=k_active,
                                       noise_sigma=noise, seed=1)
        d_sae = expansion * d
        shapes = sorted({m.shape for m in manifolds})
        rows = []
        for k in _KLIST:
            seeds = [train_mod.train_topk_sae(X, d_sae, k, seed=s, steps=steps,
                                              batch=4096) for s in range(nseeds)]
            ctrl = train_mod.train_topk_sae(X, d_sae * nseeds, k, seed=0,
                                            steps=steps, batch=4096)
            decs = [sae_mod.get_decoder(s) for s in seeds]
            du, dc = np.vstack(decs), sae_mod.get_decoder(ctrl)
            singles, unions, controls, rhos = [], [], [], []
            for shape in shapes:
                tgt = synthetic.first_of_shape(manifolds, shape)
                probe, _ = synthetic.probe_in_context(tgt, manifolds, k_active=k_active,
                                                      n_probe=512, noise_sigma=0.0)
                cs = [sae_mod.encode_sae(s, probe) for s in seeds]
                single = np.mean([metrics.auc(
                    metrics.greedy_codes(probe, decs[i], cs[i], 8), 8)
                    for i in range(nseeds)])
                union = metrics.auc(metrics.greedy_codes(probe, du, np.hstack(cs), 8), 8)
                control = metrics.auc(
                    metrics.greedy_codes(probe, dc, sae_mod.encode_sae(ctrl, probe), 8), 8)
                singles.append(single); unions.append(union); controls.append(control)
                rhos.append(_rho(metrics.greedy_codes(probe, decs[0], cs[0], 48), tgt.e))
            rows.append((k, float(np.mean(singles)), float(np.mean(unions)),
                         float(np.mean(controls)), float(np.nanmean(rhos))))
        return rows

    def ksweep_or_cached(key, force):
        if key in _KSWEEP_CACHE:
            return _KSWEEP_CACHE[key]
        if not force:
            return None
        _KSWEEP_CACHE[key] = _sweep(*key)
        return _KSWEEP_CACHE[key]

    return (ksweep_or_cached,)


@app.cell
def _(c, d_ambient, expansion, k_active, ksweep_or_cached, mo, noise, nseeds, plt,
      run_ksweep, steps):
    _key = (c.value, d_ambient.value, expansion.value, nseeds.value, steps.value,
            k_active.value, round(noise.value, 3))
    with mo.status.spinner(title="Sweeping k — training ~20 SAEs on CPU…"):
        _rows = ksweep_or_cached(_key, run_ksweep.value)
    mo.stop(
        _rows is None,
        mo.md("### Click **Run the sparsity sweep** (uses the c / width / # seeds above)"),
    )

    _ks = [r[0] for r in _rows]
    _single = [r[1] for r in _rows]
    _union = [r[2] for r in _rows]
    _control = [r[3] for r in _rows]
    _rho = [r[4] for r in _rows]

    _fig, _ax = plt.subplots(figsize=(8.5, 5))
    _l1, = _ax.plot(_ks, _union, "o-", color="tab:red", lw=2, label="union (pooled seeds)")
    _l2, = _ax.plot(_ks, _single, "^-", color="tab:green", label="single seed")
    _l3, = _ax.plot(_ks, _control, "v--", color="tab:orange", label="width-matched control")
    _ax.set_xlabel("k  (TopK sparsity — the regime axis)")
    _ax.set_ylabel("restricted-R² capture (AUC over 8 atoms)")
    _ax2 = _ax.twinx()
    _l4, = _ax2.plot(_ks, _rho, "s:", color="tab:blue", alpha=0.5, label="rho (capture)")
    _ax2.set_ylabel("rho  (1 = compact capture, -> 0 = dilution)", color="tab:blue")
    _ax2.tick_params(axis="y", labelcolor="tab:blue")
    _ax2.set_ylim(0, 1.05)
    _ax.set_title(f"c={c.value}, width {expansion.value*128}, {nseeds.value} seeds: "
                  f"union beats single; control dips below single in dilution")
    _ax.legend([_l1, _l2, _l3, _l4],
               ["union (pooled seeds)", "single seed", "width-matched control",
                "rho (capture)"], fontsize=8, loc="center left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Structured entanglement — correlated co-activation

    So far concepts co-fire **independently** (4 random manifolds per sample). Real
    concepts cluster — they co-occur in groups (sunset / orange / evening). This
    partitions the manifolds into groups of 8 and, with probability **corr**, draws a
    sample's active set from *within one group*, so those concepts are always seen
    together. `corr=0` is the independent benchmark; `corr=1` means a concept *only ever*
    appears with its group-mates.

    Sweeping **corr** (at the current **k**) asks whether structured entanglement breaks
    per-manifold capture. It does — but only under *near-total* correlation: single-seed
    capture holds flat until ~0.85, then crashes, and the wide **control drops below a
    single seed** (the capacity pathology) right at the break. Clearest at a capture `k`
    (small k). ~24 SAEs → **~3–4 min** first run (cached afterwards).
    """)
    return


@app.cell
def _(mo):
    run_corr = mo.ui.run_button(label="Run the correlation sweep")
    run_corr
    return (run_corr,)


@app.cell
def _(metrics, np, sae_mod, synthetic, train_mod):
    _CORR_CACHE = {}
    _CORRS = (0.0, 0.5, 0.7, 0.85, 0.95, 1.0)

    def _gen_correlated(manifolds, n, k_active, groups, corr, seed, noise):
        rng = np.random.default_rng(seed)
        c = len(manifolds)
        dim = manifolds[0].V.shape[1]
        full = [g for g in groups if len(g) >= k_active]
        active = np.empty((n, k_active), dtype=int)
        coin = rng.random(n)
        for s in range(n):
            if corr > 0 and coin[s] < corr and full:
                g = full[rng.integers(len(full))]
                active[s] = rng.choice(g, size=k_active, replace=False)
            else:
                active[s] = rng.choice(c, size=k_active, replace=False)
        X = np.zeros((n, dim), dtype=np.float32)
        for i, m in enumerate(manifolds):
            rows = np.where((active == i).any(axis=1))[0]
            if len(rows):
                X[rows] += m.contribution(len(rows), rng)
        X += rng.normal(0, noise, X.shape).astype(np.float32)
        return X

    def _corr_sweep(c, d, expansion, nseeds, steps, k, k_active, noise):
        manifolds = synthetic.build_dictionary(c=c, d=d, seed=0)
        groups = [list(range(i, min(i + 8, c))) for i in range(0, c, 8)]
        shapes = sorted({m.shape for m in manifolds})
        d_sae = expansion * d
        rows = []
        for corr in _CORRS:
            X = _gen_correlated(manifolds, 20000, k_active, groups, corr, seed=1,
                                noise=noise)
            seeds = [train_mod.train_topk_sae(X, d_sae, k, seed=s, steps=steps,
                                              batch=4096) for s in range(nseeds)]
            ctrl = train_mod.train_topk_sae(X, d_sae * nseeds, k, seed=0,
                                            steps=steps, batch=4096)
            decs = [sae_mod.get_decoder(s) for s in seeds]
            du, dc = np.vstack(decs), sae_mod.get_decoder(ctrl)
            sng, uni, ctl = [], [], []
            for shape in shapes:
                tgt = synthetic.first_of_shape(manifolds, shape)
                probe, _ = synthetic.probe_in_context(tgt, manifolds, k_active=k_active,
                                                      n_probe=512, noise_sigma=0.0)
                cs = [sae_mod.encode_sae(s, probe) for s in seeds]
                sng.append(np.mean([metrics.auc(
                    metrics.greedy_codes(probe, decs[i], cs[i], 8), 8)
                    for i in range(nseeds)]))
                uni.append(metrics.auc(metrics.greedy_codes(probe, du, np.hstack(cs), 8), 8))
                ctl.append(metrics.auc(
                    metrics.greedy_codes(probe, dc, sae_mod.encode_sae(ctrl, probe), 8), 8))
            rows.append((corr, float(np.mean(sng)), float(np.mean(uni)),
                         float(np.mean(ctl))))
        return rows

    def corr_or_cached(key, force):
        if key in _CORR_CACHE:
            return _CORR_CACHE[key]
        if not force:
            return None
        _CORR_CACHE[key] = _corr_sweep(*key)
        return _CORR_CACHE[key]

    return (corr_or_cached,)


@app.cell
def _(c, corr_or_cached, d_ambient, expansion, k_active, ksae, mo, noise, nseeds, plt,
      run_corr, steps):
    _key = (c.value, d_ambient.value, expansion.value, nseeds.value, steps.value,
            int(ksae.value), k_active.value, round(noise.value, 3))
    with mo.status.spinner(title="Correlation sweep — training ~24 SAEs on CPU…"):
        _rows = corr_or_cached(_key, run_corr.value)
    mo.stop(
        _rows is None,
        mo.md("### Click **Run the correlation sweep** (uses c / width / # seeds / k above)"),
    )

    _cr = [r[0] for r in _rows]
    _single = [r[1] for r in _rows]
    _union = [r[2] for r in _rows]
    _control = [r[3] for r in _rows]

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _lu, = _ax.plot(_cr, _union, "o-", color="tab:red", lw=2, label="union (pooled seeds)")
    _ls, = _ax.plot(_cr, _single, "^-", color="tab:green", label="single seed")
    _lc, = _ax.plot(_cr, _control, "v--", color="tab:orange", label="width-matched control")
    _ax.set_xlabel("corr  (within-group co-activation — structured entanglement)")
    _ax.set_ylabel("restricted-R² capture (AUC over 8 atoms)")
    _ax.set_title(f"k={int(ksae.value)}: capture holds, then correlated co-occurrence "
                  f"entangles the group")
    _ax.legend([_lu, _ls, _lc],
               ["union (pooled seeds)", "single seed", "width-matched control"],
               fontsize=8, loc="lower left")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
