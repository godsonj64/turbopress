"""Generate all paper figures directly from the measured results JSON.

Every number plotted is read from results/*.json (no hand-typed values), so
the figures cannot drift from the experiments. Run: python paper/make_figures.py
Outputs vector PDFs (for LaTeX) and PNG previews into paper/figures/.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def _load(name):
    path = ROOT / "results" / name
    if not path.exists():
        raise SystemExit(
            f"missing {path} - regenerate it first (see module docstrings in "
            "turbopress/: harness, real_model, allocate, ldlq_micro)"
        )
    return json.loads(path.read_text())


qwen = _load("real_model_qwen06b.json")
alloc = _load("allocation_qwen06b.json")
sml = _load("real_model_v2.json")
equil06 = _load("equil_baseline_qwen06b.json")
equil17 = _load("equil_baseline_qwen17b.json")
harness = _load("harness_results.json")
ldlq_micro = _load("ldlq_micro.json")

C = {"nearest": "#4c4c4c", "tcq": "#1f77b4", "gptq": "#d62728",
     "mixed": "#2ca02c", "uniform": "#9467bd"}


def rows(res, prefix):
    return [r for r in res["results"] if r["label"].startswith(prefix)]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---- Fig 1: rate-distortion (bits vs KL) on Qwen3-0.6B ----------------------
def fig_rate_distortion():
    fig, (axk, axp) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    families = [("nearest", "Nearest (rotated scalar)", "o", "--"),
                ("tcq", "TCQ + equil.", "s", "-"),
                ("gptq", "GPTQ/LDLQ + equil.", "^", "-")]
    for key, lab, mk, ls in families:
        rs = rows(qwen, "nearest" if key == "nearest" else key)
        rs = [r for r in rs if r["label"].startswith(key)]
        rs = sorted(rs, key=lambda r: r["bits_per_weight"])
        b = [r["bits_per_weight"] for r in rs]
        kl = [r["mean_kl"] for r in rs]
        pp = [r["ppl_q"] for r in rs]
        axk.plot(b, kl, ls, marker=mk, color=C[key], label=lab)
        axp.plot(b, pp, ls, marker=mk, color=C[key], label=lab)
    for ax in (axk, axp):
        ax.set_yscale("log")
        ax.set_xlabel("bits / weight")
        ax.set_xticks([2, 3, 4])
    axk.set_ylabel(r"KL$(p_\mathrm{fp}\,\|\,p_q)$  (nats)")
    axp.set_ylabel("perplexity")
    axp.axhline(qwen["ppl_fp"], color="k", lw=0.8, ls=":", label=f"fp16 ({qwen['ppl_fp']:.1f})")
    axk.set_title("(a) Distortion vs. rate")
    axp.set_title("(b) Perplexity vs. rate")
    axk.legend(fontsize=7.5, loc="upper right")
    axp.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Qwen3-0.6B, all 196 decoder linears quantized", fontsize=10)
    save(fig, "fig1_rate_distortion")


# ---- Fig 2: error feedback / trellis vs nearest, KL bars per bit ------------
def fig_method_bars():
    bits = [2, 3, 4]
    def get(prefix, field):
        out = []
        for bb in bits:
            r = [x for x in qwen["results"]
                 if x["label"].startswith(prefix) and abs(x["bits_per_weight"] - bb) < 0.2][0]
            out.append(r[field])
        return out
    near = get("nearest", "mean_kl")
    tcq = get("tcq", "mean_kl")
    gptq = get("gptq", "mean_kl")
    x = np.arange(len(bits)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w, near, w, color=C["nearest"], label="Nearest")
    ax.bar(x, tcq, w, color=C["tcq"], label="TCQ+eq")
    ax.bar(x + w, gptq, w, color=C["gptq"], label="GPTQ/LDLQ+eq")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel(r"KL$(p_\mathrm{fp}\,\|\,p_q)$  (nats)")
    ax.set_title("Error feedback and trellis coding vs. nearest rounding")
    for xi, (n, t, g) in enumerate(zip(near, tcq, gptq)):
        ax.text(xi - w, n * 1.15, f"{n:.2f}", ha="center", fontsize=6.5)
        ax.text(xi, t * 1.15, f"{t:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + w, g * 1.15, f"{g:.2f}", ha="center", fontsize=6.5)
    ax.legend(fontsize=8)
    save(fig, "fig2_method_bars")


# ---- Fig 3: scale-up 135M -> 0.6B ------------------------------------------
def fig_scaleup():
    def tcq_eq(res, bb):
        return [x for x in res["results"]
                if "tcq" in x["label"] and "eq" in x["label"]
                and abs(x["bits_per_weight"] - bb) < 0.2][0]
    models = [("SmolLM2-135M", sml), ("Qwen3-0.6B", qwen)]
    fig, (axr, axk) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    width = 0.35
    xs = np.arange(2)
    for i, bb in enumerate([2, 4]):
        ratios = [tcq_eq(m, bb)["ppl_q"] / m["ppl_fp"] for _, m in models]
        axr.bar(xs + (i - 0.5) * width, ratios, width,
                label=f"{bb}-bit", color=[C["tcq"], C["gptq"]][i])
    axr.set_xticks(xs); axr.set_xticklabels([n for n, _ in models])
    axr.axhline(1.0, color="k", lw=0.8, ls=":")
    axr.set_ylabel(r"perplexity ratio  $\mathrm{ppl}_q/\mathrm{ppl}_\mathrm{fp}$")
    axr.set_title("(a) Relative perplexity gap (TCQ+eq)")
    axr.legend(fontsize=8)
    for i, bb in enumerate([2, 3, 4]):
        kls = [tcq_eq(m, bb)["mean_kl"] for _, m in models]
        axk.plot([0, 1], kls, "-o", label=f"{bb}-bit",
                 color=[C["nearest"], C["tcq"], C["gptq"]][i])
    axk.set_xticks([0, 1]); axk.set_xticklabels([n for n, _ in models])
    axk.set_yscale("log")
    axk.set_ylabel(r"KL$(p_\mathrm{fp}\,\|\,p_q)$  (nats)")
    axk.set_title("(b) KL vs. model size (TCQ+eq)")
    axk.legend(fontsize=8)
    save(fig, "fig3_scaleup")


# ---- Fig 4: per-layer sensitivity map --------------------------------------
def fig_sensitivity_map():
    sens = alloc["sensitivity"]
    order = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    short = ["q", "k", "v", "o", "gate", "up", "down"]
    layers = sorted({int(k.split(".", 1)[0]) for k in sens})
    M = np.full((len(order), len(layers)), np.nan)
    for k, v in sens.items():
        li, name = k.split(".", 1)
        if name in order:
            M[order.index(name), layers.index(int(li))] = v["2"]
    fig, ax = plt.subplots(figsize=(9.0, 2.9))
    im = ax.imshow(np.log10(M), aspect="auto", cmap="magma")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(short)
    ax.set_xticks(range(0, len(layers), 2))
    ax.set_xticklabels(range(0, len(layers), 2))
    ax.set_xlabel("decoder block index")
    ax.set_title("Per-layer 2-bit KL sensitivity  (single-layer KL, others fp)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(r"$\log_{10}$ KL")
    save(fig, "fig4_sensitivity_map")


# ---- Fig 5: allocation vs uniform ------------------------------------------
def fig_allocation():
    res = alloc["results"]
    fig, (axh, axk) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    # stacked bit histogram per target
    targets = [r["target_bits"] for r in res]
    b2 = [r["bit_histogram"]["2"] for r in res]
    b3 = [r["bit_histogram"]["3"] for r in res]
    b4 = [r["bit_histogram"]["4"] for r in res]
    x = np.arange(len(targets))
    axh.bar(x, b2, 0.5, label="2-bit", color="#c6dbef")
    axh.bar(x, b3, 0.5, bottom=b2, label="3-bit", color="#6baed6")
    axh.bar(x, b4, 0.5, bottom=np.array(b2) + np.array(b3), label="4-bit", color="#2171b5")
    axh.set_xticks(x); axh.set_xticklabels([f"budget {t}" for t in targets])
    axh.set_ylabel("number of layers")
    axh.set_title("(a) Allocated widths")
    axh.legend(fontsize=8)
    # KL: mixed vs uniform
    w = 0.35
    mk = [r["mixed_kl"] for r in res]
    uk = [r["uniform_kl"] for r in res]
    axk.bar(x - w / 2, mk, w, color=C["mixed"], label="mixed (allocated)")
    axk.bar(x + w / 2, uk, w, color=C["uniform"], label="uniform")
    for xi, r in enumerate(res):
        axk.text(xi - w / 2, r["mixed_kl"] + 0.02, f"{r['mixed_bits_per_weight']:.2f}b\n{r['mixed_kl']:.3f}",
                 ha="center", fontsize=6.5)
        axk.text(xi + w / 2, r["uniform_kl"] + 0.02, f"{r['uniform_bits_per_weight']:.2f}b\n{r['uniform_kl']:.3f}",
                 ha="center", fontsize=6.5)
    axk.set_xticks(x); axk.set_xticklabels([f"budget {t}" for t in targets])
    axk.set_ylabel(r"KL$(p_\mathrm{fp}\,\|\,p_q)$  (nats)")
    axk.set_title("(b) Mixed vs. uniform")
    axk.legend(fontsize=8)
    save(fig, "fig5_allocation")


# ---- Fig 6: QJL-for-weights negative result (depth compounding) ------------
def fig_qjl_negative():
    # Round-1 synthetic 16-layer chain at equal ~3-bit budget, read from the
    # harness results (isotropic activations).
    comp = {
        r["label"]: r
        for r in harness["results"]["compounding"]
        if r["act_mode"] == "isotropic"
    }
    pair = [comp["nearest 3b"], comp["2b + QJL k=d (=3b)"]]
    labels = [f"nearest 3b\n({pair[0]['eff_bits']:.2f}b)",
              f"2b+QJL k=d\n({pair[1]['eff_bits']:.2f}b)"]
    kl = [r["final_kl"] for r in pair]
    top1 = [r["top1_agreement"] for r in pair]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, kl, w, color=C["nearest"], label="KL(fp||q)")
    ax.set_ylabel("KL (nats)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, top1, w, color=C["tcq"], label="top-1 agree")
    ax2.set_ylabel("top-1 agreement"); ax2.set_ylim(0, 1)
    ax2.grid(False)
    ax.set_title("QJL bias-correction for weights (16-layer chain, equal budget)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    save(fig, "fig6_qjl_negative")


# ---- Fig 7: LDLQ controlled microbenchmark ---------------------------------
def fig_ldlq_micro():
    # Reproducible measurement: python -m turbopress.ldlq_micro
    res = ldlq_micro["results"]
    bits = [r["bits"] for r in res]
    near = [r["nearest_loss_mean"] for r in res]
    near_sd = [r["nearest_loss_std"] for r in res]
    ldlq = [r["ldlq_loss_mean"] for r in res]
    ldlq_sd = [r["ldlq_loss_std"] for r in res]
    x = np.arange(len(bits)); w = 0.35
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.bar(x - w / 2, near, w, yerr=near_sd, capsize=2,
           color=C["nearest"], label="Nearest")
    ax.bar(x + w / 2, ldlq, w, yerr=ldlq_sd, capsize=2,
           color=C["gptq"], label="LDLQ")
    for xi, r in enumerate(res):
        ax.text(xi, max(r["nearest_loss_mean"], r["ldlq_loss_mean"]) * 1.06,
                f"{r['gain']:.2f}$\\times$", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel(r"$\mathrm{tr}[(W-\hat W)H(W-\hat W)^\top]/nd$")
    n_seeds = ldlq_micro["settings"]["seeds"]
    ax.set_title("LDLQ reduces Hessian-weighted error\n"
                 f"(synthetic, correlated $H$; mean$\\pm$sd over {n_seeds} seeds)")
    ax.legend(fontsize=8)
    save(fig, "fig7_ldlq_micro")


# ---- Fig 8: equilibration exponent ablation (Proposition 1) -----------------
def fig_equil_exponent():
    # Controlled comparison: identical rotated scalar quantizer, only the
    # activation fold differs. no-eq floor vs AWQ sqrt fold vs quarter fold.
    models = [("Qwen3-0.6B", equil06), ("Qwen3-1.7B", equil17)]
    variants = [("no-eq", "no equilibration", C["nearest"]),
                ("awq-sqrt", r"AWQ fold $s_j = m_j^{1/2}$", C["uniform"]),
                ("quarter", r"quarter fold $s_j = (m_j/c_j)^{1/4}$", C["tcq"])]
    bits = [2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, (mname, res) in zip(axes, models):
        by_label = {r["label"]: r for r in res["results"]}
        x = np.arange(len(bits)); w = 0.26
        for vi, (vkey, vlab, color) in enumerate(variants):
            kls = [by_label[f"scalar {b}b {vkey}"]["mean_kl"] for b in bits]
            ax.bar(x + (vi - 1) * w, kls, w, color=color, label=vlab)
            for xi, kl in zip(x, kls):
                ax.text(xi + (vi - 1) * w, kl * 1.12, f"{kl:.2f}",
                        ha="center", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([f"{b}-bit" for b in bits])
        ax.set_title(mname)
    axes[0].set_ylabel(r"KL$(p_\mathrm{fp}\,\|\,p_q)$  (nats)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Equilibration exponent ablation (rotated scalar quantizer)",
                 fontsize=10)
    save(fig, "fig8_equil_exponent")


for f in [fig_rate_distortion, fig_method_bars, fig_scaleup, fig_sensitivity_map,
          fig_allocation, fig_qjl_negative, fig_ldlq_micro, fig_equil_exponent]:
    f()
    print("wrote", f.__name__)
print("figures ->", FIG)
