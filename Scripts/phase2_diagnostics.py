"""Supervisor-driven §5.1/§6 diagnostics for CORAL alignment.

For each source→target pair and a given feature space (default shared22):
  1. Load labeled windows (binary_label) from phase3_training/<space>/.
  2. Run CORAL on source to align it with target (reuses coral_align).
  3. Emit 2x2 PCA panel: rows = {before, after}, cols = {color by domain, color by label}.
  4. Emit 2x2 t-SNE panel with the same layout.
  5. Annotate PC1+PC2 variance explained on each panel (answers the supervisor's
     "PCA looked aligned, why didn't classification follow" question: the plot is
     a low-variance subspace of the 22D alignment).

Outputs: bgp_unified_results/phase2_diagnostics/<src>__to__<tgt>/
  - alignment_pca_labeled.{png,pdf}
  - alignment_pca_multi_domain.{png,pdf}
  - alignment_pca_multi_label.{png,pdf}
  - pca_scree.{png,pdf}
  - alignment_tsne_labeled.{png,pdf}
  - diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = PROJECT_ROOT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))
from coral_phase2_runner import coral_align, META_COLUMNS  # type: ignore

RNG_SEED = 42
DEFAULT_PAIRS = [
    ("rrc04_as12880", "rrc04_as3352"),   # D1 -> D2
    ("rrc04_as12880", "rrc05_as12880"),  # D1 -> D3
    ("rrc04_as12880", "rrc05_as3352"),   # D1 -> D4
    ("rrc04_as3352",  "rrc05_as12880"),  # D2 -> D3
    ("rrc04_as3352",  "rrc05_as3352"),   # D2 -> D4
    ("rrc05_as12880", "rrc05_as3352"),   # D3 -> D4
]

DOMAIN_CMAP = {"source": "#d1495b", "target": "#2f6690"}
LABEL_CMAP = {0: "#2f6690", 1: "#e07b00"}  # normal / anomaly


@dataclass
class PairFrames:
    source: pd.DataFrame
    target: pd.DataFrame
    features: list[str]


def load_pair(space: str, source: str, target: str) -> PairFrames:
    root = PROJECT_ROOT / "dataset" / "phase3_training" / space
    s = pd.read_csv(root / f"{source}.csv")
    t = pd.read_csv(root / f"{target}.csv")
    drop = set(META_COLUMNS) | {"binary_label", "discovered_label",
                                "event_id", "provenance", "incident_type"}
    features = [c for c in s.columns if c not in drop]
    features = [c for c in features if c in t.columns]
    return PairFrames(source=s, target=t, features=features)


def subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= cap else rng.choice(n, size=cap, replace=False)


def scatter_domain(ax, coords_sb, coords_sa, coords_tg, side: str):
    x = coords_sb if side == "before" else coords_sa
    ax.scatter(coords_tg[:, 0], coords_tg[:, 1], s=8, alpha=0.3,
               color=DOMAIN_CMAP["target"], label="Target", rasterized=True)
    ax.scatter(x[:, 0], x[:, 1], s=8, alpha=0.3,
               color=DOMAIN_CMAP["source"],
               label=f"Source ({side})", rasterized=True)


def scatter_label(ax, coords_sb, coords_sa, coords_tg,
                  s_lab, t_lab, side: str):
    x = coords_sb if side == "before" else coords_sa
    for lab, marker in [(0, "."), (1, "x")]:
        m = t_lab == lab
        if m.any():
            ax.scatter(coords_tg[m, 0], coords_tg[m, 1], s=10, alpha=0.35,
                       marker=marker, color=LABEL_CMAP[lab],
                       label=f"target {'normal' if lab==0 else 'anomaly'}",
                       rasterized=True)
    for lab, marker in [(0, "o"), (1, "^")]:
        m = s_lab == lab
        if m.any():
            ax.scatter(x[m, 0], x[m, 1], s=10, alpha=0.35,
                       marker=marker, color=LABEL_CMAP[lab],
                       edgecolor="black", linewidth=0.1,
                       label=f"source {'normal' if lab==0 else 'anomaly'} ({side})",
                       rasterized=True)


def run_pca(stacked: np.ndarray):
    """Fit PCA with all components so cumulative variance is known."""
    n_comp = stacked.shape[1]
    pca = PCA(n_components=n_comp, random_state=RNG_SEED)
    coords = pca.fit_transform(stacked)
    return coords, pca.explained_variance_ratio_


def components_for_variance(var_ratio: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> dict:
    cum = np.cumsum(var_ratio)
    out = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum, t) + 1)
        out[f"n_for_{int(t * 100)}pct"] = int(min(idx, len(cum)))
    return out


def save_fig(fig, out: Path):
    """Save a figure as both PNG (raster preview) and PDF (vector, for the paper).

    Scatter layers are created with rasterized=True, so the PDF keeps the dense
    point clouds as embedded raster (small file) while axes, ticks, titles and
    legends stay as selectable vector text.
    """
    out = Path(out)
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    # dpi=300 controls the resolution of the rasterized scatter layers inside
    # the PDF; axes, ticks, titles and legends remain vector regardless.
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def plot_scree(var_ratio: np.ndarray, thresholds_info: dict, out: Path):
    cum = np.cumsum(var_ratio)
    xs = np.arange(1, len(var_ratio) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(xs, var_ratio * 100, color="#d1495b", alpha=0.7, label="per-PC variance (%)")
    ax1.set_xlabel("PC index")
    ax1.set_ylabel("variance explained (%)", color="#d1495b")
    ax1.tick_params(axis="y", labelcolor="#d1495b")
    ax1.set_xticks(xs)
    ax2 = ax1.twinx()
    ax2.plot(xs, cum * 100, color="#2f6690", marker="o", label="cumulative (%)")
    ax2.set_ylabel("cumulative variance (%)", color="#2f6690")
    ax2.tick_params(axis="y", labelcolor="#2f6690")
    ax2.set_ylim(0, 105)
    for pct, color in [(90, "#888"), (95, "#555"), (99, "#222")]:
        ax2.axhline(pct, linestyle=":", color=color, linewidth=0.8)
        k = thresholds_info[f"n_for_{pct}pct"]
        ax2.annotate(f"{pct}% @ PC{k}", xy=(k, pct), xytext=(k + 0.4, pct - 4),
                     fontsize=8, color=color)
    fig.suptitle("Scree: per-PC and cumulative variance explained (22D feature space)")
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def run_tsne(stacked: np.ndarray):
    perplexity = min(30, max(5, (stacked.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, random_state=RNG_SEED, init="pca",
                perplexity=perplexity, max_iter=1000, learning_rate="auto")
    return tsne.fit_transform(stacked)


def plot_pc_grid(pca_coords_sb, pca_coords_sa, pca_coords_tg, s_lab, t_lab,
                 pca_var: np.ndarray, color_mode: str, out: Path):
    """3 rows (PC1-2, PC3-4, PC5-6) x 2 cols (before, after) coloured by `color_mode`."""
    pair_idx = [(0, 1), (2, 3), (4, 5)]
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for row, (i, j) in enumerate(pair_idx):
        ve = (pca_var[i] + pca_var[j]) * 100
        sb = pca_coords_sb[:, [i, j]]
        sa = pca_coords_sa[:, [i, j]]
        tg = pca_coords_tg[:, [i, j]]
        for col, (coords, side) in enumerate([(sb, "before"), (sa, "after")]):
            ax = axes[row, col]
            if color_mode == "domain":
                ax.scatter(tg[:, 0], tg[:, 1], s=8, alpha=0.3,
                           color=DOMAIN_CMAP["target"], label="Target", rasterized=True)
                ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.3,
                           color=DOMAIN_CMAP["source"],
                           label=f"Source ({side})", rasterized=True)
            else:  # label
                for lab, marker in [(0, "."), (1, "x")]:
                    m = t_lab == lab
                    if m.any():
                        ax.scatter(tg[m, 0], tg[m, 1], s=10, alpha=0.35,
                                   marker=marker, color=LABEL_CMAP[lab],
                                   label=f"target {'normal' if lab == 0 else 'anomaly'}",
                                   rasterized=True)
                for lab, marker in [(0, "o"), (1, "^")]:
                    m = s_lab == lab
                    if m.any():
                        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.35,
                                   marker=marker, color=LABEL_CMAP[lab],
                                   edgecolor="black", linewidth=0.1,
                                   label=f"source {'normal' if lab == 0 else 'anomaly'} ({side})",
                                   rasterized=True)
            ax.set_title(f"PC{i+1}-PC{j+1}  ({side} CORAL) — {ve:.1f}% var")
            ax.set_xlabel(f"PC{i+1}")
            ax.set_ylabel(f"PC{j+1}")
            ax.grid(alpha=0.2)
        if row == 0:
            axes[row, 1].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(
        f"Lower-variance PC pairs often carry the class-conditional structure that PC1-PC2 hides. "
        f"Colour mode: {color_mode}.",
        fontsize=10, y=0.995,
    )
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def plot_2x2(coords_sb, coords_sa, coords_tg, s_lab, t_lab,
             title_prefix: str, var_explained: str, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    scatter_domain(axes[0, 0], coords_sb, coords_sa, coords_tg, "before")
    scatter_domain(axes[0, 1], coords_sb, coords_sa, coords_tg, "after")
    scatter_label(axes[1, 0], coords_sb, coords_sa, coords_tg, s_lab, t_lab, "before")
    scatter_label(axes[1, 1], coords_sb, coords_sa, coords_tg, s_lab, t_lab, "after")
    axes[0, 0].set_title(f"{title_prefix} — by domain, before CORAL")
    axes[0, 1].set_title(f"{title_prefix} — by domain, after CORAL")
    axes[1, 0].set_title(f"{title_prefix} — by label, before CORAL")
    axes[1, 1].set_title(f"{title_prefix} — by label, after CORAL")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"{title_prefix.split()[0]} 1")
        ax.set_ylabel(f"{title_prefix.split()[0]} 2")
    axes[0, 1].legend(loc="best", frameon=False, fontsize=8)
    axes[1, 1].legend(loc="best", frameon=False, fontsize=8)
    if var_explained:
        fig.suptitle(var_explained, fontsize=10, y=0.995)
    fig.tight_layout()
    save_fig(fig, out)
    plt.close(fig)


def diagnose_pair(space: str, source: str, target: str,
                  out_root: Path, max_samples: int = 3000) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    frames = load_pair(space, source, target)
    feats = frames.features

    src_X = frames.source[feats].to_numpy(dtype=float)
    tgt_X = frames.target[feats].to_numpy(dtype=float)
    src_y = frames.source["binary_label"].to_numpy()
    tgt_y = frames.target["binary_label"].to_numpy()

    scaler = StandardScaler().fit(np.vstack([src_X, tgt_X]))
    src_s = scaler.transform(src_X)
    tgt_s = scaler.transform(tgt_X)
    src_aligned, _ = coral_align(src_s, tgt_s, reg=1e-6)

    s_idx = subsample(len(src_s), max_samples, rng)
    t_idx = subsample(len(tgt_s), max_samples, rng)
    sb, sa, tg = src_s[s_idx], src_aligned[s_idx], tgt_s[t_idx]
    s_lab, t_lab = src_y[s_idx], tgt_y[t_idx]

    # PCA on stacked (sb + sa + tg) so the 2D space is common.
    stacked_pca = np.vstack([sb, sa, tg])
    pca_coords, pca_var = run_pca(stacked_pca)
    n_sb, n_sa = len(sb), len(sa)
    pca_sb_full = pca_coords[:n_sb]
    pca_sa_full = pca_coords[n_sb:n_sb + n_sa]
    pca_tg_full = pca_coords[n_sb + n_sa:]
    pca_sb = pca_sb_full[:, :2]
    pca_sa = pca_sa_full[:, :2]
    pca_tg = pca_tg_full[:, :2]
    thresholds_info = components_for_variance(pca_var)
    pca_header = (
        f"PCA variance explained: PC1={pca_var[0]*100:.1f}%  "
        f"PC2={pca_var[1]*100:.1f}%  "
        f"(PC1+PC2={pca_var[:2].sum()*100:.1f}%; "
        f"90% at PC{thresholds_info['n_for_90pct']}, "
        f"95% at PC{thresholds_info['n_for_95pct']}, "
        f"99% at PC{thresholds_info['n_for_99pct']} of {len(pca_var)}D)."
    )

    # t-SNE on the same stacked matrix for consistency.
    tsne_coords = run_tsne(stacked_pca)
    tsne_sb = tsne_coords[:n_sb]
    tsne_sa = tsne_coords[n_sb:n_sb + n_sa]
    tsne_tg = tsne_coords[n_sb + n_sa:]

    out_dir = out_root / f"{source}__to__{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_2x2(pca_sb, pca_sa, pca_tg, s_lab, t_lab,
             "PC", pca_header, out_dir / "alignment_pca_labeled.png")
    plot_pc_grid(pca_sb_full, pca_sa_full, pca_tg_full, s_lab, t_lab,
                 pca_var, "domain", out_dir / "alignment_pca_multi_domain.png")
    plot_pc_grid(pca_sb_full, pca_sa_full, pca_tg_full, s_lab, t_lab,
                 pca_var, "label", out_dir / "alignment_pca_multi_label.png")
    plot_scree(pca_var, thresholds_info, out_dir / "pca_scree.png")
    plot_2x2(tsne_sb, tsne_sa, tsne_tg, s_lab, t_lab,
             "tSNE", "t-SNE preserves local neighborhoods; class-conditional separation visible where PCA compresses it away.",
             out_dir / "alignment_tsne_labeled.png")

    diag = {
        "source": source,
        "target": target,
        "feature_space": space,
        "n_features": len(feats),
        "n_source": int(len(src_s)),
        "n_target": int(len(tgt_s)),
        "source_anomaly_rate": float((src_y == 1).mean()),
        "target_anomaly_rate": float((tgt_y == 1).mean()),
        "pca_var_explained_top2": [float(v) for v in pca_var[:2]],
        "pca_var_explained_top6": [float(v) for v in pca_var[:6]],
        "pca_cum_top2": float(pca_var[:2].sum()),
        "pca_cum_top6": float(pca_var[:6].sum()),
        "pca_components_total": int(len(pca_var)),
        "pca_components_for_90pct": thresholds_info["n_for_90pct"],
        "pca_components_for_95pct": thresholds_info["n_for_95pct"],
        "pca_components_for_99pct": thresholds_info["n_for_99pct"],
        "pca_full_explained_variance_ratio": [float(v) for v in pca_var],
        "plot_subsample": int(max_samples),
    }
    with open(out_dir / "diagnostics.json", "w") as h:
        json.dump(diag, h, indent=2)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="shared22")
    ap.add_argument("--pairs", nargs="*",
                    help="Pairs as src:tgt; defaults to all 6.")
    ap.add_argument("--out-root", default=str(
        PROJECT_ROOT / "bgp_unified_results" / "phase2_diagnostics"))
    ap.add_argument("--max-samples", type=int, default=3000)
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = DEFAULT_PAIRS if not args.pairs else [tuple(p.split(":")) for p in args.pairs]
    summary = []
    for src, tgt in pairs:
        print(f"[diag] {src} -> {tgt}")
        d = diagnose_pair(args.space, src, tgt, out_root, args.max_samples)
        summary.append(d)
        print(f"    PC1={d['pca_var_explained_top2'][0]*100:.1f}%  "
              f"PC2={d['pca_var_explained_top2'][1]*100:.1f}%  "
              f"sum(PC1..PC6)={d['pca_cum_top6']*100:.1f}%  "
              f"|  90%@PC{d['pca_components_for_90pct']}  "
              f"95%@PC{d['pca_components_for_95pct']}  "
              f"99%@PC{d['pca_components_for_99pct']}")
    with open(out_root / "diagnostics_summary.json", "w") as h:
        json.dump(summary, h, indent=2)
    print(f"[diag] wrote {len(summary)} pair diagnostics to {out_root}")


if __name__ == "__main__":
    main()
