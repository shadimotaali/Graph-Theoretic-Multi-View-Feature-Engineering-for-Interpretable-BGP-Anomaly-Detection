"""SHAP explainability for deployment models (Phase E).

Produces:
  1. Global beeswarm plots for MLP (AS12880) and TopoGPS (AS3352)
  2. Waterfall plots for the top anomaly windows in each historical event
  3. Combined side-by-side beeswarm figure for the paper

Output: figures/shap_*.pdf and figures/shap_*.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))

from sklearn.preprocessing import StandardScaler
from phase3_view_partition import SHARED22_STAT_VIEW, SHARED22_ALL
from phase3_pipeline import _fit_calibrator
from phase3_deep_models import fit_mlp

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAINING_DIR = ROOT / "dataset" / "phase3_training" / "shared22"
CORAL_ROOT = ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned"
HIST_DIR = ROOT / "bgp_unified_results" / "phase3_fusion" / "deployment_historical"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = list(SHARED22_STAT_VIEW)
FUSION_EARLY_COLS = list(SHARED22_ALL)

SEEDS = [0, 1, 2, 3, 4]
N_BACKGROUND = 100  # KernelExplainer background sample size
N_EXPLAIN = 200     # number of windows to explain for beeswarm
N_WATERFALL = 3     # top anomaly windows per event for waterfall plots

EXCLUDE_FEATURES = set()


# ============================================================================
# Utility
# ============================================================================

def filter_explanation(expl: shap.Explanation, raw_cols: list[str]) -> shap.Explanation:
    """Drop excluded features from a SHAP Explanation object."""
    keep = [i for i, c in enumerate(raw_cols) if c not in EXCLUDE_FEATURES]
    return shap.Explanation(
        values=expl.values[..., keep],
        base_values=expl.base_values,
        data=expl.data[..., keep] if expl.data is not None else None,
        feature_names=[expl.feature_names[i] for i in keep],
    )


def short_feature_name(name: str) -> str:
    """Shorten feature names for plot readability."""
    renames = {
        "innermost_core_size": "core_size",
        "rich_club_p75": "RC_p75",
        "rich_club_p90": "RC_p90",
        "rich_club_p95": "RC_p95",
        "symmetry_ratio": "symmetry",
        "std_edge_ixp_cosine_dist": "ixp_cos_std",
        "clustering_avg_local": "clustering",
        "frac_p2p_edges": "frac_p2p",
        "avg_ixp_cosine_dist": "ixp_cos_avg",
        "as_path_avg": "AS_path_avg",
        "as_path_max": "AS_path_max",
        "as_path_std": "AS_path_std",
        "edit_distance_avg": "edit_dist_avg",
        "edit_distance_max": "edit_dist_max",
        "unique_as_path_max": "uniq_AS_path",
        "vf_rate_delta": "VF_rate_Δ",
        "ego_filter_ratio": "ego_filt",
        "ego_origin_violations": "ego_origin_viol",
        "unique_peers": "uniq_peers",
        "origin_changes": "origin_chg",
    }
    return renames.get(name, name)


# ============================================================================
# MLP SHAP (AS12880)
# ============================================================================

def run_mlp_shap():
    """KernelExplainer on the deployed MLP ensemble for AS12880."""
    print("=" * 72)
    print("SHAP: AS12880 — MLP (stat view, 13 features)")
    print("=" * 72)

    # --- Train models (same as deployment) ---
    train_df = pd.read_csv(TRAINING_DIR / "rrc04_as12880.csv")
    train_df = train_df[train_df["provenance"] == "study"].reset_index(drop=True)
    train_df = train_df[train_df["binary_label"].isin([0, 1])].reset_index(drop=True)

    X_train = train_df[STAT_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["binary_label"].to_numpy(dtype=np.int32)

    models = []
    for seed in SEEDS:
        print(f"  Seed {seed}: fitting MLP...", end=" ", flush=True)
        t0 = time.time()
        model = fit_mlp(X_train, y_train, seed)
        print(f"done ({time.time()-t0:.1f}s)")
        models.append(model)

    # Calibrator
    mean_scores_train = np.zeros(len(y_train), dtype=np.float64)
    for m in models:
        mean_scores_train += m.predict_proba(X_train)[:, 1]
    mean_scores_train /= len(models)

    calibrator = _fit_calibrator(
        mean_scores_train, y_train,
        method="platt", tau_policy="prior_quantile",
    )
    if calibrator is None:
        raise RuntimeError("Calibrator fitting returned None")

    def mlp_predict(X: np.ndarray) -> np.ndarray:
        """Ensemble mean calibrated score."""
        scores = np.zeros(len(X), dtype=np.float64)
        for m in models:
            scores += m.predict_proba(X.astype(np.float32))[:, 1]
        scores /= len(models)
        return calibrator.apply(scores)

    # --- Background sample (stratified) ---
    rng = np.random.RandomState(42)
    idx_normal = np.where(y_train == 0)[0]
    idx_anomaly = np.where(y_train == 1)[0]
    n_bg_anom = min(N_BACKGROUND // 5, len(idx_anomaly))
    n_bg_norm = min(N_BACKGROUND - n_bg_anom, len(idx_normal))
    bg_idx = np.concatenate([
        rng.choice(idx_normal, n_bg_norm, replace=False),
        rng.choice(idx_anomaly, n_bg_anom, replace=False),
    ])
    X_background = X_train[bg_idx]

    print(f"  KernelExplainer: background={len(X_background)}")
    explainer = shap.KernelExplainer(mlp_predict, X_background)

    # --- Global beeswarm: explain a sample of training data ---
    idx_explain_norm = rng.choice(idx_normal, min(N_EXPLAIN * 4 // 5, len(idx_normal)), replace=False)
    idx_explain_anom = rng.choice(idx_anomaly, min(N_EXPLAIN // 5, len(idx_anomaly)), replace=False)
    idx_explain = np.concatenate([idx_explain_norm, idx_explain_anom])
    X_explain = X_train[idx_explain]

    print(f"  Computing SHAP values for {len(X_explain)} windows...")
    t0 = time.time()
    shap_values = explainer.shap_values(X_explain)
    print(f"  Done ({time.time()-t0:.1f}s)")

    feature_names = [short_feature_name(c) for c in STAT_COLS]
    explanation_raw = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_explain,
        feature_names=feature_names,
    )
    explanation = filter_explanation(explanation_raw, STAT_COLS)
    n_display = len([c for c in STAT_COLS if c not in EXCLUDE_FEATURES])

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.sca(ax)
    shap.plots.beeswarm(explanation, show=False, max_display=n_display)
    ax.set_title("AS12880 — MLP deployment model", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shap_beeswarm_as12880_mlp.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved beeswarm: figures/shap_beeswarm_as12880_mlp.pdf")

    # --- Historical event waterfall ---
    event_csv = HIST_DIR / "as12880_shutdown_2019_mlp_scores.csv"
    if event_csv.exists():
        print("\n  Waterfall: AS12880 shutdown 2019")
        ev_df = pd.read_csv(event_csv)
        ev_data = pd.read_csv(
            ROOT / "runs/20260425T214649Z-c4ba70e1/workspace/output/"
            "unified_full_rrc04_AS12880_2hop_2019-11-16_2019-11-22_5min.csv"
        )
        ev_in = ev_df["in_bgp_event"] == 1
        ev_scores = ev_df.loc[ev_in, "y_score_calibrated"].to_numpy()
        top_idx = np.argsort(ev_scores)[::-1][:N_WATERFALL]
        global_idx = ev_df.index[ev_in].to_numpy()[top_idx]

        X_event = ev_data[STAT_COLS].to_numpy(dtype=np.float32)
        X_event = np.nan_to_num(X_event, nan=0.0)
        X_top = X_event[global_idx]

        print(f"    Computing SHAP for top-{N_WATERFALL} anomaly windows...")
        shap_top = explainer.shap_values(X_top)

        for i, (gi, si) in enumerate(zip(global_idx, top_idx)):
            ts = ev_df.loc[gi, "window_start"]
            score = ev_scores[si]
            expl_raw = shap.Explanation(
                values=shap_top[i],
                base_values=explainer.expected_value,
                data=X_top[i],
                feature_names=feature_names,
            )
            expl = filter_explanation(expl_raw, STAT_COLS)
            fig, ax = plt.subplots(figsize=(7, 4))
            plt.sca(ax)
            shap.plots.waterfall(expl, show=False, max_display=n_display)
            ax.set_title(f"AS12880 shutdown — {ts} (score={score:.3f})",
                         fontsize=10)
            plt.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(
                    FIG_DIR / f"shap_waterfall_as12880_shutdown_{i+1}.{ext}",
                    dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"    Saved {N_WATERFALL} waterfall plots")

    return explanation


# ============================================================================
# TopoGPS SHAP (AS3352)
# ============================================================================

def run_gnn_shap():
    """KernelExplainer on the deployed TopoGPS ensemble for AS3352.

    Strategy: freeze the graph encoder, extract the fixed 32-dim embedding
    for AS3352, then explain the 22 window features through the MLP head.
    The explainer sees [22 window features] → score, with the graph embedding
    treated as a constant context vector (not explained, since those 17 node
    features are static topology, not per-window signals).
    """
    print("\n" + "=" * 72)
    print("SHAP: AS3352 — TopoGPS (fusion_early, 22 features)")
    print("=" * 72)

    from gnn_classifiers import (
        load_pair, train_topo_gps, TopoGPS,
        topo_key_for_domain, target_as_for_domain, window_to_stamp,
        load_snapshot, NODE_FEATURE_COLS, standardize,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    pair = "rrc04_as3352__to__rrc05_as3352"
    data = load_pair(
        pair=pair,
        coral_kind="study_only",
        balance_fit="none",
        alignment="after_coral",
        view="fusion_early",
        source_labels_from=None,
    )
    print(f"  Source: {data.src_feats.shape[0]} rows")

    # Train all 5 seeds
    gnn_entries = []
    for seed in SEEDS:
        print(f"\n  Seed {seed}: training TopoGPS...", end=" ", flush=True)
        t0 = time.time()
        y_score_target, y_pred_target, meta = train_topo_gps(
            data,
            source_domain="rrc04_as3352",
            target_domain="rrc05_as3352",
            seed=seed,
            epochs=50,
            device=device,
            patience=3,
        )
        print(f"done ({time.time()-t0:.1f}s)")
        gnn_entries.append(meta)

    # Load one topology snapshot for the fixed embedding
    topo_key = "rrc04_as3352"
    topo_dir = ROOT / "dataset" / "gnn_graphs" / topo_key
    available_stamps = sorted([d.name for d in topo_dir.iterdir() if d.is_dir()])
    last_stamp = available_stamps[-1]
    print(f"\n  Topology snapshot: {last_stamp}")
    sg = load_snapshot(topo_key, last_stamp)
    target_as = 3352

    # Extract fixed graph embeddings from all 5 models
    tgt_embeddings = []
    for meta in gnn_entries:
        model = meta["model"]
        node_mu = meta["node_mu"]
        node_sd = meta["node_sd"]
        model.eval()
        mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
        sd_t = torch.from_numpy((node_sd + 1e-6).astype(np.float32)).to(device)
        with torch.no_grad():
            sg_x = (sg.x.to(device) - mu_t) / sd_t
            sg_ei = sg.edge_index.to(device)
            h = model.encode(sg_x, sg_ei)
            emb = h[sg.asn_to_idx[target_as]]
        tgt_embeddings.append(emb)

    # Prepare standardized training features (same as train_topo_gps does)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)
    scaler = StandardScaler().fit(data.src_feats)
    src_mu = scaler.mean_.astype(np.float32)
    src_sd = scaler.scale_.astype(np.float32)

    def gnn_predict_window(X_window: np.ndarray) -> np.ndarray:
        """Predict from 22 window features using ensemble of TopoGPS MLP heads.

        X_window: (n, 22) already standardized window features
        Returns: (n,) mean sigmoid score
        """
        X_t = torch.from_numpy(X_window.astype(np.float32)).to(device)
        all_scores = []
        for meta, emb in zip(gnn_entries, tgt_embeddings):
            model = meta["model"]
            model.eval()
            with torch.no_grad():
                emb_exp = emb.unsqueeze(0).expand(len(X_t), -1)
                z = torch.cat([emb_exp, X_t], dim=-1)
                logits = model.mlp(z).squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
        return np.mean(all_scores, axis=0)

    # Standardize source features for explanation
    X_src_std = (data.src_feats.astype(np.float32) - src_mu) / src_sd
    y_src = data.src_labels

    # Background sample (stratified)
    rng = np.random.RandomState(42)
    idx_normal = np.where(y_src == 0)[0]
    idx_anomaly = np.where(y_src == 1)[0]
    n_bg_anom = min(N_BACKGROUND // 5, len(idx_anomaly))
    n_bg_norm = min(N_BACKGROUND - n_bg_anom, len(idx_normal))
    bg_idx = np.concatenate([
        rng.choice(idx_normal, n_bg_norm, replace=False),
        rng.choice(idx_anomaly, n_bg_anom, replace=False),
    ])
    X_background = X_src_std[bg_idx]

    print(f"  KernelExplainer: background={len(X_background)}")
    explainer = shap.KernelExplainer(gnn_predict_window, X_background)

    # Global beeswarm
    idx_explain_norm = rng.choice(idx_normal, min(N_EXPLAIN * 4 // 5, len(idx_normal)), replace=False)
    idx_explain_anom = rng.choice(idx_anomaly, min(N_EXPLAIN // 5, len(idx_anomaly)), replace=False)
    idx_explain = np.concatenate([idx_explain_norm, idx_explain_anom])
    X_explain = X_src_std[idx_explain]

    feature_names = [short_feature_name(c) for c in FUSION_EARLY_COLS]

    print(f"  Computing SHAP values for {len(X_explain)} windows...")
    t0 = time.time()
    shap_values = explainer.shap_values(X_explain)
    print(f"  Done ({time.time()-t0:.1f}s)")

    explanation_raw = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_explain,
        feature_names=feature_names,
    )
    explanation = filter_explanation(explanation_raw, FUSION_EARLY_COLS)
    n_display_gnn = len([c for c in FUSION_EARLY_COLS if c not in EXCLUDE_FEATURES])

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.sca(ax)
    shap.plots.beeswarm(explanation, show=False, max_display=n_display_gnn)
    ax.set_title("AS3352 — TopoGPS deployment model", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shap_beeswarm_as3352_topogps.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved beeswarm: figures/shap_beeswarm_as3352_topogps.pdf")

    # --- Historical event waterfall ---
    event_csv = HIST_DIR / "as3352_outage_2025_gnn_scores.csv"
    if event_csv.exists():
        print("\n  Waterfall: AS3352 outage 2025")
        ev_df = pd.read_csv(event_csv)
        ev_raw = pd.read_csv(
            ROOT / "runs/20260425T214653Z-ce5295f8/workspace/output/"
            "unified_full_rrc04_AS3352_2hop_2025-04-28_2025-04-30_5min.csv"
        )

        # Preprocess event features same as inference
        coral_arts = np.load(
            CORAL_ROOT / "study_only" / "pairs" / "rrc04_as3352__to__rrc05_as3352"
            / "artifacts.npz",
            allow_pickle=True,
        )
        fill_values = coral_arts["fill_values"]
        feat_mean = coral_arts["feature_mean"]
        feat_std = coral_arts["feature_std"]
        coral_src_mean = coral_arts["source_mean"]
        coral_tgt_mean = coral_arts["target_mean"]
        coral_transform = coral_arts["transform"]

        X_ev_raw = ev_raw[FUSION_EARLY_COLS].to_numpy(dtype=np.float32)
        for i in range(X_ev_raw.shape[1]):
            mask = np.isnan(X_ev_raw[:, i])
            if mask.any():
                X_ev_raw[mask, i] = fill_values[i]
        X_std_ev = (X_ev_raw - feat_mean) / (feat_std + 1e-9)
        X_aligned = (X_std_ev - coral_src_mean) @ coral_transform + coral_tgt_mean
        X_ev_scaled = (X_aligned.astype(np.float32) - src_mu) / src_sd

        ev_in = ev_df["in_bgp_event"] == 1
        ev_scores = ev_df.loc[ev_in, "y_score_raw"].to_numpy()
        top_idx = np.argsort(ev_scores)[::-1][:N_WATERFALL]
        global_idx = ev_df.index[ev_in].to_numpy()[top_idx]

        X_top = X_ev_scaled[global_idx]

        print(f"    Computing SHAP for top-{N_WATERFALL} anomaly windows...")
        shap_top = explainer.shap_values(X_top)

        for i, (gi, si) in enumerate(zip(global_idx, top_idx)):
            ts = ev_df.loc[gi, "window_start"]
            score = ev_scores[si]
            expl_raw = shap.Explanation(
                values=shap_top[i],
                base_values=explainer.expected_value,
                data=X_top[i],
                feature_names=feature_names,
            )
            expl = filter_explanation(expl_raw, FUSION_EARLY_COLS)
            fig, ax = plt.subplots(figsize=(7, 4))
            plt.sca(ax)
            shap.plots.waterfall(expl, show=False, max_display=n_display_gnn)
            ax.set_title(f"AS3352 outage — {ts} (score={score:.3f})",
                         fontsize=10)
            plt.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(
                    FIG_DIR / f"shap_waterfall_as3352_outage_{i+1}.{ext}",
                    dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"    Saved {N_WATERFALL} waterfall plots")

    return explanation


# ============================================================================
# Combined figure for paper
# ============================================================================

def make_combined_beeswarm(expl_mlp=None, expl_gnn=None):
    """Side-by-side beeswarm figure (Fig. X in paper).

    If Explanation objects are passed, renders directly with shap.plots.
    Otherwise falls back to stitching saved PNGs.
    """
    if expl_mlp is not None and expl_gnn is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        plt.sca(axes[0])
        shap.plots.beeswarm(expl_mlp, show=False, max_display=13)
        axes[0].set_title("(a) AS12880 — MLP", fontsize=12)

        plt.sca(axes[1])
        shap.plots.beeswarm(expl_gnn, show=False, max_display=22)
        axes[1].set_title("(b) AS3352 — TopoGPS", fontsize=12)

        plt.tight_layout(w_pad=3)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"shap_summary_as12880_as3352.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved combined: figures/shap_summary_as12880_as3352.pdf")
        return

    mlp_png = FIG_DIR / "shap_beeswarm_as12880_mlp.png"
    gnn_png = FIG_DIR / "shap_beeswarm_as3352_topogps.png"
    if not (mlp_png.exists() and gnn_png.exists()):
        print("  Skipping combined figure — individual beeswarms not found")
        return

    from PIL import Image
    img_mlp = Image.open(mlp_png)
    img_gnn = Image.open(gnn_png)

    h = max(img_mlp.height, img_gnn.height)
    if img_mlp.height != h:
        w = int(img_mlp.width * h / img_mlp.height)
        img_mlp = img_mlp.resize((w, h), Image.LANCZOS)
    if img_gnn.height != h:
        w = int(img_gnn.width * h / img_gnn.height)
        img_gnn = img_gnn.resize((w, h), Image.LANCZOS)

    combined = Image.new("RGB", (img_mlp.width + img_gnn.width + 40, h), "white")
    combined.paste(img_mlp, (0, 0))
    combined.paste(img_gnn, (img_mlp.width + 40, 0))
    combined.save(FIG_DIR / "shap_summary_as12880_as3352.png", dpi=(200, 200))
    combined.save(FIG_DIR / "shap_summary_as12880_as3352.pdf")
    print("  Saved combined: figures/shap_summary_as12880_as3352.pdf")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for deployment models")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--gnn-only", action="store_true")
    args = parser.parse_args()

    run_mlp = not args.gnn_only
    run_gnn = not args.mlp_only

    expl_mlp, expl_gnn = None, None
    if run_mlp:
        expl_mlp = run_mlp_shap()
    if run_gnn:
        expl_gnn = run_gnn_shap()

    make_combined_beeswarm(expl_mlp, expl_gnn)

    print("\n" + "=" * 72)
    print("SHAP DONE — all figures saved to figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
