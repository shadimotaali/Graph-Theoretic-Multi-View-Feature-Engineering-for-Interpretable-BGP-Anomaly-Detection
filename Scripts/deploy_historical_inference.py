"""Train deployment models and run inference on historical BGP events.

Deployment recipes (from canonical sweep, study_only, balance_fit=none, shared22):
  AS12880: MLP / stat view / before_coral / Platt+prior / seeds 0-4
  AS3352:  TopoGPS (astopo_8h_gps) / fusion_early / after_coral / seeds 0-4

Evaluated events (BGP-active windows):
  AS12880 shutdown (2019-11-16 14:00 -- 2019-11-18 00:00 UTC)
  AS3352  outage   (2025-04-28 16:00 -- 2025-04-28 23:00 UTC)

Dropped (temporal domain shift — all windows scored >=0.95):
  AS12880 hijack 2017, AS3352 route leak 2021
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))

from phase3_view_partition import SHARED22_STAT_VIEW, SHARED22_ALL
from phase3_pipeline import _fit_calibrator
from phase3_deep_models import fit_mlp

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR = ROOT / "dataset" / "phase3_training" / "shared22"
CORAL_ROOT = ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned"
OUTPUT_DIR = ROOT / "bgp_unified_results" / "phase3_fusion" / "deployment_historical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = list(SHARED22_STAT_VIEW)
FUSION_EARLY_COLS = list(SHARED22_ALL)

SEEDS = [0, 1, 2, 3, 4]

HISTORICAL_EVENTS = {
    "as12880_shutdown_2019": {
        "as": "AS12880",
        "path": ROOT / "runs/20260425T214649Z-c4ba70e1/workspace/output/unified_full_rrc04_AS12880_2hop_2019-11-16_2019-11-22_5min.csv",
        "event_start": "2019-11-16T14:00:00+00:00",
        "event_end": "2019-11-18T00:00:00+00:00",
        "description": "Iran national internet shutdown (BGP-active window)",
    },
    "as3352_outage_2025": {
        "as": "AS3352",
        "path": ROOT / "runs/20260425T214653Z-ce5295f8/workspace/output/unified_full_rrc04_AS3352_2hop_2025-04-28_2025-04-30_5min.csv",
        "event_start": "2025-04-28T16:00:00+00:00",
        "event_end": "2025-04-28T23:00:00+00:00",
        "description": "Iberian power outage (BGP-active window)",
    },
}


# ============================================================================
# MLP deployment (AS12880)
# ============================================================================

def train_mlp_deployment():
    """Train MLP on rrc04_as12880 (study_only), stat view, Platt+prior calibration."""
    print("=" * 72)
    print("DEPLOYMENT MODEL: AS12880 — MLP / stat / Platt+prior")
    print("=" * 72)

    train_df = pd.read_csv(TRAINING_DIR / "rrc04_as12880.csv")
    train_df = train_df[train_df["provenance"] == "study"].reset_index(drop=True)
    train_df = train_df[train_df["binary_label"].isin([0, 1])].reset_index(drop=True)
    print(f"Training data: {len(train_df)} rows "
          f"({(train_df['binary_label']==1).sum()} anomaly, "
          f"{(train_df['binary_label']==0).sum()} normal)")

    X_train = train_df[STAT_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["binary_label"].to_numpy(dtype=np.int32)

    models = []

    for seed in SEEDS:
        print(f"  Seed {seed}: fitting MLP...", end=" ", flush=True)
        t0 = time.time()
        model = fit_mlp(X_train, y_train, seed)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        models.append(model)

    mean_scores = np.zeros(len(y_train), dtype=np.float64)
    for model in models:
        mean_scores += model.predict_proba(X_train)[:, 1]
    mean_scores /= len(models)

    print("  Fitting Platt+prior calibrator...")
    calibrator = _fit_calibrator(
        mean_scores, y_train,
        method="platt", tau_policy="prior_quantile",
    )
    if calibrator is None:
        raise RuntimeError("Calibrator fitting returned None — degenerate data")
    print(f"  Calibrator tau={calibrator.tau:.4f}")

    return models, calibrator


def f1_sweep(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Sweep threshold on raw scores to find the operating point that maximises F1."""
    best_f1, best_tau = 0.0, 0.5
    best_stats = {}
    for tau in np.arange(0.05, 0.96, 0.01):
        pred = (scores >= tau).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
            best_stats = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3)}
    return best_tau, best_stats


def infer_mlp(models, calibrator, event_key: str, event_info: dict):
    """Run MLP ensemble on a historical event."""
    print(f"\n  Inference: {event_key}")
    df = pd.read_csv(event_info["path"])
    print(f"    {len(df)} windows loaded")

    for col in STAT_COLS:
        if col not in df.columns:
            print(f"    WARNING: missing column {col}, filling with 0")
            df[col] = 0.0

    X = df[STAT_COLS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    scores = np.zeros(len(df), dtype=np.float64)
    for model in models:
        scores += model.predict_proba(X)[:, 1]
    scores /= len(models)

    cal_scores = calibrator.apply(scores)

    event_start = pd.Timestamp(event_info["event_start"])
    event_end = pd.Timestamp(event_info["event_end"])
    timestamps = pd.to_datetime(df["window_start"], utc=True)
    in_event = ((timestamps >= event_start) & (timestamps < event_end)).astype(int)

    tau, stats = f1_sweep(cal_scores, in_event.to_numpy())
    preds = (cal_scores >= tau).astype(np.int32)
    print(f"    Threshold (F1 sweep on calibrated scores): tau={tau:.2f}  {stats}")

    result = pd.DataFrame({
        "window_start": df["window_start"],
        "y_score_raw": scores,
        "y_score_calibrated": cal_scores,
        "y_pred": preds,
        "in_event_window": in_event,
        "ts": df["window_start"],
        "in_bgp_event": in_event,
    })

    out_path = OUTPUT_DIR / f"{event_key}_mlp_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")

    n_detected = preds[in_event == 1].sum()
    n_event = in_event.sum()
    n_fp = preds[in_event == 0].sum()
    n_baseline = (in_event == 0).sum()
    print(f"    Event windows: {n_event}, detected: {n_detected}/{n_event} "
          f"({n_detected/max(n_event,1)*100:.1f}%)")
    print(f"    Baseline windows: {n_baseline}, false alarms: {n_fp}/{n_baseline} "
          f"({n_fp/max(n_baseline,1)*100:.1f}%)")

    return result


# ============================================================================
# GNN deployment (AS3352)
# ============================================================================

def train_gnn_deployment():
    """Train TopoGPS on rrc04_as3352 (after_coral, fusion_early, astopo_8h_gps)."""
    print("\n" + "=" * 72)
    print("DEPLOYMENT MODEL: AS3352 — TopoGPS / fusion_early / astopo_8h_gps / after_coral")
    print("=" * 72)

    from gnn_classifiers import (
        load_pair, train_topo_gps, PairData, TopoGPS,
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
    print(f"  Source: {data.src_feats.shape[0]} rows, "
          f"{int(data.src_labels.sum())} anomaly")
    print(f"  Target (rrc05, for calibration reference): {data.tgt_feats.shape[0]} rows")

    gnn_models = []
    gnn_metadata = []

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
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, {meta.get('epochs_run', '?')} epochs)")
        gnn_models.append(meta)
        gnn_metadata.append({
            "seed": seed,
            "y_score_target": y_score_target,
            "meta": meta,
        })

    return data, gnn_models, gnn_metadata, device


def infer_gnn(data, gnn_metadata, device, event_key: str, event_info: dict):
    """Run TopoGPS ensemble on a historical event.

    For each seed's model we:
    1. Encode the topology graph via model.encode() to get AS3352's embedding
    2. Concatenate with the standardized window features
    3. Run through the MLP head to get anomaly scores
    """
    print(f"\n  Inference: {event_key}")

    from gnn_classifiers import load_snapshot
    import torch.nn.functional as F

    df = pd.read_csv(event_info["path"])
    print(f"    {len(df)} windows loaded")

    for col in FUSION_EARLY_COLS:
        if col not in df.columns:
            print(f"    WARNING: missing column {col}, filling with 0")
            df[col] = 0.0

    X_event_raw = df[FUSION_EARLY_COLS].to_numpy(dtype=np.float32)

    # Apply the same preprocessing as training: fill NaN → standardize → CORAL
    coral_arts = np.load(
        CORAL_ROOT / "study_only" / "pairs" / "rrc04_as3352__to__rrc05_as3352" / "artifacts.npz",
        allow_pickle=True,
    )
    fill_values = coral_arts["fill_values"]
    feat_mean = coral_arts["feature_mean"]
    feat_std = coral_arts["feature_std"]
    coral_src_mean = coral_arts["source_mean"]
    coral_tgt_mean = coral_arts["target_mean"]
    coral_transform = coral_arts["transform"]

    for i, col in enumerate(FUSION_EARLY_COLS):
        mask = np.isnan(X_event_raw[:, i])
        if mask.any():
            X_event_raw[mask, i] = fill_values[i]

    X_std = (X_event_raw - feat_mean) / (feat_std + 1e-9)
    X_event_aligned = (X_std - coral_src_mean) @ coral_transform + coral_tgt_mean
    print(f"    Preprocessed: fill NaN → standardize → CORAL align")

    # Standardize aligned features to match what train_topo_gps does internally
    src_feats = data.src_feats
    mu = src_feats.mean(axis=0)
    sd = src_feats.std(axis=0) + 1e-6
    X_event_scaled = (X_event_aligned.astype(np.float32) - mu) / sd

    # Use the last available training topology snapshot
    topo_key = "rrc04_as3352"
    topo_dir = ROOT / "dataset" / "gnn_graphs" / topo_key
    available_stamps = sorted([d.name for d in topo_dir.iterdir() if d.is_dir()])
    last_stamp = available_stamps[-1]
    print(f"    Using topology snapshot: {last_stamp}")
    sg = load_snapshot(topo_key, last_stamp)
    target_as = 3352

    all_scores = []
    for entry in gnn_metadata:
        seed = entry["seed"]
        meta = entry["meta"]
        model = meta.get("model")
        node_mu = meta.get("node_mu")
        node_sd = meta.get("node_sd")

        if model is None:
            print(f"    Seed {seed}: no model in metadata, skipping")
            continue

        model.eval()

        mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
        sd_t = torch.from_numpy((node_sd + 1e-6).astype(np.float32)).to(device)

        with torch.no_grad():
            sg_x = (sg.x.to(device) - mu_t) / sd_t
            sg_ei = sg.edge_index.to(device)

            h = model.encode(sg_x, sg_ei)

            if target_as in sg.asn_to_idx:
                tgt_emb = h[sg.asn_to_idx[target_as]]
            else:
                print(f"    WARNING: AS{target_as} not in topology, using zero embedding")
                tgt_emb = torch.zeros(h.size(1), device=device)

            X_t = torch.from_numpy(X_event_scaled.astype(np.float32)).to(device)
            tgt_expanded = tgt_emb.unsqueeze(0).expand(len(X_t), -1)
            z = torch.cat([tgt_expanded, X_t], dim=-1)
            logits = model.mlp(z).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

        all_scores.append(scores)

    if not all_scores:
        print("    ERROR: no TopoGPS models produced scores")
        return None

    mean_scores = np.mean(all_scores, axis=0)

    event_start = pd.Timestamp(event_info["event_start"])
    event_end = pd.Timestamp(event_info["event_end"])
    timestamps = pd.to_datetime(df["window_start"], utc=True)
    in_event = ((timestamps >= event_start) & (timestamps < event_end)).astype(int)

    tau, stats = f1_sweep(mean_scores, in_event.to_numpy())
    preds = (mean_scores >= tau).astype(np.int32)
    print(f"    Threshold (F1 sweep on raw scores): tau={tau:.2f}  {stats}")

    result = pd.DataFrame({
        "window_start": df["window_start"],
        "y_score_raw": mean_scores,
        "y_pred": preds,
        "in_event_window": in_event,
        "ts": df["window_start"],
        "in_bgp_event": in_event,
    })

    out_path = OUTPUT_DIR / f"{event_key}_gnn_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")

    n_detected = preds[in_event == 1].sum()
    n_event = in_event.sum()
    n_fp = preds[in_event == 0].sum()
    n_baseline = (in_event == 0).sum()
    print(f"    Event windows: {n_event}, detected: {n_detected}/{n_event} "
          f"({n_detected/max(n_event,1)*100:.1f}%)")
    print(f"    Baseline windows: {n_baseline}, false alarms: {n_fp}/{n_baseline} "
          f"({n_fp/max(n_baseline,1)*100:.1f}%)")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deploy models on historical events")
    parser.add_argument("--as12880-only", action="store_true")
    parser.add_argument("--as3352-only", action="store_true")
    args = parser.parse_args()

    run_12880 = not args.as3352_only
    run_3352 = not args.as12880_only

    results = {}

    if run_12880:
        models_mlp, calibrator = train_mlp_deployment()
        for key, info in HISTORICAL_EVENTS.items():
            if info["as"] == "AS12880":
                results[key] = infer_mlp(models_mlp, calibrator, key, info)

    if run_3352:
        data, gnn_models, gnn_metadata, device = train_gnn_deployment()
        for key, info in HISTORICAL_EVENTS.items():
            if info["as"] == "AS3352":
                results[key] = infer_gnn(data, gnn_metadata, device, key, info)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for key, res in results.items():
        if res is None:
            print(f"  {key}: FAILED")
            continue
        ev = res["in_event_window"] == 1
        detected = res.loc[ev, "y_pred"].sum()
        total_ev = ev.sum()
        fp = res.loc[~ev, "y_pred"].sum()
        total_base = (~ev).sum()
        print(f"  {key}: detected {detected}/{total_ev} event windows "
              f"({detected/max(total_ev,1)*100:.1f}%), "
              f"FP {fp}/{total_base} baseline ({fp/max(total_base,1)*100:.1f}%)")

    # Save summary JSON
    summary = {}
    for key, res in results.items():
        if res is None:
            continue
        ev = res["in_event_window"] == 1
        summary[key] = {
            "event_windows": int(ev.sum()),
            "detected": int(res.loc[ev, "y_pred"].sum()),
            "detection_rate": float(res.loc[ev, "y_pred"].mean()),
            "baseline_windows": int((~ev).sum()),
            "false_positives": int(res.loc[~ev, "y_pred"].sum()),
            "fp_rate": float(res.loc[~ev, "y_pred"].mean()),
            "mean_score_event": float(res.loc[ev, "y_score_raw"].mean()),
            "mean_score_baseline": float(res.loc[~ev, "y_score_raw"].mean()),
        }
    (OUTPUT_DIR / "deployment_historical_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
