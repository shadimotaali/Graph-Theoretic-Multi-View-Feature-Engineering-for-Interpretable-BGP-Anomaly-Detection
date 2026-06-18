#!/usr/bin/env python3
"""
Per-class Frobenius distance of cov(X_s|y) vs cov(X_t|y) before/after CORAL.

Loads only real data from bgp_unified_results/phase3_fusion/coral_aligned/study_only/.
For each directed pair (src -> tgt) computes, on the 22 shared features:

  global: ||cov(X_s)          - cov(X_t)||_F       before / after
  y=0:    ||cov(X_s | y=0)    - cov(X_t | y=0)||_F before / after
  y=1:    ||cov(X_s | y=1)    - cov(X_t | y=1)||_F before / after

Global-before is cross-checked against metrics.json::cov_gap_before as a sanity
check (expected exact match; we also print cov_gap_after for reference).

Output: bgp_unified_results/phase3_fusion/summaries/coral_frobenius_per_class.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORAL_ALIGNED_ROOT = PROJECT_ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned"
OUT_DIR = PROJECT_ROOT / "bgp_unified_results" / "phase3_fusion" / "summaries"
OUT_CSV = OUT_DIR / "coral_frobenius_per_class.csv"

BALANCE_FIT_TO_KIND_LABEL = {
    "none": "study_only",
    "equal": "study_only_balanced_equal",
}

FEATURES_22 = [
    "innermost_core_size", "rich_club_p75", "rich_club_p90", "rich_club_p95",
    "symmetry_ratio", "std_edge_ixp_cosine_dist", "clustering_avg_local",
    "frac_p2p_edges", "avg_ixp_cosine_dist", "as_path_avg", "as_path_max",
    "as_path_std", "edit_distance_avg", "edit_distance_max",
    "unique_as_path_max", "flaps", "nadas", "vf_rate_delta",
    "ego_filter_ratio", "ego_origin_violations", "unique_peers",
    "origin_changes",
]

DOMAINS = {
    "D1": "rrc04_as12880",
    "D2": "rrc04_as3352",
    "D3": "rrc05_as12880",
    "D4": "rrc05_as3352",
}
DOMAIN_OF = {v: k for k, v in DOMAINS.items()}


def safe_reduction_pct(before: float, after: float) -> float:
    """Match coral_phase2_runner.reduction_pct convention: return 0.0 when
    before is 0 (no reduction possible) or when either operand is NaN."""
    if np.isnan(before) or np.isnan(after):
        return 0.0
    if before == 0.0:
        return 0.0
    return 100.0 * (before - after) / before


def load_feature_matrix(
    csv_path: Path, scaler: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load features + binary_label. If `scaler` given, apply the pair's joint
    median-imputation + (x - mean) / std to move raw values into the same
    scaled space as target_reference.csv / aligned_source.csv.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES_22 if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    if "binary_label" not in df.columns:
        raise ValueError(f"{csv_path} missing binary_label column")
    X = df[FEATURES_22].to_numpy(dtype=float)
    y_raw = df["binary_label"].to_numpy()

    if pd.isna(y_raw).any():
        n_bad = int(pd.isna(y_raw).sum())
        raise ValueError(f"{csv_path} has {n_bad} rows with missing binary_label")
    unique_labels = set(np.unique(y_raw).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{csv_path} binary_label has non-binary values: "
            f"{sorted(unique_labels - {0, 1})}"
        )
    y = y_raw.astype(int)

    if scaler is not None:
        # replicate coral_phase2_runner.prepare_numeric_matrices:
        #   inf -> nan, nan -> fill_values (joint median), then (x-mean)/std
        X = np.where(np.isinf(X), np.nan, X)
        fill = scaler["fill_values"]
        nan_mask = np.isnan(X)
        if nan_mask.any():
            idx_cols = np.where(nan_mask)
            X[idx_cols] = fill[idx_cols[1]]
        X = (X - scaler["feature_mean"]) / scaler["feature_std"]

    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        raise ValueError(
            f"{csv_path} has {n_bad} rows with non-finite feature values "
            f"(NaN or inf) after loading"
            + ("" if scaler is None else " (and scaler applied)")
        )
    return X, y


def cov_frob(X_s: np.ndarray, X_t: np.ndarray) -> float:
    # rowvar=False so rows are observations; ddof=1 for sample covariance
    if X_s.shape[0] < 2 or X_t.shape[0] < 2:
        return float("nan")
    C_s = np.cov(X_s, rowvar=False, ddof=1)
    C_t = np.cov(X_t, rowvar=False, ddof=1)
    return float(np.linalg.norm(C_s - C_t, ord="fro"))


def compute_pair(src: str, tgt: str, coral_root: Path, balance_fit: str) -> dict:
    pair_dir = coral_root / "pairs" / f"{src}__to__{tgt}"
    before_csv = coral_root / "prepared_sources" / f"{src}_binary_source.csv"
    after_csv = pair_dir / "aligned_source.csv"
    target_csv = pair_dir / "target_reference.csv"
    metrics_path = pair_dir / "metrics.json"
    artifacts_path = pair_dir / "artifacts.npz"

    for p in (before_csv, after_csv, target_csv, metrics_path, artifacts_path):
        if not p.exists():
            raise FileNotFoundError(p)

    arts = np.load(artifacts_path, allow_pickle=True)
    feat_names = list(arts["feature_names"])
    if feat_names != FEATURES_22:
        raise ValueError(
            f"{artifacts_path} feature_names order differs from FEATURES_22"
        )
    scaler = {
        "fill_values": np.asarray(arts["fill_values"], dtype=float),
        "feature_mean": np.asarray(arts["feature_mean"], dtype=float),
        "feature_std": np.asarray(arts["feature_std"], dtype=float),
    }

    # before-CORAL source: apply the pair's scaler so we compare cov's in the
    # same standardized space that CORAL aligned to (same space as aligned_source
    # and target_reference CSVs).
    Xs_before, ys_before = load_feature_matrix(before_csv, scaler=scaler)
    Xs_after, ys_after = load_feature_matrix(after_csv)
    Xt, yt = load_feature_matrix(target_csv)

    # Source row count and labels must be identical before/after CORAL
    # (CORAL only shifts/re-colors features; it does not drop or re-label).
    if Xs_before.shape != Xs_after.shape:
        raise ValueError(
            f"shape mismatch before/after for {src}->{tgt}: "
            f"{Xs_before.shape} vs {Xs_after.shape}"
        )
    if not np.array_equal(ys_before, ys_after):
        raise ValueError(f"binary_label mismatch before/after for {src}->{tgt}")

    metrics = json.loads(metrics_path.read_text())

    row = {
        "balance_fit": balance_fit,
        "pair": f"{src}__to__{tgt}",
        "src": src,
        "tgt": tgt,
        "src_domain": DOMAIN_OF.get(src, ""),
        "tgt_domain": DOMAIN_OF.get(tgt, ""),
        "n_src": int(Xs_before.shape[0]),
        "n_src_y0": int((ys_before == 0).sum()),
        "n_src_y1": int((ys_before == 1).sum()),
        "n_tgt": int(Xt.shape[0]),
        "n_tgt_y0": int((yt == 0).sum()),
        "n_tgt_y1": int((yt == 1).sum()),
        "frob_global_before": cov_frob(Xs_before, Xt),
        "frob_global_after": cov_frob(Xs_after, Xt),
        "frob_y0_before": cov_frob(Xs_before[ys_before == 0], Xt[yt == 0]),
        "frob_y0_after": cov_frob(Xs_after[ys_after == 0], Xt[yt == 0]),
        "frob_y1_before": cov_frob(Xs_before[ys_before == 1], Xt[yt == 1]),
        "frob_y1_after": cov_frob(Xs_after[ys_after == 1], Xt[yt == 1]),
        "metrics_json_cov_gap_before": float(metrics["cov_gap_before"]),
        "metrics_json_cov_gap_after": float(metrics["cov_gap_after"]),
    }
    # reduction percentages (0.0 when before==0 or either side is NaN, matching
    # coral_phase2_runner.reduction_pct upstream)
    for key in ("global", "y0", "y1"):
        row[f"frob_{key}_reduction_pct"] = safe_reduction_pct(
            row[f"frob_{key}_before"], row[f"frob_{key}_after"]
        )
    return row


def run_variant(balance_fit: str) -> list[dict]:
    kind_label = BALANCE_FIT_TO_KIND_LABEL[balance_fit]
    coral_root = CORAL_ALIGNED_ROOT / kind_label
    if not coral_root.is_dir():
        raise FileNotFoundError(coral_root)
    print(f"\n=== balance_fit={balance_fit}  root={coral_root} ===", flush=True)
    pair_dirs = sorted((coral_root / "pairs").iterdir())
    rows = []
    for pd_dir in pair_dirs:
        if not pd_dir.is_dir():
            continue
        name = pd_dir.name
        if "__to__" not in name:
            continue
        src, tgt = name.split("__to__", 1)
        print(f"[frob][{balance_fit}] {src} -> {tgt}", flush=True)
        row = compute_pair(src, tgt, coral_root, balance_fit)
        diff_before = abs(row["frob_global_before"] - row["metrics_json_cov_gap_before"])
        diff_after = abs(row["frob_global_after"] - row["metrics_json_cov_gap_after"])
        print(
            f"  global_before={row['frob_global_before']:.6f} "
            f"(metrics.json={row['metrics_json_cov_gap_before']:.6f}  |diff|={diff_before:.2e})",
            flush=True,
        )
        print(
            f"  global_after ={row['frob_global_after']:.6e} "
            f"(metrics.json={row['metrics_json_cov_gap_after']:.6e}  |diff|={diff_after:.2e})",
            flush=True,
        )
        print(
            f"  y=0 before={row['frob_y0_before']:.4f} after={row['frob_y0_after']:.4f} "
            f"red%={row['frob_y0_reduction_pct']:.2f}  (n_src_y0={row['n_src_y0']}, n_tgt_y0={row['n_tgt_y0']})",
            flush=True,
        )
        print(
            f"  y=1 before={row['frob_y1_before']:.4f} after={row['frob_y1_after']:.4f} "
            f"red%={row['frob_y1_reduction_pct']:.2f}  (n_src_y1={row['n_src_y1']}, n_tgt_y1={row['n_tgt_y1']})",
            flush=True,
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balance-fit",
        choices=["none", "equal", "both"],
        default="both",
        help="which CORAL variant to process (default: both)",
    )
    args = parser.parse_args()

    variants = ["none", "equal"] if args.balance_fit == "both" else [args.balance_fit]
    all_rows: list[dict] = []
    for bf in variants:
        all_rows.extend(run_variant(bf))

    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] wrote {OUT_CSV}  rows={len(df)}  variants={variants}")


if __name__ == "__main__":
    main()
