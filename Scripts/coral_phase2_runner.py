#!/usr/bin/env python3
"""
Phase 2 CORAL alignment runner for the selected BGP feature spaces.

This script runs four experiment blocks:
1. Full unlabeled matrix on the shared 41-feature selected space.
2. Likely-normal matrix on the shared 36-feature selected space.
3. Anomaly matrix on the shared 30-feature selected space.
4. Labeled-to-unlabeled matrix on the shared 22-feature space, where each
   source domain is formed by stacking likely-normal and anomaly subsets with
   binary labels and each target domain uses the full unlabeled selected set.

Outputs are written under:
    bgp_unified_results/domain_adaptation_coral/

Each pair directory contains:
    - aligned_source.csv
    - target_reference.csv
    - metrics.json
    - artifacts.npz
    - artifacts_meta.json
    - alignment_pca.png

Usage:
    MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python Scripts/coral_phase2_runner.py
    MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python Scripts/coral_phase2_runner.py --block full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = PROJECT_ROOT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


RNG_SEED = 42
META_COLUMNS = ["window_start", "window_id", "collector"]
OUTPUT_ROOT = PROJECT_ROOT / "bgp_unified_results" / "domain_adaptation_coral"
DEFAULT_EIGEN_FLOOR = 1e-12
CONSTANT_STD_EPS = 1e-12
EXPECTED_FEATURE_COUNTS = {
    "full": 41,
    "likely_normal": 36,
    "anomalies": 30,
    "shared22": 22,
}


@dataclass(frozen=True)
class DomainSpec:
    key: str
    pretty: str
    collector: str
    asn: str
    full_path: Path
    likely_normal_path: Path
    anomaly_path: Path


DOMAINS: dict[str, DomainSpec] = {
    "rrc04_as12880": DomainSpec(
        key="rrc04_as12880",
        pretty="RRC04 / AS12880",
        collector="rrc04",
        asn="AS12880",
        full_path=PROJECT_ROOT / "dataset" / "phase1_raw_features" / "selected_full_rrc04_AS12880_2hop_2025-11-01_2025-11-30_5min.csv",
        likely_normal_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc04_AS12880_2hop_2025-11-01_2025-11-30_5min_likely_normal.csv",
        anomaly_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc04_AS12880_2hop_2025-11-01_2025-11-30_5min_anomalies.csv",
    ),
    "rrc04_as3352": DomainSpec(
        key="rrc04_as3352",
        pretty="RRC04 / AS3352",
        collector="rrc04",
        asn="AS3352",
        full_path=PROJECT_ROOT / "dataset" / "phase1_raw_features" / "selected_full_rrc04_AS3352_2hop_2025-11-01_2025-11-30_5min.csv",
        likely_normal_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc04_AS3352_2hop_2025-11-01_2025-11-30_5min_likely_normal.csv",
        anomaly_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc04_AS3352_2hop_2025-11-01_2025-11-30_5min_anomalies.csv",
    ),
    "rrc05_as12880": DomainSpec(
        key="rrc05_as12880",
        pretty="RRC05 / AS12880",
        collector="rrc05",
        asn="AS12880",
        full_path=PROJECT_ROOT / "dataset" / "phase1_raw_features" / "selected_full_rrc05_AS12880_2hop_2025-11-01_2025-11-30_5min.csv",
        likely_normal_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc05_AS12880_2hop_2025-11-01_2025-11-30_5min_likely_normal.csv",
        anomaly_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc05_AS12880_2hop_2025-11-01_2025-11-30_5min_anomalies.csv",
    ),
    "rrc05_as3352": DomainSpec(
        key="rrc05_as3352",
        pretty="RRC05 / AS3352",
        collector="rrc05",
        asn="AS3352",
        full_path=PROJECT_ROOT / "dataset" / "phase1_raw_features" / "selected_full_rrc05_AS3352_2hop_2025-11-01_2025-11-30_5min.csv",
        likely_normal_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc05_AS3352_2hop_2025-11-01_2025-11-30_5min_likely_normal.csv",
        anomaly_path=PROJECT_ROOT / "dataset" / "phase1_labels_hdbscan" / "selected_full_rrc05_AS3352_2hop_2025-11-01_2025-11-30_5min_anomalies.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 CORAL alignment blocks.")
    parser.add_argument(
        "--block",
        choices=["all", "full", "likely_normal", "anomalies", "shared22"],
        default="all",
        help="Which block to run.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Directory where results should be written.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-6,
        help="Diagonal regularization added to covariance matrices.",
    )
    parser.add_argument(
        "--mmd-samples",
        type=int,
        default=1024,
        help="Maximum rows per domain used for the MMD estimate.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=800,
        help="Maximum rows per domain used in each PCA plot.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PCA projection plots.",
    )
    return parser.parse_args()


def header_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def feature_columns(path: Path) -> list[str]:
    return [col for col in header_columns(path) if col not in META_COLUMNS]


def ordered_intersection(paths: Iterable[Path]) -> list[str]:
    paths = list(paths)
    first_features = feature_columns(paths[0])
    shared = set(first_features)
    for path in paths[1:]:
        shared &= set(feature_columns(path))
    return [feature for feature in first_features if feature in shared]


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_covariance(X: np.ndarray, reg: float = 0.0) -> np.ndarray:
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    if reg:
        cov = cov + reg * np.eye(cov.shape[0], dtype=float)
    return cov


def symmetric_sqrt(
    matrix: np.ndarray,
    inverse: bool = False,
    floor: float = DEFAULT_EIGEN_FLOOR,
    label: str = "matrix",
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    raw_eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    effective_floor = max(float(floor), DEFAULT_EIGEN_FLOOR)
    negative_count = int((raw_eigenvalues < 0.0).sum())
    clip_mask = raw_eigenvalues < effective_floor
    clip_count = int(clip_mask.sum())
    if negative_count or clip_count:
        warnings.warn(
            (
                f"{label}: clipped {clip_count} eigenvalue(s) below "
                f"{effective_floor:.2e}; raw min={raw_eigenvalues.min():.2e}, "
                f"negative count={negative_count}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    eigenvalues = np.clip(raw_eigenvalues, effective_floor, None)
    if inverse:
        scaled = np.diag(1.0 / np.sqrt(eigenvalues))
    else:
        scaled = np.diag(np.sqrt(eigenvalues))
    sqrt_matrix = eigenvectors @ scaled @ eigenvectors.T
    meta = {
        "label": label,
        "eigen_floor": effective_floor,
        "raw_min_eigenvalue": float(raw_eigenvalues.min()),
        "raw_max_eigenvalue": float(raw_eigenvalues.max()),
        "negative_eigenvalue_count": negative_count,
        "clipped_eigenvalue_count": clip_count,
    }
    return sqrt_matrix, meta


def coral_align(source: np.ndarray, target: np.ndarray, reg: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_cov = compute_covariance(source_centered, reg=reg)
    target_cov = compute_covariance(target_centered, reg=reg)

    source_cov_inv_sqrt, source_sqrt_meta = symmetric_sqrt(
        source_cov,
        inverse=True,
        floor=max(reg, DEFAULT_EIGEN_FLOOR),
        label="source_cov",
    )
    target_cov_sqrt, target_sqrt_meta = symmetric_sqrt(
        target_cov,
        inverse=False,
        floor=max(reg, DEFAULT_EIGEN_FLOOR),
        label="target_cov",
    )
    transform = source_cov_inv_sqrt @ target_cov_sqrt
    aligned = source_centered @ transform + target_mean

    artifacts = {
        "source_mean": source_mean.squeeze(0),
        "target_mean": target_mean.squeeze(0),
        "transform": transform,
        "source_cov": source_cov,
        "target_cov": target_cov,
        "source_sqrt_meta": source_sqrt_meta,
        "target_sqrt_meta": target_sqrt_meta,
    }
    return aligned, artifacts


def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(X * X, axis=1)[:, None]
    y_norm = np.sum(Y * Y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * X @ Y.T, 0.0)


def sampled_indices(n_rows: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows <= max_samples:
        return np.arange(n_rows)
    return rng.choice(n_rows, size=max_samples, replace=False)


def median_bandwidth_sq(X: np.ndarray, Y: np.ndarray) -> float:
    sample = np.vstack([X, Y])
    if len(sample) < 2:
        return 1.0
    dists = pairwise_sq_dists(sample, sample)
    upper = dists[np.triu_indices_from(dists, k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_mmd_sq(X: np.ndarray, Y: np.ndarray, bandwidth_sq: float) -> float:
    bandwidth_sq = max(float(bandwidth_sq), DEFAULT_EIGEN_FLOOR)
    gamma = 1.0 / (2.0 * bandwidth_sq)
    K_xx = np.exp(-gamma * pairwise_sq_dists(X, X))
    K_yy = np.exp(-gamma * pairwise_sq_dists(Y, Y))
    K_xy = np.exp(-gamma * pairwise_sq_dists(X, Y))
    return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())


def reduction_pct(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return (before - after) / before * 100.0


def scenario_label(source_spec: DomainSpec, target_spec: DomainSpec) -> str:
    same_as = source_spec.asn == target_spec.asn
    same_collector = source_spec.collector == target_spec.collector
    if same_as and not same_collector:
        return "same_as_diff_collector"
    if same_collector and not same_as:
        return "same_collector_diff_as"
    if not same_as and not same_collector:
        return "diff_as_diff_collector"
    return "same_domain"


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return frame.astype(float)
    except ValueError:
        return frame.apply(pd.to_numeric, errors="coerce")


def prepare_numeric_matrices(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    source_numeric = coerce_numeric_frame(source_df.loc[:, features]).replace([np.inf, -np.inf], np.nan)
    target_numeric = coerce_numeric_frame(target_df.loc[:, features]).replace([np.inf, -np.inf], np.nan)

    source_missing = int(source_numeric.isna().sum().sum())
    target_missing = int(target_numeric.isna().sum().sum())

    combined = pd.concat([source_numeric, target_numeric], axis=0, ignore_index=True)
    fill_values = combined.median(axis=0, numeric_only=True).fillna(0.0)

    source_filled = source_numeric.fillna(fill_values)
    target_filled = target_numeric.fillna(fill_values)

    combined_filled = pd.concat([source_filled, target_filled], axis=0, ignore_index=True)
    feature_mean = combined_filled.mean(axis=0)
    raw_feature_std = combined_filled.std(axis=0, ddof=0)
    constant_mask = raw_feature_std <= CONSTANT_STD_EPS
    constant_features = raw_feature_std.index[constant_mask].tolist()
    if constant_features:
        preview = ", ".join(constant_features[:6])
        suffix = " ..." if len(constant_features) > 6 else ""
        warnings.warn(
            (
                f"Detected {len(constant_features)} constant feature(s) after "
                f"joint imputation/standardization prep: {preview}{suffix}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    feature_std = raw_feature_std.mask(constant_mask, 1.0)

    source_scaled = ((source_filled - feature_mean) / feature_std).to_numpy(dtype=float)
    target_scaled = ((target_filled - feature_mean) / feature_std).to_numpy(dtype=float)

    meta = {
        "fill_values": fill_values.to_numpy(dtype=float),
        "feature_mean": feature_mean.to_numpy(dtype=float),
        "feature_std": feature_std.to_numpy(dtype=float),
        "source_missing": source_missing,
        "target_missing": target_missing,
        "scaling_mode": "joint_transductive",
        "constant_feature_count": len(constant_features),
        "constant_features": constant_features,
    }
    return source_scaled, target_scaled, meta


def write_projection_plot(
    source_before: np.ndarray,
    source_after: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    rng: np.random.Generator,
    max_samples: int,
) -> None:
    source_idx = sampled_indices(len(source_before), max_samples, rng)
    target_idx = sampled_indices(len(target), max_samples, rng)
    sb = source_before[source_idx]
    sa = source_after[source_idx]
    tg = target[target_idx]
    stacked = np.vstack([sb, sa, tg])
    pca = PCA(n_components=2, random_state=RNG_SEED)
    coords = pca.fit_transform(stacked)

    n_sb = len(sb)
    n_sa = len(sa)
    coords_sb = coords[:n_sb]
    coords_sa = coords[n_sb : n_sb + n_sa]
    coords_tg = coords[n_sb + n_sa :]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    panels = [
        (axes[0], coords_sb, "Source Before CORAL"),
        (axes[1], coords_sa, "Source After CORAL"),
    ]
    for axis, source_coords, title in panels:
        axis.scatter(
            coords_tg[:, 0],
            coords_tg[:, 1],
            s=10,
            alpha=0.35,
            label="Target",
            color="#2f6690",
        )
        axis.scatter(
            source_coords[:, 0],
            source_coords[:, 1],
            s=10,
            alpha=0.35,
            label=title.replace("Source ", ""),
            color="#d1495b",
        )
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("PC2")
    axes[1].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_output_dataframe(
    base_df: pd.DataFrame,
    features: list[str],
    transformed: np.ndarray,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    output = base_df.loc[:, META_COLUMNS].copy()
    if extra_columns:
        for column in extra_columns:
            output[column] = base_df[column].values
    for index, feature in enumerate(features):
        output[feature] = transformed[:, index]
    return output


def pair_output_name(source_key: str, target_key: str) -> str:
    return f"{source_key}__to__{target_key}"


def run_pair(
    block_name: str,
    source_name: str,
    target_name: str,
    source_spec: DomainSpec,
    target_spec: DomainSpec,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    reg: float,
    mmd_samples: int,
    plot_samples: int,
    skip_plots: bool,
) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_material = f"{block_name}|{source_name}|{target_name}|{RNG_SEED}".encode("utf-8")
    pair_seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16) % (2**32)
    pair_rng = np.random.default_rng(pair_seed)

    source_scaled, target_scaled, prep_meta = prepare_numeric_matrices(source_df, target_df, features)
    source_aligned, coral_meta = coral_align(source_scaled, target_scaled, reg=reg)

    cov_before = float(np.linalg.norm(compute_covariance(source_scaled) - compute_covariance(target_scaled), ord="fro"))
    cov_after = float(np.linalg.norm(compute_covariance(source_aligned) - compute_covariance(target_scaled), ord="fro"))
    source_mmd_idx = sampled_indices(len(source_scaled), mmd_samples, pair_rng)
    target_mmd_idx = sampled_indices(len(target_scaled), mmd_samples, pair_rng)
    source_mmd_sample = source_scaled[source_mmd_idx]
    source_aligned_mmd_sample = source_aligned[source_mmd_idx]
    target_mmd_sample = target_scaled[target_mmd_idx]
    mmd_bandwidth_sq = median_bandwidth_sq(source_mmd_sample, target_mmd_sample)
    mmd_before = rbf_mmd_sq(source_mmd_sample, target_mmd_sample, bandwidth_sq=mmd_bandwidth_sq)
    mmd_after = rbf_mmd_sq(source_aligned_mmd_sample, target_mmd_sample, bandwidth_sq=mmd_bandwidth_sq)

    extra_source_columns = [column for column in source_df.columns if column not in META_COLUMNS and column not in features]
    extra_target_columns = [column for column in target_df.columns if column not in META_COLUMNS and column not in features]

    aligned_source_df = build_output_dataframe(source_df, features, source_aligned, extra_columns=extra_source_columns)
    target_reference_df = build_output_dataframe(target_df, features, target_scaled, extra_columns=extra_target_columns)

    aligned_source_df.to_csv(output_dir / "aligned_source.csv", index=False)
    target_reference_df.to_csv(output_dir / "target_reference.csv", index=False)

    np.savez(
        output_dir / "artifacts.npz",
        feature_names=np.array(features, dtype=object),
        fill_values=np.asarray(prep_meta["fill_values"], dtype=float),
        feature_mean=np.asarray(prep_meta["feature_mean"], dtype=float),
        feature_std=np.asarray(prep_meta["feature_std"], dtype=float),
        source_mean=np.asarray(coral_meta["source_mean"], dtype=float),
        target_mean=np.asarray(coral_meta["target_mean"], dtype=float),
        transform=np.asarray(coral_meta["transform"], dtype=float),
        source_cov=np.asarray(coral_meta["source_cov"], dtype=float),
        target_cov=np.asarray(coral_meta["target_cov"], dtype=float),
    )

    artifacts_meta = {
        "block": block_name,
        "source": source_name,
        "target": target_name,
        "n_features": len(features),
        "features": features,
        "source_missing_values": int(prep_meta["source_missing"]),
        "target_missing_values": int(prep_meta["target_missing"]),
        "scaling_mode": prep_meta["scaling_mode"],
        "constant_feature_count": int(prep_meta["constant_feature_count"]),
        "constant_features": prep_meta["constant_features"],
        "source_extra_columns": extra_source_columns,
        "target_extra_columns": extra_target_columns,
        "source_sqrt_meta": coral_meta["source_sqrt_meta"],
        "target_sqrt_meta": coral_meta["target_sqrt_meta"],
    }
    with open(output_dir / "artifacts_meta.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts_meta, handle, indent=2)

    metrics = {
        "block": block_name,
        "source": source_name,
        "target": target_name,
        "source_pretty": source_spec.pretty,
        "target_pretty": target_spec.pretty,
        "scenario": scenario_label(source_spec, target_spec),
        "n_source": int(len(source_df)),
        "n_target": int(len(target_df)),
        "n_features": int(len(features)),
        "cov_gap_before": cov_before,
        "cov_gap_after": cov_after,
        "cov_gap_reduction_pct": reduction_pct(cov_before, cov_after),
        "mmd_before": mmd_before,
        "mmd_after": mmd_after,
        "mmd_reduction_pct": reduction_pct(mmd_before, mmd_after),
        "mmd_bandwidth_sq": mmd_bandwidth_sq,
        "mmd_source_sample_size": int(len(source_mmd_sample)),
        "mmd_target_sample_size": int(len(target_mmd_sample)),
        "source_missing_values": int(prep_meta["source_missing"]),
        "target_missing_values": int(prep_meta["target_missing"]),
        "constant_feature_count": int(prep_meta["constant_feature_count"]),
        "source_cov_eigen_clipped": int(coral_meta["source_sqrt_meta"]["clipped_eigenvalue_count"]),
        "target_cov_eigen_clipped": int(coral_meta["target_sqrt_meta"]["clipped_eigenvalue_count"]),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if not skip_plots:
        write_projection_plot(
            source_before=source_scaled,
            source_after=source_aligned,
            target=target_scaled,
            output_path=output_dir / "alignment_pca.png",
            rng=pair_rng,
            max_samples=plot_samples,
        )

    return metrics


def block_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby("scenario", dropna=False)[
            [
                "cov_gap_before",
                "cov_gap_after",
                "cov_gap_reduction_pct",
                "mmd_before",
                "mmd_after",
                "mmd_reduction_pct",
            ]
        ]
        .mean()
        .reset_index()
    )
    return summary


def write_block_manifest(
    output_dir: Path,
    block_name: str,
    description: str,
    features: list[str],
    extra: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "block": block_name,
        "description": description,
        "n_features": len(features),
        "features": features,
        "scaling_mode": "joint_transductive",
    }
    if extra:
        manifest.update(extra)
    with open(output_dir / "feature_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_matrix_block(
    block_name: str,
    description: str,
    source_frames: dict[str, pd.DataFrame],
    features: list[str],
    output_root: Path,
    reg: float,
    mmd_samples: int,
    plot_samples: int,
    skip_plots: bool,
    source_role: str,
    target_role: str,
    target_frames: dict[str, pd.DataFrame] | None = None,
    extra_manifest: dict[str, object] | None = None,
    extra_record_fields: dict[str, dict[str, int | str]] | None = None,
) -> pd.DataFrame:
    block_dir = output_root / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    write_block_manifest(block_dir, block_name, description, features, extra=extra_manifest)

    if target_frames is None:
        target_frames = source_frames
    extra_record_fields = extra_record_fields or {}

    records: list[dict[str, float | int | str]] = []
    source_keys = list(source_frames.keys())
    target_keys = list(target_frames.keys())
    total_pairs = sum(1 for source_key in source_keys for target_key in target_keys if source_key != target_key)
    print(f"\nRunning {block_name} with {len(features)} features across {total_pairs} source-target pairs.")
    pair_counter = 0
    for source_key in source_keys:
        for target_key in target_keys:
            if source_key == target_key:
                continue
            pair_counter += 1
            print(f"  [{pair_counter}/{total_pairs}] {source_key} -> {target_key}")
            pair_dir = block_dir / "pairs" / pair_output_name(source_key, target_key)
            record = run_pair(
                block_name=block_name,
                source_name=source_key,
                target_name=target_key,
                source_spec=DOMAINS[source_key],
                target_spec=DOMAINS[target_key],
                source_df=source_frames[source_key],
                target_df=target_frames[target_key],
                features=features,
                output_dir=pair_dir,
                reg=reg,
                mmd_samples=mmd_samples,
                plot_samples=plot_samples,
                skip_plots=skip_plots,
            )
            record["source_role"] = source_role
            record["target_role"] = target_role
            record.update(extra_record_fields.get(source_key, {}))
            records.append(record)

    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.sort_values(["scenario", "source", "target"]).to_csv(block_dir / "pairwise_metrics.csv", index=False)
    block_summary(metrics_df).to_csv(block_dir / "scenario_summary.csv", index=False)
    return metrics_df


def build_binary_sources(shared_features: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    sources: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    for key, spec in DOMAINS.items():
        likely_normal_df = load_frame(spec.likely_normal_path)
        anomaly_df = load_frame(spec.anomaly_path)

        likely_normal_df = likely_normal_df.loc[:, META_COLUMNS + shared_features].copy()
        anomaly_df = anomaly_df.loc[:, META_COLUMNS + shared_features].copy()

        likely_normal_df["binary_label"] = 0
        likely_normal_df["binary_label_name"] = "likely_normal"
        anomaly_df["binary_label"] = 1
        anomaly_df["binary_label_name"] = "anomaly"

        combined = pd.concat([likely_normal_df, anomaly_df], axis=0, ignore_index=True)
        sources[key] = combined
        counts[key] = {
            "likely_normal": int(len(likely_normal_df)),
            "anomaly": int(len(anomaly_df)),
            "total": int(len(combined)),
        }
    return sources, counts


def write_binary_sources(output_root: Path, sources: dict[str, pd.DataFrame]) -> None:
    source_dir = output_root / "shared22_labeled_to_unlabeled" / "prepared_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for key, df in sources.items():
        df.to_csv(source_dir / f"{key}_binary_source.csv", index=False)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    full_paths = [spec.full_path for spec in DOMAINS.values()]
    likely_normal_paths = [spec.likely_normal_path for spec in DOMAINS.values()]
    anomaly_paths = [spec.anomaly_path for spec in DOMAINS.values()]

    full_features = ordered_intersection(full_paths)
    likely_normal_features = ordered_intersection(likely_normal_paths)
    anomaly_features = ordered_intersection(anomaly_paths)
    shared22_features = ordered_intersection(full_paths + likely_normal_paths + anomaly_paths)

    for name, features in {
        "full": full_features,
        "likely_normal": likely_normal_features,
        "anomalies": anomaly_features,
        "shared22": shared22_features,
    }.items():
        expected = EXPECTED_FEATURE_COUNTS[name]
        if len(features) != expected:
            raise ValueError(
                f"Expected {expected} features for '{name}', but found {len(features)}. "
                "The selected CSV schemas may have changed."
            )

    feature_registry = {
        "full": full_features,
        "likely_normal": likely_normal_features,
        "anomalies": anomaly_features,
        "shared22": shared22_features,
    }
    with open(output_root / "feature_registry.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                block: {
                    "n_features": len(features),
                    "features": features,
                }
                for block, features in feature_registry.items()
            },
            handle,
            indent=2,
        )

    blocks_to_run = [args.block] if args.block != "all" else ["full", "likely_normal", "anomalies", "shared22"]

    all_metrics: list[pd.DataFrame] = []

    if "full" in blocks_to_run:
        frames = {key: load_frame(spec.full_path) for key, spec in DOMAINS.items()}
        all_metrics.append(
            run_matrix_block(
                block_name="full_41",
                description="Pairwise CORAL matrix on the full unlabeled selected space.",
                source_frames=frames,
                features=full_features,
                output_root=output_root,
                reg=args.reg,
                mmd_samples=args.mmd_samples,
                plot_samples=args.plot_samples,
                skip_plots=args.skip_plots,
                source_role="full_unlabeled",
                target_role="full_unlabeled",
                extra_manifest={"expected_feature_count": 41},
            )
        )

    if "likely_normal" in blocks_to_run:
        frames = {key: load_frame(spec.likely_normal_path) for key, spec in DOMAINS.items()}
        all_metrics.append(
            run_matrix_block(
                block_name="likely_normal_36",
                description="Pairwise CORAL matrix on the filtered likely-normal selected space.",
                source_frames=frames,
                features=likely_normal_features,
                output_root=output_root,
                reg=args.reg,
                mmd_samples=args.mmd_samples,
                plot_samples=args.plot_samples,
                skip_plots=args.skip_plots,
                source_role="likely_normal",
                target_role="likely_normal",
                extra_manifest={"expected_feature_count": 36},
            )
        )

    if "anomalies" in blocks_to_run:
        frames = {key: load_frame(spec.anomaly_path) for key, spec in DOMAINS.items()}
        all_metrics.append(
            run_matrix_block(
                block_name="anomalies_30",
                description="Pairwise CORAL matrix on the anomaly selected space.",
                source_frames=frames,
                features=anomaly_features,
                output_root=output_root,
                reg=args.reg,
                mmd_samples=args.mmd_samples,
                plot_samples=args.plot_samples,
                skip_plots=args.skip_plots,
                source_role="anomaly",
                target_role="anomaly",
                extra_manifest={"expected_feature_count": 30},
            )
        )

    if "shared22" in blocks_to_run:
        binary_sources, class_counts = build_binary_sources(shared22_features)
        write_binary_sources(output_root, binary_sources)
        target_frames = {key: load_frame(spec.full_path).loc[:, META_COLUMNS + shared22_features].copy() for key, spec in DOMAINS.items()}
        metrics_df = run_matrix_block(
            block_name="shared22_labeled_to_unlabeled",
            description="CORAL matrix from binary labeled sources to full unlabeled targets on the shared 22-feature space.",
            source_frames=binary_sources,
            target_frames=target_frames,
            features=shared22_features,
            output_root=output_root,
            reg=args.reg,
            mmd_samples=args.mmd_samples,
            plot_samples=args.plot_samples,
            skip_plots=args.skip_plots,
            source_role="binary_labeled_source",
            target_role="full_unlabeled_target",
            extra_manifest={"class_counts": class_counts, "expected_feature_count": 22},
            extra_record_fields={
                key: {
                    "source_likely_normal_count": class_counts[key]["likely_normal"],
                    "source_anomaly_count": class_counts[key]["anomaly"],
                }
                for key in binary_sources
            },
        )
        all_metrics.append(metrics_df)

    if all_metrics:
        combined = pd.concat(all_metrics, axis=0, ignore_index=True)
        combined.to_csv(output_root / "all_blocks_metrics.csv", index=False)

        block_overview = (
            combined.groupby("block")[
                [
                    "n_features",
                    "cov_gap_before",
                    "cov_gap_after",
                    "cov_gap_reduction_pct",
                    "mmd_before",
                    "mmd_after",
                    "mmd_reduction_pct",
                ]
            ]
            .mean()
            .reset_index()
        )
        block_overview.to_csv(output_root / "block_overview.csv", index=False)

        print("Completed CORAL blocks:")
        for _, row in block_overview.iterrows():
            print(
                f"  {row['block']}: "
                f"features={int(row['n_features'])}, "
                f"mean cov reduction={row['cov_gap_reduction_pct']:.2f}%, "
                f"mean MMD reduction={row['mmd_reduction_pct']:.2f}%"
            )

        print("\nFeature counts:")
        print(f"  full: {len(full_features)}")
        print(f"  likely_normal: {len(likely_normal_features)}")
        print(f"  anomalies: {len(anomaly_features)}")
        print(f"  shared22: {len(shared22_features)}")


if __name__ == "__main__":
    main()
