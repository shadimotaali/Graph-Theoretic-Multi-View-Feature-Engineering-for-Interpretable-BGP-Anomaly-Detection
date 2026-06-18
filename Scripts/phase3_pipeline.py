#!/usr/bin/env python3
"""
Phase 3 unified pipeline.

Single entry point for the entire Phase 3 multi-view fusion experiment:

    python Scripts/phase3_pipeline.py manifest
    python Scripts/phase3_pipeline.py coral [--balance-fit equal]
    python Scripts/phase3_pipeline.py experiment [--coral-kind study_only]
    python Scripts/phase3_pipeline.py analyze [--input-dir ...]
    python Scripts/phase3_pipeline.py diagnostics
    python Scripts/phase3_pipeline.py all        # manifest → coral → experiment → analyze

Subcommands
-----------
manifest    Build the v2 event manifest (gap-based study events + historical).
coral       Re-fit CORAL on unified shared22 source/target.
experiment  Run XGBoost fusion experiment (3 views × 2 alignments × 5 seeds).
analyze     Per-event metrics, detection rates, paired Wilcoxon tests.
diagnostics D1→D2 same-collector test + pseudo-label conditional CORAL.
all         Run manifest → coral → experiment → analyze sequentially.

Library dependency: ``phase3_view_partition.py`` (kept separate — it defines
shared constants imported here).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))
from coral_phase2_runner import (  # noqa: E402
    META_COLUMNS,
    build_output_dataframe,
    compute_covariance,
    coral_align,
    median_bandwidth_sq,
    prepare_numeric_matrices,
    rbf_mmd_sq,
    reduction_pct,
    sampled_indices,
    write_projection_plot,
)
from phase3_view_partition import (  # noqa: E402
    CORE10_ALL,
    SHARED22_ALL,
    SHARED22_GRAPH_VIEW,
    SHARED22_STAT_VIEW,
    validate_core10_partition,
    validate_partition,
)

# Stage 3 (deep models + attention fusion). Torch is optional; if not installed,
# only classifiers that don't need it (xgboost, random_forest) remain available.
try:
    from phase3_deep_models import (  # noqa: E402
        AttentionFusionWeights,
        LSTMParams,
        LSTMPipeline,
        MLPParams,
        TORCH_AVAILABLE,
        fit_attention_fusion,
        fit_lstm,
        fit_mlp,
    )
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    fit_mlp = fit_lstm = fit_attention_fusion = None  # type: ignore
    LSTMPipeline = MLPParams = LSTMParams = AttentionFusionWeights = None  # type: ignore

# ============================================================================
# Shared paths and constants
# ============================================================================
MANIFEST_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "phase3_labeled"
MANIFEST_CSV = MANIFEST_OUTPUT_DIR / "event_manifest_v2.csv"
MANIFEST_JSON = MANIFEST_OUTPUT_DIR / "event_manifest_v2_summary.json"
CORAL_ALIGNED_ROOT = PROJECT_ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned"
FUSION_OUTPUT_ROOT = PROJECT_ROOT / "bgp_unified_results" / "phase3_fusion"
DIAGNOSTICS_DIR = FUSION_OUTPUT_ROOT / "coral_diagnostics"

DOMAINS = ["rrc04_as12880", "rrc04_as3352", "rrc05_as12880", "rrc05_as3352"]
SOURCE_DOMAIN = "rrc04_as3352"
TARGET_DOMAIN = "rrc05_as3352"

ANOMALY_LABELS = {"likely_anomaly", "high_confidence_anomaly"}
NORMAL_LABELS = {"likely_normal"}

# Feature-set registry — §5.3 shared22-vs-core10 ablation. The active set is
# selected via --feature-set on every subcommand and applied by
# set_active_feature_set() before any pipeline work runs. Shared22 is the
# default; core10 collapses the view dimension to a single flat view (no
# graph/stat sub-views, therefore no late-fusion or attention-fusion views).
_FEATURE_SETS: dict[str, dict] = {
    "shared22": {
        "unified_dir": PROJECT_ROOT / "dataset" / "phase3_training" / "shared22",
        "view_configs": {
            "graph": list(SHARED22_GRAPH_VIEW),
            "stat": list(SHARED22_STAT_VIEW),
            "fusion_early": list(SHARED22_ALL),
        },
        "base_views": ("graph", "stat", "fusion_early"),
        "late_views": ("fusion_late_mean", "fusion_late_or"),
        "attention_view": "fusion_late_attention",
        "all_features": list(SHARED22_ALL),
        "validator": validate_partition,
    },
    "core10": {
        "unified_dir": PROJECT_ROOT / "dataset" / "phase3_training" / "core10",
        "view_configs": {
            "core10": list(CORE10_ALL),
        },
        "base_views": ("core10",),
        "late_views": (),
        "attention_view": None,
        "all_features": list(CORE10_ALL),
        "validator": validate_core10_partition,
    },
}

FEATURE_SET: str = "shared22"
UNIFIED_DIR: Path = _FEATURE_SETS["shared22"]["unified_dir"]
VIEW_CONFIGS: dict[str, list[str]] = _FEATURE_SETS["shared22"]["view_configs"]
BASE_VIEWS: tuple[str, ...] = _FEATURE_SETS["shared22"]["base_views"]
LATE_VIEWS: tuple[str, ...] = _FEATURE_SETS["shared22"]["late_views"]
ATTENTION_VIEW: str | None = _FEATURE_SETS["shared22"]["attention_view"]
ALL_FEATURES: list[str] = _FEATURE_SETS["shared22"]["all_features"]
_VALIDATE_PARTITION = _FEATURE_SETS["shared22"]["validator"]
ALL_VIEWS = BASE_VIEWS + LATE_VIEWS + ((ATTENTION_VIEW,) if ATTENTION_VIEW else ())


def set_active_feature_set(name: str) -> None:
    """Swap the active feature set. Must be called before any pipeline work."""
    global FEATURE_SET, UNIFIED_DIR, VIEW_CONFIGS, BASE_VIEWS, LATE_VIEWS
    global ATTENTION_VIEW, ALL_FEATURES, ALL_VIEWS, _VALIDATE_PARTITION
    if name not in _FEATURE_SETS:
        raise ValueError(
            f"unknown feature set: {name!r} (choices: {sorted(_FEATURE_SETS)})"
        )
    spec = _FEATURE_SETS[name]
    FEATURE_SET = name
    UNIFIED_DIR = spec["unified_dir"]
    VIEW_CONFIGS = spec["view_configs"]
    BASE_VIEWS = spec["base_views"]
    LATE_VIEWS = spec["late_views"]
    ATTENTION_VIEW = spec["attention_view"]
    ALL_FEATURES = spec["all_features"]
    _VALIDATE_PARTITION = spec["validator"]
    ALL_VIEWS = BASE_VIEWS + LATE_VIEWS + ((ATTENTION_VIEW,) if ATTENTION_VIEW else ())
# Flat classifiers consume (X, y) arrays; sequence classifiers consume the full DataFrame.
FLAT_CLASSIFIERS = ("xgboost", "random_forest", "mlp")
SEQ_CLASSIFIERS = ("lstm",)
CLASSIFIERS = FLAT_CLASSIFIERS + SEQ_CLASSIFIERS
ALIGNMENTS = ("before_coral", "after_coral")
SEEDS = (0, 1, 2, 3, 4)
RNG_SEED = 42
DEFAULT_MAX_GAP_MINUTES = 120

AGG_METRICS = (
    "macro_f1",
    "recall_anomaly",
    "precision_anomaly",
    "pr_auc_anomaly",
    "balanced_accuracy",
)


# ============================================================================
# Shared helpers
# ============================================================================
def load_unified(domain: str) -> pd.DataFrame:
    path = UNIFIED_DIR / f"{domain}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"unified {FEATURE_SET} file not found: {path}"
        )
    df = pd.read_csv(path)
    _VALIDATE_PARTITION(df.columns)
    return df


def load_study(domain: str) -> pd.DataFrame:
    df = load_unified(domain)
    return df[df["provenance"] == "study"].reset_index(drop=True)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg) / float(n_pos) if n_pos > 0 else 1.0


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, seed: int,
                **overrides) -> XGBClassifier:
    spw = compute_scale_pos_weight(y_train)
    defaults = dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, tree_method="hist",
        random_state=seed, n_jobs=4, verbosity=0,
    )
    defaults.update(overrides)
    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray, seed: int,
                      **overrides) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=seed, n_jobs=4,
    )
    defaults.update(overrides)
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


CLASSIFIER_FITTERS = {
    "xgboost": fit_xgboost,
    "random_forest": fit_random_forest,
}
# MLP is registered at import time iff torch is available.
if TORCH_AVAILABLE and fit_mlp is not None:
    CLASSIFIER_FITTERS["mlp"] = fit_mlp

# Tuning grids (source-only CV). Deliberately small to keep Stage 3 tractable.
TUNING_GRIDS = {
    "xgboost": [
        {"max_depth": 4, "learning_rate": 0.05},
        {"max_depth": 6, "learning_rate": 0.05},
        {"max_depth": 6, "learning_rate": 0.1},
        {"max_depth": 8, "learning_rate": 0.05},
    ],
    "random_forest": [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 400, "max_depth": None},
        {"n_estimators": 400, "max_depth": 10},
    ],
    "mlp": [
        {"hidden": (64, 32), "dropout": 0.2, "lr": 1e-3},
        {"hidden": (128, 64), "dropout": 0.2, "lr": 1e-3},
        {"hidden": (128, 64), "dropout": 0.3, "lr": 5e-4},
    ],
    "lstm": [
        {"seq_len": 5, "hidden": 32, "dropout": 0.2},
        {"seq_len": 10, "hidden": 64, "dropout": 0.2},
        {"seq_len": 20, "hidden": 64, "dropout": 0.3},
    ],
}


def derive_kind_label(source_kind: str, balance_fit: str) -> str:
    """Canonical kind_label used by both CORAL output paths and experiment lookup.

    Single source of truth: if this formula ever changes, both writer (cmd_coral)
    and reader (cmd_experiment via cmd_all) stay in sync automatically.

    When a non-default feature set is active (e.g. core10), the label is
    suffixed with ``__<feature_set>`` so CORAL outputs do not collide with the
    shared22 artifacts at the same source_kind/balance_fit.
    """
    base = source_kind if balance_fit == "none" else f"{source_kind}_balanced_{balance_fit}"
    if FEATURE_SET == "shared22":
        return base
    return f"{base}__{FEATURE_SET}"


def balanced_fit_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    anom_idx = np.where(labels == 1)[0]
    norm_idx = np.where(labels == 0)[0]
    k = int(min(len(anom_idx), len(norm_idx)))
    if k == 0:
        raise ValueError("balanced_fit_indices: one of the classes is empty")
    if len(anom_idx) > k:
        anom_idx = rng.choice(anom_idx, size=k, replace=False)
    if len(norm_idx) > k:
        norm_idx = rng.choice(norm_idx, size=k, replace=False)
    combined = np.concatenate([anom_idx, norm_idx])
    combined.sort()
    return combined


# ============================================================================
# 1. MANIFEST
# ============================================================================
def _load_domain_ts(domain: str) -> pd.DataFrame:
    path = UNIFIED_DIR / f"{domain}.csv"
    df = pd.read_csv(path)
    df["window_start"] = pd.to_datetime(df["window_start"], format="ISO8601", utc=True)
    df = df.sort_values("window_start", kind="mergesort").reset_index(drop=True)
    return df


def _build_study_events(df_study: pd.DataFrame, domain: str, max_gap_minutes: int) -> list[dict]:
    anomaly = df_study[df_study["discovered_label"].isin(ANOMALY_LABELS)].copy()
    if anomaly.empty:
        return []
    anomaly = anomaly.sort_values("window_start", kind="mergesort").reset_index(drop=True)
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    new_event = anomaly["window_start"].diff().isna() | (anomaly["window_start"].diff() > max_gap)
    run_id = new_event.cumsum()
    events: list[dict] = []
    for run, grp in anomaly.groupby(run_id, sort=True):
        dominant = grp["discovered_label"].value_counts().idxmax()
        events.append({
            "domain": domain, "event_id": f"{domain}_evt_{int(run):03d}",
            "provenance": "study", "incident_type": "",
            "start_window": grp["window_id"].iloc[0],
            "end_window": grp["window_id"].iloc[-1],
            "start_time": grp["window_start"].iloc[0].isoformat(),
            "end_time": grp["window_start"].iloc[-1].isoformat(),
            "n_windows": int(len(grp)), "n_anomaly_windows": int(len(grp)),
            "n_normal_windows": 0, "dominant_label": dominant,
        })
    return events


def _build_historical_events(df_hist: pd.DataFrame, domain: str) -> list[dict]:
    if df_hist.empty:
        return []
    events: list[dict] = []
    for event_id, grp in df_hist.groupby("event_id", sort=True):
        grp = grp.sort_values("window_start", kind="mergesort")
        dominant = grp["discovered_label"].value_counts().idxmax()
        inc_series = grp["incident_type"].dropna()
        inc_type = str(inc_series.iloc[0]) if not inc_series.empty else ""
        events.append({
            "domain": domain, "event_id": str(event_id),
            "provenance": "historical", "incident_type": inc_type,
            "start_window": grp["window_id"].iloc[0],
            "end_window": grp["window_id"].iloc[-1],
            "start_time": grp["window_start"].iloc[0].isoformat(),
            "end_time": grp["window_start"].iloc[-1].isoformat(),
            "n_windows": int(len(grp)),
            "n_anomaly_windows": int((grp["binary_label"] == 1).sum()),
            "n_normal_windows": int((grp["binary_label"] == 0).sum()),
            "dominant_label": dominant,
        })
    return events


def build_manifest(domains: Iterable[str], max_gap_minutes: int) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for domain in domains:
        df = _load_domain_ts(domain)
        df_study = df[df["provenance"] == "study"].copy()
        df_hist = df[df["provenance"] == "historical"].copy()
        study_events = _build_study_events(df_study, domain, max_gap_minutes)
        hist_events = _build_historical_events(df_hist, domain)
        rows.extend(study_events)
        rows.extend(hist_events)
        n_sa = int((df_study["binary_label"] == 1).sum())
        n_sn = int((df_study["binary_label"] == 0).sum())
        summary[domain] = {
            "study_normal_windows": n_sn, "study_anomaly_windows": n_sa,
            "historical_anomaly_windows": int((df_hist["binary_label"] == 1).sum()),
            "n_study_events": len(study_events),
            "n_historical_events": len(hist_events),
            "avg_windows_per_study_event": round(n_sa / len(study_events), 2) if study_events else 0.0,
            "study_event_size_min": min((e["n_windows"] for e in study_events), default=0),
            "study_event_size_max": max((e["n_windows"] for e in study_events), default=0),
        }
    manifest = pd.DataFrame(rows).sort_values(
        ["domain", "provenance", "start_time"], kind="mergesort"
    ).reset_index(drop=True)
    return manifest, summary


def cmd_manifest(args: argparse.Namespace) -> None:
    MANIFEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, summary = build_manifest(DOMAINS, args.max_gap_minutes)
    manifest.to_csv(MANIFEST_CSV, index=False)
    payload = {
        "universal_labelling_rule": {
            "normal": sorted(NORMAL_LABELS), "anomaly": sorted(ANOMALY_LABELS),
            "dropped": ["uncertain"],
        },
        "event_definition": (
            f"Study events = runs of anomaly windows where consecutive windows "
            f"are at most {args.max_gap_minutes} minutes apart."
        ),
        "max_gap_minutes": args.max_gap_minutes,
        "per_domain": summary,
        "total_events": int(len(manifest)),
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {MANIFEST_CSV}  ({len(manifest)} events, max_gap={args.max_gap_minutes} min)")
    print(f"Wrote {MANIFEST_JSON}")
    for domain, info in summary.items():
        print(f"  {domain}: {info['n_study_events']} study + {info['n_historical_events']} historical events")


# ============================================================================
# 2. CORAL
# ============================================================================
def cmd_coral(args: argparse.Namespace) -> None:
    kind_label = derive_kind_label(args.source_kind, args.balance_fit)
    kind_root = CORAL_ALIGNED_ROOT / kind_label
    prepared_dir = kind_root / "prepared_sources"
    pairs_dir = kind_root / "pairs" / f"{args.source_domain}__to__{args.target_domain}"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    features = list(ALL_FEATURES)
    print("=" * 72)
    print(f"CORAL re-runner (feature_set={FEATURE_SET}, d={len(features)})")
    print("=" * 72)
    print(f"source_kind:  {args.source_kind}  |  balance_fit: {args.balance_fit}")
    print(f"source:       {args.source_domain}  →  target: {args.target_domain}")
    print(f"output:       {kind_root}")
    print()

    raw_source = load_unified(args.source_domain)
    raw_target = load_unified(args.target_domain)

    if args.source_kind == "study_only":
        source_df = raw_source[raw_source["provenance"] == "study"].reset_index(drop=True)
    elif args.source_kind == "study_plus_historical":
        source_df = raw_source[raw_source["provenance"].isin(("study", "historical"))].reset_index(drop=True)
    else:
        raise ValueError(f"unknown source_kind: {args.source_kind}")
    target_df = raw_target[raw_target["provenance"] == "study"].reset_index(drop=True)

    print(f"source: {len(source_df)} rows ({(source_df['binary_label']==1).sum()} anom, "
          f"{(source_df['binary_label']==0).sum()} norm)")
    print(f"target: {len(target_df)} rows ({(target_df['binary_label']==1).sum()} anom, "
          f"{(target_df['binary_label']==0).sum()} norm)")

    rng = np.random.default_rng(RNG_SEED)
    src_scaled, tgt_scaled, prep_meta = prepare_numeric_matrices(source_df, target_df, features)

    if args.balance_fit == "equal":
        # balance_fit=equal: subsample normals once on the source so the SAME
        # balanced set flows into (a) the CORAL covariance fit and (b) the
        # saved source CSVs that downstream classifier training reads. Target
        # is only balanced for the CORAL covariance estimate; the full target
        # is kept as the evaluation reference.
        src_bal = balanced_fit_indices(source_df["binary_label"].to_numpy(), rng)
        tgt_bal = balanced_fit_indices(target_df["binary_label"].to_numpy(), rng)
        print(f"balanced fit: source {len(src_bal)} rows (applied to CORAL + classifier), "
              f"target {len(tgt_bal)} rows (CORAL only)")
        source_df = source_df.iloc[src_bal].reset_index(drop=True)
        src_scaled = src_scaled[src_bal]
        _, coral_meta = coral_align(src_scaled, tgt_scaled[tgt_bal], reg=args.reg)
        source_aligned = (src_scaled - coral_meta["source_mean"]) @ coral_meta["transform"] + coral_meta["target_mean"]
    else:
        source_aligned, coral_meta = coral_align(src_scaled, tgt_scaled, reg=args.reg)

    # Save raw source (for before_coral training) — after any balancing above
    source_df.to_csv(prepared_dir / f"{args.source_domain}_binary_source.csv", index=False)

    # Metrics on full data
    cov_before = float(np.linalg.norm(compute_covariance(src_scaled) - compute_covariance(tgt_scaled), ord="fro"))
    cov_after = float(np.linalg.norm(compute_covariance(source_aligned) - compute_covariance(tgt_scaled), ord="fro"))
    si = sampled_indices(len(src_scaled), 1024, rng)
    ti = sampled_indices(len(tgt_scaled), 1024, rng)
    bw = median_bandwidth_sq(src_scaled[si], tgt_scaled[ti])
    mmd_before = rbf_mmd_sq(src_scaled[si], tgt_scaled[ti], bandwidth_sq=bw)
    mmd_after = rbf_mmd_sq(source_aligned[si], tgt_scaled[ti], bandwidth_sq=bw)

    # Write outputs
    extra_src = [c for c in source_df.columns if c not in META_COLUMNS and c not in features]
    extra_tgt = [c for c in target_df.columns if c not in META_COLUMNS and c not in features]
    build_output_dataframe(source_df, features, source_aligned, extra_columns=extra_src).to_csv(
        pairs_dir / "aligned_source.csv", index=False
    )
    build_output_dataframe(target_df, features, tgt_scaled, extra_columns=extra_tgt).to_csv(
        pairs_dir / "target_reference.csv", index=False
    )
    np.savez(
        pairs_dir / "artifacts.npz",
        feature_names=np.array(features, dtype=object),
        fill_values=np.asarray(prep_meta["fill_values"], dtype=float),
        feature_mean=np.asarray(prep_meta["feature_mean"], dtype=float),
        feature_std=np.asarray(prep_meta["feature_std"], dtype=float),
        source_mean=np.asarray(coral_meta["source_mean"], dtype=float),
        target_mean=np.asarray(coral_meta["target_mean"], dtype=float),
        transform=np.asarray(coral_meta["transform"], dtype=float),
    )
    metrics = {
        "source": args.source_domain, "target": args.target_domain,
        "balance_fit": args.balance_fit, "kind_label": kind_label,
        "n_source": len(source_df), "n_target": len(target_df),
        "cov_gap_before": cov_before, "cov_gap_after": cov_after,
        "cov_gap_reduction_pct": reduction_pct(cov_before, cov_after),
        "mmd_before": mmd_before, "mmd_after": mmd_after,
        "mmd_reduction_pct": reduction_pct(mmd_before, mmd_after),
    }
    (pairs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (kind_root / "run_config.json").write_text(json.dumps({
        "protocol": "Phase 3 CORAL re-run", "kind_label": kind_label,
        "balance_fit": args.balance_fit, "source_kind": args.source_kind,
        "source_domain": args.source_domain, "target_domain": args.target_domain,
        "n_features": len(features), "reg": args.reg, "metrics": metrics,
    }, indent=2))

    try:
        write_projection_plot(
            source_before=src_scaled, source_after=source_aligned, target=tgt_scaled,
            output_path=pairs_dir / "alignment_pca.png", rng=rng, max_samples=800,
        )
    except Exception as exc:  # plot is diagnostic-only; don't fail the run
        warnings.warn(
            f"PCA projection plot failed ({type(exc).__name__}: {exc}); "
            f"CORAL metrics/artifacts are unaffected.",
            RuntimeWarning, stacklevel=2,
        )

    print(f"\n  cov_gap reduction: {metrics['cov_gap_reduction_pct']:.2f}%")
    print(f"  mmd reduction:     {metrics['mmd_reduction_pct']:.2f}%")
    print(f"Wrote: {kind_root}")


# ============================================================================
# 3. EXPERIMENT
# ============================================================================
def _load_experiment_source(alignment: str, raw_path: Path, aligned_path: Path) -> pd.DataFrame:
    path = raw_path if alignment == "before_coral" else aligned_path
    df = pd.read_csv(path)
    _VALIDATE_PARTITION(df.columns)
    if "binary_label" not in df.columns:
        raise ValueError(f"source file missing binary_label: {path}")
    return df


def _apply_source_label_override(src: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    """Override source labels from a relabel CSV (post-CORAL HDBSCAN relabel).

    Joins on ``window_start``; overrides ``binary_label`` with ``binary_label_new``;
    drops rows where the relabel CSV's ``discovered_label == 'uncertain'`` (universal
    drop-uncertain rule per feedback_labelling_rule.md — applies to any new label set).
    """
    rel = pd.read_csv(labels_path)
    required = {"window_start", "binary_label_new", "discovered_label"}
    missing = required - set(rel.columns)
    if missing:
        raise ValueError(
            f"relabel CSV missing required columns {sorted(missing)}: {labels_path}"
        )
    rel_keep = rel[rel["discovered_label"] != "uncertain"][
        ["window_start", "binary_label_new"]
    ].copy()
    n_before = len(src)
    merged = src.merge(rel_keep, on="window_start", how="inner")
    if len(merged) == 0:
        raise RuntimeError(
            f"source-labels-from join produced 0 rows; check window_start alignment "
            f"between source ({len(src)} rows) and {labels_path} ({len(rel_keep)} non-uncertain rows)."
        )
    merged["binary_label"] = merged["binary_label_new"].astype(int)
    merged = merged.drop(columns=["binary_label_new"])
    n_dropped = n_before - len(merged)
    print(f"  [source-labels-from] {labels_path.name}: "
          f"{n_before} -> {len(merged)} rows ({n_dropped} dropped as uncertain or unmatched); "
          f"anom={(merged['binary_label']==1).sum()}")
    return merged


def _derive_stage_prefix(classifiers, views, *, enable_late_fusion,
                         enable_attention_fusion, enable_tuning) -> tuple[str, bool, bool, bool]:
    """Single source of truth for stage label and which passes are active.

    Returns ``(prefix, late_fusion_enabled, attention_enabled, is_stage3)``.
    Used by both ``cmd_experiment`` and ``cmd_all`` to keep output dirs consistent.
    """
    view_set = set(views)
    late_fusion_enabled = bool(enable_late_fusion) and {"graph", "stat"}.issubset(view_set)
    attention_enabled = bool(enable_attention_fusion) and {"graph", "stat"}.issubset(view_set)
    tuning_enabled = bool(enable_tuning)
    uses_deep = any(c in ("mlp", "lstm") for c in classifiers)
    is_stage3 = uses_deep or attention_enabled or tuning_enabled
    is_stage2 = len(classifiers) > 1 or late_fusion_enabled
    if is_stage3:
        prefix = "stage3_deep_tuning_attention"
    elif is_stage2:
        prefix = "stage2_classifier_and_latefusion"
    else:
        prefix = "stage1_minimal"
    return prefix, late_fusion_enabled, attention_enabled, is_stage3


def _verify_coral_artifact(coral_root: Path, expected_source: str,
                           expected_target: str, expected_kind: str) -> dict:
    """Assert the CORAL artifact matches the caller's expectations.

    Validates against PAIR-LOCAL ``pairs/{src}__to__{tgt}/metrics.json`` so that
    multiple (source, target) pairs can coexist under the same ``kind_label``
    without the last-writer clobbering validation state. The root
    ``run_config.json`` is only consulted for shared metadata (reg, source_kind);
    it may reflect a different pair's settings when kind_label hosts several.

    Catches the silent-mismatch class of bug (e.g. running experiment with
    --coral-kind study_only against artifacts re-fit under study_plus_historical,
    or mixing up source/target domains).
    """
    pairs_dir = coral_root / "pairs" / f"{expected_source}__to__{expected_target}"
    pair_metrics_path = pairs_dir / "metrics.json"
    if not pair_metrics_path.exists():
        raise FileNotFoundError(
            f"CORAL pair metrics missing at {pair_metrics_path}.\n"
            f"Re-run: python Scripts/phase3_pipeline.py coral "
            f"--source-domain {expected_source} --target-domain {expected_target} "
            f"(with the --source-kind / --balance-fit that produce kind_label={expected_kind!r})."
        )
    pair_cfg = json.loads(pair_metrics_path.read_text())

    # Pair-local field names are "source"/"target"; both are authoritative for this pair.
    mismatches = []
    if pair_cfg.get("kind_label") != expected_kind:
        mismatches.append(f"kind_label: artifact={pair_cfg.get('kind_label')!r} vs expected={expected_kind!r}")
    if pair_cfg.get("source") != expected_source:
        mismatches.append(f"source: artifact={pair_cfg.get('source')!r} vs expected={expected_source!r}")
    if pair_cfg.get("target") != expected_target:
        mismatches.append(f"target: artifact={pair_cfg.get('target')!r} vs expected={expected_target!r}")
    if mismatches:
        raise RuntimeError(
            "CORAL pair artifact does not match the experiment configuration:\n  - "
            + "\n  - ".join(mismatches)
            + f"\nartifact path: {pair_metrics_path}\n"
            f"Re-run `phase3_pipeline.py coral` with matching --source-kind / "
            f"--balance-fit / --source-domain / --target-domain."
        )

    # Optionally enrich with root-level shared metadata if present (non-fatal).
    root_cfg_path = coral_root / "run_config.json"
    if root_cfg_path.exists():
        try:
            root_cfg = json.loads(root_cfg_path.read_text())
            for k in ("reg", "source_kind", "protocol", "n_features"):
                if k in root_cfg and k not in pair_cfg:
                    pair_cfg[k] = root_cfg[k]
        except json.JSONDecodeError:
            pass  # root config is advisory only
    return pair_cfg


def _load_target_study(target_domain: str) -> pd.DataFrame:
    path = UNIFIED_DIR / f"{target_domain}.csv"
    df = pd.read_csv(path)
    df = df[df["provenance"] == "study"].reset_index(drop=True)
    _VALIDATE_PARTITION(df.columns)
    if "binary_label" not in df.columns:
        raise ValueError(f"target file missing binary_label: {path}")
    return df


def _load_event_map(target_df: pd.DataFrame, target_domain: str) -> pd.Series:
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest = manifest[
        (manifest["domain"] == target_domain) & (manifest["provenance"] == "study")
    ].copy()
    manifest["start_time"] = pd.to_datetime(manifest["start_time"], format="ISO8601", utc=True)
    manifest["end_time"] = pd.to_datetime(manifest["end_time"], format="ISO8601", utc=True)
    target_ts = pd.to_datetime(target_df["window_start"], format="ISO8601", utc=True)
    event_ids = pd.Series([""] * len(target_df), index=target_df.index, dtype=object)
    anomaly_mask = target_df["binary_label"] == 1
    for _, row in manifest.iterrows():
        in_event = anomaly_mask & (target_ts >= row["start_time"]) & (target_ts <= row["end_time"])
        event_ids.loc[in_event] = row["event_id"]
    unmatched = anomaly_mask & (event_ids == "")
    if unmatched.any():
        raise RuntimeError(f"{int(unmatched.sum())} anomaly rows unmatched to events")
    return event_ids


def _row_level_metrics(y_true, y_pred, y_score) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = float("nan")
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    finite = np.isfinite(y_score_arr)
    if finite.any() and finite.sum() == len(y_score_arr):
        ece = _expected_calibration_error(np.clip(y_score_arr, 0.0, 1.0), y_true_arr)
        brier = _brier_score(np.clip(y_score_arr, 0.0, 1.0), y_true_arr)
    else:
        ece = float("nan")
        brier = float("nan")
    return {
        "macro_f1": float(macro_f1),
        "precision_normal": float(p[0]), "precision_anomaly": float(p[1]),
        "recall_normal": float(r[0]), "recall_anomaly": float(r[1]),
        "f1_normal": float(f1[0]), "f1_anomaly": float(f1[1]),
        "pr_auc_anomaly": float(pr_auc),
        "ece": float(ece),
        "brier": float(brier),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_rows": int(len(y_true)),
        "n_anomaly": int((y_true == 1).sum()),
        "n_normal": int((y_true == 0).sum()),
    }


def _per_event_metrics(y_true, y_pred, event_ids) -> pd.DataFrame:
    rows: list[dict] = []
    for eid, idx in event_ids.groupby(event_ids).groups.items():
        if eid == "":
            continue
        idx_arr = np.asarray(list(idx))
        n_w = len(idx_arr)
        n_hit = int((y_pred[idx_arr] == 1).sum())
        rows.append({
            "event_id": eid, "n_windows": n_w,
            "n_detected_windows": n_hit,
            "recall": n_hit / n_w if n_w else float("nan"),
            "detected": bool(n_hit > 0),
        })
    return pd.DataFrame(rows)


def _make_cv_splitter(classifier: str, y: np.ndarray, requested_folds: int, seed: int):
    """Pick a CV splitter and return (splitter, effective_folds).

    - Flat classifiers: StratifiedKFold with shuffling.
    - Sequence classifiers (lstm): TimeSeriesSplit (no stratification, no
      shuffling) to avoid leaking future into past.

    Returns ``(None, 0)`` to signal "skip tuning entirely" when either class has
    fewer than 2 samples (StratifiedKFold requires each class in every split).
    Otherwise caps folds at ``min(requested, minority_count)`` for flat
    classifiers and ``min(requested, len(y)-1)`` for TimeSeriesSplit.
    """
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    minority = min(n_pos, n_neg)

    if classifier in SEQ_CLASSIFIERS:
        # TimeSeriesSplit needs at least n_splits + 1 samples total.
        if n < 3:
            warnings.warn(f"Skipping tuning: only {n} samples (need >= 3).")
            return None, 0
        effective = max(2, min(requested_folds, n - 1))
        if effective < requested_folds:
            warnings.warn(
                f"Reducing TimeSeriesSplit folds from {requested_folds} to {effective} "
                f"(only {n} samples)."
            )
        return TimeSeriesSplit(n_splits=effective), effective

    # Flat classifiers: StratifiedKFold needs >= 2 samples per class; if the
    # minority has < 2, we cannot run any stratified CV -- skip tuning for this
    # (classifier, view, alignment) and let the default params be used.
    if minority < 2:
        warnings.warn(
            f"Skipping tuning: minority class has {minority} sample(s); "
            "need >= 2 for StratifiedKFold."
        )
        return None, 0
    effective = max(2, min(requested_folds, minority))
    if effective < requested_folds:
        warnings.warn(
            f"Reducing CV folds from {requested_folds} to {effective} "
            f"because minority class has only {minority} samples."
        )
    return StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed), effective


def _tune_classifier(classifier: str, view: str, source_df: pd.DataFrame,
                     seed: int, cv_folds: int) -> dict:
    """Source-only CV over TUNING_GRIDS[classifier]; returns best params by macro-F1.

    Splitter choice (via ``_make_cv_splitter``):
    - Flat classifiers (xgb, rf, mlp): StratifiedKFold, shuffled.
    - Sequence classifier (lstm): TimeSeriesSplit, chronologically ordered.

    For LSTM, the source_df is sorted by (domain, window_start) before splitting
    so that train rows precede validation rows in time.
    """
    grid = TUNING_GRIDS.get(classifier, [])
    if not grid:
        return {}
    feature_cols = VIEW_CONFIGS[view]

    if classifier in SEQ_CLASSIFIERS:
        sort_keys = ["domain", "window_start"] if "domain" in source_df.columns else ["window_start"]
        df = source_df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    else:
        df = source_df.reset_index(drop=True)
    y_full = df["binary_label"].to_numpy(dtype=np.int32)
    splitter, effective_folds = _make_cv_splitter(classifier, y_full, cv_folds, seed)
    if splitter is None:
        # Use None (not NaN) so the tuning log is strict-JSON-clean.
        return {"best_params": {}, "best_cv_macro_f1": None,
                "effective_folds": 0, "grid": [],
                "note": "tuning skipped (insufficient data for CV)"}

    best_score = -np.inf
    best_params: dict = {}
    scored: list[dict] = []
    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in splitter.split(np.zeros(len(df)), y_full):
            if classifier in SEQ_CLASSIFIERS:
                tr_df = df.iloc[tr_idx].copy()
                va_df = df.iloc[va_idx].copy()
                model = fit_lstm(tr_df, feature_cols, seed, **params)
                p1 = model.predict_proba(va_df)[:, 1]
                y_va = va_df["binary_label"].to_numpy(dtype=np.int32)
            else:
                X_tr = df.iloc[tr_idx][feature_cols].to_numpy(dtype=np.float32)
                y_tr = y_full[tr_idx]
                X_va = df.iloc[va_idx][feature_cols].to_numpy(dtype=np.float32)
                y_va = y_full[va_idx]
                fitter = CLASSIFIER_FITTERS[classifier]
                model = fitter(X_tr, y_tr, seed, **params)
                p1 = model.predict_proba(X_va)[:, 1]
            y_pred = (p1 >= 0.5).astype(np.int32)
            fold_scores.append(f1_score(y_va, y_pred, average="macro"))
        mean_score = float(np.mean(fold_scores))
        scored.append({"params": params, "cv_macro_f1_mean": mean_score,
                       "cv_macro_f1_std": float(np.std(fold_scores))})
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)
    return {
        "best_params": best_params, "best_cv_macro_f1": best_score,
        "effective_folds": effective_folds, "grid": scored,
    }


def _fit_base_model(classifier: str, view: str, source_df: pd.DataFrame, seed: int,
                    params: dict | None = None):
    """Fit the right base model for ``classifier``; return a predict_proba-capable object."""
    params = params or {}
    feature_cols = VIEW_CONFIGS[view]
    if classifier in SEQ_CLASSIFIERS:
        return fit_lstm(source_df, feature_cols, seed, **params)
    X = source_df[feature_cols].to_numpy(dtype=np.float32)
    y = source_df["binary_label"].to_numpy(dtype=np.int32)
    return CLASSIFIER_FITTERS[classifier](X, y, seed, **params)


def _predict_proba(classifier: str, model, df: pd.DataFrame, view: str) -> np.ndarray:
    feature_cols = VIEW_CONFIGS[view]
    if classifier in SEQ_CLASSIFIERS:
        return model.predict_proba(df)[:, 1]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    return model.predict_proba(X)[:, 1]


# ============================================================================
# Inference-latency instrumentation (§5.8)
# ============================================================================
# Measured on the same fitted model that produces the reported detection
# metrics, so F1/PR-AUC and latency are never out of sync. Two regimes:
#   * single-window: one new window arrives, produce one score. For flat
#     classifiers this is a 1-row predict; for LSTM it is a seq_len-row
#     rolling slice (the real streaming cost of LSTMPipeline, which rebuilds
#     sequences internally and has no single-sequence fast path).
#   * batched: full target_df predict divided by n_windows (throughput-style).
# All values are milliseconds. Used only for reporting; does not affect the
# detection outputs.
LATENCY_SINGLE_REPEATS = 30
LATENCY_BATCHED_REPEATS = 3


def _summarise_timings(ms: list[float]) -> tuple[float, float]:
    if not ms:
        return (float("nan"), float("nan"))
    arr = np.asarray(ms, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _time_inference(classifier: str, model, df: pd.DataFrame, view: str,
                    seed: int) -> dict[str, float]:
    """Measure single-window + batched inference latency for a fitted model.

    Flat classifiers: single-window = 1-row predict.
    LSTM: single-window = seq_len-row rolling slice ending at the sampled
    index (matches streaming use of LSTMPipeline, which rebuilds sequences
    inside every ``predict_proba`` call).
    """
    n = len(df)
    nan_block = {
        "single_window_ms_median": float("nan"),
        "single_window_ms_p95": float("nan"),
        "batched_ms_per_window_median": float("nan"),
    }
    if n == 0:
        return nan_block

    rng = np.random.default_rng(seed)
    singles: list[float] = []
    if classifier in SEQ_CLASSIFIERS:
        seq_len = int(getattr(model.params, "seq_len", 0) or 0)
        if seq_len <= 0 or n < seq_len:
            # Can't form a streaming window; leave NaN rather than fake it.
            pass
        else:
            end_choices = rng.integers(seq_len - 1, n, size=LATENCY_SINGLE_REPEATS)
            for end in end_choices:
                slab = df.iloc[int(end) - seq_len + 1 : int(end) + 1]
                t0 = time.perf_counter()
                _predict_proba(classifier, model, slab, view)
                singles.append((time.perf_counter() - t0) * 1000.0)
    else:
        idxs = rng.integers(0, n, size=LATENCY_SINGLE_REPEATS)
        for i in idxs:
            slab = df.iloc[int(i) : int(i) + 1]
            t0 = time.perf_counter()
            _predict_proba(classifier, model, slab, view)
            singles.append((time.perf_counter() - t0) * 1000.0)

    batched_per_window: list[float] = []
    for _ in range(LATENCY_BATCHED_REPEATS):
        t0 = time.perf_counter()
        _predict_proba(classifier, model, df, view)
        batched_per_window.append((time.perf_counter() - t0) * 1000.0 / n)

    med, p95 = _summarise_timings(singles)
    return {
        "single_window_ms_median": med,
        "single_window_ms_p95": p95,
        "batched_ms_per_window_median": float(np.median(batched_per_window))
        if batched_per_window else float("nan"),
    }


def _time_closure(fn, n_windows: int, seed: int) -> dict[str, float]:
    """Time a pure-numpy closure that runs the full target set in one call.

    Used for late/attention fusion where the per-window cost is sub-microsecond
    numpy, so we time the closure in bulk and derive per-window numbers.
    ``fn`` must be callable with no arguments and must return something (we
    store the last return to avoid dead-code elimination concerns).
    """
    if n_windows <= 0:
        return {"single_window_ms_median": float("nan"),
                "single_window_ms_p95": float("nan"),
                "batched_ms_per_window_median": float("nan")}
    # One warm-up pass (primes caches, triggers any lazy compilation).
    fn()
    ms: list[float] = []
    for _ in range(LATENCY_BATCHED_REPEATS):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ms, dtype=np.float64)
    per_win = arr / n_windows
    return {
        "single_window_ms_median": float(np.median(per_win)),
        "single_window_ms_p95": float(np.percentile(per_win, 95)),
        "batched_ms_per_window_median": float(np.median(per_win)),
    }


def _compute_oof_source_scores(
    classifier: str, view: str, source_df: pd.DataFrame,
    seed: int, params: dict | None, cv_folds: int,
) -> np.ndarray:
    """Out-of-fold source anomaly probabilities via k-fold CV.

    Returns an array aligned to ``source_df`` row order. For LSTM the
    source_df is sorted by (domain, window_start) internally and mapped back,
    so callers get scores in the original order. Rows that fall outside any
    validation fold (can happen with TimeSeriesSplit's first chunk) are NaN.
    """
    params = params or {}
    feature_cols = VIEW_CONFIGS[view]
    if classifier in SEQ_CLASSIFIERS:
        sort_keys = ["domain", "window_start"] if "domain" in source_df.columns else ["window_start"]
        sort_idx = source_df.sort_values(sort_keys, kind="mergesort").index.to_numpy()
        df = source_df.loc[sort_idx].reset_index(drop=True)
    else:
        sort_idx = np.arange(len(source_df))
        df = source_df.reset_index(drop=True)
    y_full = df["binary_label"].to_numpy(dtype=np.int32)
    splitter, _ = _make_cv_splitter(classifier, y_full, cv_folds, seed)
    oof = np.full(len(df), np.nan, dtype=np.float64)
    if splitter is None:
        # Map NaN back to original order and return; the caller must handle masking.
        inverse = np.empty_like(sort_idx)
        inverse[sort_idx] = np.arange(len(sort_idx))
        return oof[inverse]
    for tr_idx, va_idx in splitter.split(np.zeros(len(df)), y_full):
        if classifier in SEQ_CLASSIFIERS:
            tr_df = df.iloc[tr_idx].copy()
            va_df = df.iloc[va_idx].copy()
            model = fit_lstm(tr_df, feature_cols, seed, **params)
            oof[va_idx] = model.predict_proba(va_df)[:, 1]
        else:
            X_tr = df.iloc[tr_idx][feature_cols].to_numpy(dtype=np.float32)
            y_tr = y_full[tr_idx]
            X_va = df.iloc[va_idx][feature_cols].to_numpy(dtype=np.float32)
            model = CLASSIFIER_FITTERS[classifier](X_tr, y_tr, seed, **params)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
    inverse = np.empty_like(sort_idx)
    inverse[sort_idx] = np.arange(len(sort_idx))
    return oof[inverse]


TAU_BOUNDS = (0.1, 0.9)


@dataclass
class Calibrator:
    """Probability remapping plus a decision threshold, both fitted on source OOF.

    The scaler can be isotonic regression (non-parametric, risks clip-induced
    rank collapse under severe shift) or Platt scaling (1-parameter logistic,
    extrapolates smoothly). The threshold can be chosen by macro-F1 sweep
    (overfits source) or by a target-prior quantile (distribution-aware).
    Both are fitted on source out-of-fold predictions so no target labels leak.
    """
    method: str  # "isotonic" | "platt"
    tau_policy: str  # "f1_sweep" | "prior_quantile"
    tau: float
    iso: IsotonicRegression | None = None
    platt: LogisticRegression | None = None

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == "isotonic":
            return self.iso.transform(p)
        if self.method == "platt":
            return self.platt.predict_proba(p.reshape(-1, 1))[:, 1]
        raise ValueError(f"unknown calibration method: {self.method}")

    def predict(self, p: np.ndarray) -> np.ndarray:
        return (self.apply(p) >= self.tau).astype(np.int32)


def _sweep_threshold(p_cal: np.ndarray, y: np.ndarray) -> float:
    """Pick threshold maximising macro-F1 over all unique OOF score cuts.

    Source-overfit: best τ on source does not always transfer to target under
    severe distribution shift. Prefer ``_prior_quantile_threshold`` when the
    anomaly prior is known and the transfer is strong (e.g. cross-AS).
    """
    candidates = np.unique(p_cal)
    candidates = np.concatenate(([0.0], candidates, [1.0]))
    best_tau = 0.5
    best_score = -np.inf
    for tau in candidates:
        pred = (p_cal >= tau).astype(np.int32)
        score = f1_score(y, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def _prior_quantile_threshold(p_cal: np.ndarray, y: np.ndarray,
                              target_scores: np.ndarray | None = None) -> float:
    """Pick τ so that the predicted-positive rate matches the anomaly prior.

    Uses the source anomaly prior (``y.mean()``) as the target rate. If
    ``target_scores`` is provided, applies the prior to target scores
    directly (quantile on target distribution), which is distribution-aware
    without touching target labels. Clamped to ``TAU_BOUNDS``.
    """
    prior = float(np.clip(y.mean(), 0.01, 0.5))
    scores_for_quantile = p_cal if target_scores is None else target_scores
    if len(scores_for_quantile) == 0:
        return 0.5
    tau = float(np.quantile(scores_for_quantile, 1.0 - prior))
    return float(np.clip(tau, TAU_BOUNDS[0], TAU_BOUNDS[1]))


def _fit_calibrator(
    p_raw: np.ndarray, y: np.ndarray,
    *, method: str = "isotonic", tau_policy: str = "f1_sweep",
    target_scores: np.ndarray | None = None,
) -> Calibrator | None:
    """Fit a probability scaler and decision threshold on source OOF.

    - ``method="isotonic"``: non-parametric; good for non-linear miscal but
      clips at train range, hurting PR-AUC under shift.
    - ``method="platt"``: 1-parameter sigmoid; lower variance; extrapolates.
    - ``tau_policy="f1_sweep"``: maximise macro-F1 on source OOF (overfits).
    - ``tau_policy="prior_quantile"``: pick τ matching source/target anomaly
      prior; bounded to ``TAU_BOUNDS``; less source-overfit.

    Returns None if OOF is degenerate (<5 rows, single class, or all NaN).
    """
    mask = ~np.isnan(p_raw)
    if mask.sum() < 5:
        return None
    p_valid = np.clip(p_raw[mask], 0.0, 1.0)
    y_valid = y[mask]
    if len(np.unique(y_valid)) < 2:
        return None

    iso: IsotonicRegression | None = None
    platt: LogisticRegression | None = None
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_valid, y_valid)
        p_cal = iso.transform(p_valid)
    elif method == "platt":
        platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt.fit(p_valid.reshape(-1, 1), y_valid)
        p_cal = platt.predict_proba(p_valid.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"unknown calibration method: {method}")

    if tau_policy == "f1_sweep":
        tau = _sweep_threshold(p_cal, y_valid)
    elif tau_policy == "prior_quantile":
        tgt_cal = None
        if target_scores is not None and method == "isotonic":
            tgt_cal = iso.transform(np.clip(target_scores, 0.0, 1.0))
        elif target_scores is not None and method == "platt":
            tgt_cal = platt.predict_proba(np.clip(target_scores, 0.0, 1.0).reshape(-1, 1))[:, 1]
        tau = _prior_quantile_threshold(p_cal, y_valid, target_scores=tgt_cal)
    else:
        raise ValueError(f"unknown tau_policy: {tau_policy}")

    return Calibrator(method=method, tau_policy=tau_policy, tau=float(tau),
                      iso=iso, platt=platt)


def _expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Binned ECE: weighted mean |conf - acc| per bin."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def _brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def _run_single_config(view, alignment, classifier, seed,
                       source_df, target_df, event_ids, predictions_dir,
                       tuned_params: dict | None = None,
                       calibrator: Calibrator | None = None):
    """Train a single (view, alignment, classifier, seed) config and return metrics + scores.

    Returns: (row_metrics, per_event_df, y_score_target, y_score_source,
    y_score_target_raw). ``y_score_target`` is calibrated when a calibrator is
    passed in; ``y_score_target_raw`` is always the model's unmodified proba,
    kept so late/attention fusion can enforce a consistent raw-vs-calibrated
    policy across both base views. Source-side scores are in-sample and kept
    only for diagnostics — the attention stacker is fitted on OOF source
    scores computed in a separate pass to avoid leakage.
    """
    feature_cols = VIEW_CONFIGS[view]
    y_test = target_df["binary_label"].to_numpy(dtype=np.int32)

    t0 = time.perf_counter()
    model = _fit_base_model(classifier, view, source_df, seed, params=tuned_params)
    t1 = time.perf_counter()
    y_score_tgt_raw = _predict_proba(classifier, model, target_df, view)
    t2 = time.perf_counter()
    y_score_src = _predict_proba(classifier, model, source_df, view)
    t3 = time.perf_counter()
    if calibrator is not None:
        y_score_tgt = calibrator.apply(y_score_tgt_raw)
        y_pred = (y_score_tgt >= calibrator.tau).astype(np.int32)
    else:
        y_score_tgt = y_score_tgt_raw
        y_pred = (y_score_tgt >= 0.5).astype(np.int32)
    fit_ms = (t1 - t0) * 1000.0
    predict_target_ms = (t2 - t1) * 1000.0
    predict_source_ms = (t3 - t2) * 1000.0
    elapsed = t3 - t0
    # Always keep raw target scores alongside the (possibly calibrated)
    # y_score_tgt so downstream fusion can enforce a consistent raw-vs-cal
    # policy per seed even when a calibrator is missing for one base view.
    y_score_tgt_raw = np.asarray(y_score_tgt_raw, dtype=np.float64)

    # Per-window inference latency (measured on the same fitted model that
    # produced the metrics above — no parallel retrain).
    latency = _time_inference(classifier, model, target_df, view, seed)

    y_train = source_df["binary_label"].to_numpy(dtype=np.int32)
    row_m = _row_level_metrics(y_test, y_pred, y_score_tgt)
    row_m.update({
        "view": view, "classifier": classifier, "alignment": alignment,
        "seed": seed, "fit_seconds": round(elapsed, 2),
        "fit_ms": round(fit_ms, 3),
        "predict_target_ms": round(predict_target_ms, 3),
        "predict_source_ms": round(predict_source_ms, 3),
        "single_window_ms_median": round(latency["single_window_ms_median"], 4)
            if np.isfinite(latency["single_window_ms_median"]) else float("nan"),
        "single_window_ms_p95": round(latency["single_window_ms_p95"], 4)
            if np.isfinite(latency["single_window_ms_p95"]) else float("nan"),
        "batched_ms_per_window_median": round(latency["batched_ms_per_window_median"], 4)
            if np.isfinite(latency["batched_ms_per_window_median"]) else float("nan"),
        "n_features": len(feature_cols),
        "scale_pos_weight": compute_scale_pos_weight(y_train),
    })
    ev_m = _per_event_metrics(y_test, y_pred, event_ids)
    ev_m["view"] = view
    ev_m["classifier"] = classifier
    ev_m["alignment"] = alignment
    ev_m["seed"] = seed

    pred_df = pd.DataFrame({
        "window_start": target_df["window_start"].values,
        "window_id": target_df["window_id"].values,
        "binary_label": y_test, "y_pred": y_pred,
        "y_score": y_score_tgt, "event_id": event_ids.values,
    })
    pred_df.to_csv(
        predictions_dir / f"{view}__{alignment}__{classifier}__seed{seed}.csv",
        index=False,
    )
    return row_m, ev_m, y_score_tgt, y_score_src, y_score_tgt_raw


def _run_late_fusion(rule, alignment, classifier, seed,
                     p_graph, p_stat, target_df, event_ids, predictions_dir,
                     mean_calibrator: Calibrator | None = None,
                     tau_graph: float | None = None,
                     tau_stat: float | None = None):
    """Derive a late-fusion result from cached graph + stat scores.

    - ``fusion_late_mean``: averaged probability. If ``mean_calibrator`` is
      provided (fitted on source OOF of the same mean), apply it and threshold
      at its τ; otherwise threshold the raw mean at 0.5.
    - ``fusion_late_or``: hard OR of per-view thresholded predictions. When
      ``tau_graph``/``tau_stat`` are provided (from per-view calibrators),
      they replace the 0.5 defaults. Hard OR is intentionally *not*
      re-calibrated (no probability output to calibrate).
    """
    y_test = target_df["binary_label"].to_numpy(dtype=np.int32)
    n_win = len(target_df)
    if rule == "fusion_late_mean":
        y_score_raw = (p_graph + p_stat) / 2.0
        if mean_calibrator is not None:
            y_score = mean_calibrator.apply(y_score_raw)
            threshold = mean_calibrator.tau
        else:
            y_score = y_score_raw
            threshold = 0.5
        y_pred = (y_score >= threshold).astype(np.int32)
        def _fuse():
            raw = (p_graph + p_stat) / 2.0
            s = mean_calibrator.apply(raw) if mean_calibrator is not None else raw
            thr = mean_calibrator.tau if mean_calibrator is not None else 0.5
            return (s >= thr).astype(np.int32)
    elif rule == "fusion_late_or":
        tg = 0.5 if tau_graph is None else float(tau_graph)
        ts = 0.5 if tau_stat is None else float(tau_stat)
        y_pred = ((p_graph >= tg) | (p_stat >= ts)).astype(np.int32)
        y_score = np.maximum(p_graph, p_stat)
        def _fuse():
            return ((p_graph >= tg) | (p_stat >= ts)).astype(np.int32)
    else:
        raise ValueError(f"unknown late-fusion rule: {rule}")

    # Timing is the fusion-only step: base-view predicts are already counted
    # against the graph/stat rows written by _run_single_config.
    latency = _time_closure(_fuse, n_win, seed)

    row_m = _row_level_metrics(y_test, y_pred, y_score)
    row_m.update({
        "view": rule, "classifier": classifier, "alignment": alignment,
        "seed": seed, "fit_seconds": 0.0,
        "fit_ms": 0.0,
        "predict_target_ms": float("nan"),
        "predict_source_ms": float("nan"),
        "single_window_ms_median": round(latency["single_window_ms_median"], 6),
        "single_window_ms_p95": round(latency["single_window_ms_p95"], 6),
        "batched_ms_per_window_median": round(latency["batched_ms_per_window_median"], 6),
        "n_features": len(VIEW_CONFIGS["graph"]) + len(VIEW_CONFIGS["stat"]),
        "scale_pos_weight": float("nan"),
    })
    ev_m = _per_event_metrics(y_test, y_pred, event_ids)
    ev_m["view"] = rule
    ev_m["classifier"] = classifier
    ev_m["alignment"] = alignment
    ev_m["seed"] = seed

    pred_df = pd.DataFrame({
        "window_start": target_df["window_start"].values,
        "window_id": target_df["window_id"].values,
        "binary_label": y_test, "y_pred": y_pred,
        "y_score": y_score, "event_id": event_ids.values,
    })
    pred_df.to_csv(
        predictions_dir / f"{rule}__{alignment}__{classifier}__seed{seed}.csv",
        index=False,
    )
    return row_m, ev_m


def _run_attention_fusion(alignment, classifier, seed,
                          p_graph_src, p_stat_src, y_src,
                          p_graph_tgt, p_stat_tgt,
                          target_df, event_ids, predictions_dir,
                          calibrate_stacker: bool = False):
    """Learned late fusion: logistic regression on source probs, applied at target.

    Returns (row_metrics, per_event_df, attention_weights_dict, stacker_tau).

    When ``calibrate_stacker`` is True, apply the fitted stacker back to its
    source OOF inputs, run a macro-F1 threshold sweep on the resulting source
    scores, and use that τ at target time instead of 0.5. The stacker's own
    sigmoid output is already a probability so no isotonic remap is added on
    top (that would over-fit on a 2-parameter model).
    """
    if fit_attention_fusion is None:
        raise RuntimeError("attention fusion requires phase3_deep_models import")
    y_test = target_df["binary_label"].to_numpy(dtype=np.int32)
    n_win = len(target_df)

    t0 = time.perf_counter()
    weights = fit_attention_fusion(p_graph_src, p_stat_src, y_src, seed=seed)
    t1 = time.perf_counter()
    y_score = weights.apply(p_graph_tgt, p_stat_tgt)
    t2 = time.perf_counter()
    stacker_tau = 0.5
    if calibrate_stacker:
        stacker_src_score = weights.apply(p_graph_src, p_stat_src)
        stacker_tau = _sweep_threshold(
            np.clip(stacker_src_score, 0.0, 1.0), y_src.astype(np.int32)
        )
    y_pred = (y_score >= stacker_tau).astype(np.int32)
    fit_ms = (t1 - t0) * 1000.0
    predict_target_ms = (t2 - t1) * 1000.0
    elapsed = time.perf_counter() - t0

    # Stacker-only latency: time weights.apply() + threshold. Base-view
    # predicts are billed to the graph/stat rows in _run_single_config.
    tau_local = float(stacker_tau)
    def _fuse():
        s = weights.apply(p_graph_tgt, p_stat_tgt)
        return (s >= tau_local).astype(np.int32)
    latency = _time_closure(_fuse, n_win, seed)

    row_m = _row_level_metrics(y_test, y_pred, y_score)
    row_m.update({
        "view": ATTENTION_VIEW, "classifier": classifier, "alignment": alignment,
        "seed": seed, "fit_seconds": round(elapsed, 4),
        "fit_ms": round(fit_ms, 3),
        "predict_target_ms": round(predict_target_ms, 3),
        "predict_source_ms": float("nan"),
        "single_window_ms_median": round(latency["single_window_ms_median"], 6),
        "single_window_ms_p95": round(latency["single_window_ms_p95"], 6),
        "batched_ms_per_window_median": round(latency["batched_ms_per_window_median"], 6),
        "n_features": len(VIEW_CONFIGS["graph"]) + len(VIEW_CONFIGS["stat"]),
        "scale_pos_weight": float("nan"),
    })
    ev_m = _per_event_metrics(y_test, y_pred, event_ids)
    ev_m["view"] = ATTENTION_VIEW
    ev_m["classifier"] = classifier
    ev_m["alignment"] = alignment
    ev_m["seed"] = seed

    pred_df = pd.DataFrame({
        "window_start": target_df["window_start"].values,
        "window_id": target_df["window_id"].values,
        "binary_label": y_test, "y_pred": y_pred,
        "y_score": y_score, "event_id": event_ids.values,
    })
    pred_df.to_csv(
        predictions_dir / f"{ATTENTION_VIEW}__{alignment}__{classifier}__seed{seed}.csv",
        index=False,
    )
    weights_d = {
        "intercept": weights.intercept,
        "w_graph": weights.w_graph,
        "w_stat": weights.w_stat,
        "use_sigmoid": weights.use_sigmoid,
        "tau": float(stacker_tau),
    }
    return row_m, ev_m, weights_d, float(stacker_tau)


def cmd_experiment(args: argparse.Namespace) -> None:
    src_dom = args.source_domain
    tgt_dom = args.target_domain
    coral_root = CORAL_ALIGNED_ROOT / args.coral_kind
    raw_src_path = coral_root / "prepared_sources" / f"{src_dom}_binary_source.csv"
    aligned_src_path = coral_root / "pairs" / f"{src_dom}__to__{tgt_dom}" / "aligned_source.csv"

    # Strong compatibility check: artifacts must match (source, target, kind).
    # Guards against stale/mismatched CORAL outputs being silently reused.
    coral_cfg = _verify_coral_artifact(
        coral_root, expected_source=src_dom,
        expected_target=tgt_dom, expected_kind=args.coral_kind,
    )
    for p in (raw_src_path, aligned_src_path):
        if not p.exists():
            raise FileNotFoundError(f"CORAL artifact not found: {p}")

    default_prefix, late_fusion_enabled, attention_enabled, _ = _derive_stage_prefix(
        args.classifiers, args.views,
        enable_late_fusion=args.enable_late_fusion,
        enable_attention_fusion=getattr(args, "enable_attention_fusion", False),
        enable_tuning=getattr(args, "enable_tuning", False),
    )
    tuning_enabled = bool(getattr(args, "enable_tuning", False))
    uses_deep = any(c in ("mlp", "lstm") for c in args.classifiers)
    output_suffix = args.output_suffix or f"{default_prefix}_{args.coral_kind}"
    output_root = FUSION_OUTPUT_ROOT / output_suffix
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: deep models require torch.
    if uses_deep and not TORCH_AVAILABLE:
        raise RuntimeError(
            "Classifiers {mlp, lstm} require PyTorch. Install with: pip install torch"
        )

    print("=" * 72)
    print("Fusion experiment (I1 zero-shot)")
    print("=" * 72)
    print(f"{src_dom} -> {tgt_dom}  |  CORAL kind: {args.coral_kind}")
    print(f"Classifiers: {args.classifiers}  |  Views: {args.views}  |  "
          f"Late fusion: {'on' if late_fusion_enabled else 'off'}  |  "
          f"Attention fusion: {'on' if attention_enabled else 'off'}  |  "
          f"Tuning: {'on' if tuning_enabled else 'off'}  |  Seeds: {args.seeds}")
    print(f"Output: {output_root}")
    print()

    target_df = _load_target_study(tgt_dom)
    event_ids = _load_event_map(target_df, tgt_dom)
    n_events = event_ids[event_ids != ""].nunique()
    print(f"Target: {len(target_df)} rows ({(target_df['binary_label']==1).sum()} anom), {n_events} events")

    source_labels_from = getattr(args, "source_labels_from", None)
    if source_labels_from is not None:
        if "before_coral" in args.alignments:
            raise ValueError(
                "--source-labels-from is only meaningful for after_coral; "
                "remove before_coral from --alignments."
            )
        source_labels_from = Path(source_labels_from)
        if not source_labels_from.exists():
            raise FileNotFoundError(f"--source-labels-from path not found: {source_labels_from}")

    source_cache: dict[str, pd.DataFrame] = {}
    for alignment in args.alignments:
        src = _load_experiment_source(alignment, raw_src_path, aligned_src_path)
        if source_labels_from is not None and alignment == "after_coral":
            src = _apply_source_label_override(src, source_labels_from)
        source_cache[alignment] = src
        print(f"Source ({alignment}): {len(src)} rows ({(src['binary_label']==1).sum()} anom)")
    print()

    # --- Pass 0: hyperparameter tuning (source-only stratified CV) -----------
    # Tuned per (classifier, view, alignment) and reused for all seeds.
    tuned_params: dict[tuple, dict] = {}
    tuning_log: dict = {}
    if tuning_enabled:
        print(f"Tuning ({args.tuning_cv_folds}-fold source-only CV)...")
        for classifier in args.classifiers:
            for view in args.views:
                for alignment in args.alignments:
                    res = _tune_classifier(
                        classifier, view, source_cache[alignment],
                        seed=RNG_SEED, cv_folds=args.tuning_cv_folds,
                    )
                    if not res:
                        continue
                    tuned_params[(classifier, view, alignment)] = res["best_params"]
                    tuning_log.setdefault(classifier, {}).setdefault(view, {})[alignment] = res
                    score = res["best_cv_macro_f1"]
                    score_str = f"{score:.4f}" if score is not None else "skipped"
                    note = f"  [{res['note']}]" if res.get("note") else ""
                    print(f"  {classifier:<14} | {view:<18} | {alignment:<12}  "
                          f"best_cv_macro_f1={score_str}  "
                          f"params={res['best_params']}{note}")
        (output_root / "tuned_params.json").write_text(
            json.dumps(tuning_log, indent=2, default=str, allow_nan=False)
        )
        print()

    all_row: list[dict] = []
    all_event: list[pd.DataFrame] = []
    # Score caches keyed by (classifier, view, alignment, seed). Source-side
    # scores here are in-sample (diagnostics only); the attention stacker is
    # fitted on OOF source scores in a dedicated pass below.
    score_cache_tgt: dict[tuple, np.ndarray] = {}
    score_cache_tgt_raw: dict[tuple, np.ndarray] = {}

    # --- Pass 0.5: OOF source scores for base views (graph, stat, fusion_early).
    # Required by:
    #   - calibration (isotonic + τ sweep)
    #   - attention stacker (logistic regression on OOF source predictions)
    # Keyed by (classifier, view, alignment, seed). Stored raw; calibrators
    # are fitted in Pass 0.75 when calibration is enabled.
    calibration_enabled = bool(getattr(args, "enable_calibration", False))
    calib_cv_folds = int(getattr(args, "calibration_cv_folds", 5))
    calib_method = str(getattr(args, "calibration_method", "isotonic"))
    tau_policy = str(getattr(args, "tau_policy", "f1_sweep"))
    score_cache_src_oof: dict[tuple, np.ndarray] = {}
    calibrator_cache: dict[tuple, Calibrator] = {}
    calibration_log: dict = {}
    oof_needed_views: list[str] = []
    if attention_enabled:
        oof_needed_views.extend(v for v in ("graph", "stat") if v in args.views)
    if calibration_enabled:
        for v in args.views:
            if v not in oof_needed_views:
                oof_needed_views.append(v)
    if oof_needed_views:
        print()
        print(f"Pass 0.5: computing OOF source predictions "
              f"(views={oof_needed_views}, cv={calib_cv_folds})...")
        for classifier in args.classifiers:
            for view in oof_needed_views:
                for alignment in args.alignments:
                    hp = tuned_params.get((classifier, view, alignment))
                    for seed in args.seeds:
                        oof = _compute_oof_source_scores(
                            classifier, view, source_cache[alignment],
                            seed, params=hp, cv_folds=calib_cv_folds,
                        )
                        score_cache_src_oof[(classifier, view, alignment, seed)] = oof

    if calibration_enabled:
        print()
        print(f"Pass 0.75: fitting {calib_method}+{tau_policy} calibrators on source OOF...")
        for classifier in args.classifiers:
            for view in args.views:
                for alignment in args.alignments:
                    y_src_full = source_cache[alignment]["binary_label"].to_numpy(dtype=np.int32)
                    for seed in args.seeds:
                        oof = score_cache_src_oof.get((classifier, view, alignment, seed))
                        if oof is None:
                            continue
                        calib = _fit_calibrator(
                            oof, y_src_full,
                            method=calib_method, tau_policy=tau_policy,
                        )
                        if calib is None:
                            continue
                        calibrator_cache[(classifier, view, alignment, seed)] = calib
                        log_entry: dict = {
                            "method": calib.method,
                            "tau_policy": calib.tau_policy,
                            "tau": float(calib.tau),
                        }
                        if calib.iso is not None:
                            log_entry["iso_X"] = [float(x) for x in calib.iso.X_thresholds_]
                            log_entry["iso_Y"] = [float(y) for y in calib.iso.y_thresholds_]
                        if calib.platt is not None:
                            log_entry["platt_coef"] = float(calib.platt.coef_[0, 0])
                            log_entry["platt_intercept"] = float(calib.platt.intercept_[0])
                        calibration_log.setdefault(classifier, {}) \
                            .setdefault(view, {}) \
                            .setdefault(alignment, {})[str(seed)] = log_entry
        print(f"  fitted {len(calibrator_cache)} calibrators "
              f"out of {len(args.classifiers) * len(args.views) * len(args.alignments) * len(args.seeds)} configs")

    # --- Pass 1: base views --------------------------------------------------
    for classifier in args.classifiers:
        for view in args.views:
            for alignment in args.alignments:
                hp = tuned_params.get((classifier, view, alignment))
                for seed in args.seeds:
                    calib = calibrator_cache.get((classifier, view, alignment, seed))
                    rm, em, y_tgt, _y_src_in_sample, y_tgt_raw = _run_single_config(
                        view, alignment, classifier, seed,
                        source_cache[alignment], target_df, event_ids, predictions_dir,
                        tuned_params=hp, calibrator=calib,
                    )
                    all_row.append(rm)
                    all_event.append(em)
                    if view in ("graph", "stat"):
                        score_cache_tgt[(classifier, view, alignment, seed)] = y_tgt
                        score_cache_tgt_raw[(classifier, view, alignment, seed)] = y_tgt_raw
                    cal_tag = f" cal(τ={calib.tau:.3f})" if calib is not None else ""
                    print(f"  {classifier:<14} | {view:<18} | {alignment:<12} | seed={seed}  "
                          f"macro_f1={rm['macro_f1']:.4f}  rec_anom={rm['recall_anomaly']:.4f}  "
                          f"pr_auc={rm['pr_auc_anomaly']:.4f}  ({rm['fit_seconds']:.1f}s){cal_tag}")

    # --- Pass 2: derived late-fusion views (mean / OR; no re-training) -------
    # When calibration is enabled:
    #   - fusion_late_mean: fit a second-stage calibrator on (oof_g_cal + oof_s_cal)/2 vs y_src.
    #   - fusion_late_or:   threshold each calibrated base-view score at its own τ; hard OR.
    if late_fusion_enabled:
        print()
        print("Deriving late-fusion (mean, OR) from cached base-view scores...")
        for classifier in args.classifiers:
            for alignment in args.alignments:
                for seed in args.seeds:
                    key_g = (classifier, "graph", alignment, seed)
                    key_s = (classifier, "stat", alignment, seed)
                    if key_g not in score_cache_tgt or key_s not in score_cache_tgt:
                        continue
                    calib_g = calibrator_cache.get(key_g)
                    calib_s = calibrator_cache.get(key_s)
                    # All-or-none: only use the calibrated path when BOTH base
                    # views have a fitted calibrator for this seed; otherwise
                    # fall back to raw target probs + 0.5 thresholds so the
                    # two views share one scale.
                    both_calibrated = (
                        calibration_enabled
                        and calib_g is not None
                        and calib_s is not None
                    )
                    if both_calibrated:
                        p_g = score_cache_tgt[key_g]
                        p_s = score_cache_tgt[key_s]
                        tau_g = calib_g.tau
                        tau_s = calib_s.tau
                    else:
                        p_g = score_cache_tgt_raw[key_g]
                        p_s = score_cache_tgt_raw[key_s]
                        tau_g = None
                        tau_s = None
                    mean_calibrator: Calibrator | None = None
                    if both_calibrated:
                        oof_g = score_cache_src_oof.get(key_g)
                        oof_s = score_cache_src_oof.get(key_s)
                        if oof_g is not None and oof_s is not None:
                            mask = ~(np.isnan(oof_g) | np.isnan(oof_s))
                            if mask.sum() >= 5:
                                y_src_full = source_cache[alignment]["binary_label"].to_numpy(dtype=np.int32)
                                mean_raw_oof = (calib_g.apply(oof_g[mask]) + calib_s.apply(oof_s[mask])) / 2.0
                                mean_calibrator = _fit_calibrator(
                                    mean_raw_oof, y_src_full[mask],
                                    method=calib_method, tau_policy=tau_policy,
                                )
                                if mean_calibrator is not None:
                                    m_entry: dict = {
                                        "method": mean_calibrator.method,
                                        "tau_policy": mean_calibrator.tau_policy,
                                        "tau": float(mean_calibrator.tau),
                                    }
                                    if mean_calibrator.iso is not None:
                                        m_entry["iso_X"] = [float(x) for x in mean_calibrator.iso.X_thresholds_]
                                        m_entry["iso_Y"] = [float(y) for y in mean_calibrator.iso.y_thresholds_]
                                    if mean_calibrator.platt is not None:
                                        m_entry["platt_coef"] = float(mean_calibrator.platt.coef_[0, 0])
                                        m_entry["platt_intercept"] = float(mean_calibrator.platt.intercept_[0])
                                    calibration_log.setdefault(classifier, {}) \
                                        .setdefault("fusion_late_mean", {}) \
                                        .setdefault(alignment, {})[str(seed)] = m_entry
                    for rule in LATE_VIEWS:
                        rm, em = _run_late_fusion(
                            rule, alignment, classifier, seed,
                            p_g, p_s, target_df, event_ids, predictions_dir,
                            mean_calibrator=mean_calibrator if rule == "fusion_late_mean" else None,
                            tau_graph=tau_g if rule == "fusion_late_or" else None,
                            tau_stat=tau_s if rule == "fusion_late_or" else None,
                        )
                        all_row.append(rm)
                        all_event.append(em)
                        tag = ""
                        if rule == "fusion_late_mean" and mean_calibrator is not None:
                            tag = f" cal(τ={mean_calibrator.tau:.3f})"
                        elif rule == "fusion_late_or" and (tau_g is not None or tau_s is not None):
                            tag = f" τg={tau_g or 0.5:.3f} τs={tau_s or 0.5:.3f}"
                        print(f"  {classifier:<14} | {rule:<18} | {alignment:<12} | seed={seed}  "
                              f"macro_f1={rm['macro_f1']:.4f}  rec_anom={rm['recall_anomaly']:.4f}  "
                              f"pr_auc={rm['pr_auc_anomaly']:.4f}  (derived){tag}")

    # --- Pass 3: learned attention fusion (logistic stacker on OOF source) --
    # Stacker is fitted on calibrated OOF source predictions when calibration
    # is enabled (so base views and stacker see the same probability scale),
    # otherwise raw OOF. A τ sweep on the stacker's source output picks the
    # decision threshold when calibrate_stacker=True.
    attention_log: dict = {}
    if attention_enabled:
        print()
        print("Fitting attention fusion (logistic stacker on OOF source predictions)...")
        for classifier in args.classifiers:
            for alignment in args.alignments:
                for seed in args.seeds:
                    key_g = (classifier, "graph", alignment, seed)
                    key_s = (classifier, "stat", alignment, seed)
                    if (key_g not in score_cache_tgt or key_s not in score_cache_tgt
                            or key_g not in score_cache_src_oof or key_s not in score_cache_src_oof):
                        continue
                    y_src_full = source_cache[alignment]["binary_label"].to_numpy(dtype=np.int32)
                    oof_g = score_cache_src_oof[key_g]
                    oof_s = score_cache_src_oof[key_s]
                    mask = ~(np.isnan(oof_g) | np.isnan(oof_s))
                    if mask.sum() < 2 or len(np.unique(y_src_full[mask])) < 2:
                        continue
                    calib_g = calibrator_cache.get(key_g)
                    calib_s = calibrator_cache.get(key_s)
                    # All-or-none: stacker source features and target features
                    # must share a single scale. Only take the calibrated path
                    # when BOTH base-view calibrators exist for this seed.
                    both_calibrated = (
                        calibration_enabled
                        and calib_g is not None
                        and calib_s is not None
                    )
                    if both_calibrated:
                        src_g_in = calib_g.apply(oof_g[mask])
                        src_s_in = calib_s.apply(oof_s[mask])
                        tgt_g_in = score_cache_tgt[key_g]
                        tgt_s_in = score_cache_tgt[key_s]
                    else:
                        src_g_in = oof_g[mask]
                        src_s_in = oof_s[mask]
                        tgt_g_in = score_cache_tgt_raw[key_g]
                        tgt_s_in = score_cache_tgt_raw[key_s]
                    rm, em, weights, stacker_tau = _run_attention_fusion(
                        alignment, classifier, seed,
                        src_g_in, src_s_in, y_src_full[mask],
                        tgt_g_in, tgt_s_in,
                        target_df, event_ids, predictions_dir,
                        calibrate_stacker=both_calibrated,
                    )
                    all_row.append(rm)
                    all_event.append(em)
                    attention_log.setdefault(classifier, {}).setdefault(alignment, {})[str(seed)] = weights
                    if both_calibrated:
                        calibration_log.setdefault(classifier, {}) \
                            .setdefault(ATTENTION_VIEW, {}) \
                            .setdefault(alignment, {})[str(seed)] = {
                                "method": calib_method,
                                "tau_policy": tau_policy,
                                "tau": float(stacker_tau),
                            }
                    cal_tag = f" τ={stacker_tau:.3f}" if both_calibrated else ""
                    print(f"  {classifier:<14} | {ATTENTION_VIEW:<18} | {alignment:<12} | seed={seed}  "
                          f"macro_f1={rm['macro_f1']:.4f}  rec_anom={rm['recall_anomaly']:.4f}  "
                          f"pr_auc={rm['pr_auc_anomaly']:.4f}  "
                          f"w=(g={weights['w_graph']:+.3f}, s={weights['w_stat']:+.3f}, b={weights['intercept']:+.3f}){cal_tag}")
        (output_root / "attention_fusion_weights.json").write_text(
            json.dumps(attention_log, indent=2)
        )

    if calibration_enabled:
        (output_root / "calibration_log.json").write_text(
            json.dumps(calibration_log, indent=2)
        )

    row_df = pd.DataFrame(all_row)
    event_df = pd.concat(all_event, ignore_index=True)
    row_df.to_csv(output_root / "metrics_row_level.csv", index=False)
    event_df.to_csv(output_root / "metrics_per_event.csv", index=False)
    (output_root / "run_config.json").write_text(json.dumps({
        "protocol": "I1 zero-shot", "source_domain": src_dom,
        "target_domain": tgt_dom, "coral_kind": args.coral_kind,
        "coral_artifact_root": str(coral_root),
        "coral_artifact_metrics": {
            k: coral_cfg[k] for k in (
                "cov_gap_before", "cov_gap_after", "cov_gap_reduction_pct",
                "mmd_before", "mmd_after", "mmd_reduction_pct",
                "n_source", "n_target", "balance_fit",
            ) if k in coral_cfg
        },
        "classifiers": list(args.classifiers),
        "views": list(args.views),
        "late_views": list(LATE_VIEWS) if late_fusion_enabled else [],
        "attention_view": ATTENTION_VIEW if attention_enabled else None,
        "late_fusion_enabled": bool(late_fusion_enabled),
        "attention_fusion_enabled": bool(attention_enabled),
        "tuning_enabled": bool(tuning_enabled),
        "tuning_cv_folds": int(args.tuning_cv_folds) if tuning_enabled else None,
        "calibration_enabled": bool(calibration_enabled),
        "calibration_cv_folds": int(calib_cv_folds) if calibration_enabled else None,
        "calibration_method": calib_method if calibration_enabled else None,
        "tau_policy": tau_policy if calibration_enabled else None,
        "n_calibrators_fitted": int(len(calibrator_cache)) if calibration_enabled else 0,
        "alignments": list(args.alignments),
        "seeds": args.seeds, "n_target_rows": len(target_df),
        "n_study_events": int(n_events), "primary_metric": "macro_f1",
        "torch_available": bool(TORCH_AVAILABLE),
    }, indent=2))

    print("\n" + "=" * 72)
    print("Summary (mean over seeds):")
    print("=" * 72)
    summary = row_df.groupby(["classifier", "view", "alignment"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        recall_anomaly_mean=("recall_anomaly", "mean"),
        pr_auc_mean=("pr_auc_anomaly", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    print(f"\nWrote: {output_root}")


# ============================================================================
# 4. ANALYZE
# ============================================================================
def _per_event_matrix(event_df: pd.DataFrame, metric: str = "recall",
                      classifier: str | None = None) -> pd.DataFrame:
    """Pivot event-level metrics into a wide (event_id × (view, alignment)) table.

    If ``classifier`` is given, slice the event_df to that classifier first.
    Otherwise, pivot over all rows (useful when only one classifier is present).
    """
    df = event_df if classifier is None else event_df[event_df["classifier"] == classifier]
    mean = df.groupby(["event_id", "view", "alignment"])[metric].mean().reset_index()
    return mean.pivot_table(index="event_id", columns=["view", "alignment"], values=metric, aggfunc="first")


def _paired_wilcoxon(a, b, label_a, label_b) -> dict:
    # Fix 3: filter NaN / non-finite pairs before computing diffs
    finite_mask = np.isfinite(a) & np.isfinite(b)
    a_clean = a[finite_mask]
    b_clean = b[finite_mask]
    n_dropped = int((~finite_mask).sum())
    diffs = a_clean - b_clean
    n_nz = int((diffs != 0).sum())
    result = {
        "a": label_a, "b": label_b,
        "n_events_total": int(len(a)),
        "n_events_retained": int(len(a_clean)),
        "n_dropped_nan": n_dropped,
        "n_nonzero_pairs": n_nz,
        "mean_a": float(np.mean(a_clean)) if len(a_clean) else float("nan"),
        "mean_b": float(np.mean(b_clean)) if len(b_clean) else float("nan"),
        "mean_diff": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "median_diff": float(np.median(diffs)) if len(diffs) else float("nan"),
        "n_a_gt_b": int((diffs > 0).sum()), "n_b_gt_a": int((diffs < 0).sum()),
        "n_tied": int((diffs == 0).sum()),
    }
    if n_nz < 1:
        result.update({"statistic": float("nan"), "p_value": float("nan"), "note": "all tied or empty"})
        return result
    stat, p = wilcoxon(a_clean, b_clean, zero_method="wilcox", alternative="two-sided")
    result.update({"statistic": float(stat), "p_value": float(p)})
    return result


def _print_test_block(title: str, tests: list[dict]) -> None:
    if not tests:
        return
    print(title)
    for t in tests:
        p = t["p_value"]
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        n_ret = t.get("n_events_retained", "?")
        print(f"  {t['a']:<40} vs {t['b']:<40}  "
              f"diff={t['mean_diff']:+.4f}  p={p:.4g}{sig}  (n={n_ret})")
    print()


def _analyze_classifier(event_df: pd.DataFrame, row_df: pd.DataFrame,
                        classifier: str, primary_alignment: str) -> dict:
    event_wide = _per_event_matrix(event_df, classifier=classifier)
    row_clf = row_df[row_df["classifier"] == classifier]

    # Post-hoc best alignment per view (exploratory reporting only).
    best = {}
    means = row_clf.groupby(["view", "alignment"])["macro_f1"].mean().reset_index()
    for v, g in means.groupby("view"):
        best[str(v)] = str(g.loc[g["macro_f1"].idxmax(), "alignment"])

    # Within-view: after_coral vs before_coral (fixed a priori).
    within: list[dict] = []
    for view in sorted({v for v, _ in event_wide.columns}):
        try:
            a = event_wide[(view, "after_coral")].to_numpy()
            b = event_wide[(view, "before_coral")].to_numpy()
        except KeyError:
            continue
        within.append(_paired_wilcoxon(a, b, f"{view}/after_coral", f"{view}/before_coral"))

    # Cross-view confirmatory: fusion_early vs graph/stat at primary alignment.
    cross_confirmatory: list[dict] = []
    if ("fusion_early", primary_alignment) in event_wide.columns:
        fe = event_wide[("fusion_early", primary_alignment)].to_numpy()
        for other in ("graph", "stat"):
            if (other, primary_alignment) not in event_wide.columns:
                continue
            cross_confirmatory.append(_paired_wilcoxon(
                fe, event_wide[(other, primary_alignment)].to_numpy(),
                f"fusion_early/{primary_alignment}", f"{other}/{primary_alignment}",
            ))

    # Cross-view exploratory (post-hoc best alignment; optimistically biased).
    cross_exploratory: list[dict] = []
    if "fusion_early" in best:
        fe_al = best["fusion_early"]
        fe = event_wide[("fusion_early", fe_al)].to_numpy()
        for other in ("graph", "stat"):
            if other not in best:
                continue
            o_al = best[other]
            cross_exploratory.append(_paired_wilcoxon(
                fe, event_wide[(other, o_al)].to_numpy(),
                f"fusion_early/{fe_al}", f"{other}/{o_al}",
            ))

    # Fusion-strategy comparison at primary alignment (confirmatory):
    # pair each late-fusion rule against fusion_early on matched events.
    fusion_strategy: list[dict] = []
    if ("fusion_early", primary_alignment) in event_wide.columns:
        fe = event_wide[("fusion_early", primary_alignment)].to_numpy()
        for rule in ("fusion_late_mean", "fusion_late_or"):
            if (rule, primary_alignment) not in event_wide.columns:
                continue
            fusion_strategy.append(_paired_wilcoxon(
                fe, event_wide[(rule, primary_alignment)].to_numpy(),
                f"fusion_early/{primary_alignment}", f"{rule}/{primary_alignment}",
            ))

    return {
        "event_wide": event_wide,
        "best_alignment": best,
        "within": within,
        "cross_confirmatory": cross_confirmatory,
        "cross_exploratory": cross_exploratory,
        "fusion_strategy": fusion_strategy,
    }


def _wilcoxon_classifiers(event_df: pd.DataFrame, classifiers: list[str],
                          primary_alignment: str) -> list[dict]:
    """Classifier comparison per view at the primary alignment."""
    if len(classifiers) < 2:
        return []
    tests: list[dict] = []
    # Pairwise across classifiers, per view.
    for i, ca in enumerate(classifiers):
        for cb in classifiers[i + 1:]:
            wide_a = _per_event_matrix(event_df, classifier=ca)
            wide_b = _per_event_matrix(event_df, classifier=cb)
            common_views = sorted({v for v, _ in wide_a.columns} & {v for v, _ in wide_b.columns})
            for view in common_views:
                if (view, primary_alignment) not in wide_a.columns:
                    continue
                if (view, primary_alignment) not in wide_b.columns:
                    continue
                a = wide_a[(view, primary_alignment)]
                b = wide_b[(view, primary_alignment)]
                # Align on event_id (inner join) to ensure paired events.
                joined = pd.concat([a, b], axis=1, join="inner").dropna()
                if joined.empty:
                    continue
                tests.append(_paired_wilcoxon(
                    joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy(),
                    f"{ca}/{view}/{primary_alignment}", f"{cb}/{view}/{primary_alignment}",
                ))
    return tests


def cmd_analyze(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    row_df = pd.read_csv(input_dir / "metrics_row_level.csv")
    event_df = pd.read_csv(input_dir / "metrics_per_event.csv")
    primary_alignment = args.primary_alignment

    classifiers = sorted(row_df["classifier"].unique().tolist())
    views_present = sorted(row_df["view"].unique().tolist())
    has_late = any(v.startswith("fusion_late_") for v in views_present)

    print("=" * 72)
    print(f"Analyzer -- {input_dir}")
    print("=" * 72)
    print(f"rows: {len(row_df)}  |  events: {event_df['event_id'].nunique()}  |  "
          f"classifiers: {classifiers}  |  late-fusion: {'on' if has_late else 'off'}")
    print()

    # Row-level summary (classifier x view x alignment).
    agg = {m: ["mean", "std"] for m in AGG_METRICS}
    summary = row_df.groupby(["classifier", "view", "alignment"]).agg(agg)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    print("Row-level metrics (mean +/- std):")
    for _, r in summary.iterrows():
        parts = [f"{r['classifier']:<14} {r['view']:<18} {r['alignment']:<14}"]
        for m in AGG_METRICS:
            parts.append(f"{m}={r[f'{m}_mean']:.4f}+/-{r[f'{m}_std']:.4f}")
        print("  " + "  ".join(parts))
    print()

    # Event detection rate (classifier x view x alignment).
    det = (
        event_df.assign(d=lambda d: d["detected"].astype(int))
        .groupby(["classifier", "view", "alignment", "seed"])["d"].mean().reset_index()
        .groupby(["classifier", "view", "alignment"])["d"].agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "detect_rate_mean", "std": "detect_rate_std"})
    )
    print("Event detection rate:")
    print(det.to_string(index=False))
    print()

    out = input_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "row_level_summary.csv", index=False)
    det.to_csv(out / "event_detection_summary.csv", index=False)

    per_classifier_payload: dict[str, dict] = {}
    for clf in classifiers:
        print("=" * 72)
        print(f"Classifier: {clf}")
        print("=" * 72)
        r = _analyze_classifier(event_df, row_df, clf, primary_alignment)
        print(f"Best alignment per view: {r['best_alignment']}")
        print()
        _print_test_block("Within-view: after_coral(A) vs before_coral(B)", r["within"])
        _print_test_block(
            f"Cross-view CONFIRMATORY: fusion_early vs single (both @ {primary_alignment})",
            r["cross_confirmatory"],
        )
        _print_test_block(
            "Cross-view EXPLORATORY (post-hoc best; optimistically biased)",
            r["cross_exploratory"],
        )
        _print_test_block(
            f"Fusion strategy CONFIRMATORY: fusion_early vs late rules (@ {primary_alignment})",
            r["fusion_strategy"],
        )
        r["event_wide"].to_csv(out / f"per_event_recall_wide__{clf}.csv")
        per_classifier_payload[clf] = {
            "best_alignment_per_view": r["best_alignment"],
            "within_view": r["within"],
            "cross_view_confirmatory": {
                "primary_alignment": primary_alignment,
                "tests": r["cross_confirmatory"],
            },
            "cross_view_exploratory": {
                "warning": "post-hoc best alignment selected from target macro-F1; "
                           "p-values are optimistically biased -- exploratory only.",
                "tests": r["cross_exploratory"],
            },
            "fusion_strategy_confirmatory": {
                "primary_alignment": primary_alignment,
                "tests": r["fusion_strategy"],
            },
        }

    # Classifier comparison (only if >1 classifier).
    classifier_tests = _wilcoxon_classifiers(event_df, classifiers, primary_alignment)
    if classifier_tests:
        print("=" * 72)
        print("Classifier comparison (pairwise, per view, @ primary alignment)")
        print("=" * 72)
        _print_test_block(
            f"Classifiers at {primary_alignment}", classifier_tests,
        )

    (out / "wilcoxon_by_classifier.json").write_text(json.dumps(per_classifier_payload, indent=2))
    (out / "wilcoxon_classifier_comparison.json").write_text(json.dumps({
        "primary_alignment": primary_alignment,
        "tests": classifier_tests,
    }, indent=2))
    print(f"Wrote: {out}")


# ============================================================================
# 5. DIAGNOSTICS
# ============================================================================
def _quick_eval(X_tr, y_tr, X_te, y_te, seed=0):
    model = fit_xgboost(X_tr, y_tr, seed)
    y_score = model.predict_proba(X_te)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    mf1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    tp = int(((y_pred == 1) & (y_te == 1)).sum())
    fn = int(((y_pred == 0) & (y_te == 1)).sum())
    fp = int(((y_pred == 1) & (y_te == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {"macro_f1": mf1, "recall_anomaly": rec, "precision_anomaly": prec}


def _diag1_same_collector() -> dict:
    print("=" * 72)
    print("Diagnostic 1: D1→D2 same-collector (RRC04)")
    print("=" * 72)
    d1 = load_study("rrc04_as12880")
    d2 = load_study("rrc04_as3352")
    feat = list(SHARED22_STAT_VIEW)
    y_src = d1["binary_label"].to_numpy(dtype=int)
    y_tgt = d2["binary_label"].to_numpy(dtype=int)
    src_s, tgt_s, _ = prepare_numeric_matrices(d1, d2, feat)
    aligned, _ = coral_align(src_s, tgt_s, reg=1e-6)

    before = _quick_eval(src_s, y_src, tgt_s, y_tgt)
    after = _quick_eval(aligned, y_src, tgt_s, y_tgt)
    print(f"  before_coral: macro_f1={before['macro_f1']:.4f}  rec={before['recall_anomaly']:.4f}")
    print(f"  after_coral:  macro_f1={after['macro_f1']:.4f}  rec={after['recall_anomaly']:.4f}")
    print(f"  CORAL helped: {after['macro_f1'] > before['macro_f1']}")
    return {"test": "D1→D2 same-collector", "before_coral": before, "after_coral": after,
            "coral_helped": after["macro_f1"] > before["macro_f1"]}


def _diag2_pseudo_label_coral() -> dict:
    print()
    print("=" * 72)
    print("Diagnostic 2: Pseudo-label conditional CORAL (D2→D4)")
    print("=" * 72)
    d2 = load_study(SOURCE_DOMAIN)
    d4 = load_study(TARGET_DOMAIN)
    feat = list(SHARED22_STAT_VIEW)
    y_src = d2["binary_label"].to_numpy(dtype=int)
    y_tgt = d4["binary_label"].to_numpy(dtype=int)
    src_s, tgt_s, _ = prepare_numeric_matrices(d2, d4, feat)

    # Step 1: pre-alignment pseudo-labels
    pre_model = fit_xgboost(src_s, y_src, seed=0)
    tgt_scores = pre_model.predict_proba(tgt_s)[:, 1]
    lo, hi = 0.15, 0.85
    pn_mask = tgt_scores < lo
    pa_mask = tgt_scores > hi
    n_pn, n_pa = int(pn_mask.sum()), int(pa_mask.sum())
    print(f"  pseudo-labels: {n_pn} normal (score<{lo}), {n_pa} anomaly (score>{hi})")

    if n_pn < 10 or n_pa < 10:
        lo, hi = 0.25, 0.75
        pn_mask = tgt_scores < lo
        pa_mask = tgt_scores > hi
        n_pn, n_pa = int(pn_mask.sum()), int(pa_mask.sum())
        print(f"  relaxed: {n_pn} normal, {n_pa} anomaly")

    if n_pn < 5 or n_pa < 5:
        msg = f"too few pseudo-labels ({n_pn} norm, {n_pa} anom)"
        print(f"  ABORT: {msg}")
        return {"test": "pseudo-label conditional CORAL", "error": msg}

    # Step 2: class-conditional CORAL
    src_norm_mask = y_src == 0
    src_anom_mask = y_src == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        aligned_norm, _ = coral_align(src_s[src_norm_mask], tgt_s[pn_mask], reg=1e-6)
        aligned_anom, _ = coral_align(src_s[src_anom_mask], tgt_s[pa_mask], reg=1e-6)

    src_cond = np.empty_like(src_s)
    src_cond[src_norm_mask] = aligned_norm
    src_cond[src_anom_mask] = aligned_anom

    vanilla_aligned, _ = coral_align(src_s, tgt_s, reg=1e-6)

    before = _quick_eval(src_s, y_src, tgt_s, y_tgt)
    vanilla = _quick_eval(vanilla_aligned, y_src, tgt_s, y_tgt)
    conditional = _quick_eval(src_cond, y_src, tgt_s, y_tgt)

    print(f"  before_coral:      macro_f1={before['macro_f1']:.4f}  rec={before['recall_anomaly']:.4f}")
    print(f"  vanilla_coral:     macro_f1={vanilla['macro_f1']:.4f}  rec={vanilla['recall_anomaly']:.4f}")
    print(f"  conditional_coral: macro_f1={conditional['macro_f1']:.4f}  rec={conditional['recall_anomaly']:.4f}")
    print(f"  Conditional beat vanilla: {conditional['macro_f1'] > vanilla['macro_f1']}")
    print(f"  Conditional beat before:  {conditional['macro_f1'] > before['macro_f1']}")

    return {
        "test": "pseudo-label conditional CORAL (D2→D4)",
        "before_coral": before, "vanilla_coral": vanilla,
        "conditional_coral": conditional,
        "conditional_beat_vanilla": conditional["macro_f1"] > vanilla["macro_f1"],
        "conditional_beat_before": conditional["macro_f1"] > before["macro_f1"],
    }


def cmd_diagnostics(args: argparse.Namespace) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    d1 = _diag1_same_collector()
    (DIAGNOSTICS_DIR / "diagnostic1_same_collector.json").write_text(json.dumps(d1, indent=2))
    d2 = _diag2_pseudo_label_coral()
    (DIAGNOSTICS_DIR / "diagnostic2_pseudo_label_coral.json").write_text(json.dumps(d2, indent=2))
    print(f"\nWrote: {DIAGNOSTICS_DIR}")


# ============================================================================
# 6. ALL (manifest → coral → experiment → analyze)
# ============================================================================
def cmd_all(args: argparse.Namespace) -> None:
    # Derive coral_kind from source_kind + balance_fit so it stays consistent
    kind_label = derive_kind_label(args.source_kind, args.balance_fit)
    args.coral_kind = kind_label

    print("\n>>> STAGE: manifest")
    cmd_manifest(args)
    print("\n>>> STAGE: coral")
    cmd_coral(args)
    print("\n>>> STAGE: experiment")
    cmd_experiment(args)
    print("\n>>> STAGE: analyze")
    default_prefix, _, _, _ = _derive_stage_prefix(
        args.classifiers, args.views,
        enable_late_fusion=args.enable_late_fusion,
        enable_attention_fusion=getattr(args, "enable_attention_fusion", False),
        enable_tuning=getattr(args, "enable_tuning", False),
    )
    output_suffix = args.output_suffix or f"{default_prefix}_{kind_label}"
    args.input_dir = FUSION_OUTPUT_ROOT / output_suffix
    cmd_analyze(args)


# ============================================================================
# CLI
# ============================================================================
def _positive_int_ge(minimum: int):
    """argparse type validator: int >= ``minimum``."""
    def _check(value: str) -> int:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"expected int, got {value!r}")
        if iv < minimum:
            raise argparse.ArgumentTypeError(
                f"must be >= {minimum}, got {iv}"
            )
        return iv
    return _check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase3_pipeline",
        description="Phase 3 unified pipeline for multi-view fusion experiments.",
    )
    parser.add_argument("--feature-set", choices=tuple(_FEATURE_SETS.keys()),
                        default="shared22",
                        help="Feature-space for CORAL + experiment. "
                             "'shared22' (default, 22 features, graph+stat+fusion views) "
                             "or 'core10' (10 features, single flat view — no graph/stat "
                             "sub-views, so no late-fusion/attention-fusion views).")
    sub = parser.add_subparsers(dest="command", required=True)

    # manifest
    p_man = sub.add_parser("manifest", help="Build event manifest (v2)")
    p_man.add_argument("--max-gap-minutes", type=int, default=DEFAULT_MAX_GAP_MINUTES)

    # coral
    p_cor = sub.add_parser("coral", help="Re-fit CORAL on unified target")
    p_cor.add_argument("--source-kind", choices=("study_only", "study_plus_historical"), default="study_only")
    p_cor.add_argument("--balance-fit", choices=("none", "equal"), default="none")
    p_cor.add_argument("--source-domain", default=SOURCE_DOMAIN)
    p_cor.add_argument("--target-domain", default=TARGET_DOMAIN)
    p_cor.add_argument("--reg", type=float, default=1e-6)

    # experiment
    p_exp = sub.add_parser("experiment", help="Run fusion experiment")
    p_exp.add_argument("--coral-kind", default="study_only")
    p_exp.add_argument("--source-domain", default=SOURCE_DOMAIN)
    p_exp.add_argument("--target-domain", default=TARGET_DOMAIN)
    p_exp.add_argument("--output-suffix", default=None)
    p_exp.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p_exp.add_argument("--views", nargs="+", default=list(VIEW_CONFIGS.keys()),
                       choices=list(VIEW_CONFIGS.keys()))
    p_exp.add_argument("--alignments", nargs="+", default=list(ALIGNMENTS),
                       choices=list(ALIGNMENTS))
    p_exp.add_argument("--classifiers", nargs="+",
                       default=list(("xgboost", "random_forest")),
                       choices=list(CLASSIFIERS))
    p_exp.add_argument("--enable-late-fusion", dest="enable_late_fusion",
                       action="store_true", default=True)
    p_exp.add_argument("--disable-late-fusion", dest="enable_late_fusion",
                       action="store_false")
    p_exp.add_argument("--enable-attention-fusion", dest="enable_attention_fusion",
                       action="store_true", default=False,
                       help="Pass 3 stage-3 learned-fusion stacker on graph+stat probs.")
    p_exp.add_argument("--enable-tuning", dest="enable_tuning",
                       action="store_true", default=False,
                       help="Source-only stratified k-fold CV per (classifier, view, alignment).")
    p_exp.add_argument("--tuning-cv-folds", type=_positive_int_ge(2), default=5,
                       help="Source-only CV fold count for tuning (min 2).")
    p_exp.add_argument("--enable-calibration", dest="enable_calibration",
                       action="store_true", default=False,
                       help="Post-hoc isotonic calibration + macro-F1 threshold sweep "
                            "on source OOF predictions (base views, late_mean, attention).")
    p_exp.add_argument("--calibration-cv-folds", type=_positive_int_ge(2), default=5,
                       help="Source-only CV fold count for OOF used by calibration "
                            "and/or attention stacker (min 2).")
    p_exp.add_argument("--calibration-method", choices=("isotonic", "platt"),
                       default="isotonic",
                       help="Probability scaler: isotonic (non-parametric, clips at "
                            "train range) or platt (1-parameter sigmoid, extrapolates).")
    p_exp.add_argument("--tau-policy", choices=("f1_sweep", "prior_quantile"),
                       default="f1_sweep",
                       help="Threshold policy: f1_sweep (max macro-F1 on source OOF) "
                            "or prior_quantile (match anomaly prior on target scores).")
    p_exp.add_argument("--source-labels-from", dest="source_labels_from",
                       type=Path, default=None,
                       help="Override source binary_label from a relabel CSV "
                            "(joined on window_start). Drops rows where the "
                            "relabel CSV's discovered_label == 'uncertain'. "
                            "After-CORAL only.")

    # analyze
    p_ana = sub.add_parser("analyze", help="Analyze results with Wilcoxon tests")
    p_ana.add_argument("--input-dir", type=Path,
                       default=FUSION_OUTPUT_ROOT / "stage1_minimal_study_only")
    p_ana.add_argument("--primary-alignment", choices=list(ALIGNMENTS),
                       default="before_coral",
                       help="Pre-committed alignment for confirmatory cross-view tests "
                            "(default: before_coral — the Phase 3 primary condition)")

    # diagnostics
    sub.add_parser("diagnostics", help="CORAL diagnostic experiments")

    # all
    p_all = sub.add_parser("all", help="Run manifest → coral → experiment → analyze")
    p_all.add_argument("--max-gap-minutes", type=int, default=DEFAULT_MAX_GAP_MINUTES)
    p_all.add_argument("--source-kind", choices=("study_only", "study_plus_historical"), default="study_only")
    p_all.add_argument("--balance-fit", choices=("none", "equal"), default="none")
    p_all.add_argument("--source-domain", default=SOURCE_DOMAIN)
    p_all.add_argument("--target-domain", default=TARGET_DOMAIN)
    p_all.add_argument("--reg", type=float, default=1e-6)
    p_all.add_argument("--output-suffix", default=None)
    p_all.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p_all.add_argument("--views", nargs="+", default=list(VIEW_CONFIGS.keys()),
                       choices=list(VIEW_CONFIGS.keys()))
    p_all.add_argument("--alignments", nargs="+", default=list(ALIGNMENTS),
                       choices=list(ALIGNMENTS))
    p_all.add_argument("--primary-alignment", choices=list(ALIGNMENTS),
                       default="before_coral",
                       help="Pre-committed alignment for confirmatory cross-view tests")
    p_all.add_argument("--classifiers", nargs="+",
                       default=list(("xgboost", "random_forest")),
                       choices=list(CLASSIFIERS))
    p_all.add_argument("--enable-late-fusion", dest="enable_late_fusion",
                       action="store_true", default=True)
    p_all.add_argument("--disable-late-fusion", dest="enable_late_fusion",
                       action="store_false")
    p_all.add_argument("--enable-attention-fusion", dest="enable_attention_fusion",
                       action="store_true", default=False)
    p_all.add_argument("--enable-tuning", dest="enable_tuning",
                       action="store_true", default=False)
    p_all.add_argument("--tuning-cv-folds", type=_positive_int_ge(2), default=5,
                       help="Source-only CV fold count for tuning (min 2).")
    p_all.add_argument("--enable-calibration", dest="enable_calibration",
                       action="store_true", default=False,
                       help="Post-hoc isotonic calibration + macro-F1 threshold sweep "
                            "on source OOF predictions (base views, late_mean, attention).")
    p_all.add_argument("--calibration-cv-folds", type=_positive_int_ge(2), default=5,
                       help="Source-only CV fold count for OOF used by calibration "
                            "and/or attention stacker (min 2).")
    p_all.add_argument("--calibration-method", choices=("isotonic", "platt"),
                       default="isotonic")
    p_all.add_argument("--tau-policy", choices=("f1_sweep", "prior_quantile"),
                       default="f1_sweep")

    return parser


def main() -> None:
    # Two-pass parse so --feature-set swaps VIEW_CONFIGS before build_parser()
    # bakes `choices=list(VIEW_CONFIGS.keys())` into the --views argument.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--feature-set", choices=tuple(_FEATURE_SETS.keys()),
                            default="shared22")
    pre_args, _ = pre_parser.parse_known_args()
    set_active_feature_set(pre_args.feature_set)

    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "manifest": cmd_manifest,
        "coral": cmd_coral,
        "experiment": cmd_experiment,
        "analyze": cmd_analyze,
        "diagnostics": cmd_diagnostics,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
