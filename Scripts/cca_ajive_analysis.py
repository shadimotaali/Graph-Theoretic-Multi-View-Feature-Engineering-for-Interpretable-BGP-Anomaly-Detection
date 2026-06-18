#!/usr/bin/env python3
"""
Cross-view complementarity analysis via CCA + AJIVE.

This script supports both:
1. Legacy `unified_SELECTED_*.csv` files
2. Newer `selected_full_*.csv` files with the 41-feature selected schema

Examples
--------
./.venv/bin/python Scripts/cca_ajive_analysis.py dataset/selected_full_rrc04_AS3352_2hop_2025-11-01_2025-11-30_5min.csv
./.venv/bin/python Scripts/cca_ajive_analysis.py dataset/selected_full_*_5min.csv
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("path")  # set this to your project root
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "bgp_unified_results"
    / "output"
    / "unified_SELECTED_rrc04_AS12880_2hop_2025-11-07_2025-11-08_5min.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "bgp_unified_results" / "cross_view_redundancy_analysis"

META_COLUMNS = {
    "window_start",
    "window_id",
    "collector",
    "segment",
    "asn",
    "ego_nodes",
    "ego_edges",
    "ego_nodes_dynamic",
    "snapshot_id",
}

# Statistical features from the unified/statistical pipeline. Any non-meta
# feature not in this set is treated as graph-side.
STATISTICAL_FEATURES = {
    "announcements",
    "withdrawals",
    "total_updates",
    "ann_rate",
    "wd_rate",
    "wd_ann_ratio",
    "ann_wd_ratio",
    "unique_prefixes_ann",
    "unique_prefixes_wd",
    "new_prefixes",
    "dups",
    "flaps",
    "origin_IGP",
    "origin_INCOMPLETE",
    "origin_changes",
    "as_path_avg",
    "as_path_max",
    "as_path_std",
    "unique_as_path_max",
    "edit_distance_avg",
    "edit_distance_max",
    "edit_distance_dict_0",
    "edit_distance_dict_1",
    "edit_distance_dict_2",
    "edit_distance_dict_3",
    "edit_distance_dict_4",
    "edit_distance_dict_5",
    "edit_distance_dict_6",
    "edit_distance_unique_dict_0",
    "edit_distance_unique_dict_1",
    "number_rare_ases",
    "rare_ases_ratio",
    "imp_wd",
    "imp_wd_spath",
    "imp_wd_dpath",
    "nadas",
    "unique_peers",
    "ego_filter_ratio",
    "ego_updates_total",
    "ego_updates_filtered",
    # In the selected 41-feature schema, these summarize routing/ego-monitoring
    # behavior rather than pure topology, so we treat them as statistical-view.
    "vf_rate_delta",
    "ego_origin_violations",
    "ego_valley_free_paths_total",
    "ego_avg_violation_depth",
    "ego_max_violation_depth",
    "ego_valley_free_violations",
}

BOOTSTRAP_RESAMPLES_DEFAULT = 100
AJIVE_PARALLEL_SIMS_DEFAULT = 50
AJIVE_PARALLEL_QUANTILE_DEFAULT = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CCA + AJIVE complementarity analysis on one or more CSV datasets."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or directories containing CSV files. Defaults to the legacy unified_SELECTED sample.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base directory for outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of permutations for the CCA significance test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation/AJIVE routines.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES_DEFAULT,
        help="Bootstrap resamples for a 95%% confidence interval on rho_1.",
    )
    parser.add_argument(
        "--ajive-parallel-sims",
        type=int,
        default=AJIVE_PARALLEL_SIMS_DEFAULT,
        help="Number of Horn parallel-analysis simulations for AJIVE signal ranks.",
    )
    parser.add_argument(
        "--ajive-parallel-quantile",
        type=float,
        default=AJIVE_PARALLEL_QUANTILE_DEFAULT,
        help="Quantile cutoff for AJIVE parallel-analysis rank estimation.",
    )
    return parser.parse_args()


def resolve_input_paths(raw_inputs: list[str]) -> list[Path]:
    if not raw_inputs:
        return [DEFAULT_INPUT]

    resolved: list[Path] = []
    for item in raw_inputs:
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.is_dir():
            resolved.extend(sorted(candidate.glob("*.csv")))
        else:
            resolved.append(candidate)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    return unique_paths


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def ratio_text(numerator: float, denominator: float, decimals: int = 1) -> str:
    if abs(denominator) <= 1e-12:
        return "undefined (no joint variance estimated)"
    return f"{numerator / denominator:.{decimals}f}x"


def detect_view_membership(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    stat_cols = [col for col in feature_cols if col in STATISTICAL_FEATURES]
    graph_cols = [col for col in feature_cols if col not in STATISTICAL_FEATURES]
    return graph_cols, stat_cols


def fit_cca_correlations(X_g: np.ndarray, X_s: np.ndarray, n_components: int) -> list[float]:
    cca = CCA(n_components=n_components, max_iter=5000)
    X_c, Y_c = cca.fit_transform(X_g, X_s)

    correlations: list[float] = []
    for idx in range(n_components):
        corr = np.corrcoef(X_c[:, idx], Y_c[:, idx])[0, 1]
        correlations.append(float(abs(corr)) if np.isfinite(corr) else 0.0)
    return correlations


def bootstrap_rho1_ci(
    X_g: np.ndarray,
    X_s: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    if n_bootstrap <= 0:
        return {
            "n_bootstrap": 0,
            "n_successful": 0,
            "rho1_bootstrap": [],
            "rho1_ci_low": float("nan"),
            "rho1_ci_high": float("nan"),
            "rho1_bootstrap_mean": float("nan"),
            "rho1_bootstrap_std": float("nan"),
        }

    n_obs = X_g.shape[0]
    rng = check_random_state(random_state)
    rho1_samples: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_obs, size=n_obs)
        Xg_b = StandardScaler().fit_transform(X_g[idx, :])
        Xs_b = StandardScaler().fit_transform(X_s[idx, :])
        try:
            rho1_samples.append(fit_cca_correlations(Xg_b, Xs_b, 1)[0])
        except Exception:
            continue

    if not rho1_samples:
        return {
            "n_bootstrap": n_bootstrap,
            "n_successful": 0,
            "rho1_bootstrap": [],
            "rho1_ci_low": float("nan"),
            "rho1_ci_high": float("nan"),
            "rho1_bootstrap_mean": float("nan"),
            "rho1_bootstrap_std": float("nan"),
        }

    rho1_array = np.asarray(rho1_samples, dtype=float)
    ci_low, ci_high = np.percentile(rho1_array, [2.5, 97.5])

    return {
        "n_bootstrap": n_bootstrap,
        "n_successful": len(rho1_samples),
        "rho1_bootstrap": rho1_samples,
        "rho1_ci_low": float(ci_low),
        "rho1_ci_high": float(ci_high),
        "rho1_bootstrap_mean": float(rho1_array.mean()),
        "rho1_bootstrap_std": float(rho1_array.std(ddof=1) if len(rho1_array) > 1 else 0.0),
    }


def prepare_views(df: pd.DataFrame) -> dict:
    feat_df = df.drop(columns=[col for col in df.columns if col in META_COLUMNS], errors="ignore").copy()
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

    all_nan = feat_df.columns[feat_df.isna().all()].tolist()
    constant = feat_df.columns[feat_df.nunique(dropna=False) <= 1].tolist()
    dropped = sorted(set(all_nan + constant))

    feat_df = feat_df.drop(columns=dropped, errors="ignore")
    if feat_df.empty:
        raise ValueError("No usable numeric feature columns remain after dropping NaN/constant columns.")

    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    graph_cols, stat_cols = detect_view_membership(feat_df.columns.tolist())

    if not graph_cols:
        raise ValueError("No graph-view features were detected in the input file.")
    if not stat_cols:
        raise ValueError("No statistical-view features were detected in the input file.")

    scaler_g = StandardScaler()
    scaler_s = StandardScaler()
    X_g = scaler_g.fit_transform(feat_df[graph_cols].to_numpy())
    X_s = scaler_s.fit_transform(feat_df[stat_cols].to_numpy())

    membership_df = pd.DataFrame(
        {
            "Feature": graph_cols + stat_cols,
            "View": ["Graph"] * len(graph_cols) + ["Statistical"] * len(stat_cols),
        }
    )

    return {
        "feat_df": feat_df,
        "graph_cols": graph_cols,
        "stat_cols": stat_cols,
        "dropped": dropped,
        "X_g": X_g,
        "X_s": X_s,
        "membership_df": membership_df,
    }


def compute_cca_results(
    X_g: np.ndarray,
    X_s: np.ndarray,
    graph_cols: list[str],
    stat_cols: list[str],
    n_perms: int,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    n_components = min(len(graph_cols), len(stat_cols))
    canon_corrs = fit_cca_correlations(X_g, X_s, n_components)

    perm_rho1 = np.zeros(n_perms, dtype=float)
    rng = check_random_state(random_state)

    for perm_idx in range(n_perms):
        shuffled_idx = rng.permutation(X_s.shape[0])
        X_s_perm = X_s[shuffled_idx, :]
        try:
            perm_rho1[perm_idx] = fit_cca_correlations(X_g, X_s_perm, 1)[0]
        except Exception:
            perm_rho1[perm_idx] = 0.0

    rho1 = canon_corrs[0]
    p_value = float(np.mean(perm_rho1 >= rho1))
    bootstrap = bootstrap_rho1_ci(X_g, X_s, n_bootstrap=n_bootstrap, random_state=random_state + 7919)

    wilks_terms = 1.0 - np.clip(np.square(canon_corrs), 0.0, 0.999999999999)
    wilks_lambda = float(np.prod(wilks_terms))
    n_obs = X_g.shape[0]
    p_dim = len(graph_cols)
    q_dim = len(stat_cols)
    chi2 = float(
        max(
            0.0,
            -(n_obs - 1 - 0.5 * (p_dim + q_dim + 1)) * np.log(max(wilks_lambda, 1e-12)),
        )
    )
    dof = p_dim * q_dim
    wilks_p = float(1.0 - stats.chi2.cdf(chi2, dof))

    total_shared_r2 = float(np.sum(np.square(canon_corrs)))
    mean_r2 = float(total_shared_r2 / n_components)

    return {
        "n_components": n_components,
        "canon_corrs": canon_corrs,
        "perm_rho1": perm_rho1,
        "p_value": p_value,
        "wilks_lambda": wilks_lambda,
        "wilks_chi2": chi2,
        "wilks_dof": dof,
        "wilks_p": wilks_p,
        "total_shared_r2": total_shared_r2,
        "mean_r2": mean_r2,
        "rho1": rho1,
        "bootstrap": bootstrap,
    }


def estimate_signal_rank_parallel(
    X: np.ndarray,
    n_simulations: int,
    quantile: float,
    random_state: int,
) -> dict:
    n_obs, n_features = X.shape
    covariance = np.atleast_2d(np.cov(X, rowvar=False))
    observed_eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    rng = np.random.default_rng(random_state)
    null_eigenvalues = np.zeros((n_simulations, n_features), dtype=float)

    for sim_idx in range(n_simulations):
        random_matrix = rng.normal(size=(n_obs, n_features))
        random_matrix = StandardScaler().fit_transform(random_matrix)
        random_covariance = np.atleast_2d(np.cov(random_matrix, rowvar=False))
        null_eigenvalues[sim_idx, :] = np.linalg.eigvalsh(random_covariance)[::-1]

    threshold_eigenvalues = np.quantile(null_eigenvalues, quantile, axis=0)
    rank = int(np.sum(observed_eigenvalues > threshold_eigenvalues))
    rank = max(1, min(rank, n_features))

    return {
        "rank": rank,
        "method": "parallel_analysis",
        "n_simulations": n_simulations,
        "quantile": quantile,
        "observed_eigenvalues": observed_eigenvalues,
        "threshold_eigenvalues": threshold_eigenvalues,
    }


def compute_ajive(
    X_list: list[np.ndarray],
    initial_ranks: list[int] | None = None,
    parallel_sims: int = AJIVE_PARALLEL_SIMS_DEFAULT,
    parallel_quantile: float = AJIVE_PARALLEL_QUANTILE_DEFAULT,
    random_state: int = 42,
) -> dict:
    """
    Lightweight AJIVE-style decomposition for two standardized views.
    """
    k_views = len(X_list)
    n_obs = X_list[0].shape[0]
    rank_details: list[dict] = []

    if initial_ranks is None:
        initial_ranks = []
        for view_idx, X in enumerate(X_list, start=1):
            rank_info = estimate_signal_rank_parallel(
                X,
                n_simulations=parallel_sims,
                quantile=parallel_quantile,
                random_state=random_state + view_idx,
            )
            rank_k = rank_info["rank"]
            initial_ranks.append(rank_k)
            rank_details.append(rank_info)
            print(
                f"  View {view_idx}: estimated signal rank = {rank_k} "
                f"(of {X.shape[1]} features) via parallel analysis"
            )
            print(f"    Leading observed eigenvalues: {np.round(rank_info['observed_eigenvalues'][:6], 4)}")
            print(
                f"    Null {int(parallel_quantile * 100)}th-percentile cutoff: "
                f"{np.round(rank_info['threshold_eigenvalues'][:6], 4)}"
            )

    signal_bases = []
    for rank_k, X in zip(initial_ranks, X_list):
        U, _, _ = svd(X, full_matrices=False)
        signal_bases.append(U[:, :rank_k])

    stacked_bases = np.hstack(signal_bases)
    U_stacked, singular_vals_stacked, _ = svd(stacked_bases, full_matrices=False)

    joint_threshold = np.sqrt(k_views) * 0.9
    joint_rank = int(np.sum(singular_vals_stacked > joint_threshold))
    joint_rank = min(joint_rank, min(initial_ranks))

    print("\n  Joint rank estimation (Wedin-style threshold):")
    print(f"    Threshold: {joint_threshold:.4f}")
    print(f"    Singular values: {np.round(singular_vals_stacked[:min(10, len(singular_vals_stacked))], 4)}")
    print(f"    Joint rank: {joint_rank}")

    if joint_rank > 0:
        U_joint = U_stacked[:, :joint_rank]
        P_joint = U_joint @ U_joint.T
    else:
        P_joint = np.zeros((n_obs, n_obs))

    results = {
        "joint_rank": joint_rank,
        "initial_ranks": initial_ranks,
        "singular_values_stacked": singular_vals_stacked,
        "rank_method": "parallel_analysis",
        "rank_details": rank_details,
        "views": [],
    }

    total_var_all = 0.0
    joint_var_all = 0.0
    indiv_var_all = 0.0
    noise_var_all = 0.0

    for rank_k, X in zip(initial_ranks, X_list):
        total_var = float(np.sum(X**2))
        joint_component = P_joint @ X
        joint_var = float(np.sum(joint_component**2))

        residual = X - joint_component
        U_res, s_res, Vt_res = svd(residual, full_matrices=False)
        indiv_rank = max(0, min(rank_k - joint_rank, len(s_res)))

        if indiv_rank > 0:
            individual_component = (
                U_res[:, :indiv_rank]
                @ np.diag(s_res[:indiv_rank])
                @ Vt_res[:indiv_rank, :]
            )
        else:
            individual_component = np.zeros_like(X)

        indiv_var = float(np.sum(individual_component**2))
        noise_component = X - joint_component - individual_component
        noise_var = float(np.sum(noise_component**2))

        results["views"].append(
            {
                "total_var": total_var,
                "joint_var": joint_var,
                "indiv_var": indiv_var,
                "noise_var": noise_var,
                "joint_pct": safe_ratio(joint_var, total_var) * 100.0,
                "indiv_pct": safe_ratio(indiv_var, total_var) * 100.0,
                "noise_pct": safe_ratio(noise_var, total_var) * 100.0,
            }
        )

        total_var_all += total_var
        joint_var_all += joint_var
        indiv_var_all += indiv_var
        noise_var_all += noise_var

    results["overall"] = {
        "total_var": total_var_all,
        "joint_var": joint_var_all,
        "indiv_var": indiv_var_all,
        "noise_var": noise_var_all,
        "joint_pct": safe_ratio(joint_var_all, total_var_all) * 100.0,
        "indiv_pct": safe_ratio(indiv_var_all, total_var_all) * 100.0,
        "noise_pct": safe_ratio(noise_var_all, total_var_all) * 100.0,
    }

    return results


def interpret_cross_view_results(cca_res: dict, ajive_res: dict) -> dict:
    rho1_r2_pct = cca_res["rho1"]**2 * 100.0
    overall_joint_pct = ajive_res["overall"]["joint_pct"]
    overall_indiv_pct = ajive_res["overall"]["indiv_pct"]

    if rho1_r2_pct >= 50.0 and overall_joint_pct <= 25.0:
        short = (
            "CCA finds strong alignment along a few specific linear directions, "
            "but AJIVE shows that most total variance remains view-specific."
        )
        label = "specific_shared_but_globally_complementary"
    elif rho1_r2_pct >= 50.0 and overall_joint_pct > 25.0:
        short = "Both CCA and AJIVE indicate material cross-view redundancy."
        label = "broad_shared_structure"
    elif rho1_r2_pct < 10.0 and overall_indiv_pct >= 60.0:
        short = "Both CCA and AJIVE support strong complementarity across the two views."
        label = "strong_complementarity"
    elif rho1_r2_pct < 25.0 and overall_indiv_pct >= 60.0:
        short = (
            "CCA detects moderate directional coupling, but the overall variance "
            "decomposition confirms that the views are predominantly complementary."
        )
        label = "moderate_coupling_complementary"
    else:
        short = "The views show moderate coupling together with substantial view-specific structure."
        label = "moderate_mixed_structure"

    detail = (
        "CCA emphasizes the single best aligned latent direction, whereas AJIVE summarizes "
        "variance across the entire view, so high rho1 and modest joint variance can coexist."
    )

    return {
        "label": label,
        "short": short,
        "detail": detail,
    }


def pretty_dataset_name(dataset_label: str) -> str:
    tokens = dataset_label.replace(".csv", "").split("_")
    collector = next((token.upper() for token in tokens if token.startswith("rrc")), None)
    target_as = next((token for token in tokens if token.startswith("AS")), None)
    if collector and target_as:
        return f"{collector} / {target_as}"
    return dataset_label


def build_batch_consistency_summary(batch_df: pd.DataFrame) -> dict:
    rho1_min = float(batch_df["cca_rho1"].min())
    rho1_max = float(batch_df["cca_rho1"].max())
    joint_min = float(batch_df["ajive_overall_joint_pct"].min())
    joint_max = float(batch_df["ajive_overall_joint_pct"].max())
    indiv_min = float(batch_df["ajive_overall_indiv_pct"].min())
    indiv_max = float(batch_df["ajive_overall_indiv_pct"].max())

    all_significant = bool((batch_df["cca_perm_p_value"] < 0.05).all())
    all_majority_individual = bool((batch_df["ajive_overall_indiv_pct"] >= 50.0).all())
    all_low_joint = bool((batch_df["ajive_overall_joint_pct"] <= 25.0).all())

    if all_significant and all_majority_individual and all_low_joint:
        verdict = (
            "Consistently complementary across ASes and collectors: every dataset shows "
            "majority individual AJIVE variance (>{:.0f}%) with limited joint structure "
            "(<={:.0f}%), confirming that both views contribute predominantly unique information.".format(
                indiv_min, joint_max
            )
        )
        level = "consistently_complementary"
    elif all_significant and all_majority_individual:
        verdict = (
            "All datasets show majority individual variance, but the joint structure "
            "varies meaningfully across ASes and collectors."
        )
        level = "qualitatively_stable"
    elif all_significant:
        verdict = (
            "All datasets are statistically linked across views, but the strength and variance "
            "decomposition shift meaningfully by AS/collector."
        )
        level = "mixed_stability"
    else:
        verdict = "The pattern is not stable enough to claim consistent cross-dataset behavior."
        level = "unstable"

    detail = (
        f"rho1 ranges from {rho1_min:.3f} to {rho1_max:.3f}; "
        f"AJIVE overall joint variance ranges from {joint_min:.1f}% to {joint_max:.1f}%; "
        f"AJIVE overall individual variance ranges from {indiv_min:.1f}% to {indiv_max:.1f}%."
    )

    return {
        "level": level,
        "verdict": verdict,
        "detail": detail,
    }


def build_batch_latex_table(batch_df: pd.DataFrame, consistency: dict) -> str:
    lines = [
        "% Auto-generated cross-dataset CCA/AJIVE comparison table",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Cross-dataset comparison of CCA and AJIVE scores for the selected feature set.}",
        "\\label{tab:cca-ajive-batch}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Dataset & $p_G$ & $p_S$ & $\\rho_1$ & 95\\% CI & $R_1^2$ & AJIVE joint & AJIVE individual \\\\",
        "\\midrule",
    ]

    for _, row in batch_df.iterrows():
        lines.append(
            f"{row['dataset_pretty']} & "
            f"{int(row['graph_feature_count'])} & "
            f"{int(row['stat_feature_count'])} & "
            f"{row['cca_rho1']:.3f} & "
            f"[{row['cca_rho1_ci_low']:.3f}, {row['cca_rho1_ci_high']:.3f}] & "
            f"{row['cca_rho1_r2_pct']:.1f}\\% & "
            f"{row['ajive_overall_joint_pct']:.1f}\\% & "
            f"{row['ajive_overall_indiv_pct']:.1f}\\% \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            f"\\paragraph{{Consistency.}} {consistency['verdict']} {consistency['detail']}",
            "",
        ]
    )
    return "\n".join(lines)


def print_cca_report(cca_res: dict) -> None:
    print("=" * 70)
    print("CANONICAL CORRELATION ANALYSIS (CCA)")
    print("=" * 70)
    print(f"\nCanonical correlations (ρ₁ ≥ ρ₂ ≥ ... ≥ ρ_{cca_res['n_components']}):")
    print("-" * 50)

    for idx, corr in enumerate(cca_res["canon_corrs"], start=1):
        r2_pct = corr**2 * 100.0
        bar = "█" * int(corr * 40)
        print(f"  ρ_{idx:>2} = {corr:>7.4f}   (R² = {r2_pct:>5.1f}%)  {bar}")

    print(f"\n  ρ₁ (strongest possible cross-view link): {cca_res['rho1']:.4f}")
    print(f"  R² of ρ₁: {cca_res['rho1']**2:.4f} ({cca_res['rho1']**2 * 100:.1f}% shared variance)")
    if np.isfinite(cca_res["bootstrap"]["rho1_ci_low"]):
        print(
            "  Bootstrap 95% CI for ρ₁: "
            f"[{cca_res['bootstrap']['rho1_ci_low']:.4f}, {cca_res['bootstrap']['rho1_ci_high']:.4f}] "
            f"from {cca_res['bootstrap']['n_successful']}/{cca_res['bootstrap']['n_bootstrap']} resamples"
        )

    print(f"\n{'─' * 50}")
    print("Permutation test (H₀: views are independent)")
    print(f"{'─' * 50}")
    print(f"  Observed ρ₁:          {cca_res['rho1']:.4f}")
    print(f"  Permutation mean ρ₁:  {cca_res['perm_rho1'].mean():.4f}")
    print(f"  Permutation 95th %:   {np.percentile(cca_res['perm_rho1'], 95):.4f}")
    print(f"  Permutation 99th %:   {np.percentile(cca_res['perm_rho1'], 99):.4f}")
    print(f"  p-value:              {cca_res['p_value']:.4f}")

    print(f"\n{'─' * 50}")
    print("Wilks' Lambda test (all canonical correlations = 0?)")
    print(f"{'─' * 50}")
    print(f"  Wilks' Λ = {cca_res['wilks_lambda']:.6f}")
    print(
        f"  χ² = {cca_res['wilks_chi2']:.2f}, "
        f"df = {cca_res['wilks_dof']}, p = {cca_res['wilks_p']:.6f}"
    )

    print(f"\n{'─' * 50}")
    print("Summary: Total shared variance")
    print(f"{'─' * 50}")
    print(f"  Sum of all R²:  {cca_res['total_shared_r2']:.4f} out of max {cca_res['n_components']}")
    print(f"  Mean R² per CC: {cca_res['mean_r2']:.4f} ({cca_res['mean_r2'] * 100:.1f}%)")


def print_ajive_report(ajive_res: dict) -> None:
    print("\n" + "=" * 70)
    print("AJIVE: Joint and Individual Variation Explained")
    print("=" * 70)
    print(f"\n{'─' * 60}")
    print("AJIVE Variance Decomposition")
    print(f"{'─' * 60}")
    print(f"\n  {'Component':<20} {'Graph View':>12} {'Stat View':>12} {'Overall':>12}")
    print(f"  {'─' * 56}")

    vg, vs = ajive_res["views"]
    ov = ajive_res["overall"]
    print(
        f"  {'Joint (shared)':<20} "
        f"{vg['joint_pct']:>11.1f}% {vs['joint_pct']:>11.1f}% {ov['joint_pct']:>11.1f}%"
    )
    print(
        f"  {'Individual (unique)':<20} "
        f"{vg['indiv_pct']:>11.1f}% {vs['indiv_pct']:>11.1f}% {ov['indiv_pct']:>11.1f}%"
    )
    print(
        f"  {'Noise':<20} "
        f"{vg['noise_pct']:>11.1f}% {vs['noise_pct']:>11.1f}% {ov['noise_pct']:>11.1f}%"
    )

    print(f"\n  Joint rank: {ajive_res['joint_rank']}")
    print(
        f"  Signal ranks: Graph={ajive_res['initial_ranks'][0]}, "
        f"Stat={ajive_res['initial_ranks'][1]}"
    )
    print(f"  Rank estimation: {ajive_res['rank_method']}")

    graph_ratio = safe_ratio(vg["indiv_var"], max(vg["joint_var"], 1e-10))
    stat_ratio = safe_ratio(vs["indiv_var"], max(vs["joint_var"], 1e-10))
    print(f"\n  Individual-to-Joint ratio:")
    print(f"    Graph: {ratio_text(vg['indiv_var'], vg['joint_var'])}")
    print(f"    Stat:  {ratio_text(vs['indiv_var'], vs['joint_var'])}")


def print_cross_view_interpretation(interpretation: dict) -> None:
    print(f"\n{'─' * 60}")
    print("CCA vs AJIVE Interpretation")
    print(f"{'─' * 60}")
    print(f"  {interpretation['short']}")
    print(f"  {interpretation['detail']}")


def save_visualization(
    dataset_label: str,
    output_dir: Path,
    cca_res: dict,
    ajive_res: dict,
    interpretation: dict,
) -> Path:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    vg, vs = ajive_res["views"]
    ov = ajive_res["overall"]

    ax1 = fig.add_subplot(gs[0, 0])
    colors_cc = [
        "#c0392b" if cc > 0.30 else "#f39c12" if cc > 0.15 else "#27ae60"
        for cc in cca_res["canon_corrs"]
    ]
    ax1.bar(
        range(1, cca_res["n_components"] + 1),
        cca_res["canon_corrs"],
        color=colors_cc,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.axhline(y=0.30, color="red", linestyle="--", linewidth=1, label="Moderate threshold")
    ax1.axhline(y=0.70, color="darkred", linestyle="--", linewidth=1, label="Strong threshold")
    ax1.set_xlabel("Canonical Component", fontsize=11)
    ax1.set_ylabel("Canonical Correlation (ρ)", fontsize=11)
    ax1.set_title("CCA: Canonical Correlations", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(1, cca_res["n_components"] + 1))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        cca_res["perm_rho1"],
        bins=40,
        color="#bdc3c7",
        edgecolor="gray",
        alpha=0.8,
        label="Permuted ρ₁",
    )
    ax2.axvline(
        x=cca_res["rho1"],
        color="#c0392b",
        linewidth=2.5,
        label=f"Observed ρ₁ = {cca_res['rho1']:.3f}",
    )
    ax2.axvline(
        x=np.percentile(cca_res["perm_rho1"], 95),
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"95th percentile = {np.percentile(cca_res['perm_rho1'], 95):.3f}",
    )
    ax2.set_xlabel("First Canonical Correlation", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"Permutation Test (p = {cca_res['p_value']:.3f})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    r_squared = [cc**2 * 100.0 for cc in cca_res["canon_corrs"]]
    cumulative_r2 = np.cumsum(r_squared)
    ax3.bar(
        range(1, cca_res["n_components"] + 1),
        r_squared,
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        label="Per-component R²",
    )
    ax3.plot(
        range(1, cca_res["n_components"] + 1),
        cumulative_r2,
        "ro-",
        linewidth=1.5,
        markersize=5,
        label="Cumulative R²",
    )
    ax3.set_xlabel("Canonical Component", fontsize=11)
    ax3.set_ylabel("Shared Variance R² (%)", fontsize=11)
    ax3.set_title("CCA: Shared Variance per Component", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_xticks(range(1, cca_res["n_components"] + 1))

    ax4 = fig.add_subplot(gs[1, 0])
    views_labels = ["Graph View", "Statistical View", "Overall"]
    joint_pcts = [vg["joint_pct"], vs["joint_pct"], ov["joint_pct"]]
    indiv_pcts = [vg["indiv_pct"], vs["indiv_pct"], ov["indiv_pct"]]
    noise_pcts = [vg["noise_pct"], vs["noise_pct"], ov["noise_pct"]]
    x_pos = np.arange(len(views_labels))
    width = 0.60

    ax4.bar(x_pos, joint_pcts, width, label="Joint (shared)", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax4.bar(
        x_pos,
        indiv_pcts,
        width,
        bottom=joint_pcts,
        label="Individual (unique)",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.bar(
        x_pos,
        noise_pcts,
        width,
        bottom=[j + i for j, i in zip(joint_pcts, indiv_pcts)],
        label="Noise",
        color="#bdc3c7",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_ylabel("Variance (%)", fontsize=11)
    ax4.set_title("AJIVE: Variance Decomposition", fontsize=13, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(views_labels, fontsize=11)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.set_ylim(0, 105)

    ax5 = fig.add_subplot(gs[1, 1])
    singular_vals = ajive_res["singular_values_stacked"]
    n_sv = min(len(singular_vals), 20)
    colors_sv = [
        "#c0392b" if value > np.sqrt(2) * 0.9 else "#3498db"
        for value in singular_vals[:n_sv]
    ]
    ax5.bar(
        range(1, n_sv + 1),
        singular_vals[:n_sv],
        color=colors_sv,
        edgecolor="black",
        linewidth=0.5,
    )
    ax5.axhline(
        y=np.sqrt(2) * 0.9,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Joint threshold = {np.sqrt(2) * 0.9:.2f}",
    )
    ax5.set_xlabel("Component Index", fontsize=11)
    ax5.set_ylabel("Singular Value", fontsize=11)
    ax5.set_title("AJIVE: Stacked Signal Space SVD", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_text = (
        f"{dataset_label}\n"
        f"{'━' * min(len(dataset_label), 35)}\n\n"
        f"CCA Results:\n"
        f"  ρ₁ = {cca_res['rho1']:.4f}\n"
        f"  95% CI = [{cca_res['bootstrap']['rho1_ci_low']:.4f}, {cca_res['bootstrap']['rho1_ci_high']:.4f}]\n"
        f"  R²(ρ₁) = {cca_res['rho1']**2 * 100:.1f}%\n"
        f"  Permutation p = {cca_res['p_value']:.4f}\n"
        f"  Mean R² per CC = {cca_res['mean_r2'] * 100:.1f}%\n\n"
        f"AJIVE Results:\n"
        f"  Joint rank = {ajive_res['joint_rank']}\n"
        f"  Graph: {vg['indiv_pct']:.1f}% individual, {vg['joint_pct']:.1f}% joint\n"
        f"  Stat:  {vs['indiv_pct']:.1f}% individual, {vs['joint_pct']:.1f}% joint\n\n"
        f"Interpretation:\n"
        f"  {interpretation['short']}"
    )
    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#2c3e50", linewidth=1.5),
    )

    fig.suptitle(
        f"Cross-View Complementarity Analysis: {dataset_label}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    output_path = output_dir / "cca_ajive_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_latex_text(
    n_obs: int,
    graph_cols: list[str],
    stat_cols: list[str],
    cca_res: dict,
    ajive_res: dict,
    interpretation: dict,
    n_perms: int,
) -> str:
    vg, vs = ajive_res["views"]
    ov = ajive_res["overall"]
    if abs(vg["joint_var"]) <= 1e-12:
        graph_ratio_sentence = "The graph-view individual-to-joint ratio is undefined because AJIVE estimated zero joint variance"
    else:
        graph_ratio_sentence = (
            f"The graph-view individual-to-joint ratio is ${vg['indiv_var'] / vg['joint_var']:.1f}\\times$"
        )

    if abs(vs["joint_var"]) <= 1e-12:
        stat_ratio_sentence = (
            "the statistical-view ratio is likewise undefined because the estimated joint rank is zero"
        )
    else:
        stat_ratio_sentence = (
            f"the statistical-view ratio is ${vs['indiv_var'] / vs['joint_var']:.1f}\\times$"
        )

    return f"""
% ── Paste into Section 3.5: Formal Complementarity Analysis ──

\\subsection{{Formal Complementarity Analysis}}
\\label{{sec:complementarity}}

To verify that the observed pairwise independence extends to multivariate
combinations, we apply Canonical Correlation Analysis (CCA) and Angle-based
Joint and Individual Variation Explained (AJIVE) to the graph-theoretic
($\\mathbf{{X}}_G \\in \\mathbb{{R}}^{{{n_obs} \\times {len(graph_cols)}}}$) and statistical
($\\mathbf{{X}}_S \\in \\mathbb{{R}}^{{{n_obs} \\times {len(stat_cols)}}}$) feature matrices.

\\paragraph{{Canonical Correlation Analysis.}}
CCA finds the maximum possible correlation between any linear combination of
graph features and any linear combination of statistical features. The first
canonical correlation is $\\rho_1 = {cca_res['rho1']:.4f}$, indicating that even
the optimal linear projection of all {len(graph_cols)} graph features explains only
${cca_res['rho1']**2 * 100:.1f}\\%$ of statistical feature variance
($R^2 = {cca_res['rho1']**2:.4f}$). A permutation test ({n_perms:,} shuffles)
yields $p = {cca_res['p_value']:.4f}$, while the mean shared variance across all
{cca_res['n_components']} canonical dimensions is ${cca_res['mean_r2'] * 100:.1f}\\%$.
A bootstrap analysis ({cca_res['bootstrap']['n_successful']} successful resamples)
yields a 95\\% confidence interval of
$[{cca_res['bootstrap']['rho1_ci_low']:.4f}, {cca_res['bootstrap']['rho1_ci_high']:.4f}]$
for $\\rho_1$.

\\paragraph{{AJIVE Decomposition.}}
AJIVE decomposes each view's variation into joint (shared by both views),
individual (unique to each view), and noise components. With estimated signal
ranks of {ajive_res['initial_ranks'][0]} (graph) and {ajive_res['initial_ranks'][1]}
(statistical), estimated via Horn-style parallel analysis, and a joint rank of
{ajive_res['joint_rank']}, the decomposition reveals:

\\begin{{table}}[H]
\\centering
\\caption{{AJIVE variance decomposition of graph and statistical feature views.}}
\\label{{tab:ajive}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Component}} & \\textbf{{Graph View}} & \\textbf{{Statistical View}} & \\textbf{{Overall}} \\\\
\\midrule
Joint (shared) & ${vg['joint_pct']:.1f}\\%$ & ${vs['joint_pct']:.1f}\\%$ & ${ov['joint_pct']:.1f}\\%$ \\\\
Individual (unique) & ${vg['indiv_pct']:.1f}\\%$ & ${vs['indiv_pct']:.1f}\\%$ & ${ov['indiv_pct']:.1f}\\%$ \\\\
Noise & ${vg['noise_pct']:.1f}\\%$ & ${vs['noise_pct']:.1f}\\%$ & ${ov['noise_pct']:.1f}\\%$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{graph_ratio_sentence}, and {stat_ratio_sentence}, confirming that each
view contributes predominantly unique information whenever nonzero joint structure
is present. {interpretation['short']}
{interpretation['detail']}
""".strip() + "\n"


def export_tables(
    output_dir: Path,
    graph_cols: list[str],
    stat_cols: list[str],
    dropped: list[str],
    membership_df: pd.DataFrame,
    cca_res: dict,
    ajive_res: dict,
    interpretation: dict,
    latex_text: str,
) -> None:
    vg, vs = ajive_res["views"]
    ov = ajive_res["overall"]

    pd.DataFrame(
        {
            "Component": range(1, cca_res["n_components"] + 1),
            "Canonical_Correlation": cca_res["canon_corrs"],
            "R_squared": [cc**2 for cc in cca_res["canon_corrs"]],
            "R_squared_pct": [cc**2 * 100.0 for cc in cca_res["canon_corrs"]],
        }
    ).to_csv(output_dir / "cca_canonical_correlations.csv", index=False)

    pd.DataFrame(
        {
            "View": ["Graph", "Statistical", "Overall"],
            "Joint_pct": [vg["joint_pct"], vs["joint_pct"], ov["joint_pct"]],
            "Individual_pct": [vg["indiv_pct"], vs["indiv_pct"], ov["indiv_pct"]],
            "Noise_pct": [vg["noise_pct"], vs["noise_pct"], ov["noise_pct"]],
            "Joint_var": [vg["joint_var"], vs["joint_var"], ov["joint_var"]],
            "Individual_var": [vg["indiv_var"], vs["indiv_var"], ov["indiv_var"]],
            "Noise_var": [vg["noise_var"], vs["noise_var"], ov["noise_var"]],
        }
    ).to_csv(output_dir / "ajive_decomposition.csv", index=False)

    pd.DataFrame(
        {
            "Metric": [
                "CCA_rho1",
                "CCA_rho1_R2",
                "CCA_rho1_CI_low",
                "CCA_rho1_CI_high",
                "CCA_rho1_bootstrap_std",
                "CCA_perm_p_value",
                "CCA_mean_R2_pct",
                "CCA_wilks_lambda",
                "CCA_wilks_p",
                "AJIVE_joint_rank",
                "AJIVE_graph_joint_pct",
                "AJIVE_graph_indiv_pct",
                "AJIVE_graph_noise_pct",
                "AJIVE_stat_joint_pct",
                "AJIVE_stat_indiv_pct",
                "AJIVE_stat_noise_pct",
                "AJIVE_overall_joint_pct",
                "AJIVE_overall_indiv_pct",
            ],
            "Value": [
                cca_res["rho1"],
                cca_res["rho1"]**2,
                cca_res["bootstrap"]["rho1_ci_low"],
                cca_res["bootstrap"]["rho1_ci_high"],
                cca_res["bootstrap"]["rho1_bootstrap_std"],
                cca_res["p_value"],
                cca_res["mean_r2"] * 100.0,
                cca_res["wilks_lambda"],
                cca_res["wilks_p"],
                ajive_res["joint_rank"],
                vg["joint_pct"],
                vg["indiv_pct"],
                vg["noise_pct"],
                vs["joint_pct"],
                vs["indiv_pct"],
                vs["noise_pct"],
                ov["joint_pct"],
                ov["indiv_pct"],
            ],
        }
    ).to_csv(output_dir / "cca_ajive_summary.csv", index=False)

    membership_df.to_csv(output_dir / "view_membership.csv", index=False)
    pd.DataFrame({"Dropped_Feature": dropped}).to_csv(output_dir / "dropped_features.csv", index=False)
    with open(output_dir / "cross_view_interpretation.txt", "w", encoding="utf-8") as handle:
        handle.write(interpretation["short"] + "\n" + interpretation["detail"] + "\n")

    with open(output_dir / "cca_ajive_section.tex", "w", encoding="utf-8") as handle:
        handle.write(latex_text)

    json_payload = {
        "graph_features": graph_cols,
        "statistical_features": stat_cols,
        "dropped_features": dropped,
        "cca": {
            "n_components": cca_res["n_components"],
            "rho1": cca_res["rho1"],
            "canonical_correlations": cca_res["canon_corrs"],
            "perm_p_value": cca_res["p_value"],
            "wilks_lambda": cca_res["wilks_lambda"],
            "wilks_p": cca_res["wilks_p"],
            "mean_r2_pct": cca_res["mean_r2"] * 100.0,
            "bootstrap": cca_res["bootstrap"],
        },
        "ajive": {
            "joint_rank": ajive_res["joint_rank"],
            "initial_ranks": ajive_res["initial_ranks"],
            "rank_method": ajive_res["rank_method"],
            "rank_details": [
                {
                    "rank": detail["rank"],
                    "method": detail["method"],
                    "n_simulations": detail["n_simulations"],
                    "quantile": detail["quantile"],
                    "observed_eigenvalues": detail["observed_eigenvalues"].tolist(),
                    "threshold_eigenvalues": detail["threshold_eigenvalues"].tolist(),
                }
                for detail in ajive_res["rank_details"]
            ],
            "graph_joint_pct": vg["joint_pct"],
            "graph_indiv_pct": vg["indiv_pct"],
            "stat_joint_pct": vs["joint_pct"],
            "stat_indiv_pct": vs["indiv_pct"],
            "overall_joint_pct": ov["joint_pct"],
            "overall_indiv_pct": ov["indiv_pct"],
        },
        "interpretation": interpretation,
    }
    with open(output_dir / "cca_ajive_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)


def run_single_dataset(
    input_path: Path,
    output_root: Path,
    n_perms: int,
    n_bootstrap: int,
    ajive_parallel_sims: int,
    ajive_parallel_quantile: float,
    random_state: int,
) -> dict:
    dataset_label = input_path.stem
    output_dir = output_root / dataset_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"Dataset: {input_path}")
    print("#" * 80)

    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape}")

    prepared = prepare_views(df)
    graph_cols = prepared["graph_cols"]
    stat_cols = prepared["stat_cols"]
    feat_df = prepared["feat_df"]

    print(f"\nDropped (constant/NaN): {prepared['dropped']}")
    print(f"Graph features ({len(graph_cols)}): {graph_cols}")
    print(f"Statistical features ({len(stat_cols)}): {stat_cols}")
    print(f"Observations: {len(feat_df)}")
    print(f"X_graph: {prepared['X_g'].shape}")
    print(f"X_stat:  {prepared['X_s'].shape}")

    cca_res = compute_cca_results(
        prepared["X_g"],
        prepared["X_s"],
        graph_cols,
        stat_cols,
        n_perms=n_perms,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    print_cca_report(cca_res)

    ajive_res = compute_ajive(
        [prepared["X_g"], prepared["X_s"]],
        parallel_sims=ajive_parallel_sims,
        parallel_quantile=ajive_parallel_quantile,
        random_state=random_state,
    )
    print_ajive_report(ajive_res)
    interpretation = interpret_cross_view_results(cca_res, ajive_res)
    print_cross_view_interpretation(interpretation)

    plot_path = save_visualization(dataset_label, output_dir, cca_res, ajive_res, interpretation)
    latex_text = build_latex_text(
        n_obs=len(feat_df),
        graph_cols=graph_cols,
        stat_cols=stat_cols,
        cca_res=cca_res,
        ajive_res=ajive_res,
        interpretation=interpretation,
        n_perms=n_perms,
    )
    export_tables(
        output_dir=output_dir,
        graph_cols=graph_cols,
        stat_cols=stat_cols,
        dropped=prepared["dropped"],
        membership_df=prepared["membership_df"],
        cca_res=cca_res,
        ajive_res=ajive_res,
        interpretation=interpretation,
        latex_text=latex_text,
    )

    print(f"\nAll results exported to: {output_dir}")
    print(f"  {plot_path.name}")
    print("  cca_canonical_correlations.csv")
    print("  ajive_decomposition.csv")
    print("  cca_ajive_summary.csv")
    print("  cca_ajive_summary.json")
    print("  cca_ajive_section.tex")
    print("  view_membership.csv")
    print("  dropped_features.csv")
    print("  cross_view_interpretation.txt")

    vg, vs = ajive_res["views"]
    ov = ajive_res["overall"]

    return {
        "dataset": dataset_label,
        "dataset_pretty": pretty_dataset_name(dataset_label),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_obs": len(feat_df),
        "graph_feature_count": len(graph_cols),
        "stat_feature_count": len(stat_cols),
        "cca_rho1": cca_res["rho1"],
        "cca_rho1_ci_low": cca_res["bootstrap"]["rho1_ci_low"],
        "cca_rho1_ci_high": cca_res["bootstrap"]["rho1_ci_high"],
        "cca_rho1_bootstrap_std": cca_res["bootstrap"]["rho1_bootstrap_std"],
        "cca_rho1_r2_pct": cca_res["rho1"]**2 * 100.0,
        "cca_perm_p_value": cca_res["p_value"],
        "cca_mean_r2_pct": cca_res["mean_r2"] * 100.0,
        "cca_wilks_p": cca_res["wilks_p"],
        "ajive_joint_rank": ajive_res["joint_rank"],
        "ajive_graph_joint_pct": vg["joint_pct"],
        "ajive_graph_indiv_pct": vg["indiv_pct"],
        "ajive_stat_joint_pct": vs["joint_pct"],
        "ajive_stat_indiv_pct": vs["indiv_pct"],
        "ajive_overall_joint_pct": ov["joint_pct"],
        "ajive_overall_indiv_pct": ov["indiv_pct"],
        "interpretation_label": interpretation["label"],
        "interpretation_short": interpretation["short"],
    }


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args.inputs)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_path in input_paths:
        rows.append(
            run_single_dataset(
                input_path=input_path,
                output_root=output_root,
                n_perms=args.permutations,
                n_bootstrap=args.bootstrap_resamples,
                ajive_parallel_sims=args.ajive_parallel_sims,
                ajive_parallel_quantile=args.ajive_parallel_quantile,
                random_state=args.seed,
            )
        )

    batch_df = pd.DataFrame(rows)
    batch_path = output_root / "batch_summary.csv"
    batch_df.to_csv(batch_path, index=False)
    consistency = build_batch_consistency_summary(batch_df)

    with open(output_root / "batch_summary.tex", "w", encoding="utf-8") as handle:
        handle.write(build_batch_latex_table(batch_df, consistency))
    with open(output_root / "batch_consistency.txt", "w", encoding="utf-8") as handle:
        handle.write(consistency["verdict"] + "\n" + consistency["detail"] + "\n")
    with open(output_root / "batch_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": batch_df.to_dict(orient="records"), "consistency": consistency}, handle, indent=2)

    print("\n" + "=" * 80)
    print("BATCH SCORE SUMMARY")
    print("=" * 80)
    print(
        batch_df[
            [
                "dataset_pretty",
                "dataset",
                "graph_feature_count",
                "stat_feature_count",
                "cca_rho1",
                "cca_rho1_ci_low",
                "cca_rho1_ci_high",
                "cca_rho1_r2_pct",
                "cca_perm_p_value",
                "ajive_overall_joint_pct",
                "ajive_overall_indiv_pct",
            ]
        ].to_string(index=False)
    )
    print("\nConsistency verdict:")
    print(f"  {consistency['verdict']}")
    print(f"  {consistency['detail']}")
    print(f"\nSaved batch summary: {batch_path}")
    print(f"Saved batch LaTeX table: {output_root / 'batch_summary.tex'}")


if __name__ == "__main__":
    main()
