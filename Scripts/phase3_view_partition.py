"""
Phase 3 view partition for the shared22 feature space.

Splits the 22 shared features into the two canonical superviews defined by
the architecture doc (section 3, "Views"):

    graph view      - topology / centrality / rich-club / IXP-connectivity features
    statistical view - AS-path stats, edit-distance stats, update-stream counters,
                       ego-path violation counts

The assignment matches the ``graph_features`` / ``statistical_features`` split
used by every Phase 1 CCA/AJIVE analysis run (see
``bgp_unified_results/cross_view_redundancy_analysis/**/cca_ajive_summary.json``).
Any feature's view assignment is globally consistent across all four domains
and all three regimes, so we hard-code the mapping here.

Usage:
    from Scripts.phase3_view_partition import (
        SHARED22_GRAPH_VIEW, SHARED22_STAT_VIEW, view_columns, validate_partition,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHARED22_GRAPH_VIEW: tuple[str, ...] = (
    "innermost_core_size",
    "rich_club_p75",
    "rich_club_p90",
    "rich_club_p95",
    "symmetry_ratio",
    "std_edge_ixp_cosine_dist",
    "clustering_avg_local",
    "frac_p2p_edges",
    "avg_ixp_cosine_dist",
)

SHARED22_STAT_VIEW: tuple[str, ...] = (
    "as_path_avg",
    "as_path_max",
    "as_path_std",
    "edit_distance_avg",
    "edit_distance_max",
    "unique_as_path_max",
    "flaps",
    "nadas",
    "vf_rate_delta",
    "ego_filter_ratio",
    "ego_origin_violations",
    "unique_peers",
    "origin_changes",
)

SHARED22_ALL: tuple[str, ...] = SHARED22_GRAPH_VIEW + SHARED22_STAT_VIEW

# Curated 10-feature portable-floor set for the §5.3 shared22-vs-core10 ablation.
# Selection: intersection of the three regime cores (full-dataset, likely-normal,
# anomaly) with a de-redundancy refinement that (i) collapses the three
# rich_club quantiles to p90 only, (ii) drops unique_as_path_max as collinear
# with as_path_max, and (iii) reintroduces one feature per functional axis —
# frac_p2p_edges (mesh-vs-tree discriminator), edit_distance_max (path-rewrite
# extreme), flaps (update-churn). Evaluated as a single flat view;
# graph/stat partitioning is not applied at this dimensionality.
CORE10_ALL: tuple[str, ...] = (
    "innermost_core_size",
    "rich_club_p90",
    "symmetry_ratio",
    "frac_p2p_edges",
    "std_edge_ixp_cosine_dist",
    "as_path_max",
    "edit_distance_max",
    "flaps",
    "ego_origin_violations",
    "unique_peers",
)


def view_columns(view: str) -> tuple[str, ...]:
    view = view.lower()
    if view == "graph":
        return SHARED22_GRAPH_VIEW
    if view in {"stat", "statistical"}:
        return SHARED22_STAT_VIEW
    if view in {"both", "all", "fusion"}:
        return SHARED22_ALL
    if view == "core10":
        return CORE10_ALL
    raise ValueError(f"unknown view: {view!r} (expected 'graph', 'stat', 'both', or 'core10')")


def validate_partition(available_columns: Iterable[str]) -> None:
    available = set(available_columns)
    missing_graph = [f for f in SHARED22_GRAPH_VIEW if f not in available]
    missing_stat = [f for f in SHARED22_STAT_VIEW if f not in available]
    if missing_graph or missing_stat:
        raise ValueError(
            "shared22 view partition mismatch.\n"
            f"  missing graph-view features: {missing_graph}\n"
            f"  missing stat-view features:  {missing_stat}"
        )
    if len(SHARED22_ALL) != 22:
        raise AssertionError(
            f"shared22 partition must have 22 features, got {len(SHARED22_ALL)}"
        )
    if set(SHARED22_GRAPH_VIEW) & set(SHARED22_STAT_VIEW):
        raise AssertionError("graph and stat views overlap")


def validate_core10_partition(available_columns: Iterable[str]) -> None:
    available = set(available_columns)
    missing = [f for f in CORE10_ALL if f not in available]
    if missing:
        raise ValueError(
            "core10 feature set mismatch.\n"
            f"  missing core10 features: {missing}"
        )
    if len(CORE10_ALL) != 10:
        raise AssertionError(
            f"core10 feature set must have 10 features, got {len(CORE10_ALL)}"
        )
    extra_in_shared = set(CORE10_ALL) - set(SHARED22_ALL)
    if extra_in_shared:
        raise AssertionError(
            f"core10 feature not drawn from shared22: {sorted(extra_in_shared)}"
        )


if __name__ == "__main__":
    import pandas as pd

    sample_path = PROJECT_ROOT / "dataset" / "phase3_training" / "shared22" / "rrc05_as3352.csv"
    df_head = pd.read_csv(sample_path, nrows=1)
    validate_partition(df_head.columns)
    print(f"shared22 view partition OK ({len(SHARED22_ALL)} features)")
    print(f"  graph view ({len(SHARED22_GRAPH_VIEW)}):")
    for f in SHARED22_GRAPH_VIEW:
        print(f"    - {f}")
    print(f"  stat view ({len(SHARED22_STAT_VIEW)}):")
    for f in SHARED22_STAT_VIEW:
        print(f"    - {f}")

    core10_path = PROJECT_ROOT / "dataset" / "phase3_training" / "core10" / "rrc05_as3352.csv"
    if core10_path.exists():
        df_core = pd.read_csv(core10_path, nrows=1)
        validate_core10_partition(df_core.columns)
        print(f"core10 feature set OK ({len(CORE10_ALL)} features):")
        for f in CORE10_ALL:
            print(f"    - {f}")
