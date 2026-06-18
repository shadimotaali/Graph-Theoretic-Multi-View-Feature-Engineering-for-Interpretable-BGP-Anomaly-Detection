#!/usr/bin/env python3
"""
5-minute AS-topology snapshot extractor (astopo_5m, stretch goal for GNN §5.7).

Algorithm (agreed with supervisor 2026-04-21):
  1. Load bview.YYYYMMDD.HHMM.gz at every 8-hour boundary (00:00 / 08:00 / 16:00)
     via IncrementalTopology.load_rib() to reset the topology and prevent
     drift from accumulated incremental errors.
  2. Between 8h boundaries, apply updates.YYYYMMDD.HHMM.gz files in 5-minute
     increments via IncrementalTopology.apply_window_updates(). After each
     5-min batch, extract the 2-hop ego subgraph around the target AS and
     compute node-level features matching the existing 8h snapshot schema.
  3. Emit per-window CSVs:
         dataset/gnn_graphs_5min/<pair>/<YYYYMMDD_HHMM>/nodes.csv
         dataset/gnn_graphs_5min/<pair>/<YYYYMMDD_HHMM>/edges.csv
     and a master index at dataset/gnn_graphs_5min/snapshots_index.csv.

Reads MRT data from shared_cache/mrt_files/{rrc04,rrc05}/ — no network access.

Pairs processed: D1 (rrc04_as12880), D2 (rrc04_as3352),
                 D3 (rrc05_as12880), D4 (rrc05_as3352).
Study window: 2025-11-01 00:00 UTC .. 2025-11-30 00:00 UTC  (29 days).

Resumable: already-present snapshot dirs are skipped. Safe to kill/restart.

Usage (CPU-only, expect ~18-20h for all 4 pairs sequentially):
    tmux new -s astopo_5m
    python Scripts/extract_astopo_5min.py                    # all 4 pairs
    python Scripts/extract_astopo_5min.py --pair rrc04_as12880  # one pair
    python Scripts/extract_astopo_5min.py --days 1 --pair rrc04_as12880  # smoke test
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import networkx as nx
import pandas as pd

import networkit as nk  # required for fast centralities

from graph_features_lib import (
    IncrementalTopology,
    annotate_graph_relationships,
    build_ixp_feature_vectors,
    compute_ixp_cosine_features,
    extract_graph_level_features,
    extract_node_level_features,
    extract_relationship_features,
    load_caida_relationships,
    load_peeringdb_ixp_memberships,
)

from historical_incident_extractor import parse_updates_bgpkit

# ============================================================================
# Configuration
# ============================================================================

MRT_CACHE = PROJECT_ROOT / "shared_cache" / "mrt_files"
CAIDA_DIR = PROJECT_ROOT / "bgp_unified_results" / "reference_data" / "caida"
PEERINGDB_JSON = PROJECT_ROOT / "bgp_unified_results" / "temp" / "peeringdb_2_dump_2025_11_06.json"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "gnn_graphs_5min"
INDEX_PATH = OUTPUT_DIR / "snapshots_index.csv"

EGO_K_HOP = 2
WINDOW_SIZE_MINUTES = 5
RIB_RESYNC_HOURS = 8

STUDY_START = datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)
STUDY_END   = datetime(2025, 11, 30, 0, 0, tzinfo=timezone.utc)

# Matches the CAIDA vintage the 8h pipeline used for the Nov-2025 study run.
CAIDA_FILE = CAIDA_DIR / "20240101.as-rel2.txt"

# Node column schema — must match existing dataset/gnn_graphs/<pair>/*/nodes.csv
NODE_COLS = [
    "asn", "degree", "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality", "pagerank",
    "local_clustering", "avg_neighbor_degree", "node_clique_number",
    "eccentricity", "core_number", "n_providers", "n_customers", "n_peers",
    "provider_ratio", "is_tier1", "avg_ixp_cosine_dist", "ixp_vector_norm",
    "n_ixp_memberships",
]
EDGE_COLS = ["source", "target", "weight"]

GRAPH_CONFIG = {
    "target_as": None,   # set per-pair
    "k_hop": EGO_K_HOP,
    "mode": "ego",
    "compute_spectral": True,
    "betweenness_sample_k": None,
    "max_nodes_for_clique": 5000,
}

PAIRS = {
    "rrc04_as12880": ("rrc04", 12880),
    "rrc04_as3352":  ("rrc04", 3352),
    "rrc05_as12880": ("rrc05", 12880),
    "rrc05_as3352":  ("rrc05", 3352),
}

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("astopo_5m")


# ============================================================================
# MRT path helpers
# ============================================================================

def rib_path(collector: str, dt: datetime) -> Path:
    return MRT_CACHE / collector / f"bview.{dt:%Y%m%d.%H%M}.gz"


def update_path(collector: str, dt: datetime) -> Path:
    return MRT_CACHE / collector / f"updates.{dt:%Y%m%d.%H%M}.gz"


def floor_8h(dt: datetime) -> datetime:
    h = (dt.hour // RIB_RESYNC_HOURS) * RIB_RESYNC_HOURS
    return dt.replace(hour=h, minute=0, second=0, microsecond=0)


# ============================================================================
# Feature extraction for one ego subgraph
# ============================================================================

def compute_node_features(ego_sub: nx.Graph, target_as: int,
                          rel_map: dict, caida_meta: dict, asn_to_ixps) -> pd.DataFrame:
    """Return node feature table matching NODE_COLS schema for every ego node."""
    config = dict(GRAPH_CONFIG)
    config["target_as"] = target_as

    # Build parallel networkit graph (fast centralities).
    nx2nk = {}
    nk2nx = {}
    G_nk = nk.Graph(ego_sub.number_of_nodes(), weighted=False, directed=False)
    for i, node in enumerate(ego_sub.nodes()):
        nx2nk[node] = i
        nk2nx[i] = node
    for u, v in ego_sub.edges():
        if u in nx2nk and v in nx2nk:
            G_nk.addEdge(nx2nk[u], nx2nk[v])

    # Graph-level features populate shared_data with caches (betweenness etc.)
    graph_feats, graph_shared = extract_graph_level_features(
        ego_sub, G_nk, nk2nx, config
    )

    shared_data = {
        "target_node_nx": target_as,
        "target_node_nk": nx2nk.get(target_as),
    }
    shared_data.update(graph_shared)

    node_df, _extra = extract_node_level_features(
        ego_sub, G_nk, nx2nk, nk2nx, shared_data, config
    )

    # Relationship features: n_providers, n_customers, n_peers, provider_ratio,
    # is_tier1. as_paths=[] — we only need per-node counts, not valley-free.
    _, rel_node_df = extract_relationship_features(
        ego_sub, rel_map, caida_meta, target_as=target_as, as_paths=[]
    )
    if rel_node_df is not None and not rel_node_df.empty:
        node_df = node_df.join(rel_node_df, how="left")

    # IXP cosine / membership features.
    if asn_to_ixps:
        try:
            ixp_vectors_df, _ = build_ixp_feature_vectors(ego_sub, asn_to_ixps)
            _, ixp_node_df = compute_ixp_cosine_features(
                ego_sub, ixp_vectors_df, asn_to_ixps
            )
            if ixp_node_df is not None and not ixp_node_df.empty:
                node_df = node_df.join(ixp_node_df, how="left")
        except Exception as e:
            warnings.warn(f"IXP features failed: {e}")

    node_df = node_df.reset_index()
    # Ensure all columns present; fill missing as NaN, then reorder.
    for c in NODE_COLS:
        if c not in node_df.columns:
            node_df[c] = pd.NA
    return node_df[NODE_COLS]


def ego_edges_df(ego_sub: nx.Graph) -> pd.DataFrame:
    """Return edges.csv schema DataFrame (source, target, weight)."""
    rows = []
    for u, v, data in ego_sub.edges(data=True):
        w = data.get("weight", 1)
        if u <= v:
            rows.append((u, v, int(w)))
        else:
            rows.append((v, u, int(w)))
    return pd.DataFrame(rows, columns=EDGE_COLS)


# ============================================================================
# Per-pair driver
# ============================================================================

def process_pair(
    pair_key: str,
    collector: str,
    target_as: int,
    study_start: datetime,
    study_end: datetime,
    rel_map: dict,
    caida_meta: dict,
    asn_to_ixps: dict,
) -> None:
    pair_dir = OUTPUT_DIR / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== pair={pair_key}  collector={collector}  target_as={target_as} ===")
    log.info(f"   window: {study_start.isoformat()}  ->  {study_end.isoformat()}")

    index_rows: list[dict] = []
    topo: IncrementalTopology | None = None
    last_rib_boundary: datetime | None = None
    current = study_start
    n_done = 0
    n_skipped = 0
    n_failed = 0
    t_pair = time.time()

    while current < study_end:
        stamp = current.strftime("%Y%m%d_%H%M")
        win_dir = pair_dir / stamp
        nodes_csv = win_dir / "nodes.csv"
        edges_csv = win_dir / "edges.csv"

        # Skip complete outputs (resumable).
        if nodes_csv.exists() and edges_csv.exists():
            n_skipped += 1
            current += timedelta(minutes=WINDOW_SIZE_MINUTES)
            # Keep index reflecting what's on disk.
            try:
                nodes_n = sum(1 for _ in open(nodes_csv)) - 1
                edges_n = sum(1 for _ in open(edges_csv)) - 1
            except Exception:
                nodes_n = edges_n = -1
            index_rows.append({
                "pair": pair_key, "collector": collector, "stamp": stamp,
                "n_nodes": nodes_n, "n_edges": edges_n, "n_self_loops": 0,
                "n_dup_edges": 0, "snapshot_ts": current.isoformat(),
                "source": "cached",
            })
            continue

        # RIB resync at every 8h boundary.
        boundary = floor_8h(current)
        if last_rib_boundary != boundary:
            rib_p = rib_path(collector, boundary)
            if not rib_p.exists():
                log.warning(f"   missing RIB {rib_p.name}, skipping 8h block")
                last_rib_boundary = boundary
                current = boundary + timedelta(hours=RIB_RESYNC_HOURS)
                continue
            log.info(f"   [rib] loading {rib_p.name}")
            t0 = time.time()
            topo = IncrementalTopology()
            topo.load_rib(str(rib_p))
            annotate_graph_relationships(topo.G, rel_map)
            log.info(f"   [rib] loaded in {time.time()-t0:.1f}s "
                     f"({topo.G.number_of_nodes()} nodes, {topo.G.number_of_edges()} edges)")
            last_rib_boundary = boundary

            # Replay any 5-min updates that fall between the boundary and
            # `current` so we catch up if we're resuming mid-block.
            catchup = boundary
            while catchup < current:
                up_p = update_path(collector, catchup)
                if up_p.exists():
                    df_up = parse_updates_bgpkit(up_p)
                    if not df_up.empty:
                        recs = [(r["type"], r["peer_as"], r["prefix"],
                                 r["as_path_clean"] or r["as_path"])
                                for _, r in df_up.iterrows()]
                        topo.apply_window_updates(recs)
                catchup += timedelta(minutes=WINDOW_SIZE_MINUTES)

        # Apply this 5-min window's updates.
        up_p = update_path(collector, current)
        if up_p.exists():
            try:
                df_up = parse_updates_bgpkit(up_p)
                if not df_up.empty:
                    recs = [(r["type"], r["peer_as"], r["prefix"],
                             r["as_path_clean"] or r["as_path"])
                            for _, r in df_up.iterrows()]
                    topo.apply_window_updates(recs)
            except Exception as e:
                log.warning(f"   [upd] {up_p.name} parse failed: {e}")

        # Extract ego subgraph around target AS.
        try:
            target_int = int(target_as)
            if target_int not in topo.G:
                log.info(f"   [win {stamp}] target AS not in topology — skipping")
                n_failed += 1
                current += timedelta(minutes=WINDOW_SIZE_MINUTES)
                continue
            ego_sub = topo.get_ego_subgraph(target_int, EGO_K_HOP)
            if ego_sub.number_of_nodes() < 3:
                log.info(f"   [win {stamp}] ego<3 nodes — skipping")
                n_failed += 1
                current += timedelta(minutes=WINDOW_SIZE_MINUTES)
                continue

            nodes_df = compute_node_features(ego_sub, target_int, rel_map,
                                             caida_meta, asn_to_ixps)
            edges_df = ego_edges_df(ego_sub)

            win_dir.mkdir(parents=True, exist_ok=True)
            nodes_df.to_csv(nodes_csv, index=False)
            edges_df.to_csv(edges_csv, index=False)
            n_done += 1
            if n_done % 24 == 0:
                rate = n_done / (time.time() - t_pair + 1e-9)
                eta_h = ((study_end - current).total_seconds() / 60 / WINDOW_SIZE_MINUTES) / max(rate, 1e-6) / 3600
                log.info(f"   [win {stamp}] done={n_done}  skipped={n_skipped}  "
                         f"failed={n_failed}  rate={rate:.2f} win/s  ETA~{eta_h:.1f}h")

            index_rows.append({
                "pair": pair_key, "collector": collector, "stamp": stamp,
                "n_nodes": int(ego_sub.number_of_nodes()),
                "n_edges": int(ego_sub.number_of_edges()),
                "n_self_loops": int(nx.number_of_selfloops(ego_sub)),
                "n_dup_edges": 0,
                "snapshot_ts": current.isoformat(),
                "source": "fresh",
            })
        except Exception as e:
            log.warning(f"   [win {stamp}] extract failed: {e}")
            n_failed += 1

        current += timedelta(minutes=WINDOW_SIZE_MINUTES)

    log.info(f"=== pair={pair_key} done: {n_done} fresh / {n_skipped} cached / "
             f"{n_failed} failed in {(time.time()-t_pair)/3600:.2f}h ===")

    # Append to master index.
    if index_rows:
        new_df = pd.DataFrame(index_rows)
        if INDEX_PATH.exists():
            old = pd.read_csv(INDEX_PATH)
            merged = pd.concat([old, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["pair", "stamp"], keep="last")
        else:
            merged = new_df
        merged.to_csv(INDEX_PATH, index=False)
        log.info(f"   index: {len(merged)} rows total at {INDEX_PATH}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS.keys()), default=None,
                    help="Process a single pair (default: all 4).")
    ap.add_argument("--days", type=int, default=None,
                    help="Process only the first N days (smoke test).")
    args = ap.parse_args()

    if not CAIDA_FILE.exists():
        log.error(f"CAIDA file missing: {CAIDA_FILE}")
        sys.exit(1)
    if not PEERINGDB_JSON.exists():
        log.error(f"PeeringDB file missing: {PEERINGDB_JSON}")
        sys.exit(1)

    log.info(f"Loading CAIDA relationships: {CAIDA_FILE.name}")
    rel_map, caida_meta = load_caida_relationships(CAIDA_FILE)
    log.info(f"   {len(rel_map)//2:,} relationships, "
             f"{len(caida_meta.get('clique', set()))} tier-1, "
             f"{len(caida_meta.get('ixp_ases', set()))} IXP ASes")

    log.info(f"Loading PeeringDB IXP memberships: {PEERINGDB_JSON.name}")
    asn_to_ixps = load_peeringdb_ixp_memberships(PEERINGDB_JSON)
    log.info(f"   {len(asn_to_ixps):,} ASNs with IXP memberships")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs_to_run = [args.pair] if args.pair else list(PAIRS.keys())
    study_end = STUDY_END
    if args.days is not None:
        study_end = STUDY_START + timedelta(days=args.days)

    for pk in pairs_to_run:
        collector, target_as = PAIRS[pk]
        process_pair(pk, collector, target_as, STUDY_START, study_end,
                     rel_map, caida_meta, asn_to_ixps)

    log.info("ALL DONE")


if __name__ == "__main__":
    main()
