#!/usr/bin/env python3
"""
Historical Incident Feature Extractor

Processes documented BGP anomaly events through the same Phase 0 feature
extraction pipeline used for the study-period data, producing per-incident
CSVs that can be merged into Phase 3 training datasets.

This script:
  1. Downloads RIB + UPDATE MRT files for each incident date from RIPE RIS.
  2. Initialises the IncrementalTopology from the closest RIB snapshot.
  3. Processes 5-minute windows through the full graph + statistical pipeline.
  4. Outputs per-incident feature CSVs with incident metadata columns.

Requirements:
  - bgpkit-parser (pip install bgpkit-parser)
  - graph_features_lib.py and stat_features_lib.py in Scripts/
  - CAIDA relationship file and PeeringDB IXP data in bgp_unified_results/

Usage:
    ./.venv/bin/python Scripts/historical_incident_extractor.py --list
    ./.venv/bin/python Scripts/historical_incident_extractor.py --incident facebook_outage_2021
    ./.venv/bin/python Scripts/historical_incident_extractor.py --priority-batch
    ./.venv/bin/python Scripts/historical_incident_extractor.py --priority-batch --collector rrc05
    ./.venv/bin/python Scripts/historical_incident_extractor.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import numpy as np
import pandas as pd

import networkit as nk  # fail fast if missing — required for graph features

from graph_features_lib import (
    IncrementalTopology,
    annotate_graph_relationships,
    download_caida_rel2,
    download_mrt_file,
    build_ixp_feature_vectors,
    extract_graph_level_features,
    extract_node_level_features,
    extract_relationship_features,
    compute_ixp_cosine_features,
    find_caida_date,
    load_caida_relationships,
    load_peeringdb_ixp_memberships,
)
from stat_features_lib import (
    extract_statistical_features,
    parse_as_path_clean,
)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = PROJECT_ROOT / "bgp_unified_results"
REFERENCE_DIR = BASE_DIR / "reference_data"
CAIDA_DIR = REFERENCE_DIR / "caida"
MRT_CACHE = PROJECT_ROOT / "shared_cache" / "mrt_files"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "phase3_labeled" / "historical_incidents"

TARGET_ASES = [12880, 3352]
EGO_K_HOP = 2
WINDOW_SIZE_MINUTES = 5
# Earliest monthly CAIDA snapshot known to be available in our archive.
# Incidents before this date will skip CAIDA relationship features.
CAIDA_MIN_SUPPORTED_DATE = datetime(2015, 12, 1).date()

GRAPH_CONFIG = {
    "target_as": None,   # set per-run
    "k_hop": EGO_K_HOP,
    "mode": "ego",
    "compute_spectral": True,
    "betweenness_sample_k": None,
    "max_nodes_for_clique": 5000,
}


# ============================================================================
# Priority incident catalog (curated for visibility at rrc04)
# ============================================================================
# Selected criteria:
#   - rrc04 collector (Amsterdam) — matches our study setup
#   - High-impact events likely to affect AS3352 (Telefonica) or AS12880
#     ego-networks (via Tier-1 path changes, global route leaks)
#   - Mix of PH, PM, DoS types
#   - Post-2008 (better MRT availability)

PRIORITY_INCIDENTS = {
    # === PREFIX HIJACKING ===
    "pakistan_youtube": {
        "name": "Pakistan-YouTube Hijack",
        "date": "2008-02-24",
        "start_time": "18:47",
        "end_time": "21:01",
        "rrc": "rrc04",
        "type": "PH",
        "label": "prefix_hijacking",
        "duration_minutes": 134,
    },
    "rostelecom": {
        "name": "Rostelecom Route Leak (8800+ prefixes)",
        "date": "2020-04-01",
        "start_time": "19:28",
        "end_time": "20:30",
        "rrc": "rrc04",
        "type": "PH",
        "label": "prefix_hijacking",
        "duration_minutes": 62,
    },
    "mainone_google": {
        "name": "MainOne-Google Hijack",
        "date": "2018-11-12",
        "start_time": "21:10",
        "end_time": "22:35",
        "rrc": "rrc04",
        "type": "PH",
        "label": "prefix_hijacking",
        "duration_minutes": 85,
    },
    "klayswap": {
        "name": "KlaySwap BGP Hijack ($1.9M theft)",
        "date": "2022-02-03",
        "start_time": "10:04",
        "end_time": "13:00",
        "rrc": "rrc04",
        "type": "PH",
        "label": "prefix_hijacking",
        "duration_minutes": 176,
    },
    "orange_spain_2024": {
        "name": "Orange Spain RIPE Account Hijack",
        "date": "2024-01-03",
        "start_time": "09:30",
        "end_time": "13:00",
        "rrc": "rrc04",
        "type": "PH",
        "label": "prefix_hijacking",
        "duration_minutes": 210,
    },

    # === PATH MANIPULATION / ROUTE LEAKS ===
    "verizon_dqe_leak": {
        "name": "Verizon/DQE Route Leak (Cloudflare, Discord, Reddit)",
        "date": "2019-06-24",
        "start_time": "10:30",
        "end_time": "12:12",
        "rrc": "rrc04",
        "type": "PM",
        "label": "path_manipulation",
        "duration_minutes": 102,
    },
    "centurylink_leak_2018": {
        "name": "CenturyLink Major Route Leak (911 disrupted)",
        "date": "2018-12-27",
        "start_time": "09:00",
        "end_time": "22:00",
        "rrc": "rrc04",
        "type": "PM",
        "label": "path_manipulation",
        "duration_minutes": 780,
    },
    "china_telecom": {
        "name": "China Telecom Route Leak (~15% global routes)",
        "date": "2010-04-08",
        "start_time": "07:00",
        "end_time": "07:18",
        "rrc": "rrc04",
        "type": "PM",
        "label": "path_manipulation",
        "duration_minutes": 18,
    },
    "cogent_route_leak_2021": {
        "name": "Cogent Major Route Leak",
        "date": "2021-07-22",
        "start_time": "14:00",
        "end_time": "16:00",
        "rrc": "rrc04",
        "type": "PM",
        "label": "path_manipulation",
        "duration_minutes": 120,
    },
    "china_telecom_safehost": {
        "name": "SafeHost-China Telecom Leak (40K routes)",
        "date": "2019-06-06",
        "start_time": "12:00",
        "end_time": "16:00",
        "rrc": "rrc04",
        "type": "PM",
        "label": "path_manipulation",
        "duration_minutes": 240,
    },

    # === DoS / OUTAGE ===
    "facebook_outage_2021": {
        "name": "Facebook/Meta Global Outage (3.5B users)",
        "date": "2021-10-04",
        "start_time": "15:39",
        "end_time": "21:00",
        "rrc": "rrc04",
        "type": "DoS",
        "label": "dos_attack",
        "duration_minutes": 321,
    },
    "level3_leak": {
        "name": "Level 3 Route Leak (Netflix, Comcast)",
        "date": "2017-11-06",
        "start_time": "17:47",
        "end_time": "19:20",
        "rrc": "rrc04",
        "type": "DoS",
        "label": "dos_attack",
        "duration_minutes": 93,
    },
    "google_japan": {
        "name": "Google Japan BGP Leak (135K prefixes)",
        "date": "2017-08-25",
        "start_time": "03:22",
        "end_time": "03:30",
        "rrc": "rrc04",
        "type": "DoS",
        "label": "dos_attack",
        "duration_minutes": 8,
    },
    "dyn_dns_attack_2016": {
        "name": "Dyn DNS DDoS (Mirai Botnet)",
        "date": "2016-10-21",
        "start_time": "11:10",
        "end_time": "18:36",
        "rrc": "rrc04",
        "type": "DoS",
        "label": "dos_attack",
        "duration_minutes": 446,
    },
    "vodafone_india": {
        "name": "Vodafone India Route Leak (30K prefixes)",
        "date": "2021-04-17",
        "start_time": "09:00",
        "end_time": "15:00",
        "rrc": "rrc04",
        "type": "DoS",
        "label": "dos_attack",
        "duration_minutes": 360,
    },
}


# ============================================================================
# Helpers
# ============================================================================


def closest_rib_time(incident_dt: datetime) -> datetime:
    """Find the closest RIB snapshot time (every 8h: 00:00, 08:00, 16:00) BEFORE the incident."""
    hour = (incident_dt.hour // 8) * 8
    rib_dt = incident_dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    if rib_dt > incident_dt:
        rib_dt -= timedelta(hours=8)
    return rib_dt


def rib_url(collector: str, rib_dt: datetime) -> str:
    return f"https://data.ris.ripe.net/{collector}/{rib_dt:%Y.%m}/bview.{rib_dt:%Y%m%d.%H%M}.gz"


def update_url(collector: str, ts: datetime) -> str:
    return f"https://data.ris.ripe.net/{collector}/{ts:%Y.%m}/updates.{ts:%Y%m%d.%H%M}.gz"


def download_with_retry(url: str, dest_dir: Path, collector: str, max_retries: int = 3) -> Path | None:
    """Download an MRT file with retries."""
    for attempt in range(max_retries):
        try:
            return download_mrt_file(url, dest_dir, collector=collector)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return None


def load_caida_for_date(incident_date_str: str) -> tuple[dict, dict, dict]:
    """Load CAIDA relationship data matched to the incident date.

    Returns (rel_map, caida_meta, caida_info) where caida_info records which
    file was actually used (for auditability in the summary).
    """
    caida_date = find_caida_date(incident_date_str)  # e.g. "20170801"
    CAIDA_DIR.mkdir(parents=True, exist_ok=True)

    # Look for an existing file matching this date (explicit .txt to avoid
    # picking compressed artifacts like .as-rel2.txt.gz)
    caida_path = None
    for candidate in sorted(CAIDA_DIR.glob("*.as-rel2.txt")):
        if caida_date in candidate.name:
            caida_path = candidate
            break

    if caida_path is None:
        print(f"  Downloading CAIDA file for {caida_date}...")
        try:
            caida_path = download_caida_rel2(caida_date, CAIDA_DIR)
        except Exception as e:
            warnings.warn(
                f"CAIDA download for {caida_date} failed ({e}). "
                f"Falling back to latest available — relationship features may "
                f"reflect a different time period than the incident."
            )
            all_caida = sorted(CAIDA_DIR.glob("*.as-rel2.txt"))
            if not all_caida:
                for root in [REFERENCE_DIR, PROJECT_ROOT / "runs"]:
                    if root.exists():
                        all_caida = sorted(root.rglob("*.as-rel2.txt"))
                        if all_caida:
                            break
            if not all_caida:
                raise FileNotFoundError("No CAIDA relationship file found anywhere.")
            caida_path = all_caida[-1]

    caida_exact_match = caida_date in caida_path.name

    print(f"  CAIDA file: {caida_path.name}"
          f"{'' if caida_exact_match else ' [FALLBACK — date mismatch]'}")

    rel_map, caida_meta = load_caida_relationships(caida_path)
    print(f"  CAIDA: {len(rel_map)//2:,} relationships, "
          f"{len(caida_meta.get('clique', set()))} tier-1, "
          f"{len(caida_meta.get('ixp_ases', set()))} IXP ASes")

    caida_info = {
        "requested_date": caida_date,
        "file_used": caida_path.name,
        "exact_match": caida_exact_match,
    }

    return rel_map, caida_meta, caida_info


def parse_updates_bgpkit(mrt_path: Path) -> pd.DataFrame:
    """Parse an MRT UPDATE file using bgpkit-parser into the DataFrame format
    expected by extract_statistical_features."""
    import bgpkit

    records = []
    try:
        parser = bgpkit.Parser(url=str(mrt_path))
        for elem in parser:
            ts = datetime.fromtimestamp(elem.timestamp, tz=timezone.utc)
            entry_type = elem.elem_type
            if entry_type not in ("A", "W"):
                continue

            as_path_raw = elem.as_path or ""
            as_path_clean = parse_as_path_clean(as_path_raw) if as_path_raw else ""
            origin = ""
            if as_path_clean:
                parts = as_path_clean.split()
                origin = parts[-1] if parts else ""

            records.append({
                "timestamp": ts,
                "type": entry_type,
                "peer_ip": elem.peer_ip or "",
                "peer_as": str(elem.peer_asn or ""),
                "prefix": elem.prefix or "",
                "as_path": as_path_raw,
                "as_path_clean": as_path_clean,
                "origin": origin,
                "next_hop": elem.next_hop or "",
            })
    except Exception as e:
        print(f"    Warning: bgpkit parse error on {mrt_path.name}: {e}")

    if not records:
        return pd.DataFrame(columns=[
            "timestamp", "type", "peer_ip", "peer_as", "prefix",
            "as_path", "as_path_clean", "origin", "next_hop",
        ])

    return pd.DataFrame(records)


# ============================================================================
# Core extraction
# ============================================================================


def extract_incident_features(
    incident_key: str,
    incident_cfg: dict,
    target_as: int,
    rel_map: dict,
    caida_meta: dict,
    asn_to_ixps: dict,
    enable_relationship_features: bool = True,
) -> pd.DataFrame | None:
    """
    Extract features for a single incident and target AS.
    Returns a DataFrame with one row per 5-minute window.
    """
    collector = incident_cfg["rrc"]
    start_time = datetime.strptime(
        f"{incident_cfg['date']} {incident_cfg['start_time']}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(
        f"{incident_cfg['date']} {incident_cfg['end_time']}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=timezone.utc)
    if end_time <= start_time:
        end_time += timedelta(days=1)

    # Extend collection window: 1h before, 1h after
    collection_start = start_time - timedelta(hours=1)
    collection_end = end_time + timedelta(hours=1)

    print(f"\n  Target AS: {target_as}")
    print(f"  Collection window: {collection_start:%H:%M} - {collection_end:%H:%M} UTC")

    # --- Download RIB ---
    rib_dt = closest_rib_time(collection_start)
    # download_mrt_file appends collector/ to dest_dir, so pass MRT_CACHE directly
    MRT_CACHE.mkdir(parents=True, exist_ok=True)

    rib_file_url = rib_url(collector, rib_dt)
    print(f"  Downloading RIB ({rib_dt:%Y%m%d.%H%M})...")
    rib_path = download_with_retry(rib_file_url, MRT_CACHE, collector)
    if rib_path is None:
        print(f"    ERROR: Could not download RIB. Skipping.")
        return None

    # --- Download UPDATE files ---
    # Warm-up updates: from RIB time to collection_start (evolve topology only)
    # Feature-extraction updates: from collection_start to collection_end
    warmup_paths = []
    extraction_paths = []

    # Round to 5-minute boundary
    warmup_start = rib_dt.replace(second=0, microsecond=0)
    warmup_start = warmup_start.replace(minute=(warmup_start.minute // 5) * 5)
    extraction_start = collection_start.replace(second=0, microsecond=0)
    extraction_start = extraction_start.replace(minute=(extraction_start.minute // 5) * 5)

    # Download warm-up updates (RIB time → collection_start)
    current = warmup_start
    while current < extraction_start:
        url_str = update_url(collector, current)
        path = download_with_retry(url_str, MRT_CACHE, collector)
        if path is not None:
            warmup_paths.append((path, current))
        current += timedelta(minutes=5)

    # Download extraction-window updates (collection_start → collection_end)
    current = extraction_start
    while current <= collection_end:
        url_str = update_url(collector, current)
        path = download_with_retry(url_str, MRT_CACHE, collector)
        if path is not None:
            extraction_paths.append((path, current))
        current += timedelta(minutes=5)

    print(f"  Warm-up UPDATE files: {len(warmup_paths)} (RIB → collection start)")
    print(f"  Extraction UPDATE files: {len(extraction_paths)}")
    if not extraction_paths:
        print(f"    ERROR: No extraction UPDATE files downloaded. Skipping.")
        return None

    # --- Initialize topology ---
    print(f"  Loading RIB into topology...")
    topo = IncrementalTopology()
    topo.load_rib(str(rib_path))
    info = topo.snapshot_info()
    print(f"    Topology: {info['n_nodes']:,} nodes, {info['n_edges']:,} edges")

    target_int = int(target_as)
    if target_int not in topo.G:
        print(f"    WARNING: AS{target_as} not found in topology. Skipping.")
        return None

    # Annotate relationships only when CAIDA relationships are enabled.
    if enable_relationship_features:
        annotate_graph_relationships(topo.G, rel_map)
    else:
        print(
            f"  CAIDA relationship features: disabled "
            f"(incident before {CAIDA_MIN_SUPPORTED_DATE.isoformat()})"
        )

    # --- Warm-up: apply updates from RIB time to collection_start ---
    # This brings the topology to the correct state before feature extraction
    print(f"  Warming up topology ({len(warmup_paths)} files)...")
    for warmup_path, warmup_ts in warmup_paths:
        df_warmup = parse_updates_bgpkit(warmup_path)
        if not df_warmup.empty:
            records_for_topo = [
                (row["type"], row["peer_as"], row["prefix"], row["as_path_clean"] or row["as_path"])
                for _, row in df_warmup.iterrows()
            ]
            topo.apply_window_updates(records_for_topo)

    info_after_warmup = topo.snapshot_info()
    print(f"    Topology after warm-up: {info_after_warmup['n_nodes']:,} nodes, "
          f"{info_after_warmup['n_edges']:,} edges")

    # --- Process extraction windows ---
    config = dict(GRAPH_CONFIG)
    config["target_as"] = target_as

    results = []
    prev_prefixes = None

    # Process feature-extraction windows
    for update_path, update_ts in extraction_paths:
        window_id = update_ts.strftime("%Y%m%d_%H%M")
        window_end = update_ts + timedelta(minutes=WINDOW_SIZE_MINUTES)
        is_during_incident = (update_ts < end_time) and (window_end > start_time)

        # Parse UPDATE file
        df_update = parse_updates_bgpkit(update_path)
        if df_update.empty:
            continue

        # Apply to topology
        records_for_topo = [
            (row["type"], row["peer_as"], row["prefix"], row["as_path_clean"] or row["as_path"])
            for _, row in df_update.iterrows()
        ]
        topo.apply_window_updates(records_for_topo)

        # Extract ego subgraph
        try:
            ego_sub = topo.get_ego_subgraph(target_as, EGO_K_HOP)
        except Exception:
            continue

        if ego_sub.number_of_nodes() < 3:
            continue

        # --- Graph features ---
        try:
            nx2nk_map = {}
            nk2nx_map = {}
            G_nk = nk.Graph(ego_sub.number_of_nodes(), weighted=False, directed=False)
            for i, node in enumerate(ego_sub.nodes()):
                nx2nk_map[node] = i
                nk2nx_map[i] = node
            for u, v in ego_sub.edges():
                if u in nx2nk_map and v in nx2nk_map:
                    G_nk.addEdge(nx2nk_map[u], nx2nk_map[v])

            graph_feats, graph_shared = extract_graph_level_features(ego_sub, G_nk, nk2nx_map, config)

            shared_data = {
                "target_node_nx": target_int,
                "target_node_nk": nx2nk_map.get(target_int),
            }
            shared_data.update(graph_shared)
            node_feats_df, extra_graph_feats = extract_node_level_features(
                ego_sub, G_nk, nx2nk_map, nk2nx_map, shared_data, config
            )
            graph_feats.update(extra_graph_feats)

            # node_feats_df is a DataFrame indexed by ASN (all ego nodes).
            # Extract only the target AS row as a scalar dict.
            if target_int in node_feats_df.index:
                node_feats = node_feats_df.loc[target_int].to_dict()
            else:
                node_feats = {}

            # Relationship features (optional by incident date policy)
            rel_feats = {}
            rel_node_feats = {}
            if enable_relationship_features:
                as_paths = df_update[df_update["as_path_clean"].notna()]["as_path_clean"].tolist()
                rel_feats, rel_node_df = extract_relationship_features(
                    ego_sub, rel_map, caida_meta, target_as=target_int, as_paths=as_paths
                )
                # Extract target AS node-level relationship features
                if rel_node_df is not None and target_int in rel_node_df.index:
                    rel_node_feats = rel_node_df.loc[target_int].to_dict()

            # IXP cosine features
            ixp_feats = {}
            ixp_node_feats = {}
            if asn_to_ixps:
                try:
                    ixp_vectors_df, _ = build_ixp_feature_vectors(ego_sub, asn_to_ixps)
                    ixp_graph_feats, ixp_node_df = compute_ixp_cosine_features(
                        ego_sub, ixp_vectors_df, asn_to_ixps
                    )
                    ixp_feats = ixp_graph_feats
                    # Extract target AS node-level IXP features
                    if ixp_node_df is not None and target_int in ixp_node_df.index:
                        ixp_node_feats = ixp_node_df.loc[target_int].to_dict()
                except Exception as e:
                    warnings.warn(f"IXP feature extraction failed for {window_id}: {e}")

        except Exception as e:
            warnings.warn(f"Graph feature extraction failed for {window_id}: {e}")
            graph_feats = {}
            node_feats = {}
            rel_feats = {}
            ixp_feats = {}

        # --- Ego filter ratio ---
        # Matches study-period pipeline: ratio of updates involving ego-network ASes
        ego_nodes = set(ego_sub.nodes())
        total_updates = len(df_update)
        if total_updates > 0:
            ego_mask = df_update["as_path_clean"].fillna("").apply(
                lambda p: bool(ego_nodes & {int(a) for a in p.split() if a.isdigit()})
            )
            ego_updates = int(ego_mask.sum())
            ego_filter_ratio = ego_updates / total_updates
        else:
            ego_updates = 0
            ego_filter_ratio = 0.0

        # --- Statistical features ---
        try:
            stat_feats, current_prefixes = extract_statistical_features(
                df_update, prev_prefixes=prev_prefixes, rare_as_threshold=3
            )
            prev_prefixes = current_prefixes
        except Exception as e:
            warnings.warn(f"Stat feature extraction failed for {window_id}: {e}")
            stat_feats = {}

        # Combine
        row = {
            "window_start": update_ts.isoformat(),
            "window_id": window_id,
            "collector": collector,
        }
        row.update(graph_feats)
        row.update(node_feats)
        row.update(rel_feats)
        row.update(rel_node_feats)
        row.update(ixp_feats)
        row.update(ixp_node_feats)
        row.update(stat_feats)

        # Ego filtering stats (matches study-period pipeline)
        row["ego_updates_total"] = total_updates
        row["ego_updates_filtered"] = ego_updates
        row["ego_filter_ratio"] = round(ego_filter_ratio, 6)

        # Incident metadata and event identity
        row["incident_key"] = incident_key
        row["incident_name"] = incident_cfg["name"]
        row["incident_type"] = incident_cfg["type"]
        row["incident_label"] = incident_cfg["label"]
        row["is_during_incident"] = int(is_during_incident)
        row["binary_label"] = 1 if is_during_incident else 0
        # Unique event_id: ties this window to a specific incident for
        # event-level splitting.  Format: hist_{incident}_{domain}
        # All windows from the same incident+domain share one event_id.
        # Windows outside the incident window get event_id = "none".
        domain_key_local = f"{collector}_as{target_as}"
        row["event_id"] = (
            f"hist_{incident_key}_{domain_key_local}" if is_during_incident else "none"
        )
        row["provenance"] = "historical"

        results.append(row)

    if not results:
        print(f"    No windows extracted.")
        return None

    df = pd.DataFrame(results)
    print(f"    Extracted {len(df)} windows ({df['binary_label'].sum()} during incident)")
    return df


# ============================================================================
# Post-extraction visibility check
# ============================================================================

# Minimum thresholds for declaring an incident "visible"
VISIBILITY_MIN_DEVIANT_FEATURES = 3    # at least N features must deviate
VISIBILITY_ZSCORE_THRESHOLD = 2.0      # deviation must exceed this z-score
VISIBILITY_MAHALANOBIS_FLOOR = 0.5     # minimum normalized Mahalanobis distance
# Skip features whose padding std is below this fraction of |padding mean|
# (or below the absolute floor) — they are essentially constant and any
# incident-window difference is meaningless noise that would otherwise
# produce astronomical z-scores.
VISIBILITY_MIN_REL_STD = 1e-4   # std must be at least 0.01% of |mean|
VISIBILITY_MIN_ABS_STD = 1e-6   # absolute floor for features with mean ~ 0


def assess_visibility(df: pd.DataFrame, incident_key: str, domain_key: str) -> dict:
    """
    Check whether incident windows show a measurable feature deviation from
    the padding (non-incident) windows.

    Strategy:
      1. Split into padding (binary_label=0) and incident (binary_label=1).
      2. Compute per-feature z-scores of the incident-window means relative
         to the padding distribution.
      3. Count how many features exceed the z-score threshold.
      4. Compute a global Mahalanobis-like distance (mean z-score across
         top-k most deviant features).
      5. Flag the incident as visible / marginal / invisible.

    Returns a dict with diagnostics and a verdict.
    """
    meta_and_label_cols = {
        "window_start", "window_id", "collector",
        "incident_key", "incident_name", "incident_type",
        "incident_label", "is_during_incident", "binary_label",
        "event_id", "provenance", "visibility",
        "ego_updates_total", "ego_updates_filtered",
    }
    feature_cols = [c for c in df.columns if c not in meta_and_label_cols]

    padding = df[df["binary_label"] == 0]
    incident = df[df["binary_label"] == 1]

    result = {
        "incident": incident_key,
        "domain": domain_key,
        "n_padding_windows": len(padding),
        "n_incident_windows": len(incident),
    }

    if len(incident) == 0:
        result.update({"verdict": "no_incident_windows", "visible": False})
        return result

    if len(padding) < 3:
        # Not enough padding to compute meaningful statistics; trust the label
        result.update({
            "verdict": "insufficient_padding",
            "visible": True,
            "note": "Too few padding windows for statistical comparison; "
                    "incident label retained by default.",
        })
        return result

    # Compute numeric features only — features whose padding values are
    # essentially constant (std below the relative/absolute floor) are
    # discarded entirely, since any incident-window difference is meaningless
    # numerical noise that would otherwise produce astronomical z-scores.
    numeric_features = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if padding[col].notna().sum() < 3:
            continue
        pad_std = padding[col].std(ddof=1)
        pad_mean = padding[col].mean()
        if not np.isfinite(pad_std) or not np.isfinite(pad_mean):
            continue
        min_meaningful_std = max(
            VISIBILITY_MIN_REL_STD * abs(pad_mean),
            VISIBILITY_MIN_ABS_STD,
        )
        if pad_std < min_meaningful_std:
            continue  # essentially constant — uninformative
        numeric_features.append(col)

    if not numeric_features:
        result.update({"verdict": "no_numeric_features", "visible": False})
        return result

    # Per-feature z-score of incident mean vs padding distribution.
    # By construction every feature in `numeric_features` has a meaningful std.
    z_scores = {}
    for col in numeric_features:
        pad_mean = padding[col].mean()
        pad_std = padding[col].std(ddof=1)
        inc_mean = incident[col].mean()
        z = abs(inc_mean - pad_mean) / pad_std
        z_scores[col] = round(z, 3)

    # Count deviant features
    deviant = {f: z for f, z in z_scores.items() if z >= VISIBILITY_ZSCORE_THRESHOLD}
    n_deviant = len(deviant)

    # Global distance: mean of top-10 z-scores
    top_z = sorted(z_scores.values(), reverse=True)[:10]
    mean_top_z = float(np.mean(top_z)) if top_z else 0.0

    # Verdict
    if n_deviant >= VISIBILITY_MIN_DEVIANT_FEATURES and mean_top_z >= VISIBILITY_MAHALANOBIS_FLOOR:
        verdict = "visible"
        visible = True
    elif n_deviant >= 1 and mean_top_z >= VISIBILITY_MAHALANOBIS_FLOOR:
        verdict = "marginal"
        visible = True
    else:
        verdict = "invisible"
        visible = False

    # Top deviant features for reporting
    top_deviant = dict(sorted(deviant.items(), key=lambda x: -x[1])[:10])

    # Full z-score table sorted descending (saved to disk, not just top-10)
    all_z_sorted = dict(sorted(z_scores.items(), key=lambda x: -x[1]))

    # Per-feature detail: padding mean, incident mean, z-score
    feature_detail = {}
    for col in numeric_features:
        feature_detail[col] = {
            "padding_mean": round(float(padding[col].mean()), 6),
            "padding_std": round(float(padding[col].std(ddof=1)), 6),
            "incident_mean": round(float(incident[col].mean()), 6),
            "z_score": z_scores[col],
            "deviant": z_scores[col] >= VISIBILITY_ZSCORE_THRESHOLD,
        }

    result.update({
        "verdict": verdict,
        "visible": visible,
        "n_numeric_features": len(numeric_features),
        "n_deviant_features": n_deviant,
        "mean_top10_zscore": round(mean_top_z, 3),
        "top_deviant_features": top_deviant,
        "all_z_scores": all_z_sorted,
        "feature_detail": feature_detail,
        "thresholds": {
            "z_score_threshold": VISIBILITY_ZSCORE_THRESHOLD,
            "min_deviant_features": VISIBILITY_MIN_DEVIANT_FEATURES,
            "mahalanobis_floor": VISIBILITY_MAHALANOBIS_FLOOR,
            "min_rel_std": VISIBILITY_MIN_REL_STD,
            "min_abs_std": VISIBILITY_MIN_ABS_STD,
        },
    })

    return result


def save_visibility_report(vis: dict, output_dir: Path) -> Path:
    """Save the full visibility report to a JSON file for later inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{vis['incident']}_{vis['domain']}_visibility.json"
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vis, f, indent=2, default=str)
    return path


def print_visibility_report(vis: dict) -> None:
    """Pretty-print a visibility assessment."""
    verdict = vis["verdict"]
    tag = {"visible": "VISIBLE", "marginal": "MARGINAL", "invisible": "INVISIBLE"}
    symbol = {"visible": "+", "marginal": "~", "invisible": "-"}

    s = symbol.get(verdict, "?")
    t = tag.get(verdict, verdict.upper())

    print(f"    Visibility: [{s}] {t}")
    if "n_deviant_features" in vis:
        print(f"      Deviant features: {vis['n_deviant_features']}/{vis['n_numeric_features']} "
              f"(z >= {VISIBILITY_ZSCORE_THRESHOLD})")
        print(f"      Mean top-10 z-score: {vis['mean_top10_zscore']:.3f}")
        top = vis.get("top_deviant_features", {})
        if top:
            preview = ", ".join(f"{f}={z:.1f}" for f, z in list(top.items())[:5])
            print(f"      Top deviating: {preview}")
    if "note" in vis:
        print(f"      Note: {vis['note']}")


# ============================================================================
# Rebuild combined outputs from existing per-incident CSVs
# ============================================================================


def _rebuild_combined_outputs() -> None:
    """
    Rebuild all combined output files from the existing per-incident CSVs
    and visibility reports.  This is fast (no MRT downloads) and useful
    after fixing visibility logic or when re-combining after separate
    collector batch runs.
    """
    print("Rebuilding combined outputs from existing per-incident CSVs...")
    vis_dir = OUTPUT_DIR / "visibility_reports"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Visibility will be recomputed from each per-incident CSV below so
    #    that any change to the visibility logic (e.g. z-score floor) takes
    #    effect without re-downloading MRTs.  We still keep a dict of fresh
    #    verdicts for the second pass.
    visibility: dict[str, dict] = {}

    # 2. Load all per-incident CSVs and classify
    all_incident_csvs = sorted(OUTPUT_DIR.glob("*_rrc0*_as*.csv"))
    # Exclude combined files
    exclude_prefixes = ("all_historical", "visible_historical", "unified_historical",
                        "extraction_summary", "historical_event_manifest",
                        "historical_as_unified")
    per_incident_csvs = [
        f for f in all_incident_csvs
        if not f.name.startswith(exclude_prefixes)
    ]
    print(f"  Found {len(per_incident_csvs)} per-incident CSVs")

    # 3. Build per-domain combined files (visible only) and summary
    domain_dfs: dict[str, list[pd.DataFrame]] = {}
    summary_rows = []
    hist_manifest_rows = []

    for csv_path in per_incident_csvs:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Extract incident key and domain from filename
        # Format: {incident_key}_{collector}_as{asn}.csv
        fname = csv_path.stem  # e.g., "rostelecom_rrc05_as12880"
        # Find domain part: rrc0X_asYYYY
        parts = fname.split("_")
        domain_idx = None
        for i, p in enumerate(parts):
            if p.startswith("rrc0"):
                domain_idx = i
                break
        if domain_idx is None:
            continue
        domain_key = "_".join(parts[domain_idx:])  # e.g., "rrc05_as12880"
        incident_key = "_".join(parts[:domain_idx])  # e.g., "rostelecom"

        vis_key = f"{incident_key}_{domain_key}"
        # Recompute visibility from the CSV so the latest assessment logic
        # (z-score floor, etc.) is applied — then persist the fresh report.
        vis_info = assess_visibility(df, incident_key, domain_key)
        try:
            save_visibility_report(vis_info, vis_dir)
        except Exception as exc:
            print(f"  WARN: failed to save visibility report for {vis_key}: {exc}")
        visibility[vis_key] = vis_info
        verdict = vis_info.get("verdict", "unknown")
        is_visible = vis_info.get("visible", False)

        # Get collector and target_as
        collector = domain_key.split("_")[0]  # rrc05
        target_as_str = domain_key.split("_as")[-1]  # 12880
        target_as = int(target_as_str) if target_as_str.isdigit() else 0

        # Get incident config
        inc_cfg = PRIORITY_INCIDENTS.get(incident_key, {})

        # Summary row
        n_incident_windows = int(df["binary_label"].sum()) if "binary_label" in df.columns else 0
        summary_rows.append({
            "incident": incident_key,
            "domain": domain_key,
            "target_as": target_as,
            "total_windows": len(df),
            "anomaly_windows": n_incident_windows,
            "type": inc_cfg.get("type", ""),
            "elapsed_sec": 0.0,  # not available in rebuild
            "verdict": verdict,
            "visible": is_visible,
            "n_deviant_features": vis_info.get("n_deviant_features", 0),
            "mean_top10_zscore": vis_info.get("mean_top10_zscore", 0.0),
            "caida_file": "rebuilt",
            "caida_exact_match": True,
            "caida_enabled": True,
            "caida_reason": "",
        })

        # Only include visible incidents in combined outputs
        if is_visible:
            domain_dfs.setdefault(domain_key, []).append(df)

            # Manifest rows for visible incident windows
            event_windows = df[df.get("event_id", pd.Series(dtype=str)) != "none"]
            if not event_windows.empty:
                for event_id, group in event_windows.groupby("event_id"):
                    group_sorted = group.sort_values("window_start")
                    hist_manifest_rows.append({
                        "domain": domain_key,
                        "collector": collector,
                        "event_id": event_id,
                        "start_window": group_sorted["window_id"].iloc[0],
                        "end_window": group_sorted["window_id"].iloc[-1],
                        "start_time": str(group_sorted["window_start"].iloc[0]),
                        "end_time": str(group_sorted["window_start"].iloc[-1]),
                        "n_windows": len(group_sorted),
                        "n_anomaly_windows": len(group_sorted),
                        "n_absorbed_uncertain": 0,
                        "dominant_label": group_sorted.get(
                            "incident_label", pd.Series(["anomaly"])
                        ).iloc[0],
                        "incident_key": incident_key,
                        "incident_type": inc_cfg.get("type", ""),
                        "provenance": "historical",
                    })

    # 4. Write combined per-domain files (visible only)
    for domain_key, dfs in domain_dfs.items():
        combined = pd.concat(dfs, ignore_index=True)
        out_path = OUTPUT_DIR / f"all_historical_incidents_{domain_key}.csv"
        combined.to_csv(out_path, index=False)
        n_anom = int(combined["binary_label"].sum()) if "binary_label" in combined.columns else 0
        print(f"  {out_path.name}: {len(combined)} windows ({n_anom} anomaly)")

    # 5. Write unified visible file
    all_visible = []
    for dfs in domain_dfs.values():
        all_visible.extend(dfs)
    if all_visible:
        vis_combined = pd.concat(all_visible, ignore_index=True)
        vis_combined.to_csv(OUTPUT_DIR / "visible_historical_incidents.csv", index=False)
        print(f"  visible_historical_incidents.csv: {len(vis_combined)} windows")

    # 6. Write summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_DIR / "extraction_summary.csv", index=False)
        with open(OUTPUT_DIR / "extraction_summary.json", "w") as f:
            json.dump(summary_rows, f, indent=2, default=str)
        print(f"  extraction_summary.csv: {len(summary_df)} rows")

    # 7. Write event manifest
    if hist_manifest_rows:
        manifest_df = pd.DataFrame(hist_manifest_rows)
        manifest_df.to_csv(OUTPUT_DIR / "historical_event_manifest.csv", index=False)
        print(f"  historical_event_manifest.csv: {len(manifest_df)} events")

    # 8. Print overview
    print(f"\n{'='*70}")
    print("REBUILD SUMMARY")
    print(f"{'='*70}")
    n_visible = sum(1 for r in summary_rows if r["visible"])
    n_invisible = sum(1 for r in summary_rows if not r["visible"])
    print(f"Total incident-domain pairs: {len(summary_rows)}")
    print(f"  Visible: {n_visible}")
    print(f"  Invisible/no-data: {n_invisible}")
    for dk, dfs in sorted(domain_dfs.items()):
        combined = pd.concat(dfs, ignore_index=True)
        n_anom = int(combined["binary_label"].sum())
        n_events = combined[combined.get("event_id", pd.Series(dtype=str)) != "none"]["event_id"].nunique()
        print(f"  {dk}: {len(combined)} windows, {n_anom} anomaly, {n_events} events")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Extract features from historical BGP incidents.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List priority incidents.")
    group.add_argument("--incident", type=str, help="Process a single incident by key.")
    group.add_argument("--priority-batch", action="store_true", help="Process all 15 priority incidents.")
    group.add_argument("--all", action="store_true", help="Process all incidents from both catalogs.")
    group.add_argument(
        "--rebuild-combined", action="store_true",
        help=(
            "Rebuild combined output files (all_historical_incidents_*.csv, "
            "visible_historical_incidents.csv, extraction_summary.csv, "
            "historical_event_manifest.csv) from existing per-incident CSVs "
            "and visibility reports.  No MRT downloads or feature extraction."
        ),
    )
    parser.add_argument(
        "--collector",
        type=str,
        default=None,
        help=(
            "Override the incident collector for all processed incidents "
            "(e.g., rrc05). Defaults to each incident's configured collector."
        ),
    )
    parser.add_argument("--target-as", type=int, nargs="+", default=TARGET_ASES,
                        help=f"Target ASes (default: {TARGET_ASES})")
    args = parser.parse_args()

    if args.list:
        print(f"Priority incidents ({len(PRIORITY_INCIDENTS)}):\n")
        for key, cfg in PRIORITY_INCIDENTS.items():
            print(f"  [{cfg['type']:>3}] {cfg['date']} | {cfg['duration_minutes']:>5}min | {key}")
            print(f"        {cfg['name']}")
        return

    if args.rebuild_combined:
        _rebuild_combined_outputs()
        return

    # Select incidents
    if args.incident:
        if args.incident not in PRIORITY_INCIDENTS:
            print(f"Unknown incident: {args.incident}")
            print("Use --list to see available incidents.")
            return
        incidents = {args.incident: dict(PRIORITY_INCIDENTS[args.incident])}
    elif args.priority_batch:
        incidents = {k: dict(v) for k, v in PRIORITY_INCIDENTS.items()}
    else:
        incidents = {k: dict(v) for k, v in PRIORITY_INCIDENTS.items()}  # extend later with full catalog

    if args.collector:
        collector_override = args.collector.strip().lower()
        for cfg in incidents.values():
            cfg["rrc"] = collector_override
        print(
            f"Collector override enabled: {collector_override} "
            f"(applied to {len(incidents)} incident(s))"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MRT_CACHE.mkdir(parents=True, exist_ok=True)

    # PeeringDB IXP memberships are date-independent enough to load once
    print("Loading PeeringDB IXP memberships...")
    pdb_files: list[Path] = []
    # Search multiple known locations (reference_data, temp, graph results)
    pdb_search_dirs = [
        REFERENCE_DIR,
        BASE_DIR / "temp",
        PROJECT_ROOT / "bgp_graph_features_results" / "data" / "peeringdb",
    ]
    for pdb_dir in pdb_search_dirs:
        if pdb_dir.exists():
            pdb_files.extend(sorted(pdb_dir.glob("peeringdb*.json")))
    # Also search run workspaces as a last resort
    if not pdb_files:
        runs_dir = PROJECT_ROOT / "runs"
        if runs_dir.exists():
            pdb_files = sorted(runs_dir.rglob("peeringdb*.json"))

    asn_to_ixps = {}
    if pdb_files:
        try:
            asn_to_ixps = load_peeringdb_ixp_memberships(pdb_files[-1])
            print(f"  PeeringDB: {pdb_files[-1].name} — "
                  f"{len(asn_to_ixps):,} ASes with IXP memberships")
        except Exception as e:
            warnings.warn(f"PeeringDB load failed: {e}")
    else:
        print("  WARNING: No PeeringDB file found — IXP cosine features will be empty")

    # Process incidents
    all_results = []
    summary = []

    # Cache CAIDA data per date to avoid re-downloading (for supported dates)
    caida_cache: dict[str, tuple[dict, dict, dict]] = {}

    for incident_key, incident_cfg in incidents.items():
        print(f"\n{'='*70}")
        print(f"Incident: {incident_cfg['name']}")
        print(f"Date: {incident_cfg['date']}  Type: {incident_cfg['type']}  Duration: {incident_cfg['duration_minutes']}min")
        print(f"{'='*70}")

        incident_date = datetime.strptime(incident_cfg["date"], "%Y-%m-%d").date()
        use_caida_relationships = incident_date >= CAIDA_MIN_SUPPORTED_DATE

        if use_caida_relationships:
            # Load CAIDA data matched to incident date (cached across same-month incidents)
            caida_date = find_caida_date(incident_cfg["date"])
            if caida_date not in caida_cache:
                print(f"  Loading CAIDA for {caida_date}...")
                rel_map, caida_meta, caida_info = load_caida_for_date(incident_cfg["date"])
                caida_cache[caida_date] = (rel_map, caida_meta, caida_info)
            else:
                rel_map, caida_meta, caida_info = caida_cache[caida_date]
            caida_info["enabled"] = True
            caida_info["reason"] = ""
        else:
            print(
                f"  Skipping CAIDA relationship features: incident date "
                f"{incident_cfg['date']} is before minimum supported "
                f"{CAIDA_MIN_SUPPORTED_DATE.isoformat()}."
            )
            rel_map = {}
            caida_meta = {"clique": set(), "ixp_ases": set(), "sources": []}
            caida_info = {
                "requested_date": find_caida_date(incident_cfg["date"]),
                "file_used": "none",
                "exact_match": False,
                "enabled": False,
                "reason": f"incident_before_{CAIDA_MIN_SUPPORTED_DATE.isoformat()}",
            }

        for target_as in args.target_as:
            domain_key = f"{incident_cfg['rrc']}_as{target_as}"

            t0 = time.perf_counter()
            df = extract_incident_features(
                incident_key, incident_cfg, target_as,
                rel_map, caida_meta, asn_to_ixps,
                enable_relationship_features=use_caida_relationships,
            )
            elapsed = time.perf_counter() - t0

            if df is not None and not df.empty:
                # Visibility check
                vis = assess_visibility(df, incident_key, domain_key)
                print_visibility_report(vis)

                # Save full visibility report to disk
                vis_dir = OUTPUT_DIR / "visibility_reports"
                vis_path = save_visibility_report(vis, vis_dir)
                print(f"    Visibility report: {vis_path.name}")

                # Tag each row with the visibility verdict
                df["visibility"] = vis["verdict"]

                # Save per-incident CSV
                out_path = OUTPUT_DIR / f"{incident_key}_{domain_key}.csv"
                df.to_csv(out_path, index=False)
                print(f"    Saved: {out_path.name} ({elapsed:.1f}s)")

                if vis["visible"]:
                    all_results.append(df)

                summary.append({
                    "incident": incident_key,
                    "domain": domain_key,
                    "target_as": target_as,
                    "total_windows": len(df),
                    "anomaly_windows": int(df["binary_label"].sum()),
                    "type": incident_cfg["type"],
                    "elapsed_sec": round(elapsed, 1),
                    "verdict": vis["verdict"],
                    "visible": vis["visible"],
                    "n_deviant_features": vis.get("n_deviant_features", 0),
                    "mean_top10_zscore": vis.get("mean_top10_zscore", 0.0),
                    "caida_file": caida_info["file_used"],
                    "caida_exact_match": caida_info["exact_match"],
                    "caida_enabled": caida_info["enabled"],
                    "caida_reason": caida_info["reason"],
                })
            else:
                print(f"    No features extracted ({elapsed:.1f}s)")
                summary.append({
                    "incident": incident_key,
                    "domain": domain_key,
                    "target_as": target_as,
                    "total_windows": 0,
                    "anomaly_windows": 0,
                    "type": incident_cfg["type"],
                    "elapsed_sec": round(elapsed, 1),
                    "verdict": "no_data",
                    "visible": False,
                    "n_deviant_features": 0,
                    "mean_top10_zscore": 0.0,
                    "caida_file": caida_info["file_used"],
                    "caida_exact_match": caida_info["exact_match"],
                    "caida_enabled": caida_info["enabled"],
                    "caida_reason": caida_info["reason"],
                })

    # ---- Build historical event manifest ----
    # Each incident-domain pair is one event. This manifest can be merged
    # with the study-period event_manifest.csv for Phase 3 event-level splitting.
    hist_manifest_rows = []
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        # Merge with existing visible data from prior batch runs
        vis_path = OUTPUT_DIR / "visible_historical_incidents.csv"
        if vis_path.exists():
            existing_vis = pd.read_csv(vis_path)
            # Remove domains being re-processed in this run
            new_domains = set(combined["collector"].astype(str) + "_as" +
                             combined["incident_key"].str.extract(r"_as(\d+)$", expand=False).fillna(""))
            # Simpler approach: remove by (incident_key, collector) overlap
            new_keys = set(zip(combined["incident_key"], combined["collector"]))
            keep_mask = ~existing_vis.apply(
                lambda r: (r.get("incident_key", ""), r.get("collector", "")) in new_keys,
                axis=1,
            )
            combined = pd.concat([existing_vis[keep_mask], combined], ignore_index=True)
        combined.to_csv(vis_path, index=False)
        print(f"\nCombined (visible only): {len(combined)} windows total")

        # Build manifest from visible incidents
        for event_id, group in combined[combined["event_id"] != "none"].groupby("event_id"):
            group_sorted = group.sort_values("window_start")
            hist_manifest_rows.append({
                "domain": group_sorted["collector"].iloc[0] + "_as" + str(
                    event_id.split("_as")[-1] if "_as" in event_id else ""
                ),
                "collector": group_sorted["collector"].iloc[0],
                "event_id": event_id,
                "start_window": group_sorted["window_id"].iloc[0],
                "end_window": group_sorted["window_id"].iloc[-1],
                "start_time": str(group_sorted["window_start"].iloc[0]),
                "end_time": str(group_sorted["window_start"].iloc[-1]),
                "n_windows": len(group_sorted),
                "n_anomaly_windows": len(group_sorted),
                "n_absorbed_uncertain": 0,
                "dominant_label": group_sorted["incident_label"].iloc[0],
                "incident_key": group_sorted["incident_key"].iloc[0],
                "incident_type": group_sorted["incident_type"].iloc[0],
                "provenance": "historical",
            })

    if hist_manifest_rows:
        hist_manifest = pd.DataFrame(hist_manifest_rows)
        manifest_path = OUTPUT_DIR / "historical_event_manifest.csv"
        if manifest_path.exists():
            existing_manifest = pd.read_csv(manifest_path)
            new_event_ids = set(hist_manifest["event_id"])
            keep_mask = ~existing_manifest["event_id"].isin(new_event_ids)
            hist_manifest = pd.concat(
                [existing_manifest[keep_mask], hist_manifest], ignore_index=True
            )
        hist_manifest.to_csv(manifest_path, index=False)
        print(f"Historical event manifest: {len(hist_manifest)} events")
    else:
        hist_manifest = pd.DataFrame()

    # Summary — merge with any existing summary from prior batch runs
    # (e.g., rrc04 run followed by rrc05 run).  Deduplicates on
    # (incident, domain) so re-running the same batch is safe.
    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "extraction_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove rows that will be replaced by this run
        key_cols = ["incident", "domain"]
        new_keys = set(zip(summary_df["incident"], summary_df["domain"]))
        keep_mask = ~existing.apply(
            lambda r: (r["incident"], r["domain"]) in new_keys, axis=1
        )
        merged = pd.concat([existing[keep_mask], summary_df], ignore_index=True)
    else:
        merged = summary_df
    merged.to_csv(summary_path, index=False)
    # Also save JSON version
    with open(OUTPUT_DIR / "extraction_summary.json", "w") as f:
        json.dump(merged.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")
    total_windows = sum(s["total_windows"] for s in summary)
    total_anomaly = sum(s["anomaly_windows"] for s in summary)
    n_visible = sum(1 for s in summary if s.get("verdict") == "visible")
    n_marginal = sum(1 for s in summary if s.get("verdict") == "marginal")
    n_invisible = sum(1 for s in summary if s.get("verdict") == "invisible")
    n_nodata = sum(1 for s in summary if s.get("verdict") == "no_data")
    print(f"Incidents processed: {len(incidents)}")
    print(f"Domain-incident pairs: {len(summary)}")
    print(f"Total windows extracted: {total_windows}")
    print(f"Anomaly windows: {total_anomaly}")
    print(f"\nVisibility verdicts:")
    print(f"  [+] Visible:   {n_visible}")
    print(f"  [~] Marginal:  {n_marginal}")
    print(f"  [-] Invisible: {n_invisible}")
    print(f"  [x] No data:   {n_nodata}")
    if n_invisible > 0:
        inv = [s for s in summary if s.get("verdict") == "invisible"]
        print(f"\n  Invisible incidents (excluded from combined output):")
        for s in inv:
            print(f"    {s['incident']} / {s['domain']}: "
                  f"{s['n_deviant_features']} deviant features, "
                  f"mean z={s['mean_top10_zscore']:.2f}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Visibility reports: {OUTPUT_DIR / 'visibility_reports'}")

    # Print event identity guidance
    print(f"\n{'='*70}")
    print("EVENT IDENTITY FOR PHASE 3")
    print(f"{'='*70}")
    print("Each historical incident has a unique event_id: hist_{incident}_{domain}")
    print("These are SEPARATE events for event-level splitting:")
    print("  - Never mix windows from the same event across train/test")
    print("  - Historical events can be used as SOURCE training anomalies")
    print("  - Study-period events remain the primary TEST events")
    if not hist_manifest.empty:
        print(f"\nHistorical events ready for Phase 3:")
        for _, row in hist_manifest.iterrows():
            print(f"  {row['event_id']}: {row['n_windows']} windows "
                  f"({row['incident_type']}, {row['incident_key']})")


if __name__ == "__main__":
    main()
