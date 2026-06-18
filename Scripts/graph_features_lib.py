# =============================================================================
# graph_features_lib.py  —  Pure-function library for BGP graph feature extraction
# =============================================================================
# Import this module; do NOT execute as a script.
# All globals (REMOVE_IXP_ASNS, IXP_ASNS, HAS_NETWORKIT) are module-level
# constants that the caller may override before calling the functions.
# =============================================================================

import bz2
import json
import logging
import os
import threading
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import bgpkit
from scipy.sparse.linalg import eigsh
from scipy import stats as sp_stats
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)

# --- Optional NetworKit ---
try:
    import networkit as nk
    HAS_NETWORKIT = True
except ImportError:
    HAS_NETWORKIT = False

# --- Module-level constants (caller may override) ---
PRIVATE_ASN_RANGES: Tuple[Tuple[int, int], ...] = (
    (64512, 65534),
    (4200000000, 4294967294),
)
RESERVED_ASNS: Set[int] = {0, 23456, 65535, 4294967295}
REMOVE_IXP_ASNS: bool = False
IXP_ASNS: Set[int] = set()


# =============================================================================
# ASN Validation
# =============================================================================

def is_valid_public_asn(asn: int) -> bool:
    """Return True if the ASN is a valid public (non-private, non-reserved) ASN."""
    if asn in RESERVED_ASNS:
        return False
    for start, end in PRIVATE_ASN_RANGES:
        if start <= asn <= end:
            return False
    return True


# =============================================================================
# AS-Path Parsing
# =============================================================================

def parse_as_path(as_path_str: str) -> List[int]:
    """
    Parse an AS_PATH string into a deduplicated list of valid public ASNs (integers).
    - Removes AS prepending duplicates
    - Skips AS-SETs entirely
    - Filters private/reserved ASNs (RFC 6996, RFC 7300)
    - Optionally filters IXP route-server ASNs (if REMOVE_IXP_ASNS is True)
    """
    if not as_path_str:
        return []
    tokens = as_path_str.split()
    deduped: List[int] = []
    in_as_set = False
    for token in tokens:
        if '{' in token:
            in_as_set = True
        if in_as_set:
            if '}' in token:
                in_as_set = False
            continue
        if '}' in token:
            continue
        try:
            asn = int(token)
        except ValueError:
            continue
        if not is_valid_public_asn(asn):
            continue
        if REMOVE_IXP_ASNS and asn in IXP_ASNS:
            continue
        if not deduped or asn != deduped[-1]:
            deduped.append(asn)
    return deduped


def parse_as_path_to_str(as_path_str: str) -> str:
    """
    Same as parse_as_path but returns a space-separated string of ASNs.
    Useful for statistical feature functions that expect string paths.
    """
    return ' '.join(str(a) for a in parse_as_path(as_path_str))


def extract_edges_from_as_path(as_path: List[int]) -> List[Tuple[int, int]]:
    """Extract sorted pairwise AS adjacency edges from a parsed AS path list."""
    edges = []
    for i in range(len(as_path) - 1):
        edge = tuple(sorted([as_path[i], as_path[i + 1]]))
        if edge[0] != edge[1]:
            edges.append(edge)
    return edges


# =============================================================================
# MRT File Download & Parsing
# =============================================================================

def download_mrt_file(url: str, dest_dir: Path, collector: str = None) -> Path:
    """Download an MRT file (cached). Returns the local Path."""
    filename = url.split('/')[-1]
    if collector:
        local_path = (dest_dir / collector / filename)
        local_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        local_path = dest_dir / filename
    if local_path.exists():
        return local_path
    logger.info(f"  Downloading: {filename}")
    t0 = time.time()
    temp_path = local_path.with_name(
        f".{local_path.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    if temp_path.exists():
        temp_path.unlink()
    urllib.request.urlretrieve(url, str(temp_path))
    if local_path.exists() and local_path.stat().st_size > 0:
        temp_path.unlink(missing_ok=True)
        return local_path
    os.replace(temp_path, local_path)
    logger.info(f"  Saved: {filename} ({local_path.stat().st_size/(1024*1024):.1f} MB, {time.time()-t0:.1f}s)")
    return local_path


def parse_mrt_to_rows(file_path: str) -> Tuple[List[dict], dict]:
    """
    Parse a single MRT RIB dump file via bgpkit.Parser.
    Returns (rows, stats) where rows are dicts with RIB fields.
    """
    rows = []
    stats = {'total_elements': 0, 'announcements': 0, 'withdrawals': 0,
             'unique_prefixes': set(), 'unique_peers': set(), 'parse_errors': 0}
    logger.info(f"  Parsing RIB: {Path(file_path).name}")
    t0 = time.time()
    try:
        parser = bgpkit.Parser(url=str(file_path))
        for elem in parser:
            stats['total_elements'] += 1
            if elem.elem_type == 'W':
                stats['withdrawals'] += 1
                continue
            stats['announcements'] += 1
            if elem.prefix:
                stats['unique_prefixes'].add(elem.prefix)
            if elem.peer_asn:
                stats['unique_peers'].add(elem.peer_asn)
            ts = datetime.fromtimestamp(elem.timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            communities = ' '.join(str(c) for c in elem.communities) if elem.communities else ''
            aggr_asn = getattr(elem, 'aggr_asn', None)
            aggr_ip = getattr(elem, 'aggr_ip', None)
            aggregator = f"{aggr_asn} {aggr_ip}".strip() if aggr_asn and aggr_ip else (str(aggr_asn) if aggr_asn else '')
            rows.append({
                'MRT_Type': 'TABLE_DUMP2', 'Timestamp': ts, 'Entry_Type': 'B',
                'Peer_IP': elem.peer_ip or '', 'Peer_AS': elem.peer_asn or '',
                'Prefix': elem.prefix or '', 'AS_Path': elem.as_path or '',
                'Origin': elem.origin or '', 'Next_Hop': elem.next_hop or '',
                'Local_Pref': elem.local_pref if elem.local_pref is not None else '',
                'MED': elem.med if elem.med is not None else '',
                'Community': communities, 'Atomic_Aggregate': 'AG' if elem.atomic else '',
                'Aggregator': aggregator,
            })
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        stats['parse_errors'] += 1
    elapsed = time.time() - t0
    stats['unique_prefixes'] = len(stats['unique_prefixes'])
    stats['unique_peers'] = len(stats['unique_peers'])
    stats['rows_parsed'] = len(rows)
    stats['parse_time_sec'] = round(elapsed, 2)
    logger.info(f"  -> {stats['total_elements']:,} elements, {len(rows):,} rows in {elapsed:.1f}s")
    return rows, stats


def build_edges_from_csv(csv_path_or_df):
    """
    Load a per-snapshot CSV and build AS topology edges (optimized).
    Returns (all_edges: set, edge_counts: Counter).
    """
    from collections import Counter
    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df, usecols=['AS_Path'])
    else:
        df = csv_path_or_df
    paths = df['AS_Path'].dropna().values.astype(str)
    edge_counts = Counter()
    for as_path_raw in paths:
        if not as_path_raw or as_path_raw == 'nan':
            continue
        tokens = as_path_raw.split()
        deduped_prev = -1
        in_as_set = False
        for token in tokens:
            if '{' in token:
                in_as_set = True
            if in_as_set:
                if '}' in token:
                    in_as_set = False
                continue
            if '}' in token:
                continue
            try:
                asn = int(token)
            except ValueError:
                continue
            if not is_valid_public_asn(asn):
                continue
            if REMOVE_IXP_ASNS and asn in IXP_ASNS:
                continue
            if asn == deduped_prev:
                continue
            if deduped_prev > 0 and asn != deduped_prev:
                a, b = (deduped_prev, asn) if deduped_prev < asn else (asn, deduped_prev)
                edge_counts[(a, b)] += 1
            deduped_prev = asn
    return set(edge_counts.keys()), edge_counts


# =============================================================================
# CAIDA AS Relationship Loader
# =============================================================================

def find_caida_date(start_date_str: str) -> str:
    """Return YYYYMM01 for the CAIDA file matching the given analysis start date."""
    return datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%Y%m01')


def download_caida_rel2(caida_date: str, dest_dir: Path) -> Path:
    """Download and decompress CAIDA Serial-2 AS relationship file (cached)."""
    txt_path = dest_dir / f"{caida_date}.as-rel2.txt"
    bz2_path = dest_dir / f"{caida_date}.as-rel2.txt.bz2"
    if txt_path.exists():
        logger.info(f"  CAIDA cached: {txt_path.name}")
        return txt_path
    urls = [
        f"https://publicdata.caida.org/datasets/as-relationships/serial-2/{caida_date}.as-rel2.txt.bz2",
        f"https://data.caida.org/datasets/as-relationships/serial-2/{caida_date}.as-rel2.txt.bz2",
    ]
    downloaded = False
    for url in urls:
        try:
            logger.info(f"  Downloading CAIDA: {url}")
            urllib.request.urlretrieve(url, str(bz2_path))
            downloaded = True
            break
        except urllib.error.HTTPError as e:
            logger.warning(f"  HTTP {e.code} from {url.split('/')[2]}")
        except Exception as e:
            logger.warning(f"  Error: {e}")
    if not downloaded:
        dt = datetime.strptime(caida_date, '%Y%m%d')
        prev = (dt.replace(day=1) - timedelta(days=1)).replace(day=1)
        prev_date = prev.strftime('%Y%m01')
        txt_path_prev = dest_dir / f"{prev_date}.as-rel2.txt"
        bz2_path = dest_dir / f"{prev_date}.as-rel2.txt.bz2"
        if txt_path_prev.exists():
            return txt_path_prev
        txt_path = txt_path_prev
        for url in [
            f"https://publicdata.caida.org/datasets/as-relationships/serial-2/{prev_date}.as-rel2.txt.bz2",
            f"https://data.caida.org/datasets/as-relationships/serial-2/{prev_date}.as-rel2.txt.bz2",
        ]:
            try:
                urllib.request.urlretrieve(url, str(bz2_path))
                downloaded = True
                break
            except Exception:
                continue
    if not downloaded:
        raise FileNotFoundError(f"Could not download CAIDA file for {caida_date}")
    logger.info(f"  Decompressing {bz2_path.name}...")
    with bz2.open(str(bz2_path), 'rt', encoding='utf-8') as fin, \
         open(str(txt_path), 'w', encoding='utf-8') as fout:
        fout.writelines(fin)
    try:
        bz2_path.unlink()
    except Exception:
        pass
    return txt_path


def load_caida_relationships(filepath: Path) -> Tuple[dict, dict]:
    """
    Parse CAIDA Serial-2 file into a bidirectional relationship lookup.
    Returns (rel_map, meta) where:
      rel_map[(as1, as2)] = +1 (as2 is provider), -1 (as2 is customer), 0 (peer)
      meta = {'clique': set(), 'ixp_ases': set(), 'sources': []}
    """
    rel_map = {}
    meta = {'clique': set(), 'ixp_ases': set(), 'sources': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# inferred clique:') or line.startswith('# input clique:'):
                meta['clique'] = {int(a) for a in line.split(':', 1)[1].split() if a.isdigit()}
                continue
            if line.startswith('# IXP ASes:'):
                meta['ixp_ases'] = {int(a) for a in line.split(':', 1)[1].split() if a.isdigit()}
                continue
            if line.startswith('# source:'):
                meta['sources'].append(line)
                continue
            if line.startswith('#') or not line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            try:
                as1, as2, rel = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            rel_map[(as1, as2)] = rel
            if rel == -1:
                rel_map[(as2, as1)] = 1
            elif rel == 0:
                rel_map[(as2, as1)] = 0
    return rel_map, meta


# =============================================================================
# Graph Utilities
# =============================================================================

def nx_to_nk(G_nx):
    """Convert a NetworkX graph to NetworKit format. Returns (G_nk, nx2nk, nk2nx)."""
    if not HAS_NETWORKIT:
        return None, None, None
    import networkit as nk_mod
    node_list = sorted(G_nx.nodes())
    nx2nk_map = {n: i for i, n in enumerate(node_list)}
    nk2nx_map = {i: n for n, i in nx2nk_map.items()}
    G_nk = nk_mod.Graph(len(node_list), weighted=False, directed=False)
    for u, v in G_nx.edges():
        G_nk.addEdge(nx2nk_map[u], nx2nk_map[v])
    return G_nk, nx2nk_map, nk2nx_map


def extract_ego_subgraph(G, target_as: int, k_hop: int):
    """
    Extract k-hop ego subgraph around target_as via BFS.
    Returns (G_ego, ego_info_dict).
    """
    if target_as not in G:
        return None, {'error': f'AS {target_as} not in graph'}
    ego_nodes = {target_as}
    frontier = {target_as}
    for _ in range(k_hop):
        next_frontier = set()
        for node in frontier:
            for nbr in G.neighbors(node):
                if nbr not in ego_nodes:
                    next_frontier.add(nbr)
                    ego_nodes.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    G_ego = G.subgraph(ego_nodes).copy()
    ego_info = {
        'target_as': target_as, 'k_hop': k_hop,
        'ego_nodes': len(ego_nodes), 'ego_edges': G_ego.number_of_edges(),
        'target_degree_full': G.degree(target_as),
    }
    if not nx.is_connected(G_ego):
        for comp in nx.connected_components(G_ego):
            if target_as in comp:
                G_ego = G_ego.subgraph(comp).copy()
                break
        ego_info['ego_connected'] = False
        ego_info['ego_nodes_after_lcc'] = G_ego.number_of_nodes()
    else:
        ego_info['ego_connected'] = True
    return G_ego, ego_info


# =============================================================================
# Graph-Level Feature Extraction
# =============================================================================

def extract_graph_level_features(G_lcc, G_nk, nk2nx_map, config: dict):
    """
    Extract graph-level topological features from a snapshot's LCC.

    config keys used:
        compute_spectral (bool, default True)
        betweenness_sample_k (int|None, default None = exact)
        max_nodes_for_clique (int, default 5000)

    Returns (features: dict, shared_data: dict).
    shared_data contains '_bc_map' and '_core_map' for node-level reuse.
    """
    features = {}
    shared_data = {}
    n_nodes = G_lcc.number_of_nodes()
    n_edges = G_lcc.number_of_edges()
    features['n_nodes'] = n_nodes
    features['n_edges'] = n_edges

    A_sparse = nx.adjacency_matrix(G_lcc, weight=None).astype(float)
    L_sparse = nx.laplacian_matrix(G_lcc, weight=None).astype(float)
    degrees = [d for _, d in G_lcc.degree(weight=None)]

    # 1. Assortativity
    try:
        features['assortativity'] = nx.degree_assortativity_coefficient(G_lcc, weight=None)
    except Exception:
        features['assortativity'] = None

    # 2. Density
    features['density'] = nx.density(G_lcc)

    # 3. Clustering
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            features['clustering_global'] = nk_mod.globals.ClusteringCoefficient.exactGlobal(G_nk)
            features['clustering_avg_local'] = nk_mod.globals.ClusteringCoefficient.sequentialAvgLocal(G_nk)
        else:
            features['clustering_global'] = nx.transitivity(G_lcc)
            features['clustering_avg_local'] = nx.average_clustering(G_lcc)
    except Exception:
        features['clustering_global'] = None
        features['clustering_avg_local'] = None

    # 4. Diameter & Average path length
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            diam_algo = nk_mod.distance.Diameter(G_nk, algo=nk_mod.distance.DiameterAlgo.AUTOMATIC)
            diam_algo.run()
            features['diameter'] = diam_algo.getDiameter()[0]
        elif n_nodes < 50000:
            features['diameter'] = nx.diameter(G_lcc)
        else:
            sample = np.random.choice(list(G_lcc.nodes()), size=min(100, n_nodes), replace=False)
            features['diameter'] = max(nx.eccentricity(G_lcc, v=node) for node in sample)
    except Exception:
        features['diameter'] = None

    try:
        if n_nodes < 20000:
            features['avg_path_length'] = nx.average_shortest_path_length(G_lcc)
        elif HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            sample_size = min(500, n_nodes)
            sample_nk_ids = np.random.choice(G_nk.numberOfNodes(), size=sample_size, replace=False)
            total_dist, count = 0, 0
            for nk_id in sample_nk_ids:
                bfs = nk_mod.distance.BFS(G_nk, int(nk_id))
                bfs.run()
                dists = bfs.getDistances()
                total_dist += sum(dists)
                count += len(dists) - 1
            features['avg_path_length'] = total_dist / count if count > 0 else float('inf')
        else:
            sample = np.random.choice(list(G_lcc.nodes()), size=min(500, n_nodes), replace=False)
            total_dist, count = 0, 0
            for node in sample:
                lengths = nx.single_source_shortest_path_length(G_lcc, node)
                total_dist += sum(lengths.values())
                count += len(lengths) - 1
            features['avg_path_length'] = total_dist / count if count > 0 else float('inf')
    except Exception:
        features['avg_path_length'] = None

    # 5. Algebraic connectivity
    try:
        try:
            eigs = eigsh(L_sparse, k=2, sigma=1e-6, which='LM', maxiter=5000, return_eigenvectors=False)
        except Exception:
            eigs = eigsh(L_sparse, k=2, which='SM', maxiter=n_nodes, return_eigenvectors=False)
        features['algebraic_connectivity'] = float(np.sort(eigs)[1])
    except Exception:
        features['algebraic_connectivity'] = None

    # 6. Spectral radius
    try:
        sr_vals = eigsh(A_sparse, k=1, which='LM', maxiter=5000, return_eigenvectors=False)
        features['spectral_radius'] = float(sr_vals[0])
    except Exception:
        features['spectral_radius'] = None

    # 7. Percolation limit
    features['percolation_limit'] = (
        1.0 / features['spectral_radius'] if features.get('spectral_radius') else None
    )

    # 8-11. Spectral metrics
    adjacency_eigs = None
    if config.get('compute_spectral', True):
        n_eigs = min(n_nodes - 2, 50)
        use_full = n_nodes < 5000
        try:
            if use_full:
                L_dense = L_sparse.toarray()
                A_dense = A_sparse.toarray()
                laplacian_eigs = np.sort(np.real(np.linalg.eigvalsh(L_dense)))
                adjacency_eigs = np.sort(np.real(np.linalg.eigvalsh(A_dense)))[::-1]
            else:
                try:
                    laplacian_eigs = np.sort(eigsh(L_sparse, k=min(n_eigs, n_nodes-2),
                                                   sigma=1e-6, which='LM', maxiter=5000,
                                                   return_eigenvectors=False))
                except Exception:
                    laplacian_eigs = np.sort(eigsh(L_sparse, k=min(n_eigs, n_nodes-2),
                                                   which='SM', maxiter=n_nodes,
                                                   return_eigenvectors=False))
                adjacency_eigs = np.sort(eigsh(A_sparse, k=min(n_eigs, n_nodes-2),
                                               which='LM', maxiter=5000,
                                               return_eigenvectors=False))[::-1]

            distinct_eigs = len(np.unique(np.round(adjacency_eigs, 8)))
            D = features.get('diameter', 10) or 10
            features['symmetry_ratio'] = distinct_eigs / (D + 1)

            max_eig = np.max(adjacency_eigs)
            shifted = np.exp(adjacency_eigs - max_eig)
            if use_full:
                features['natural_connectivity'] = float(max_eig + np.log(np.mean(shifted)))
            else:
                features['natural_connectivity'] = float(max_eig + np.log(np.sum(shifted) / n_nodes))

            nonzero_lap = laplacian_eigs[laplacian_eigs > 1e-10]
            if use_full and len(nonzero_lap) > 0:
                features['kirchhoff_index'] = float(n_nodes * np.sum(1.0 / nonzero_lap))
                features['log_spanning_trees'] = float(np.sum(np.log(nonzero_lap)) - np.log(n_nodes))
            else:
                features['kirchhoff_index'] = None
                features['log_spanning_trees'] = None
        except Exception as e:
            logger.warning(f"Spectral metrics failed: {e}")
            for k in ['symmetry_ratio', 'natural_connectivity', 'kirchhoff_index', 'log_spanning_trees']:
                features.setdefault(k, None)
    else:
        for k in ['symmetry_ratio', 'natural_connectivity', 'kirchhoff_index', 'log_spanning_trees']:
            features[k] = None

    # 12. Edge & node connectivity
    try:
        min_deg = min(d for _, d in G_lcc.degree())
        if min_deg <= 1 or nx.has_bridges(G_lcc):
            features['edge_connectivity'] = 1 if min_deg >= 1 else 0
        elif n_nodes < 5000:
            features['edge_connectivity'] = nx.edge_connectivity(G_lcc)
        else:
            G_unit = nx.Graph()
            G_unit.add_edges_from(G_lcc.edges())
            min_deg_node = min(G_lcc.nodes(), key=lambda n: G_lcc.degree(n))
            ec = min_deg
            for nbr in G_lcc.neighbors(min_deg_node):
                ec = min(ec, int(nx.maximum_flow_value(G_unit, min_deg_node, nbr,
                         flow_func=nx.algorithms.flow.shortest_augmenting_path)))
                if ec <= 1:
                    break
            features['edge_connectivity'] = ec
    except Exception:
        features['edge_connectivity'] = None

    try:
        if features.get('edge_connectivity') == 0:
            features['node_connectivity'] = 0
        elif list(nx.articulation_points(G_lcc)):
            features['node_connectivity'] = 1
        elif n_nodes < 5000:
            features['node_connectivity'] = nx.node_connectivity(G_lcc)
        else:
            min_deg_node = min(G_lcc.nodes(), key=lambda n: G_lcc.degree(n))
            nc = features.get('edge_connectivity', 1) or 1
            for nbr in G_lcc.neighbors(min_deg_node):
                nc = min(nc, nx.node_connectivity(G_lcc, min_deg_node, nbr))
                if nc <= 1:
                    break
            features['node_connectivity'] = nc
    except Exception:
        features['node_connectivity'] = None

    # 13. Rich-club coefficient
    try:
        rc = nx.rich_club_coefficient(G_lcc, normalized=False)
        rc_keys = sorted(rc.keys())
        def _nearest_rc(k):
            if not rc_keys:
                return None
            idx = min(int(np.searchsorted(rc_keys, k)), len(rc_keys) - 1)
            return rc[rc_keys[idx]]
        for pct in [25, 50, 75, 90, 95]:
            k_val = int(np.percentile(degrees, pct))
            features[f'rich_club_p{pct}'] = _nearest_rc(k_val)
    except Exception:
        for pct in [25, 50, 75, 90, 95]:
            features[f'rich_club_p{pct}'] = None

    # 14. Betweenness centrality distribution (stored in shared_data for node-level reuse)
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            if config.get('betweenness_sample_k'):
                bc_algo = nk_mod.centrality.ApproxBetweenness(G_nk, epsilon=0.01, delta=0.1)
            else:
                bc_algo = nk_mod.centrality.Betweenness(G_nk, normalized=True)
            bc_algo.run()
            bc_scores_nk = bc_algo.scores()
            _bc_map = {nk2nx_map[i]: bc_scores_nk[i] for i in range(len(bc_scores_nk))}
        else:
            _bc_map = nx.betweenness_centrality(G_lcc, k=config.get('betweenness_sample_k'), normalized=True)
        bc_scores = np.array(list(_bc_map.values()))
        features['betweenness_mean'] = float(np.mean(bc_scores))
        features['betweenness_max'] = float(np.max(bc_scores))
        features['betweenness_std'] = float(np.std(bc_scores))
        features['betweenness_skewness'] = float(sp_stats.skew(bc_scores))
        shared_data['_bc_map'] = _bc_map
    except Exception as e:
        logger.warning(f"Betweenness failed: {e}")
        for k in ['betweenness_mean', 'betweenness_max', 'betweenness_std', 'betweenness_skewness']:
            features[k] = None

    # 15. K-core decomposition (stored in shared_data for node-level reuse)
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            cd = nk_mod.centrality.CoreDecomposition(G_nk)
            cd.run()
            _core_scores_nk = cd.scores()
            _core_map = {nk2nx_map[i]: int(_core_scores_nk[i]) for i in range(len(_core_scores_nk))}
            features['degeneracy'] = int(cd.maxCoreNumber())
        else:
            _core_map = nx.core_number(G_lcc)
            features['degeneracy'] = int(max(_core_map.values()))
        core_numbers = np.array(list(_core_map.values()))
        features['core_mean'] = float(np.mean(core_numbers))
        features['core_std'] = float(np.std(core_numbers))
        features['core_median'] = float(np.median(core_numbers))
        features['innermost_core_size'] = int(np.sum(core_numbers == features['degeneracy']))
        shared_data['_core_map'] = _core_map
    except Exception as e:
        logger.warning(f"K-core failed: {e}")
        for k in ['degeneracy', 'core_mean', 'core_std', 'core_median', 'innermost_core_size']:
            features[k] = None

    # 16. Spectral gap
    if config.get('compute_spectral', True) and adjacency_eigs is not None:
        try:
            sorted_eigs = np.sort(adjacency_eigs)[::-1]
            if len(sorted_eigs) >= 2:
                features['spectral_gap'] = float(sorted_eigs[0] - sorted_eigs[1])
                features['adj_eig_ratio_1_2'] = (
                    float(sorted_eigs[0] / sorted_eigs[1]) if sorted_eigs[1] != 0 else None
                )
            else:
                features['spectral_gap'] = None
                features['adj_eig_ratio_1_2'] = None
        except Exception:
            features['spectral_gap'] = None
            features['adj_eig_ratio_1_2'] = None
    else:
        features['spectral_gap'] = None
        features['adj_eig_ratio_1_2'] = None

    # 17. Pair-based similarity (Jaccard, Adamic-Adar)
    try:
        edges_list = list(G_lcc.edges())
        if edges_list:
            features['mean_jaccard'] = float(np.mean([s for _, _, s in nx.jaccard_coefficient(G_lcc, edges_list)]))
            features['mean_adamic_adar'] = float(np.mean([s for _, _, s in nx.adamic_adar_index(G_lcc, edges_list)]))
        else:
            features['mean_jaccard'] = None
            features['mean_adamic_adar'] = None
    except Exception:
        features['mean_jaccard'] = None
        features['mean_adamic_adar'] = None

    shared_data['degrees'] = degrees
    return features, shared_data


# =============================================================================
# Node-Level Feature Extraction
# =============================================================================

def extract_node_level_features(G_lcc, G_nk, nx2nk_map, nk2nx_map, shared_data: dict, config: dict):
    """
    Extract 10 node-level features for a single snapshot.
    Returns (node_df: DataFrame indexed by ASN, extra_graph_features: dict).
    """
    n_nodes = G_lcc.number_of_nodes()
    extra_graph_features = {}
    node_features = pd.DataFrame(index=sorted(G_lcc.nodes()))
    node_features.index.name = 'asn'

    # 1. Degree centrality
    try:
        dc = nx.degree_centrality(G_lcc)
        node_features['degree_centrality'] = node_features.index.map(dc)
        node_features['degree'] = node_features.index.map(dict(G_lcc.degree()))
    except Exception as e:
        logger.warning(f"Degree centrality failed: {e}")

    # 2. Betweenness centrality (reused from graph-level)
    try:
        _bc_map = shared_data.get('_bc_map') or nx.betweenness_centrality(G_lcc, normalized=True)
        node_features['betweenness_centrality'] = node_features.index.map(_bc_map)
    except Exception as e:
        logger.warning(f"Betweenness failed: {e}")

    # 3. Closeness centrality
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            cc_algo = nk_mod.centrality.Closeness(G_nk, True, nk_mod.centrality.ClosenessVariant.GENERALIZED)
            cc_algo.run()
            cc_map = {nk2nx_map[i]: cc_algo.scores()[i] for i in range(G_nk.numberOfNodes())}
        else:
            cc_map = nx.closeness_centrality(G_lcc, wf_improved=True)
        node_features['closeness_centrality'] = node_features.index.map(cc_map)
    except Exception as e:
        logger.warning(f"Closeness failed: {e}")

    # 4. Eigenvector centrality
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            ev_algo = nk_mod.centrality.EigenvectorCentrality(G_nk, tol=1e-8)
            ev_algo.run()
            ev_map = {nk2nx_map[i]: ev_algo.scores()[i] for i in range(G_nk.numberOfNodes())}
        else:
            try:
                ev_map = nx.eigenvector_centrality(G_lcc, max_iter=200, tol=1e-6)
            except nx.PowerIterationFailedConvergence:
                ev_map = nx.eigenvector_centrality_numpy(G_lcc)
        node_features['eigenvector_centrality'] = node_features.index.map(ev_map)
    except Exception as e:
        logger.warning(f"Eigenvector failed: {e}")

    # 5. PageRank
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            pr_algo = nk_mod.centrality.PageRank(G_nk, damp=0.85, tol=1e-8)
            pr_algo.run()
            pr_map = {nk2nx_map[i]: pr_algo.scores()[i] for i in range(G_nk.numberOfNodes())}
        else:
            pr_map = nx.pagerank(G_lcc, alpha=0.85)
        node_features['pagerank'] = node_features.index.map(pr_map)
    except Exception as e:
        logger.warning(f"PageRank failed: {e}")

    # 6. Local clustering coefficient
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            lcc_algo = nk_mod.centrality.LocalClusteringCoefficient(G_nk, turbo=True)
            lcc_algo.run()
            lcc_map = {nk2nx_map[i]: lcc_algo.scores()[i] for i in range(G_nk.numberOfNodes())}
        else:
            lcc_map = nx.clustering(G_lcc)
        node_features['local_clustering'] = node_features.index.map(lcc_map)
    except Exception as e:
        logger.warning(f"Local clustering failed: {e}")

    # 7. Average neighbor degree
    try:
        node_features['avg_neighbor_degree'] = node_features.index.map(nx.average_neighbor_degree(G_lcc))
    except Exception as e:
        logger.warning(f"Avg neighbor degree failed: {e}")

    # 8. Node clique number
    try:
        max_nodes_clique = config.get('max_nodes_for_clique', 5000)
        if n_nodes <= max_nodes_clique:
            ncn = nx.node_clique_number(G_lcc)
        else:
            _core_map = shared_data.get('_core_map', {})
            k_max = max(_core_map.values()) if _core_map else 0
            ncn = {}
            if k_max > 0:
                core_sg = nx.k_core(G_lcc, k=k_max)
                if core_sg.number_of_nodes() <= max_nodes_clique:
                    ncn.update(nx.node_clique_number(core_sg))
            for node in G_lcc.nodes():
                if node not in ncn:
                    clique = {node}
                    for cand in sorted(G_lcc.neighbors(node), key=lambda x: G_lcc.degree(x), reverse=True):
                        if all(G_lcc.has_edge(cand, c) for c in clique):
                            clique.add(cand)
                    ncn[node] = len(clique)
        node_features['node_clique_number'] = node_features.index.map(ncn)
    except Exception as e:
        logger.warning(f"Clique number failed: {e}")

    # 9. Eccentricity
    try:
        if HAS_NETWORKIT and G_nk is not None:
            import networkit as nk_mod
            ecc_map = {}
            for nk_id in range(G_nk.numberOfNodes()):
                bfs = nk_mod.distance.BFS(G_nk, nk_id)
                bfs.run()
                ecc_map[nk2nx_map[nk_id]] = int(max(bfs.getDistances()))
            node_features['eccentricity'] = node_features.index.map(ecc_map)
            extra_graph_features['radius'] = min(ecc_map.values())
        elif n_nodes < 10000:
            ecc = nx.eccentricity(G_lcc)
            node_features['eccentricity'] = node_features.index.map(ecc)
            extra_graph_features['radius'] = min(ecc.values())
        else:
            sample = np.random.choice(list(G_lcc.nodes()), size=min(500, n_nodes), replace=False)
            ecc_sample = {}
            for node in sample:
                lengths = nx.single_source_shortest_path_length(G_lcc, node)
                ecc_sample[node] = max(lengths.values())
            node_features['eccentricity'] = node_features.index.map(ecc_sample)
            extra_graph_features['radius'] = min(ecc_sample.values()) if ecc_sample else None
    except Exception as e:
        logger.warning(f"Eccentricity failed: {e}")
        extra_graph_features['radius'] = None

    # 10. K-shell / core number (reused)
    try:
        _core_map = shared_data.get('_core_map') or nx.core_number(G_lcc)
        node_features['core_number'] = node_features.index.map(_core_map)
    except Exception as e:
        logger.warning(f"Core number failed: {e}")

    return node_features, extra_graph_features


# =============================================================================
# Relationship-Aware Feature Extraction
# =============================================================================

def annotate_graph_relationships(G, rel_map: dict) -> float:
    """Add CAIDA relationship type as edge attribute. Returns coverage fraction."""
    n_edges = G.number_of_edges()
    if n_edges == 0:
        return 0.0
    matched = 0
    for u, v in G.edges():
        rel = rel_map.get((u, v))
        G.edges[u, v]['rel_type'] = rel
        if rel is not None:
            matched += 1
    return matched / n_edges


def check_valley_free(as_path: list, rel_map: dict) -> tuple:
    """
    Check if an AS-PATH is valley-free.
    Returns (is_valid: bool, violation_index: int|None, n_known_hops: int).
    """
    if len(as_path) < 2:
        return True, None, 0
    phase = 'up'
    n_known = 0
    for i in range(len(as_path) - 1):
        rel = rel_map.get((as_path[i], as_path[i + 1]))
        if rel is None:
            continue
        n_known += 1
        if phase == 'up':
            if rel == 0:
                phase = 'peer'
            elif rel == -1:
                phase = 'down'
        elif phase == 'peer':
            if rel == -1:
                phase = 'down'
            else:
                return False, i, n_known
        elif phase == 'down':
            if rel != -1:
                return False, i, n_known
    return True, None, n_known


def load_as_paths_from_csv(csv_path) -> list:
    """Load and parse AS paths from a per-snapshot CSV. Returns list of int-lists."""
    df = pd.read_csv(csv_path, usecols=['AS_Path'])
    return normalize_as_paths(df['AS_Path'].dropna().values.astype(str))


def normalize_as_paths(as_paths) -> list:
    """
    Normalize an iterable of AS paths into a list of parsed int-lists.

    Each item may be:
      - a raw AS_PATH string
      - a cleaned space-separated ASN string
      - an iterable of ASNs
    Paths shorter than 2 hops are dropped.
    """
    normalized = []
    if as_paths is None:
        return normalized

    for raw in as_paths:
        if raw is None:
            continue
        if isinstance(raw, str):
            if not raw or raw == 'nan':
                continue
            parsed = parse_as_path(raw)
        else:
            try:
                parsed = [int(asn) for asn in raw]
            except (TypeError, ValueError):
                continue
            parsed = [asn for asn in parsed if is_valid_public_asn(asn)]
            if REMOVE_IXP_ASNS:
                parsed = [asn for asn in parsed if asn not in IXP_ASNS]
            deduped = []
            for asn in parsed:
                if not deduped or asn != deduped[-1]:
                    deduped.append(asn)
            parsed = deduped

        if len(parsed) >= 2:
            normalized.append(parsed)

    return normalized


def extract_relationship_features(G_sub, rel_map: dict, caida_meta: dict,
                                   csv_path=None, target_as: int = None,
                                   as_paths=None) -> tuple:
    """
    Extract relationship-aware graph-level and node-level features.
    AS paths may be supplied directly via as_paths, or loaded from csv_path.
    Returns (graph_rel_feats: dict, node_rel_df: DataFrame indexed by ASN).
    """
    graph_rel_feats = {}
    coverage = annotate_graph_relationships(G_sub, rel_map)
    graph_rel_feats['rel_coverage'] = round(coverage, 4)
    n_total = G_sub.number_of_edges()
    n_p2c = n_p2p = n_unknown = 0
    for u, v in G_sub.edges():
        rel = G_sub.edges[u, v].get('rel_type')
        if rel is None:
            n_unknown += 1
        elif rel == 0:
            n_p2p += 1
        else:
            n_p2c += 1
    graph_rel_feats['frac_p2c_edges'] = round(n_p2c / n_total, 4) if n_total else 0
    graph_rel_feats['frac_p2p_edges'] = round(n_p2p / n_total, 4) if n_total else 0
    graph_rel_feats['frac_unknown_edges'] = round(n_unknown / n_total, 4) if n_total else 0
    nodes_set = set(G_sub.nodes())
    clique = caida_meta.get('clique', set())
    ixp_ases = caida_meta.get('ixp_ases', set())
    graph_rel_feats['n_tier1_in_subgraph'] = len(clique & nodes_set)
    graph_rel_feats['n_ixp_in_subgraph'] = len(ixp_ases & nodes_set)

    normalized_paths = []
    if as_paths is not None:
        normalized_paths = normalize_as_paths(as_paths)
    elif csv_path is not None:
        try:
            normalized_paths = load_as_paths_from_csv(csv_path)
        except Exception as e:
            logger.warning(f"Could not load AS paths: {e}")

    ego_nodes = set(int(n) for n in G_sub.nodes()) if target_as else set()

    def _vf_stats(paths_to_check):
        violations = total_checked = 0
        depths = []
        for path in paths_to_check:
            is_valid, viol_idx, n_known = check_valley_free(path, rel_map)
            if n_known == 0:
                continue
            total_checked += 1
            if not is_valid:
                violations += 1
                if viol_idx is not None:
                    depths.append(viol_idx / (len(path) - 1))
        return violations, total_checked, depths

    violations, total_checked, depths = _vf_stats(normalized_paths)
    if total_checked > 0:
        graph_rel_feats['valley_free_violations'] = violations
        graph_rel_feats['valley_free_violation_rate'] = round(violations / total_checked, 6)
        graph_rel_feats['valley_free_paths_checked'] = total_checked
        graph_rel_feats['avg_violation_depth'] = round(float(np.mean(depths)), 4) if depths else 0
        graph_rel_feats['max_violation_depth'] = round(float(max(depths)), 4) if depths else 0
    else:
        for k in ['valley_free_violations', 'valley_free_violation_rate',
                  'avg_violation_depth', 'max_violation_depth']:
            graph_rel_feats[k] = None
        graph_rel_feats['valley_free_paths_checked'] = 0

    ego_paths = [p for p in normalized_paths if set(p) & ego_nodes] if normalized_paths and ego_nodes else []
    ego_viol, ego_checked, ego_depths = _vf_stats(ego_paths)
    ego_total_paths = len(ego_paths)
    if ego_checked > 0:
        graph_rel_feats['ego_valley_free_violations'] = ego_viol
        graph_rel_feats['ego_valley_free_violation_rate'] = round(ego_viol / ego_checked, 6)
        graph_rel_feats['ego_valley_free_paths_checked'] = ego_checked
        graph_rel_feats['ego_avg_violation_depth'] = round(float(np.mean(ego_depths)), 4) if ego_depths else 0
        graph_rel_feats['ego_max_violation_depth'] = round(float(max(ego_depths)), 4) if ego_depths else 0
        graph_rel_feats['ego_valley_free_paths_total'] = ego_total_paths
        global_rate = graph_rel_feats.get('valley_free_violation_rate')
        ego_rate = graph_rel_feats['ego_valley_free_violation_rate']
        graph_rel_feats['vf_rate_delta'] = (
            round(ego_rate - global_rate, 6) if global_rate is not None else None
        )
        global_checked = graph_rel_feats.get('valley_free_paths_checked', 0)
        graph_rel_feats['ego_vf_paths_frac'] = (
            round(ego_checked / global_checked, 6) if global_checked > 0 else None
        )
        ego_transit = ego_origin = 0
        for path in ego_paths:
            is_valid, _, n_known = check_valley_free(path, rel_map)
            if n_known == 0 or is_valid:
                continue
            if target_as and target_as in set(path):
                if path[0] == target_as or path[-1] == target_as:
                    ego_origin += 1
                else:
                    ego_transit += 1
        graph_rel_feats['ego_transit_violations'] = ego_transit
        graph_rel_feats['ego_origin_violations'] = ego_origin
        graph_rel_feats['ego_transit_vf_rate'] = round(ego_transit / ego_checked, 6)
    else:
        for k in ['ego_valley_free_violations', 'ego_valley_free_violation_rate',
                  'ego_avg_violation_depth', 'ego_max_violation_depth',
                  'vf_rate_delta', 'ego_vf_paths_frac',
                  'ego_transit_violations', 'ego_origin_violations', 'ego_transit_vf_rate']:
            graph_rel_feats[k] = None
        graph_rel_feats['ego_valley_free_paths_checked'] = 0
        graph_rel_feats['ego_valley_free_paths_total'] = ego_total_paths

    node_records = []
    for node in G_sub.nodes():
        neighbors = list(G_sub.neighbors(node))
        deg = len(neighbors)
        n_prov = n_cust = n_peer = n_unk = 0
        for nbr in neighbors:
            rel = rel_map.get((node, nbr))
            if rel == 1:
                n_prov += 1
            elif rel == -1:
                n_cust += 1
            elif rel == 0:
                n_peer += 1
            else:
                n_unk += 1
        node_records.append({
            'asn': node, 'n_providers': n_prov, 'n_customers': n_cust,
            'n_peers': n_peer, 'n_unknown_rel': n_unk,
            'p2c_ratio': round(n_cust / deg, 4) if deg > 0 else 0,
            'p2p_ratio': round(n_peer / deg, 4) if deg > 0 else 0,
            'provider_ratio': round(n_prov / deg, 4) if deg > 0 else 0,
            'is_tier1': 1 if node in clique else 0,
            'is_ixp': 1 if node in ixp_ases else 0,
        })
    return graph_rel_feats, pd.DataFrame(node_records).set_index('asn')


# =============================================================================
# IXP Cosine Distance Features (PeeringDB)
# =============================================================================

def download_peeringdb_dump(date_str: str, dest_dir: Path) -> Optional[Path]:
    """Download PeeringDB JSON dump from CAIDA's archive (cached, tries 30 days back)."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for offset in range(31):
        cur = dt - timedelta(days=offset)
        filename = f"peeringdb_2_dump_{cur.year}_{cur.strftime('%m')}_{cur.strftime('%d')}.json"
        local_path = dest_dir / filename
        if local_path.exists():
            return local_path
        url = (f"https://publicdata.caida.org/datasets/peeringdb/"
               f"{cur.year}/{cur.strftime('%m')}/{filename}")
        try:
            urllib.request.urlretrieve(url, str(local_path))
            return local_path
        except urllib.error.HTTPError:
            continue
        except Exception as e:
            logger.warning(f"PeeringDB download error: {e}")
            continue
    logger.warning("Could not find PeeringDB dump within 30 days of analysis date")
    return None


def load_peeringdb_ixp_memberships(peeringdb_json_path: Path) -> Dict[int, Set[int]]:
    """Parse PeeringDB JSON dump → {ASN: set of IXP IDs}."""
    with open(peeringdb_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    asn_to_ixps: Dict[int, Set[int]] = {}
    for entry in data.get('netixlan', {}).get('data', []):
        asn = entry.get('asn')
        ix_id = entry.get('ix_id')
        if asn is not None and ix_id is not None:
            asn_to_ixps.setdefault(asn, set()).add(ix_id)
    return asn_to_ixps


def build_ixp_feature_vectors(G: nx.Graph,
                               asn_to_ixps: Dict[int, Set[int]]) -> Tuple[pd.DataFrame, dict]:
    """Build IXP neighborhood membership vectors for each node in G."""
    all_ixp_ids: Set[int] = set()
    for node in G.nodes():
        if node in asn_to_ixps:
            all_ixp_ids.update(asn_to_ixps[node])
        for nbr in G.neighbors(node):
            if nbr in asn_to_ixps:
                all_ixp_ids.update(asn_to_ixps[nbr])
    if not all_ixp_ids:
        return pd.DataFrame(index=sorted(G.nodes())), {}
    sorted_ixps = sorted(all_ixp_ids)
    ixp_mapping = {ixp_id: idx for idx, ixp_id in enumerate(sorted_ixps)}
    n_ixps = len(sorted_ixps)
    node_list = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    vectors = np.zeros((len(node_list), n_ixps), dtype=np.float64)
    for node in node_list:
        i = node_idx[node]
        for nbr in G.neighbors(node):
            if nbr in asn_to_ixps:
                for ixp_id in asn_to_ixps[nbr]:
                    vectors[i, ixp_mapping[ixp_id]] += 1
    return pd.DataFrame(vectors, index=node_list, columns=sorted_ixps), ixp_mapping


def compute_ixp_cosine_features(G: nx.Graph, ixp_vectors_df: pd.DataFrame,
                                  asn_to_ixps: Dict[int, Set[int]]) -> Tuple[dict, pd.DataFrame]:
    """Compute per-edge IXP cosine distances → graph- and node-level features."""
    graph_feats = {}
    nodes = sorted(G.nodes())
    n_memberships = [len(asn_to_ixps.get(n, set())) for n in nodes]
    if ixp_vectors_df.shape[1] > 0:
        raw_values = ixp_vectors_df.loc[nodes].values
        row_norms = np.linalg.norm(raw_values, axis=1)
        normalized = Normalizer(norm='l2').transform(raw_values)
    else:
        row_norms = np.zeros(len(nodes))
        normalized = np.zeros((len(nodes), 0))
    node_idx = {n: i for i, n in enumerate(nodes)}
    edge_cosine_dists = []
    node_cosine_sums: Dict[int, list] = defaultdict(list)
    for u, v in G.edges():
        if ixp_vectors_df.shape[1] == 0:
            dist = -1.0
        else:
            nu, nv = row_norms[node_idx[u]], row_norms[node_idx[v]]
            if nu == 0 or nv == 0:
                dist = -1.0
            else:
                dist = float(cosine_distance(normalized[node_idx[u]], normalized[node_idx[v]]))
        edge_cosine_dists.append(dist)
        node_cosine_sums[u].append(dist)
        node_cosine_sums[v].append(dist)
    valid_dists = [d for d in edge_cosine_dists if d >= 0]
    if valid_dists:
        graph_feats['avg_edge_ixp_cosine_dist'] = round(float(np.mean(valid_dists)), 6)
        graph_feats['median_edge_ixp_cosine_dist'] = round(float(np.median(valid_dists)), 6)
        graph_feats['std_edge_ixp_cosine_dist'] = round(float(np.std(valid_dists)), 6)
    else:
        graph_feats['avg_edge_ixp_cosine_dist'] = None
        graph_feats['median_edge_ixp_cosine_dist'] = None
        graph_feats['std_edge_ixp_cosine_dist'] = None
    graph_feats['n_edges_with_ixp_data'] = len(valid_dists)
    graph_feats['n_edges_missing_ixp_data'] = len(edge_cosine_dists) - len(valid_dists)
    node_records = []
    for i, node in enumerate(nodes):
        valid_nbr = [d for d in node_cosine_sums.get(node, []) if d >= 0]
        node_records.append({
            'asn': node,
            'n_ixp_memberships': n_memberships[i],
            'ixp_vector_norm': round(float(row_norms[i]), 6),
            'avg_ixp_cosine_dist': round(float(np.mean(valid_nbr)), 6) if valid_nbr else None,
        })
    return graph_feats, pd.DataFrame(node_records).set_index('asn')


# =============================================================================
# IncrementalTopology Class
# =============================================================================

class IncrementalTopology:
    """
    Maintains an AS-level topology that evolves with BGP updates.
    Uses reference counting per (peer_as, prefix) path.
    Reference: Sanchez et al., BigDAMA 2019 / DFOH, NSDI 2024.
    """

    def __init__(self):
        self.G = nx.Graph()
        self.edge_ref_count: Dict[tuple, int] = defaultdict(int)
        self.prefix_edges: Dict[tuple, set] = {}
        self._stats = {'rib_paths': 0, 'ann_applied': 0, 'wd_applied': 0,
                       'edges_added': 0, 'edges_removed': 0}

    def _normalize_edge(self, u, v) -> tuple:
        return (u, v) if u <= v else (v, u)

    def _remove_supported_edges(self, edge_set: Set[tuple]) -> None:
        G = self.G
        edge_ref_count = self.edge_ref_count
        stats = self._stats

        for edge in edge_set:
            remaining = edge_ref_count[edge] - 1
            if remaining > 0:
                edge_ref_count[edge] = remaining
                continue

            if G.has_edge(*edge):
                G.remove_edge(*edge)
                stats['edges_removed'] += 1

            u, v = edge
            if u in G and not G.adj[u]:
                G.remove_node(u)
            if v in G and not G.adj[v]:
                G.remove_node(v)

            del edge_ref_count[edge]

    def load_rib(self, rib_file_path: str):
        """Initialize topology from a RIB dump file via bgpkit.Parser."""
        logger.info(f"Loading RIB: {Path(rib_file_path).name}")
        t0 = time.time()
        parser = bgpkit.Parser(url=str(rib_file_path))
        G = self.G
        edge_ref_count = self.edge_ref_count
        prefix_edges = self.prefix_edges
        stats = self._stats
        normalize_edge = self._normalize_edge

        for elem in parser:
            if elem.elem_type == 'W':
                continue
            as_path = parse_as_path(elem.as_path or '')
            if len(as_path) < 2:
                continue
            key = (str(elem.peer_asn), elem.prefix or '')
            edge_set = set()
            edge_set_add = edge_set.add
            for u, v in extract_edges_from_as_path(as_path):
                edge = normalize_edge(u, v)
                edge_set_add(edge)
                edge_ref_count[edge] += 1
                if not G.has_edge(*edge):
                    G.add_edge(*edge)
                    stats['edges_added'] += 1
            prefix_edges[key] = edge_set
            stats['rib_paths'] += 1
        elapsed = time.time() - t0
        logger.info(f"  RIB loaded: {stats['rib_paths']:,} paths, "
                    f"{G.number_of_nodes():,} nodes, "
                    f"{G.number_of_edges():,} edges in {elapsed:.1f}s")

    def apply_announcement(self, peer_as, prefix: str, as_path_str: str):
        """Process an announcement: implicit withdrawal then add new path's edges."""
        as_path = parse_as_path(as_path_str)
        if len(as_path) < 2:
            return
        normalize_edge = self._normalize_edge
        new_edges = {normalize_edge(u, v) for u, v in extract_edges_from_as_path(as_path)}
        key = (str(peer_as), prefix)
        previous_edges = self.prefix_edges.get(key)
        if previous_edges:
            self._remove_supported_edges(previous_edges)

        self.prefix_edges[key] = new_edges
        G = self.G
        edge_ref_count = self.edge_ref_count
        stats = self._stats
        for edge in new_edges:
            edge_ref_count[edge] += 1
            if not G.has_edge(*edge):
                G.add_edge(*edge)
                stats['edges_added'] += 1
        self._stats['ann_applied'] += 1

    def apply_withdrawal(self, peer_as, prefix: str):
        """Process a withdrawal: remove the path's edges."""
        key = (str(peer_as), prefix)
        previous_edges = self.prefix_edges.pop(key, None)
        if previous_edges:
            self._remove_supported_edges(previous_edges)
        self._stats['wd_applied'] += 1

    def apply_window_updates(self, records) -> None:
        """Apply a window of ordered update records with less per-row overhead."""
        G = self.G
        edge_ref_count = self.edge_ref_count
        prefix_edges = self.prefix_edges
        stats = self._stats
        normalize_edge = self._normalize_edge
        remove_supported_edges = self._remove_supported_edges

        for elem_type, peer_as, prefix, as_path_str in records:
            key = (str(peer_as), prefix)

            if elem_type == 'A':
                if not as_path_str:
                    continue

                as_path = parse_as_path(as_path_str)
                if len(as_path) < 2:
                    continue

                new_edges = {normalize_edge(u, v) for u, v in extract_edges_from_as_path(as_path)}
                previous_edges = prefix_edges.get(key)
                if previous_edges:
                    remove_supported_edges(previous_edges)

                prefix_edges[key] = new_edges
                for edge in new_edges:
                    edge_ref_count[edge] += 1
                    if not G.has_edge(*edge):
                        G.add_edge(*edge)
                        stats['edges_added'] += 1
                stats['ann_applied'] += 1
            elif elem_type == 'W':
                previous_edges = prefix_edges.pop(key, None)
                if previous_edges:
                    remove_supported_edges(previous_edges)
                stats['wd_applied'] += 1

    def get_ego_subgraph(self, target_as, k_hop: int) -> nx.Graph:
        """Extract k-hop ego subgraph around target AS."""
        target = int(target_as) if isinstance(target_as, str) else target_as
        if target not in self.G:
            return nx.Graph()
        return nx.ego_graph(self.G, target, radius=k_hop).copy()

    def snapshot_info(self) -> dict:
        return {
            'n_nodes': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'n_tracked_paths': len(self.prefix_edges),
            **self._stats,
        }
