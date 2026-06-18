"""GNN classifier family for the §5.2 ablation.

Three edge-construction variants share a single GAT encoder:

  gnn_temporal   nodes = 5-min windows;   edges = t <-> t-k for k \u2208 [1..K]
  gnn_knn        nodes = 5-min windows;   edges = mutual k-NN in 22-d feature space
  gnn_astopo_8h  nodes = ASes in the 8-h ego snapshot;  edges = real AS peering
                 (headline GAT: per-snapshot encoder \u2192 target-AS embedding,
                  concatenated with the 5-min window features, MLP head)

Training is inductive: source graph \u2192 target graph (no shared nodes).

Output prediction CSV matches the existing classifier layout so the
`build_supervisor_metrics_csv.py` and `build_retrain_comparison.py` aggregators
pick up GNN rows with zero changes:

  transfer_matrix/<bucket>/<pair>[__relabeled]/predictions/<view>__after_coral__gnn_<edge>__seed<n>.csv
  columns: window_start, window_id, binary_label, y_pred, y_score, event_id
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, GATv2Conv, GINConv, GPSConv, HGTConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from phase3_view_partition import (  # noqa: E402
    CORE10_ALL, SHARED22_ALL, SHARED22_GRAPH_VIEW, SHARED22_STAT_VIEW,
)
from gnn_caida import classify_edges, load_caida  # noqa: E402

VIEW_COLS = {
    "graph": list(SHARED22_GRAPH_VIEW),
    "stat": list(SHARED22_STAT_VIEW),
    "fusion_early": list(SHARED22_ALL),
    "core10": list(CORE10_ALL),
}

CORAL_ROOT = ROOT / "bgp_unified_results" / "phase3_fusion" / "coral_aligned"
TM_ROOT = ROOT / "bgp_unified_results" / "phase3_fusion" / "transfer_matrix"
TOPO_ROOT = ROOT / "dataset" / "gnn_graphs"

# -------------------------------------------------------------------- data load

@dataclass
class PairData:
    pair: str
    src_feats: np.ndarray          # (N_src, F)
    src_labels: np.ndarray         # (N_src,)
    src_meta: pd.DataFrame         # window_start, window_id, event_id, binary_label
    tgt_feats: np.ndarray          # (N_tgt, F)
    tgt_labels: np.ndarray         # (N_tgt,)
    tgt_meta: pd.DataFrame
    feature_cols: list[str]
    # Chronological val mask over src_* arrays: True = validation (last VAL_FRAC of source).
    # Training loss uses ~src_val_mask; early stop PR-AUC uses src_val_mask only.
    src_val_mask: np.ndarray = None  # (N_src,) bool


def _apply_source_label_override(src: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    """Mirror of phase3_pipeline._apply_source_label_override (kept local to avoid
    importing a 2000-line module for a 10-line helper)."""
    rel = pd.read_csv(labels_path)
    required = {"window_start", "binary_label_new", "discovered_label"}
    missing = required - set(rel.columns)
    if missing:
        raise ValueError(f"relabel CSV missing required columns {sorted(missing)}: {labels_path}")
    keep = rel[rel["discovered_label"] != "uncertain"][["window_start", "binary_label_new"]].copy()
    merged = src.merge(keep, on="window_start", how="inner")
    if len(merged) == 0:
        raise RuntimeError(f"source-labels-from join produced 0 rows: {labels_path}")
    merged["binary_label"] = merged["binary_label_new"].astype(int)
    merged = merged.drop(columns=["binary_label_new"])
    print(f"  [source-labels-from] {labels_path.name}: "
          f"{len(src)} -> {len(merged)} rows  "
          f"anomaly_rate {src['binary_label'].mean():.3f} -> "
          f"{merged['binary_label'].mean():.3f}")
    return merged


def build_kind_label(coral_kind: str, balance_fit: str, view: str) -> str:
    """Mirror of phase3_pipeline.derive_kind_label for GNN lookups."""
    base = coral_kind if balance_fit == "none" else f"{coral_kind}_balanced_{balance_fit}"
    if view == "core10":
        return f"{base}__core10"
    return base


def pair_dir_suffix(balance_fit: str, view: str) -> str:
    """Matches aggregator regex in build_supervisor_metrics_csv.parse_dir_name.
    Order: __core10 before __balanced_equal (matches on-disk convention)."""
    parts = []
    if view == "core10":
        parts.append("__core10")
    if balance_fit != "none":
        parts.append(f"__balanced_{balance_fit}")
    return "".join(parts)


VAL_FRAC = 0.10  # last 10% of the chronologically-sorted source is held out for early-stop PR-AUC


def _chronological_val_mask(n: int, src_labels: np.ndarray,
                            frac: float = VAL_FRAC) -> np.ndarray:
    """Boolean mask over N source rows; True for the last `frac` fraction.
    Grows the tail backward if the tail has <5 positives OR <5 negatives, so
    the val PR-AUC signal is meaningful even on low-anomaly pairs.
    """
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    cut = max(1, int(round(n * frac)))
    mask[n - cut:] = True
    MIN_PER_CLASS = 5
    while mask.sum() < n:
        tail_labels = src_labels[mask]
        n_pos = int((tail_labels == 1).sum())
        n_neg = int((tail_labels == 0).sum())
        if n_pos >= MIN_PER_CLASS and n_neg >= MIN_PER_CLASS:
            break
        cut = min(n, cut + max(1, int(round(n * 0.02))))
        mask[:] = False
        mask[n - cut:] = True
    return mask


def load_pair(pair: str, coral_kind: str, balance_fit: str, alignment: str,
              view: str, source_labels_from: Path | None) -> PairData:
    kind_label = build_kind_label(coral_kind, balance_fit, view)
    kind_root = CORAL_ROOT / kind_label
    base = kind_root / "pairs" / pair
    src_dom = pair.split("__to__")[0]
    if alignment == "after_coral":
        src = pd.read_csv(base / "aligned_source.csv")
    else:  # before_coral: raw pre-CORAL source (mirrors phase3_pipeline._load_experiment_source)
        src = pd.read_csv(kind_root / "prepared_sources" / f"{src_dom}_binary_source.csv")
    tgt = pd.read_csv(base / "target_reference.csv")
    if source_labels_from is not None:
        src = _apply_source_label_override(src, source_labels_from)
    # Temporal edges rely on row order matching chronological order.
    # CORAL output is already sorted in practice; defensive sort below
    # guarantees the invariant if the upstream ever changes.
    if "window_start" in src.columns:
        src = src.sort_values("window_start", kind="mergesort").reset_index(drop=True)
    if "window_start" in tgt.columns:
        tgt = tgt.sort_values("window_start", kind="mergesort").reset_index(drop=True)
    cols = VIEW_COLS[view]
    meta_cols = ["window_start", "window_id", "event_id", "binary_label"]
    for c in meta_cols:
        if c not in src.columns:
            src[c] = np.nan
        if c not in tgt.columns:
            tgt[c] = np.nan
    src_labels = src["binary_label"].to_numpy(dtype=np.int64)
    val_mask = _chronological_val_mask(len(src), src_labels, frac=VAL_FRAC)
    return PairData(
        pair=pair,
        src_feats=src[cols].to_numpy(dtype=np.float32),
        src_labels=src_labels,
        src_meta=src[meta_cols].reset_index(drop=True),
        tgt_feats=tgt[cols].to_numpy(dtype=np.float32),
        tgt_labels=tgt["binary_label"].to_numpy(dtype=np.int64),
        tgt_meta=tgt[meta_cols].reset_index(drop=True),
        feature_cols=cols,
        src_val_mask=val_mask,
    )


# -------------------------------------------------------------------- edges

def temporal_edges(n: int, k: int = 3) -> torch.Tensor:
    """Each node t is connected to t-1..t-k (and vice versa)."""
    src, dst = [], []
    for lag in range(1, k + 1):
        a = np.arange(lag, n)
        b = a - lag
        src.extend(a.tolist() + b.tolist())
        dst.extend(b.tolist() + a.tolist())
    return torch.tensor([src, dst], dtype=torch.long)


def knn_edges(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """True mutual k-NN in euclidean space on the provided feature matrix.
    An undirected edge (i, j) is emitted iff i is in the k nearest neighbours
    of j AND j is in the k nearest neighbours of i (self-matches excluded)."""
    n = len(X)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    _, idx = nn.kneighbors(X)
    neighbour_sets = [set(int(j) for j in row[1:]) for row in idx]  # drop self
    src, dst = [], []
    for i, nbrs in enumerate(neighbour_sets):
        for j in nbrs:
            if i in neighbour_sets[j]:
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)
    if not src:
        # Mutual k-NN can be empty for tiny N with pathological geometry;
        # fall back to symmetrised k-NN in that rare case so training runs.
        for i, nbrs in enumerate(neighbour_sets):
            for j in nbrs:
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)
    ei = np.array([src, dst])
    ei = np.unique(ei.T, axis=0).T
    return torch.tensor(ei, dtype=torch.long)


# -------------------------------------------------------------------- topology

@dataclass
class SnapshotGraph:
    x: torch.Tensor            # (N, F_node)  float32, scaled
    edge_index: torch.Tensor   # (2, E_bi)   E_bi = 2*E (undirected as both dirs)
    edge_attr: torch.Tensor    # (E_bi, 1)   float32, log1p(weight)
    asn_to_idx: dict[int, int]


_SNAPSHOT_CACHE: dict[tuple[str, str], SnapshotGraph] = {}
# Exclude identifier + booleans. Keep numeric node features only.
NODE_FEATURE_COLS = [
    "degree", "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality", "pagerank",
    "local_clustering", "avg_neighbor_degree", "node_clique_number",
    "eccentricity", "core_number", "n_providers", "n_customers", "n_peers",
    "provider_ratio", "ixp_vector_norm", "n_ixp_memberships",
]  # 17 numeric features; avg_ixp_cosine_dist dropped (can be null)


def load_snapshot(pair_topo_key: str, stamp: str) -> SnapshotGraph:
    """Load (and cache) the 8-h snapshot graph for a (collector, AS) pair."""
    key = (pair_topo_key, stamp)
    if key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[key]
    d = TOPO_ROOT / pair_topo_key / stamp
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    asns = nodes["asn"].to_numpy()
    asn_to_idx = {int(a): i for i, a in enumerate(asns)}
    src = edges["source"].map(asn_to_idx).to_numpy()
    dst = edges["target"].map(asn_to_idx).to_numpy()
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    # Edge weights: parsed value is a count-like multiplicity; log1p keeps the
    # dynamic range compact so a dense core edge doesn't dominate GAT scoring.
    w = edges["weight"].to_numpy(dtype=np.float32)
    ea = np.log1p(np.concatenate([w, w])).reshape(-1, 1)  # duplicated for both dirs
    x = nodes[NODE_FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    sg = SnapshotGraph(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei).long(),
        edge_attr=torch.from_numpy(ea),
        asn_to_idx=asn_to_idx,
    )
    _SNAPSHOT_CACHE[key] = sg
    return sg


def window_to_stamp(window_start: str) -> str:
    """2025-11-01T00:00:00+00:00  ->  20251101_0000 rounded down to 8h bucket."""
    ts = pd.Timestamp(window_start)
    bucket_h = (ts.hour // 8) * 8
    return f"{ts.year:04d}{ts.month:02d}{ts.day:02d}_{bucket_h:02d}00"


def topo_key_for_domain(domain: str) -> str:
    """domain='rrc04_as12880' -> 'rrc04_as12880' (matches gnn_graphs/ layout)."""
    return domain


def target_as_for_domain(domain: str) -> int:
    # domain = 'rrc04_as12880' -> 12880
    return int(domain.split("_as")[-1])


# -------------------------------------------------------------------- models

class WindowGAT(nn.Module):
    """Two-layer GAT for temporal/knn variants. Produces per-node logit."""
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.g2 = GATConv(hidden * heads, hidden, heads=1, concat=False,
                          dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.elu(self.g1(x, edge_index))
        h = self.drop(h)
        h = F.elu(self.g2(h, edge_index))
        return self.out(h).squeeze(-1)


class TopoGAT(nn.Module):
    """Per-snapshot GAT over the AS-topology graph, concatenated with the
    5-min window feature vector, then an MLP head. Node features are the
    17 numeric node-level properties parsed from the HTML snapshot.

    When `use_edge_weights=True`, the two attention layers become GATv2 with
    edge_dim=1 so the learned attention can condition on log1p(weight)."""
    def __init__(self, node_in: int, window_in: int, embed: int = 16,
                 heads: int = 8, hidden: int = 64, dropout: float = 0.3,
                 use_edge_weights: bool = False):
        super().__init__()
        self.use_edge_weights = use_edge_weights
        if use_edge_weights:
            self.g1 = GATv2Conv(node_in, embed, heads=heads, dropout=dropout,
                                edge_dim=1)
            self.g2 = GATv2Conv(embed * heads, embed, heads=1, concat=False,
                                dropout=dropout, edge_dim=1)
        else:
            self.g1 = GATConv(node_in, embed, heads=heads, dropout=dropout)
            self.g2 = GATConv(embed * heads, embed, heads=1, concat=False,
                              dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed + window_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_snapshot(self, sg: SnapshotGraph):
        if self.use_edge_weights:
            h = F.elu(self.g1(sg.x, sg.edge_index, sg.edge_attr))
            return F.elu(self.g2(h, sg.edge_index, sg.edge_attr))
        h = F.elu(self.g1(sg.x, sg.edge_index))
        return F.elu(self.g2(h, sg.edge_index))

    def forward_batch(self, snap_embeds_target_as: torch.Tensor,
                      window_feats: torch.Tensor) -> torch.Tensor:
        z = torch.cat([snap_embeds_target_as, window_feats], dim=-1)
        return self.mlp(z).squeeze(-1)


# -------------------------------------------------------------------- trainer

def set_seed(seed: int, strict: bool = True) -> None:
    """Seed python/numpy/torch and enforce strict CUDA determinism by default.

    Mirrors phase3_deep_models.set_determinism so flat classifiers and the
    GNN family share one reproducibility contract.  Strict mode (the default
    since the 2026-04-22 reproducibility audit) pins cuDNN to deterministic
    kernels and requires CUBLAS_WORKSPACE_CONFIG=:4096:8.
    """
    import os as _os
    import random as _random
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)


def class_pos_weight(labels: np.ndarray) -> torch.Tensor:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    w = float(n_neg) / max(1, n_pos)
    return torch.tensor([w], dtype=torch.float32)


def standardize(X_src: np.ndarray, X_tgt: np.ndarray):
    sc = StandardScaler().fit(X_src)
    return sc.transform(X_src).astype(np.float32), sc.transform(X_tgt).astype(np.float32)


class EarlyStop:
    """Patience-based early stop on a held-out validation PR-AUC signal.

    Keeps a deep copy of the model's state_dict at every improvement so the
    returned model is the best-seen, not last-epoch. `patience` is the number
    of consecutive eval rounds with no improvement over `best + min_delta`
    before halting.
    """
    def __init__(self, patience: int = 3, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.best_epoch = -1
        self.bad = 0
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module | None = None,
             epoch: int | None = None) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            if epoch is not None:
                self.best_epoch = epoch
            if model is not None:
                # deepcopy via .clone() on each tensor avoids aliasing the live weights
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load best-seen weights back into the model. No-op if no step was kept."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_window_gnn(data: PairData, edge: Literal["temporal", "knn"],
                     seed: int, epochs: int, device: torch.device,
                     patience: int = 5,
                     k_temporal: int = 3, k_knn: int = 10) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)
    if edge == "temporal":
        e_src = temporal_edges(len(Xs), k=k_temporal)
        e_tgt = temporal_edges(len(Xt), k=k_temporal)
    else:
        e_src = knn_edges(Xs, k=k_knn)
        e_tgt = knn_edges(Xt, k=k_knn)

    # Transductive val split: edges span the full source graph so val nodes
    # can receive messages from train nodes (realistic at inference time),
    # but the loss only backpropagates through train nodes.
    val_mask = (data.src_val_mask if data.src_val_mask is not None
                else np.zeros(len(Xs), dtype=bool))
    train_mask = ~val_mask
    y_tr_np = data.src_labels[train_mask].astype(np.int32)
    y_val_np = data.src_labels[val_mask].astype(np.int32)

    x_src = torch.from_numpy(Xs).to(device)
    x_tgt = torch.from_numpy(Xt).to(device)
    y_src = torch.from_numpy(data.src_labels.astype(np.float32)).to(device)
    tr_idx = torch.from_numpy(np.where(train_mask)[0]).long().to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).long().to(device)
    e_src = e_src.to(device); e_tgt = e_tgt.to(device)

    model = WindowGAT(in_dim=Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels[train_mask]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    stopper = EarlyStop(patience=patience, min_delta=1e-3)
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_src, e_src)
        loss = loss_fn(logits[tr_idx], y_src[tr_idx])
        loss.backward()
        opt.step()
        with torch.no_grad():
            scores_all = torch.sigmoid(logits).detach().cpu().numpy()
        try:
            tr_prauc = float(average_precision_score(y_tr_np, scores_all[train_mask]))
            val_prauc = float(average_precision_score(y_val_np, scores_all[val_mask])) \
                if val_mask.any() and len(set(y_val_np.tolist())) > 1 else tr_prauc
        except Exception:
            tr_prauc = val_prauc = 0.0
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            preds = (scores_all > 0.5).astype(np.int32)
            tr_acc = float((preds[train_mask] == y_tr_np).mean())
            print(f"    ep {ep+1:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"tr_acc={tr_acc:.3f}  tr_prauc={tr_prauc:.3f}  val_prauc={val_prauc:.3f}")
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            print(f"    [early-stop] ep={ep+1}  best_val_prauc={stopper.best:.3f}  "
                  f"best_ep={stopper.best_epoch}  (patience={patience})")
            break
    stopper.restore(model)

    model.eval()
    with torch.no_grad():
        logits = model(x_tgt, e_tgt)
        scores = torch.sigmoid(logits).cpu().numpy()
    info = {"best_val_prauc": float(stopper.best), "best_epoch": int(stopper.best_epoch),
            "stopped_ep": int(ep + 1), "patience": int(patience)}
    return scores, (scores > 0.5).astype(np.int64), info


def train_topo_gnn(data: PairData, source_domain: str, target_domain: str,
                   seed: int, epochs: int, device: torch.device,
                   patience: int = 3,
                   batch: int = 256,
                   use_edge_weights: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)

    src_topo = topo_key_for_domain(source_domain)
    tgt_topo = topo_key_for_domain(target_domain)
    src_as = target_as_for_domain(source_domain)
    tgt_as = target_as_for_domain(target_domain)

    src_stamps = [window_to_stamp(w) for w in data.src_meta["window_start"]]
    tgt_stamps = [window_to_stamp(w) for w in data.tgt_meta["window_start"]]

    # Pre-scan node-feature stats for standardization (using source snapshots).
    unique_src_stamps = sorted(set(src_stamps))
    node_feats_stack = []
    for s in unique_src_stamps:
        sg = load_snapshot(src_topo, s)
        node_feats_stack.append(sg.x.numpy())
    node_mu = np.concatenate(node_feats_stack).mean(axis=0)
    node_sd = np.concatenate(node_feats_stack).std(axis=0) + 1e-6
    mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
    sd_t = torch.from_numpy(node_sd.astype(np.float32)).to(device)

    # Do NOT mutate _SNAPSHOT_CACHE in place: the global cache holds RAW
    # node features and must stay raw so repeated train_topo_gnn() calls in
    # the same process each recompute their own standardization. We keep a
    # per-call standardized view on the device.
    standardized_x: dict[tuple[str, str], torch.Tensor] = {}

    def get_standardized_x(topo_key: str, stamp: str) -> torch.Tensor:
        key = (topo_key, stamp)
        if key not in standardized_x:
            sg = load_snapshot(topo_key, stamp)
            standardized_x[key] = (sg.x.to(device) - mu_t) / sd_t
        return standardized_x[key]

    model = TopoGAT(node_in=len(NODE_FEATURE_COLS),
                    window_in=Xs.shape[1],
                    use_edge_weights=use_edge_weights).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def forward_windows(X: np.ndarray, stamps: list[str], topo_key: str,
                        target_as: int, indices: np.ndarray) -> torch.Tensor:
        logits = []
        # Group by snapshot so each GAT pass serves all batched windows in that bucket.
        by_stamp: dict[str, list[int]] = {}
        for i in indices:
            by_stamp.setdefault(stamps[i], []).append(int(i))
        for stamp, idxs in by_stamp.items():
            sg = load_snapshot(topo_key, stamp)
            sg_x = get_standardized_x(topo_key, stamp); sg_ei = sg.edge_index.to(device)
            # Local GAT encode (inline to get grad); can't reuse encode_snapshot w/
            # module.train() vs eval() easily, so inline.
            if use_edge_weights:
                sg_ea = sg.edge_attr.to(device)
                h = F.elu(model.g1(sg_x, sg_ei, sg_ea))
                h = F.elu(model.g2(h, sg_ei, sg_ea))
            else:
                h = F.elu(model.g1(sg_x, sg_ei))
                h = F.elu(model.g2(h, sg_ei))
            if target_as not in sg.asn_to_idx:
                # Target AS not in ego graph at this slot (rare) -> zero embedding.
                tgt_emb = torch.zeros(h.size(1), device=device).unsqueeze(0).expand(len(idxs), -1)
            else:
                tgt_emb = h[sg.asn_to_idx[target_as]].unsqueeze(0).expand(len(idxs), -1)
            w_feat = torch.from_numpy(X[idxs]).to(device)
            z = torch.cat([tgt_emb, w_feat], dim=-1)
            out = model.mlp(z).squeeze(-1)
            for k, i in enumerate(idxs):
                logits.append((i, out[k]))
        logits.sort(key=lambda t: t[0])
        return torch.stack([v for _, v in logits])

    n_src = len(Xs)
    rng = np.random.default_rng(seed)
    val_mask = (data.src_val_mask if data.src_val_mask is not None
                else np.zeros(n_src, dtype=bool))
    train_idx_all = np.where(~val_mask)[0]
    val_idx_all = np.where(val_mask)[0]
    y_tr_np = data.src_labels[train_idx_all].astype(np.int32)
    y_val_np = data.src_labels[val_idx_all].astype(np.int32)
    stopper = EarlyStop(patience=patience, min_delta=1e-3)
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx_all)  # iterate train windows only
        total, n_batches = 0.0, 0
        tr_scores = np.zeros(n_src, dtype=np.float32)
        for start in range(0, len(order), batch):
            idxs = order[start:start + batch]
            idxs.sort()
            opt.zero_grad()
            logit_batch = forward_windows(Xs, src_stamps, src_topo, src_as, idxs)
            y_batch = torch.from_numpy(
                data.src_labels[idxs].astype(np.float32)).to(device)
            loss = loss_fn(logit_batch, y_batch)
            loss.backward()
            opt.step()
            total += loss.item(); n_batches += 1
            tr_scores[idxs] = torch.sigmoid(logit_batch).detach().cpu().numpy()
        # Val pass in eval mode (dropout off, no grads)
        model.eval()
        val_scores = np.zeros(n_src, dtype=np.float32)
        if len(val_idx_all):
            with torch.no_grad():
                for start in range(0, len(val_idx_all), batch):
                    idxs = val_idx_all[start:start + batch]
                    lb = forward_windows(Xs, src_stamps, src_topo, src_as, idxs)
                    val_scores[idxs] = torch.sigmoid(lb).cpu().numpy()
        try:
            tr_prauc = float(average_precision_score(y_tr_np, tr_scores[train_idx_all]))
            val_prauc = (float(average_precision_score(y_val_np, val_scores[val_idx_all]))
                         if len(val_idx_all) and len(set(y_val_np.tolist())) > 1 else tr_prauc)
        except Exception:
            tr_prauc = val_prauc = 0.0
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    ep {ep+1:3d}/{epochs}  loss_mean={total/max(1,n_batches):.4f}  "
                  f"tr_prauc={tr_prauc:.3f}  val_prauc={val_prauc:.3f}")
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            print(f"    [early-stop] ep={ep+1}  best_val_prauc={stopper.best:.3f}  "
                  f"best_ep={stopper.best_epoch}  (patience={patience})")
            break
    stopper.restore(model)

    model.eval()
    with torch.no_grad():
        all_idx = np.arange(len(Xt))
        scores = np.zeros(len(Xt), dtype=np.float32)
        for start in range(0, len(Xt), batch):
            idxs = all_idx[start:start + batch]
            logit_batch = forward_windows(Xt, tgt_stamps, tgt_topo, tgt_as, idxs)
            scores[idxs] = torch.sigmoid(logit_batch).cpu().numpy()
    info = {"best_val_prauc": float(stopper.best), "best_epoch": int(stopper.best_epoch),
            "stopped_ep": int(ep + 1), "patience": int(patience),
            "model": model, "node_mu": node_mu, "node_sd": node_sd}
    return scores, (scores > 0.5).astype(np.int64), info


# -------------------------------------------------------------------- HGT
# Heterogeneous AS-topology GNN typed by CAIDA p2p / p2c / c2p relationships.

HGT_EDGE_TYPES = [
    ("as", "p2p", "as"),
    ("as", "p2c", "as"),
    ("as", "c2p", "as"),
    ("as", "unknown", "as"),  # endpoints not in CAIDA's AS-Relationships snapshot
]
HGT_METADATA = (["as"], HGT_EDGE_TYPES)


@dataclass
class HeteroSnapshot:
    x: torch.Tensor                                    # (N, F_node)
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    asn_to_idx: dict[int, int]


_HETERO_SNAPSHOT_CACHE: dict[tuple[str, str], HeteroSnapshot] = {}


def load_snapshot_hetero(pair_topo_key: str, stamp: str,
                         p2c: set[tuple[int, int]],
                         p2p: set[tuple[int, int]]) -> HeteroSnapshot:
    key = (pair_topo_key, stamp)
    if key in _HETERO_SNAPSHOT_CACHE:
        return _HETERO_SNAPSHOT_CACHE[key]
    d = TOPO_ROOT / pair_topo_key / stamp
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    asns = nodes["asn"].to_numpy()
    asn_to_idx = {int(a): i for i, a in enumerate(asns)}
    si = edges["source"].map(asn_to_idx).to_numpy()
    di = edges["target"].map(asn_to_idx).to_numpy()
    typed = classify_edges(si, di, asns, p2c, p2p)
    # Append a dedicated sentinel node (all-zero features, not in asn_to_idx)
    # so empty edge-type placeholders (below) never bias a real AS embedding —
    # in particular, avoids contaminating node 0 whose ASN could be the target.
    x_real = nodes[NODE_FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    sentinel = np.zeros((1, x_real.shape[1]), dtype=np.float32)
    x = np.concatenate([x_real, sentinel], axis=0)
    sentinel_idx = len(asns)  # row index of the sentinel
    # Keep 'unknown' as an explicit 4th edge type so endpoints missing from
    # CAIDA still pass messages (~3 % of edges here; methodologically cleaner
    # than collapsing them into p2p or dropping).
    eid: dict[tuple[str, str, str], torch.Tensor] = {}
    for name, rel in (("p2p", ("as", "p2p", "as")),
                      ("p2c", ("as", "p2c", "as")),
                      ("c2p", ("as", "c2p", "as")),
                      ("unknown", ("as", "unknown", "as"))):
        arr = typed[name]
        if arr.shape[1] == 0:
            # HGTConv needs a non-empty edge_index per type; self-loop the
            # sentinel so no real node receives spurious messages from it.
            arr = np.array([[sentinel_idx], [sentinel_idx]], dtype=np.int64)
        eid[rel] = torch.from_numpy(arr).long()
    hs = HeteroSnapshot(
        x=torch.from_numpy(x),
        edge_index_dict=eid,
        asn_to_idx=asn_to_idx,
    )
    _HETERO_SNAPSHOT_CACHE[key] = hs
    return hs


class TopoHGT(nn.Module):
    """Heterogeneous AS-topology GNN with CAIDA-typed edges."""
    def __init__(self, node_in: int, window_in: int, embed: int = 16,
                 heads: int = 4, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lin_in = nn.Linear(node_in, embed * heads)
        self.hgt1 = HGTConv(
            in_channels=embed * heads,
            out_channels=embed * heads,
            metadata=HGT_METADATA,
            heads=heads,
        )
        self.hgt2 = HGTConv(
            in_channels=embed * heads,
            out_channels=embed,
            metadata=HGT_METADATA,
            heads=1,
        )
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed + window_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor,
               edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        x_dict = {"as": self.lin_in(x)}
        x_dict = self.hgt1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = {k: self.drop(v) for k, v in x_dict.items()}
        x_dict = self.hgt2(x_dict, edge_index_dict)
        return F.elu(x_dict["as"])


def train_topo_hgt(data: PairData, source_domain: str, target_domain: str,
                   seed: int, epochs: int, device: torch.device,
                   patience: int = 3,
                   batch: int = 256) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)

    src_topo = topo_key_for_domain(source_domain)
    tgt_topo = topo_key_for_domain(target_domain)
    src_as = target_as_for_domain(source_domain)
    tgt_as = target_as_for_domain(target_domain)

    src_stamps = [window_to_stamp(w) for w in data.src_meta["window_start"]]
    tgt_stamps = [window_to_stamp(w) for w in data.tgt_meta["window_start"]]

    # Pick the CAIDA snapshot nearest to the study start date (not lexicographic).
    study_ymd = src_stamps[0].split("_")[0] if src_stamps else None
    p2c, p2p = load_caida(ROOT / "runs", target_ymd=study_ymd)
    print(f"  [CAIDA] p2c={len(p2c):,} pairs  p2p={len(p2p):,} pairs  "
          f"(date_anchor={study_ymd})")

    # Node-feature stats from source snapshots (raw).
    unique_src_stamps = sorted(set(src_stamps))
    stacks = [load_snapshot_hetero(src_topo, s, p2c, p2p).x.numpy()
              for s in unique_src_stamps]
    node_mu = np.concatenate(stacks).mean(axis=0)
    node_sd = np.concatenate(stacks).std(axis=0) + 1e-6
    mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
    sd_t = torch.from_numpy(node_sd.astype(np.float32)).to(device)

    standardized_x: dict[tuple[str, str], torch.Tensor] = {}

    def get_std_x(topo_key: str, stamp: str) -> torch.Tensor:
        key = (topo_key, stamp)
        if key not in standardized_x:
            hs = load_snapshot_hetero(topo_key, stamp, p2c, p2p)
            standardized_x[key] = (hs.x.to(device) - mu_t) / sd_t
        return standardized_x[key]

    model = TopoHGT(node_in=len(NODE_FEATURE_COLS),
                    window_in=Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def forward_windows(X: np.ndarray, stamps: list[str], topo_key: str,
                        target_as: int, indices: np.ndarray) -> torch.Tensor:
        logits = []
        by_stamp: dict[str, list[int]] = {}
        for i in indices:
            by_stamp.setdefault(stamps[i], []).append(int(i))
        for stamp, idxs in by_stamp.items():
            hs = load_snapshot_hetero(topo_key, stamp, p2c, p2p)
            x_std = get_std_x(topo_key, stamp)
            eid = {k: v.to(device) for k, v in hs.edge_index_dict.items()}
            h = model.encode(x_std, eid)
            if target_as not in hs.asn_to_idx:
                tgt_emb = torch.zeros(h.size(1), device=device).unsqueeze(0).expand(len(idxs), -1)
            else:
                tgt_emb = h[hs.asn_to_idx[target_as]].unsqueeze(0).expand(len(idxs), -1)
            w_feat = torch.from_numpy(X[idxs]).to(device)
            z = torch.cat([tgt_emb, w_feat], dim=-1)
            out = model.mlp(z).squeeze(-1)
            for k, i in enumerate(idxs):
                logits.append((i, out[k]))
        logits.sort(key=lambda t: t[0])
        return torch.stack([v for _, v in logits])

    n_src = len(Xs)
    rng = np.random.default_rng(seed)
    val_mask = (data.src_val_mask if data.src_val_mask is not None
                else np.zeros(n_src, dtype=bool))
    train_idx_all = np.where(~val_mask)[0]
    val_idx_all = np.where(val_mask)[0]
    y_tr_np = data.src_labels[train_idx_all].astype(np.int32)
    y_val_np = data.src_labels[val_idx_all].astype(np.int32)
    stopper = EarlyStop(patience=patience, min_delta=1e-3)
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx_all)
        total, nb = 0.0, 0
        tr_scores = np.zeros(n_src, dtype=np.float32)
        for start in range(0, len(order), batch):
            idxs = order[start:start + batch]; idxs.sort()
            opt.zero_grad()
            logit_batch = forward_windows(Xs, src_stamps, src_topo, src_as, idxs)
            y_batch = torch.from_numpy(
                data.src_labels[idxs].astype(np.float32)).to(device)
            loss = loss_fn(logit_batch, y_batch)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
            tr_scores[idxs] = torch.sigmoid(logit_batch).detach().cpu().numpy()
        model.eval()
        val_scores = np.zeros(n_src, dtype=np.float32)
        if len(val_idx_all):
            with torch.no_grad():
                for start in range(0, len(val_idx_all), batch):
                    idxs = val_idx_all[start:start + batch]
                    lb = forward_windows(Xs, src_stamps, src_topo, src_as, idxs)
                    val_scores[idxs] = torch.sigmoid(lb).cpu().numpy()
        try:
            tr_prauc = float(average_precision_score(y_tr_np, tr_scores[train_idx_all]))
            val_prauc = (float(average_precision_score(y_val_np, val_scores[val_idx_all]))
                         if len(val_idx_all) and len(set(y_val_np.tolist())) > 1 else tr_prauc)
        except Exception:
            tr_prauc = val_prauc = 0.0
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    ep {ep+1:3d}/{epochs}  loss_mean={total/max(1,nb):.4f}  "
                  f"tr_prauc={tr_prauc:.3f}  val_prauc={val_prauc:.3f}")
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            print(f"    [early-stop] ep={ep+1}  best_val_prauc={stopper.best:.3f}  "
                  f"best_ep={stopper.best_epoch}  (patience={patience})")
            break
    stopper.restore(model)

    model.eval()
    with torch.no_grad():
        all_idx = np.arange(len(Xt))
        scores = np.zeros(len(Xt), dtype=np.float32)
        for start in range(0, len(Xt), batch):
            idxs = all_idx[start:start + batch]
            logit_batch = forward_windows(Xt, tgt_stamps, tgt_topo, tgt_as, idxs)
            scores[idxs] = torch.sigmoid(logit_batch).cpu().numpy()
    info = {"best_val_prauc": float(stopper.best), "best_epoch": int(stopper.best_epoch),
            "stopped_ep": int(ep + 1), "patience": int(patience)}
    return scores, (scores > 0.5).astype(np.int64), info


# -------------------------------------------------------------------- GPS
# GraphGPS: local MPNN (GIN) + global attention (Transformer) per snapshot.
# Uses the same SnapshotGraph as TopoGAT (homogeneous, no edge types required).

_PE_CACHE: dict[tuple[str, str, int], torch.Tensor] = {}


def compute_lap_pe(pair_topo_key: str, stamp: str, k: int) -> torch.Tensor:
    """Laplacian positional encoding: k smallest non-trivial eigenvectors of
    the normalized Laplacian. Zero-padded when the graph has fewer than k+2
    nodes. Returns (N, k) float32; eigenvector signs are random per call
    (sign-flip augmentation happens at the train loop level)."""
    key = (pair_topo_key, stamp, k)
    if key in _PE_CACHE:
        return _PE_CACHE[key]
    sg = load_snapshot(pair_topo_key, stamp)
    n = sg.x.size(0)
    k_eff = max(1, min(k, n - 2))
    pe_tensor = torch.zeros(n, k, dtype=torch.float32)
    try:
        d = Data(x=sg.x, edge_index=sg.edge_index, num_nodes=n)
        d = AddLaplacianEigenvectorPE(k=k_eff, is_undirected=True,
                                      attr_name="pe")(d)
        pe = d.pe.to(torch.float32)
        pe_tensor[:, :pe.size(1)] = pe
    except Exception:
        # Disconnected tiny graphs can fail scipy eigsh; fall back to zeros.
        pass
    _PE_CACHE[key] = pe_tensor
    return pe_tensor


class TopoGPS(nn.Module):
    """Two GPSConv blocks over the AS-topology ego-graph.

    When `pe_dim > 0`, node features are concatenated with a `pe_dim`-wide
    Laplacian positional encoding at forward time — the canonical GraphGPS
    recipe lets self-attention reason about position in the graph, which the
    base GINConv local message-passing can only see indirectly.
    """
    def __init__(self, node_in: int, window_in: int, channels: int = 32,
                 heads: int = 4, hidden: int = 64, dropout: float = 0.3,
                 pe_dim: int = 0):
        super().__init__()
        self.pe_dim = pe_dim
        self.lin_in = nn.Linear(node_in + pe_dim, channels)

        def gin_block() -> GINConv:
            return GINConv(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.ReLU(),
                    nn.Linear(channels, channels),
                ),
                train_eps=True,
            )

        self.gps1 = GPSConv(channels, conv=gin_block(), heads=heads,
                            dropout=dropout)
        self.gps2 = GPSConv(channels, conv=gin_block(), heads=heads,
                            dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(channels + window_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               pe: torch.Tensor | None = None) -> torch.Tensor:
        if self.pe_dim > 0:
            assert pe is not None, "TopoGPS(pe_dim>0) needs `pe` at forward time"
            x = torch.cat([x, pe], dim=-1)
        h = self.lin_in(x)
        h = self.gps1(h, edge_index)
        h = F.elu(h)
        h = self.gps2(h, edge_index)
        return F.elu(h)


def train_topo_gps(data: PairData, source_domain: str, target_domain: str,
                   seed: int, epochs: int, device: torch.device,
                   patience: int = 3,
                   batch: int = 256,
                   pe_dim: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """When pe_dim > 0, concatenate a Laplacian PE to node features each pass
    (with random sign flips during training for eigenvector-sign invariance)."""
    set_seed(seed)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)

    src_topo = topo_key_for_domain(source_domain)
    tgt_topo = topo_key_for_domain(target_domain)
    src_as = target_as_for_domain(source_domain)
    tgt_as = target_as_for_domain(target_domain)

    src_stamps = [window_to_stamp(w) for w in data.src_meta["window_start"]]
    tgt_stamps = [window_to_stamp(w) for w in data.tgt_meta["window_start"]]

    unique_src_stamps = sorted(set(src_stamps))
    stacks = [load_snapshot(src_topo, s).x.numpy() for s in unique_src_stamps]
    node_mu = np.concatenate(stacks).mean(axis=0)
    node_sd = np.concatenate(stacks).std(axis=0) + 1e-6
    mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
    sd_t = torch.from_numpy(node_sd.astype(np.float32)).to(device)

    standardized_x: dict[tuple[str, str], torch.Tensor] = {}

    def get_std_x(topo_key: str, stamp: str) -> torch.Tensor:
        key = (topo_key, stamp)
        if key not in standardized_x:
            sg = load_snapshot(topo_key, stamp)
            standardized_x[key] = (sg.x.to(device) - mu_t) / sd_t
        return standardized_x[key]

    model = TopoGPS(node_in=len(NODE_FEATURE_COLS),
                    window_in=Xs.shape[1], pe_dim=pe_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Per-snapshot PE cache on the target device; random sign flip is applied
    # each training step (not at cache time) to respect eigenvector sign
    # ambiguity without memoizing it.
    pe_cache_device: dict[tuple[str, str], torch.Tensor] = {}

    def get_pe(topo_key: str, stamp: str) -> torch.Tensor:
        key = (topo_key, stamp)
        if key not in pe_cache_device:
            pe_cache_device[key] = compute_lap_pe(topo_key, stamp, pe_dim).to(device)
        return pe_cache_device[key]

    def forward_windows(X: np.ndarray, stamps: list[str], topo_key: str,
                        target_as: int, indices: np.ndarray,
                        pe_flip: bool) -> torch.Tensor:
        logits = []
        by_stamp: dict[str, list[int]] = {}
        for i in indices:
            by_stamp.setdefault(stamps[i], []).append(int(i))
        for stamp, idxs in by_stamp.items():
            sg = load_snapshot(topo_key, stamp)
            sg_x = get_std_x(topo_key, stamp)
            sg_ei = sg.edge_index.to(device)
            if pe_dim > 0:
                pe = get_pe(topo_key, stamp)
                if pe_flip:
                    signs = torch.randint(0, 2, (pe_dim,), device=device,
                                          dtype=pe.dtype) * 2 - 1
                    pe = pe * signs.unsqueeze(0)
                h = model.encode(sg_x, sg_ei, pe=pe)
            else:
                h = model.encode(sg_x, sg_ei)
            if target_as not in sg.asn_to_idx:
                tgt_emb = torch.zeros(h.size(1), device=device).unsqueeze(0).expand(len(idxs), -1)
            else:
                tgt_emb = h[sg.asn_to_idx[target_as]].unsqueeze(0).expand(len(idxs), -1)
            w_feat = torch.from_numpy(X[idxs]).to(device)
            z = torch.cat([tgt_emb, w_feat], dim=-1)
            out = model.mlp(z).squeeze(-1)
            for k, i in enumerate(idxs):
                logits.append((i, out[k]))
        logits.sort(key=lambda t: t[0])
        return torch.stack([v for _, v in logits])

    n_src = len(Xs)
    rng = np.random.default_rng(seed)
    val_mask = (data.src_val_mask if data.src_val_mask is not None
                else np.zeros(n_src, dtype=bool))
    train_idx_all = np.where(~val_mask)[0]
    val_idx_all = np.where(val_mask)[0]
    y_tr_np = data.src_labels[train_idx_all].astype(np.int32)
    y_val_np = data.src_labels[val_idx_all].astype(np.int32)
    stopper = EarlyStop(patience=patience, min_delta=1e-3)
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx_all)
        total, nb = 0.0, 0
        tr_scores = np.zeros(n_src, dtype=np.float32)
        for start in range(0, len(order), batch):
            idxs = order[start:start + batch]; idxs.sort()
            opt.zero_grad()
            logit_batch = forward_windows(Xs, src_stamps, src_topo, src_as,
                                          idxs, pe_flip=(pe_dim > 0))
            y_batch = torch.from_numpy(
                data.src_labels[idxs].astype(np.float32)).to(device)
            loss = loss_fn(logit_batch, y_batch)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
            tr_scores[idxs] = torch.sigmoid(logit_batch).detach().cpu().numpy()
        model.eval()
        val_scores = np.zeros(n_src, dtype=np.float32)
        if len(val_idx_all):
            with torch.no_grad():
                for start in range(0, len(val_idx_all), batch):
                    idxs = val_idx_all[start:start + batch]
                    lb = forward_windows(Xs, src_stamps, src_topo, src_as,
                                         idxs, pe_flip=False)
                    val_scores[idxs] = torch.sigmoid(lb).cpu().numpy()
        try:
            tr_prauc = float(average_precision_score(y_tr_np, tr_scores[train_idx_all]))
            val_prauc = (float(average_precision_score(y_val_np, val_scores[val_idx_all]))
                         if len(val_idx_all) and len(set(y_val_np.tolist())) > 1 else tr_prauc)
        except Exception:
            tr_prauc = val_prauc = 0.0
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    ep {ep+1:3d}/{epochs}  loss_mean={total/max(1,nb):.4f}  "
                  f"tr_prauc={tr_prauc:.3f}  val_prauc={val_prauc:.3f}")
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            print(f"    [early-stop] ep={ep+1}  best_val_prauc={stopper.best:.3f}  "
                  f"best_ep={stopper.best_epoch}  (patience={patience})")
            break
    stopper.restore(model)

    model.eval()
    with torch.no_grad():
        all_idx = np.arange(len(Xt))
        scores = np.zeros(len(Xt), dtype=np.float32)
        for start in range(0, len(Xt), batch):
            idxs = all_idx[start:start + batch]
            logit_batch = forward_windows(Xt, tgt_stamps, tgt_topo, tgt_as,
                                          idxs, pe_flip=False)
            scores[idxs] = torch.sigmoid(logit_batch).cpu().numpy()
    info = {"best_val_prauc": float(stopper.best), "best_epoch": int(stopper.best_epoch),
            "stopped_ep": int(ep + 1), "patience": int(patience),
            "model": model, "node_mu": node_mu, "node_sd": node_sd}
    return scores, (scores > 0.5).astype(np.int64), info


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-domain", required=True)
    ap.add_argument("--target-domain", required=True)
    ap.add_argument("--bucket", choices=("cross_as", "same_as"), required=True)
    ap.add_argument("--edge",
                    choices=("temporal", "knn", "astopo_8h",
                             "astopo_8h_weighted", "astopo_8h_hgt",
                             "astopo_8h_gps", "astopo_8h_gps_pe"),
                    required=True)
    ap.add_argument("--pe-k", type=int, default=8,
                    help="Laplacian PE eigenvector count for astopo_8h_gps_pe")
    ap.add_argument("--view", choices=tuple(VIEW_COLS.keys()), default="fusion_early")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coral-kind", default="study_only",
                    choices=("study_only", "study_plus_historical"))
    ap.add_argument("--balance-fit", default="none",
                    choices=("none", "equal"),
                    help="source-balancing regime used during CORAL fit")
    ap.add_argument("--alignment", default="after_coral",
                    choices=("before_coral", "after_coral"),
                    help="source feature matrix: raw (before) or CORAL-aligned (after)")
    ap.add_argument("--source-labels-from", type=Path, default=None)
    ap.add_argument("--output-suffix", default=None,
                    help="override output dir; default derives from view/balance_fit")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=-1,
                    help="EarlyStop patience on held-out val PR-AUC. "
                         "-1 = use trainer default (5 for window, 3 for topo).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pair = f"{args.source_domain}__to__{args.target_domain}"
    # Auto-compute pair_dir suffix so the aggregator parses feature_set and
    # balance_fit correctly (see build_supervisor_metrics_csv.parse_dir_name).
    pair_dir_name = pair + pair_dir_suffix(args.balance_fit, args.view)
    # Historical-augmented CORAL routes to stage1b_historical/, matching the
    # layout other classifiers use for the study_plus_historical arm.
    root_bucket = ("stage1b_historical"
                   if args.coral_kind == "study_plus_historical"
                   else "transfer_matrix")
    suffix = args.output_suffix or f"{root_bucket}/{args.bucket}/{pair_dir_name}"
    out_dir = ROOT / "bgp_unified_results" / "phase3_fusion" / suffix / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_name = f"gnn_{args.edge}"
    out_csv = out_dir / f"{args.view}__{args.alignment}__{clf_name}__seed{args.seed}.csv"

    print(f"[GNN] pair={pair}  edge={args.edge}  view={args.view}  "
          f"seed={args.seed}  device={args.device}")
    print(f"[GNN] coral_kind={args.coral_kind}  balance_fit={args.balance_fit}  "
          f"alignment={args.alignment}")
    print(f"[GNN] out={out_csv.relative_to(ROOT)}")

    data = load_pair(pair, args.coral_kind, args.balance_fit, args.alignment,
                     args.view, args.source_labels_from)
    print(f"[GNN] src={data.src_feats.shape} anom={(data.src_labels==1).sum()} "
          f"| tgt={data.tgt_feats.shape} anom={(data.tgt_labels==1).sum()}")

    device = torch.device(args.device)

    # Per-family default patience (matches "separately tuned per classifier" spec).
    default_patience = 5 if args.edge in ("temporal", "knn") else 3
    patience = args.patience if args.patience > 0 else default_patience
    print(f"[GNN] patience={patience}  val_frac={VAL_FRAC}  "
          f"val_n={int(data.src_val_mask.sum())}  "
          f"val_anom={int(data.src_labels[data.src_val_mask].sum())}")

    if args.edge in ("temporal", "knn"):
        scores, preds, train_info = train_window_gnn(
            data, edge=args.edge, seed=args.seed, epochs=args.epochs,
            device=device, patience=patience)
    elif args.edge == "astopo_8h_hgt":
        scores, preds, train_info = train_topo_hgt(
            data, source_domain=args.source_domain,
            target_domain=args.target_domain, seed=args.seed,
            epochs=args.epochs, device=device, patience=patience)
    elif args.edge in ("astopo_8h_gps", "astopo_8h_gps_pe"):
        pe_dim = args.pe_k if args.edge == "astopo_8h_gps_pe" else 0
        scores, preds, train_info = train_topo_gps(
            data, source_domain=args.source_domain,
            target_domain=args.target_domain, seed=args.seed,
            epochs=args.epochs, device=device, patience=patience,
            pe_dim=pe_dim)
    else:  # astopo_8h or astopo_8h_weighted
        scores, preds, train_info = train_topo_gnn(
            data, source_domain=args.source_domain,
            target_domain=args.target_domain, seed=args.seed,
            epochs=args.epochs, device=device, patience=patience,
            use_edge_weights=(args.edge == "astopo_8h_weighted"))

    out = data.tgt_meta.copy()
    out["y_pred"] = preds.astype(np.int64)
    out["y_score"] = scores.astype(np.float64)
    out = out[["window_start", "window_id", "binary_label", "y_pred", "y_score", "event_id"]]
    out.to_csv(out_csv, index=False)
    # Side-car: held-out val PR-AUC, stop epoch, patience — used by patience tuner.
    import json
    summary = {
        "pair": pair, "edge": args.edge, "view": args.view, "seed": args.seed,
        "coral_kind": args.coral_kind, "balance_fit": args.balance_fit,
        "alignment": args.alignment, "bucket": args.bucket,
        "val_frac": VAL_FRAC,
        "val_n": int(data.src_val_mask.sum()),
        "val_anom": int(data.src_labels[data.src_val_mask].sum()),
        **train_info,
    }
    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    from sklearn.metrics import f1_score, average_precision_score, recall_score
    print(f"[GNN] wrote {out_csv.name}  rows={len(out)}  anom_rate={out['binary_label'].mean():.3f}")
    print(f"[GNN] summary: best_val_prauc={train_info['best_val_prauc']:.4f}  "
          f"best_ep={train_info['best_epoch']}  stopped_ep={train_info['stopped_ep']}")
    print(f"  macro_f1       = {f1_score(out['binary_label'], out['y_pred'], average='macro'):.4f}")
    print(f"  recall_anomaly = {recall_score(out['binary_label'], out['y_pred']):.4f}")
    try:
        print(f"  pr_auc_anomaly = {average_precision_score(out['binary_label'], out['y_score']):.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()