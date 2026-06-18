#!/usr/bin/env python3
"""Post-canonical per-window latency measurement for GNN classifiers.

Background
----------
`Scripts/gnn_classifiers.py` trains and emits prediction CSVs but does NOT
write any latency columns (unlike `phase3_pipeline.py` for flat classifiers,
which already emits `single_window_ms_median` etc. into `metrics_row_level.csv`).

This script measures end-to-end per-window inference latency for GNN cells:

* `temporal` / `knn` (window-as-node GNNs): graph is built once over all
  target windows, then a single batched forward produces scores. Per-window
  latency = batched_forward_ms / n_target_windows.

* `astopo_8h` / `astopo_8h_weighted` (AS-topology GAT): per-window inference
  loads the 8-hour topology snapshot and forwards through GAT. We measure
  snapshot-load + forward separately because deployment cost depends on
  topology refresh cadence.

Methodology
-----------
- Full training under the post-determinism contract (strict cuDNN,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `set_seed(seed, strict=True)`).
- N=100 timed single-window (or batched) inferences with 10 warm-up discards.
- Median + p95 reported. Times include CUDA stream synchronization.

Usage
-----
Run ONE cell:
    python Scripts/measure_gnn_latency.py one \
        --source-domain rrc04_as12880 --target-domain rrc05_as12880 \
        --bucket same_as --edge astopo_8h --seed 0

Run many cells from a manifest CSV (columns: source_domain, target_domain,
bucket, edge, view, seed, coral_kind, balance_fit, alignment):
    python Scripts/measure_gnn_latency.py manifest \
        --manifest latency_cells.csv

Outputs
-------
- `summaries/gnn_latency_canonical.csv` (append-friendly) with columns:
  source, target, bucket, edge, view, seed, coral_kind, balance_fit,
  alignment, n_target_windows, train_seconds, graph_build_ms_median,
  graph_build_ms_p95, forward_ms_median, forward_ms_p95,
  single_window_ms_median, single_window_ms_p95,
  batched_ms_per_window_median.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo Scripts/ on sys.path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Strict-determinism env var must be set before torch imports cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gnn_classifiers as _gnn_mod  # noqa: E402  (for _SNAPSHOT_CACHE access)
from gnn_classifiers import (  # noqa: E402
    load_pair,
    standardize,
    temporal_edges,
    knn_edges,
    WindowGAT,
    TopoGAT,
    class_pos_weight,
    EarlyStop,
    set_seed,
    load_snapshot,
    topo_key_for_domain,
    target_as_for_domain,
    window_to_stamp,
    NODE_FEATURE_COLS,
    VAL_FRAC,
)
from run_astopo_5m import load_snapshot_5min, window_to_stamp_5min  # noqa: E402


def _nan_latency_row(source: str, target: str, bucket: str, edge: str,
                     view: str, seed: int, coral_kind: str,
                     balance_fit: str, alignment: str,
                     n_target_windows: int, reason: str) -> dict:
    """Return a schema-compatible row with NaN timings when measurement is unsafe."""
    return {
        "source": source, "target": target, "bucket": bucket, "edge": edge,
        "view": view, "seed": seed, "coral_kind": coral_kind,
        "balance_fit": balance_fit, "alignment": alignment,
        "n_target_windows": n_target_windows,
        "train_seconds": float("nan"),
        "graph_build_ms_once": float("nan"),
        "graph_build_ms_median": float("nan"),
        "graph_build_ms_p95": float("nan"),
        "forward_ms_median": float("nan"),
        "forward_ms_p95": float("nan"),
        "single_window_ms_median": float("nan"),
        "single_window_ms_p95": float("nan"),
        "batched_ms_per_window_median": float("nan"),
        "skip_reason": reason,
    }

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = REPO_ROOT / "bgp_unified_results" / "phase3_fusion" / "summaries"
OUT_CSV = SUMMARIES / "gnn_latency_canonical.csv"

N_SAMPLES = 100
N_WARMUP = 10


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_window_gnn(source: str, target: str, bucket: str, edge: str,
                       view: str, seed: int, coral_kind: str,
                       balance_fit: str, alignment: str,
                       epochs: int, patience: int, device: torch.device,
                       k_temporal: int = 3, k_knn: int = 10) -> dict:
    """Train a temporal or kNN window-as-node GAT, then time batched forward."""
    assert edge in ("temporal", "knn")
    pair = f"{source}__to__{target}"
    data = load_pair(pair, coral_kind, balance_fit, alignment, view, None)

    Xs, Xt = standardize(data.src_feats, data.tgt_feats)
    n_tgt = len(Xt)
    if n_tgt == 0:
        return _nan_latency_row(source, target, bucket, edge, view, seed,
                                coral_kind, balance_fit, alignment,
                                n_target_windows=0,
                                reason="empty target split (len(Xt)==0)")

    # Build source edges (training-side — not counted in deployment latency)
    if edge == "temporal":
        e_src = temporal_edges(len(Xs), k=k_temporal)
    else:
        e_src = knn_edges(Xs, k=k_knn)

    # Build target edges — this is the one-off inference-side graph-build cost
    # a deployed system pays once, separately timed.
    t0 = time.perf_counter()
    if edge == "temporal":
        e_tgt = temporal_edges(n_tgt, k=k_temporal)
    else:
        e_tgt = knn_edges(Xt, k=k_knn)
    graph_build_once_ms = (time.perf_counter() - t0) * 1000.0

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
    e_src = e_src.to(device)
    e_tgt = e_tgt.to(device)

    set_seed(seed, strict=True)
    model = WindowGAT(in_dim=Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels[train_mask]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    stopper = EarlyStop(patience=patience, min_delta=1e-3)
    from sklearn.metrics import average_precision_score

    t_train_start = time.perf_counter()
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
            val_prauc = (float(average_precision_score(y_val_np, scores_all[val_mask]))
                         if val_mask.any() and len(set(y_val_np.tolist())) > 1 else tr_prauc)
        except Exception:
            tr_prauc = val_prauc = 0.0
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            break
    stopper.restore(model)
    train_seconds = time.perf_counter() - t_train_start

    # --- Latency measurement -------------------------------------------------
    model.eval()

    # Warm-up batched forwards (discard timings)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x_tgt, e_tgt)
        _sync(device)

    # Batched forward (the natural deployment pattern for window-as-node GNN):
    # whole target graph → all verdicts in one pass.
    ms_batched = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_tgt, e_tgt)
        _sync(device)
        ms_batched.append((time.perf_counter() - t0) * 1000.0)

    batched_per_win = np.array(ms_batched) / max(n_tgt, 1)

    return {
        "source": source, "target": target, "bucket": bucket, "edge": edge,
        "view": view, "seed": seed, "coral_kind": coral_kind,
        "balance_fit": balance_fit, "alignment": alignment,
        "n_target_windows": n_tgt,
        "train_seconds": round(train_seconds, 3),
        # Graph is built once per deployment, not per window — report target-side
        # (inference-side) one-off cost only; source-side edge build is a training
        # concern folded into train_seconds.
        "graph_build_ms_once": round(graph_build_once_ms, 3),
        "graph_build_ms_median": float("nan"),
        "graph_build_ms_p95": float("nan"),
        # Forward here = full batched target forward. Per-window = batched/n.
        "forward_ms_median": round(float(np.median(ms_batched)), 4),
        "forward_ms_p95": round(float(np.percentile(ms_batched, 95)), 4),
        "single_window_ms_median": round(float(np.median(batched_per_win)), 4),
        "single_window_ms_p95": round(float(np.percentile(batched_per_win, 95)), 4),
        "batched_ms_per_window_median": round(float(np.median(batched_per_win)), 4),
        "skip_reason": "",
    }


def measure_topo_gnn(source: str, target: str, bucket: str, edge: str,
                     view: str, seed: int, coral_kind: str,
                     balance_fit: str, alignment: str,
                     epochs: int, patience: int, device: torch.device,
                     use_edge_weights: bool = False, batch: int = 256) -> dict:
    """Train astopo_8h TopoGAT, then time per-window snapshot load + forward."""
    assert edge in ("astopo_8h", "astopo_8h_weighted", "astopo_5m")
    pair = f"{source}__to__{target}"
    data = load_pair(pair, coral_kind, balance_fit, alignment, view, None)
    Xs, Xt = standardize(data.src_feats, data.tgt_feats)

    src_topo = topo_key_for_domain(source)
    tgt_topo = topo_key_for_domain(target)
    src_as = target_as_for_domain(source)
    tgt_as = target_as_for_domain(target)

    src_stamps = [window_to_stamp(w) for w in data.src_meta["window_start"]]
    tgt_stamps = [window_to_stamp(w) for w in data.tgt_meta["window_start"]]

    # Pre-scan node-feature stats for standardization (sources only, matching trainer).
    unique_src_stamps = sorted(set(src_stamps))
    node_feats_stack = []
    for s in unique_src_stamps:
        sg = load_snapshot(src_topo, s)
        node_feats_stack.append(sg.x.numpy())
    node_mu = np.concatenate(node_feats_stack).mean(axis=0)
    node_sd = np.concatenate(node_feats_stack).std(axis=0) + 1e-6
    mu_t = torch.from_numpy(node_mu.astype(np.float32)).to(device)
    sd_t = torch.from_numpy(node_sd.astype(np.float32)).to(device)

    set_seed(seed, strict=True)
    model = TopoGAT(node_in=len(NODE_FEATURE_COLS),
                    window_in=Xs.shape[1],
                    use_edge_weights=use_edge_weights).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    pos_w = class_pos_weight(data.src_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Standardized-x cache (per-process), same pattern as gnn_classifiers.
    standardized_x: dict[tuple[str, str], torch.Tensor] = {}

    def get_std_x(topo_key: str, stamp: str) -> torch.Tensor:
        key = (topo_key, stamp)
        if key not in standardized_x:
            sg = load_snapshot(topo_key, stamp)
            standardized_x[key] = (sg.x.to(device) - mu_t) / sd_t
        return standardized_x[key]

    def forward_windows(X, stamps, topo_key, tgt_as_local, indices):
        logits = []
        by_stamp: dict[str, list[int]] = {}
        for i in indices:
            by_stamp.setdefault(stamps[i], []).append(int(i))
        for stamp, idxs in by_stamp.items():
            sg = load_snapshot(topo_key, stamp)
            sg_x = get_std_x(topo_key, stamp)
            sg_ei = sg.edge_index.to(device)
            if use_edge_weights:
                sg_ea = sg.edge_attr.to(device)
                h = F.elu(model.g1(sg_x, sg_ei, sg_ea))
                h = F.elu(model.g2(h, sg_ei, sg_ea))
            else:
                h = F.elu(model.g1(sg_x, sg_ei))
                h = F.elu(model.g2(h, sg_ei))
            if tgt_as_local not in sg.asn_to_idx:
                tgt_emb = torch.zeros(h.size(1), device=device).unsqueeze(0).expand(len(idxs), -1)
            else:
                tgt_emb = h[sg.asn_to_idx[tgt_as_local]].unsqueeze(0).expand(len(idxs), -1)
            w_feat = torch.from_numpy(X[idxs]).to(device)
            z = torch.cat([tgt_emb, w_feat], dim=-1)
            out = model.mlp(z).squeeze(-1)
            for k, i in enumerate(idxs):
                logits.append((i, out[k]))
        logits.sort(key=lambda t: t[0])
        return torch.stack([v for _, v in logits])

    # --- Training ------------------------------------------------------------
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

    t_train_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx_all)
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
        if stopper.step(val_prauc, model=model, epoch=ep + 1):
            break
    stopper.restore(model)
    train_seconds = time.perf_counter() - t_train_start

    # --- Latency measurement: per-window snapshot load + forward -------------
    #
    # Codex 2026-04-22: `load_snapshot` holds a module-level cache
    # (`gnn_classifiers._SNAPSHOT_CACHE`); without explicit invalidation,
    # first-touch loads get absorbed by warm-up iterations and later timed
    # samples measure near-free cache hits. We therefore
    #   1. warm up GPU kernels separately, on a FIXED stamp outside the
    #      timing set, so warm-up does not pollute the timing cache;
    #   2. sample *unique* stamps for the timing loop;
    #   3. invalidate each timed stamp's cache entry immediately before
    #      it is loaded, so every timed sample pays first-touch cost.
    #
    # This produces the worst-case per-window graph-build cost. Amortised
    # cost for astopo_8h = measured / (96 5-min slots per 8-hour snapshot)
    # and should be reported alongside in §5.10 prose.
    if len(Xt) == 0:
        return _nan_latency_row(source, target, bucket, edge, view, seed,
                                coral_kind, balance_fit, alignment,
                                n_target_windows=0,
                                reason="empty target split (len(Xt)==0)")

    model.eval()
    rng_lat = np.random.default_rng(seed)

    # 1. Warm-up on a fixed stamp OUTSIDE the timing set (stamp index 0 by
    #    construction is the earliest; we don't sample it below). Its cache
    #    entry stays hot afterwards, but that does not affect timed samples.
    unique_tgt_stamps = sorted(set(tgt_stamps))
    warmup_stamp = unique_tgt_stamps[0]
    with torch.no_grad():
        for _ in range(N_WARMUP):
            sg_w = load_snapshot(tgt_topo, warmup_stamp)
            sg_x_w = (sg_w.x.to(device) - mu_t) / sd_t
            sg_ei_w = sg_w.edge_index.to(device)
            if use_edge_weights:
                sg_ea_w = sg_w.edge_attr.to(device)
                h_w = F.elu(model.g1(sg_x_w, sg_ei_w, sg_ea_w))
                h_w = F.elu(model.g2(h_w, sg_ei_w, sg_ea_w))
            else:
                h_w = F.elu(model.g1(sg_x_w, sg_ei_w))
                h_w = F.elu(model.g2(h_w, sg_ei_w))
            tgt_emb_w = (h_w[sg_w.asn_to_idx[tgt_as]]
                         if tgt_as in sg_w.asn_to_idx
                         else torch.zeros(h_w.size(1), device=device))
            tgt_idx_w = tgt_stamps.index(warmup_stamp)
            w_feat_w = torch.from_numpy(Xt[tgt_idx_w:tgt_idx_w + 1]).to(device)
            z_w = torch.cat([tgt_emb_w.unsqueeze(0), w_feat_w], dim=-1)
            _ = model.mlp(z_w).squeeze(-1)
        _sync(device)

    # 2. Build the timing set from the remaining unique stamps.
    timing_stamps = [s for s in unique_tgt_stamps if s != warmup_stamp]
    if not timing_stamps:
        return _nan_latency_row(source, target, bucket, edge, view, seed,
                                coral_kind, balance_fit, alignment,
                                n_target_windows=len(Xt),
                                reason="only one unique target stamp available")
    n_unique_available = len(timing_stamps)
    if n_unique_available < N_SAMPLES:
        print(f"[WARN] only {n_unique_available} unique target stamps "
              f"available for timing (< N_SAMPLES={N_SAMPLES}); sampling "
              f"with replacement, cache will still be invalidated per sample")
        sample_stamps = rng_lat.choice(timing_stamps, size=N_SAMPLES, replace=True)
    else:
        sample_stamps = rng_lat.choice(timing_stamps, size=N_SAMPLES, replace=False)

    # Map each stamp to any index into Xt that uses it (first occurrence).
    stamp_to_idx: dict[str, int] = {}
    for idx, s in enumerate(tgt_stamps):
        stamp_to_idx.setdefault(s, idx)

    ms_graph: list[float] = []
    ms_forward: list[float] = []
    ms_total: list[float] = []

    with torch.no_grad():
        for stamp in sample_stamps:
            # 3. Invalidate the global snapshot cache for this stamp so
            #    load_snapshot pays first-touch cost. Also clear our local
            #    standardized_x cache (key was populated during training on
            #    source stamps; target stamps won't collide, but be safe).
            _gnn_mod._SNAPSHOT_CACHE.pop((tgt_topo, stamp), None)
            standardized_x.pop((tgt_topo, stamp), None)

            i = stamp_to_idx[stamp]
            t0 = time.perf_counter()
            sg = load_snapshot(tgt_topo, stamp)
            sg_x = (sg.x.to(device) - mu_t) / sd_t  # standardize (cache-bypass above)
            sg_ei = sg.edge_index.to(device)
            _sync(device)
            t1 = time.perf_counter()

            if use_edge_weights:
                sg_ea = sg.edge_attr.to(device)
                h = F.elu(model.g1(sg_x, sg_ei, sg_ea))
                h = F.elu(model.g2(h, sg_ei, sg_ea))
            else:
                h = F.elu(model.g1(sg_x, sg_ei))
                h = F.elu(model.g2(h, sg_ei))
            if tgt_as in sg.asn_to_idx:
                tgt_emb = h[sg.asn_to_idx[tgt_as]]
            else:
                tgt_emb = torch.zeros(h.size(1), device=device)
            w_feat = torch.from_numpy(Xt[i:i + 1]).to(device)
            z = torch.cat([tgt_emb.unsqueeze(0), w_feat], dim=-1)
            _ = model.mlp(z).squeeze(-1)
            _sync(device)
            t2 = time.perf_counter()

            ms_graph.append((t1 - t0) * 1000.0)
            ms_forward.append((t2 - t1) * 1000.0)
            ms_total.append((t2 - t0) * 1000.0)

    return {
        "source": source, "target": target, "bucket": bucket, "edge": edge,
        "view": view, "seed": seed, "coral_kind": coral_kind,
        "balance_fit": balance_fit, "alignment": alignment,
        "n_target_windows": len(Xt),
        "train_seconds": round(train_seconds, 3),
        "graph_build_ms_once": float("nan"),  # per-window in topo GNN, not one-off
        "graph_build_ms_median": round(float(np.median(ms_graph)), 4),
        "graph_build_ms_p95": round(float(np.percentile(ms_graph, 95)), 4),
        "forward_ms_median": round(float(np.median(ms_forward)), 4),
        "forward_ms_p95": round(float(np.percentile(ms_forward, 95)), 4),
        "single_window_ms_median": round(float(np.median(ms_total)), 4),
        "single_window_ms_p95": round(float(np.percentile(ms_total, 95)), 4),
        "batched_ms_per_window_median": round(float(np.median(ms_total)), 4),
        "skip_reason": "",
    }


def run_one(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    default_patience = 5 if args.edge in ("temporal", "knn") else 3
    patience = args.patience if args.patience > 0 else default_patience
    common = dict(
        source=args.source_domain, target=args.target_domain,
        bucket=args.bucket, edge=args.edge, view=args.view, seed=args.seed,
        coral_kind=args.coral_kind, balance_fit=args.balance_fit,
        alignment=args.alignment, epochs=args.epochs, patience=patience,
        device=device,
    )
    if args.edge in ("temporal", "knn"):
        return measure_window_gnn(**common)
    elif args.edge in ("astopo_8h", "astopo_8h_weighted"):
        return measure_topo_gnn(**common,
                                use_edge_weights=(args.edge == "astopo_8h_weighted"))
    elif args.edge == "astopo_5m":
        _gnn_mod.load_snapshot = load_snapshot_5min
        _gnn_mod.window_to_stamp = window_to_stamp_5min
        return measure_topo_gnn(**common, use_edge_weights=False)
    else:
        sys.exit(f"edge variant not supported by this script: {args.edge} "
                 f"(supported: temporal, knn, astopo_8h, astopo_8h_weighted, astopo_5m)")


def append_row(row: dict) -> None:
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    header = list(row.keys())
    new = pd.DataFrame([row], columns=header)
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        # Upsert: drop any existing row with the same cell key
        key_cols = ["source", "target", "bucket", "edge", "view", "seed",
                    "coral_kind", "balance_fit", "alignment"]
        key_match = pd.DataFrame([{c: row[c] for c in key_cols}])
        merged = existing.merge(key_match.assign(_drop=True), on=key_cols, how="left")
        existing = merged.loc[merged["_drop"].isna()].drop(columns=["_drop"])
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new
    out.to_csv(OUT_CSV, index=False)
    print(f"[latency written] {OUT_CSV}")


def cmd_one(args: argparse.Namespace) -> None:
    row = run_one(args)
    for k, v in row.items():
        print(f"  {k}: {v}")
    append_row(row)


def cmd_manifest(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    required = {"source_domain", "target_domain", "bucket", "edge"}
    missing = required - set(manifest.columns)
    if missing:
        sys.exit(f"manifest missing required columns: {missing}")
    manifest = manifest.fillna({
        "view": "fusion_early", "seed": 0, "coral_kind": "study_only",
        "balance_fit": "none", "alignment": "after_coral",
    })
    for i, r in manifest.iterrows():
        ns = argparse.Namespace(
            source_domain=r["source_domain"], target_domain=r["target_domain"],
            bucket=r["bucket"], edge=r["edge"], view=r["view"], seed=int(r["seed"]),
            coral_kind=r["coral_kind"], balance_fit=r["balance_fit"],
            alignment=r["alignment"], epochs=args.epochs, patience=args.patience,
            device=args.device,
        )
        print(f"\n=== cell {i+1}/{len(manifest)}: {r['source_domain']} -> "
              f"{r['target_domain']}  edge={r['edge']}  seed={int(r['seed'])} ===")
        row = run_one(ns)
        for k, v in row.items():
            print(f"  {k}: {v}")
        append_row(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("one", help="measure a single cell")
    p1.add_argument("--source-domain", required=True)
    p1.add_argument("--target-domain", required=True)
    p1.add_argument("--bucket", choices=("cross_as", "same_as"), required=True)
    p1.add_argument("--edge", choices=("temporal", "knn", "astopo_8h",
                                        "astopo_8h_weighted", "astopo_5m"), required=True)
    p1.add_argument("--view", default="fusion_early")
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--coral-kind", default="study_only",
                    choices=("study_only", "study_plus_historical"))
    p1.add_argument("--balance-fit", default="none", choices=("none", "equal"))
    p1.add_argument("--alignment", default="after_coral",
                    choices=("before_coral", "after_coral"))
    p1.add_argument("--epochs", type=int, default=200)
    p1.add_argument("--patience", type=int, default=-1,
                    help="-1 = edge-specific default")
    p1.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p1.set_defaults(func=cmd_one)

    p2 = sub.add_parser("manifest", help="run multiple cells from CSV")
    p2.add_argument("--manifest", required=True, type=Path)
    p2.add_argument("--epochs", type=int, default=200)
    p2.add_argument("--patience", type=int, default=-1)
    p2.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p2.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
