"""CAIDA AS-Relationships (serial-2) loader.

The file is pipe-separated with rows `source|target|relationship|source_info`
and relationship codes:

  -1  provider-to-customer (source is provider, target is customer)
   0  peer-to-peer
  +1  customer-to-provider  (not emitted in serial-2; derived by reversing -1)

For HGT we classify an ego-graph edge (a, b) into one of four types:

  p2p   there is a CAIDA p2p entry between {a, b}            (symmetric)
  p2c   CAIDA has a -> b as provider->customer              (directed a->b)
  c2p   CAIDA has b -> a as provider->customer              (directed a->b)
  unknown  neither endpoint ordering is in CAIDA            (fallback; rare)

`classify_edges` returns four `edge_index` arrays sharing the same node
indexing, ready to be dropped into a PyG `HeteroData` with one node type.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_CAIDA_FNAME_RE = __import__("re").compile(r"^(\d{8})\.as-rel2\.txt$")


def _list_caida_files(runs_root: Path) -> list[tuple[str, Path]]:
    """Return all (YYYYMMDD, path) CAIDA snapshots found under runs/."""
    hits = list(runs_root.glob("*/workspace/reference_data/caida/*.as-rel2.txt"))
    dated: dict[str, Path] = {}
    for p in hits:
        m = _CAIDA_FNAME_RE.match(p.name)
        if not m:
            continue
        ymd = m.group(1)
        # Multiple runs may ship the same dated snapshot; keep one (they match).
        dated.setdefault(ymd, p)
    if not dated:
        raise FileNotFoundError(
            f"no YYYYMMDD.as-rel2.txt under {runs_root}/*/workspace/reference_data/caida/"
        )
    return sorted(dated.items())


def _pick_caida_file(runs_root: Path, target_ymd: str | None) -> Path:
    """Pick the CAIDA snapshot nearest to `target_ymd` (YYYYMMDD).

    If `target_ymd` is None, pick the single available file (and error if
    there are several — caller then owes a date). CAIDA AS-relationships
    drift slowly, so nearest-by-calendar-day is the right rule.
    """
    files = _list_caida_files(runs_root)
    if target_ymd is None:
        if len(files) == 1:
            return files[0][1]
        raise ValueError(
            f"{len(files)} CAIDA files available; caller must supply target_ymd "
            f"to disambiguate (got {[y for y, _ in files]})."
        )
    tgt = pd.to_datetime(target_ymd, format="%Y%m%d")
    return min(files, key=lambda kv: abs(pd.to_datetime(kv[0], format="%Y%m%d") - tgt))[1]


_CACHE: dict[Path, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {}


def load_caida(runs_root: Path,
               target_ymd: str | None = None
               ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Returns (p2c_pairs, p2p_pairs) as sets of (src_asn, tgt_asn).

    `target_ymd` = YYYYMMDD string; picks the CAIDA snapshot nearest to that
    calendar date. Pass the first day of the study window.

    For p2c the orientation is src=provider, tgt=customer.
    For p2p the set stores only (min, max) so membership checks are order-free.
    """
    path = _pick_caida_file(runs_root, target_ymd)
    if path in _CACHE:
        return _CACHE[path]
    df = pd.read_csv(
        path, sep="|", comment="#", header=None,
        names=["source", "target", "rel", "info"],
        dtype={"source": np.int64, "target": np.int64, "rel": np.int8},
        usecols=["source", "target", "rel"],
    )
    known = df["rel"].isin([-1, 0])
    if not known.all():
        # Serial-2 only emits {-1, 0}; surface any surprise for audit.
        surprises = sorted(set(df.loc[~known, "rel"].tolist()))
        print(f"  [caida WARN] unexpected rel codes in {path.name}: {surprises}")
        df = df[known]
    p2c_rows = df[df["rel"] == -1]
    p2p_rows = df[df["rel"] == 0]
    p2c = set(zip(p2c_rows["source"].tolist(), p2c_rows["target"].tolist()))
    p2p = set()
    for s, t in zip(p2p_rows["source"].tolist(), p2p_rows["target"].tolist()):
        a, b = (int(s), int(t)) if s < t else (int(t), int(s))
        p2p.add((a, b))
    _CACHE[path] = (p2c, p2p)
    return p2c, p2p


def classify_edges(src_idx: np.ndarray, dst_idx: np.ndarray,
                   idx_to_asn: np.ndarray,
                   p2c: set[tuple[int, int]],
                   p2p: set[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Route each undirected edge into one of {p2p, p2c, c2p, unknown}.

    `src_idx`/`dst_idx` are *node indices* into the snapshot's ASN vector
    `idx_to_asn`.  The function emits the relational convention required by
    HGTConv: for p2p both directions in `edge_index`; for p2c the single
    direction provider->customer (emitted as a->b if CAIDA has a as provider);
    for c2p the single direction customer->provider.  `unknown` gets both
    directions so a fallback homogeneous message-passing layer can still learn.

    Returns a dict[str, (2, E_type) int64 array].
    """
    p2p_s, p2p_d = [], []
    p2c_s, p2c_d = [], []
    c2p_s, c2p_d = [], []
    unk_s, unk_d = [], []
    for i, j in zip(src_idx, dst_idx):
        a = int(idx_to_asn[i]); b = int(idx_to_asn[j])
        ab = (a, b); ba = (b, a)
        p2p_key = (min(a, b), max(a, b))
        if p2p_key in p2p:
            p2p_s.append(i); p2p_d.append(j)
            p2p_s.append(j); p2p_d.append(i)
        elif ab in p2c:
            # a is provider, b is customer; CAIDA emits a->b as p2c.
            # From b's side the edge is c2p (b->a as c2p).
            p2c_s.append(i); p2c_d.append(j)
            c2p_s.append(j); c2p_d.append(i)
        elif ba in p2c:
            p2c_s.append(j); p2c_d.append(i)
            c2p_s.append(i); c2p_d.append(j)
        else:
            unk_s.append(i); unk_d.append(j)
            unk_s.append(j); unk_d.append(i)
    def _arr(s, d):
        if not s:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array([s, d], dtype=np.int64)
    return {
        "p2p": _arr(p2p_s, p2p_d),
        "p2c": _arr(p2c_s, p2c_d),
        "c2p": _arr(c2p_s, c2p_d),
        "unknown": _arr(unk_s, unk_d),
    }