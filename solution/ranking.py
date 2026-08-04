from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from kg_io import KnowledgeGraph
from optimisation import clear_device_memory, is_memory_error

TOP_K = 20


# ─────────────────────────────────────────────────────────────────────────────
# Ranked-list generation
# ─────────────────────────────────────────────────────────────────────────────


def compute_ranked_lists(
    graph: KnowledgeGraph,
    device: torch.device,
    ent_emb: torch.Tensor,
    rel_emb: torch.Tensor,
    translation_mode: bool,
    logger: Callable,
    init_batch: int = 16,
    margin: float = 1.0,
) -> dict[str, list[str]]:
    """
    Rank every evaluation query over the full entity vocabulary under the filtered protocol.

    Queries are grouped by **prediction side first** (all tail queries, then all head
    queries), then subdivided by relation within each side. This grouping order differs
    from the original implementation, which grouped by (relation, side) jointly.

    Known-true entities for each query are suppressed (set to -inf) before top-K
    extraction, except for the gold answer being evaluated.

    Parameters
    ----------
    translation_mode : True  → TransE L2 scoring (negative squared L2 distance)
                       False → DistMult bilinear scoring
    init_batch       : starting query batch size (halved adaptively on OOM)
    margin           : unused for DistMult; kept for API compatibility
    """
    # Partition queries: side -> relation -> [query]
    by_side: dict[str, dict[str, list]] = {
        "tail": defaultdict(list),
        "head": defaultdict(list),
    }
    for q in graph.queries:
        side = q["target"]
        by_side.get(side, by_side["tail"])[q["relation"]].append(q)

    out: dict[str, list[str]] = {}
    bsz = init_batch

    for side in ("tail", "head"):
        rel_groups = by_side[side]
        for relation in sorted(rel_groups.keys()):
            queries_for_rel = rel_groups[relation]
            r_vec = rel_emb[graph.rid[relation]]
            i = 0

            while i < len(queries_for_rel):
                window = queries_for_rel[i: i + bsz]
                src_ids = torch.tensor(
                    [graph.eid[q["source_entity"]] for q in window], device=device
                )
                src_vecs = ent_emb[src_ids]

                try:
                    if translation_mode:
                        if side == "tail":
                            # query point = h + r, score = -||query - t||₂²
                            query_pt = (src_vecs + r_vec).unsqueeze(1)   # (B, 1, d)
                            diff = query_pt - ent_emb.unsqueeze(0)       # (B, N, d)
                            scores = -(diff ** 2).sum(-1)                # (B, N)
                        else:
                            # For head prediction: t - r = h, so query = t - r
                            query_pt = (src_vecs - r_vec).unsqueeze(1)   # (B, 1, d)
                            diff = query_pt - ent_emb.unsqueeze(0)       # (B, N, d)
                            scores = -(diff ** 2).sum(-1)                # (B, N)
                    else:
                        # DistMult: (h * r) @ E^T
                        scores = (src_vecs * r_vec) @ ent_emb.t()       # (B, N)
                except RuntimeError as exc:
                    if not is_memory_error(exc) or bsz <= 1:
                        raise
                    clear_device_memory(device)
                    bsz = max(1, bsz // 2)
                    logger(f"ranking OOM — batch now {bsz}")
                    continue

                for k, q in enumerate(window):
                    gold = graph.val_answers.get(q["query_id"])
                    if side == "tail":
                        known = graph.tail_filter[q["source_entity"], relation]
                    else:
                        known = graph.head_filter[relation, q["source_entity"]]
                    suppress = {graph.eid[e] for e in known if e in graph.eid and e != gold}
                    if q["source_entity"] in graph.eid and q["source_entity"] != gold:
                        suppress.add(graph.eid[q["source_entity"]])

                    row = scores[k].clone()
                    if suppress:
                        suppress_t = torch.tensor(
                            sorted(suppress), dtype=torch.long, device=device
                        )
                        row[suppress_t] = -torch.inf

                    top_indices = torch.topk(row, TOP_K).indices.tolist()
                    out[q["query_id"]] = [graph.entities[j] for j in top_indices]

                i += bsz

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────


def eval_metrics(
    ranked: dict[str, list[str]],
    query_ids: list[str],
    answers: dict[str, str],
) -> dict:
    """
    Compute filtered MRR and Hits@1/3/10 for a collection of query IDs.

    Answers ranked beyond TOP_K are treated as not found and contribute 0.
    """
    total_rr = 0.0
    hits: dict[int, int] = {1: 0, 3: 0, 10: 0}
    n = len(query_ids)

    for qid in query_ids:
        gold = answers[qid]
        found_rank = 0
        for rank, candidate in enumerate(ranked.get(qid, [])[:TOP_K], start=1):
            if candidate == gold:
                found_rank = rank
                break
        if found_rank:
            total_rr += 1.0 / found_rank
            for k in hits:
                if found_rank <= k:
                    hits[k] += 1

    safe_n = max(n, 1)
    return {
        "mrr":     total_rr   / safe_n,
        "hits@1":  hits[1]    / safe_n,
        "hits@3":  hits[3]    / safe_n,
        "hits@10": hits[10]   / safe_n,
        "n":       n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spearman rank correlation  (Pearson-on-ranks approach)
# ─────────────────────────────────────────────────────────────────────────────


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """
    Spearman rank-order correlation via the Pearson-on-ranks formula.

    Converts each input sequence to ranks, then computes the Pearson
    product-moment correlation between those rank sequences. This is
    mathematically equivalent to the d² formula for unique values but
    handles tied ranks correctly via the mean-rank convention.

    Returns 0.0 for fewer than 2 observations or zero-variance inputs.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    def to_ranks(vals: list[float]) -> np.ndarray:
        arr = np.array(vals, dtype=np.float64)
        order = np.argsort(arr)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1.0, n + 1.0)
        return ranks

    rx = to_ranks(xs)
    ry = to_ranks(ys)

    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────


def export_rankings(path: Path, rows: list[tuple]) -> None:
    """Write query_id / ranked_entities pairs to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["query_id", "ranked_entities"])
        writer.writerows(rows)
