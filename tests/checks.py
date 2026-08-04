from __future__ import annotations

import csv
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
TESTS_DIR = Path(__file__).resolve().parent
GRAPH_DIR = TESTS_DIR / "verifier_data"

MODELS = ["transe", "rgcn"]
SCORERS = {"transe": ["transe_l1", "transe_l2"], "rgcn": ["distmult", "complex"]}
SCORER_FAMILY = {"transe": "translation", "rgcn": "bilinear"}
TOP_K = 20
SAMPLE_QUERIES = 60
RANK_AGREEMENT_MIN = 0.5
MODELS_DIFFER_MIN = 0.10
MRR_CEILING = 0.55
HITS10_CEILING = 0.85
MIN_RELATION_EVAL = 30
ENCODER_MAX_LAYERS = 4
ENCODER_COSINE_MIN = 0.98
ENCODER_RELERR_MAX = 0.10
TRANSFORM_DISTINCT_MIN = 0.5
MESSAGE_CONTRIBUTION_MIN = 0.01
ENCODER_VARIANTS = [("in_degree", "relu"), ("in_degree", "identity"),
                    ("none", "relu"), ("none", "identity")]
SPLIT_ADVANTAGE_MAX = 1.10
SPLIT_ADVANTAGE_MARGIN = 0.02

_cache: dict[str, Any] = {}


def cached(key: str, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def hidden() -> dict:
    return cached("hidden",
                  lambda: json.loads((TESTS_DIR / "hidden_labels.json").read_text(encoding="utf-8")))


def entity_rows() -> list[dict]:
    return cached("entity_rows", lambda: read_tsv(GRAPH_DIR / "kg" / "entities.tsv"))


def entities() -> list[str]:
    return cached("entities", lambda: [r["entity_id"] for r in entity_rows()])


def entity_types() -> dict[str, str]:
    return cached("etypes", lambda: {r["entity_id"]: r["type"] for r in entity_rows()})


def relations() -> list[str]:
    return cached("relations",
                  lambda: [r["relation"] for r in read_tsv(GRAPH_DIR / "kg" / "relations.tsv")])


def triples(name: str) -> list[tuple[str, str, str]]:
    return cached(f"triples_{name}", lambda: [
        (r["head"], r["relation"], r["tail"])
        for r in read_tsv(GRAPH_DIR / "kg" / f"{name}.tsv")])


def queries() -> dict[str, dict]:
    def build():
        return {r["query_id"]: r for r in read_tsv(GRAPH_DIR / "lp" / "eval_queries.tsv")}
    return cached("queries", build)


def known_answers(scope: str = "full") -> tuple[dict, dict]:
    def build_agent():
        tail, head = defaultdict(set), defaultdict(set)
        for h, r, t in triples("train") + triples("valid"):
            tail[h, r].add(t)
            head[r, t].add(h)
        return tail, head

    def build_full():
        agent_tail, agent_head = known_answers("agent")
        tail = defaultdict(set, {k: set(v) for k, v in agent_tail.items()})
        head = defaultdict(set, {k: set(v) for k, v in agent_head.items()})
        for qid, gold in hidden()["lp"]["answers"].items():
            q = queries().get(qid)
            if q is None:
                continue
            if q["target"] == "tail":
                tail[q["source_entity"], q["relation"]].add(gold)
            else:
                head[q["relation"], q["source_entity"]].add(gold)
        return tail, head

    if scope == "agent":
        return cached("known_agent", build_agent)
    return cached("known_full", build_full)


def split_ids(kind: str, split: str) -> list[str]:
    def build():
        table = hidden()[kind]["split"]
        return sorted(k for k, v in table.items() if v == split)
    return cached(f"split_{kind}_{split}", build)


def app_path(*parts: str) -> Path:
    return APP_DIR.joinpath(*parts)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise AssertionError(f"Missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON in {path}: {exc.msg} (line {exc.lineno})") from exc


def read_ranking(model: str) -> dict[str, list[str]]:
    def build():
        path = app_path("artifacts", "predictions", f"lp_{model}.csv")
        assert path.exists(), f"Missing predictions file: {path}"
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, f"{path} is empty"
        cols = set(rows[0])
        assert {"query_id", "ranked_entities"} <= cols, (
            f"{path} must have header query_id,ranked_entities; got {sorted(cols)}")
        out: dict[str, list[str]] = {}
        for r in rows:
            qid = str(r["query_id"]).strip()
            assert qid not in out, f"{path}: duplicate query_id {qid}"
            raw = str(r["ranked_entities"] or "")
            out[qid] = [c.strip() for c in raw.split("|") if c.strip()][:TOP_K]
        return out
    return cached(f"ranking_{model}", build)


def safe_ranking(model: str) -> dict[str, list[str]]:
    try:
        return read_ranking(model)
    except Exception:  # noqa: BLE001
        return {}


def filtered_rank(qid: str, ranked: list[str], gold: str, scope: str = "full") -> int:
    tail, head = known_answers(scope)
    q = queries()[qid]
    known = (tail[q["source_entity"], q["relation"]] if q["target"] == "tail"
             else head[q["relation"], q["source_entity"]])
    position = 0
    for cand in ranked[:TOP_K]:
        if cand == gold:
            return position + 1
        if cand in known or cand == q["source_entity"]:
            continue
        position += 1
    return 0


def lp_metrics(model: str, qids: list[str], scope: str = "full") -> dict[str, float]:
    ranking = safe_ranking(model)
    answers = hidden()["lp"]["answers"]
    rr = h1 = h3 = h10 = 0.0
    for qid in qids:
        gold = answers.get(qid)
        if gold is None:
            continue
        rank = filtered_rank(qid, ranking.get(qid, []), gold, scope)
        if rank:
            rr += 1.0 / rank
            h1 += rank <= 1
            h3 += rank <= 3
            h10 += rank <= 10
    n = max(len(qids), 1)
    return {"mrr": rr / n, "hits@1": h1 / n, "hits@3": h3 / n, "hits@10": h10 / n,
            "n": len(qids)}


def relation_query_ids(split: str) -> dict[str, list[str]]:
    def build():
        out = defaultdict(list)
        table = hidden()["lp"]["relation"]
        for qid in split_ids("lp", split):
            out[table[qid]].append(qid)
        return dict(out)
    return cached(f"relation_ids_{split}", build)


def analysis_relations() -> list[str]:
    def build():
        return sorted(r for r, ids in relation_query_ids("val").items()
                      if len(ids) >= MIN_RELATION_EVAL)
    return cached("analysis_relations", build)


def load_embeddings(model: str):
    ent = np.load(app_path("artifacts", "embeddings", f"{model}_entity.npy"))
    rel = np.load(app_path("artifacts", "embeddings", f"{model}_relation.npy"))
    ent, rel = ent.astype(np.float64), rel.astype(np.float64)
    scale = max(float(np.abs(ent).max(initial=0.0)), float(np.abs(rel).max(initial=0.0)))
    if np.isfinite(scale) and scale > 0:
        ent, rel = ent / scale, rel / scale
    return ent, rel


def score_all(kind: str, src: np.ndarray, rel: np.ndarray, ent: np.ndarray,
              target: str) -> np.ndarray:
    if kind in ("transe_l1", "transe_l2"):
        shifted = src + rel if target == "tail" else src - rel
        if kind == "transe_l2":
            return -(np.square(shifted).sum(1)[:, None] - 2 * shifted @ ent.T
                     + np.square(ent).sum(1)[None, :])
        out = np.empty((shifted.shape[0], ent.shape[0]), dtype=np.float64)
        for a in range(0, shifted.shape[0], 4):
            block_q = shifted[a:a + 4]
            for b in range(0, ent.shape[0], 4096):
                block_e = ent[b:b + 4096]
                out[a:a + 4, b:b + 4096] = -np.abs(
                    block_q[:, None, :] - block_e[None, :, :]).sum(-1)
        return out
    if kind == "distmult":
        return (src * rel) @ ent.T
    half = ent.shape[1] // 2
    sr, si = src[:, :half], src[:, half:half * 2]
    rr, ri = rel[:half], rel[half:half * 2]
    er, ei = ent[:, :half], ent[:, half:half * 2]
    if target == "tail":
        qr, qi = sr * rr - si * ri, sr * ri + si * rr
    else:
        qr, qi = sr * rr + si * ri, sr * ri - si * rr
    return qr @ er.T + qi @ ei.T


def rank_agreement(model: str) -> float:
    _cache[f"scorer_agreements:{model}"] = {}
    ent, rel = load_embeddings(model)
    ids = entities()
    index = {e: i for i, e in enumerate(ids)}
    rel_index = {r: i for i, r in enumerate(relations())}
    ranking = safe_ranking(model)
    # Sample both splits. Covering only the graded half would let a submission write the
    # shipped validation answers straight into its ranking, inflating the validation score
    # that check_results_plausible bounds the held-out score against.
    pool: list[str] = []
    for split in ("hidden", "val"):
        split_qids = split_ids("lp", split)
        take = max(1, SAMPLE_QUERIES // 2)
        step = max(1, len(split_qids) // take)
        pool.extend(split_qids[::step][:take])
    sample = [q for q in pool if ranking.get(q) and ranking[q][0] in index]
    if len(sample) < 10:
        return 0.0
    if ent.shape[0] != len(ids) or rel.shape[0] != len(relations()):
        return 0.0
    if ent.shape[1] != rel.shape[1]:
        return 0.0
    tail, head = known_answers("agent")
    answers = hidden()["lp"]["answers"]
    groups = defaultdict(list)
    for qid in sample:
        q = queries()[qid]
        groups[q["relation"], q["target"]].append(qid)
    kinds = [k for k in SCORERS[model]
             if k != "complex" or ent.shape[1] % 2 == 0]
    best = 0.0
    for kind in kinds:
        hits: dict[str, int] = defaultdict(int)
        total = 0
        for (relation, target), qids in sorted(groups.items()):
            src_rows = np.array([index[queries()[q]["source_entity"]] for q in qids])
            with np.errstate(all="ignore"):
                scores = score_all(kind, ent[src_rows], rel[rel_index[relation]], ent, target)
            for row, qid in enumerate(qids):
                q = queries()[qid]
                gold = answers.get(qid)
                known = (tail[q["source_entity"], q["relation"]] if target == "tail"
                         else head[q["relation"], q["source_entity"]])
                blocked = np.array([index[e] for e in known
                                    if e in index and e != gold], dtype=int)
                raw = scores[row]
                filtered = raw.copy()
                if blocked.size:
                    filtered[blocked] = -np.inf
                top = index[ranking[qid][0]]
                for name, values in (("raw", raw), ("filtered", filtered)):
                    if np.isfinite(values[top]) and (values > values[top]).sum() < 10:
                        hits[name] += 1
                total += 1
        if total:
            score = max([count / total for count in hits.values()] or [0.0])
            _cache[f"scorer_agreements:{model}"][kind] = score
            best = max(best, score)
    return best


def scorer_agreements(model: str) -> dict[str, float]:
    """Per scoring rule, how well it reproduces the submitted ranking."""
    key = f"scorer_agreements:{model}"
    if key not in _cache:
        rank_agreement(model)
    return _cache.get(key, {})


def encoder_layers() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load the saved R-GCN encoder parameters, one entry per layer."""
    out = []
    for layer in range(ENCODER_MAX_LAYERS):
        paths = {kind: app_path("artifacts", "encoder", f"{kind}_layer{layer}.npy")
                 for kind in ("bases", "coeff", "self")}
        if not any(p.exists() for p in paths.values()):
            break
        missing = [k for k, p in paths.items() if not p.exists()]
        assert not missing, (
            f"encoder layer {layer} is missing {', '.join(sorted(missing))}_layer{layer}.npy "
            f"— save the basis tensor, the per-relation basis coefficients and the "
            f"self-loop transform for every encoder layer")
        try:
            bases = np.load(paths["bases"]).astype(np.float64)
            coeff = np.load(paths["coeff"]).astype(np.float64)
            self_w = np.load(paths["self"]).astype(np.float64)
        except Exception as exc:
            raise AssertionError(f"encoder layer {layer} tensors unreadable ({exc})") from exc
        n_rowtypes = 2 * len(relations())
        assert bases.ndim == 3, (
            f"bases_layer{layer}.npy has shape {bases.shape}, expected "
            f"(n_bases, dim_in, dim_out)")
        assert coeff.shape == (n_rowtypes, bases.shape[0]), (
            f"coeff_layer{layer}.npy has shape {coeff.shape}, expected "
            f"({n_rowtypes}, {bases.shape[0]}) — one basis coefficient row per relation "
            f"per direction, forward relations first then inverse")
        assert self_w.shape == (bases.shape[1], bases.shape[2]), (
            f"self_layer{layer}.npy has shape {self_w.shape}, expected "
            f"({bases.shape[1]}, {bases.shape[2]})")
        for name, arr in (("bases", bases), ("coeff", coeff), ("self", self_w)):
            assert np.isfinite(arr).all(), f"{name}_layer{layer}.npy has non-finite values"
        out.append((bases, coeff, self_w))
    assert out, (
        "no R-GCN encoder parameters found under /app/artifacts/encoder — the rgcn slot "
        "must be a relation-aware encoder, and its per-layer basis tensor, per-relation "
        "basis coefficients and self-loop transform all have to be saved")
    return out


def edge_buckets() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per relation-direction: (src_sorted, norm_sorted, unique_dst, reduceat_starts)."""
    def build():
        ids = entities()
        index = {e: i for i, e in enumerate(ids)}
        rel_index = {r: i for i, r in enumerate(relations())}
        n_rel = len(rel_index)
        raw: list[tuple[list[int], list[int]]] = [([], []) for _ in range(2 * n_rel)]
        for row in read_tsv(GRAPH_DIR / "kg" / "train.tsv"):
            hi, ti = index.get(row["head"]), index.get(row["tail"])
            ri = rel_index.get(row["relation"])
            if hi is None or ti is None or ri is None:
                continue
            raw[ri][0].append(hi)
            raw[ri][1].append(ti)
            raw[ri + n_rel][0].append(ti)
            raw[ri + n_rel][1].append(hi)
        out = []
        for src, dst in raw:
            if not src:
                empty_i = np.empty(0, dtype=np.int64)
                out.append((empty_i, np.empty(0), empty_i, empty_i))
                continue
            src_a = np.asarray(src, dtype=np.int64)
            dst_a = np.asarray(dst, dtype=np.int64)
            counts = np.bincount(dst_a, minlength=len(ids)).astype(np.float64)
            norm = 1.0 / counts[dst_a]
            order = np.argsort(dst_a, kind="stable")
            dst_sorted = dst_a[order]
            uniq, starts = np.unique(dst_sorted, return_index=True)
            out.append((src_a[order], norm[order], uniq, starts))
        return out
    return cached("edge_buckets", build)


def rgcn_replay(ablate_messages: bool = False, normalisation: str = "in_degree",
                activation: str = "relu") -> np.ndarray:
    """Recompute the encoder output from the saved input embeddings and transforms.

    The task does not dictate one encoder, so the caller sweeps the ordinary choices
    (mean or unnormalised messages, ReLU or no activation) and keeps whichever
    reproduces the submitted embeddings.

    With ablate_messages the neighbour aggregation is dropped and only the self-loop
    path survives, which is what a decoder-only submission collapses to.
    """
    layers = encoder_layers()
    path = app_path("artifacts", "embeddings", "rgcn_input.npy")
    assert path.exists(), (
        "rgcn_input.npy is missing — save the encoder's input entity embeddings in "
        "entities.tsv row order so the encoder output can be reproduced")
    h = np.load(path).astype(np.float64)
    n_ent = len(entities())
    assert h.ndim == 2 and h.shape[0] == n_ent, (
        f"rgcn_input.npy has shape {h.shape}, expected ({n_ent}, dim)")
    assert np.isfinite(h).all(), "rgcn_input.npy has non-finite values"
    buckets = edge_buckets()
    for layer, (bases, coeff, self_w) in enumerate(layers):
        assert h.shape[1] == bases.shape[1], (
            f"encoder layer {layer} expects input dim {bases.shape[1]} but the incoming "
            f"representation has dim {h.shape[1]}")
        with np.errstate(all="ignore"):
            agg = np.zeros((n_ent, bases.shape[2]), dtype=np.float64)
            if not ablate_messages:
                projected = np.stack([h @ bases[b] for b in range(bases.shape[0])])
                for rt, (src, norm, uniq, starts) in enumerate(buckets):
                    if src.size == 0:
                        continue
                    transformed = np.tensordot(coeff[rt], projected, axes=(0, 0))
                    vals = transformed[src]
                    if normalisation == "in_degree":
                        vals = vals * norm[:, None]
                    np.add.at(agg, uniq, np.add.reduceat(vals, starts, axis=0))
            h = agg + h @ self_w
            if activation == "relu":
                h = np.maximum(h, 0.0)
    return h


def transform_distinctness(coeff: np.ndarray) -> float:
    """Fraction of relation-direction rows whose basis coefficients are distinct."""
    rounded = np.round(coeff, 6)
    distinct = {row.tobytes() for row in rounded}
    return len(distinct) / max(coeff.shape[0], 1)


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals):
        order = np.argsort(vals)
        out = np.empty(len(vals))
        out[order] = np.arange(len(vals))
        return out
    rx = ranks(xs) - (len(xs) - 1) / 2
    ry = ranks(ys) - (len(ys) - 1) / 2
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def check_kg_stats_valid() -> str:
    stats = load_json(app_path("artifacts", "kg_stats.json"))
    problems = []
    if stats.get("n_entities") != len(entities()):
        problems.append(f"n_entities {stats.get('n_entities')} != {len(entities())}")
    if stats.get("n_relations") != len(relations()):
        problems.append(f"n_relations {stats.get('n_relations')} != {len(relations())}")
    train = triples("train")
    if stats.get("n_train_triples") != len(train):
        problems.append(f"n_train_triples {stats.get('n_train_triples')} != {len(train)}")
    counts = defaultdict(int)
    for _, r, _ in train:
        counts[r] += 1
    reported = stats.get("relation_counts", {})
    for rel in relations():
        if reported.get(rel) != counts[rel]:
            problems.append(f"relation_counts[{rel}]={reported.get(rel)} != {counts[rel]}")
    type_counts = defaultdict(int)
    for t in entity_types().values():
        type_counts[t] += 1
    reported_types = stats.get("entity_type_counts", {})
    for t, n in sorted(type_counts.items()):
        if reported_types.get(t) != n:
            problems.append(f"entity_type_counts[{t}]={reported_types.get(t)} != {n}")
    assert not problems, "; ".join(problems[:8])
    return (f"kg_stats matches the shipped graph ({len(entities())} entities, "
            f"{len(train)} train triples)")


def check_embeddings_valid() -> str:
    problems = []
    for model in MODELS:
        for kind, expected in (("entity", len(entities())), ("relation", len(relations()))):
            path = app_path("artifacts", "embeddings", f"{model}_{kind}.npy")
            if not path.exists():
                problems.append(f"{model}_{kind}.npy missing")
                continue
            try:
                arr = np.load(path)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{model}_{kind}.npy unreadable ({exc})")
                continue
            if arr.ndim != 2 or arr.shape[0] != expected:
                problems.append(f"{model}_{kind}.npy shape {arr.shape}, expected "
                                f"({expected}, d)")
                continue
            if arr.shape[1] < 32:
                problems.append(f"{model}_{kind}.npy dim {arr.shape[1]} < 32")
            if not np.isfinite(arr).all():
                problems.append(f"{model}_{kind}.npy has non-finite values")
            if float(arr.std()) <= 1e-6:
                problems.append(f"{model}_{kind}.npy is constant")
            if kind == "entity":
                sampled = {arr[i].tobytes() for i in range(0, arr.shape[0],
                                                           max(1, arr.shape[0] // 200))}
                if len(sampled) < 100:
                    problems.append(f"{model}_entity.npy has only {len(sampled)} distinct "
                                    f"rows in a 200-row sample")
        try:
            ent, rel = load_embeddings(model)
            if ent.shape[1] != rel.shape[1]:
                problems.append(f"{model}: entity dim {ent.shape[1]} != relation dim "
                                f"{rel.shape[1]}")
        except Exception:  # noqa: BLE001, S110
            pass
    assert not problems, "; ".join(problems[:8])
    return "entity and relation embeddings present and well formed for both models"


def check_lp_predictions_complete() -> str:
    expected = set(queries())
    valid_entities = set(entities())
    problems = []
    for model in MODELS:
        try:
            ranking = read_ranking(model)
        except AssertionError as exc:
            problems.append(str(exc))
            continue
        got = set(ranking)
        if got != expected:
            problems.append(f"lp_{model}: covers {len(got & expected)}/{len(expected)} "
                            f"queries" + (f", {len(got - expected)} unknown ids"
                                          if got - expected else ""))
        empty = sum(1 for v in ranking.values() if not v)
        if empty:
            problems.append(f"lp_{model}: {empty} queries have an empty candidate list")
        bad = sum(1 for v in ranking.values()
                  if any(c not in valid_entities for c in v))
        if bad > 0.01 * max(len(ranking), 1):
            problems.append(f"lp_{model}: {bad} rows contain unknown entity ids")
        dupes = sum(1 for v in ranking.values() if len(set(v)) != len(v))
        if dupes > 0.01 * max(len(ranking), 1):
            problems.append(f"lp_{model}: {dupes} rows repeat a candidate")
        short = sum(1 for v in ranking.values() if len(v) < 5)
        if short > 0.20 * max(len(ranking), 1):
            problems.append(f"lp_{model}: {short} rows carry fewer than 5 candidates")
    assert not problems, "; ".join(problems[:8])
    return f"both ranking files cover all {len(expected)} eval queries"


def check_lp_rankings_match_embeddings() -> str:
    details = []
    for model in MODELS:
        try:
            agreement = rank_agreement(model)
        except Exception as exc:
            raise AssertionError(f"{model}: could not score the saved embeddings ({exc})") from exc
        details.append(f"{model}={agreement:.2f}")
        assert agreement >= RANK_AGREEMENT_MIN, (
            f"{model}: the submitted top-ranked entity sits outside the top 10 of the "
            f"ranking implied by the saved {model} embeddings under a "
            f"{SCORER_FAMILY[model]} scoring rule on {(1 - agreement) * 100:.0f}% of "
            f"sampled queries (agreement {agreement:.2f} < {RANK_AGREEMENT_MIN}) — "
            f"predictions must come from the model whose parameters were saved")
    return f"rankings reproduce from the saved embeddings ({', '.join(details)})"


def check_rgcn_encoder_reproduces() -> str:
    layers = encoder_layers()
    saved = np.load(app_path("artifacts", "embeddings", "rgcn_entity.npy")).astype(np.float64)
    scale = float(np.linalg.norm(saved))

    # The instruction asks for an R-GCN, not for one particular R-GCN, so sweep the
    # ordinary variants and keep whichever reproduces what was saved. Prescribing a
    # single propagation rule would dictate the architecture rather than the result.
    best = (-1.0, float("inf"), ENCODER_VARIANTS[0], None)
    for variant in ENCODER_VARIANTS:
        norm_mode, act = variant
        replayed = rgcn_replay(normalisation=norm_mode, activation=act)
        if replayed.shape != saved.shape:
            raise AssertionError(
                f"replaying the saved encoder gives shape {replayed.shape} but "
                f"rgcn_entity.npy has shape {saved.shape} — the saved transforms must be "
                f"the ones that produced the saved embeddings")
        err = float(np.linalg.norm(replayed - saved) / scale) if scale > 0 else float("inf")
        num = (replayed * saved).sum(1)
        den = np.linalg.norm(replayed, axis=1) * np.linalg.norm(saved, axis=1)
        live = den > 0
        cos = float(np.mean(num[live] / den[live])) if live.any() else 0.0
        if cos > best[0]:
            best = (cos, err, variant, replayed)
    cosine, rel_err, (norm_mode, act), replayed = best

    assert cosine >= ENCODER_COSINE_MIN and rel_err <= ENCODER_RELERR_MAX, (
        f"replaying the saved R-GCN encoder over the training graph does not reproduce "
        f"rgcn_entity.npy under any of the {len(ENCODER_VARIANTS)} propagation variants "
        f"tried (best: mean row cosine {cosine:.4f}, relative error {rel_err:.4f}; needs "
        f"{ENCODER_COSINE_MIN} and {ENCODER_RELERR_MAX}) — the rgcn slot must be a real "
        f"relation-aware encoder whose saved input embeddings and per-relation transforms "
        f"regenerate its output embeddings by message passing, not a decoder-only model")

    # Reproduction alone is spoofable: set the input embeddings equal to the output,
    # the self-loop to the identity and the bases to ~0, and relu(0 + X @ I) == X for a
    # post-ReLU output. So ablate the neighbour aggregation and require the replay to
    # break — message passing has to be doing real work, not decorating a decoder.
    ablated = rgcn_replay(ablate_messages=True, normalisation=norm_mode, activation=act)
    span = float(np.linalg.norm(replayed))
    contribution = (float(np.linalg.norm(replayed - ablated) / span) if span > 0
                    else 0.0)
    assert contribution >= MESSAGE_CONTRIBUTION_MIN, (
        f"dropping neighbour aggregation changes the encoder output by only "
        f"{contribution:.2e} of its norm (minimum {MESSAGE_CONTRIBUTION_MIN}) — the saved "
        f"encoder reproduces its embeddings through the self-loop path alone, so the rgcn "
        f"slot is a decoder with an inert encoder bolted on rather than a model whose "
        f"representations come from the graph")

    problems = []
    for layer, (bases, coeff, _self_w) in enumerate(layers):
        share = transform_distinctness(coeff)
        if share < TRANSFORM_DISTINCT_MIN:
            problems.append(f"layer {layer} reuses the same basis coefficients across "
                            f"relations ({share:.2f} of rows distinct)")
        if float(np.abs(bases).max(initial=0.0)) <= 1e-8:
            problems.append(f"layer {layer} basis tensor is all zeros")
    assert not problems, (
        "; ".join(problems) + " — per-relation transforms must actually differ by relation")
    return (f"the saved encoder replays to rgcn_entity.npy over {len(layers)} layer(s) "
            f"under {norm_mode}/{act} propagation (cosine {cosine:.4f}, relative error "
            f"{rel_err:.4f}) and neighbour aggregation carries {contribution:.2f} of the "
            f"output norm")


def check_models_differ() -> str:
    a, b = safe_ranking("transe"), safe_ranking("rgcn")
    shared = [q for q in queries() if a.get(q) and b.get(q)]
    assert shared, "no query is covered by both ranking files"
    diff = sum(1 for q in shared if a[q][0] != b[q][0]) / len(shared)
    assert diff >= MODELS_DIFFER_MIN, (
        f"transe and rgcn pick the same top entity on {(1 - diff) * 100:.1f}% of queries "
        f"(disagreement {diff:.3f} < {MODELS_DIFFER_MIN}) — the two models must be trained "
        f"independently, not copied")
    return f"transe and rgcn rankings disagree on {diff:.3f} of queries"


def check_lp_above_popularity() -> str:
    baseline = hidden()["lp"]["popularity_baseline_mrr"]["hidden"]
    details = []
    for model in MODELS:
        mrr = lp_metrics(model, split_ids("lp", "hidden"))["mrr"]
        details.append(f"{model}={mrr:.4f}")
        assert mrr > baseline + 0.02, (
            f"{model} hidden MRR {mrr:.4f} does not clear the popularity baseline "
            f"{baseline:.4f} by a margin — the model is not learning the graph")
    return f"both models beat the popularity baseline {baseline:.4f} ({', '.join(details)})"


def check_results_plausible() -> str:
    details = []
    hidden_ids = split_ids("lp", "hidden")
    val_ids = split_ids("lp", "val")
    for model in MODELS:
        stats = lp_metrics(model, hidden_ids)
        mrr, hits = stats["mrr"], stats["hits@10"]
        val_mrr = lp_metrics(model, val_ids)["mrr"]
        details.append(f"{model}={mrr:.4f} (val {val_mrr:.4f})")
        assert mrr <= MRR_CEILING, (
            f"{model} hidden MRR {mrr:.4f} exceeds {MRR_CEILING}, which is far above what "
            f"this benchmark supports — the held-out triples were part of training or were "
            f"looked up rather than predicted")
        assert hits <= HITS10_CEILING, (
            f"{model} hidden Hits@10 {hits:.4f} exceeds {HITS10_CEILING} — the held-out "
            f"answers were looked up rather than predicted")
        assert val_mrr <= MRR_CEILING, (
            f"{model} validation MRR {val_mrr:.4f} exceeds {MRR_CEILING} — the validation "
            f"answers are shipped, so a score that high means they were written into the "
            f"ranking rather than predicted")
        # A fixed ceiling only stops a naive leaker. Both splits are held out of training
        # and drawn the same way, so an honest model scores about the same on each; a
        # held-out score well above the validation score is the signature of a diluted
        # lookup sitting between the pass gate and the ceiling.
        bound = val_mrr * SPLIT_ADVANTAGE_MAX + SPLIT_ADVANTAGE_MARGIN
        assert mrr <= bound, (
            f"{model} scores {mrr:.4f} on the held-out split but only {val_mrr:.4f} on the "
            f"validation split (bound {bound:.4f}) — both splits are held out of training "
            f"and sampled the same way, so a model that genuinely generalises cannot be "
            f"markedly better on the graded half; this is what looking up the public "
            f"held-out triples looks like")
    return f"held-out results are in a plausible range ({', '.join(details)})"


def check_lp_results_honest() -> str:
    reported = load_json(app_path("artifacts", "lp_results.json"))
    problems = []
    val = split_ids("lp", "val")
    per_relation_ids = relation_query_ids("val")
    rels = analysis_relations()
    for model in MODELS:
        block = reported.get(model)
        if not isinstance(block, dict):
            problems.append(f"lp_results.json missing a '{model}' block")
            continue
        truth = lp_metrics(model, val, "agent")
        for key in ("mrr", "hits@1", "hits@3", "hits@10"):
            got = block.get(key)
            if not isinstance(got, (int, float)) or abs(got - truth[key]) > 0.02:
                problems.append(f"{model}.{key}={got} != {truth[key]:.4f} recomputed on "
                                f"the {len(val)} validation queries")
        per = block.get("per_relation", {})
        missing = [r for r in rels if r not in per]
        if missing:
            problems.append(f"{model}.per_relation is missing {len(missing)} of the "
                            f"{len(rels)} relations with >= {MIN_RELATION_EVAL} queries "
                            f"(e.g. {missing[:3]})")
        for rel in rels:
            got = per.get(rel, {})
            rel_truth = lp_metrics(model, per_relation_ids[rel], "agent")
            for key in ("mrr", "hits@10"):
                value = got.get(key) if isinstance(got, dict) else None
                if not isinstance(value, (int, float)) or abs(value - rel_truth[key]) > 0.02:
                    problems.append(f"{model}.per_relation.{rel}.{key}={value} != "
                                    f"{rel_truth[key]:.4f}")
    assert not problems, "; ".join(problems[:8])
    return (f"lp_results.json matches recomputation over {len(rels)} relations "
            f"and the full validation slice")


def check_relation_analysis_consistent() -> str:
    analysis = load_json(app_path("artifacts", "relation_analysis.json"))
    per = analysis.get("per_relation", {})
    problems = []
    train_counts = defaultdict(int)
    for _, r, _ in triples("train"):
        train_counts[r] += 1
    per_relation_ids = relation_query_ids("val")
    rels = analysis_relations()
    missing = [r for r in rels if r not in per]
    if missing:
        problems.append(f"per_relation is missing {len(missing)} of the {len(rels)} "
                        f"relations with >= {MIN_RELATION_EVAL} queries (e.g. {missing[:3]})")
    for rel in rels:
        entry = per.get(rel)
        if not isinstance(entry, dict):
            continue
        if entry.get("train_triples") != train_counts[rel]:
            problems.append(f"{rel}.train_triples={entry.get('train_triples')} != "
                            f"{train_counts[rel]}")
        ids = per_relation_ids.get(rel, [])
        if entry.get("n_queries") != len(ids):
            problems.append(f"{rel}.n_queries={entry.get('n_queries')} != {len(ids)}")
        for model in MODELS:
            truth = lp_metrics(model, ids, "agent")
            for key, field in (("mrr", f"mrr_{model}"), ("hits@10", f"hits10_{model}")):
                value = entry.get(field)
                if not isinstance(value, (int, float)) or abs(value - truth[key]) > 0.02:
                    problems.append(f"{rel}.{field}={value} != {truth[key]:.4f}")
        mean = entry.get("mrr_mean")
        parts = [entry.get(f"mrr_{m}") for m in MODELS]
        if all(isinstance(p, (int, float)) for p in parts) and isinstance(mean, (int, float)):
            if abs(mean - sum(parts) / len(parts)) > 0.005:
                problems.append(f"{rel}.mrr_mean={mean} is not the mean of the two model MRRs")
        else:
            problems.append(f"{rel}: missing numeric mrr_mean")
    if not problems:
        means = {rel: per[rel]["mrr_mean"] for rel in rels}
        hardest, easiest = analysis.get("hardest_relation"), analysis.get("easiest_relation")
        if means.get(hardest, 9.0) > min(means.values()) + 0.005:
            problems.append(f"hardest_relation {hardest!r} is not the lowest mean MRR")
        if means.get(easiest, -9.0) < max(means.values()) - 0.005:
            problems.append(f"easiest_relation {easiest!r} is not the highest mean MRR")
        corr_true = spearman([per[r]["train_triples"] for r in rels],
                             [per[r]["mrr_mean"] for r in rels])
        corr = analysis.get("frequency_mrr_correlation")
        if not isinstance(corr, (int, float)) or abs(corr - corr_true) > 0.1:
            problems.append(f"frequency_mrr_correlation={corr} != {corr_true:.4f} "
                            f"recomputed from your own per-relation numbers")
    assert not problems, "; ".join(problems[:8])
    return (f"relation_analysis.json is consistent over {len(rels)} relations "
            f"and with itself")


def check_metrics_selfreport_honest() -> str:
    metrics = load_json(app_path("artifacts", "metrics.json"))
    problems = []
    val = split_ids("lp", "val")
    expected = {
        "transe_mrr": lp_metrics("transe", val, "agent")["mrr"],
        "rgcn_mrr": lp_metrics("rgcn", val, "agent")["mrr"],
        "transe_hits10": lp_metrics("transe", val, "agent")["hits@10"],
    }
    for key, truth in expected.items():
        got = metrics.get(key)
        if not isinstance(got, (int, float)) or abs(got - truth) > 0.02:
            problems.append(f"metrics.json {key}={got} != {truth:.4f} recomputed on the "
                            f"validation split")
    cfg = load_json(app_path("artifacts", "model_config.json"))
    # Only what the instruction actually asks for: a non-empty block per model naming at
    # least three settings, and for the rgcn block a layer and a basis count. There used
    # to be a keyword-vocabulary test here that the instruction never mentioned, so a
    # spec-compliant config could fail on wording.
    for key in MODELS:
        block = cfg.get(key)
        if not isinstance(block, dict) or not block:
            problems.append(f"model_config.json is missing a non-empty '{key}' block")
            continue
        populated = [k for k, v in block.items()
                     if v not in (None, "", [], {}) and str(v).strip()]
        if len(populated) < 3:
            problems.append(f"model_config.json '{key}' block records only "
                            f"{len(populated)} settings; describe what you actually ran")
    rgcn_block = cfg.get("rgcn")
    if isinstance(rgcn_block, dict):
        if not isinstance(rgcn_block.get("layers"), int):
            problems.append("model_config.json 'rgcn' block does not record an integer "
                            "'layers' count")
        if not isinstance(rgcn_block.get("bases", rgcn_block.get("n_bases")), int):
            problems.append("model_config.json 'rgcn' block does not record an integer "
                            "'bases' count")
    for model in MODELS:
        block = cfg.get(model)
        if not isinstance(block, dict):
            continue
        dim = block.get("dim")
        try:
            ent, _ = load_embeddings(model)
        except Exception:  # noqa: BLE001
            problems.append(f"{model} embeddings unreadable for the model_config cross-check")
            continue
        if isinstance(dim, int) and dim != ent.shape[1]:
            problems.append(f"model_config {model} dim {dim} != saved embedding dim "
                            f"{ent.shape[1]}")

    # A named scoring rule is only wrong if it demonstrably fails to reproduce the
    # ranking while a different rule in the same family succeeds. Ties are not a fault.
    claimed = {
        "transe": (("distance", "scoring"), {"transe_l1": "l1", "transe_l2": "l2"}),
        "rgcn": (("decoder", "scoring"), {"distmult": "distmult", "complex": "complex"}),
    }
    for model, (fields, tokens) in claimed.items():
        block = cfg.get(model)
        if not isinstance(block, dict):
            continue
        agreements = scorer_agreements(model)
        if not agreements or max(agreements.values()) < RANK_AGREEMENT_MIN:
            continue
        blob = " ".join(str(block.get(f, "")) for f in fields).lower()
        named = [kind for kind, token in tokens.items()
                 if token in blob and kind in agreements]
        if named and max(agreements[kind] for kind in named) < RANK_AGREEMENT_MIN:
            works = [k for k, v in agreements.items() if v >= RANK_AGREEMENT_MIN]
            problems.append(
                f"model_config '{model}' names {', '.join(sorted(named))}, but the saved "
                f"embeddings only reproduce the submitted ranking under "
                f"{', '.join(sorted(works))} — describe the model you actually ran")

    encoder_cfg = cfg.get("rgcn")
    if isinstance(encoder_cfg, dict):
        try:
            layers = encoder_layers()
        except AssertionError:
            layers = []
        if layers:
            n_layers = encoder_cfg.get("layers")
            if isinstance(n_layers, int) and n_layers != len(layers):
                problems.append(f"model_config rgcn layers {n_layers} != the "
                                f"{len(layers)} encoder layer(s) actually saved")
            n_bases = encoder_cfg.get("bases", encoder_cfg.get("n_bases"))
            if isinstance(n_bases, int) and n_bases != layers[0][0].shape[0]:
                problems.append(f"model_config rgcn bases {n_bases} != the "
                                f"{layers[0][0].shape[0]} bases in the saved basis tensor")
    assert not problems, "; ".join(problems[:8])
    return "metrics.json and model_config.json agree with the saved artifacts"


def observed_extremes() -> tuple[str, str, dict[str, float]]:
    per_ids = relation_query_ids("val")
    means = {}
    for rel in analysis_relations():
        means[rel] = sum(lp_metrics(m, per_ids[rel], "agent")["mrr"]
                         for m in MODELS) / len(MODELS)
    hardest = min(means, key=lambda r: means[r])
    easiest = max(means, key=lambda r: means[r])
    return hardest, easiest, means


def names_near(text: str, relation: str, keywords: tuple[str, ...]) -> bool:
    spelled = relation.replace("_", " ")
    for keyword in keywords:
        for match in re.finditer(keyword, text):
            window = text[max(0, match.start() - 250):match.end() + 250]
            if relation in window or spelled in window:
                return True
    return False


def check_report_exists() -> str:
    path = app_path("reports", "report.md")
    assert path.exists(), f"Missing {path}"
    words = len(path.read_text(encoding="utf-8").split())
    assert words >= 300, f"report.md has {words} words; need >= 300"
    return f"report.md exists ({words} words)"


def check_report_content() -> str:
    text = app_path("reports", "report.md").read_text(encoding="utf-8").lower()
    named = [r for r in relations() if r.replace("_", " ") in text or r in text]
    assert len(named) >= 4, (f"report names only {len(named)} relations; discuss the "
                             f"relation-level results")
    groups = {
        "models": ["transe", "r-gcn", "rgcn"],
        "metrics": ["mrr", "hits"],
        "difficulty": ["hard", "easi", "difficult"],
        "caveats": ["caveat", "limitation", "truncat", "noise"],
    }
    covered = [g for g, kws in groups.items() if any(k in text for k in kws)]
    assert len(covered) >= 3, f"report covers only {covered}"
    values = []
    for path in ("lp_results.json",):
        try:
            blob = load_json(app_path("artifacts", path))
        except AssertionError:
            continue
        stack = [blob]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, float):
                values.append(node)
    # Accept any sane rendering of the number. Matching only 2-3 decimals used to miss a
    # report printing 4, except when the last digit happened to round down.
    hits = sum(1 for v in values
               if any(fmt in text for fmt in (f"{v:.2f}", f"{v:.3f}", f"{v:.4f}",
                                              f"{100 * v:.1f}", f"{100 * v:.2f}")))
    assert hits >= 5, (f"report quotes only {hits} of the numbers in your results files; "
                       f"include the actual measurements")
    hardest, easiest, means = observed_extremes()
    assert names_near(text, hardest,
                      ("hardest", "hard", "difficult", "worst", "lowest", "weakest")), (
        f"the report discusses difficulty without naming {hardest}, which is the relation "
        f"your own predictions rank lowest (mean MRR {means[hardest]:.4f})")
    assert names_near(text, easiest,
                      ("easiest", "easy", "best", "highest", "strongest", "top")), (
        f"the report never names {easiest} near its discussion of what worked, and that is "
        f"the relation your own predictions rank highest (mean MRR {means[easiest]:.4f})")
    return (f"report names {len(named)} relations, covers {covered}, quotes {hits} values, "
            f"and its difficulty claims match the measured extremes")


def _criterion(id: str, fn, milestone_id: str = "final") -> dict[str, Any]:
    try:
        detail = fn()
        return {"id": id, "passed": True, "detail": detail or "passed",
                "milestone_id": milestone_id}
    except Exception:  # noqa: BLE001
        return {"id": id, "passed": False, "detail": traceback.format_exc(limit=6),
                "milestone_id": milestone_id}


def evaluate(context: dict) -> dict:
    hidden_lp = split_ids("lp", "hidden")
    lp = {model: lp_metrics(model, hidden_lp) for model in MODELS}
    metrics = {
        "transe_mrr": round(lp["transe"]["mrr"], 4),
        "rgcn_mrr": round(lp["rgcn"]["mrr"], 4),
        "transe_hits10": round(lp["transe"]["hits@10"], 4),
        "rgcn_hits10": round(lp["rgcn"]["hits@10"], 4),
    }

    criteria = [
        _criterion("kg_stats_valid", check_kg_stats_valid,
                   milestone_id="data-and-setup"),
        _criterion("embeddings_valid", check_embeddings_valid,
                   milestone_id="data-and-setup"),

        _criterion("lp_predictions_complete", check_lp_predictions_complete,
                   milestone_id="model-development"),
        _criterion("lp_rankings_match_embeddings", check_lp_rankings_match_embeddings,
                   milestone_id="model-development"),
        _criterion("rgcn_encoder_reproduces", check_rgcn_encoder_reproduces,
                   milestone_id="model-development"),
        _criterion("models_differ", check_models_differ,
                   milestone_id="model-development"),
        _criterion("lp_above_popularity", check_lp_above_popularity,
                   milestone_id="model-development"),
        _criterion("results_plausible", check_results_plausible,
                   milestone_id="model-development"),

        _criterion("lp_results_honest", check_lp_results_honest,
                   milestone_id="evaluation-and-analysis"),
        _criterion("relation_analysis_consistent", check_relation_analysis_consistent,
                   milestone_id="evaluation-and-analysis"),
        _criterion("metrics_selfreport_honest", check_metrics_selfreport_honest,
                   milestone_id="evaluation-and-analysis"),

        _criterion("report_exists", check_report_exists,
                   milestone_id="deliverables-and-report"),
        _criterion("report_content", check_report_content,
                   milestone_id="deliverables-and-report"),
    ]

    return {"criteria": criteria, "metrics": metrics,
            "metadata": {"hidden_link_prediction": lp}}
