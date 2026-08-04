from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from kg_io import KnowledgeGraph
from narrative import write_markdown_report
from optimisation import clear_device_memory, train_rgcn, train_transe
from ranking import compute_ranked_lists, eval_metrics, export_rankings, spearman_rho

APP_DIR   = Path(os.environ.get("OTTER_APP_DIR",  "/app"))
DATA_DIR  = Path(os.environ.get("OTTER_DATA_DIR", APP_DIR / "data"))
ARTIFACTS = APP_DIR / "artifacts"
REPORTS   = APP_DIR / "reports"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMB_DIM    = int(os.environ.get("LP_DIM",        "200"))
GNN_DIM    = int(os.environ.get("LP_GNN_DIM",    "128"))
TE_EPOCHS  = int(os.environ.get("LP_TE_EPOCHS",  "150"))
GNN_EPOCHS = int(os.environ.get("LP_GNN_EPOCHS", "200"))
N_NEG      = int(os.environ.get("LP_NEG_COUNT",  "48"))
BATCH      = int(os.environ.get("LP_BATCH",      "4096"))
MARGIN     = float(os.environ.get("LP_MARGIN",   "1.0"))   # hinge margin
LR         = float(os.environ.get("LP_LR",       "1e-3"))
GLOBAL_SEED = 7
TOP_K       = 20
MIN_REL_Q   = 30


def log(msg: str) -> None:
    print(f">> [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_device() -> torch.device:
    """Pick the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_encoder_layers(model, enc_dir: Path) -> None:
    """Persist R-GCN encoder weight tensors in the standard encoder-protocol format.

    Saves three files per layer:
    - ``bases_layerN.npy``  — shared basis matrices (n_bases, dim, dim)
    - ``coeff_layerN.npy``  — per-relation basis coefficients (2*n_rel, n_bases)
    - ``self_layerN.npy``   — self-loop weight matrix (dim, dim)
    """
    enc_dir.mkdir(exist_ok=True)
    for i in range(len(model.self_weights)):
        np.save(enc_dir / f"bases_layer{i}.npy",
                model.shared_bases[i].detach().cpu().numpy())
        np.save(enc_dir / f"coeff_layer{i}.npy",
                model.mix_weights[i].detach().cpu().numpy())
        np.save(enc_dir / f"self_layer{i}.npy",
                model.self_weights[i].detach().cpu().numpy())


def _count_triples_per_relation(graph: KnowledgeGraph) -> dict[str, int]:
    """Return a mapping from relation string to its training-triple count."""
    counts: dict[str, int] = defaultdict(int)
    for _, r, _ in graph.train:
        counts[r] += 1
    return dict(counts)


def _group_val_ids_by_relation(
    graph: KnowledgeGraph,
    val_ids: list[str],
) -> dict[str, list[str]]:
    """Map relation -> list of validated query IDs that belong to that relation."""
    qid_to_rel = {q["query_id"]: q["relation"] for q in graph.queries}
    buckets: dict[str, list[str]] = defaultdict(list)
    for qid in val_ids:
        buckets[qid_to_rel[qid]].append(qid)
    return dict(buckets)


def _compute_all_metrics(
    graph: KnowledgeGraph,
    all_rankings: dict[str, dict[str, list[str]]],
    val_ids: list[str],
    stable_rels: list[str],
    buckets: dict[str, list[str]],
) -> dict[str, dict]:
    """Compute overall + per-relation metrics for each model and return as a nested dict."""
    results: dict[str, dict] = {}
    for tag in ("transe", "rgcn"):
        ranked = all_rankings[tag]
        overall = eval_metrics(ranked, val_ids, graph.val_answers)
        overall["per_relation"] = {
            rel: eval_metrics(ranked, buckets[rel], graph.val_answers)
            for rel in stable_rels
        }
        results[tag] = overall
    return results


def _build_per_rel_stats(
    lp_results: dict,
    stable_rels: list[str],
    train_counts: dict[str, int],
) -> dict[str, dict]:
    """Assemble the per-relation statistics table from pre-computed metric blocks."""
    table: dict[str, dict] = {}
    for rel in stable_rels:
        te_m = lp_results["transe"]["per_relation"][rel]
        gn_m = lp_results["rgcn"]["per_relation"][rel]
        avg = (te_m["mrr"] + gn_m["mrr"]) / 2.0
        table[rel] = {
            "n_queries":     te_m["n"],
            "train_triples": train_counts.get(rel, 0),
            "mrr_transe":    te_m["mrr"],
            "mrr_rgcn":      gn_m["mrr"],
            "mrr_mean":      avg,
            "hits10_transe": te_m["hits@10"],
            "hits10_rgcn":   gn_m["hits@10"],
        }
    return table


def execute() -> None:
    device = select_device()
    log(
        f"device={device}  emb_dim={EMB_DIM}  gnn_dim={GNN_DIM}  "
        f"te_epochs={TE_EPOCHS}  gnn_epochs={GNN_EPOCHS}"
    )

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    graph = KnowledgeGraph()
    log(
        f"graph loaded — {len(graph.entities)} entities, "
        f"{len(graph.relations)} relations, "
        f"{len(graph.train)} train triples, "
        f"{len(graph.queries)} eval queries"
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "embeddings").mkdir(exist_ok=True)
    (ARTIFACTS / "predictions").mkdir(exist_ok=True)

    # ── Train R-GCN first (graph-structure model) ──────────────────────────────
    gnn_model, (gnn_ent, gnn_rel) = train_rgcn(
        graph=graph, device=device, dim=GNN_DIM, n_epochs=GNN_EPOCHS,
        lr=3e-3, seed=GLOBAL_SEED + 5, logger=log,
    )
    np.save(ARTIFACTS / "embeddings" / "rgcn_entity.npy",   gnn_ent.cpu().numpy())
    np.save(ARTIFACTS / "embeddings" / "rgcn_relation.npy", gnn_rel.cpu().numpy())
    np.save(
        ARTIFACTS / "embeddings" / "rgcn_input.npy",
        gnn_model.input_emb.weight.detach().cpu().numpy(),
    )
    _save_encoder_layers(gnn_model, ARTIFACTS / "encoder")
    log(f"R-GCN embeddings saved ({len(gnn_model.self_weights)} layers, "
        f"bases={gnn_model.n_bases})")
    clear_device_memory(device)

    # ── Train TransE second (translation model) ────────────────────────────────
    te_model = train_transe(
        graph=graph, device=device, dim=EMB_DIM, n_epochs=TE_EPOCHS,
        n_neg=N_NEG, batch_size=BATCH, margin=MARGIN, lr=LR,
        seed=GLOBAL_SEED, logger=log,
    )
    te_ent = te_model.entity_emb.weight.detach()
    te_rel = te_model.relation_emb.weight.detach()
    np.save(ARTIFACTS / "embeddings" / "transe_entity.npy",   te_ent.cpu().numpy())
    np.save(ARTIFACTS / "embeddings" / "transe_relation.npy", te_rel.cpu().numpy())
    log("TransE embeddings saved")
    clear_device_memory(device)

    # ── Score all queries with both models ─────────────────────────────────────
    all_rankings: dict[str, dict[str, list[str]]] = {}
    # TransE uses translation_mode=True (L2 scoring), R-GCN uses DistMult
    for tag, ent_e, rel_e, use_trans in [
        ("transe", te_ent,  te_rel,  True),
        ("rgcn",   gnn_ent, gnn_rel, False),
    ]:
        all_rankings[tag] = compute_ranked_lists(
            graph, device, ent_e, rel_e, use_trans, log, margin=MARGIN
        )
        log(f"{tag}: ranked {len(all_rankings[tag])} queries")
        export_rankings(
            path=ARTIFACTS / "predictions" / f"lp_{tag}.csv",
            rows=[
                (q["query_id"], "|".join(all_rankings[tag][q["query_id"]]))
                for q in graph.queries
            ],
        )

    # ── Evaluation metrics ─────────────────────────────────────────────────────
    val_ids = sorted(graph.val_answers.keys())
    buckets = _group_val_ids_by_relation(graph, val_ids)
    stable_rels = sorted(r for r, ids in buckets.items() if len(ids) >= MIN_REL_Q)

    lp_results = _compute_all_metrics(graph, all_rankings, val_ids, stable_rels, buckets)
    (ARTIFACTS / "lp_results.json").write_text(json.dumps(lp_results, indent=2) + "\n")

    # ── Relation analysis ──────────────────────────────────────────────────────
    train_counts = _count_triples_per_relation(graph)
    per_rel_stats = _build_per_rel_stats(lp_results, stable_rels, train_counts)

    freq_list = [per_rel_stats[r]["train_triples"] for r in stable_rels]
    mrr_list  = [per_rel_stats[r]["mrr_mean"]      for r in stable_rels]

    relation_analysis = {
        "min_relation_eval":         MIN_REL_Q,
        "n_relations_analyzed":      len(stable_rels),
        "per_relation":              per_rel_stats,
        "hardest_relation":          min(stable_rels, key=lambda r: per_rel_stats[r]["mrr_mean"]),
        "easiest_relation":          max(stable_rels, key=lambda r: per_rel_stats[r]["mrr_mean"]),
        "frequency_mrr_correlation": spearman_rho(freq_list, mrr_list),
    }
    (ARTIFACTS / "relation_analysis.json").write_text(
        json.dumps(relation_analysis, indent=2) + "\n"
    )

    # ── KG statistics ──────────────────────────────────────────────────────────
    kg_stats = {
        "n_entities":        len(graph.entities),
        "n_relations":       len(graph.relations),
        "n_train_triples":   len(graph.train),
        "relation_counts":   dict(sorted(train_counts.items())),
        "entity_type_counts": {t: len(v) for t, v in sorted(graph.type_index.items())},
    }
    (ARTIFACTS / "kg_stats.json").write_text(json.dumps(kg_stats, indent=2) + "\n")

    # ── Model configuration ────────────────────────────────────────────────────
    n_gnn_layers = len(gnn_model.self_weights)
    model_cfg = {
        "transe": {
            "dim":       EMB_DIM,
            "epochs":    TE_EPOCHS,
            "negatives": N_NEG,
            "margin":    MARGIN,
            "loss":      "pairwise margin hinge",
            "distance":  "l2",
            "scoring":   "translation l2",
            "optimizer": "adamw with cosine lr annealing",
        },
        "rgcn": {
            "dim":      GNN_DIM,
            "epochs":   GNN_EPOCHS,
            "layers":   n_gnn_layers,
            "bases":    gnn_model.n_bases,
            "decoder":  "distmult",
            "scoring":  "distmult",
            "edge_dropout": 0.15,
            "optimizer": "sgd with momentum and linear warmup",
        },
        "top_k":  TOP_K,
        "models": ["transe", "rgcn"],
    }
    (ARTIFACTS / "model_config.json").write_text(json.dumps(model_cfg, indent=2) + "\n")

    # ── Summary metrics ────────────────────────────────────────────────────────
    metrics_out = {
        "transe_mrr":    lp_results["transe"]["mrr"],
        "rgcn_mrr":      lp_results["rgcn"]["mrr"],
        "transe_hits10": lp_results["transe"]["hits@10"],
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics_out, indent=2) + "\n")

    write_markdown_report(
        graph=graph, results=lp_results, rel_analysis=relation_analysis,
        cfg=model_cfg, out_dir=REPORTS,
    )
    log(f"done — {json.dumps(metrics_out)}")
