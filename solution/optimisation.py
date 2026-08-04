from __future__ import annotations

import math
import os
from collections import Counter

import torch
from embedding_models import RelationalGCN, PairwiseTranslationKGE, distmult_bce_loss
from kg_io import KnowledgeGraph

# Minimum batch / chunk sizes before we stop halving on OOM
_TRANSE_BATCH_FLOOR = 256
_RGCN_CHUNK_FLOOR = 512


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_memory_error(exc: Exception) -> bool:
    """Return True when the exception signals GPU / MPS memory exhaustion."""
    if not isinstance(exc, RuntimeError):
        return False
    lowered = str(exc).lower()
    return "out of memory" in lowered or "can't allocate" in lowered


def clear_device_memory(device: torch.device) -> None:
    """Free cached memory allocations on GPU or MPS devices."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Regularisation
# ─────────────────────────────────────────────────────────────────────────────


def _l1_reg(node_emb: torch.Tensor, rel_emb: torch.Tensor, scale: float) -> torch.Tensor:
    """Mean absolute L1 penalty on entity and relation embeddings."""
    return scale * (node_emb.abs().mean() + rel_emb.abs().mean())


# ─────────────────────────────────────────────────────────────────────────────
# Graph adjacency construction
# ─────────────────────────────────────────────────────────────────────────────


def build_relation_adjacency(graph: KnowledgeGraph, device: torch.device) -> list:
    """
    Build per-relation edge lists with in-degree normalisation weights.

    Returns one (src, dst, norm) tensor triple for every directed relation type.
    Forward relations occupy indices 0..R-1; inverse relations R..2R-1.
    Each normalisation weight equals 1 / in_degree(dst) for that edge,
    computed from destination-node edge counts across the training graph.
    """
    n_rel = len(graph.relations)
    src_lists = [[] for _ in range(2 * n_rel)]
    dst_lists = [[] for _ in range(2 * n_rel)]

    for head_str, rel_str, tail_str in graph.train:
        h_i = graph.eid[head_str]
        t_i = graph.eid[tail_str]
        r_i = graph.rid[rel_str]
        # Forward edge: head -> tail
        src_lists[r_i].append(h_i)
        dst_lists[r_i].append(t_i)
        # Inverse edge: tail -> head
        src_lists[r_i + n_rel].append(t_i)
        dst_lists[r_i + n_rel].append(h_i)

    adjacency: list = []
    for s_list, d_list in zip(src_lists, dst_lists):
        in_degree = Counter(d_list)
        norms = [1.0 / in_degree[d] for d in d_list]
        adjacency.append((
            torch.tensor(s_list, dtype=torch.long,  device=device),
            torch.tensor(d_list, dtype=torch.long,  device=device),
            torch.tensor(norms,  dtype=torch.float, device=device),
        ))
    return adjacency


# ─────────────────────────────────────────────────────────────────────────────
# OOM handler
# ─────────────────────────────────────────────────────────────────────────────


def _handle_oom(
    opt: torch.optim.Optimizer,
    device: torch.device,
    current_size: int,
    floor: int,
    logger,
    label: str,
) -> int:
    """Zero gradients, free GPU cache, halve the batch/chunk size, return new size."""
    opt.zero_grad(set_to_none=True)
    clear_device_memory(device)
    new_size = max(floor, current_size // 2)
    logger(f"{label} shrunk to {new_size} after memory error")
    return new_size


# ─────────────────────────────────────────────────────────────────────────────
# TransE training  (AdamW + cosine LR annealing)
# ─────────────────────────────────────────────────────────────────────────────


def train_transe(
    graph: KnowledgeGraph,
    device: torch.device,
    dim: int,
    n_epochs: int,
    n_neg: int,
    batch_size: int,
    margin: float,
    lr: float,
    seed: int,
    logger,
) -> PairwiseTranslationKGE:
    """
    Train a translation-based KGE model using AdamW and cosine LR annealing.

    Key differences from a vanilla Adam-based TransE:
    - Optimiser: **AdamW** with weight_decay=1e-4 (decoupled L2 regularisation).
    - Schedule: **cosine annealing** reduces LR from `lr` to `lr/100` over training.
    - Loss: **pairwise margin hinge** instead of adversarial BCE.
    - Entity vectors are L2-normalised after every gradient step.
    - OOM events halve the batch size and retry the step.
    """
    torch.manual_seed(seed)
    n_ent = len(graph.entities)
    n_rel = len(graph.relations)

    model = PairwiseTranslationKGE(n_ent, n_rel, dim, margin).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=lr / 100.0
    )

    trips = torch.tensor(graph.index_triples(graph.train), device=device)
    bsz = batch_size

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(trips), device=device)
        epoch_loss = 0.0
        pos = 0

        while pos < len(trips):
            batch = trips[perm[pos: pos + bsz]]
            try:
                loss = model(batch, n_ent, n_neg)
                opt.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                # Project entities back onto unit sphere after each update
                model.normalise_entities()
                epoch_loss += loss.item() * len(batch)
                pos += bsz
            except RuntimeError as exc:
                if not is_memory_error(exc) or bsz <= _TRANSE_BATCH_FLOOR:
                    raise
                bsz = _handle_oom(opt, device, bsz, _TRANSE_BATCH_FLOOR, logger, "transe batch")

        scheduler.step()

        if ep % 25 == 0 or ep == n_epochs - 1:
            cur_lr = scheduler.get_last_lr()[0]
            logger(f"transe  epoch={ep:4d}  loss={epoch_loss / len(trips):.4f}  lr={cur_lr:.2e}")

    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# R-GCN training  (SGD + momentum + warmup)
# ─────────────────────────────────────────────────────────────────────────────


def _warmup_factor(ep: int, warmup_epochs: int) -> float:
    """Linear LR warmup from 0 to 1 over the first `warmup_epochs` epochs."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (ep + 1) / warmup_epochs)


def train_rgcn(
    graph: KnowledgeGraph,
    device: torch.device,
    dim: int,
    n_epochs: int,
    lr: float,
    seed: int,
    logger,
) -> tuple[RelationalGCN, tuple]:
    """
    Train an R-GCN encoder + DistMult decoder.

    Key differences from the previous implementation:
    - Optimiser: **SGD with momentum** (0.9) instead of Adam.
    - LR schedule: **linear warmup** (10 epochs) then constant.
    - Regularisation: **L1 penalty** on embeddings instead of L2.
    - Each epoch: one full-graph encode pass, then chunked mini-batch backward.
    - OOM events halve the chunk size and retry the epoch.

    Returns (model, (entity_emb, relation_emb)) with both tensors detached.
    """
    torch.manual_seed(seed)

    n_ent = len(graph.entities)
    n_rel = len(graph.relations)
    model = RelationalGCN(n_ent, n_rel, dim).to(device)

    eff_lr = float(os.environ.get("LP_GNN_LR", str(lr)))
    opt = torch.optim.SGD(model.parameters(), lr=eff_lr, momentum=0.9, weight_decay=0.0)

    warmup_epochs = 10
    adj = build_relation_adjacency(graph, device)
    trips = torch.tensor(graph.index_triples(graph.train), device=device)
    n_total = len(trips)
    chunk = int(os.environ.get("LP_GNN_CHUNK", "20000"))
    n_neg = int(os.environ.get("LP_GNN_NEG", "16"))

    for ep in range(n_epochs):
        # Apply linear warmup by adjusting the learning rate manually
        wf = _warmup_factor(ep, warmup_epochs)
        for pg in opt.param_groups:
            pg["lr"] = eff_lr * wf

        perm = torch.randperm(n_total, device=device)
        ep_loss = 0.0

        try:
            model.train()
            z = model.encode(adj, edge_dropout=0.15)
            opt.zero_grad()

            for begin in range(0, n_total, chunk):
                mb = trips[perm[begin: begin + chunk]]
                n_mb = len(mb)
                frac = n_mb / n_total

                h_e = z[mb[:, 0]]
                r_e = model.rel_vecs[mb[:, 1]]
                t_e = z[mb[:, 2]]

                pos_sc = model.score(h_e, r_e, t_e)

                neg_t = torch.randint(0, n_ent, (n_mb, n_neg), device=device)
                neg_h = torch.randint(0, n_ent, (n_mb, n_neg), device=device)
                neg_sc = torch.cat([
                    model.score(h_e.unsqueeze(1), r_e.unsqueeze(1), z[neg_t]),
                    model.score(z[neg_h], r_e.unsqueeze(1), t_e.unsqueeze(1)),
                ], dim=1)

                chunk_loss = distmult_bce_loss(pos_sc, neg_sc) * frac
                chunk_loss = chunk_loss + _l1_reg(z, model.rel_vecs, 5e-7 * frac)
                more_chunks = (begin + chunk < n_total)
                chunk_loss.backward(retain_graph=more_chunks)
                ep_loss += chunk_loss.item() * n_total

            opt.step()

        except RuntimeError as exc:
            if not is_memory_error(exc) or chunk <= _RGCN_CHUNK_FLOOR:
                raise
            chunk = _handle_oom(opt, device, chunk, _RGCN_CHUNK_FLOOR, logger, "rgcn chunk")
            continue

        if ep % 25 == 0 or ep == n_epochs - 1:
            logger(f"rgcn   epoch={ep:4d}  loss={ep_loss / n_total:.4f}  lr={eff_lr * wf:.2e}")

    model.eval()
    with torch.no_grad():
        final_z = model.encode(adj)
    return model, (final_z.detach(), model.rel_vecs.detach())
