from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────


def pairwise_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss (hinge loss) for knowledge graph embedding.

    For each positive triple score p and each corresponding negative score n,
    penalises configurations where n is not at least `margin` below p:
        loss = mean( max(0, margin - p + n) )

    This is structurally different from adversarial BCE: it uses a hard margin
    boundary rather than a log-sigmoid boundary, and weights all negatives equally
    rather than up-weighting hard ones.

    Parameters
    ----------
    pos_scores : (B,)   score for each positive triple
    neg_scores : (B, K) scores for K negatives per positive
    margin     : gap the model is trained to maintain
    """
    # Expand positives to (B, K) for broadcasting
    pos_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)
    hinge = torch.clamp(margin - pos_exp + neg_scores, min=0.0)
    return hinge.mean()


def distmult_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss used for the bilinear (DistMult) decoder.

    Treats the problem as binary classification: positives should score high,
    negatives should score low. Applies sigmoid + BCELoss rather than the
    self-adversarially weighted variant used elsewhere.

    Parameters
    ----------
    pos_scores : (B,)   score for each positive triple
    neg_scores : (B, K) scores for K negatives per positive
    """
    pos_target = torch.ones_like(pos_scores)
    neg_target = torch.zeros_like(neg_scores)

    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_target)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_target)
    return pos_loss + neg_loss


# ─────────────────────────────────────────────────────────────────────────────
# TransE model  (L2-distance scoring, pairwise margin loss)
# ─────────────────────────────────────────────────────────────────────────────


class PairwiseTranslationKGE(torch.nn.Module):
    """
    Translation-based knowledge graph embedding trained with pairwise hinge loss.

    Like TransE, each entity and relation gets a learned vector and plausible
    triples satisfy h + r ≈ t.  Differences from a vanilla TransE:

    * Scoring uses **L2 (Euclidean) distance** instead of L1 (Manhattan).
    * Training uses **pairwise margin ranking loss** instead of adversarial BCE.
    * Entity embeddings are **L2-normalised** after every gradient step (standard
      TransE regularisation) by dividing by their per-row norm.
    * Embeddings are initialised from a **Kaiming-uniform** distribution scaled to
      the embedding dimension rather than the classical 6/sqrt(d) uniform.

    Parameters
    ----------
    n_ent   : number of entities
    n_rel   : number of relation types
    dim     : embedding dimensionality
    margin  : hinge margin separating positives from negatives
    """

    def __init__(self, n_ent: int, n_rel: int, dim: int, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.dim = dim
        self.entity_emb = torch.nn.Embedding(n_ent, dim)
        self.relation_emb = torch.nn.Embedding(n_rel, dim)

        # Kaiming-uniform init — spread proportional to 1/sqrt(dim)
        bound = 1.0 / math.sqrt(dim)
        torch.nn.init.uniform_(self.entity_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.relation_emb.weight, -bound, bound)

        # Normalise entity embeddings at startup
        with torch.no_grad():
            self.entity_emb.weight.div_(
                self.entity_emb.weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            )

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Negative L2 distance: higher score = more plausible triple."""
        return -((h + r - t) ** 2).sum(dim=-1).sqrt()

    def normalise_entities(self) -> None:
        """Project all entity vectors onto the unit hypersphere (in-place)."""
        with torch.no_grad():
            norms = self.entity_emb.weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.entity_emb.weight.div_(norms)

    def forward(self, batch: torch.Tensor, n_ent: int, n_neg: int) -> torch.Tensor:
        """
        Compute pairwise margin loss for a mini-batch of positive triples.

        Corrupts each triple by replacing either the tail or the head with a
        uniformly sampled random entity.
        """
        h = self.entity_emb(batch[:, 0])
        r = self.relation_emb(batch[:, 1])
        t = self.entity_emb(batch[:, 2])

        pos_scores = self.score(h, r, t)

        # Corrupt tails
        neg_t_ids = torch.randint(0, n_ent, (len(batch), n_neg), device=batch.device)
        neg_t_scores = self.score(
            h.unsqueeze(1),
            r.unsqueeze(1),
            self.entity_emb(neg_t_ids),
        )

        # Corrupt heads
        neg_h_ids = torch.randint(0, n_ent, (len(batch), n_neg), device=batch.device)
        neg_h_scores = self.score(
            self.entity_emb(neg_h_ids),
            r.unsqueeze(1),
            t.unsqueeze(1),
        )

        neg_scores = torch.cat([neg_t_scores, neg_h_scores], dim=1)
        return pairwise_margin_loss(pos_scores, neg_scores, margin=self.margin)


# ─────────────────────────────────────────────────────────────────────────────
# R-GCN encoder  (3 layers, 4 bases, bias, DistMult decoder)
# ─────────────────────────────────────────────────────────────────────────────


class RelationalGCN(torch.nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) encoder.

    Builds entity representations via multiple rounds of message passing over
    the relational graph. Each relation type (forward and inverse) has a distinct
    weight matrix expressed as a weighted mixture of a shared set of basis matrices
    (basis decomposition) plus a per-layer self-loop transform.

    Differences from the reference R-GCN:
    * Uses **3 propagation layers** instead of 2.
    * Uses **4 shared basis matrices** per layer instead of 3.
    * Parameters initialised with **Kaiming-normal** rather than Xavier-uniform.

    The decoder is DistMult (element-wise product of head, relation, tail vectors).

    Parameters
    ----------
    n_ent    : number of entities
    n_rel    : number of relation types
    dim      : uniform representation size across all layers
    n_layers : propagation depth (default 3)
    n_bases  : shared basis count per layer (default 4)
    """

    def __init__(
        self,
        n_ent: int,
        n_rel: int,
        dim: int,
        n_layers: int = 3,
        n_bases: int = 4,
    ) -> None:
        super().__init__()
        self.n_rel = n_rel
        self.n_bases = n_bases
        self.n_layers = n_layers

        # Input entity embeddings
        self.input_emb = torch.nn.Embedding(n_ent, dim)
        torch.nn.init.kaiming_normal_(self.input_emb.weight, nonlinearity="relu")

        # Per-relation vectors for DistMult decoder
        self.rel_vecs = torch.nn.Parameter(torch.empty(n_rel, dim))
        torch.nn.init.kaiming_normal_(self.rel_vecs, nonlinearity="relu")

        # Per-layer parameters
        self.self_weights = torch.nn.ParameterList()
        self.shared_bases = torch.nn.ParameterList()
        self.mix_weights = torch.nn.ParameterList()

        for _ in range(n_layers):
            # Self-loop transform
            sw = torch.nn.Parameter(torch.empty(dim, dim))
            torch.nn.init.kaiming_normal_(sw, nonlinearity="relu")
            self.self_weights.append(sw)

            # Shared basis matrices: (n_bases, dim, dim)
            B = torch.nn.Parameter(torch.empty(n_bases, dim, dim))
            torch.nn.init.kaiming_normal_(B, nonlinearity="relu")
            self.shared_bases.append(B)

            # Per-relation basis coefficients: (2*n_rel, n_bases)
            C = torch.nn.Parameter(torch.empty(2 * n_rel, n_bases))
            torch.nn.init.kaiming_normal_(C, nonlinearity="relu")
            self.mix_weights.append(C)

        self.inter_drop = torch.nn.Dropout(p=0.2)

    def encode(self, adjacency: list, edge_dropout: float = 0.0) -> torch.Tensor:
        """
        Propagate messages through the relational graph and return entity representations.

        Parameters
        ----------
        adjacency    : list of (src_idx, dst_idx, norm_weights) tensors,
                       one per directed relation type (forward then inverse).
        edge_dropout : fraction of edges to stochastically suppress per layer.
        """
        h = self.input_emb.weight

        for layer in range(self.n_layers):
            # Project all entity states through each basis: (n_bases, N, dim)
            basis_projections = torch.stack(
                [h @ self.shared_bases[layer][b] for b in range(self.n_bases)]
            )
            agg = torch.zeros_like(h)

            for rel_type, (esrc, edst, enorm) in enumerate(adjacency):
                if esrc.numel() == 0:
                    continue
                s, d, nw = esrc, edst, enorm
                if edge_dropout > 0.0:
                    keep = torch.rand(s.shape[0], device=s.device) >= edge_dropout
                    s, d, nw = s[keep], d[keep], nw[keep]
                    if s.numel() == 0:
                        continue

                # Relation-specific mixture of basis projections
                alpha = self.mix_weights[layer][rel_type].view(self.n_bases, 1, 1)
                rel_proj = (basis_projections * alpha).sum(dim=0)   # (N, dim)
                agg.index_add_(0, d, rel_proj[s] * nw.view(-1, 1))

            # Combine neighbour aggregation and self-loop transform
            h = torch.relu(agg + h @ self.self_weights[layer])
            if layer < self.n_layers - 1:
                h = self.inter_drop(h)

        return h

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DistMult bilinear score: Σ(h ⊙ r ⊙ t) over the embedding dimension."""
        return (h * r * t).sum(dim=-1)
