from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
DATA_DIR = Path(os.environ.get("OTTER_DATA_DIR", APP_DIR / "data"))


def _parse_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file line-by-line and return a list of row dicts."""
    with path.open(encoding="utf-8") as fh:
        raw_lines = fh.read().splitlines()
    if not raw_lines:
        return []
    header = raw_lines[0].split("\t")
    records: list[dict[str, str]] = []
    for line in raw_lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        records.append({header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))})
    return records


class KnowledgeGraph:
    """
    Loads and indexes a CoDEx-style knowledge graph for link prediction.

    Entities and relations are indexed by their row order in the TSV files.
    Training and validation triples are loaded from the kg/ subdirectory.
    Evaluation queries and their partial answers are read from lp/.

    Attributes
    ----------
    entities      : ordered list of entity ID strings
    entity_names  : entity_id -> human-readable name
    entity_types  : entity_id -> type label string
    eid           : entity_id -> integer row index
    relations     : ordered list of relation strings
    rid           : relation -> integer row index
    train         : list of (head, relation, tail) string triples
    valid         : list of (head, relation, tail) string triples
    queries       : list of query dicts with keys query_id/source_entity/relation/target
    val_answers   : query_id -> answer entity string (labelled subset)
    type_index    : entity type -> list of entity integer indices
    tail_filter   : (head, relation) -> set of known tail entities
    head_filter   : (relation, tail) -> set of known head entities
    """

    def __init__(self) -> None:
        manifest_path = DATA_DIR / "dataset_manifest.json"
        self.manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8"))

        # ── Entities ──────────────────────────────────────────────────────────
        ent_rows = _parse_tsv(DATA_DIR / "kg" / "entities.tsv")
        self.entities: list[str] = [row["entity_id"] for row in ent_rows]
        self.entity_names: dict[str, str] = {row["entity_id"]: row["name"] for row in ent_rows}
        self.entity_types: dict[str, str] = {row["entity_id"]: row["type"] for row in ent_rows}
        self.eid: dict[str, int] = {e: idx for idx, e in enumerate(self.entities)}

        # ── Relations ─────────────────────────────────────────────────────────
        rel_rows = _parse_tsv(DATA_DIR / "kg" / "relations.tsv")
        self.relations: list[str] = [row["relation"] for row in rel_rows]
        self.rid: dict[str, int] = {r: idx for idx, r in enumerate(self.relations)}

        # ── Triples ───────────────────────────────────────────────────────────
        self.train: list[tuple[str, str, str]] = self._load_triples("train.tsv")
        self.valid: list[tuple[str, str, str]] = self._load_triples("valid.tsv")

        # ── Evaluation queries and labelled answers ────────────────────────────
        self.queries: list[dict] = _parse_tsv(DATA_DIR / "lp" / "eval_queries.tsv")
        self.val_answers: dict[str, str] = {
            row["query_id"]: row["answer_entity"]
            for row in _parse_tsv(DATA_DIR / "lp" / "val_answers.tsv")
        }

        # ── Entity-type index ─────────────────────────────────────────────────
        self.type_index: dict[str, list[int]] = defaultdict(list)
        for eid_str in self.entities:
            self.type_index[self.entity_types[eid_str]].append(self.eid[eid_str])

        # ── Filtered-evaluation answer sets ───────────────────────────────────
        self.tail_filter: dict[tuple, set] = defaultdict(set)
        self.head_filter: dict[tuple, set] = defaultdict(set)
        for h, r, t in self.train + self.valid:
            self.tail_filter[h, r].add(t)
            self.head_filter[r, t].add(h)

    # ── Backward-compat aliases ────────────────────────────────────────────────
    # (some callers from previous iterations used these names)

    def _load_triples(self, filename: str) -> list[tuple[str, str, str]]:
        rows = _parse_tsv(DATA_DIR / "kg" / filename)
        return [(row["head"], row["relation"], row["tail"]) for row in rows]

    def index_triples(
        self, triples: list[tuple[str, str, str]]
    ):
        """Convert (head, relation, tail) string triples to an int64 numpy array."""
        import numpy as np

        return np.array(
            [[self.eid[h], self.rid[r], self.eid[t]] for h, r, t in triples],
            dtype=np.int64,
        )


# Legacy alias so other modules can still import 'GraphData'
GraphData = KnowledgeGraph
