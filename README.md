# otter/codex-linkpred — EC notes

Link prediction on the CoDEx-M knowledge graph. The agent trains TransE and
an R-GCN on the shipped triples, ranks held-out queries over the full entity
set under the filtered protocol, and breaks the results down by relation
type.

This task began as a KG20C scholarly-graph task. KG20C is released for
research use only, which the guide does not allow, so the graph was swapped
for **CoDEx-M** — an openly licensed benchmark of the same scale that keeps
the two-model link-prediction design intact. The link-prediction task,
verifier, and anti-cheat structure are unchanged; only the underlying graph
and the per-relation breadth differ.

## Source and provenance

- Dataset: CoDEx-M, from Safavi & Koutra, *CoDEx: A Comprehensive Knowledge
  Graph Completion Benchmark*, EMNLP 2020,
  [arXiv:2009.07810](https://arxiv.org/abs/2009.07810).
- Repository: <https://github.com/tsafavi/codex>. Files used, unmodified
  apart from re-keying: `data/triples/codex-m/{train,valid,test}.txt`
  (185,584 / 10,310 / 10,311 triples), `data/entities/en/entities.json`
  (labels), `data/relations/en/relations.json` (labels),
  `data/types/entity2types.json` and `data/types/en/types.json` (entity
  types). 17,050 entities, 51 relations. Counts match the repository.
- **License: MIT.** The CoDEx repository ships under the MIT license
  (`LICENSE`, Copyright (c) 2021 Tara Safavi), and the underlying entities,
  relations and types are extracted from Wikidata, which is released under
  CC0 1.0 (public domain). Both are on the platform's accepted enum; MIT is
  used in `task.toml`. Nothing here is research-use restricted.
- Entity labels are Wikidata entity labels; relation names are slugified
  Wikidata property labels (e.g. P17 "country" -> `country`). Entity types
  are the primary Wikidata type label per entity.

## Design

- **Split.** The official CoDEx-M splits are kept, so numbers stay
  comparable with published baselines. `train.tsv` and `valid.tsv` ship to
  the agent. Each valid and test triple becomes **one** evaluation query —
  one direction, chosen by a seeded coin flip per triple, 20,621 in total.
  The 10,310 from valid are answered in `/app/data/lp/val_answers.tsv` so
  the agent can measure itself, and the 10,311 from test are graded from
  `tests/hidden_labels.json`.

  Shipping both directions of a triple was a real leak and is described
  under Residual risks; one direction per triple is what closes it. Query
  ids are renumbered contiguously under a seeded shuffle so the discarded
  direction leaves no gap to read structure from.
- **Anonymised keys.** Entity ids are re-keyed to `E00000...E17049` under a
  seeded permutation, and Wikidata ids are not shipped, which raises the bar
  on looking up held-out answers from the public repo. Labels are kept so
  the relation analysis is readable.
- **Ranking protocol.** The agent submits its top 20 candidates per query,
  ranked over the full entity set. The verifier walks that list, skips
  entities that are true answers for the query anywhere in the full graph,
  and takes the position of the gold answer — the standard filtered
  protocol, truncated at 20. CoDEx relations do not have clean single-type
  target slots, so ranking is over all entities rather than type-restricted;
  this is the canonical CoDEx protocol and keeps the candidate pool large.
- **Per-relation analysis** is reported over the 27 relations with at least
  30 validation queries (`MIN_RELATION_EVAL = 30`), computed identically in
  the oracle and the verifier so the two agree. Relations with too few
  queries carry too much noise for a stable per-relation MRR.
- **What the agent self-reports** is measured on the labelled validation
  slice and honesty-checked against recomputation using only the filtering
  the agent can see; what decides pass/fail is recomputed on the held-out
  slice with full filtering.

## Oracle

`solution/src/pipeline.py`, run through `solution/solve.sh`. No external
model downloads; everything trains from scratch on the shipped triples.

- **TransE**: dim 200, 150 epochs, 48 negatives per positive, margin 12,
  L1 translation distance, self-adversarial sigmoid loss, Adam 1e-3.
- **R-GCN**: two layers, basis-decomposed forward and inverse transforms per
  relation (3 bases) plus a self-loop transform, in-degree message
  normalisation, edge dropout 0.2, DistMult decoder over the encoder output,
  200 epochs at lr 2e-3. One full-graph encode per epoch keeps it feasible
  on the larger CoDEx-M graph.
- **Ranking**: full-entity filtered protocol, dropping entities already
  known true for the query.

Epoch counts are fixed, so the schedule is identical on every machine and
the recorded numbers are reproducible rather than hardware-dependent. Both
training loops and the ranking step halve their batch and retry on an
out-of-memory error rather than dying. TransE embeddings are saved the
moment that phase finishes, so a kill mid-R-GCN does not discard TransE.

## Calibration

Re-measured end to end on **2026-07-29** on the one-direction query set
against the current 13-criterion verifier, on an M4 laptop (MPS) with the
shipped configuration. The full verifier returns **reward 1.0 and rubric
1.0, all thirteen criteria passing**. Whole run: 76 minutes (TransE 27,
R-GCN 47, ranking 2).

| model | held-out MRR | Hits@10 | validation MRR | held-out / validation |
|---|---|---|---|---|
| TransE | 0.2593 | 0.4261 | 0.2608 | 0.9942 |
| R-GCN | 0.1263 | 0.2572 | 0.1289 | 0.9798 |

On the validation slice the same run gives TransE Hits@1 0.1788, Hits@3
0.2952, Hits@10 0.4229. Halving the query set moved the numbers by under a
point against the previous bidirectional run (0.2659 / 0.1271 / 0.4308),
which is what you would expect from scoring a random half of the same
triples — the leak was an attack available to agents, not something the
oracle was using.

Rank agreement with the saved embeddings is 1.00 for both models under
their required scoring family; the two models disagree on 0.917 of
top-ranked entities. The popularity baseline is 0.0398 (held-out). The
saved R-GCN encoder replays to its own output embeddings at mean row cosine
1.0000 under `in_degree/relu` propagation, with neighbour aggregation
carrying **0.57** of the output norm.

The last two columns are load-bearing: the split-advantage bound in
`results_plausible` assumes an honest model scores about the same on both
halves, and at ratios of 0.9942 and 0.9798 it does.

Relation difficulty (27 relations with >= 30 validation queries): hardest
by mean MRR are `cast_member`, `narrative_location` and `cause_of_death`;
easiest are `languages_spoken_written_or_signed`, `spouse` and `sibling`.
This is the expected shape — relations whose answer is drawn from a small,
tightly constrained set (a person's sibling, spouse, or language) are easy,
while relations whose answer is a large sparse pool (a film's cast, a
work's setting) stay hard. Spearman between per-relation training frequency
and mean MRR is -0.0824, so on this graph how much data a relation has says
almost nothing about how hard it is.

TransE outperforming this R-GCN is normal for CoDEx-M at these budgets; the
R-GCN gate is therefore a floor on the second model being real and trained,
while quality discrimination rests on `transe_mrr` and `transe_hits10`.

## Thresholds

| metric | oracle (held out) | gate | margin |
|---|---|---|---|
| `transe_mrr` | 0.2593 | 0.21 | 24% |
| `rgcn_mrr` | 0.1263 | 0.085 | 49% |
| `transe_hits10` | 0.4261 | 0.35 | 22% |

Three gates must clear simultaneously, so the margins are wider than a
single-metric task would need. A deliberately under-trained baseline (dim
64, 40 epochs) lands well below all three while still passing every
deterministic criterion — the calibration point that proves the task
discriminates on model quality rather than on artifact bookkeeping. Its
measured numbers are recorded in the validation matrix below.

## Anti-cheat

- `lp_rankings_match_embeddings` re-scores a sample of queries from the saved
  entity/relation matrices and requires the submitted top-ranked entity to
  land inside the recomputed top 10 on at least half of them. Each model is
  scored only under the family the instruction names for it — translation
  (L1/L2) for TransE, bilinear (DistMult/ComplEx) for the R-GCN — so shipping
  a second TransE in the R-GCN slot fails. The sample is drawn from **both**
  splits, half from each. Sampling only the graded half would let a
  submission write the shipped validation answers straight into its ranking,
  which matters because `results_plausible` now uses the validation score as
  a reference point.
- `results_plausible` has an absolute and a relative arm.
  - Absolute: held-out MRR at most 0.55, held-out Hits@10 at most 0.85, and
    validation MRR at most 0.55. CoDEx-M does not support numbers near that
    under the filtered protocol.
  - Relative: held-out MRR at most `1.10 x validation MRR + 0.02`. Both
    splits are held out of training and drawn the same way, so an honest
    model scores about the same on each. A fixed ceiling alone only stops a
    naive leaker; it leaves the whole band between the 0.21 gate and the 0.55
    ceiling usable by a leaker who dilutes. The relative arm collapses that
    band to the neighbourhood of the model's own validation score.
- `models_differ` requires the two models to disagree on at least 10% of
  top-ranked entities (oracle: 0.917).
- `lp_above_popularity` compares against the type-free popularity baseline
  computed over every graded query (hidden MRR@20 0.0398).
- `rgcn_encoder_reproduces` replays the saved encoder. It loads
  `rgcn_input.npy` and, for each layer, the basis tensor, the per-relation
  basis coefficients and the self-loop transform, rebuilds the message
  passing over the training graph read from `tests/verifier_data/`, and
  requires the result to land back on `rgcn_entity.npy` (mean row cosine
  at least 0.98, relative Frobenius error at most 0.10). It also requires
  the per-relation coefficient rows to be genuinely distinct.

  The replay does not assume one architecture. It sweeps four variants —
  messages mean-normalised by in-degree or left unnormalised, crossed with
  ReLU or no activation — and keeps whichever reproduces the submission
  best, then ablates using that same variant. The whole sweep costs about
  six seconds against a 900 s budget.

  **Reproduction alone is not enough, and this was a real hole.** Set the
  input embeddings equal to the output, the self-loop to the identity, and
  the bases to about 1e-7: because the honest output is post-ReLU and
  therefore non-negative, `relu(0 + X @ I) == X`, so the replay reproduces
  perfectly while message passing does nothing. Measured on the real graph,
  that spoof scored cosine 1.00000000 at relative error 5.3e-06 and passed.
  So the check now also **ablates the neighbour aggregation** and requires
  the output to move by at least 1% of its norm without it. Measured: a real
  encoder over the real graph moves 0.0893 (dim 128) to 0.1297 (dim 64); the
  spoof moves 5.25e-06. Four orders of magnitude of separation, and the
  threshold sits about 9x below the genuine value.
- Honesty checks recompute every reported number from the prediction files,
  using only agent-visible filtering for the validation-slice checks so an
  honest submission is not penalised.
- `metrics_selfreport_honest` cross-checks `model_config.json` against the
  artifacts rather than only reading it: entity dimension for both models,
  layer count and basis count against the tensors actually saved, and the
  scoring rule each block names against the rule that empirically
  reproduces that model's ranking. Naming L2 while shipping an L1 model, or
  claiming three layers while saving two, fails.

## Residual risks

- **Paired directions (closed).** Every triple used to ship as two queries,
  one asking for the head and one for the tail. The answer to a tail
  question was therefore the source of the paired head question on the same
  relation, and `val_answers.tsv` identifies which half is graded, so the
  candidate set collapsed from 17,050 entities to the sources of the graded
  queries on that relation. Measured on the old file: the gold answer sat
  inside that pool for **20,622 of 20,622** graded queries. A reviewer
  scored 0.2116 MRR and 0.3680 Hits@10 against gates of 0.21 and 0.35
  *without training anything* — restricting to the pool and sorting by
  shared neighbours. The rubric checks would still have caught a submission
  built only on that, but the three gates had stopped measuring model
  quality, and it cut the other way too: a genuinely good model ranking
  inside so small a pool could overshoot the 0.55 ceiling and be zeroed as
  a suspected leak.

  Shipping one direction per triple closes it. Gold-in-pool falls from
  100% to 43.7%, which is ordinary graph overlap rather than a shortcut,
  and the same no-training attack now scores **0.0190 MRR / 0.0359
  Hits@10** — below both gates and below the 0.0398 popularity baseline.

- **Public graph.** `allow_internet` must be true, and CoDEx is public, so
  an agent could re-identify the anonymised ids by label and read the
  held-out triples. The hidden split is exactly the public CoDEx-M test
  split (10,311 triples) and entity labels ship as plain text, so
  re-identification is realistic, not theoretical.

  Injecting answers wholesale fails: the ranking has to reproduce from the
  saved embeddings, and the absolute ceilings catch the result. The
  remaining exposure is a **diluted** leak that lifts only some held-out
  queries. The relative arm of `results_plausible` bounds this, and the
  bound is real but not total.

  The multiplier is now **1.10**, set from measurement rather than caution.
  The oracle run scores 0.2593 held-out against 0.2608 validation for
  TransE and 0.1263 against 0.1289 for the R-GCN — ratios of 0.9942 and
  0.9798, so the two halves agree to within 2% and a 10% allowance is
  ample. At 1.10 the oracle clears its own bound by 16% (TransE) and 28%
  (R-GCN). A leaker starting from an oracle-shaped profile can lift roughly
  6% of held-out queries before tripping it, against about 11% at the
  interim 1.25 and about 38% under the old fixed ceiling alone. The
  exploitable band is roughly 6x narrower than it was — narrow, not gone.
- **Architecture verification** was previously an accepted residual: the
  bilinear reproduction check ruled out a second TransE in the R-GCN slot
  but not an R-GCN versus a plain DistMult with no encoder. That gap is now
  closed by `rgcn_encoder_reproduces`, which re-runs the encoder and
  requires it to reproduce the saved embeddings **and** to depend on the
  neighbour aggregation to do so. To pass, a submission has to ship input
  embeddings and per-relation transforms whose message passing over the
  training graph regenerates its own output embeddings, with the graph
  carrying a non-trivial share of that output — which is training an R-GCN,
  not describing one.

  Two things remain open, both narrower. The replay sweeps four propagation
  variants (messages mean-normalised or unnormalised, crossed with ReLU or
  no activation) and keeps whichever reproduces the submission, so the
  common implementations are covered — but a genuinely unusual encoder
  outside that set would still fail. `/app/data/encoder_protocol.md` states
  which variants are replayable and tells the agent to stay inside them, so
  this is a stated contract rather than something to guess. And the
  ablation threshold is a floor, not a proof:
  an adversary who solves for input embeddings that reproduce a
  decoder-shaped output while letting the graph contribute just over 1%
  would clear it. That fixed point has to be solved through the real
  propagation and still clear `rgcn_mrr`, which is the work the criterion is
  trying to force, but it is a bound rather than an impossibility.

## Local validation

```bash
WORK=$(mktemp -d); LOGS=$(mktemp -d)
OTTER_APP_DIR=$WORK OTTER_DATA_DIR=$PWD/environment/task_inputs \
  bash solution/solve.sh
OTTER_APP_DIR=$WORK OTTER_DATA_DIR=$PWD/environment/task_inputs \
  OTTER_LOG_DIR=$LOGS OTTER_TESTS_DIR=$PWD/tests bash tests/test.sh
cat $LOGS/reward.json
```

`codex_colab_calibration.ipynb` runs the same thing on a Colab T4 and prints
the calibration block; that is how the shipped numbers were produced.

Regenerating the data (needs the CoDEx checkout / raw files):

```bash
python3 scripts/prepare_data.py /path/to/codex_raw .
```

## Validation matrix

Every row below was executed on 2026-07-29 against the **real oracle
artifacts** from the 68-minute M4 MPS run, with the current 13-criterion
verifier. Each scenario copies those artifacts, mutates one thing, and
re-runs `tests/test.sh`.

| scenario | reward | rubric | criteria that failed |
|---|---|---|---|
| real oracle | 1.00 | 1.000 | none |
| **encoder spoof** (input := output, self := identity, bases := 1e-7) | 0.00 | 0.900 | `rgcn_encoder_reproduces` alone |
| **DistMult in the rgcn slot**, no encoder shipped | 0.00 | 0.800 | rank agreement, `rgcn_encoder_reproduces` |
| **diluted leak**, 25% of held-out lifted to rank 1 | 0.00 | 0.925 | `results_plausible` alone |
| **scored answer filtered out of validation predictions** | 0.00 | 0.650 | `results_plausible`, `lp_results_honest`, `relation_analysis_consistent`, `metrics_selfreport_honest` |
| R-GCN predictions copied from TransE | 0.00 | 0.575 | rank agreement, `models_differ`, honesty x3 |
| self-report inflated (`metrics.json` only) | 0.00 | 0.925 | `metrics_selfreport_honest` alone |
| empty workdir | 0.00 | 0.075 | 12 of 13, clean exit, no verifier crash |

The first three bold rows are exploits raised in review; each now fails,
and two of them on **exactly one** criterion — the one written to catch
them — rather than collaterally, which is what tells you the check is doing
its own work.

The fourth bold row is not an attack. It is an honest agent following the
old instruction wording, which told it to drop every entity the data marks
as a true answer — including, for the answered queries, the answer itself.
That zeroed its validation MRR and took down four criteria including a
critical one. The oracle quietly kept the gold, so nothing caught it. The
wording now says the scored answer stays in and only the other known-true
entities come out, which is what the verifier has always done. This row
stays in the matrix as a regression test on the wording.

3,000 held-out triples appended to the agent-visible `valid.tsv` do not
move the verdict — the structural guarantee is that `checks.py` contains
zero references to `/app/data`, so nothing the agent can edit feeds
grading. The two load-bearing rows are the weak baseline (proves the task
discriminates on model quality, not bookkeeping: every deterministic
criterion passes, only the gates fail) and the empty workdir (proves the
verifier degrades rather than crashes).

### Encoder criterion, verified separately

`rgcn_encoder_reproduces` was validated on its own against the real
CoDEx-M training graph (17,050 entities, 51 relations, 102 relation
directions), driving the shipped `RGCN.encode` in eval mode and replaying
its parameters through the verifier. This exercises the real graph and the
real propagation code; the encoder is untrained, which the criterion does
not depend on. Replay agreement: relative Frobenius error 2.6e-07, mean row
cosine 1.00000000 at the oracle's `RGCN_DIM` of 128. Replay cost is a few
seconds, against a 900 s verifier budget.

| scenario | verdict |
|---|---|
| genuine R-GCN artifacts | pass |
| **plain DistMult in the rgcn slot, no encoder shipped** | **fail** — no encoder parameters found |
| **input := output, self-loop := identity, bases := 1e-7** | **fail** — aggregation carries 5.25e-06 of the norm |
| encoder directory removed | fail — no encoder parameters found |
| `rgcn_input.npy` removed | fail — input embeddings missing |
| one transform shared across all relations | fail — replay does not reproduce |
| output embeddings swapped, encoder kept | fail — replay does not reproduce |
| second encoder layer dropped | fail — replay does not reproduce |

Rows two and three are the two ways a decoder-only model has been shown to
reach the R-GCN slot. Row three passed reproduction at cosine 1.00000000
before the ablation arm was added.

### Leak bound, verified separately

`check_results_plausible` was exercised against constructed ranking files
with an oracle-shaped profile (55% of queries with no gold inside the top
20, giving MRR near 0.26 and Hits@10 near 0.43 on both splits). Only the
ranking files vary; the graded answers are the real ones.

| scenario | held-out MRR | validation MRR | verdict |
|---|---|---|---|
| honest, both splits alike | 0.262 | 0.260 | pass |
| diluted leak, 20% of held-out to rank 1 | 0.407 | 0.260 | fail — relative arm |
| diluted leak, 30% of held-out to rank 1 | 0.481 | 0.260 | fail — relative arm |
| naive leak, all held-out to rank 1 | 1.000 | 0.260 | fail — absolute ceiling |
| validation stuffed to mask the ratio | 0.481 | 1.000 | fail — validation ceiling |

The last row is the obvious counter-move against a relative bound, which is
why the validation ceiling and the two-split rank-agreement sample are part
of the same fix rather than separate polish.

## Review history

The dataset swap (round 5) resolves the standing licence blocker: KG20C was
research-use-only, which the guide does not allow, so the graph was replaced
with the openly licensed CoDEx-M. Earlier rounds (on KG20C) closed a
verifier hole where grading read the agent-editable `/app/data`, switched
the oracle to fixed epoch counts, fixed the validation-slice honesty check
to filter only with agent-visible triples, removed a template-generated QA
half, and set `gpu_types`. Those fixes carry into this task unchanged; the
only new surface is the graph and the wider per-relation breadth.

Round 6 answers a Harbor trial that failed on a build-time
`ConnectionError` and a robustness review that recorded two
`weak_assertion` findings.

- **Build.** Dependencies now install from `requirements.lock` with
  `--require-hashes --no-deps`, matching the other tasks in this family.
  The previous plain `pip install -r requirements.txt` let the resolver
  walk the transitive tree of `torch_geometric` at build time, which is
  the widest network surface in the image and the most likely source of
  the failed connect. Every hash is carried over from the verified lock of
  an already-built task rather than generated here. `networkx` was dropped:
  nothing imports it and no verified hash was available for it.
- **Finding 1, the R-GCN slot.** Closed by `rgcn_encoder_reproduces`; see
  Anti-cheat and the encoder matrix above.
- **Finding 2, `model_config.json`.** `metrics_selfreport_honest` now
  cross-checks the config against the artifacts on several axes instead of
  the transe dimension alone; see Anti-cheat.

Round 7 answers a human review that found two gameability holes the
automated ROBUST pass missed, both reproduced here before being fixed.

- **The encoder replay was spoofable by a decoder-only model** — the exact
  thing the criterion exists to prevent. Reproduced at cosine 1.00000000.
  Fixed by ablating the neighbour aggregation and requiring the output to
  depend on it. See Anti-cheat and the encoder matrix.
- **The anti-leak defence was ceiling-only**, leaving the band between the
  0.21 gate and the 0.55 ceiling usable by a diluted leaker. Fixed by
  bounding held-out MRR against the submission's own validation MRR, plus a
  validation ceiling and a two-split rank-agreement sample so the reference
  point cannot be inflated. See Anti-cheat and the leak table.

Round 8 fixes a failed Harbor build. The static checker rejected
`instruction.md` on `not_over_prescribed`: round 7's encoder paragraph
spelled out the propagation formula, layer count, normalisation, activation
and dropout, which is the *how* that `task-advice1` says never to put in an
instruction. It was written that way because the verifier replays the
encoder and needed a known rule.

Resolved by removing the need for the rule rather than by hiding it.

- The replay now sweeps four ordinary propagation variants and keeps
  whichever reproduces the submission, so the task no longer dictates one
  encoder. Verified that this does not weaken either round-7 fix: the
  spoof and all six negative scenarios still fail, the genuine encoder
  still reproduces at cosine 1.00000000.
- The serialisation contract moved to `/app/data/encoder_protocol.md`,
  which ships in the image and is agent-visible. This is also the
  `PROTOCOL.md` split the round-7 review asked for. `instruction.md` is
  down to 754 words from 912 and now states the requirement and the two
  properties checked, not the algorithm.

Round 10 states the three pass thresholds verbatim in `instruction.md`
(`transe_mrr` 0.21, `rgcn_mrr` 0.085, `transe_hits10` 0.35). The static
checker requires it and blocks the build otherwise.

This is a real contradiction with the guidance, worth recording so the next
round does not quietly undo it. `task-advice1` section 7 says an
instruction should give no hint of "where the difficulty lives", and none
of the sibling tasks in this family print their gate numbers — the closest
is a prose warning that the bar is hard. The checker disagrees and it
blocks, so per `task-advice2` section 9 the checker wins. The wording keeps
the damage small: it names the bar and says the numbers are recomputed on
the graded half, while the agent's own `metrics.json` covers the answered
half, so the thresholds read as a target rather than as a map of where the
task is hard. Difficulty is unaffected — no gate moved, and agents were
already failing on model quality rather than on not knowing the number.

Round 11 answers a review that found a structural leak and an instruction
bug, both reproduced here before being fixed.

- **Paired directions.** Every triple shipped as two queries, so the answer
  to one was the source of the other. Measured on the old file, the gold
  sat in the same-relation source pool for **20,622 of 20,622** graded
  queries, and the reviewer scored 0.2116 MRR / 0.3680 Hits@10 against
  gates of 0.21 / 0.35 with no training at all. Fixed by shipping one
  direction per triple: gold-in-pool drops to 43.7% and the same attack now
  scores **0.0190 / 0.0359**, below both gates and below the popularity
  baseline. The oracle was re-run from scratch on the new query set and
  still returns reward 1.0; the metrics moved by under a point, confirming
  the leak was an agent-side shortcut rather than something the oracle
  relied on.
- **Filtering wording.** The instruction told the agent to drop every
  entity the data marks as a true answer, which for the answered queries
  includes the answer being scored. An agent doing exactly that zeroes its
  validation MRR and fails four criteria including a critical one — now a
  row in the validation matrix. The verifier always kept the gold and the
  oracle did too, so only the wording was wrong. It now says the scored
  answer stays in and the other known-true entities come out.
- **Report number matching.** `check_report_content` matched values at two
  or three decimals while the shipped report prints four, so it only lined
  up when the last digit happened to round down. It now accepts two, three
  or four decimals and two percentage forms.
- **model_config keywords.** The check required two keywords per model from
  a hard-coded vocabulary the instruction never mentions. Removed. What
  remains is what the instruction actually asks for: a non-empty block per
  model with at least three settings, an integer layer and basis count for
  the rgcn block, and the artifact cross-checks on dimension, layer count,
  basis count and scoring family.

Note on the split-advantage bound: it assumes the agent keeps the scored
answer in its ranking, which is now stated plainly. The bound is
deliberately not relaxed for submissions whose validation score collapses —
softening it there would hand back the diluted-leak band to anyone willing
to tank their own validation numbers.

Static-check warnings accepted deliberately, per `task-advice1` section 6:
unpinned apt packages (pinning blind against an image that cannot be built
locally trades a warning for a hard failure) and the non-slim base image
(GPU work). Both are listed there as acceptable with a note.

Round 9 is the oracle re-run itself, and it found a real bug.

`lp_rankings_match_embeddings` returned agreement 0.00 for both models, so
**the oracle failed**. Round 7's two-split sampling loop bound its loop
variable to `ids`, shadowing the entity list that the very next guard
compares against — `ent.shape[0] != len(ids)` became 17,050 vs 20,620 and
the function bailed out before scoring anything. A one-word rename fixed
it; agreement is back to 1.00 for both models. It is worth naming plainly:
three rounds of local scenario tests all passed while the oracle was
broken, because every one of them called the new checks directly and none
ran the whole verifier against real artifacts end to end.

With that fixed the oracle scores **reward 1.0, rubric 1.0, all thirteen
criteria green**, and the run supplied the two numbers earlier rounds had
to guess at:

- Neighbour aggregation carries **0.57** of the trained encoder's output
  norm, against 0.089 untrained and a 0.01 threshold. The floor is 57x
  clear on the real model, so it is conservative rather than lucky.
- Held-out and validation MRR agree closely (ratios 1.0153 and 0.9837 on
  that run), so the leak multiplier came down from 1.25 to **1.10** as the
  round-7 notes said it should if the evidence supported it. Re-verified at
  the tighter setting: still reward 1.0.

Static-check warnings accepted deliberately, per `task-advice1` section 6:
unpinned apt packages (pinning blind against an image that cannot be built
locally trades a warning for a hard failure) and the non-slim base image
(GPU work). Both are listed there as acceptable with a note.

Still open, deliberately not changed: the instruction is duplicated into
`task.toml`'s `description`, and `gpu_types` declares A100 while these
numbers come from an M4 laptop (MPS) and an earlier Colab T4. The two agree
closely, and epoch counts are fixed so the schedule is hardware-independent,
but the declaration still does not match the hardware any recorded run used.
Either run the oracle on an A100, or set `gpu_types = ["T4"]`, which is what
`task-advice2` recommends when calibrating on Colab.
