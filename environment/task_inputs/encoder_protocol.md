# Encoder replay protocol

The R-GCN you put in the `rgcn` slot is checked by re-running its encoder from
the parameters you save and comparing the result with the entity embeddings you
save. This file describes what that replay reads and what it does, so that a
working encoder is not failed on a file-layout detail.

The architecture is your call. This is a serialisation contract and a list of
propagation variants the replay knows how to reproduce — not a required design.

## Files to save

All arrays are `.npy`, float32 or float64, rows in `entities.tsv` order where an
entity axis is involved.

| path | shape | contents |
|---|---|---|
| `/app/artifacts/embeddings/rgcn_input.npy` | n_entities by dim_in | the entity embeddings that go **into** the encoder |
| `/app/artifacts/embeddings/rgcn_entity.npy` | n_entities by dim_out | the entity embeddings that come **out** of it, the ones your ranking uses |
| `/app/artifacts/encoder/bases_layer0.npy` | n_bases by dim_in by dim_out | layer 0 basis matrices |
| `/app/artifacts/encoder/coeff_layer0.npy` | 2 x n_relations by n_bases | layer 0 basis coefficients, one row per relation per direction |
| `/app/artifacts/encoder/self_layer0.npy` | dim_in by dim_out | layer 0 self-loop transform |

Number the layers from 0 upward, `bases_layer1.npy` and so on, up to four
layers. All three files must be present for every layer you ship. Coefficient
row `i` is relation `i` of `relations.tsv` in the head-to-tail direction; row
`i + n_relations` is the same relation tail-to-head.

Relation transforms are expressed in basis-decomposed form: the transform for
one relation-direction is the sum over bases of its coefficient times that basis
matrix. If your encoder holds a full matrix per relation-direction rather than a
basis decomposition, save it as a one-basis decomposition — `n_bases` of 1, with
the coefficient rows carrying the scaling.

## What the replay does

For each layer, starting from `rgcn_input.npy`:

1. Every training triple sends a message from head to tail, and another from
   tail to head.
2. A message is the source entity's current row put through the transform for
   that relation and direction.
3. Messages arriving at an entity are summed.
4. The entity's own current row through the self-loop transform is added.
5. An activation closes the layer, and the result feeds the next layer.

The output of the last layer is compared with `rgcn_entity.npy`. It has to match
to a mean row cosine of 0.98 with relative Frobenius error no worse than 0.10.

## Variants the replay tries

Steps 3 and 5 are where implementations differ, so the replay tries each
combination below and keeps whichever reproduces your embeddings best. You do
not have to declare which one you used.

- **Message normalisation**: divide each message by the number of messages
  arriving at that destination for that relation and direction, or leave
  messages unnormalised.
- **Activation**: ReLU, or none.

Two consequences worth knowing. Turn dropout off for the copy you save, since
the replay has no way to reproduce a random mask. And if you use a propagation
rule outside these variants, the replay will not reproduce your embeddings and
the check will fail even though your encoder is sound — so either stay inside
them or save a copy of the encoder output produced by a run that does.

## The ablation

The replay is run a second time with step 1 removed, so only the self-loop path
survives. The output has to move by at least 1% of its norm between the two
runs. This is what separates an encoder from a decoder with an inert encoder
attached: an input equal to the output, an identity self-loop and near-zero
bases reproduce perfectly while the graph does nothing, and that does not pass.
Let the message passing carry real weight.
