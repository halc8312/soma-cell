# SOMA-CELL 0.6.8-GPU A4 contract

Status: A4 development contract through slice A4.3. This is not an A4
promotion.

## Authority

- The promoted A3 checkpoint remains the engineering baseline.
- Frozen SOMA-CELL 0.6.6 CPU behavior remains the scientific oracle.
- `full_gpu_world_step` remains `false`.
- A4 may change representation and execution location, but not biology,
  material costs, event order, RNG order, or insertion order.

## A4.1 retained foundation

A4.1 packs one world's cells into one fixed-capacity ragged genome arena. It
preserves complete genomes in cell/list order, active replication template and
partial copy, and the exact transient genome-lesion vector. The arena uploads
once to an explicit Torch CPU/CUDA device and is read back only at an explicit
boundary. CPU reconstruction validates the full batch before the first commit.

The sequence order within one cell remains:

```text
complete genome 0 ... complete genome N-1,
replication template (only when active),
replication copy (only when active, including length zero)
```

The template is a reference copy, not additional material. Material symbols
remain `complete genomes + partial replication copy`.

## A4.2 scope

A4.2 adds one pure derived cache and no world/scheduler integration:

1. Decode delimited genes from complete genomes in one resident batch.
2. Exclude replication template and partial copy from gene-cache authority.
3. Match frozen `parse_genes`: scan left to right, advance one symbol after an
   invalid candidate, and advance exactly 16 symbols after a valid gene.
4. Fold occurrences in cell -> complete-genome-list -> local-start order.
5. Preserve the first record for each `(cell, fingerprint)` and increment only
   its `copy_number` for later duplicates.
6. Preserve all 12 payload symbols, the first local start, exact base-8
   fingerprint, and copy count. Host materialization deterministically restores
   every frozen `gene_specs` field and adds reaction fields only for generic
   genes.

The Torch path uses a fixed candidate matrix, a host-constant maximum of
`floor(max_sequence_symbols / 16)` greedy rounds, and fixed-shape stable group
and compaction buffers. It does not call `.item()`, `.cpu()`, `.numpy()`,
`nonzero`, masked selection, or variable-length `unique` inside the decoder.
All Torch outputs remain on the input device until one explicit
`A4GeneCacheBatch.to_numpy()` readback.

`A4GeneCacheBatch` is disposable derived data. `A4RaggedGenomeBatch` remains
the sole genome authority; no second world state or save format is introduced.
The cache has no deserialization constructor. A4.3 consumes only a cache
produced directly from the same validated resident arena, not an external or
restored cache whose biological values cannot be proven without that source.

## A4.3 scope

A4.3 adds one pure, fixed-shape paid-translation plan. It does not replace the
A3 scheduler or mutate a CPU cell. `bind_a4_translation()` decodes its cache
internally from the exact ragged object, binds that cache to a one-time
physiology/protein snapshot, and returns no serializable authority. NumPy and
Torch plans return an updated `A4TranslationStateBatch`; inputs, RNG, cells,
world, and scheduler receipts remain unchanged.

The plan reproduces the actual Formal066 dynamic dispatch, not only the A3
single-cell test helper:

- 0.3 paid translation, quiescence, misfolding, and protein sync thresholds;
- 0.5 ecology-localized regulator needs;
- 0.4 sensor/effector needs, repair-localized chaperone activity, and
  behavioural quiescence;
- P0 neural-attachment osmolyte contribution to inherited membrane tension;
- 0.2 non-regulator protein needs and material stoichiometry.

The neural attachment contribution is an ordered, read-only CPU-authoritative
snapshot. A4.3 does not move neural dynamics, tissue material, or attachment
ownership to the device.

All weights observe the pre-translation state. Protein payment then follows
gene-cache order, preserves the ATP reserve `0.042`, and consumes exactly
`0.64 fuel + 0.36 mineral + 0.52 ATP` per translated unit. Active and damaged
protein dictionaries retain their independent insertion orders. Existing keys
keep their positions; new keys append in gene-cache order only when the final
mass survives the frozen active `>1e-10` or damaged `>1e-11` sync threshold.
The Torch path uses fixed arrays, a host-constant gene-rank loop batched across
cells, and scatter to the prevalidated fixed protein capacity. The loop keeps
the frozen per-gene fp64 payment recurrence instead of substituting a
closed-form prefix budget at exhausted-resource boundaries. It performs no
scalar readback or variable-length output.

## Fail-closed invariants

Ragged foundation invariants remain unchanged:

- Symbols are integer values in the frozen alphabet `[0, 7]`.
- Used offsets start at zero, are monotonic, and terminate at the used count.
- Unused offsets are `-1`; unused symbols/material values are zero.
- Cell IDs/order are preserved; duplicate IDs are rejected.
- Per-cell sequence count is `complete_genomes + 2 * active_replication`.
- Active template is non-empty and partial copy cannot exceed it.
- Inactive replication has no copy, template lesion, or fractional progress.
- Lesion vector length cannot exceed complete-genome count.
- Exact capacity passes; capacity + 1 fails before allocation or commit.

Gene-cache invariants are:

- Entry order is the frozen first-insertion order; fingerprint sorting is not
  externally observable.
- `cell_capacity <= 134217727`; this keeps `(cell, fingerprint)` composite
  int64 keys strictly below the reserved `INT64_MAX` sentinel.
- Entry mask is a true prefix; cell entry offsets terminate at entry count.
- Payload/fingerprint relation is recomputed and checked exactly.
- Fingerprints are unique within each cell cache; copy counts are positive.
- Unused fingerprint/start values are `-1`; payload/copy tails are zero.
- Capacity is derived, not auto-grown:
  `min(symbol_capacity // 16,
       sequence_capacity * (max_sequence_symbols // 16))`.
  Accepted genes are non-overlapping 16-symbol intervals, so valid input cannot
  exceed this bound. No record is clipped.
- NumPy decode fully validates its source. Torch decode accepts only the
  metadata/dtypes/shapes of an already validated upload and does not perform a
  hidden diagnostic D2H. Full value validation occurs before upload and after
  explicit readback.
- Decode is pure: source arena, CPU cells, RNG, scheduler receipts, material
  pools, and protein dictionaries are unchanged.

Translation-state invariants are:

- Cell IDs/order and the derived genome count, lesion mean, replication flag,
  and material-symbol count agree with the bound ragged arena.
- `source_provenance` is a lowercase 64-hex SHA-256 of the fully validated
  NumPy ragged source. Host binding recomputes it; resident binding accepts the
  carried digest only while both ragged and physiology tensor pointers and
  version counters still match their trusted upload or trusted clone. A
  resident clone verifies its source attestation before re-attesting new
  storage, so mutation followed by clone cannot mint new trust.
- The internally decoded cache is attested for the binding lifetime: host
  arrays use a canonical SHA-256 and resident tensors use pointer/version
  counters. Mutation between bind and plan is rejected.
- Neural attachment osmolyte is the ordered sum of the current CPU-authoritative
  attachment states. It is a translation input only, not neural state ownership.
- Active and damaged protein rows are independent true prefixes in their
  original dictionary insertion orders; fingerprints are unique per mapping.
- Catalyst and damaged-protein pools equal their ordered mapping sums.
- The three paid pools (fuel, mineral, ATP) may retain a signed fp64 payment
  residual no smaller than `-2e-12`; every other pool remains nonnegative and
  larger negative biology is rejected. This tolerance preserves the frozen
  sequential subtraction result and is not free material or clipping.
- Packing proves both `existing keys union current gene fingerprints` fit the
  declared per-cell protein capacity. Capacity is never grown on device.
- The cache is created inside the binding from the same ragged arena; restored
  or caller-supplied cache data is not accepted.
- `last_translation` always resets when the operation is invoked.
  `last_quiescence` changes only after the translator/gene/positive-weight
  gates, matching the frozen early-return boundary.
- A4.3 output is a pure plan. It is not applied to the A3 world and cannot run
  beside the current `translation_cpu` event.

## Explicit exclusions through A4.3

- No scheduler/world integration or CPU-cell protein/material commit.
- No replication, proofreading, mutation, symbol hydrolysis, or RNG kernel.
- No scheduler replacement and no change to A3 `gene_refresh`,
  `translation_cpu`, or `replication_cpu` authority.
- No division, death, corpse/eDNA/HGT, neural, or causal-system port.
- No fp32 claim, mixed precision, `torch.compile`, CUDA Graph, Triton, custom
  CUDA, multi-stream, multi-GPU, or online-GPU abstraction.
- No formal 13-spec/65-measurement benchmark and no speedup claim.

## A4.3 acceptance

- All A4.1 lossless/corruption/capacity/residency tests remain PASS.
- Nested valid markers and invalid-outer/valid-inner recovery match frozen
  greedy parsing exactly.
- Multiple complete genomes preserve first-insertion order and copy counts;
  active template/copy genes are excluded.
- Full host `gene_specs`, including 0.6.5 grammar payload and conditional
  reaction keys, match the frozen CPU oracle.
- NumPy, Torch CPU, and RTX CUDA fixed-shape outputs have exact discrete state
  and agree within the registered fp64 tolerance after explicit readback;
  decoder input pointers and values do not change.
- Exact derived entry capacity passes; the next complete gene is rejected at
  the enclosing fixed symbol/sequence boundary.
- Focused A3 regressions remain PASS and `full_gpu_world_step=false`.

- Direct `Formal066ProtoCell.translate` comparison covers rich material,
  exact ATP reserve, mid-order material exhaustion, no-translator early return,
  repair/sensor/effector/ecology regulators, behavioural quiescence, external
  translator weighting, and a real P0 neural attachment osmolyte contribution.
- Pools, `last_translation`, conditional `last_quiescence`, active/damaged
  amounts, and both dictionary orders match the CPU oracle in fp64.
- NumPy, Torch CPU, and RTX CUDA plans agree after one explicit readback;
  resident ragged/cache/physiology storage remains unchanged.
- Exact protein-union capacity passes and capacity + 1 fails before upload.
- A4.3 does not alter A3 `translation_cpu`, replication authority, or the
  promoted baseline.

The next slice may add replication/proofreading/material mutation state and
RNG planning. Scheduler replacement remains later, after a contiguous resident
chain can commit without recreating A3's per-cell host/device round trips.
