# SOMA-CELL 0.6.8-GPU A4 contract

Status: A4 development contract through slice A4.5a. This is not an A4
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

## A4.4b scope

A4.4b retains A4.4a's pre-existing active-template, mutation-free,
non-completing pure plan and adds only its deterministic frozen controls:
proofreading, inherited quiescence, behavioural quiescence, and the external
replicase contribution.  It remains neither a scheduler replacement nor a
CPU-cell commit.  A supported row has a non-empty active template, a shorter
partial copy, at least one complete genome, effective replicase above the
frozen gate, nonnegative ATP at replication entry, and remains incomplete
after the literal resource gates are applied.  Every derived fp64 value that
controls a discrete replication branch must also remain outside the registered
ambiguity bands described below; a row inside a band remains CPU-authoritative.

The supported configuration requires genome replication on and mutation off.
Proofreading, external replicase, inherited quiescence, and the behavioural
quiescence effector may independently be on or off.  The two quiescence flags
must exactly match the attested physiology snapshot.  Template selection has
already occurred.

Within that boundary the plan reproduces the Formal066 wrapper and frozen 0.4
operation order: scale `dt` by `max(0.25, eco66_replication_rate_scale)`, fold
endogenous replicase in active-protein dictionary order, then add `0.85` for
an enabled external replicase.  Proofreading and quiescence repair signals use
only repair-localized regulator proteins, literal `(mass * efficiency) *
promoter` grouping, active-dictionary left folds, and the frozen aggregate
inhibition.  Inherited and behavioural quiescence are combined in the same MRO
order as Formal066.  These deterministic factors scale speed before it is
added to the fractional carry.  The full integer request is removed before
resources are checked.

CPU and CUDA fp64 division are permitted to differ by a few ulps even when
both follow the same expression order.  Because truncating that result can
turn a continuous difference into a different DNA-symbol count, A4.4b does
not guess at an integer boundary.  For a positive progress increment it marks
the row out of scope when
`abs(fractional_total - rint(fractional_total)) <=
4096 * eps64 * max(1, abs(fractional_total))` and the nearest integer is at
least one.  Nonfinite increments or totals are rejected by the same scope
code.  A zero
increment remains a valid no-op.  This narrow engineering guard preserves the
CPU oracle without adding software fp64 division or silently accepting a
different discrete result.

The same factor supplies a relative comparison band,
`4096 * eps64 * max(abs(left), abs(right))`, around the effective-replicase
`> 1e-6` gate and the proofreading-derived ATP payment gate.  The ATP band is
needed only when proofreading is enabled; the proofreading-free constant gate
remains exact.  The nucleotide comparison uses the unchanged resident input
and exact monomer constant and therefore remains outside this derived-value
guard.  If an ATP ambiguity appears after an earlier symbol was tentatively
planned, the entire row is rolled back to its input pools, telemetry, and empty
append plan before it is marked out of scope.

Accepted symbols then consume one exact `MONOMER_MASS` and
`0.0012 + 0.00075 * proof_fraction` ATP, while retaining the frozen `0.022`
ATP gate reserve.  Unpaid requests are not restored.  The input physiology
now carries `cumulative_proofreading_atp`; the plan returns
`cumulative_proofreading_atp_after` and adds the extra proofreading ATP once
per accepted symbol in the same fixed symbol loop as the CPU.  It is not
reconstructed later as `before + aggregate_delta`, which could change fp64
rounding.

Although mutation is disabled, the observable effective error rate is updated
from configured mutation rate, template lesion, reactive concentration, and
proofreading reduction.  RNG state, world dissipated energy, complete genomes,
lesions, and the gene cache remain unchanged.  Inputs remain immutable and the
fixed output carries only the append plan, paid pools, fractional progress,
and replication telemetry needed for a later atomic commit.  A signed ATP
residual permitted at the preceding translation boundary is explicitly
outside this replication slice; it is rejected rather than fed into rate
math.  A successful plan therefore requires nonnegative ATP on output as
well; only unchanged fuel/mineral translation residuals retain their
registered signed ledger tolerance.

NumPy rejects an out-of-scope row before returning a plan.  The CUDA path does
not perform a hidden scalar readback; it carries a fixed scope-valid/error-code
result which must be rejected at the explicit readback or future commit
boundary.  Error code 6 identifies every fp64 discrete-boundary guard.  An
unsupported row is never a successful replication result, and the remaining
resident fields of an invalid plan are not commit authority before explicit
readback and full validation succeed.

## A4.5a scope

A4.5a overlays scalar PCG64 substitution decisions on the already validated
A4.4b schedule.  Its boundary is deliberately narrow: the template is already
active, the paid copy remains incomplete, and mutation is enabled only for
same-length per-symbol substitution.  Inactive-template start, completion,
structural/material mutation, hydrolysis, CPU-cell commit, and scheduler
replacement remain outside this slice.

Tape preparation takes an explicit canonical NumPy PCG64 state and clones it;
it never advances the live world RNG.  In cell-input order and then accepted
symbol order it performs exactly one scalar `random()` call per paid append,
including when effective error is zero.  Only a strict `uniform <
effective_error` hit immediately performs scalar `integers(0, 7)`.  Resource-
blocked requests consume no draw.  The bounded integer call is kept as a
NumPy high-level call so its internal rejection and `has_uint32`/`uinteger`
cache behavior are not guessed.

`A4SubstitutionRngTape` is fixed-shape, private-factory, event-local data.  It
carries draw/replacement prefixes, per-row counts and effective errors, full
PCG64 before/after states, source/config/dt/schedule digests, and immutable
scalar plus resident pointer/version attestations.  Host validation replays
the complete scalar call sequence and proves the after-state exactly.  The
tape has no deserialization, save, or live-RNG authority.

The resident plan applies the CPU-resolved replacement mask; it does not use a
second Torch RNG.  It nevertheless recomputes the deterministic schedule and
checks cell identity, append counts, effective-error bits, and every recorded
decision on device.  A draw within the registered 4096-epsilon comparison
band is code 6.  A nonambiguous schedule or decision mismatch is code 7.
Because all rows share one ordered PCG64 stream, either failure rolls back and
invalidates the complete used batch, not only the row where it was detected.
This also prevents a valid tape made from a different physiology snapshot but
the same ragged genome from being reused.

The returned `rng_after_state` is evidence for a later atomic commit only.
A4.5a leaves the actual world RNG unchanged.  It is a direct-call oracle, not
a scheduler integration: the current world step interleaves other per-cell RNG
events between replication calls, so a batched tape cannot be installed as
world authority until that global order is represented transactionally.

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
- Ragged, physiology, and cache scalar metadata (schema, capacities, used
  counts, host flags, and provenance) are snapshotted at bind and must remain
  exactly unchanged for the binding lifetime; this check requires no D2H.
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

Replication-elongation-plan invariants are:

- Every used row is bound to the same attested ragged/cache/physiology source.
- Template and partial-copy order are derived from the A4.1 sequence layout;
  no genome, template, or copy is sorted or sliced.
- Every supported append symbol equals the next template byte exactly.
- Fractional progress consumes the entire integer request before material
  gates, matching the frozen CPU even when no symbol can be paid.
- Nucleotide and ATP are subtracted in symbol order with no free DNA, refund,
  clipping, or contribution to world dissipated energy.
- Exact fixed capacity passes; a required additional symbol beyond capacity
  is rejected before any commit.
- The plan is pure: input ragged/cache/physiology, CPU cells, world, RNG, and
  scheduler receipts remain unchanged.
- Template selection, completion, or any enabled postponed mechanism makes the
  row explicitly unsupported; it cannot silently fall through to a partial
  GPU result plus a second CPU replication call.

## Explicit exclusions through A4.5a

- No scheduler/world integration or CPU-cell protein/material commit.
- No inactive-template start, completion transaction, new complete genome,
  lesion inheritance, gene-cache refresh, structural/material mutation,
  symbol hydrolysis, or device RNG kernel.
- No scheduler replacement and no change to A3 `gene_refresh`,
  `translation_cpu`, or `replication_cpu` authority.
- No division, death, corpse/eDNA/HGT, neural, or causal-system port.
- No fp32 claim, mixed precision, `torch.compile`, CUDA Graph, Triton, custom
  CUDA, multi-stream, multi-GPU, or online-GPU abstraction.
- No formal 13-spec/65-measurement benchmark and no speedup claim.

## Acceptance through A4.5a

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

- A pre-existing active-template, non-completing Formal066 CPU fixture matches
  the NumPy plan for appended bytes/count, nucleotide and ATP pools,
  fractional carry, `last_replication_symbols`,
  `last_effective_error_rate`, and the sequential
  `cumulative_proofreading_atp_after` value.
- Requested-zero, exact and immediately-below nucleotide/ATP gates, and
  resource exhaustion preserve the frozen ordered behavior.
- Proofreading on/off, inherited and behavioural quiescence, endogenous plus
  external replicase, distinct complete-genome/template lesion signals, and a
  nonzero large cumulative proofreading value match direct Formal066 calls.
- CPU oracle RNG state is byte-exact before/after because the supported slice
  consumes zero random draws.
- NumPy, Torch CPU, and explicit RTX CUDA fp64 plans agree after one explicit
  readback; all source pointers and values remain unchanged.
- Scope and capacity failure are atomic and fail closed.  Completion,
  inactive-template start, mutation, or negative replication-entry ATP is not
  reported as migrated.
- Two crafted CPU/CUDA integer-boundary cases plus exact/nextafter
  proofreading-ATP comparisons are rejected with scope code 6 on NumPy,
  Torch CPU, and RTX CUDA, while controls displaced by `1e-10` remain
  supported with the same requested/payment result.  The replicase comparison
  guard is also exercised, and the NumPy pairwise fold used for membrane
  damage is checked bit-exactly on both Torch devices.
- A4.4b leaves A3 `replication_cpu` authoritative and makes no speed claim.

- A mutation-enabled, active, non-completing Formal066 fixture matches the
  A4.5a suffix bytes, substitution-event counts, and full PCG64 after-state.
- Effective error zero still consumes one threshold draw per paid symbol;
  only hits consume the immediate bounded-integer call, and resource stops
  consume no later draw.
- NumPy, Torch CPU, and explicit RTX CUDA apply the same attested fixed tape
  without changing the source arena, physiology, cache, tape storage, or live
  RNG.
- Forged/tampered scalar, PCG64, tail, config, dt, and source schedules fail
  closed. A foreign-physiology tape is code 7; an on-device decision boundary
  is code 6; both invalidate and roll back the whole used batch.
- All 28 A4 development tests and the focused A3 regressions pass while A3
  `replication_cpu` remains the only scheduler authority.

Known A4.4b integration blockers are recorded rather than hidden.  On the
measured six-cell development fixture the current fixed symbol-rank Torch plan
was about 503 ms per call versus about 10.1 ms for NumPy, so it is not a
performance candidate.  A4.4b replaces branch-critical fp64 reductions with
host-constant device loops and is validated with deterministic algorithms
enabled on the recorded PyTorch/CUDA environment, but this is a correctness
result rather than a performance improvement or a promise for untested older
PyTorch releases.  The 4096-epsilon discrete guard is the next power of two
above the worst normalized distance (2987.54) observed in 50,000
boundary-focused, up-to-639-symbol proofreading-payment recurrences.  It is a
conservative measured engineering bound, not a proof over every possible
real-valued input; rows in
its ambiguity band deliberately retain CPU authority.  Exact software
division was rejected as disproportionate complexity.  The launch-heavy path
must be redesigned and remeasured before scheduler authority, promotion, or
any speedup claim.

Inactive-template start, completion/structural mutation, and hydrolysis remain
separate later slices. Scheduler replacement remains later,
after a contiguous resident chain can commit without recreating A3's per-cell
host/device round trips.
