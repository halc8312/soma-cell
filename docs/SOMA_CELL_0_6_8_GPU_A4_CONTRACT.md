# SOMA-CELL 0.6.8-GPU A4 contract

Status: A4 development contract through slice A4.8c7. This is not an A4
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

## A4.5b scope

A4.5b extends only the mutation-enabled A4.5a substitution-tape path to the
frozen inactive-template start case with exactly one non-empty complete genome.
The public mutation-free deterministic elongation path remains unchanged and
CPU-authoritative for inactive starts.  Rows with zero or two-or-more complete
genomes, rows below the replicase gate, and rows which complete in the same
call remain explicit scope failures rather than ordinary no-op integration.

After the frozen early gates, an eligible row performs scalar
`integers(0, 1)` on the cloned PCG64 generator, records selected index zero,
uses a byte-exact copy of complete genome zero as its virtual template, takes
the corresponding lesion or `0.0` when absent, and starts from an empty copy
and zero fractional progress.  It then continues proofreading, quiescence,
payment, threshold draws, and conditional replacement draws in that same
cell call.  Across cells the only allowed replay order is therefore
`selection-if-needed -> that cell's threshold/conditional-integer draws ->
next cell`; selections are never batched ahead of substitution draws.  The
high-level index-zero call is replayed even though the recorded NumPy/PCG64
implementation leaves the complete before-state, including its uint32 cache,
unchanged.

The pure plan adds `template_start_events`, `selected_template_indices`, and
`template_storage_symbols`; the tape independently attests the start mask and
selection indices in its schedule digest and replay.  These fields describe a
future atomic topology transaction only.  A4.5b does not modify or re-attest
the resident ragged arena, does not commit the live RNG after-state, and is not
save or scheduler authority.

Before a start is reported successful the batch proves both future capacities:
`sequence_count + 2 * start_count <= sequence_capacity` and
`symbol_count + sum(start_template_lengths) + sum(paid_append_counts) <=
symbol_capacity`.  The empty copy still consumes one of the two sequence
slots.  Exact capacity passes; either capacity one short fails atomically
before a tape or plan can become commit authority.  With sufficient capacity,
a start which reaches completion stays code 3 for A4.6; the inherited
batch-global capacity check retains its later precedence and may instead make
the whole invalid batch code 4.  Neither result is exposed as a partial start
plus a second CPU replication call.

## A4.6a scope

A4.6a adds one separate mutation-free completion descriptor for a
pre-existing active, non-empty template.  The template must be incomplete on
entry and every used row must reach completion in this call after the frozen
ordered resource gates.  A non-completing or otherwise unsupported row makes
the whole descriptor event invalid.  The legacy elongation and substitution
APIs continue to report completion as scope code 3; completion is reachable
only through the public `paid_replication_completion_plan` dispatch and no
public mode flag is exposed.

The completed polymer is the exact existing paid partial-copy prefix followed
by this call's paid append.  It is never reconstructed by copying the template,
so substitutions produced by earlier calls are preserved.  For each successful
row the fixed plan records:

- `completion_events`, `completed_symbols`, and `completed_lengths`;
- the frozen mutation-free inherited lesion
  `template_lesion * (0.28 + 0.22 * (1 - proof_fraction)) +
  effective_error * completed_length * 0.06`;
- `replication_cycle_deltas = 1`;
- `topology_sequence_deltas = -1`, because template plus copy become one
  complete genome sequence; and
- `topology_symbol_deltas = append_count - completed_length`, the net arena
  change from the original active representation, not an additional material
  payment.

Fractional progress resets to zero in the descriptor.  Nucleotide, ATP,
proofreading ATP, last-copy count, and effective-error telemetry retain the
same ordered A4.4b calculation.  With `mutation=false`, frozen
`mutate_sequence` returns before RNG or structural/material changes, so the
CPU RNG is byte-exact unchanged.

This is a payload/ledger/topology descriptor, not a state commit.  It does not
mutate or re-attest the ragged arena, append a live complete genome or lesion,
refresh the gene cache, apply `novel_path_first_age`, advance the live RNG,
update a CPU cell, or replace the scheduler.  Conceptual post-state packing
and A4.2 re-decode are validation evidence only.  Actual atomic arena/cache/
physiology reissue remains a later bounded slice.

The generic plan validator proves fixed schema, canonical tails, paid suffix,
range, and topology self-consistency.  It does not accept a plan as commit
authority or rederive the historical completed prefix and lesion from a source
binding.  A future atomic commit must consume only an internally generated
plan and revalidate those two semantic relations against the attested source;
caller-forged but self-consistent payload/lesion values are outside A4.6a's
trust boundary.

## A4.6b1 scope

A4.6b1 fixes the combined PCG64 call schedule for mutation-enabled completion
without yet applying structural mutations on the device.  Every used row has a
pre-existing active, non-empty template, is incomplete on entry, and completes
in this call.  Template selection, non-completing rows, and mixed batches stay
outside this event tape.  The public A4.6a mutation-free descriptor is not
weakened or given a mode flag.

Tape preparation first re-derives the mutation-free completion schedule from
the attested binding.  It then replays one cloned NumPy PCG64 stream in literal
cell order.  For each cell it performs all paid-append threshold draws and
their immediate conditional `integers(0, 7)` replacements, then immediately
performs that cell's structural chain before advancing to the next cell.  It
is forbidden to prepare all cells' append substitutions first and append a
second all-cell structural tape, because the resulting global RNG state would
not match Formal066.

The structural chain records, without inventing raw-draw counts:

- scalar insertion, deletion, duplication, inversion, and transposition
  threshold calls and the short-circuit conditions which decide whether each
  call exists;
- the exact scalar count/position/ordinal bounds and returned values for an
  applied operation;
- insertion and minimum-length padding as the original high-level NumPy
  `uint8` vector calls, never scalarized;
- the pre-budget length, material budget, post-budget length, committed
  material delta, and frozen per-operation event counts; and
- the complete `state`, `inc`, `has_uint32`, and `uinteger` PCG64 before/after
  states.

The frozen structural mutator receives `mutation_rate=0` after paid append
substitutions, so it performs no second base-substitution vector.  Nucleotide
budget is computed once from the post-elongation pool.  The tape stores the
outcome-equivalent effective budget capped at frozen `MAX_GENOME_LENGTH`,
because no structural expansion can consume more symbols.  The frozen fp64
division and Python integer conversion are performed first; a nonfinite raw
quotient fails closed instead of inventing a successful A4 state where the CPU
reference raises.
Material-budget tail
trim occurs only after every structural RNG call and event count.  It is an
authoritative material rule, not arena overflow clipping: event counters keep
their pre-trim values, while nucleotide payment or refund uses only the final
length delta.  Structural mutation consumes no ATP.  Arena sequence/symbol/
lesion capacity remains a separate fail-closed condition and is never repaired
by extra trimming.  Both the final per-sequence length and the aggregate future
symbol count, including structural material deltas, must fit exactly.

`A4CompletionMutationRngTape` is a private-factory, event-local object with a
binding-aware host validator.  Host arrays are protected by a canonical digest;
resident tensors use scalar metadata plus pointer/version attestation and carry
the validated host-content digest unchanged through resident clones.  Explicit
readback must match that original digest before a NumPy tape is reissued, which
also rejects `.data` or shared-NumPy writes that bypass Torch `_version`.  The
validator re-derives the completion schedule and replays every high-level call,
bound, dtype, vector payload, event count, material delta, and final PCG64 state
against the same source/config/dt.  Any disagreement invalidates the complete
shared stream.  The tape stores operation parameters, not a CPU-produced final
genome; A4.6b2 must derive the transformed polymer with fixed NumPy/Torch
operations and perform its own device-side semantic checks before use.

A4.6b1 does not apply the tape, produce mutation-enabled completion authority,
advance a live RNG, mutate/re-attest the ragged arena, refresh a cache, update a
CPU cell, or replace the scheduler.  Its CPU/CUDA result is tape storage and
attestation evidence only.  Hydrolysis remains A4.7, while post-division grammar
mutation remains A5.

## A4.6b2 scope

A4.6b2 consumes only an internally prepared, binding-aware A4.6b1 tape and
applies its recorded operations to the mutation-free A4.6a completion payload.
It derives the final polymer on NumPy, Torch CPU, and CUDA; the tape never
contains a CPU-produced final genome.  The supported event remains bounded to
pre-existing active, non-empty templates where every used row completes in the
same call with mutation enabled.

Application follows the frozen order on fixed storage: paid-append
substitutions, insertion, deletion, gene duplication, inversion,
transposition, minimum-length padding, frozen maximum-length truncation, and
physical nucleotide-budget tail trim.  Bounds and short-circuit conditions are
recomputed on the device from the evolving polymer.  Event counts retain their
pre-material-trim meaning; only the committed final length delta is paid or
refunded in the nucleotide pool.  When the pre-structural polymer is shorter
than `MIN_GENOME_LENGTH`, padding is still drawn before material trim, but a
zero physical budget may validly trim the result back to that shorter original
length.  Structural mutation consumes no ATP.

`A4CompletionMutationPlan` records the final fixed-width polymer and length,
post-payment pools, append/substitution/structural counts, inherited lesion,
cycle and topology deltas, cleared replication intermediates, and derived
post-completion genome count, material-symbol count, and lesion mean.  The new
lesion recomputes the inherited component directly from the attested template
lesion and ordered proofreading signal, rather than subtracting it back out of
the rounded A4.6a total, and uses the final mutated length for the
effective-error term.  Ordered lesion means reduce the combined old-plus-new
lesion prefix once, using the same NumPy pairwise grouping on Torch CPU and
CUDA.

The binding-aware host tape owns the frozen Python float64 division and integer
conversion for nucleotide budget.  Torch does not re-truncate that quotient,
because CUDA may place an exact CPU integer one ulp below its boundary.  It
uses the attested tape integer for transformation and independently requires
the resident quotient to remain inside that integer bucket, allowing only the
registered fp64 comparison band.

Before consuming a resident tape, A4.6b2 compares every public tensor to a
private expected tensor copied from the validated upload, in addition to the
existing scalar, pointer, version, and host-digest attestation.  This closes
`.data` and shared-alias changes which do not increment Torch `_version`.
Any tape, operation, bound, count, material, capacity, or derived-state
disagreement invalidates and rolls back every used row; no partial polymer is
reported as successful.

A4.6b2 is still a pure descriptor.  It does not apply the final arrays to the
ragged arena, append live genome/lesion state, refresh the gene cache, advance
the live PCG64 state, update a CPU cell, or replace A3 `replication_cpu`.
Hydrolysis remains a separate event.  An atomic arena/cache/RNG/scheduler
commit is a later bounded slice and must re-attest all source relations
immediately before use.

## A4.7a scope

A4.7a records only the frozen live-genome symbol-hydrolysis RNG event for one
cell.  Its input is an internally created NumPy `A4TranslationBinding` at the
post-lesion-gain, pre-hydrolysis boundary.  Exactly one cell is allowed, and
its lesion vector must contain one value for every complete genome.  Active
replication template/copy sequences are not hydrolysis candidates.

Complete genomes are visited in their existing list/storage order.  For each
genome, the literal frozen gate is evaluated as
`lesion > 0.75 and length > MIN_GENOME_LENGTH`.  Every genome which passes
that gate consumes one scalar `Generator.random()` call.  This remains true
when `dt == 0` and the comparison probability is zero.  Only a strict
`draw < dt * 0.00065 * lesion` hit immediately consumes the scalar
`Generator.integers(0, length)` deletion-position call.  The implementation
uses a cloned canonical NumPy PCG64 state and records its full before/after
`state`, `inc`, `has_uint32`, and `uinteger`; it never advances a live RNG.

`A4HydrolysisRngTape` is private-factory and event-local.  Its scalar identity
binds sequence capacity/count, genome count, stable cell ID, source
provenance, exact `dt` hex, draw/hit counts, schedule digest, and the two PCG64
states.  Fixed arrays aligned to `sequence_capacity` are
`genome_slot_mask`, `draw_mask`, `uniform_draws`, `hit_mask`, and
`deletion_positions`.  Non-draw uniforms are zero and non-hit positions are
`-1`.  Host replay re-derives every gate and high-level call from the same
binding.

Resident tape operations verify scalar metadata and tensor pointers/versions
without hidden D2H or synchronization.  Only explicit `to_numpy()` reads the
arrays back and compares both the resident values and private expected values
with the original host-content digest; this is where `.data` changes which
bypass Torch `_version` fail closed.  A4.7a has no device-side biological
application.

The current A3 bridge drops hazard records whose derived probability is
`<= 0`, so it consumes no draw for the eligible `dt == 0` case.  That is a
known bridge difference, not a reason to change the frozen oracle.  A4.7a
preserves the frozen draw and must not replace scheduler authority until the
A3 bridge and its lockstep tests are corrected in a later integration slice.

## A4.7b scope

A4.7b consumes the trusted A4.7a tape and the same exactly-one-cell
`A4TranslationBinding` to derive a pure deletion/ledger descriptor.  Public
entry points are `genome_hydrolysis_deletion_numpy`,
`genome_hydrolysis_deletion_torch`, and
`genome_hydrolysis_deletion_plan`; full host replay is
`validate_a4_hydrolysis_deletion_plan(plan, binding, dt, tape)`.  None of
these functions changes the binding, tape, CPU cell, world, or live RNG.

`A4HydrolysisDeletionPlan` uses row-padded `final_symbols[Q,W]` rather than
claiming ownership of the compact ragged arena.  `Q` is source sequence
capacity and `W` is the frozen per-sequence capacity.  Every used sequence is
represented: hit complete genomes omit exactly the attested position, misses
are unchanged, and active replication template/copy rows remain byte-exact.
Unused symbol rows and row tails are zero.  `final_lengths[Q]` and the scalar
source identity make a later compact rebuild possible, but this slice does
not perform that rebuild.

The remaining fixed arrays are `scope_valid[1]`, `scope_error_code[1]`,
`genome_lesions_after[Q]`, `pools_after[1,POOL_COUNT]`,
`symbol_count_after[1]`, `topology_symbol_delta[1]`,
`genome_damage_event_delta[1]`, `gene_cache_dirty[1]`,
`gene_cache_refresh_count[1]`, `genome_material_symbols_after[1]`, and
`genome_lesion_mean_after[1]`.  The scalar metadata binds capacity/counts,
source symbol count, cell ID, source provenance, and the A4.7a tape schedule
digest.  The plan contains no duplicate RNG state or hit-position authority.

For every hit, complete genomes are processed in original order: delete one
symbol, add `MONOMER_MASS` to waste with one fp64 addition, multiply that
genome lesion by `0.80`, increment the damage-event delta, and record one
cache refresh.  Waste is never computed as `hit_count * MONOMER_MASS` because
that changes frozen sequential rounding.  Frozen cache refresh has no
intermediate reader, so the descriptor records the literal refresh count and
dirty bit; the final cache can be derived from the final complete-genome rows,
but no cache is made live here.

The lesion mean is formed from the complete updated lesion prefix with the
same contiguous NumPy pairwise grouping.  CPU and CUDA can differ only in the
final division by genome count; validation permits the finite, nonnegative
NumPy result or its immediate `nextafter` neighbour (at most one ULP) for this
field alone.  Symbols, lengths, lesions, sequential waste, event/material
ledgers, and all other fields remain exact.  Software fp64 division is not
introduced for this bounded diagnostic difference.

NumPy replays the tape against the full host source before application.
Resident application requires matching source provenance, capacity, `dt`,
schedule, backend, and device, then compares every public tape tensor with its
private expected tensor on device.  A `.data` bypass or semantic disagreement
invalidates the single global row and returns source-state rollback values;
no `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `torch.equal`, or implicit
synchronization is used.  Full plan/source replay occurs only after explicit
readback.

## A4.8a scope

A4.8a is the first bounded scheduler commit, and it replaces only the existing
`genome_hydrolysis_cpu_rng` event inside a new A4 wrapper. The promoted A3
source and scheduler remain byte-identical and retain authority for every
other event. `A4HydrolysisEventScheduler` subclasses the exact-once A3
scheduler, preserves the event ID and rank, and accepts no external binding,
tape, plan, or commit candidate. `Hybrid066WorldA4Hydrolysis` preserves the
explicit A4 capacity/device settings across step-boundary save, load, and
clone; active scheduler or pending-commit serialization remains forbidden.

For each invocation, the bridge packs the post-`genome_lesion_gain` live cell
as a fresh exactly-one-cell host binding. It clones the complete live PCG64
before-state, prepares the A4.7a literal tape, uploads source and tape to the
explicit Torch CPU/CUDA device, evaluates the A4.7b resident plan, reads every
resident input and output back explicitly, and performs full binding-aware
host replay. A shallow disposable CPU candidate is populated from the
validated compact rows, its final gene cache is refreshed, and a new host A4
binding proves compact topology, physiology, material count, and lesion mean.
All of this completes before the scheduler event is claimed.

Immediately before claim, the bridge rechecks the live cell object and stable
ID, generation and damage counter, exact `dt.hex()`, world-config bytes, A4
capacity/device settings, biological binding identity, and live PCG64
before-state. The private candidate before/after states must equal the A4.7
tape before/after states exactly. Retained resident ragged, physiology, cache,
tape, and plan values are explicitly read back again, so ordinary writes and
Torch `.data` version bypasses fail before meaning is committed.

Only after those checks does the bridge claim `genome_hydrolysis_cpu_rng`
exactly once and publish complete genomes, ordered lesions, pools, final gene
cache, `genome_damage_events`, and the full PCG64 after-state. Active
replication template/copy and unrelated cell state are unchanged. A no-hit
event, including an eligible `dt == 0` event, is still recorded as executed;
`work_performed` is false, while the literal eligible random draws are kept.
This corrects the zero-probability draw only in the A4 wrapper; the promoted
A3 bridge itself is not edited.

Capacity, source, trust, or ordering failure before claim leaves receipts,
biology, and live RNG unchanged and never falls back to the old CPU bridge.
If publishing fails after claim, the original list/array/dictionary/RNG object
identity and values are restored locally and the outer scheduler records an
aborted step. Event-local bindings, tape, plan, and candidate are one-shot and
disposable; the committed CPU world and PCG64 state remain the durable
step-boundary save authority. This correctness bridge performs intentional
per-cell host/device round trips and makes no performance claim.

## A4.8b scope

A4.8b adds one further bounded event replacement in a new module without
editing the promoted A3 files, the A4.3 pure plan, or the A4.8a bridge.
`A4TranslationEventScheduler` inherits the A4.8a hydrolysis override and
replaces only the existing rank-5 `translation_cpu` event after maintenance
and before the still-CPU-authoritative replication event.
`Hybrid066WorldA4Translation` preserves the derived scheduler type, fixed A4
capacities, and explicit Torch device across step-boundary save, load, and
clone. Active or pending serialization remains forbidden.

For one live post-maintenance cell, the scheduler constructs a fresh host A4
binding, independently evaluates the NumPy paid-translation oracle, uploads
the source/cache/state to the explicit Torch CPU or CUDA device, evaluates the
resident A4.3 plan, and explicitly reads every resident input and output back.
The resident readback is commit authority. The NumPy replay must have exact
cell/discrete/fingerprint/count/order state and every fp64 array must agree
within the registered absolute `2e-12` tolerance; its host values are never
substituted for the resident result.

A private one-shot CPU candidate is rebuilt only from that resident result.
Its pools, ordered active and damaged protein dictionaries,
`last_translation`, and `last_quiescence` are repacked into a fresh binding.
Live `gene_specs` must agree exactly with the genome-derived A4.2 cache at
preparation and final validation, and the candidate holds a deep copy so an
in-place dictionary alias cannot bypass source attestation. Immediately
before claim, the bridge rechecks cell/object identity, generation and alive
state, exact `dt.hex()`, the active world-config object and bytes, A4
capacity/device state, full live PCG64 identity/state, all host replay values,
all resident source/cache/plan values including Torch `.data` bypasses, the
dictionary-sync gate, and the fresh candidate binding.

Only after those checks does the scheduler claim `translation_cpu` exactly
once and publish the resident paid FUEL/MINERAL/ATP result, synchronized
CATALYST/DAMAGED_PROTEIN ledgers, ordered active/damaged dictionaries,
`last_translation`, and conditionally written `last_quiescence`. Frozen
object behavior is preserved: an early return retains both protein dictionary
objects, while reaching the weight gate replaces both through the equivalent
of `_sync_*`, even when reserve or `dt == 0` yields no translated mass. The
pools array identity is retained. Genomes, lesions, template/copy,
`gene_specs`, replication telemetry, world energy ledgers, and live PCG64 are
unchanged, and translation records zero RNG draws.

Failure before claim performs no bridge mutation and never invokes the legacy
CPU translation bridge. Failure while publishing restores the original
pool/dictionary/gene-cache/scalar/RNG identities and values and leaves an
aborted outer scheduler receipt. The event-local bindings, oracle, resident
plan, and candidate are disposable and have no save authority. The bridge is
correctness-only and makes no speed or full-world claim.

## A4.8c1 scope

A4.8c1 adds one deliberately partial replication-event replacement in a new
module without editing promoted A3, the A4.4--A4.6 pure cores, or either
earlier integration bridge. `A4ReplicationEventScheduler` inherits A4.8b
translation and A4.8a hydrolysis, then replaces only `replication_cpu` for a
live cell that already has a nonempty active template and an incomplete copy
and whose current call remains noncompletion. `Hybrid066WorldA4Replication`
preserves the scheduler type, fixed capacities, config, and explicit device
across save, load, and clone. Active or pending serialization remains
forbidden.

Mutation-free calls use the A4.4b deterministic resident elongation plan.
Mutation-enabled calls use a PCG64 substitution tape prepared from a clone of
the exact live before-state and the A4.5a resident substitution plan. Valid
active calls with `dt == 0`, requested zero, or resource-stopped zero append
remain in scope because the frozen path still updates fractional/error/last
telemetry after its active gates. Inactive start, same-call completion,
disabled/no-genome/replicase-gated early returns, negative ATP, fp64 discrete
ambiguity, and capacity failure are rejected before claim with no CPU
fallback.

The resident readback is commit authority. An independent NumPy replay must
match every discrete value, append byte and order exactly and every permitted
fp64 value within absolute `2e-12`; its host values are never substituted.
Immediately before claim the bridge recreates the mutation tape from the live
source and RNG, and reattests the host source/oracle, every resident
ragged/state/cache/tape/plan object, backend, device, pointer and value, and a
deep-isolated candidate with a fresh binding. Whole-member swaps, coordinated
host/tape changes, aliases, and Torch `.data` version bypasses fail closed.

After exactly one `replication_cpu` claim, the bridge updates the existing
pools array, extends the existing replication-copy list in order, updates
fractional/last/effective-error/proofreading telemetry, and updates the
existing mutation-event dictionary's substitution counter. Mutation-enabled
calls also publish the attested PCG64 after-state into the same live generator;
mutation-free calls leave the full state unchanged. Genomes, lesions,
template, gene-spec outer/nested dictionaries, protein dictionaries,
replication cycles, novel-path state, and world energy retain identity and
value. Preclaim failure changes nothing; postclaim failure restores all
touched and attested identities, order, values and RNG state and leaves an
aborted outer receipt.

This is a correctness-only, opt-in active-noncompletion bridge. It does not
make ordinary inactive-founder worlds generally supported and does not claim
completion, full replication authority, persistent resident state, speedup,
or a full GPU world-step.

## A4.8c2 scope

A4.8c2 adds exactly one completion branch in a new module while retaining the
A4.8c1 source unchanged. `A4ReplicationCompletionEventScheduler` accepts a
live cell with a pre-existing nonempty active template, an entry-time partial
copy, `mutation == false`, and an A4.6a resident plan that completes in this
call. A4.8c1 active noncompletion is selected only by the explicit resident
scope code and delegated to the inherited scheduler. This includes valid
active-gate `dt == 0`, requested-zero, and resource-stopped noncompletion.
Mutation-enabled completion, inactive start, and disabled/no-genome/
replicase-gated pre-active-return branches fail before claim with no frozen-
CPU fallback.

The resident A4.6a completion readback is commit authority. An independent
NumPy replay must match all discrete fields and completed bytes exactly and
permitted fp64 values within absolute `2e-12`; host values are never
substituted. The completed polymer is the existing paid partial copy followed
by this call's paid template suffix, preserving prior substitutions. Declared
source objects, oracle values, resident ragged/state/cache/plan tensor storage
and values, candidate state, fresh binding, config, device, `dt`, and the
unchanged live PCG64 state are reattested immediately before claim. Equivalent
read-only wrapper/view objects are semantic equivalents; A4.8c2 does not claim
a generic transaction over every opaque mutable member of the CPU cell.

After exactly one `replication_cpu` claim, the bridge appends the completed
genome and lesion to the existing lists, retains existing genome objects,
clears template/template-lesion/copy, resets fractional progress, increments
the cycle, refreshes the existing gene-cache dictionary, conditionally sets
`novel_path_first_age`, and publishes paid pools and replication telemetry.
Mutation counters, proteins, damaged proteins, world energy, and the complete
PCG64 object/state remain unchanged. Preclaim failure changes no bridge-owned
state; postclaim failure restores the bridge-touched and explicitly snapshotted
identities, order, values, and RNG state and leaves an aborted outer receipt.
Protected test-hook code that independently mutates an out-of-scope CPU field
is not promoted into a generic whole-cell rollback authority. Save/load/clone preserve the
A4.8c2 wrapper and scheduler type at step boundaries; active serialization is
rejected.

This remains a correctness-only, per-event explicit-readback bridge. It does
not implement mutation-enabled completion, inactive start, start-completion,
pre-active-gate early-return authority, persistent resident state, speedup, or
a full GPU world-step.

## A4.8c3 scope

A4.8c3 adds only the mutation-enabled counterpart of A4.8c2 in a new module.
`A4ReplicationCompletionMutationEventScheduler` accepts one live cell whose
non-empty template was already active on entry, whose entry copy was shorter
than that template, whose configuration has `mutation == true`, and whose
paid base-copy schedule completes in this call. It prepares the binding-aware
A4.6b1 combined PCG64 tape from a clone of the exact live before-state and
uses that tape to produce the A4.6b2 resident final-polymer plan.

The dispatch probe is resident but is not commit authority. Mutation-free
completion is explicitly delegated to A4.8c2; active mutation-free or
mutation-enabled noncompletion, including requested-zero and other valid
active zero-work cases, is delegated through A4.8c2 to A4.8c1. Inactive
template start, start-completion, and disabled/no-genome/replicase-gated or
other pre-active returns fail before claim without invoking the frozen CPU
replication bridge.

Within the owned branch, tape replay preserves the literal per-cell PCG64
order: paid-append threshold draws and immediate conditional bounded-integer
replacement calls are followed by that cell's insertion, deletion,
duplication, inversion, transposition, and minimum-padding chain. Material-
budget tail trim occurs only after those draws and recorded structural event
counts. It is not capacity clipping. The complete `state`, `inc`,
`has_uint32`, and `uinteger` before/after state is attested. Immediately before
claim the tape and NumPy final plan are regenerated from the current live
binding and RNG before-state; retained host and resident source, tape, final
plan, candidate, fresh binding, config, device, pointers, versions, and values
are revalidated.

The explicit A4.6b2 resident readback is commit authority. The independently
replayed NumPy plan must agree exactly on discrete fields and final bytes and
within absolute `2e-12` on permitted fp64 fields.  The derived
`genome_lesion_mean_after` and fresh `genome_lesion_mean` alone admit the
registered finite, nonnegative final-division difference of at most one ULP;
two ULPs are rejected before claim.  Host oracle values are never substituted.
After exactly one `replication_cpu` claim, the bridge writes only
the paid NUCLEOTIDE/ATP entries in the existing pools array, appends the final
polymer and lesion while retaining earlier genome objects, clears the active
template/copy state, resets fractional progress, increments the replication
cycle, publishes proofreading/error/last-symbol telemetry and the additive
substitution plus five structural counters, refreshes the existing gene-cache
mapping, conditionally sets `novel_path_first_age`, and installs the attested
after-state into the same live PCG64 generator.

Failure before claim changes no live biology, RNG, world ledger, or receipt.
Failure during publish restores the bridge-touched and explicitly snapshotted
pool, genome/lesion, template/copy, mutation-ledger, gene-cache, cycle,
telemetry, novel-path, world-energy, and RNG identities, order, and values,
then leaves an aborted outer receipt. This is not a generic whole-cell
transaction claim. Save/load/clone preserve the A4.8c3 world and scheduler
type, capacities, device, and inherited A4.8a/b/c1/c2 settings at step
boundaries; active or pending serialization is rejected.

A4.8c3 remains a correctness-only, per-event explicit-readback bridge. It
does not implement inactive-template start, start-completion, pre-active-gate
early-return authority, persistent resident state, a device RNG, speedup, A4
completion, or a full GPU world-step.

## A4.8c4 scope

A4.8c4 adds one mutation-enabled inactive-template-start/noncompletion branch
above A4.8c3 in a new module. `A4ReplicationStartEventScheduler` accepts one
alive cell with `mutation == true`, no active template, an empty copy, and
exactly one non-empty complete genome when the A4.5b resident plan reports one
template start and no same-call completion. All active-template sources
delegate explicitly to A4.8c3 and its A4.8c2/A4.8c1 authorities. Mutation-free
inactive start, same-call start completion, and ordinary pre-active no-op
branches stop before claim without frozen-CPU fallback.

The private one-shot tape starts from a clone of the exact live PCG64 state.
It preserves the frozen order of one template-selection `integers(0, 1)` call,
then one scalar threshold draw per paid append and an immediate scalar
`integers(0, 7)` call only on a substitution hit. The complete `state`, `inc`,
`has_uint32`, and `uinteger` before/after state is attested. Immediately before
claim, the tape is regenerated from the current live binding and RNG state and
is compared exactly with the retained host and resident tape, plan, after-
state, candidate, and fresh binding, including object/member identity, device,
pointer, version, and content attestations.

The explicit A4.5b resident plan readback is commit authority; the independent
NumPy replay is validation only. Discrete fields and bytes match exactly and
permitted fp64 fields differ by at most absolute `2e-12`; host values are never
substituted. Future capacity is checked as `Q + 2` sequences and
`S + template_length + append_count` symbols, with the exact `W` and `P`
binding capacities also required. Each one-short case fails before claim;
there is no clipping or reallocation.

After exactly one `replication_cpu` claim, the bridge selectively publishes
the paid NUCLEOTIDE/ATP pools, substitution ledger, fractional/replication
telemetry, and the attested after-state into the same live Generator. It
installs a new NumPy template which copies but does not alias the selected
genome and a new replication-copy list containing the paid append. The
template lesion is the selected genome lesion, or frozen-CPU fallback `0.0`
when that lesion is absent. `dt == 0`, requested-zero, and resource-stopped
zero-append plans still commit the template start with `work_performed=true`.
Genome/lesion lists, gene cache, cycle, topology, and other out-of-scope state
remain unchanged.

Preclaim failure changes no biology, RNG, receipt, or world ledger. Publish
failure restores the explicitly snapshotted write set, including the original
`None` template, original empty-copy list object, pools, mutation ledger,
fractional/telemetry state, live RNG object/state, and world energy, and leaves
an aborted outer receipt. This is not a generic whole-cell transaction.
Save/load/clone preserve `Hybrid066WorldA4ReplicationStart`, its scheduler,
capacities, config, device, and inherited A4.8a/b/c1/c2/c3 metadata at step
boundaries; active or pending serialization is rejected and transient
candidate/binding/tape/plan objects are not saved.

A4.8c4 remains a correctness-only, per-event explicit-readback bridge. It
does not implement mutation-free inactive start, same-call start completion,
ordinary pre-active early-return authority, persistent resident state, a
device RNG, speedup, A4 completion, or a full GPU world-step.

## A4.8c5 scope

A4.8c5 adds the mutation-free inactive-template-start/noncompletion branch
above A4.8c4 in a new module. `A4ReplicationMutationFreeStartEventScheduler`
accepts one alive cell with `mutation == false`, no active template, an empty
copy, and exactly one non-empty complete genome when the private start-capable
A4 elongation kernel reports one template start and no same-call completion.
Active sources delegate through A4.8c4 to the A4.8c3/A4.8c2/A4.8c1
authorities, and an inactive mutation-enabled source delegates to A4.8c4.
Same-call start completion and ordinary pre-active no-op branches stop before
claim without frozen-CPU fallback.

The private selection evidence binds the source identity, cell ID, exact
`dt`, mutation-free config digest, call bounds `0/1`, selected index zero,
call count one, schedule digest, and complete PCG64 before/after state. It
constructs a cloned real Generator and actually executes the frozen high-level
`integers(0, 1)` call. On the registered NumPy implementation that call is
state-neutral for both primed and unprimed PCG64 states, but it is not omitted
or replaced by direct assignment. Mutation-free append takes no threshold or
replacement draws. Immediately before claim, the deterministic NumPy plan and
selection evidence are regenerated from the current live binding and RNG and
compared exactly with their retained attestations.

Only the integration-private dispatcher may call
`_paid_replication_elongation_numpy/torch(..., allow_template_start=True)`.
The explicit resident Torch plan readback is commit authority; the independent
NumPy plan is validation only. Discrete fields and bytes match exactly and
permitted fp64 fields differ by at most absolute `2e-12`; host values are never
substituted. Future capacity is checked as `Q + 2` sequences and
`S + template_length + append_count` symbols, with the exact `W` and `P`
binding capacities also required. Each one-short case fails before claim;
there is no clipping or reallocation.

After exactly one `replication_cpu` claim, the bridge selectively publishes
the paid NUCLEOTIDE/ATP pools, a new non-aliasing template array, a new
replication-copy list, the selected lesion or missing-lesion fallback `0.0`,
fractional/last-symbol/effective-error/proofreading telemetry, and the attested
after-state into the same live Generator. The effective-error telemetry still
includes configured mutation rate, lesion, reactive, and proofreading terms
even though mutation is disabled. The mutation ledger is outside the success
write set: its object, insertion order, keys, and values receive no write,
including no same-value zero assignment. `dt == 0`, requested-zero, and
resource-stopped zero append still commit the selection and template start
with `work_performed=true`. Genome/lesion lists, gene cache, cycle, topology,
world energy, and other out-of-scope state remain unchanged.

Preclaim failure changes no biology, RNG, receipt, or world ledger. Publish
failure restores the explicitly snapshotted write set, including the original
`None` template, original empty-copy list object, pools, telemetry, unchanged
mutation-ledger identity/order/values, live RNG object/state, and world energy,
then leaves an aborted outer receipt. This is not a generic whole-cell
transaction. Save/load/clone preserve
`Hybrid066WorldA4ReplicationMutationFreeStart`, its scheduler, capacities,
config, device, and inherited A4.8a/b/c1/c2/c3/c4 metadata at step boundaries;
active or pending serialization is rejected and transient
candidate/binding/evidence/plan objects are not saved.

A4.8c5 remains a correctness-only, per-event explicit-readback bridge. It
does not implement mutation-enabled or mutation-free same-call start
completion, ordinary pre-active early-return authority, persistent resident
state, a device RNG, speedup, A4 completion, or a full GPU world-step.

## A4.8c6 scope

A4.8c6 adds the mutation-free inactive-template-start/same-call-completion
branch above A4.8c5 in a new module.
`A4ReplicationStartCompletionEventScheduler` accepts one alive cell with
`mutation == false`, no active template, an empty copy, and exactly one
non-empty complete genome only when an event-local synthetic active-start
source completes through the public A4 completion plan. Mutation-free
start/noncompletion delegates to A4.8c5; inactive mutation-enabled and active
sources retain their A4.8c4/A4.8c3/A4.8c2/A4.8c1 authorities. Ordinary
pre-active no-op branches stop before claim without frozen-CPU fallback.

Selection evidence binds the live source identity, cell ID, exact `dt`,
mutation-free config digest, call bounds `0/1`, selected index zero, call
count one, schedule digest, and complete PCG64 before/after state. It executes
the actual frozen high-level `integers(0, 1)` call on a cloned real Generator.
Mutation threshold and replacement draws are absent. On the registered NumPy
implementation the selection call is state-neutral for both primed and
unprimed PCG64 states, but it is still executed and attested.

For a source genome of length `T`, the live inactive topology is `Q=1, S=T`.
The synthetic cell is deep-isolated from live biology and holds a new
non-aliasing template copy, the selected lesion or missing-lesion fallback
`0.0`, a fresh empty copy, and zero fractional carry. Its topology is exactly
`Q=3, S=2T`; the template remains reference state rather than extra material.
The public `paid_replication_completion_plan` must append and complete exactly
`T` symbols equal to the selected genome, report sequence delta `-1`, symbol
delta `0`, cycle delta `+1`, zero substitutions, and no template-start event.
The final topology is `Q=2, S=2T`. Transaction capacity is checked at the
synthetic peak: exact `Q=3`, `S=2T`, `W=T`, and the exact protein/cache union
bound `P` pass, while every one-short case fails before claim without clipping,
reallocation, or fallback.

The explicit resident Torch public-plan readback is commit authority; a
separately rebuilt NumPy public plan is validation only. Source, selection,
synthetic binding, resident members/storage/pointers/versions/content, direct
net candidate, and fresh final binding are all re-attested immediately before
the single claim. Discrete fields and bytes match exactly and permitted fp64
fields differ by at most absolute `2e-12`; host values are never substituted.

After exactly one `replication_cpu` claim, the bridge publishes only the paid
NUCLEOTIDE/ATP pools and the attested direct net completion state. It appends
a new non-aliasing genome and its lesion to the existing lists, clears the
template, installs another fresh empty copy list, resets fractional carry,
increments the replication cycle, publishes replication/proofreading
telemetry, refreshes the existing gene-cache dictionary, and conditionally
sets `novel_path_first_age`. Prior genome objects and the outer pool/genome/
lesion/cache objects retain identity. Material symbols increase by exactly
`T`; no uncharged material is introduced. The same live Generator receives
the attested after-state. Mutation-ledger final identity, insertion order,
keys, and values are unchanged; this is a durable-state guarantee, not a
call-trace claim about the frozen CPU implementation. World energy and other
out-of-scope state remain unchanged.

Preclaim failure changes no biology, RNG, receipt, or world ledger. Publish
failure restores the complete declared snapshot, including `None` template,
the original empty-copy object, pools, genome and lesion lists, mutation
ledger, proteins and damaged proteins, nested gene cache, telemetry, cycle,
novel-path age, world energy, and the live RNG object/state, then leaves an
aborted outer receipt. Save/load/clone preserve
`Hybrid066WorldA4ReplicationStartCompletion`, its scheduler, capacities,
config, device, and inherited A4.8a/b/c1/c2/c3/c4/c5 metadata at step
boundaries; active or pending serialization is rejected and transient source,
selection, synthetic, resident-plan, candidate, and binding objects are not
saved.

A4.8c6 remains a correctness-only, per-event explicit-readback bridge. It
does not implement mutation-enabled same-call start completion, ordinary
pre-active early-return authority, persistent resident state, a device RNG,
speedup, A4 completion, or a full GPU world-step.

## A4.8c7 scope

A4.8c7 adds the mutation-enabled inactive-template-start/same-call-completion
branch above A4.8c6 in a new module.
`A4ReplicationStartCompletionMutationEventScheduler` accepts one alive cell
with `mutation == true`, no active template, an empty copy, and exactly one
non-empty complete genome when an event-local synthetic active-start source
completes through the A4.6b1/A4.6b2 completion-mutation path. Active sources,
inactive mutation-free sources, and mutation-enabled start/noncompletion retain
their A4.8c1-c6 authorities. Ordinary pre-active no-op branches stop before
claim without frozen-CPU fallback.

Dispatch uses a deep-copied `mutation == false` config only to classify the
synthetic resident public completion plan as completion or noncompletion. That
clone is never commit authority. The A4.5b start tape is not concatenated with
A4.6b1 because it rejects same-call completion and would replay the append RNG
draws twice. Instead, private factory-authenticated selection evidence executes
exactly one real high-level `integers(0, 1)` call on a cloned PCG64 state and
binds both live and synthetic identities, cell ID, exact `dt`, mutation config
digest, bounds/index/count, schedule digest, and complete before/after state.
Its after-state is exactly the A4.6b1 tape before-state. A combined chain digest
binds both factory-verified schedules, identities/provenance, config, `dt`, and
all three PCG64 boundaries and is rederived before claim and recorded in the
receipt.

The remaining frozen RNG order is one scalar threshold draw per paid append
with an immediate bounded-integer replacement draw only on a hit, followed by
insertion, deletion, duplication, inversion, transposition, and minimum-padding
draws. Maximum-length and nucleotide-budget tail trims occur after those draws,
consume no further RNG, and do not erase already recorded event counts. The
explicit A4.6b2 resident Torch final-polymer plan is commit authority; the
independently replayed NumPy plan is validation only. Discrete fields and final
bytes are exact, permitted fp64 fields use absolute tolerance at most `2e-12`,
and the registered lesion-mean division permits at most one ULP. Host oracle
values are never substituted.

For source length `T` and final length `L = T + delta`, topology moves from
live `Q=1, S=T`, through the deep-isolated synthetic active state `Q=3, S=2T`,
to final `Q=2, S=T+L`. The A4.6b2 plan is synthetic-relative, so its sequence
delta is `-1` and symbol delta is `delta`; the live net change is one complete
genome and lesion entry plus `L` material symbols. Exact transaction capacity
is `Q=3`, `S=max(2T,T+L)`, `W=max(T,L)`, and the maximum source/synthetic/final
protein and decoded-cache union `P`; every one-short case fails before claim
without clipping, reallocation, or fallback. NUCLEOTIDE pays `T` append symbols
then pays or refunds `delta`, for net cost `L`; ATP follows only the paid append.

After one claim, the bridge appends a new non-aliasing final genome and lesion,
clears template state, installs a fresh empty copy list, publishes cycle and
replication/proofreading telemetry, refreshes the existing gene-cache mapping,
conditionally sets `novel_path_first_age`, and applies the attested tape
after-state to the same live Generator. The existing mutation-ledger object,
key set, and insertion order are preserved; substitution and the five ordered
structural counters are added to their existing values while unknown keys are
untouched. This is a final-state guarantee, not a frozen-CPU internal
write-trace claim. Receipt metadata includes the ordered structural-count list,
final length, exact transaction capacity, selection and tape schedule digests,
combined chain digest, and combined high-level RNG call count.

Immediately before claim, live source/config/`dt`/RNG, selection evidence,
synthetic binding, A4.6b1 tape, NumPy oracle, resident source/tape/plan, direct
net candidate, and fresh final binding are all regenerated or re-attested,
including factory tokens, digests, object associations, non-aliasing storage,
device, pointers, versions, and values. Preclaim failure changes no live state,
RNG, or receipt. Publish failure restores the inactive `None` template, original
empty-copy object, pools, prior genome arrays and genome/lesion outer lists,
nested gene cache, mutation ledger, proteins, damaged proteins, cycle,
telemetry, novel-path age, world energy, and live RNG object/state, then leaves
an aborted outer receipt. Save/load/clone preserve the c7 world/scheduler type,
capacity, config, device, and inherited c1-c6 metadata; active or pending
serialization is rejected and all transaction artifacts remain event-local.

A4.8c7 remains a correctness-only, per-event explicit-readback bridge. It does
not implement ordinary pre-active early-return authority, persistent resident
state, a device RNG, speedup, A4 completion, or a full GPU world-step.

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
- Before the mutation-enabled tape overlay, every paid append symbol equals the
  next template byte exactly.  A recorded substitution may replace that byte
  only with a different symbol from the same frozen alphabet and never changes
  length or payment.
- Fractional progress consumes the entire integer request before material
  gates, matching the frozen CPU even when no symbol can be paid.
- Nucleotide and ATP are subtracted in symbol order with no free DNA, refund,
  clipping, or contribution to world dissipated energy.
- Exact fixed capacity passes; a required additional symbol beyond capacity
  is rejected before any commit.
- The plan is pure: input ragged/cache/physiology, CPU cells, world, RNG, and
  scheduler receipts remain unchanged.
- A4.4b's mutation-free plan still rejects template selection.  A4.5b may plan
  only the attested index-zero, mutation-enabled start above; every other
  selection remains explicitly unsupported.  A4.6a may describe only the
  mutation-free, pre-existing-active, all-row completion above.  No path may
  fall through to a partial GPU result plus a second CPU replication call.

Hydrolysis-tape invariants are:

- The source is one bound cell with a complete post-gain lesion prefix; no
  caller-supplied hazard array is accepted as a second authority.
- Complete-genome order is unchanged and active template/copy slots are never
  marked as genome candidates.
- Eligibility, threshold calls, and conditional bounded-integer calls follow
  the frozen literal order, including one eligible draw at `dt == 0`.
- The supplied PCG64 state is cloned; source binding, CPU cell, world, and live
  RNG remain unchanged.
- Host arrays have canonical tails and a content digest.  Resident ordinary
  checks perform no D2H; explicit readback rechecks the original digest.
- The tape is RNG evidence only.  It contains no deleted polymer, waste or
  lesion update, damage-event delta, cache refresh, or commit authority.

Hydrolysis-deletion-plan invariants are:

- Exactly one source cell and its attested tape are accepted; multi-cell tape
  concatenation remains outside scope.
- Row-padded output carries every used sequence without sorting.  Only hit
  complete genomes shrink by one; template/copy and sequence count are fixed.
- Symbol count, material-symbol count, topology delta, damage-event delta,
  cache dirty state, and literal refresh count all agree with the hit count.
- Waste additions preserve genome order, lesions receive one `* 0.80` per
  hit, and the derived lesion mean uses the registered grouping/one-ULP final-
  division boundary only.
- Deletion only shrinks state.  Capacity is never clipped, repurposed as a
  material trim, or silently grown; allocation failure is fail closed.
- The A4.7b result remains a pure descriptor.  A4.8a consumes only an
  internally regenerated, fully replayed descriptor and commits the one
  declared hydrolysis event; a caller-supplied descriptor is never commit
  authority.
- A4.8a failure before claim leaves biology, RNG, and receipts unchanged.
  Failure while publishing restores the original object identities and values
  and is recorded by the outer scheduler as an aborted step.

## Explicit exclusions through A4.8c7

- No scheduler/world integration other than the single-cell translation,
  pre-existing-active noncompletion, mutation-free completion, mutation-
  enabled completion, mutation-enabled inactive-template-start noncompletion
  replication, mutation-free inactive-template-start noncompletion and
  mutation-free and mutation-enabled same-call-completion replication, and
  hydrolysis event replacements in
  `Hybrid066WorldA4ReplicationStartCompletionMutation`.
- No caller-supplied tape/plan/candidate authority, persistent A4 arena, or
  multi-cell translation/replication/RNG batch. The wrapper rebuilds one
  disposable binding per event and preserves inherited per-cell event/RNG
  interleaving.
- No disabled/no-genome/replicase-gated and other ordinary pre-active-return
  authority. Those replication branches remain outside the A4.8c7 wrapper
  scope; inherited active zero-work noncompletion remains A4.8c1 authority.
- No change to promoted A3 standalone `gene_refresh`, `translation_cpu`,
  `replication_cpu`, or hydrolysis bridge code, and no change to A4.3/A4.8a
  or A4.8b/c1/c2/c3/c4/c5/c6 implementations. Their behavior is replaced only
  inside the new wrapper.
- No device RNG kernel or performance claim; the correctness bridges
  intentionally perform explicit per-cell host/device validation round trips.
- No division, death, corpse/eDNA/HGT, neural, or causal-system port.
- No fp32 claim, mixed precision, `torch.compile`, CUDA Graph, Triton, custom
  CUDA, multi-stream, multi-GPU, or online-GPU abstraction.
- No formal 13-spec/65-measurement benchmark and no speedup claim.

## Acceptance through A4.8c7

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
- A mutation-enabled inactive/active/inactive fixture matches direct Formal066
  state, lesion selection, copied suffix, telemetry, substitution counts, and
  complete PCG64 after-state in literal cell-call order.  Its index-zero start
  is replayed from both an ordinary and a primed uint32-cache state.
- A zero-dt start proves the two sequence slots and template-storage bytes with
  zero threshold draws; a paid start additionally proves all append bytes.
  Exact future sequence/symbol capacities pass and each one-short capacity
  fails atomically.
- NumPy, Torch CPU, and explicit RTX CUDA agree on start topology metadata and
  substitution results.  Start-mask/index tampering is rejected, while
  zero/two-genome, replicase-gated, mutation-off, and completing starts remain
  explicit CPU scope.
- The inherited 31 A4.5b development tests and focused A3 regressions remain
  PASS while A3 `replication_cpu` remains the only scheduler authority.

- Two pre-existing active Formal066 rows complete with the exact historical
  partial-copy prefix, paid suffix, pools, proofreading telemetry, lesion,
  cycle/reset semantics, and unchanged RNG.
- Conceptual post-state sequence/symbol/lesion counts equal the declared
  topology deltas, and A4.2 decode of that conceptual arena matches the CPU
  post-completion gene cache without making it live authority.
- NumPy, Torch CPU, and explicit RTX CUDA fp64 completion descriptors agree;
  resident input pointers and values remain unchanged.
- Legacy completion remains code 3.  Noncompletion, mixed batches, mutation,
  inactive start, and ten structural/schema corruptions fail closed; exact
  current symbol capacity passes without double-counting the paid append.
- All 34 A4 development tests and focused A3 regressions pass while A3
  `replication_cpu` remains the only scheduler authority.

- A primed PCG64 fixture replays paid append substitution and the same cell's
  insertion/deletion/duplication/inversion/transposition/padding before moving
  to the next cell, with complete before/after state exact to Formal066.
- The tape preserves vector-integer call boundaries, short-circuit draw masks,
  operation payloads, pre-trim event counts, final material delta, nucleotide
  refund/payment, and material-budget tail trim.
- NumPy tape validation and Torch CPU/explicit RTX CUDA upload/readback agree
  byte-for-byte; scalar metadata, host array digest, resident pointers/versions,
  sources, world, and live RNG remain unchanged.
- Exact future capacity passes; one-short arena or per-sequence capacity fails
  atomically and is not converted into additional material trim.
- Tampered PCG64, source/config/dt, operation payload, bounds, padding tail,
  material budget/delta, or event telemetry fails the binding-aware replay.
- All 37 A4 development tests and focused A3 regressions pass while A4.6b1
  remains a preparation/attestation slice and A3 `replication_cpu` remains the
  only scheduler authority.

- The attested A4.6b1 operation tape is applied independently by NumPy, Torch
  CPU, and explicit RTX CUDA to produce the same final polymer; no precomputed
  final genome exists in the tape.
- Direct Formal066 completion comparison covers append substitution, all five
  structural operations, minimum-length padding, maximum/material tail trim,
  positive nucleotide payment, negative-delta refund, inherited lesion,
  cycle/reset, topology, and derived genome count/material/lesion mean.
- A pre-structural length-8 completion with zero post-append nucleotide budget
  proves padding draws followed by a valid trim back to length 8.  An exact
  host budget of 19 symbols proves CUDA does not re-truncate the quotient to 18.
- Multiple lesion-prefix lengths preserve the frozen NumPy grouping exactly on
  Torch CPU and CUDA; an adversarial seven-old-plus-one-new prefix is bit-exact.
  Final tails, event telemetry, material deltas, and derived post-state fields
  are fixed-shape and fail closed when forged.
- A large finite template-lesion/reactive case proves the inherited lesion term
  is directly regrouped from template and proofreading inputs; cancellation by
  reversing the already rounded A4.6a lesion total is not accepted.
- NumPy, Torch CPU, and explicit RTX CUDA source/tape values and pointers remain
  unchanged.  A hidden CPU or CUDA `.data` tape change which leaves `_version`
  unchanged is detected before meaning is consumed, returns code 7, and rolls
  back the complete used batch.
- Nine independent completion-plan schema corruptions, capacity/schedule
  disagreement, and resident trust violations are rejected without arena,
  cache, live RNG, world, CPU-cell, or scheduler mutation.
- All 40 A4 development tests and focused A3 regressions pass while A4.6b2
  remains a pure resident descriptor and A3 `replication_cpu` remains the only
  scheduler authority.

- Direct frozen hydrolysis fixtures distinguish the strict lesion and minimum-
  length gates, preserve a random draw for an eligible `dt == 0` genome, and
  reproduce a one-cell hit/miss/hit sequence in exact complete-genome order.
  Full PCG64 cache state and the position-implied polymer, waste, lesion,
  damage-event, and refreshed-cache outcome match the Formal066 oracle.
- Full PCG64 before/after state, fixed tape arrays, host schedule/content
  digests, Torch CPU/explicit RTX CUDA storage/readback, and source/live-RNG
  non-mutation agree; factory, scalar, dictionary, array, stale-source,
  multi-cell, `dt`, RNG, and `.data` tampering fail closed at their declared
  boundary without hidden resident D2H.
- All 43 A4 development tests and focused A3 regressions pass while A4.7a
  remains a single-cell RNG tape only.  A3 hydrolysis bridge and scheduler
  authority remain unchanged, including the explicitly recorded zero-
  probability draw difference.

- Direct Formal066 hit/miss/hit and `dt == 0` fixtures reproduce final
  polymers, sequential waste, lesion attenuation, symbol/material ledgers,
  damage-event delta, cache dirty/refresh count, and derived lesion mean.
  Active template/copy rows remain byte-exact.
- NumPy, Torch CPU, and explicit RTX CUDA row-padded descriptors agree.  A
  large finite three-genome fixture proves the contiguous lesion sum is exact
  while only the final CUDA division uses the registered one-ULP allowance.
- Source/tape/world/live RNG remain unchanged.  Source, `dt`, schedule,
  position, tail, plan, and public/private resident tape corruption fail
  closed; `.data` tape changes produce one global rollback without hidden
  D2H.  Exact capacity and deletion-only shrinkage are preserved.
- All 46 A4 development tests and focused A3 regressions pass while A4.7b is
  a single-cell row-padded pure descriptor.  Compact arena/cache/live RNG/
  CPU-cell/world commit and A3 scheduler authority remain unchanged.

- Direct Formal066 hit/miss/hit, no-hit, and eligible `dt == 0` fixtures match
  the A4.8a commit for compact polymers, lesions, sequential waste, final gene
  cache, damage counter, full PCG64 state, and receipt draw/hit metadata.
- Torch CPU and explicit RTX CUDA candidates agree after explicit readback;
  retained source/tape/plan values, fresh binding provenance, and one-shot
  consumption are checked without making an external descriptor authoritative.
- Exact Q/S/W capacities pass.  One-short capacity, wrong cell or `dt`, stale
  source, scheduler config/device disagreement, candidate PCG64 tampering, and
  resident tape/plan `.data` mutation all fail before claim without changing
  biology, RNG, or receipts.
- Injected failure after claim restores the original list, array, dictionary,
  and RNG identities and values and records an aborted step.  Duplicate and
  out-of-order execution remain rejected by the inherited exact-once scheduler.
- One-step, ten-step, repair-heavy stressed three-step, and two-cell interleave
  fixtures remain lockstep with the frozen world; save/load/clone preserve the
  A4 scheduler type, capacity, device, PCG64 continuation, and next-step
  outcome.  The promoted A3 source and scheduler hashes remain unchanged.
- All 50 A4 development tests pass with explicit CUDA required while the
  promoted A3 validation remains PASS and `full_gpu_world_step=false`.

- Direct Formal066 rich, sequential-exhaustion, ATP-reserve, no-translator,
  disabled, no-genome, `dt == 0`, and signed paid-ledger-residual fixtures
  match the A4.8b resident commit. Discrete state and both dictionary orders
  are exact; fp64 values remain within `2e-12` without clipping residuals.
- Torch CPU and explicit RTX CUDA candidate/commit paths preserve source
  values, device and pointer identity, full PCG64 state, genomes, gene cache,
  replication state, and world energy ledgers. Fresh binding and one-shot
  checks pass, including the frozen early-return versus `_sync_*` dictionary-
  identity branch.
- Exact Q/S/W/P capacities pass. Every one-short capacity, wrong cell or
  `dt`, stale or aliased `gene_specs`, config/device/source disagreement,
  host-oracle mutation, resident ragged/state/cache/plan `.data` mutation,
  candidate/fresh-binding tampering, duplicate/order error, and injected
  publish failure is rejected at its declared atomic boundary.
- One-step, ten-step, translation-heavy stress, two-cell nonbatched event/RNG
  interleave, save/load/clone continuation, and active-save rejection remain
  lockstep with the frozen world while the legacy translation bridge is not
  invoked and the A4.8a hydrolysis override remains active.
- Immediately following the resident commit, Torch CPU and CUDA each match
  direct frozen translate-then-replicate behavior for ATP and nucleotide exact
  versus next-below resource gates and replicase below versus above threshold:
  all twelve discrete-copy/event/full-PCG64 controls agree without fallback.
- All 54 A4 development tests pass with explicit CUDA required while promoted
  A3 source/scheduler bytes remain unchanged and `full_gpu_world_step=false`.

- Direct Formal066 active-partial mutation-off, forced substitution miss/hit,
  `dt == 0`, requested-zero, and resource-stop cases match the A4.8c1 commit.
  Append bytes/order, paid pools, fractional/error/proof telemetry,
  substitution counters, object identities and full PCG64 state are exact.
- Torch CPU and explicit RTX CUDA candidates use resident plan authority,
  preserve preparation purity, produce a valid fresh binding, keep mutation-
  free RNG unchanged, and publish the exact mutation-enabled tape after-state.
- Exact Q/S/W/P capacities pass and each one-short case fails before claim.
  Live/config/source/RNG, host oracle/tape, resident storage/member/device/
  pointer/value, candidate alias, duplicate/order, and injected publish
  corruptions are rejected with the declared no-change or full-rollback
  identity semantics.
- Crafted active-noncompletion one-step, ten-step and two-cell nonbatched
  replication/surface/hydrolysis RNG order, save/load/clone continuation,
  active-save rejection, and legacy replication-bridge bomb pass. The
  following surface-assembly ATP next-below/exact/next-above controls match
  the direct CPU twin on Torch CPU and CUDA.
- All 58 A4 development tests pass with explicit CUDA required while promoted
  A3 and A4.8a/b sources remain unchanged and
  `full_gpu_world_step=false`.

- Direct Formal066 mutation-free active completion matches the A4.8c2 commit
  for the prior partial-copy prefix, paid suffix, completed genome/lesion,
  paid pools, proof/error telemetry, cycle/reset state, refreshed cache,
  conditional novel path, existing object identities, and unchanged PCG64.
- Torch CPU and explicit RTX CUDA candidates use resident A4.6a descriptor
  authority, preserve preparation purity, match the independent NumPy replay,
  produce a valid fresh binding, and reject candidate reuse.
- Exact Q/S/W/P capacities pass and each one-short case fails before claim.
  Used host/resident/candidate value or storage corruption, wrong `dt`,
  mutation-enabled completion, inactive start, disabled replication, duplicate
  use, and injected publish failure obey the declared no-change or bridge-
  touched identity/RNG rollback boundary.
- A4.8c1 noncompletion is explicitly delegated without legacy CPU fallback.
  Completion save/load/clone preserves wrapper/scheduler type, active save is
  rejected, and the following surface event keeps frozen event order.
- All 62 A4 development tests pass with explicit CUDA required while promoted
  A3 and A4.8a/b/c1 sources remain unchanged and
  `full_gpu_world_step=false`.

- Three direct Formal066 mutation-enabled active-completion fixtures match the
  A4.8c3 commit for paid append substitution, all five structural event
  ledgers, final polymers, material charge/refund and post-padding trim, paid
  pools, inherited lesion, topology, refreshed cache, conditional novel-path
  state, and the complete PCG64 before/after state. Existing mutation-ledger
  keys and insertion order are preserved additively.
- Torch CPU and explicit RTX CUDA candidates use the binding-aware A4.6b1 tape
  and A4.6b2 resident final plan as authority, preserve preparation purity,
  match the independent NumPy replay, produce the expected fresh binding, and
  commit the after-state into the same live generator exactly once.
- Exact Q/S/W/P capacities pass and each one-short case fails before claim.
  A legal missing-lesion prefix, host/resident source/tape/plan and `.data`
  corruption, device/pointer/member changes, live/config/RNG/candidate/fresh-
  binding tampering, wrong inputs, duplicate/order errors, excluded branches,
  and injected publish failure obey the declared no-change or bridge-touched
  and explicitly snapshotted rollback boundary.
- Mutation-free completion delegates to A4.8c2 and mutation-free or mutation-
  enabled active noncompletion delegates to A4.8c1 without frozen CPU
  fallback. Save/load/clone preserve A4.8c3 authority, active/pending saves are
  rejected, and the following surface/hydrolysis/motion order and full PCG64
  continuation match the frozen world.
- All 66 A4 development tests pass with explicit CUDA required while promoted
  A3 and prior A4.8a/b/c1/c2 sources remain unchanged and
  `full_gpu_world_step=false`.

- Direct Formal066 mutation-enabled inactive-start fixtures match the A4.8c4
  commit for the selection-plus-append PCG64 order, new non-aliasing template
  array and new replication-copy list, lesion or missing-lesion `0.0`, paid
  pools, fractional/replication telemetry, substitution ledger, and complete
  PCG64 before/after state in the same Generator. `dt == 0`, requested-zero,
  and resource-stopped zero append still report a successful template-start
  work event.
- Torch CPU and explicit RTX CUDA candidates use the A4.5b resident start plan
  as authority, preserve preparation purity, match the independent NumPy
  replay, regenerate the fresh tape from the live binding/RNG, produce the
  expected fresh binding, and reject candidate reuse.
- Exact Q/S/W/P capacities pass and each one-short case fails before claim.
  Host/resident tape or plan corruption, member/device/pointer/version/content
  changes, live/config/RNG/candidate/fresh-binding tampering, duplicate or
  ordering errors, excluded branches, and injected publish failure obey the
  declared no-change or bounded rollback semantics, including success with a
  new copy-list identity and rollback to the original empty list identity and
  `None` template.
- Active sources delegate through A4.8c3 to the A4.8c2/A4.8c1 authorities
  without frozen-CPU fallback. Save/load/clone preserve A4.8c4 authority,
  active/pending saves are rejected, and successor event order and full PCG64
  continuation match the frozen world.
- All 70 A4 development tests pass with explicit CUDA required while promoted
  A3 and prior A4.5b/A4.8a/b/c1/c2/c3 sources remain unchanged and
  `full_gpu_world_step=false`.

- Direct Formal066 mutation-free inactive-start fixtures cover normal and
  missing-lesion sources with primed and unprimed PCG64, plus `dt == 0`,
  requested-zero, and resource-stop. They execute exactly one high-level
  `integers(0, 1)` selection, no mutation draws, preserve the nonzero and extra
  mutation-ledger entries without any write, retain nonzero effective-error
  telemetry, and match the new non-aliasing template/copy, paid pools, and
  complete PCG64 state.
- Torch CPU and explicit RTX CUDA candidates use the private start-capable
  resident deterministic plan as authority, preserve preparation purity,
  match the independent NumPy replay, regenerate the selection-only evidence
  from the live binding/RNG, produce the expected fresh binding, and reject
  candidate reuse.
- Exact `Q + 2`, `S + template_length + append_count`, `W`, and `P` capacities
  pass and every one-short case fails before claim. Source/evidence/host-plan,
  resident `.data`, whole-member/device/pointer/version/content, live/config/
  RNG/candidate/fresh-binding tampering, wrong inputs, excluded scopes,
  duplicate-before-reclassification, and injected publish failure obey the
  declared no-change or `None`-aware bounded rollback semantics.
- Inactive mutation-enabled start delegates to A4.8c4, while active sources
  delegate through A4.8c4 to A4.8c3/A4.8c2/A4.8c1 without frozen-CPU fallback.
  Save/load/clone preserve A4.8c5 authority, active/pending saves are rejected,
  and successor event order and full PCG64 continuation match the frozen world.
- All 74 A4 development tests pass with explicit CUDA required while promoted
  A3 and prior A4 pure/A4.8a/b/c1/c2/c3/c4 sources remain unchanged and
  `full_gpu_world_step=false`.

- Direct Formal066 mutation-free inactive-start/same-call-completion fixtures
  cover normal and missing-lesion sources with primed and unprimed PCG64. The
  four fixed semantics cases complete genomes of length 576 or 496, execute
  exactly one high-level `integers(0, 1)` selection and no mutation draws, and
  match paid pools, final genomes/lesions, cycle/reset and telemetry, derived
  cache, conditional novel-path age, unchanged world energy and mutation-
  ledger final identity/order/keys/values, and complete PCG64 state.
- Torch CPU and explicit RTX 4060 Ti CUDA candidates prove the deep-isolated
  live-to-synthetic binding and use the public resident completion plan as
  authority. All 73 recorded resident pointers remain device-resident and
  stable through preparation; explicit readback matches the independent NumPy
  plan, the direct net candidate/fresh binding, and the frozen CPU result.
- Exact synthetic-peak `Q=3`, `S=2T`, `W=T`, and exact `P` pass on CPU/CUDA;
  every one-short case fails before claim. Nineteen resident trust probes plus
  source/evidence/synthetic/host/final/live tampering, wrong inputs, excluded
  scopes, duplicate/order errors, and injected post-publish corruption obey
  the declared no-change or full identity-preserving rollback semantics.
- Mutation-free start/noncompletion delegates to A4.8c5, inactive mutation-
  enabled start to A4.8c4, and active sources to their A4.8c3/A4.8c2/A4.8c1
  authorities on CPU and CUDA without frozen-CPU fallback. Save/load/clone
  preserve A4.8c6 authority, all inherited active/pending saves are rejected,
  and a legal full-world successor completion preserves event order.
- All 78 A4 development tests pass with explicit CUDA required while promoted
  A3 and prior A4 pure/A4.8a/b/c1/c2/c3/c4/c5 sources remain unchanged and
  `full_gpu_world_step=false`.

- Direct Formal066 mutation-enabled inactive-start/same-call-completion
  fixtures cover deletion/refund (`L=574`), expansion (`L=592`), short
  padding plus budget trim (`L=3`), and primed missing-lesion completion
  (`L=496`). They preserve the literal selection-to-append-to-structural PCG64
  call order and complete state, final polymer/lesion, paid pools, topology,
  six additive mutation counters, unknown ledger key/order, refreshed cache,
  telemetry, and novel-path behavior.
- Torch CPU and explicit RTX 4060 Ti CUDA candidates use the A4.6b2 resident
  final plan as authority. All 109 recorded resident pointers remain on the
  requested device and stable through preparation; the factory-authenticated
  selection/A4.6b1 chain digest, explicit readback, independent NumPy plan,
  fresh final binding, and direct Formal066 state agree.
- Exact `Q=3`, `S=max(2T,T+L)`, `W=max(T,L)`, and exact `P` pass on CPU/CUDA;
  every one-short host case fails before claim. Twenty-four trust mutations, seven
  no-fallback ordinary scopes, duplicate/order errors, and injected
  post-publish corruption obey the declared preclaim no-change or complete
  identity-preserving rollback semantics.
- All c1-c6 authorities delegate correctly on CPU/CUDA without frozen-CPU
  fallback. The owned c7 branch does not delegate, a legal full-world successor
  preserves event order and Formal066 state, save/load/clone retain exact c7
  authority, and all c1-c7 active/pending saves are rejected.
- All 82 A4 development tests pass with explicit CUDA required while promoted
  A3 and prior A4 pure/A4.8a/b/c1/c2/c3/c4/c5/c6 sources and the first 78 test
  names/function AST remain unchanged and `full_gpu_world_step=false`.

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

A4.8a, A4.8b, A4.8c1, A4.8c2, A4.8c3, A4.8c4, A4.8c5, A4.8c6, and A4.8c7
are bounded
exceptions to the earlier pure-descriptor boundary: the opt-in wrapper commits
paid translation, pre-existing-active noncompletion replication, mutation-free
active completion, mutation-enabled active completion, mutation-enabled and
mutation-free inactive template-start noncompletion, mutation-free and
mutation-enabled inactive template-start same-call completion, and one
hydrolysis event only after
complete resident readback/replay. The promoted A3 standalone bridges remain
byte-identical, no persistent resident chain or performance claim is implied,
and pre-active-gate early-return replication authority, division, and the A5/A6
subsystems remain separate later slices.
