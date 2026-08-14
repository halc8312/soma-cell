# SOMA-CELL 0.6.8-GPU A4 contract

Status: A4 development contract, slice A4.1 only. This is not an A4 promotion.

## Authority

- The promoted A3 checkpoint remains the engineering baseline.
- Frozen SOMA-CELL 0.6.6 CPU behavior remains the scientific oracle.
- `full_gpu_world_step` remains `false`.
- A4 may change representation and execution location, but not biology,
  material costs, event order, RNG order, or insertion order.

## A4.1 scope

This first slice implements only the smallest representation boundary needed
for later GPU-primary genome work:

1. Pack one world's cells into one fixed-capacity ragged genome arena.
2. Preserve complete genomes in cell/list order without sorting or clipping.
3. Preserve an active replication template and partial copy as two explicit
   sequences. An empty partial copy remains a real zero-length sequence.
4. Preserve the exact per-cell genome-lesion vector, including the legal
   transient state in which it is shorter than the complete-genome list.
5. Upload the arena once to Torch fp64 on an explicitly selected device and
   keep its storage resident until an explicit readback.
6. Reconstruct the CPU structural genome state atomically after full-batch
   validation, then rebuild the frozen gene cache from complete genomes.

The sequence order within one cell is fixed as:

```text
complete genome 0 ... complete genome N-1,
replication template (only when active),
replication copy (only when active, including length zero)
```

## Fail-closed invariants

- Symbols are integer values in the frozen alphabet `[0, 7]`.
- Offsets start at zero, are monotonic, and terminate at the used count.
- Unused offset slots are `-1`; unused symbol/material slots are zero.
- Cell IDs and cell order are preserved; duplicate IDs are rejected.
- Sequence count for a cell equals `complete_genomes + 2 * active_replication`.
- An active template is non-empty and its copy cannot exceed its length.
- Inactive replication has no copy, template lesion, or fractional progress.
- A lesion vector cannot exceed the number of complete genomes.
- Capacity is checked for the entire batch before allocation or commit.
- `observed == capacity` passes; `observed == capacity + 1` raises
  `A4CapacityError`. Silent slicing, truncation, and automatic growth are
  forbidden.
- Unpack validates and prepares every target before changing the first target.

The replication template is a reference copy and is not additional material.
Material genome symbols remain `complete genomes + partial replication copy`.

## Explicit exclusions from A4.1

- No translation, replication, proofreading, mutation, or hydrolysis kernel.
- No scheduler replacement and no change to A3 `translation_cpu` or
  `replication_cpu` authority.
- No actual division, death, corpse/eDNA/HGT, neural, or causal-system port.
- No fp32 claim, automatic mixed precision, `torch.compile`, CUDA Graph,
  Triton, custom CUDA, multi-stream, multi-GPU, or online-GPU abstraction.
- No formal 13-spec/65-measurement benchmark and no speedup claim.

The next slice may add batched gene decoding and paid translation only after
this representation passes lossless, corruption, capacity, atomicity, and
explicit-CUDA residency tests.

## A4.1 acceptance

- NumPy pack/unpack structural roundtrip is byte/discrete exact.
- NumPy -> Torch fp64 -> NumPy is exact for all discrete fields and fp64 values.
- Corrupt offsets, dtypes, alphabet values, tails, and replication relations
  are rejected before mutation.
- Exact capacity passes and each independent `capacity + 1` case fails
  atomically.
- A short CUDA resident checksum can run after one upload with stable input
  storage pointers; it is only a plumbing test, not a performance claim.
- Focused A3 regression tests still pass.
