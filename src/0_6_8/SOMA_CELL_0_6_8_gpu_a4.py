# coding: utf-8
"""SOMA-CELL 0.6.8-GPU A4 resident ragged-genome development slices.

This development slice intentionally does not replace the promoted A3 world
or any biological phase.  It provides the smallest fail-closed representation
boundary needed to keep variable-length complete genomes, an in-progress
replication template/copy, lesion state, a derived gene cache, and one pure
paid-translation plan resident on an explicit Torch device.  Frozen 0.6.6 CPU
behavior remains the semantic authority; scheduler translation stays on CPU.

No externally observable collection is reordered, clipped, or automatically
resized.  Internal fixed-shape grouping may sort temporary composite keys,
then explicitly restores frozen first-occurrence order.  The adapter validates
the complete batch before allocation and before CPU commit.
"""
from __future__ import division

import copy
import hashlib
import math
import os
import sys
from dataclasses import dataclass, fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a3 as a3

try:
    import torch
except Exception:  # pragma: no cover - NumPy reference remains importable
    torch = None

g2 = a3.g2
s5 = a3.s5
s4 = s5.s4
s65 = a3.s66.s65

BUILD = 'SOMA-CELL 0.6.8-GPU A4.3'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | resident ragged genomes, gene decode, and paid translation plan'
SCHEMA_VERSION = '0.6.8-GPU-A4.1-ragged-genome'
GENE_CACHE_SCHEMA_VERSION = '0.6.8-GPU-A4.2-gene-cache'
TRANSLATION_SCHEMA_VERSION = '0.6.8-GPU-A4.3-paid-translation-state'
FULL_GPU_WORLD_STEP = False
TRANSLATION_LEDGER_ATOL = 2e-12
TRANSLATION_PAID_POOL_INDICES = (
    int(a3.POOL_FUEL), int(a3.POOL_MINERAL), int(a3.POOL_ATP),
)

ALPHABET_SIZE = int(g2.ALPHABET_SIZE)
MAX_FROZEN_GENOME_SYMBOLS = int(g2.MAX_GENOME_LENGTH)
GENE_SPAN = int(g2.GENE_SPAN)
GENE_PAYLOAD = int(g2.GENE_PAYLOAD)
FINGERPRINT_RADIX = int(ALPHABET_SIZE ** GENE_PAYLOAD)
MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY = int(
    np.iinfo(np.int64).max // FINGERPRINT_RADIX
)


class A4Error(RuntimeError):
    """Base class for fail-closed A4 errors."""


class A4CapacityError(A4Error):
    """A complete source batch cannot fit the declared fixed capacities."""


class A4SchemaError(A4Error):
    """A source or packed ragged state violates the A4.1 schema."""


@dataclass(frozen=True)
class GPU068A4Config:
    """Fixed capacities shared by the small A4 development slices."""

    max_cells: int = 64
    max_sequences: int = 256
    max_symbols: int = 65536
    max_sequence_symbols: int = MAX_FROZEN_GENOME_SYMBOLS
    max_proteins_per_cell: int = 256

    def __post_init__(self):
        for name in ('max_cells', 'max_sequences', 'max_symbols',
                     'max_sequence_symbols', 'max_proteins_per_cell'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)):
                raise ValueError('%s must be an integer' % name)
            if int(value) <= 0:
                raise ValueError('%s must be positive' % name)
        if int(self.max_sequence_symbols) > MAX_FROZEN_GENOME_SYMBOLS:
            raise ValueError(
                'max_sequence_symbols exceeds frozen genome limit %d' %
                MAX_FROZEN_GENOME_SYMBOLS
            )
        if int(self.max_cells) > MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY:
            raise ValueError(
                'max_cells exceeds collision-free gene-key limit %d' %
                MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY
            )

    @classmethod
    def from_state(cls, state):
        if isinstance(state, cls):
            return state
        state = dict(state or {})
        allowed = {item.name for item in fields(cls)}
        unknown = set(state).difference(allowed)
        if unknown:
            raise ValueError('unknown A4.1 config fields: %s' % sorted(unknown))
        return cls(**state)


_ARRAY_FIELDS = (
    'symbols', 'sequence_offsets', 'cell_sequence_offsets',
    'cell_ids', 'cell_mask', 'genome_counts', 'lesion_offsets',
    'genome_lesions', 'replication_active',
    'replication_template_lesions', 'replication_fractional',
)

_UINT8_FIELDS = ('symbols',)
_INT64_FIELDS = (
    'sequence_offsets', 'cell_sequence_offsets', 'cell_ids',
    'genome_counts', 'lesion_offsets',
)
_BOOL_FIELDS = ('cell_mask', 'replication_active')
_FLOAT64_FIELDS = (
    'genome_lesions', 'replication_template_lesions',
    'replication_fractional',
)

_GENE_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'cell_entry_offsets', 'entry_count',
    'entry_mask', 'fingerprints', 'starts', 'payloads', 'copy_numbers',
)
_GENE_INT64_FIELDS = (
    'cell_ids', 'cell_entry_offsets', 'entry_count', 'fingerprints',
    'starts', 'copy_numbers',
)
_GENE_BOOL_FIELDS = ('cell_mask', 'entry_mask')
_GENE_UINT8_FIELDS = ('payloads',)

_TRANSLATION_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'pools', 'membrane', 'membrane_oxidation',
    'damage_trace', 'radius', 'current_stress', 'division_progress',
    'last_translation', 'last_quiescence', 'active_fingerprints',
    'active_mass', 'active_count', 'damaged_fingerprints', 'damaged_mass',
    'damaged_count', 'last_receptor_activity', 'last_control_vectors',
    'last_control_scalars', 'last_edna_signal', 'last_corpse_signal',
    'last_necrotoxin_signal', 'behavioural_quiescence',
    'neural_attachment_osmolyte', 'genome_count', 'genome_lesion_mean',
    'replication_active', 'genome_material_symbols',
)
_TRANSLATION_INT64_FIELDS = (
    'cell_ids', 'active_fingerprints', 'active_count',
    'damaged_fingerprints', 'damaged_count', 'genome_count',
    'genome_material_symbols',
)
_TRANSLATION_BOOL_FIELDS = ('cell_mask', 'replication_active')
_TRANSLATION_FLOAT64_FIELDS = tuple(
    name for name in _TRANSLATION_ARRAY_FIELDS
    if name not in _TRANSLATION_INT64_FIELDS + _TRANSLATION_BOOL_FIELDS
)


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _clone_array(value):
    if _is_tensor(value):
        return value.clone()
    return np.asarray(value).copy()


def _host_array(value):
    if _is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _require_capacity(observed, capacity, label):
    if int(observed) > int(capacity):
        raise A4CapacityError(
            '%s overflow: observed=%d capacity=%d' %
            (label, int(observed), int(capacity))
        )


@dataclass
class A4RaggedGenomeBatch:
    """Fixed-capacity arena for one world's structural genome state.

    Used values occupy prefixes.  Complete genomes retain cell/list order.
    When replication is active, template and copy are the final two sequences
    for that cell; a zero-length copy is represented by equal offsets.
    """

    schema_version: str
    cell_capacity: int
    sequence_capacity: int
    symbol_capacity: int
    max_sequence_symbols: int
    cell_count: int
    sequence_count: int
    symbol_count: int
    lesion_count: int
    symbols: object
    sequence_offsets: object
    cell_sequence_offsets: object
    cell_ids: object
    cell_mask: object
    genome_counts: object
    lesion_offsets: object
    genome_lesions: object
    replication_active: object
    replication_template_lesions: object
    replication_fractional: object

    def clone(self):
        if _is_tensor(self.symbols):
            if getattr(self, '_a4_translation_data_ptrs', None) != self.data_ptrs():
                raise A4SchemaError(
                    'cannot clone changed resident ragged storage'
                )
            expected_versions = getattr(
                self, '_a4_translation_versions', None,
            )
            actual_versions = {
                name: int(getattr(self, name)._version)
                for name in _ARRAY_FIELDS
            }
            if expected_versions != actual_versions:
                raise A4SchemaError(
                    'cannot clone changed resident ragged values'
                )
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value) if item.name in _ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        out = A4RaggedGenomeBatch(**values)
        if hasattr(self, '_a4_translation_provenance'):
            object.__setattr__(
                out, '_a4_translation_provenance',
                copy.deepcopy(self._a4_translation_provenance),
            )
        if _is_tensor(out.symbols):
            object.__setattr__(
                out, '_a4_translation_data_ptrs', out.data_ptrs(),
            )
            object.__setattr__(
                out, '_a4_translation_versions',
                {name: int(getattr(out, name)._version)
                 for name in _ARRAY_FIELDS},
            )
        return out

    def validate(self):
        return validate_a4_ragged(self)

    def to_torch(self, device='cpu'):
        """Perform the explicit host-to-device boundary for this slice."""
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        validate_a4_ragged(self)
        requested = str(device)
        if requested == 'auto':
            raise ValueError('A4.1 requires an explicit cpu or cuda device')
        try:
            resolved_device = torch.device(requested)
        except Exception as exc:
            raise ValueError('invalid A4.1 device: %s' % requested) from exc
        if resolved_device.type not in ('cpu', 'cuda'):
            raise ValueError('A4.1 supports explicit cpu or cuda devices only')
        if resolved_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _UINT8_FIELDS:
                value = torch.tensor(value, dtype=torch.uint8,
                                     device=resolved_device)
            elif item.name in _INT64_FIELDS:
                value = torch.tensor(value, dtype=torch.int64,
                                     device=resolved_device)
            elif item.name in _BOOL_FIELDS:
                value = torch.tensor(value, dtype=torch.bool,
                                     device=resolved_device)
            elif item.name in _FLOAT64_FIELDS:
                value = torch.tensor(value, dtype=torch.float64,
                                     device=resolved_device)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4RaggedGenomeBatch(**values)
        # The NumPy source was fully validated and every conversion above is
        # dtype/shape preserving.  Check device/dtype metadata without a
        # diagnostic D2H readback so residency begins at this boundary.
        _validate_backend_and_dtypes(out)
        object.__setattr__(
            out, '_a4_translation_provenance',
            _ragged_translation_provenance(self),
        )
        object.__setattr__(
            out, '_a4_translation_data_ptrs', out.data_ptrs(),
        )
        object.__setattr__(
            out, '_a4_translation_versions',
            {name: int(getattr(out, name)._version) for name in _ARRAY_FIELDS},
        )
        return out

    def to_numpy(self):
        """Perform an explicit readback and return canonical NumPy dtypes."""
        tensor_backed = _is_tensor(self.symbols)
        if not tensor_backed:
            validate_a4_ragged(self)
        else:
            _validate_backend_and_dtypes(self)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _ARRAY_FIELDS:
                value = _host_array(value)
                if item.name in _UINT8_FIELDS:
                    value = value.astype(np.uint8, copy=False)
                elif item.name in _INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                elif item.name in _BOOL_FIELDS:
                    value = value.astype(bool, copy=False)
                else:
                    value = value.astype(np.float64, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4RaggedGenomeBatch(**values)
        validate_a4_ragged(out)
        return out

    def data_ptrs(self):
        """Return stable input storage identities for a residency smoke test."""
        if not all(_is_tensor(getattr(self, name)) for name in _ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch-backed batch')
        return {name: int(getattr(self, name).data_ptr())
                for name in _ARRAY_FIELDS}

    def resident_symbol_checksum(self):
        """Read-only resident work; this is plumbing, not a biology kernel."""
        if _is_tensor(self.symbols):
            return torch.sum(self.symbols[:int(self.symbol_count)],
                             dtype=torch.int64)
        return np.sum(self.symbols[:int(self.symbol_count)], dtype=np.int64)

    def estimated_bytes(self):
        total = 0
        for name in _ARRAY_FIELDS:
            value = getattr(self, name)
            if _is_tensor(value):
                total += int(value.nelement() * value.element_size())
            else:
                total += int(np.asarray(value).nbytes)
        return total

    def state_dict(self):
        return {item.name: (
            _clone_array(getattr(self, item.name))
            if item.name in _ARRAY_FIELDS
            else copy.deepcopy(getattr(self, item.name))
        ) for item in fields(self)}

    @classmethod
    def from_state_dict(cls, state):
        expected = {item.name for item in fields(cls)}
        if set(state) != expected:
            raise A4SchemaError('ragged state_dict fields differ from schema')
        out = cls(**{key: copy.deepcopy(value)
                     for key, value in state.items()})
        return validate_a4_ragged(out)

    def material_symbol_counts_host(self):
        """Complete genomes plus copy; template is not additional material."""
        host = self.to_numpy() if _is_tensor(self.symbols) else self
        result = np.zeros((host.cell_capacity,), dtype=np.int64)
        for ci in range(host.cell_count):
            first = int(host.cell_sequence_offsets[ci])
            genomes = int(host.genome_counts[ci])
            for si in range(first, first + genomes):
                result[ci] += int(
                    host.sequence_offsets[si + 1] - host.sequence_offsets[si]
                )
            if bool(host.replication_active[ci]):
                copy_index = first + genomes + 1
                result[ci] += int(
                    host.sequence_offsets[copy_index + 1] -
                    host.sequence_offsets[copy_index]
                )
        return result


@dataclass
class A4GeneCacheBatch:
    """Fixed-shape derived cache decoded from complete resident genomes.

    This object is not a second biological authority.  The ragged genome
    arena remains authoritative; this cache can be discarded and rebuilt.
    Used entries form one prefix in cell/list/first-occurrence order.
    """

    schema_version: str
    cell_capacity: int
    entry_capacity: int
    max_sequence_symbols: int
    cell_count: int
    cell_ids: object
    cell_mask: object
    cell_entry_offsets: object
    entry_count: object
    entry_mask: object
    fingerprints: object
    starts: object
    payloads: object
    copy_numbers: object

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value) if item.name in _GENE_ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        return A4GeneCacheBatch(**values)

    def validate(self):
        return validate_a4_gene_cache(self)

    def to_numpy(self):
        """Perform the only supported explicit gene-cache readback."""
        tensor_backed = _is_tensor(self.entry_count)
        if tensor_backed:
            _validate_gene_backend_and_dtypes(self)
        else:
            validate_a4_gene_cache(self)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _GENE_ARRAY_FIELDS:
                value = _host_array(value)
                if item.name in _GENE_UINT8_FIELDS:
                    value = value.astype(np.uint8, copy=False)
                elif item.name in _GENE_INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                else:
                    value = value.astype(bool, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4GeneCacheBatch(**values)
        return validate_a4_gene_cache(out)

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _GENE_ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch-backed gene cache')
        return {name: int(getattr(self, name).data_ptr())
                for name in _GENE_ARRAY_FIELDS}

    def estimated_bytes(self):
        total = 0
        for name in _GENE_ARRAY_FIELDS:
            value = getattr(self, name)
            if _is_tensor(value):
                total += int(value.nelement() * value.element_size())
            else:
                total += int(np.asarray(value).nbytes)
        return total

    def state_dict(self):
        return {item.name: (
            _clone_array(getattr(self, item.name))
            if item.name in _GENE_ARRAY_FIELDS
            else copy.deepcopy(getattr(self, item.name))
        ) for item in fields(self)}

    def materialize_gene_specs_host(self):
        """Return frozen per-cell dictionaries after one explicit readback."""
        host = self.to_numpy() if _is_tensor(self.entry_count) else self
        validate_a4_gene_cache(host)
        result = []
        for ci in range(host.cell_count):
            cache = {}
            first = int(host.cell_entry_offsets[ci])
            last = int(host.cell_entry_offsets[ci + 1])
            for index in range(first, last):
                payload = tuple(int(value) for value in host.payloads[index])
                role = payload[0] % 8
                record = {
                    'start': int(host.starts[index]),
                    'payload': payload,
                    'fingerprint': int(host.fingerprints[index]),
                    'role': role,
                    'role_name': g2.ROLE_NAMES[role],
                    'parameter': payload[1] % 8,
                    'promoter': 0.18 + 1.22 * (payload[3] / 7.0),
                    'efficiency': 0.52 + 0.96 * (payload[4] / 7.0),
                    'fidelity': 0.45 + 0.54 * (payload[5] / 7.0),
                    'localisation': payload[6] % 4,
                    'regulator': payload[2] % 8,
                }
                if role == g2.ROLE_GENERIC:
                    reaction = payload[1] % len(g2.REACTION_NAMES)
                    record['reaction'] = reaction
                    record['reaction_name'] = g2.REACTION_NAMES[reaction]
                record['copy_number'] = int(host.copy_numbers[index])
                cache[record['fingerprint']] = record
            result.append(cache)
        return result


def _validate_gene_backend_and_dtypes(cache):
    kinds = set()
    devices = set()
    for name in _GENE_ARRAY_FIELDS:
        value = getattr(cache, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise A4SchemaError(
                '%s must be a NumPy array or Torch tensor' % name
            )
    if len(kinds) != 1:
        raise A4SchemaError('mixed NumPy/Torch gene cache is forbidden')
    if len(devices) > 1:
        raise A4SchemaError('Torch gene-cache arrays must share one device')
    backend = next(iter(kinds))
    for name in _GENE_UINT8_FIELDS:
        expected = torch.uint8 if backend == 'torch' else np.dtype(np.uint8)
        if getattr(cache, name).dtype != expected:
            raise A4SchemaError('%s must use uint8' % name)
    for name in _GENE_INT64_FIELDS:
        expected = torch.int64 if backend == 'torch' else np.dtype(np.int64)
        if getattr(cache, name).dtype != expected:
            raise A4SchemaError('%s must use int64' % name)
    for name in _GENE_BOOL_FIELDS:
        expected = torch.bool if backend == 'torch' else np.dtype(bool)
        if getattr(cache, name).dtype != expected:
            raise A4SchemaError('%s must use bool' % name)


def validate_a4_gene_cache(cache):
    """Validate a derived cache; Torch use is an explicit D2H boundary."""
    if not isinstance(cache, A4GeneCacheBatch):
        raise A4SchemaError('expected A4GeneCacheBatch')
    if cache.schema_version != GENE_CACHE_SCHEMA_VERSION:
        raise A4SchemaError('gene-cache schema version mismatch')
    for name in ('cell_capacity', 'entry_capacity', 'max_sequence_symbols',
                 'cell_count'):
        value = getattr(cache, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise A4SchemaError('%s must be an integer scalar' % name)
    if cache.cell_capacity <= 0 or cache.entry_capacity < 0:
        raise A4SchemaError('invalid gene-cache capacity')
    if cache.cell_capacity > MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY:
        raise A4SchemaError('cell_capacity exceeds collision-free gene-key limit')
    if cache.max_sequence_symbols <= 0:
        raise A4SchemaError('invalid max_sequence_symbols')
    if cache.max_sequence_symbols > MAX_FROZEN_GENOME_SYMBOLS:
        raise A4SchemaError('max_sequence_symbols exceeds frozen limit')
    if cache.cell_count < 0 or cache.cell_count > cache.cell_capacity:
        raise A4SchemaError('gene-cache cell_count outside capacity')
    _validate_gene_backend_and_dtypes(cache)
    expected_shapes = {
        'cell_ids': (cache.cell_capacity,),
        'cell_mask': (cache.cell_capacity,),
        'cell_entry_offsets': (cache.cell_capacity + 1,),
        'entry_count': (),
        'entry_mask': (cache.entry_capacity,),
        'fingerprints': (cache.entry_capacity,),
        'starts': (cache.entry_capacity,),
        'payloads': (cache.entry_capacity, GENE_PAYLOAD),
        'copy_numbers': (cache.entry_capacity,),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(cache, name).shape) != tuple(shape):
            raise A4SchemaError('%s shape mismatch' % name)

    raw = {name: _host_array(getattr(cache, name))
           for name in _GENE_ARRAY_FIELDS}
    count = int(raw['entry_count'])
    if count < 0 or count > cache.entry_capacity:
        raise A4SchemaError('gene-cache entry_count outside capacity')
    expected_cell_mask = np.arange(cache.cell_capacity) < cache.cell_count
    if not np.array_equal(raw['cell_mask'], expected_cell_mask):
        raise A4SchemaError('gene-cache cell_mask must be a true prefix')
    if np.any(raw['cell_ids'][:cache.cell_count] < 0):
        raise A4SchemaError('active gene-cache cell IDs must be nonnegative')
    if len(set(int(value) for value in
               raw['cell_ids'][:cache.cell_count])) != cache.cell_count:
        raise A4SchemaError('duplicate gene-cache cell IDs are forbidden')
    if np.any(raw['cell_ids'][cache.cell_count:] != -1):
        raise A4SchemaError('unused gene-cache cell IDs must be -1')
    expected_entry_mask = np.arange(cache.entry_capacity) < count
    if not np.array_equal(raw['entry_mask'], expected_entry_mask):
        raise A4SchemaError('entry_mask must be a true prefix')

    offsets = raw['cell_entry_offsets']
    prefix = offsets[:cache.cell_count + 1]
    if int(prefix[0]) != 0 or np.any(np.diff(prefix) < 0):
        raise A4SchemaError('cell_entry_offsets must be zero-based monotonic')
    if int(prefix[-1]) != count:
        raise A4SchemaError('cell_entry_offsets terminal differs from count')
    if np.any(offsets[cache.cell_count + 1:] != -1):
        raise A4SchemaError('unused cell_entry_offsets must be -1')

    if np.any(raw['fingerprints'][:count] < 0):
        raise A4SchemaError('used fingerprint must be nonnegative')
    if np.any(raw['fingerprints'][:count] >= FINGERPRINT_RADIX):
        raise A4SchemaError('fingerprint exceeds the frozen 12-symbol space')
    if np.any(raw['fingerprints'][count:] != -1):
        raise A4SchemaError('unused fingerprints must be -1')
    if np.any(raw['starts'][:count] < 0):
        raise A4SchemaError('used gene starts must be nonnegative')
    if np.any(raw['starts'][:count] + GENE_SPAN >
              cache.max_sequence_symbols):
        raise A4SchemaError('gene start exceeds sequence capacity')
    if np.any(raw['starts'][count:] != -1):
        raise A4SchemaError('unused starts must be -1')
    if np.any(raw['copy_numbers'][:count] <= 0):
        raise A4SchemaError('used copy_numbers must be positive')
    if np.any(raw['copy_numbers'][count:] != 0):
        raise A4SchemaError('unused copy_numbers must be zero')
    if np.any(raw['payloads'][:count] >= ALPHABET_SIZE):
        raise A4SchemaError('payload contains symbol outside frozen alphabet')
    if np.any(raw['payloads'][count:] != 0):
        raise A4SchemaError('unused payload tail must be zero')

    weights = np.asarray(
        [ALPHABET_SIZE ** power
         for power in range(GENE_PAYLOAD - 1, -1, -1)],
        dtype=np.int64,
    )
    expected_fp = np.sum(
        raw['payloads'][:count].astype(np.int64) * weights[None, :],
        axis=1, dtype=np.int64,
    )
    if not np.array_equal(expected_fp, raw['fingerprints'][:count]):
        raise A4SchemaError('fingerprint/payload relation differs')
    for ci in range(cache.cell_count):
        first = int(offsets[ci])
        last = int(offsets[ci + 1])
        values = [int(value) for value in raw['fingerprints'][first:last]]
        if len(values) != len(set(values)):
            raise A4SchemaError('duplicate fingerprint within one cell cache')
    return cache


def _validate_backend_and_dtypes(batch):
    kinds = set()
    devices = set()
    for name in _ARRAY_FIELDS:
        value = getattr(batch, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise A4SchemaError('%s must be a NumPy array or Torch tensor' % name)
    if len(kinds) != 1:
        raise A4SchemaError('mixed NumPy/Torch ragged state is forbidden')
    if len(devices) > 1:
        raise A4SchemaError('Torch ragged arrays must share one device')

    backend = next(iter(kinds))
    for name in _UINT8_FIELDS:
        expected = torch.uint8 if backend == 'torch' else np.dtype(np.uint8)
        if getattr(batch, name).dtype != expected:
            raise A4SchemaError('%s must use uint8' % name)
    for name in _INT64_FIELDS:
        expected = torch.int64 if backend == 'torch' else np.dtype(np.int64)
        if getattr(batch, name).dtype != expected:
            raise A4SchemaError('%s must use int64' % name)
    for name in _BOOL_FIELDS:
        expected = torch.bool if backend == 'torch' else np.dtype(bool)
        if getattr(batch, name).dtype != expected:
            raise A4SchemaError('%s must use bool' % name)
    for name in _FLOAT64_FIELDS:
        expected = torch.float64 if backend == 'torch' else np.dtype(np.float64)
        if getattr(batch, name).dtype != expected:
            raise A4SchemaError('%s must use float64 in A4.1' % name)


def validate_a4_ragged(batch):
    """Validate every relation without mutating the supplied state.

    Torch validation performs an explicit diagnostic readback.  It is a
    boundary check and must not be called inside a steady-state device loop.
    """
    if not isinstance(batch, A4RaggedGenomeBatch):
        raise A4SchemaError('expected A4RaggedGenomeBatch')
    if batch.schema_version != SCHEMA_VERSION:
        raise A4SchemaError('schema version mismatch')
    scalar_names = (
        'cell_capacity', 'sequence_capacity', 'symbol_capacity',
        'max_sequence_symbols', 'cell_count', 'sequence_count',
        'symbol_count', 'lesion_count',
    )
    for name in scalar_names:
        value = getattr(batch, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise A4SchemaError('%s must be an integer scalar' % name)
    for name in ('cell_capacity', 'sequence_capacity', 'symbol_capacity',
                 'max_sequence_symbols'):
        if int(getattr(batch, name)) <= 0:
            raise A4SchemaError('%s must be positive' % name)
    if int(batch.max_sequence_symbols) > MAX_FROZEN_GENOME_SYMBOLS:
        raise A4SchemaError('max_sequence_symbols exceeds frozen limit')
    if int(batch.cell_capacity) > MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY:
        raise A4SchemaError('cell_capacity exceeds collision-free gene-key limit')
    for count_name, capacity_name in (
        ('cell_count', 'cell_capacity'),
        ('sequence_count', 'sequence_capacity'),
        ('symbol_count', 'symbol_capacity'),
        ('lesion_count', 'sequence_capacity'),
    ):
        count = int(getattr(batch, count_name))
        capacity = int(getattr(batch, capacity_name))
        if count < 0 or count > capacity:
            raise A4SchemaError('%s outside capacity' % count_name)
    _validate_backend_and_dtypes(batch)

    expected_shapes = {
        'symbols': (batch.symbol_capacity,),
        'sequence_offsets': (batch.sequence_capacity + 1,),
        'cell_sequence_offsets': (batch.cell_capacity + 1,),
        'cell_ids': (batch.cell_capacity,),
        'cell_mask': (batch.cell_capacity,),
        'genome_counts': (batch.cell_capacity,),
        'lesion_offsets': (batch.cell_capacity + 1,),
        'genome_lesions': (batch.sequence_capacity,),
        'replication_active': (batch.cell_capacity,),
        'replication_template_lesions': (batch.cell_capacity,),
        'replication_fractional': (batch.cell_capacity,),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(batch, name).shape) != tuple(shape):
            raise A4SchemaError('%s shape mismatch' % name)

    raw = {name: _host_array(getattr(batch, name)) for name in _ARRAY_FIELDS}
    cell_count = int(batch.cell_count)
    sequence_count = int(batch.sequence_count)
    symbol_count = int(batch.symbol_count)
    lesion_count = int(batch.lesion_count)

    expected_mask = np.arange(batch.cell_capacity) < cell_count
    if not np.array_equal(raw['cell_mask'], expected_mask):
        raise A4SchemaError('cell_mask must be a true prefix')
    if np.any(raw['cell_ids'][:cell_count] < 0):
        raise A4SchemaError('active cell IDs must be nonnegative')
    active_ids = [int(value) for value in raw['cell_ids'][:cell_count]]
    if len(set(active_ids)) != len(active_ids):
        raise A4SchemaError('duplicate cell IDs are forbidden')
    if np.any(raw['cell_ids'][cell_count:] != -1):
        raise A4SchemaError('unused cell IDs must be -1')
    if np.any(raw['genome_counts'][cell_count:] != 0):
        raise A4SchemaError('unused genome counts must be zero')
    if np.any(raw['replication_active'][cell_count:]):
        raise A4SchemaError('unused replication flags must be false')

    def check_offsets(name, used, terminal):
        values = raw[name]
        prefix = values[:used + 1]
        if int(prefix[0]) != 0:
            raise A4SchemaError('%s must start at zero' % name)
        if np.any(np.diff(prefix) < 0):
            raise A4SchemaError('%s must be monotonic' % name)
        if int(prefix[-1]) != int(terminal):
            raise A4SchemaError('%s terminal offset mismatch' % name)
        if np.any(values[used + 1:] != -1):
            raise A4SchemaError('%s unused tail must be -1' % name)

    check_offsets('sequence_offsets', sequence_count, symbol_count)
    check_offsets('cell_sequence_offsets', cell_count, sequence_count)
    check_offsets('lesion_offsets', cell_count, lesion_count)

    if np.any(raw['symbols'][:symbol_count] >= ALPHABET_SIZE):
        raise A4SchemaError('symbol lies outside frozen alphabet')
    if np.any(raw['symbols'][symbol_count:] != 0):
        raise A4SchemaError('unused symbol tail must be zero')
    if not np.isfinite(raw['genome_lesions']).all():
        raise A4SchemaError('genome lesions must be finite')
    if np.any(raw['genome_lesions'][:lesion_count] < 0.0):
        raise A4SchemaError('genome lesions must be nonnegative')
    if np.any(raw['genome_lesions'][lesion_count:] != 0.0):
        raise A4SchemaError('unused lesion tail must be zero')
    if not np.isfinite(raw['replication_template_lesions']).all():
        raise A4SchemaError('template lesions must be finite')
    if not np.isfinite(raw['replication_fractional']).all():
        raise A4SchemaError('replication fractional values must be finite')

    for si in range(sequence_count):
        length = int(raw['sequence_offsets'][si + 1] -
                     raw['sequence_offsets'][si])
        if length > int(batch.max_sequence_symbols):
            raise A4SchemaError('sequence exceeds max_sequence_symbols')

    for ci in range(cell_count):
        first = int(raw['cell_sequence_offsets'][ci])
        last = int(raw['cell_sequence_offsets'][ci + 1])
        genomes = int(raw['genome_counts'][ci])
        if genomes < 0:
            raise A4SchemaError('genome count must be nonnegative')
        active = bool(raw['replication_active'][ci])
        if last - first != genomes + (2 if active else 0):
            raise A4SchemaError('cell sequence count/replication relation differs')
        lesion_first = int(raw['lesion_offsets'][ci])
        lesion_last = int(raw['lesion_offsets'][ci + 1])
        if lesion_last - lesion_first > genomes:
            raise A4SchemaError('lesion vector exceeds complete genomes')
        template_lesion = float(raw['replication_template_lesions'][ci])
        fractional = float(raw['replication_fractional'][ci])
        if template_lesion < 0.0:
            raise A4SchemaError('template lesion must be nonnegative')
        if fractional < 0.0 or fractional >= 1.0:
            raise A4SchemaError('replication_fractional must be in [0,1)')
        if active:
            template_index = first + genomes
            copy_index = template_index + 1
            template_length = int(
                raw['sequence_offsets'][template_index + 1] -
                raw['sequence_offsets'][template_index]
            )
            copy_length = int(
                raw['sequence_offsets'][copy_index + 1] -
                raw['sequence_offsets'][copy_index]
            )
            if template_length <= 0:
                raise A4SchemaError('active replication template is empty')
            if copy_length > template_length:
                raise A4SchemaError('replication copy exceeds template')
        elif template_lesion != 0.0 or fractional != 0.0:
            raise A4SchemaError(
                'inactive replication has template lesion/fractional state'
            )

    if np.any(raw['replication_template_lesions'][cell_count:] != 0.0):
        raise A4SchemaError('unused template-lesion tail must be zero')
    if np.any(raw['replication_fractional'][cell_count:] != 0.0):
        raise A4SchemaError('unused replication-fractional tail must be zero')
    return batch


def _validate_resident_ragged_metadata(batch):
    """Check a trusted uploaded arena without reading tensor values to host."""
    if not isinstance(batch, A4RaggedGenomeBatch):
        raise A4SchemaError('expected A4RaggedGenomeBatch')
    if batch.schema_version != SCHEMA_VERSION:
        raise A4SchemaError('schema version mismatch')
    for name in (
        'cell_capacity', 'sequence_capacity', 'symbol_capacity',
        'max_sequence_symbols', 'cell_count', 'sequence_count',
        'symbol_count', 'lesion_count',
    ):
        value = getattr(batch, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise A4SchemaError('%s must be an integer scalar' % name)
    for name in ('cell_capacity', 'sequence_capacity', 'symbol_capacity',
                 'max_sequence_symbols'):
        if int(getattr(batch, name)) <= 0:
            raise A4SchemaError('%s must be positive' % name)
    if int(batch.max_sequence_symbols) > MAX_FROZEN_GENOME_SYMBOLS:
        raise A4SchemaError('max_sequence_symbols exceeds frozen limit')
    for count_name, capacity_name in (
        ('cell_count', 'cell_capacity'),
        ('sequence_count', 'sequence_capacity'),
        ('symbol_count', 'symbol_capacity'),
        ('lesion_count', 'sequence_capacity'),
    ):
        count = int(getattr(batch, count_name))
        capacity = int(getattr(batch, capacity_name))
        if count < 0 or count > capacity:
            raise A4SchemaError('%s outside capacity' % count_name)
    if int(batch.cell_capacity) > MAX_CELL_CAPACITY_FOR_FINGERPRINT_KEY:
        raise A4SchemaError('cell_capacity exceeds collision-free gene-key limit')
    _validate_backend_and_dtypes(batch)
    if not all(_is_tensor(getattr(batch, name)) for name in _ARRAY_FIELDS):
        raise A4SchemaError('resident gene decode requires Torch arrays')
    expected_shapes = {
        'symbols': (batch.symbol_capacity,),
        'sequence_offsets': (batch.sequence_capacity + 1,),
        'cell_sequence_offsets': (batch.cell_capacity + 1,),
        'cell_ids': (batch.cell_capacity,),
        'cell_mask': (batch.cell_capacity,),
        'genome_counts': (batch.cell_capacity,),
        'lesion_offsets': (batch.cell_capacity + 1,),
        'genome_lesions': (batch.sequence_capacity,),
        'replication_active': (batch.cell_capacity,),
        'replication_template_lesions': (batch.cell_capacity,),
        'replication_fractional': (batch.cell_capacity,),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(batch, name).shape) != tuple(shape):
            raise A4SchemaError('%s shape mismatch' % name)
    return batch


def _gene_decode_dimensions(batch):
    rounds = int(batch.max_sequence_symbols) // GENE_SPAN
    occurrence_capacity = int(batch.sequence_capacity) * rounds
    entry_capacity = min(
        int(batch.symbol_capacity) // GENE_SPAN,
        occurrence_capacity,
    )
    return rounds, occurrence_capacity, entry_capacity


def _empty_gene_cache_numpy(batch, entry_capacity):
    C = int(batch.cell_capacity)
    G = int(entry_capacity)
    offsets = np.full((C + 1,), -1, dtype=np.int64)
    offsets[:int(batch.cell_count) + 1] = 0
    cache = A4GeneCacheBatch(
        schema_version=GENE_CACHE_SCHEMA_VERSION,
        cell_capacity=C,
        entry_capacity=G,
        max_sequence_symbols=int(batch.max_sequence_symbols),
        cell_count=int(batch.cell_count),
        cell_ids=np.asarray(batch.cell_ids, dtype=np.int64).copy(),
        cell_mask=np.asarray(batch.cell_mask, dtype=bool).copy(),
        cell_entry_offsets=offsets,
        entry_count=np.asarray(0, dtype=np.int64),
        entry_mask=np.zeros((G,), dtype=bool),
        fingerprints=np.full((G,), -1, dtype=np.int64),
        starts=np.full((G,), -1, dtype=np.int64),
        payloads=np.zeros((G, GENE_PAYLOAD), dtype=np.uint8),
        copy_numbers=np.zeros((G,), dtype=np.int64),
    )
    return validate_a4_gene_cache(cache)


def _empty_gene_cache_torch(batch, entry_capacity):
    device = batch.symbols.device
    C = int(batch.cell_capacity)
    G = int(entry_capacity)
    offsets = torch.full((C + 1,), -1, dtype=torch.int64, device=device)
    active_offsets = torch.arange(C + 1, device=device) <= int(batch.cell_count)
    offsets = torch.where(active_offsets, torch.zeros_like(offsets), offsets)
    cache = A4GeneCacheBatch(
        schema_version=GENE_CACHE_SCHEMA_VERSION,
        cell_capacity=C,
        entry_capacity=G,
        max_sequence_symbols=int(batch.max_sequence_symbols),
        cell_count=int(batch.cell_count),
        cell_ids=batch.cell_ids.clone(),
        cell_mask=batch.cell_mask.clone(),
        cell_entry_offsets=offsets,
        entry_count=torch.zeros((), dtype=torch.int64, device=device),
        entry_mask=torch.zeros((G,), dtype=torch.bool, device=device),
        fingerprints=torch.full((G,), -1, dtype=torch.int64, device=device),
        starts=torch.full((G,), -1, dtype=torch.int64, device=device),
        payloads=torch.zeros((G, GENE_PAYLOAD), dtype=torch.uint8,
                             device=device),
        copy_numbers=torch.zeros((G,), dtype=torch.int64, device=device),
    )
    _validate_gene_backend_and_dtypes(cache)
    return cache


def decode_a4_gene_cache_numpy(batch):
    """Independent NumPy transcription of frozen greedy gene decoding."""
    validate_a4_ragged(batch)
    if _is_tensor(batch.symbols):
        raise A4SchemaError('NumPy decode requires a NumPy ragged arena')
    rounds, occurrence_capacity, entry_capacity = _gene_decode_dimensions(batch)
    if rounds == 0 or occurrence_capacity == 0 or entry_capacity == 0:
        return _empty_gene_cache_numpy(batch, entry_capacity)

    C = int(batch.cell_capacity)
    Q = int(batch.sequence_capacity)
    S = int(batch.symbol_capacity)
    cell_count = int(batch.cell_count)
    sequence_count = int(batch.sequence_count)
    width = int(batch.max_sequence_symbols) - GENE_SPAN + 1
    sequence_index = np.arange(Q, dtype=np.int64)
    cell_index = np.arange(C, dtype=np.int64)
    active_cells = cell_index < cell_count
    cell_first = np.where(active_cells, batch.cell_sequence_offsets[:C], 0)
    cell_last = np.where(active_cells, batch.cell_sequence_offsets[1:C + 1], 0)
    membership = (
        active_cells[:, None]
        & (sequence_index[None, :] >= cell_first[:, None])
        & (sequence_index[None, :] < cell_last[:, None])
    )
    sequence_owner = np.argmax(membership, axis=0).astype(np.int64)
    owned = np.any(membership, axis=0)
    sequence_rank = sequence_index - cell_first[sequence_owner]
    complete = (
        (sequence_index < sequence_count)
        & owned
        & (sequence_rank < batch.genome_counts[sequence_owner])
    )
    active_sequences = sequence_index < sequence_count
    sequence_first = np.where(
        active_sequences, batch.sequence_offsets[:Q], 0,
    ).astype(np.int64)
    sequence_last = np.where(
        active_sequences, batch.sequence_offsets[1:Q + 1], 0,
    ).astype(np.int64)
    sequence_lengths = sequence_last - sequence_first

    local = np.arange(width, dtype=np.int64)
    global_start = sequence_first[:, None] + local[None, :]

    def gather(delta):
        indices = np.clip(global_start + int(delta), 0, S - 1)
        return batch.symbols[indices]

    candidates = (
        complete[:, None]
        & (local[None, :] + GENE_SPAN <= sequence_lengths[:, None])
        & (gather(0) == g2.START_MARKER[0])
        & (gather(1) == g2.START_MARKER[1])
        & (gather(GENE_SPAN - 2) == g2.STOP_MARKER[0])
        & (gather(GENE_SPAN - 1) == g2.STOP_MARKER[1])
    )
    sentinel = int(batch.max_sequence_symbols) + 1
    next_allowed = np.zeros((Q,), dtype=np.int64)
    chosen = []
    for _ in range(rounds):
        eligible = candidates & (local[None, :] >= next_allowed[:, None])
        selected = np.min(
            np.where(eligible, local[None, :], sentinel), axis=1,
        ).astype(np.int64)
        found = selected < sentinel
        chosen.append(np.where(found, selected, sentinel))
        next_allowed = np.where(found, selected + GENE_SPAN, sentinel)
    chosen = np.stack(chosen, axis=1)
    raw_sequence = np.repeat(sequence_index, rounds)
    raw_start = chosen.reshape(occurrence_capacity)
    raw_valid = raw_start < sentinel
    safe_start = np.where(raw_valid, raw_start, 0)
    payload_offset = np.arange(2, 2 + GENE_PAYLOAD, dtype=np.int64)
    payload_index = (
        sequence_first[raw_sequence, None]
        + safe_start[:, None]
        + payload_offset[None, :]
    )
    payload_index = np.clip(payload_index, 0, S - 1)
    raw_payload = batch.symbols[payload_index].astype(np.uint8, copy=True)
    raw_payload[~raw_valid] = 0
    weights = np.asarray(
        [ALPHABET_SIZE ** power
         for power in range(GENE_PAYLOAD - 1, -1, -1)],
        dtype=np.int64,
    )
    raw_fingerprint = np.sum(
        raw_payload.astype(np.int64) * weights[None, :],
        axis=1, dtype=np.int64,
    )
    raw_owner = np.repeat(sequence_owner, rounds)
    invalid_key = np.iinfo(np.int64).max
    keys = np.where(
        raw_valid,
        raw_owner * FINGERPRINT_RADIX + raw_fingerprint,
        invalid_key,
    ).astype(np.int64)

    sorted_index = np.argsort(keys, kind='stable')
    sorted_keys = keys[sorted_index]
    sorted_valid = sorted_keys != invalid_key
    group_start = sorted_valid.copy()
    if occurrence_capacity > 1:
        group_start[1:] &= sorted_keys[1:] != sorted_keys[:-1]
    group_ids = np.cumsum(group_start.astype(np.int64)) - 1
    safe_group_ids = np.maximum(group_ids, 0)
    first_by_group = np.full(
        (occurrence_capacity,), occurrence_capacity, dtype=np.int64,
    )
    first_values = np.where(sorted_valid, sorted_index, occurrence_capacity)
    np.minimum.at(first_by_group, safe_group_ids, first_values)
    copies_by_group = np.zeros((occurrence_capacity,), dtype=np.int64)
    np.add.at(copies_by_group, safe_group_ids,
              sorted_valid.astype(np.int64))
    group_count = int(np.sum(group_start, dtype=np.int64))
    if group_count > entry_capacity:
        raise A4CapacityError(
            'decoded gene cache overflow: observed=%d capacity=%d' %
            (group_count, entry_capacity)
        )
    group_order = np.argsort(first_by_group, kind='stable')
    output_group = group_order[:entry_capacity]
    output_valid = np.arange(entry_capacity) < group_count
    output_first = first_by_group[output_group]
    safe_first = np.where(output_valid, output_first, 0)
    output_owner = raw_owner[safe_first]

    fingerprints = np.where(
        output_valid, raw_fingerprint[safe_first], -1,
    ).astype(np.int64)
    starts = np.where(output_valid, raw_start[safe_first], -1).astype(np.int64)
    payloads = raw_payload[safe_first].copy()
    payloads[~output_valid] = 0
    copy_numbers = np.where(
        output_valid, copies_by_group[output_group], 0,
    ).astype(np.int64)
    entry_mask = output_valid.astype(bool)
    cell_counts = np.zeros((C,), dtype=np.int64)
    np.add.at(cell_counts, np.where(output_valid, output_owner, 0),
              output_valid.astype(np.int64))
    prefix = np.concatenate((
        np.zeros((1,), dtype=np.int64),
        np.cumsum(cell_counts, dtype=np.int64),
    ))
    cell_entry_offsets = np.full((C + 1,), -1, dtype=np.int64)
    cell_entry_offsets[:cell_count + 1] = prefix[:cell_count + 1]
    cache = A4GeneCacheBatch(
        schema_version=GENE_CACHE_SCHEMA_VERSION,
        cell_capacity=C,
        entry_capacity=entry_capacity,
        max_sequence_symbols=int(batch.max_sequence_symbols),
        cell_count=cell_count,
        cell_ids=batch.cell_ids.copy(),
        cell_mask=batch.cell_mask.copy(),
        cell_entry_offsets=cell_entry_offsets,
        entry_count=np.asarray(group_count, dtype=np.int64),
        entry_mask=entry_mask,
        fingerprints=fingerprints,
        starts=starts,
        payloads=payloads,
        copy_numbers=copy_numbers,
    )
    return validate_a4_gene_cache(cache)


def decode_a4_gene_cache_torch(batch):
    """Fixed-shape Torch decode with no scalar readback or dynamic output."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    _validate_resident_ragged_metadata(batch)
    rounds, occurrence_capacity, entry_capacity = _gene_decode_dimensions(batch)
    if rounds == 0 or occurrence_capacity == 0 or entry_capacity == 0:
        return _empty_gene_cache_torch(batch, entry_capacity)

    device = batch.symbols.device
    C = int(batch.cell_capacity)
    Q = int(batch.sequence_capacity)
    S = int(batch.symbol_capacity)
    cell_count = int(batch.cell_count)
    sequence_count = int(batch.sequence_count)
    width = int(batch.max_sequence_symbols) - GENE_SPAN + 1
    sequence_index = torch.arange(Q, dtype=torch.int64, device=device)
    cell_index = torch.arange(C, dtype=torch.int64, device=device)
    active_cells = cell_index < cell_count
    cell_first = torch.where(
        active_cells, batch.cell_sequence_offsets[:C],
        torch.zeros((C,), dtype=torch.int64, device=device),
    )
    cell_last = torch.where(
        active_cells, batch.cell_sequence_offsets[1:C + 1],
        torch.zeros((C,), dtype=torch.int64, device=device),
    )
    membership = (
        active_cells[:, None]
        & (sequence_index[None, :] >= cell_first[:, None])
        & (sequence_index[None, :] < cell_last[:, None])
    )
    sequence_owner = torch.argmax(membership.to(torch.int64), dim=0)
    owned = torch.any(membership, dim=0)
    sequence_rank = sequence_index - cell_first[sequence_owner]
    complete = (
        (sequence_index < sequence_count)
        & owned
        & (sequence_rank < batch.genome_counts[sequence_owner])
    )
    active_sequences = sequence_index < sequence_count
    zero_q = torch.zeros((Q,), dtype=torch.int64, device=device)
    sequence_first = torch.where(
        active_sequences, batch.sequence_offsets[:Q], zero_q,
    )
    sequence_last = torch.where(
        active_sequences, batch.sequence_offsets[1:Q + 1], zero_q,
    )
    sequence_lengths = sequence_last - sequence_first
    local = torch.arange(width, dtype=torch.int64, device=device)
    global_start = sequence_first[:, None] + local[None, :]

    def gather(delta):
        indices = torch.clamp(global_start + int(delta), min=0, max=S - 1)
        return batch.symbols[indices]

    candidates = (
        complete[:, None]
        & (local[None, :] + GENE_SPAN <= sequence_lengths[:, None])
        & (gather(0) == int(g2.START_MARKER[0]))
        & (gather(1) == int(g2.START_MARKER[1]))
        & (gather(GENE_SPAN - 2) == int(g2.STOP_MARKER[0]))
        & (gather(GENE_SPAN - 1) == int(g2.STOP_MARKER[1]))
    )
    sentinel = int(batch.max_sequence_symbols) + 1
    next_allowed = torch.zeros((Q,), dtype=torch.int64, device=device)
    chosen = []
    for _ in range(rounds):
        eligible = candidates & (local[None, :] >= next_allowed[:, None])
        selected = torch.amin(
            torch.where(
                eligible, local[None, :],
                torch.full_like(local[None, :], sentinel),
            ),
            dim=1,
        )
        found = selected < sentinel
        chosen.append(torch.where(
            found, selected, torch.full_like(selected, sentinel),
        ))
        next_allowed = torch.where(
            found, selected + GENE_SPAN,
            torch.full_like(selected, sentinel),
        )
    chosen = torch.stack(chosen, dim=1)
    raw_sequence = torch.repeat_interleave(sequence_index, rounds)
    raw_start = chosen.reshape(occurrence_capacity)
    raw_valid = raw_start < sentinel
    safe_start = torch.where(raw_valid, raw_start, torch.zeros_like(raw_start))
    payload_offset = torch.arange(
        2, 2 + GENE_PAYLOAD, dtype=torch.int64, device=device,
    )
    payload_index = (
        sequence_first[raw_sequence, None]
        + safe_start[:, None]
        + payload_offset[None, :]
    )
    payload_index = torch.clamp(payload_index, min=0, max=S - 1)
    raw_payload = batch.symbols[payload_index]
    raw_payload = torch.where(
        raw_valid[:, None], raw_payload, torch.zeros_like(raw_payload),
    )
    weights = torch.tensor(
        [ALPHABET_SIZE ** power
         for power in range(GENE_PAYLOAD - 1, -1, -1)],
        dtype=torch.int64, device=device,
    )
    raw_fingerprint = torch.sum(
        raw_payload.to(torch.int64) * weights[None, :], dim=1,
        dtype=torch.int64,
    )
    raw_owner = torch.repeat_interleave(sequence_owner, rounds)
    invalid_key = torch.iinfo(torch.int64).max
    keys = torch.where(
        raw_valid,
        raw_owner * FINGERPRINT_RADIX + raw_fingerprint,
        torch.full_like(raw_fingerprint, invalid_key),
    )

    sorted_index = torch.argsort(keys, stable=True)
    sorted_keys = keys[sorted_index]
    sorted_valid = sorted_keys != invalid_key
    previous_differs = torch.ones_like(sorted_valid)
    previous_differs[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_start = sorted_valid & previous_differs
    group_ids = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    safe_group_ids = torch.clamp(group_ids, min=0)
    first_by_group = torch.full(
        (occurrence_capacity,), occurrence_capacity,
        dtype=torch.int64, device=device,
    )
    first_values = torch.where(
        sorted_valid, sorted_index,
        torch.full_like(sorted_index, occurrence_capacity),
    )
    first_by_group.scatter_reduce_(
        0, safe_group_ids, first_values, reduce='amin', include_self=True,
    )
    copies_by_group = torch.zeros(
        (occurrence_capacity,), dtype=torch.int64, device=device,
    )
    copies_by_group.scatter_add_(
        0, safe_group_ids, sorted_valid.to(torch.int64),
    )
    group_count = torch.sum(group_start.to(torch.int64), dtype=torch.int64)
    group_order = torch.argsort(first_by_group, stable=True)
    output_group = group_order[:entry_capacity]
    output_valid = (
        torch.arange(entry_capacity, dtype=torch.int64, device=device)
        < group_count
    )
    output_first = first_by_group[output_group]
    safe_first = torch.where(
        output_valid, output_first, torch.zeros_like(output_first),
    )
    output_owner = raw_owner[safe_first]
    fingerprints = torch.where(
        output_valid, raw_fingerprint[safe_first],
        torch.full((entry_capacity,), -1, dtype=torch.int64, device=device),
    )
    starts = torch.where(
        output_valid, raw_start[safe_first],
        torch.full((entry_capacity,), -1, dtype=torch.int64, device=device),
    )
    payloads = torch.where(
        output_valid[:, None], raw_payload[safe_first],
        torch.zeros((entry_capacity, GENE_PAYLOAD), dtype=torch.uint8,
                    device=device),
    )
    copy_numbers = torch.where(
        output_valid, copies_by_group[output_group],
        torch.zeros((entry_capacity,), dtype=torch.int64, device=device),
    )
    cell_counts = torch.zeros((C,), dtype=torch.int64, device=device)
    safe_owner = torch.where(
        output_valid, output_owner, torch.zeros_like(output_owner),
    )
    cell_counts.scatter_add_(0, safe_owner, output_valid.to(torch.int64))
    prefix = torch.cat((
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.cumsum(cell_counts, dim=0),
    ))
    active_offsets = (
        torch.arange(C + 1, dtype=torch.int64, device=device) <= cell_count
    )
    cell_entry_offsets = torch.where(
        active_offsets, prefix,
        torch.full((C + 1,), -1, dtype=torch.int64, device=device),
    )
    cache = A4GeneCacheBatch(
        schema_version=GENE_CACHE_SCHEMA_VERSION,
        cell_capacity=C,
        entry_capacity=entry_capacity,
        max_sequence_symbols=int(batch.max_sequence_symbols),
        cell_count=cell_count,
        cell_ids=batch.cell_ids.clone(),
        cell_mask=batch.cell_mask.clone(),
        cell_entry_offsets=cell_entry_offsets,
        entry_count=group_count.reshape(()),
        entry_mask=output_valid,
        fingerprints=fingerprints,
        starts=starts,
        payloads=payloads,
        copy_numbers=copy_numbers,
    )
    _validate_gene_backend_and_dtypes(cache)
    return cache


def decode_a4_gene_cache(batch):
    """Dispatch without changing the input arena or any CPU cell object."""
    if _is_tensor(batch.symbols):
        return decode_a4_gene_cache_torch(batch)
    return decode_a4_gene_cache_numpy(batch)


def _ragged_translation_provenance(batch):
    """Hash canonical host ragged bytes for the A4.3 trust boundary."""
    if _is_tensor(batch.symbols):
        raise A4SchemaError('ragged provenance is computed before upload')
    validate_a4_ragged(batch)
    digest = hashlib.sha256()
    digest.update(batch.schema_version.encode('utf-8'))
    for name in (
        'cell_capacity', 'sequence_capacity', 'symbol_capacity',
        'max_sequence_symbols', 'cell_count', 'sequence_count',
        'symbol_count', 'lesion_count',
    ):
        digest.update(name.encode('ascii'))
        digest.update(str(int(getattr(batch, name))).encode('ascii'))
        digest.update(b'\0')
    for name in _ARRAY_FIELDS:
        value = np.ascontiguousarray(getattr(batch, name))
        digest.update(name.encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(repr(tuple(value.shape)).encode('ascii'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def _gene_cache_provenance(cache):
    """Hash a validated host cache for one ephemeral translation binding."""
    if _is_tensor(cache.entry_count):
        raise A4SchemaError('host gene-cache provenance cannot read a tensor')
    validate_a4_gene_cache(cache)
    digest = hashlib.sha256()
    digest.update(cache.schema_version.encode('utf-8'))
    for name in (
        'cell_capacity', 'entry_capacity', 'max_sequence_symbols',
        'cell_count',
    ):
        digest.update(name.encode('ascii'))
        digest.update(str(int(getattr(cache, name))).encode('ascii'))
        digest.update(b'\0')
    for name in _GENE_ARRAY_FIELDS:
        value = np.ascontiguousarray(getattr(cache, name))
        digest.update(name.encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(repr(tuple(value.shape)).encode('ascii'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


@dataclass
class A4TranslationStateBatch:
    """Fixed-shape pre/post snapshot for one pure paid-translation plan.

    Protein rows are the literal insertion order of the two frozen Python
    dictionaries.  This disposable state is not a save authority and does not
    own genomes or the gene cache.
    """

    schema_version: str
    cell_capacity: int
    protein_capacity: int
    cell_count: int
    gene_expression: bool
    external_translator: bool
    protein_repair: bool
    quiescence: bool
    quiescence_effector: bool
    source_provenance: str
    cell_ids: object
    cell_mask: object
    pools: object
    membrane: object
    membrane_oxidation: object
    damage_trace: object
    radius: object
    current_stress: object
    division_progress: object
    last_translation: object
    last_quiescence: object
    active_fingerprints: object
    active_mass: object
    active_count: object
    damaged_fingerprints: object
    damaged_mass: object
    damaged_count: object
    last_receptor_activity: object
    last_control_vectors: object
    last_control_scalars: object
    last_edna_signal: object
    last_corpse_signal: object
    last_necrotoxin_signal: object
    behavioural_quiescence: object
    neural_attachment_osmolyte: object
    genome_count: object
    genome_lesion_mean: object
    replication_active: object
    genome_material_symbols: object

    def clone(self):
        if _is_tensor(self.pools):
            if getattr(self, '_a4_translation_data_ptrs', None) != self.data_ptrs():
                raise A4SchemaError(
                    'cannot clone changed resident translation storage'
                )
            expected_versions = getattr(
                self, '_a4_translation_versions', None,
            )
            actual_versions = {
                name: int(getattr(self, name)._version)
                for name in _TRANSLATION_ARRAY_FIELDS
            }
            if expected_versions != actual_versions:
                raise A4SchemaError(
                    'cannot clone changed resident translation values'
                )
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value)
                if item.name in _TRANSLATION_ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        out = A4TranslationStateBatch(**values)
        if _is_tensor(out.pools):
            object.__setattr__(
                out, '_a4_translation_data_ptrs', out.data_ptrs(),
            )
            object.__setattr__(
                out, '_a4_translation_versions',
                {name: int(getattr(out, name)._version)
                 for name in _TRANSLATION_ARRAY_FIELDS},
            )
        return out

    def validate(self):
        return validate_a4_translation_state(self)

    def to_torch(self, device='cpu'):
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        validate_a4_translation_state(self)
        requested = torch.device(str(device))
        if requested.type not in ('cpu', 'cuda'):
            raise ValueError('A4.3 supports explicit cpu or cuda devices only')
        if requested.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _TRANSLATION_INT64_FIELDS:
                value = torch.tensor(value, dtype=torch.int64, device=requested)
            elif item.name in _TRANSLATION_BOOL_FIELDS:
                value = torch.tensor(value, dtype=torch.bool, device=requested)
            elif item.name in _TRANSLATION_FLOAT64_FIELDS:
                value = torch.tensor(value, dtype=torch.float64, device=requested)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4TranslationStateBatch(**values)
        _validate_translation_resident_metadata(out)
        object.__setattr__(out, '_a4_translation_data_ptrs', out.data_ptrs())
        object.__setattr__(
            out, '_a4_translation_versions',
            {name: int(getattr(out, name)._version)
             for name in _TRANSLATION_ARRAY_FIELDS},
        )
        return out

    def to_numpy(self):
        if _is_tensor(self.pools):
            _validate_translation_resident_metadata(self)
        else:
            validate_a4_translation_state(self)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _TRANSLATION_ARRAY_FIELDS:
                value = _host_array(value)
                if item.name in _TRANSLATION_INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                elif item.name in _TRANSLATION_BOOL_FIELDS:
                    value = value.astype(bool, copy=False)
                else:
                    value = value.astype(np.float64, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4TranslationStateBatch(**values)
        return validate_a4_translation_state(out)

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _TRANSLATION_ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch translation state')
        return {name: int(getattr(self, name).data_ptr())
                for name in _TRANSLATION_ARRAY_FIELDS}

    def state_dict(self):
        return {item.name: (
            _clone_array(getattr(self, item.name))
            if item.name in _TRANSLATION_ARRAY_FIELDS
            else copy.deepcopy(getattr(self, item.name))
        ) for item in fields(self)}


def _translation_state_provenance(state):
    """Hash one fully validated host physiology snapshot for binding lifetime."""
    if _is_tensor(state.pools):
        raise A4SchemaError('host translation provenance cannot read a tensor')
    validate_a4_translation_state(state)
    digest = hashlib.sha256()
    for item in fields(state):
        name = item.name
        value = getattr(state, name)
        digest.update(name.encode('ascii'))
        if name in _TRANSLATION_ARRAY_FIELDS:
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode('ascii'))
            digest.update(repr(tuple(array.shape)).encode('ascii'))
            digest.update(array.tobytes(order='C'))
        else:
            digest.update(repr(value).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


_TRANSLATION_BINDING_TOKEN = object()


@dataclass(frozen=True, init=False)
class A4TranslationBinding:
    """Ephemeral binding to a cache decoded from this exact ragged object."""

    ragged: A4RaggedGenomeBatch
    cache: A4GeneCacheBatch
    state: A4TranslationStateBatch
    _ragged_provenance: object
    _state_provenance: object
    _ragged_data_ptrs: object
    _ragged_versions: object
    _state_data_ptrs: object
    _state_versions: object
    _cache_provenance: object
    _cache_data_ptrs: object
    _cache_versions: object
    _token: object

    def __init__(self, *args, **kwargs):
        raise A4SchemaError(
            'A4TranslationBinding is created only by bind_a4_translation()'
        )


def _require_translation_binding(binding):
    if (not isinstance(binding, A4TranslationBinding)
            or getattr(binding, '_token', None) is not _TRANSLATION_BINDING_TOKEN):
        raise A4SchemaError('untrusted A4 translation binding')
    ragged = binding.ragged
    state = binding.state
    cache = binding.cache
    if _is_tensor(ragged.symbols):
        if (binding._ragged_data_ptrs != ragged.data_ptrs()
                or binding._state_data_ptrs != state.data_ptrs()):
            raise A4SchemaError('bound resident source storage changed')
        ragged_versions = {
            name: int(getattr(ragged, name)._version)
            for name in _ARRAY_FIELDS
        }
        state_versions = {
            name: int(getattr(state, name)._version)
            for name in _TRANSLATION_ARRAY_FIELDS
        }
        if (binding._ragged_versions != ragged_versions
                or binding._state_versions != state_versions):
            raise A4SchemaError('bound resident source values changed')
        if (getattr(ragged, '_a4_translation_provenance', None)
                != state.source_provenance):
            raise A4SchemaError('bound resident source provenance differs')
    else:
        if binding._ragged_provenance != _ragged_translation_provenance(ragged):
            raise A4SchemaError('bound host ragged values changed')
        if binding._state_provenance != _translation_state_provenance(state):
            raise A4SchemaError('bound host translation values changed')
        _validate_translation_ragged_relation(ragged, state)
    if _is_tensor(cache.entry_count):
        if binding._cache_data_ptrs != cache.data_ptrs():
            raise A4SchemaError(
                'bound resident gene-cache storage changed'
            )
        actual_versions = {
            name: int(getattr(cache, name)._version)
            for name in _GENE_ARRAY_FIELDS
        }
        if binding._cache_versions != actual_versions:
            raise A4SchemaError('bound resident gene-cache values changed')
    elif binding._cache_provenance != _gene_cache_provenance(cache):
        raise A4SchemaError('bound host gene-cache values changed')
    return binding


def _validate_translation_backend_and_dtypes(state):
    kinds = set()
    devices = set()
    for name in _TRANSLATION_ARRAY_FIELDS:
        value = getattr(state, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise A4SchemaError('%s must be a NumPy array or Torch tensor' % name)
    if len(kinds) != 1:
        raise A4SchemaError('mixed NumPy/Torch translation state is forbidden')
    if len(devices) > 1:
        raise A4SchemaError('translation arrays must share one Torch device')
    backend = next(iter(kinds))
    for name in _TRANSLATION_INT64_FIELDS:
        expected = torch.int64 if backend == 'torch' else np.dtype(np.int64)
        if getattr(state, name).dtype != expected:
            raise A4SchemaError('%s must use int64' % name)
    for name in _TRANSLATION_BOOL_FIELDS:
        expected = torch.bool if backend == 'torch' else np.dtype(bool)
        if getattr(state, name).dtype != expected:
            raise A4SchemaError('%s must use bool' % name)
    for name in _TRANSLATION_FLOAT64_FIELDS:
        expected = torch.float64 if backend == 'torch' else np.dtype(np.float64)
        if getattr(state, name).dtype != expected:
            raise A4SchemaError('%s must use float64' % name)
    return backend


def _translation_expected_shapes(state):
    C = int(state.cell_capacity)
    P = int(state.protein_capacity)
    return {
        'cell_ids': (C,), 'cell_mask': (C,),
        'pools': (C, int(a3.POOL_COUNT)),
        'membrane': (C, int(a3.MEMBRANE_SEGMENTS)),
        'membrane_oxidation': (C, int(a3.MEMBRANE_SEGMENTS)),
        'damage_trace': (C, int(a3.MEMBRANE_SEGMENTS)),
        'radius': (C,), 'current_stress': (C,),
        'division_progress': (C,), 'last_translation': (C,),
        'last_quiescence': (C,),
        'active_fingerprints': (C, P), 'active_mass': (C, P),
        'active_count': (C,), 'damaged_fingerprints': (C, P),
        'damaged_mass': (C, P), 'damaged_count': (C,),
        'last_receptor_activity': (C, int(s4.LIGAND_COUNT)),
        'last_control_vectors': (C, int(s4.CONTROL_COUNT), 2),
        'last_control_scalars': (C, int(s4.CONTROL_COUNT)),
        'last_edna_signal': (C,), 'last_corpse_signal': (C,),
        'last_necrotoxin_signal': (C,), 'behavioural_quiescence': (C,),
        'neural_attachment_osmolyte': (C,),
        'genome_count': (C,), 'genome_lesion_mean': (C,),
        'replication_active': (C,), 'genome_material_symbols': (C,),
    }


def _validate_translation_resident_metadata(state):
    if not isinstance(state, A4TranslationStateBatch):
        raise A4SchemaError('expected A4TranslationStateBatch')
    if state.schema_version != TRANSLATION_SCHEMA_VERSION:
        raise A4SchemaError('translation schema version mismatch')
    for name in ('cell_capacity', 'protein_capacity', 'cell_count'):
        value = getattr(state, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise A4SchemaError('%s must be an integer scalar' % name)
    if state.cell_capacity <= 0 or state.protein_capacity <= 0:
        raise A4SchemaError('translation capacities must be positive')
    if state.cell_count < 0 or state.cell_count > state.cell_capacity:
        raise A4SchemaError('translation cell_count outside capacity')
    for name in ('gene_expression', 'external_translator', 'protein_repair',
                 'quiescence', 'quiescence_effector'):
        if not isinstance(getattr(state, name), (bool, np.bool_)):
            raise A4SchemaError('%s must be boolean' % name)
    if not isinstance(state.source_provenance, str) or len(
            state.source_provenance) != 64:
        raise A4SchemaError('translation source provenance must be SHA-256')
    if (state.source_provenance != state.source_provenance.lower()
            or any(character not in '0123456789abcdef'
                   for character in state.source_provenance)):
        raise A4SchemaError(
            'translation source provenance must be lowercase hexadecimal'
        )
    _validate_translation_backend_and_dtypes(state)
    for name, shape in _translation_expected_shapes(state).items():
        if tuple(getattr(state, name).shape) != tuple(shape):
            raise A4SchemaError('%s shape mismatch' % name)
    return state


def validate_a4_translation_state(state):
    """Full host validation; Torch use is an explicit readback boundary."""
    _validate_translation_resident_metadata(state)
    raw = {name: _host_array(getattr(state, name))
           for name in _TRANSLATION_ARRAY_FIELDS}
    C = int(state.cell_capacity)
    P = int(state.protein_capacity)
    N = int(state.cell_count)
    if not np.array_equal(raw['cell_mask'], np.arange(C) < N):
        raise A4SchemaError('translation cell mask is not a true prefix')
    if np.any(raw['cell_ids'][:N] < 0) or len(set(
            int(value) for value in raw['cell_ids'][:N])) != N:
        raise A4SchemaError('translation cell IDs are invalid or duplicated')
    if np.any(raw['cell_ids'][N:] != -1):
        raise A4SchemaError('unused translation cell IDs must be -1')

    float_names = _TRANSLATION_FLOAT64_FIELDS
    if any(not np.isfinite(raw[name]).all() for name in float_names):
        raise A4SchemaError('translation state contains nonfinite values')
    other_pool_indices = tuple(
        index for index in range(int(a3.POOL_COUNT))
        if index not in TRANSLATION_PAID_POOL_INDICES
    )
    if np.any(raw['pools'][:N, TRANSLATION_PAID_POOL_INDICES]
              < -TRANSLATION_LEDGER_ATOL):
        raise A4SchemaError(
            'paid translation pools exceed negative ledger tolerance'
        )
    if np.any(raw['pools'][:N, other_pool_indices] < 0.0):
        raise A4SchemaError('non-payment translation pools are negative')
    nonnegative = (
        'membrane', 'membrane_oxidation', 'damage_trace', 'radius',
        'current_stress', 'division_progress', 'active_mass', 'damaged_mass',
        'last_receptor_activity', 'last_edna_signal', 'last_corpse_signal',
        'last_necrotoxin_signal', 'behavioural_quiescence',
        'neural_attachment_osmolyte', 'genome_lesion_mean',
    )
    if any(np.any(raw[name][:N] < 0.0) for name in nonnegative):
        raise A4SchemaError('translation state contains negative biology')
    if np.any(raw['radius'][:N] <= 0.0):
        raise A4SchemaError('cell radius must be positive')
    if np.any(raw['division_progress'][:N] > 1.0):
        raise A4SchemaError('division progress outside [0,1]')
    if np.any(raw['genome_count'][:N] < 0) or np.any(
            raw['genome_material_symbols'][:N] < 0):
        raise A4SchemaError('negative derived genome metadata')

    for prefix, threshold in (('active', 1e-10), ('damaged', 1e-11)):
        counts = raw[prefix + '_count']
        fingerprints = raw[prefix + '_fingerprints']
        masses = raw[prefix + '_mass']
        if np.any(counts[:N] < 0) or np.any(counts[:N] > P):
            raise A4SchemaError('%s protein count outside capacity' % prefix)
        for ci in range(N):
            count = int(counts[ci])
            used = fingerprints[ci, :count]
            if np.any(used < 0) or np.any(used >= FINGERPRINT_RADIX):
                raise A4SchemaError('%s fingerprint outside range' % prefix)
            if len(set(int(value) for value in used)) != count:
                raise A4SchemaError('duplicate %s fingerprint' % prefix)
            if np.any(masses[ci, :count] <= threshold):
                raise A4SchemaError('%s protein mass is not canonical' % prefix)
            if np.any(fingerprints[ci, count:] != -1) or np.any(
                    masses[ci, count:] != 0.0):
                raise A4SchemaError('%s protein tail is not canonical' % prefix)
        if np.any(counts[N:] != 0) or np.any(fingerprints[N:] != -1) or np.any(
                masses[N:] != 0.0):
            raise A4SchemaError('unused %s cell tail is not canonical' % prefix)

    for ci in range(N):
        active_total = float(sum(float(value) for value in
                                 raw['active_mass'][ci, :raw['active_count'][ci]]))
        damaged_total = float(sum(float(value) for value in
                                  raw['damaged_mass'][ci, :raw['damaged_count'][ci]]))
        if not math.isclose(raw['pools'][ci, a3.POOL_CATALYST], active_total,
                            rel_tol=0.0, abs_tol=2e-12):
            raise A4SchemaError('active protein/material ledger mismatch')
        if not math.isclose(raw['pools'][ci, a3.POOL_DAMAGED_PROTEIN], damaged_total,
                            rel_tol=0.0, abs_tol=2e-12):
            raise A4SchemaError('damaged protein/material ledger mismatch')

    for name in _TRANSLATION_ARRAY_FIELDS:
        if name in ('cell_ids', 'cell_mask', 'active_fingerprints',
                    'active_mass', 'active_count', 'damaged_fingerprints',
                    'damaged_mass', 'damaged_count'):
            continue
        tail = raw[name][N:]
        if tail.size and np.any(tail != 0):
            raise A4SchemaError('unused %s tail must be zero' % name)
    return state


def _strict_real_array(value, shape, label, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != tuple(shape) or raw.dtype.kind not in 'iuf':
        raise A4SchemaError('%s must be a real numeric array of shape %s' %
                            (label, tuple(shape)))
    out = raw.astype(np.float64, copy=True)
    if not np.isfinite(out).all() or (nonnegative and np.any(out < 0.0)):
        raise A4SchemaError('%s contains invalid values' % label)
    return out


def _strict_bool_flag(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise A4SchemaError('%s must be boolean' % label)
    return bool(value)


def _gene_specs_exact(left, right):
    if list(left.keys()) != list(right.keys()):
        return False
    for fingerprint in left:
        a = left[fingerprint]
        b = right[fingerprint]
        if list(a.keys()) != list(b.keys()):
            return False
        for key in a:
            if isinstance(a[key], np.ndarray) or isinstance(b[key], np.ndarray):
                if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                    return False
            elif a[key] != b[key]:
                return False
    return True


class FullFidelityA4TranslationAdapter:
    """One-time CPU snapshot builder for the pure A4.3 translation plan."""

    def __init__(self, config=None):
        self.config = GPU068A4Config.from_state(config or {})

    def pack_cells(self, cells, ragged, model_config):
        cells = list(cells)
        validate_a4_ragged(ragged)
        if _is_tensor(ragged.symbols):
            raise A4SchemaError('translation packing requires a NumPy ragged source')
        if ragged.cell_capacity != self.config.max_cells:
            raise A4SchemaError('ragged/config cell capacity mismatch')
        if len(cells) != ragged.cell_count:
            raise A4SchemaError('translation cell count differs from ragged source')
        ids = []
        for ci, cell in enumerate(cells):
            value = getattr(cell, 'cell_id', None)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)):
                raise A4SchemaError('cell[%d] cell_id must be an integer' % ci)
            ids.append(int(value))
        if ids != [int(value) for value in ragged.cell_ids[:ragged.cell_count]]:
            raise A4SchemaError('translation cell ID/order differs from ragged source')

        cache = decode_a4_gene_cache_numpy(ragged)
        expected_specs = cache.materialize_gene_specs_host()
        for ci, cell in enumerate(cells):
            if not hasattr(cell, 'gene_specs') or not _gene_specs_exact(
                    cell.gene_specs, expected_specs[ci]):
                raise A4SchemaError('cell[%d] gene cache is stale or reordered' % ci)

        flags = {}
        for name in ('gene_expression', 'external_translator', 'protein_repair',
                     'quiescence'):
            if not hasattr(model_config, name):
                raise A4SchemaError('model config lacks %s' % name)
            flags[name] = _strict_bool_flag(getattr(model_config, name), name)
        flags['quiescence_effector'] = _strict_bool_flag(
            getattr(model_config, 'quiescence_effector', False),
            'quiescence_effector',
        )

        C = int(self.config.max_cells)
        P = int(self.config.max_proteins_per_cell)
        N = len(cells)
        arrays = {
            'cell_ids': np.full((C,), -1, dtype=np.int64),
            'cell_mask': np.zeros((C,), dtype=bool),
            'pools': np.zeros((C, a3.POOL_COUNT), dtype=np.float64),
            'membrane': np.zeros((C, a3.MEMBRANE_SEGMENTS), dtype=np.float64),
            'membrane_oxidation': np.zeros((C, a3.MEMBRANE_SEGMENTS), dtype=np.float64),
            'damage_trace': np.zeros((C, a3.MEMBRANE_SEGMENTS), dtype=np.float64),
            'radius': np.zeros((C,), dtype=np.float64),
            'current_stress': np.zeros((C,), dtype=np.float64),
            'division_progress': np.zeros((C,), dtype=np.float64),
            'last_translation': np.zeros((C,), dtype=np.float64),
            'last_quiescence': np.zeros((C,), dtype=np.float64),
            'active_fingerprints': np.full((C, P), -1, dtype=np.int64),
            'active_mass': np.zeros((C, P), dtype=np.float64),
            'active_count': np.zeros((C,), dtype=np.int64),
            'damaged_fingerprints': np.full((C, P), -1, dtype=np.int64),
            'damaged_mass': np.zeros((C, P), dtype=np.float64),
            'damaged_count': np.zeros((C,), dtype=np.int64),
            'last_receptor_activity': np.zeros((C, s4.LIGAND_COUNT), dtype=np.float64),
            'last_control_vectors': np.zeros((C, s4.CONTROL_COUNT, 2), dtype=np.float64),
            'last_control_scalars': np.zeros((C, s4.CONTROL_COUNT), dtype=np.float64),
            'last_edna_signal': np.zeros((C,), dtype=np.float64),
            'last_corpse_signal': np.zeros((C,), dtype=np.float64),
            'last_necrotoxin_signal': np.zeros((C,), dtype=np.float64),
            'behavioural_quiescence': np.zeros((C,), dtype=np.float64),
            'neural_attachment_osmolyte': np.zeros((C,), dtype=np.float64),
            'genome_count': np.zeros((C,), dtype=np.int64),
            'genome_lesion_mean': np.zeros((C,), dtype=np.float64),
            'replication_active': np.zeros((C,), dtype=bool),
            'genome_material_symbols': np.zeros((C,), dtype=np.int64),
        }
        material_symbols = ragged.material_symbol_counts_host()
        for ci, cell in enumerate(cells):
            arrays['cell_ids'][ci] = ids[ci]
            arrays['cell_mask'][ci] = True
            arrays['pools'][ci] = _strict_real_array(
                cell.pools, (a3.POOL_COUNT,), 'cell[%d].pools' % ci)
            if np.any(arrays['pools'][ci, TRANSLATION_PAID_POOL_INDICES]
                      < -TRANSLATION_LEDGER_ATOL):
                raise A4SchemaError(
                    'cell[%d] paid pools exceed negative ledger tolerance' % ci
                )
            other_pool_indices = tuple(
                index for index in range(int(a3.POOL_COUNT))
                if index not in TRANSLATION_PAID_POOL_INDICES
            )
            if np.any(arrays['pools'][ci, other_pool_indices] < 0.0):
                raise A4SchemaError(
                    'cell[%d] non-payment pools are negative' % ci
                )
            arrays['membrane'][ci] = _strict_real_array(
                cell.membrane, (a3.MEMBRANE_SEGMENTS,),
                'cell[%d].membrane' % ci, True)
            arrays['membrane_oxidation'][ci] = _strict_real_array(
                cell.membrane_oxidation, (a3.MEMBRANE_SEGMENTS,),
                'cell[%d].membrane_oxidation' % ci, True)
            arrays['damage_trace'][ci] = _strict_real_array(
                cell.damage_trace, (a3.MEMBRANE_SEGMENTS,),
                'cell[%d].damage_trace' % ci, True)
            for name in ('radius', 'current_stress', 'division_progress',
                         'last_translation', 'last_quiescence',
                         'last_edna_signal', 'last_corpse_signal',
                         'last_necrotoxin_signal', 'behavioural_quiescence'):
                if not hasattr(cell, name):
                    raise A4SchemaError('cell[%d] lacks %s' % (ci, name))
                value = _strict_float_scalar(
                    getattr(cell, name), 'cell[%d].%s' % (ci, name))
                if name not in ('last_translation', 'last_quiescence') and value < 0.0:
                    raise A4SchemaError('cell[%d].%s must be nonnegative' % (ci, name))
                arrays[name][ci] = value
            arrays['last_receptor_activity'][ci] = _strict_real_array(
                cell.last_receptor_activity, (s4.LIGAND_COUNT,),
                'cell[%d].last_receptor_activity' % ci, True)
            arrays['last_control_vectors'][ci] = _strict_real_array(
                cell.last_control_vectors, (s4.CONTROL_COUNT, 2),
                'cell[%d].last_control_vectors' % ci)
            arrays['last_control_scalars'][ci] = _strict_real_array(
                cell.last_control_scalars, (s4.CONTROL_COUNT,),
                'cell[%d].last_control_scalars' % ci)
            if not hasattr(cell, 'neural_attachments') or not isinstance(
                    cell.neural_attachments, dict):
                raise A4SchemaError(
                    'cell[%d].neural_attachments must be an ordered dict' % ci
                )
            neural_osmolyte = 0.0
            for attachment_index, attachment in enumerate(
                    cell.neural_attachments.values()):
                osmolyte_method = getattr(attachment, 'osmolyte', None)
                if not callable(osmolyte_method):
                    raise A4SchemaError(
                        'cell[%d] neural attachment %d lacks osmolyte()' %
                        (ci, attachment_index)
                    )
                contribution = _strict_float_scalar(
                    osmolyte_method(),
                    'cell[%d] neural attachment %d osmolyte' %
                    (ci, attachment_index),
                )
                if contribution < 0.0:
                    raise A4SchemaError(
                        'cell[%d] neural attachment osmolyte is negative' % ci
                    )
                neural_osmolyte += contribution
                if not math.isfinite(neural_osmolyte):
                    raise A4SchemaError(
                        'cell[%d] neural attachment osmolyte overflow' % ci
                    )
            arrays['neural_attachment_osmolyte'][ci] = neural_osmolyte

            gene_keys = list(expected_specs[ci].keys())
            for prefix, mapping, threshold in (
                    ('active', cell.proteins, 1e-10),
                    ('damaged', cell.damaged_proteins, 1e-11)):
                if not isinstance(mapping, dict):
                    raise A4SchemaError('cell[%d].%s proteins must be a dict' %
                                        (ci, prefix))
                keys = []
                masses = []
                for key, amount in mapping.items():
                    if isinstance(key, (bool, np.bool_)) or not isinstance(
                            key, (int, np.integer)):
                        raise A4SchemaError('%s fingerprint must be an integer' % prefix)
                    key = int(key)
                    amount = _strict_float_scalar(amount, '%s protein mass' % prefix)
                    if key < 0 or key >= FINGERPRINT_RADIX or amount <= threshold:
                        raise A4SchemaError('%s protein mapping is not canonical' % prefix)
                    keys.append(key)
                    masses.append(amount)
                if len(keys) != len(set(keys)):
                    raise A4SchemaError('duplicate %s protein fingerprint' % prefix)
                union_count = len(set(keys).union(gene_keys))
                _require_capacity(union_count, P,
                                  'cell[%d] %s protein rows' % (ci, prefix))
                _require_capacity(len(keys), P,
                                  'cell[%d] %s protein rows' % (ci, prefix))
                count_name = prefix + '_count'
                fp_name = prefix + '_fingerprints'
                mass_name = prefix + '_mass'
                arrays[count_name][ci] = len(keys)
                arrays[fp_name][ci, :len(keys)] = keys
                arrays[mass_name][ci, :len(keys)] = masses
            arrays['genome_count'][ci] = int(ragged.genome_counts[ci])
            first = int(ragged.lesion_offsets[ci])
            last = int(ragged.lesion_offsets[ci + 1])
            arrays['genome_lesion_mean'][ci] = (
                float(np.mean(ragged.genome_lesions[first:last]))
                if last > first else 1.0
            )
            arrays['replication_active'][ci] = bool(ragged.replication_active[ci])
            arrays['genome_material_symbols'][ci] = int(material_symbols[ci])

        state = A4TranslationStateBatch(
            schema_version=TRANSLATION_SCHEMA_VERSION,
            cell_capacity=C, protein_capacity=P, cell_count=N,
            gene_expression=flags['gene_expression'],
            external_translator=flags['external_translator'],
            protein_repair=flags['protein_repair'],
            quiescence=flags['quiescence'],
            quiescence_effector=flags['quiescence_effector'],
            source_provenance=_ragged_translation_provenance(ragged),
            **arrays
        )
        return validate_a4_translation_state(state)


def pack_a4_translation_state(cells, ragged, model_config, config=None):
    return FullFidelityA4TranslationAdapter(config).pack_cells(
        cells, ragged, model_config,
    )


def _validate_translation_ragged_relation(ragged, state):
    if _ragged_translation_provenance(ragged) != state.source_provenance:
        raise A4SchemaError('translation/ragged source provenance differs')
    if ragged.cell_capacity != state.cell_capacity or ragged.cell_count != state.cell_count:
        raise A4SchemaError('translation/ragged cell metadata differs')
    if not np.array_equal(ragged.cell_ids, state.cell_ids) or not np.array_equal(
            ragged.cell_mask, state.cell_mask):
        raise A4SchemaError('translation/ragged cell identity differs')
    if not np.array_equal(ragged.genome_counts, state.genome_count):
        raise A4SchemaError('translation/ragged genome count differs')
    if not np.array_equal(ragged.replication_active, state.replication_active):
        raise A4SchemaError('translation/ragged replication state differs')
    if not np.array_equal(ragged.material_symbol_counts_host(),
                          state.genome_material_symbols):
        raise A4SchemaError('translation/ragged material symbols differ')
    means = np.zeros((ragged.cell_capacity,), dtype=np.float64)
    for ci in range(ragged.cell_count):
        first = int(ragged.lesion_offsets[ci])
        last = int(ragged.lesion_offsets[ci + 1])
        means[ci] = (float(np.mean(ragged.genome_lesions[first:last]))
                     if last > first else 1.0)
    if not np.array_equal(means, state.genome_lesion_mean):
        raise A4SchemaError('translation/ragged lesion mean differs')


def bind_a4_translation(ragged, state):
    """Decode the cache internally and create a non-serializable binding."""
    ragged_tensor = _is_tensor(ragged.symbols)
    state_tensor = _is_tensor(state.pools)
    if ragged_tensor != state_tensor:
        raise A4SchemaError('translation binding mixes host and resident state')
    if ragged_tensor:
        _validate_resident_ragged_metadata(ragged)
        _validate_translation_resident_metadata(state)
        if str(ragged.symbols.device) != str(state.pools.device):
            raise A4SchemaError('translation binding devices differ')
        if ragged.cell_capacity != state.cell_capacity or ragged.cell_count != state.cell_count:
            raise A4SchemaError('translation/ragged metadata differs')
        if getattr(ragged, '_a4_translation_provenance', None) != (
                state.source_provenance):
            raise A4SchemaError('resident translation/ragged provenance differs')
        if getattr(ragged, '_a4_translation_data_ptrs', None) != ragged.data_ptrs():
            raise A4SchemaError('resident ragged storage changed after upload')
        expected_versions = getattr(ragged, '_a4_translation_versions', None)
        actual_versions = {
            name: int(getattr(ragged, name)._version) for name in _ARRAY_FIELDS
        }
        if expected_versions != actual_versions:
            raise A4SchemaError('resident ragged values changed after upload')
        if getattr(state, '_a4_translation_data_ptrs', None) != state.data_ptrs():
            raise A4SchemaError('resident translation storage changed after upload')
        expected_state_versions = getattr(
            state, '_a4_translation_versions', None,
        )
        actual_state_versions = {
            name: int(getattr(state, name)._version)
            for name in _TRANSLATION_ARRAY_FIELDS
        }
        if expected_state_versions != actual_state_versions:
            raise A4SchemaError('resident translation values changed after upload')
    else:
        validate_a4_ragged(ragged)
        validate_a4_translation_state(state)
        _validate_translation_ragged_relation(ragged, state)
    cache = decode_a4_gene_cache(ragged)
    if cache.cell_capacity != state.cell_capacity or cache.cell_count != state.cell_count:
        raise A4SchemaError('translation/gene-cache metadata differs')
    binding = object.__new__(A4TranslationBinding)
    object.__setattr__(binding, 'ragged', ragged)
    object.__setattr__(binding, 'cache', cache)
    object.__setattr__(binding, 'state', state)
    if ragged_tensor:
        object.__setattr__(binding, '_ragged_provenance', None)
        object.__setattr__(binding, '_state_provenance', None)
        object.__setattr__(binding, '_ragged_data_ptrs', ragged.data_ptrs())
        object.__setattr__(binding, '_state_data_ptrs', state.data_ptrs())
        object.__setattr__(
            binding, '_ragged_versions',
            {name: int(getattr(ragged, name)._version)
             for name in _ARRAY_FIELDS},
        )
        object.__setattr__(
            binding, '_state_versions',
            {name: int(getattr(state, name)._version)
             for name in _TRANSLATION_ARRAY_FIELDS},
        )
    else:
        object.__setattr__(
            binding, '_ragged_provenance',
            _ragged_translation_provenance(ragged),
        )
        object.__setattr__(
            binding, '_state_provenance',
            _translation_state_provenance(state),
        )
        object.__setattr__(binding, '_ragged_data_ptrs', None)
        object.__setattr__(binding, '_ragged_versions', None)
        object.__setattr__(binding, '_state_data_ptrs', None)
        object.__setattr__(binding, '_state_versions', None)
    if _is_tensor(cache.entry_count):
        object.__setattr__(binding, '_cache_provenance', None)
        object.__setattr__(binding, '_cache_data_ptrs', cache.data_ptrs())
        object.__setattr__(
            binding, '_cache_versions',
            {name: int(getattr(cache, name)._version)
             for name in _GENE_ARRAY_FIELDS},
        )
    else:
        object.__setattr__(
            binding, '_cache_provenance', _gene_cache_provenance(cache),
        )
        object.__setattr__(binding, '_cache_data_ptrs', None)
        object.__setattr__(binding, '_cache_versions', None)
    object.__setattr__(binding, '_token', _TRANSLATION_BINDING_TOKEN)
    return binding


def _translation_dt(value):
    result = _strict_float_scalar(value, 'translation dt')
    if result < 0.0:
        raise A4SchemaError('translation dt must be nonnegative')
    return result


def _numpy_translation_cell_metrics(state, ci):
    pools = state.pools[ci]
    membrane = state.membrane[ci]
    oxidation = state.membrane_oxidation[ci]
    radius = float(state.radius[ci])
    volume = max(0.20, (radius / float(s5.BASE_RADIUS)) ** 2)
    aggregate = float(pools[a3.POOL_AGGREGATE] / volume)
    reactive = float(pools[a3.POOL_REACTIVE] / volume)
    lesion = float(state.genome_lesion_mean[ci])
    genome_factor = 1.0 / (1.0 + 0.85 * lesion)
    active_pool = max(0.0, float(pools[a3.POOL_CATALYST]))
    damaged_pool = float(pools[a3.POOL_DAMAGED_PROTEIN])
    aggregate_pool = float(pools[a3.POOL_AGGREGATE])
    functional = active_pool / max(1e-9, active_pool + damaged_pool + aggregate_pool)
    proteostasis = float(np.clip(
        functional / (1.0 + 3.6 * aggregate), 0.02, 1.0,
    ))
    required = float(g2.base.MEMBRANE_DENSITY) * (
        2.0 * math.pi * radius / a3.MEMBRANE_SEGMENTS
    )
    base_closure = np.clip(
        membrane / (max(required, 1e-9) * float(g2.base.GAP_CLOSURE_THRESHOLD)),
        0.0, 1.0,
    )
    closure_values = np.clip(
        base_closure * np.exp(-1.65 * np.clip(oxidation, 0.0, 2.5)),
        0.0, 1.0,
    )
    closure = float(np.mean(closure_values))
    genome_mass = float(state.genome_material_symbols[ci]) * float(g2.MONOMER_MASS)
    small = float(
        pools[a3.POOL_FUEL] + pools[a3.POOL_MINERAL] + pools[a3.POOL_ATP]
        + pools[a3.POOL_WASTE] + pools[a3.POOL_ALT]
        + pools[a3.POOL_INTERMEDIATE] + pools[a3.POOL_NUCLEOTIDE]
    )
    osmolyte = (
        small
        + 0.25 * float(pools[a3.POOL_MEM_PRECURSOR]
                       + pools[a3.POOL_TRANSPORTER_PRECURSOR])
        + 0.12 * float(pools[a3.POOL_CATALYST])
        + 0.03 * genome_mass
        + 0.18 * float(pools[a3.POOL_DAMAGED_PROTEIN])
        + 0.08 * float(pools[a3.POOL_AGGREGATE])
        + 0.70 * float(pools[a3.POOL_REACTIVE])
        + float(state.neural_attachment_osmolyte[ci])
    )
    capacity_radius = float(np.clip(
        np.sum(membrane) / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY)),
        float(g2.MIN_RADIUS), float(g2.MAX_RADIUS),
    ))
    osmotic = float(np.clip(
        float(s5.BASE_RADIUS) * math.sqrt(max(0.08, osmolyte / float(g2.base.TARGET_OSMOLYTE))),
        float(g2.MIN_RADIUS), float(g2.MAX_RADIUS) * 1.25,
    ))
    tension = max(0.0, osmotic / max(capacity_radius, 1e-8) - 1.0)
    protein_total = max(0.08, active_pool + damaged_pool + aggregate_pool)
    protein_damage = (damaged_pool + 1.8 * aggregate_pool) / protein_total
    weights = np.maximum(membrane, 1e-9)
    membrane_damage = float(np.average(
        np.clip(oxidation, 0.0, 2.5), weights=weights,
    ))
    burden = float(np.clip(
        0.36 * protein_damage + 0.24 * membrane_damage
        + 0.20 * reactive + 0.20 * lesion,
        0.0, 4.0,
    ))
    return {
        'volume': volume, 'aggregate': aggregate, 'reactive': reactive,
        'lesion': lesion, 'genome_factor': genome_factor,
        'proteostasis': proteostasis, 'closure': closure,
        'tension': tension, 'burden': burden,
    }


def _numpy_protein_need(state, ci, spec, metrics, role_activity):
    pools = state.pools[ci]
    role = int(spec['role'])
    if role == a3.ROLE_ENERGY:
        return float(np.clip((0.34 - pools[a3.POOL_ATP]) * 3.0 + 0.35,
                             0.12, 1.8))
    if role == a3.ROLE_MEMBRANE:
        return float(np.clip((1.02 - metrics['closure']) * 4.0
                             + metrics['tension'] * 2.2 + 0.20, 0.10, 2.0))
    if role == a3.ROLE_TRANSPORTER:
        return float(np.clip(
            0.35 + max(0.0, 0.25 - pools[a3.POOL_FUEL])
            + max(0.0, 0.23 - pools[a3.POOL_MINERAL])
            + max(0.0, 0.18 - pools[a3.POOL_ALT])
            + pools[a3.POOL_WASTE], 0.10, 1.8,
        ))
    if role == a3.ROLE_REPLICASE:
        return 1.6 if int(state.genome_count[ci]) < 2 else 0.28
    if role == a3.ROLE_TRANSLATOR:
        return float(np.clip(1.2 - role_activity * 0.22, 0.22, 1.2))
    if role == a3.ROLE_NUCLEOTIDE:
        return float(np.clip((0.22 - pools[a3.POOL_NUCLEOTIDE]) * 5.0 + 0.18,
                             0.08, 1.7))
    if role == a3.ROLE_GENERIC:
        reaction = int(spec.get('reaction', 0))
        source = int(g2.REACTION_SOURCE[reaction])
        return float(np.clip(0.18 + pools[source] * 2.2, 0.10, 1.5))

    localisation = int(spec['localisation'])
    parameter = int(spec['parameter'])
    if localisation == int(s5.LOC_ECOLOGY):
        kind = parameter % 8
        edna = float(state.last_edna_signal[ci])
        if kind == int(s5.ECO_COMPETENCE):
            need = 0.003 + 1.4 * edna
        elif kind == int(s5.ECO_RECOMBINASE):
            need = 0.002 + 0.9 * edna + 0.3 * metrics['lesion']
        elif kind == int(s5.ECO_NUCLEASE):
            need = 0.002 + 0.8 * edna
        elif kind == int(s5.ECO_RESTRICTION):
            need = 0.002 + 0.6 * edna + 0.25 * state.current_stress[ci]
        elif kind == int(s5.ECO_NECROPHAGE):
            need = 0.003 + 1.2 * state.last_corpse_signal[ci]
        elif kind == int(s5.ECO_DETOX):
            need = (0.003 + 1.5 * state.last_necrotoxin_signal[ci]
                    + 0.7 * metrics['reactive'])
        elif kind == int(s5.ECO_VESICLE):
            need = 0.0015 + 0.2 * edna
        elif kind == int(s5.ECO_MOBILE):
            need = 0.012 + 0.55 * edna
        else:
            need = 0.001 + 0.4 * edna
        return float(np.clip(need, 0.001, 1.8))
    if localisation == int(s4.LOC_SENSOR):
        ligand = parameter % int(s4.LIGAND_COUNT)
        return float(np.clip(
            0.15 + 1.4 * state.last_receptor_activity[ci, ligand]
            + 0.25 * (1.0 - metrics['closure']), 0.10, 1.55,
        ))
    if localisation == int(s4.LOC_EFFECTOR):
        channel = int(spec['regulator']) % int(s4.CONTROL_COUNT)
        drive = float(np.linalg.norm(state.last_control_vectors[ci, channel]))
        drive += abs(float(state.last_control_scalars[ci, channel]))
        return float(np.clip(
            0.16 + 1.2 * drive + 0.15 * metrics['burden'], 0.10, 1.55,
        ))

    kind = parameter % int(a3.REPAIR_COUNT)
    if kind == int(a3.REPAIR_ANTIOXIDANT):
        need = 0.030 + 2.8 * metrics['reactive'] + 0.6 * state.current_stress[ci]
    elif kind == int(a3.REPAIR_CHAPERONE):
        need = 0.025 + 3.0 * pools[a3.POOL_DAMAGED_PROTEIN]
    elif kind == int(a3.REPAIR_PROTEASE):
        need = (0.022 + 2.1 * pools[a3.POOL_DAMAGED_PROTEIN]
                + 3.3 * pools[a3.POOL_AGGREGATE])
    elif kind == int(a3.REPAIR_GENOME):
        need = 0.022 + 1.5 * metrics['lesion']
    elif kind == int(a3.REPAIR_SEGREGATION):
        need = 0.016 + metrics['burden'] * (
            0.4 + 1.4 * state.division_progress[ci]
        )
    elif kind == int(a3.REPAIR_PROOFREADING):
        need = (0.022 + 0.5 * metrics['lesion']
                + (0.6 if bool(state.replication_active[ci]) else 0.0))
    elif kind == int(a3.REPAIR_QUIESCENCE):
        need = (0.016 + 1.5 * max(0.0, metrics['burden'] - 0.20)
                + 0.7 * state.current_stress[ci])
    else:
        need = (0.022 + 2.2 * float(np.mean(state.membrane_oxidation[ci]))
                + 1.4 * (1.0 - metrics['closure']))
    return float(np.clip(need, 0.012, 2.2))


def paid_translation_plan_numpy(binding, dt):
    """Literal independent NumPy plan matching Formal066 dynamic dispatch."""
    _require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise A4SchemaError('NumPy translation requires a NumPy binding')
    validate_a4_translation_state(binding.state)
    validate_a4_gene_cache(binding.cache)
    dt = _translation_dt(dt)
    result = binding.state.clone()
    specs_by_cell = binding.cache.materialize_gene_specs_host()
    for ci in range(result.cell_count):
        result.last_translation[ci] = 0.0
        if (not result.gene_expression or int(result.genome_count[ci]) == 0):
            continue
        specs = specs_by_cell[ci]
        active = {
            int(result.active_fingerprints[ci, pi]): float(result.active_mass[ci, pi])
            for pi in range(int(result.active_count[ci]))
        }
        damaged = {
            int(result.damaged_fingerprints[ci, pi]): float(result.damaged_mass[ci, pi])
            for pi in range(int(result.damaged_count[ci]))
        }
        metrics = _numpy_translation_cell_metrics(result, ci)

        def role_activity(role):
            total = 0.0
            for fingerprint, amount in active.items():
                spec = specs.get(fingerprint)
                if spec is not None and int(spec['role']) == int(role):
                    total += amount * spec['efficiency']
            total /= 0.040
            if int(role) != int(a3.ROLE_REGULATOR):
                total *= metrics['proteostasis'] * metrics['genome_factor']
            return float(total)

        def raw_repair(kind):
            total = 0.0
            for fingerprint, amount in active.items():
                spec = specs.get(fingerprint)
                if (spec is not None
                        and int(spec['role']) == int(a3.ROLE_REGULATOR)
                        and int(spec['localisation']) == int(s4.LOC_REPAIR)
                        and int(spec['parameter']) % int(a3.REPAIR_COUNT) == int(kind)):
                    total += amount * spec['efficiency'] * spec['promoter']
            return float(total / 0.014 / (1.0 + 2.6 * metrics['aggregate']))

        translator = role_activity(a3.ROLE_TRANSLATOR)
        if result.external_translator:
            translator += 0.75
        if translator <= 1e-5 or not specs:
            continue
        weighted = []
        total_weight = 0.0
        for fingerprint, spec in specs.items():
            weight = (spec['promoter'] * spec.get('copy_number', 1)
                      * _numpy_protein_need(
                          result, ci, spec, metrics,
                          role_activity(a3.ROLE_TRANSLATOR),
                      ))
            weighted.append((fingerprint, weight))
            total_weight += weight
        if total_weight <= 0.0:
            continue

        if result.quiescence:
            signal = raw_repair(a3.REPAIR_QUIESCENCE)
            need = max(0.0, metrics['burden'] - 0.12) + 0.35 * result.current_stress[ci]
            quiescence = float(np.clip(
                (signal / (0.8 + signal)) * need * 1.45, 0.0, 0.82,
            ))
        else:
            quiescence = 0.0
        if result.quiescence_effector:
            quiescence = float(np.clip(
                max(quiescence, float(result.behavioural_quiescence[ci])),
                0.0, 0.86,
            ))
        result.last_quiescence[ci] = quiescence
        capacity = dt * 0.0060 * translator * (1.0 - 0.72 * quiescence)
        chaperone = raw_repair(a3.REPAIR_CHAPERONE) if result.protein_repair else 0.0
        misfold_fraction = float(np.clip(
            0.010 + 0.055 * metrics['reactive']
            + 0.040 * result.current_stress[ci]
            + 0.020 * metrics['aggregate'] - 0.010 * chaperone,
            0.004, 0.42,
        ))
        for fingerprint, weight in weighted:
            desired = capacity * weight / total_weight
            desired = min(
                desired,
                float(result.pools[ci, a3.POOL_FUEL]) / 0.64,
                float(result.pools[ci, a3.POOL_MINERAL]) / 0.36,
                max(0.0, float(result.pools[ci, a3.POOL_ATP]) - 0.042) / 0.52,
            )
            if desired <= 0.0:
                continue
            result.pools[ci, a3.POOL_FUEL] -= 0.64 * desired
            result.pools[ci, a3.POOL_MINERAL] -= 0.36 * desired
            result.pools[ci, a3.POOL_ATP] -= 0.52 * desired
            misfolded = desired * misfold_fraction
            active[fingerprint] = active.get(fingerprint, 0.0) + desired - misfolded
            damaged[fingerprint] = damaged.get(fingerprint, 0.0) + misfolded
            result.last_translation[ci] += desired / max(dt, 1e-9)
        active = {int(key): float(value) for key, value in active.items()
                  if math.isfinite(float(value)) and float(value) > 1e-10}
        damaged = {int(key): float(value) for key, value in damaged.items()
                   if math.isfinite(float(value)) and float(value) > 1e-11}
        if len(active) > result.protein_capacity or len(damaged) > result.protein_capacity:
            raise A4CapacityError('translation protein result exceeds fixed capacity')
        result.active_fingerprints[ci] = -1
        result.active_mass[ci] = 0.0
        result.active_count[ci] = len(active)
        result.active_fingerprints[ci, :len(active)] = list(active.keys())
        result.active_mass[ci, :len(active)] = list(active.values())
        result.damaged_fingerprints[ci] = -1
        result.damaged_mass[ci] = 0.0
        result.damaged_count[ci] = len(damaged)
        result.damaged_fingerprints[ci, :len(damaged)] = list(damaged.keys())
        result.damaged_mass[ci, :len(damaged)] = list(damaged.values())
        result.pools[ci, a3.POOL_CATALYST] = float(sum(active.values()))
        result.pools[ci, a3.POOL_DAMAGED_PROTEIN] = float(sum(damaged.values()))
    return validate_a4_translation_state(result)


def _torch_ordered_row_sum(values):
    if int(values.shape[1]) == 0:
        return torch.zeros((values.shape[0],), dtype=values.dtype,
                           device=values.device)
    return torch.cumsum(values, dim=1)[:, -1]


def paid_translation_plan_torch(binding, dt):
    """Fixed-shape resident plan; no scalar readback or dynamic output."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    _require_translation_binding(binding)
    state = binding.state
    cache = binding.cache
    if not _is_tensor(state.pools):
        raise A4SchemaError('Torch translation requires a Torch binding')
    _validate_translation_resident_metadata(state)
    _validate_gene_backend_and_dtypes(cache)
    dt = _translation_dt(dt)
    out = state.clone()
    C = int(state.cell_capacity)
    P = int(state.protein_capacity)
    K = int(cache.entry_capacity)
    device = state.pools.device
    dtype = state.pools.dtype
    out.last_translation = torch.where(
        state.cell_mask, torch.zeros_like(state.last_translation),
        state.last_translation,
    )
    if K == 0:
        return out

    rank = torch.arange(K, dtype=torch.int64, device=device)
    first = torch.clamp(cache.cell_entry_offsets[:C], min=0)
    last = torch.clamp(cache.cell_entry_offsets[1:C + 1], min=0)
    counts = torch.clamp(last - first, min=0, max=K)
    indices = first[:, None] + rank[None, :]
    safe_indices = torch.clamp(indices, min=0, max=K - 1)
    valid = (state.cell_mask[:, None] & (rank[None, :] < counts[:, None])
             & cache.entry_mask[safe_indices])
    payload = cache.payloads[safe_indices]
    fingerprints = cache.fingerprints[safe_indices]
    copies = cache.copy_numbers[safe_indices].to(dtype)
    role = torch.remainder(payload[:, :, 0].to(torch.int64), 8)
    parameter = torch.remainder(payload[:, :, 1].to(torch.int64), 8)
    regulator = torch.remainder(payload[:, :, 2].to(torch.int64), 8)
    promoter = 0.18 + 1.22 * payload[:, :, 3].to(dtype) / 7.0
    efficiency = 0.52 + 0.96 * payload[:, :, 4].to(dtype) / 7.0
    localisation = torch.remainder(payload[:, :, 6].to(torch.int64), 4)

    active_position = torch.arange(P, dtype=torch.int64, device=device)
    active_valid = active_position[None, :] < state.active_count[:, None]
    active_match = (
        valid[:, :, None] & active_valid[:, None, :]
        & (fingerprints[:, :, None] == state.active_fingerprints[:, None, :])
    )
    translator_eff = torch.sum(torch.where(
        active_match & (role[:, :, None] == int(a3.ROLE_TRANSLATOR)),
        efficiency[:, :, None], torch.zeros_like(efficiency[:, :, None]),
    ), dim=1)
    repair_factor = efficiency * promoter
    chaperone_factor = torch.sum(torch.where(
        active_match
        & (role[:, :, None] == int(a3.ROLE_REGULATOR))
        & (localisation[:, :, None] == int(s4.LOC_REPAIR))
        & (torch.remainder(parameter[:, :, None], int(a3.REPAIR_COUNT))
           == int(a3.REPAIR_CHAPERONE)),
        repair_factor[:, :, None], torch.zeros_like(repair_factor[:, :, None]),
    ), dim=1)
    quiescence_factor = torch.sum(torch.where(
        active_match
        & (role[:, :, None] == int(a3.ROLE_REGULATOR))
        & (localisation[:, :, None] == int(s4.LOC_REPAIR))
        & (torch.remainder(parameter[:, :, None], int(a3.REPAIR_COUNT))
           == int(a3.REPAIR_QUIESCENCE)),
        repair_factor[:, :, None], torch.zeros_like(repair_factor[:, :, None]),
    ), dim=1)

    pools = state.pools
    volume = torch.clamp((state.radius / float(s5.BASE_RADIUS)) ** 2, min=0.20)
    aggregate = pools[:, a3.POOL_AGGREGATE] / volume
    reactive = pools[:, a3.POOL_REACTIVE] / volume
    genome_factor = 1.0 / (1.0 + 0.85 * state.genome_lesion_mean)
    active_pool = torch.clamp(pools[:, a3.POOL_CATALYST], min=0.0)
    damaged_pool = pools[:, a3.POOL_DAMAGED_PROTEIN]
    aggregate_pool = pools[:, a3.POOL_AGGREGATE]
    functional = active_pool / torch.clamp(
        active_pool + damaged_pool + aggregate_pool, min=1e-9,
    )
    proteostasis = torch.clamp(
        functional / (1.0 + 3.6 * aggregate), min=0.02, max=1.0,
    )
    endogenous_translator = _torch_ordered_row_sum(
        state.active_mass * translator_eff,
    )
    endogenous_translator = (
        endogenous_translator / 0.040 * proteostasis * genome_factor
    )
    translator = endogenous_translator
    if state.external_translator:
        translator = translator + 0.75
    chaperone = _torch_ordered_row_sum(state.active_mass * chaperone_factor)
    chaperone = chaperone / 0.014 / (1.0 + 2.6 * aggregate)
    quiescence_signal = _torch_ordered_row_sum(
        state.active_mass * quiescence_factor,
    ) / 0.014 / (1.0 + 2.6 * aggregate)

    required = float(g2.base.MEMBRANE_DENSITY) * (
        2.0 * math.pi * state.radius / a3.MEMBRANE_SEGMENTS
    )
    base_closure = torch.clamp(
        state.membrane
        / (torch.clamp(required, min=1e-9)[:, None]
           * float(g2.base.GAP_CLOSURE_THRESHOLD)),
        min=0.0, max=1.0,
    )
    closure_values = torch.clamp(
        base_closure * torch.exp(-1.65 * torch.clamp(
            state.membrane_oxidation, min=0.0, max=2.5,
        )), min=0.0, max=1.0,
    )
    closure = torch.mean(closure_values, dim=1)
    genome_mass = state.genome_material_symbols.to(dtype) * float(g2.MONOMER_MASS)
    small = (
        pools[:, a3.POOL_FUEL] + pools[:, a3.POOL_MINERAL]
        + pools[:, a3.POOL_ATP] + pools[:, a3.POOL_WASTE]
        + pools[:, a3.POOL_ALT] + pools[:, a3.POOL_INTERMEDIATE]
        + pools[:, a3.POOL_NUCLEOTIDE]
    )
    osmolyte = (
        small + 0.25 * (pools[:, a3.POOL_MEM_PRECURSOR]
                        + pools[:, a3.POOL_TRANSPORTER_PRECURSOR])
        + 0.12 * pools[:, a3.POOL_CATALYST] + 0.03 * genome_mass
        + 0.18 * pools[:, a3.POOL_DAMAGED_PROTEIN]
        + 0.08 * pools[:, a3.POOL_AGGREGATE]
        + 0.70 * pools[:, a3.POOL_REACTIVE]
        + state.neural_attachment_osmolyte
    )
    capacity_radius = torch.clamp(
        torch.sum(state.membrane, dim=1)
        / (2.0 * math.pi * float(g2.base.MEMBRANE_DENSITY)),
        min=float(g2.MIN_RADIUS), max=float(g2.MAX_RADIUS),
    )
    osmotic = torch.clamp(
        float(s5.BASE_RADIUS) * torch.sqrt(torch.clamp(
            osmolyte / float(g2.base.TARGET_OSMOLYTE), min=0.08,
        )), min=float(g2.MIN_RADIUS), max=float(g2.MAX_RADIUS) * 1.25,
    )
    tension = torch.clamp(
        osmotic / torch.clamp(capacity_radius, min=1e-8) - 1.0, min=0.0,
    )
    protein_total = torch.clamp(active_pool + damaged_pool + aggregate_pool,
                                min=0.08)
    protein_damage = (damaged_pool + 1.8 * aggregate_pool) / protein_total
    membrane_weights = torch.clamp(state.membrane, min=1e-9)
    membrane_damage = torch.sum(
        torch.clamp(state.membrane_oxidation, min=0.0, max=2.5)
        * membrane_weights, dim=1,
    ) / torch.clamp(torch.sum(membrane_weights, dim=1), min=1e-12)
    burden = torch.clamp(
        0.36 * protein_damage + 0.24 * membrane_damage
        + 0.20 * reactive + 0.20 * state.genome_lesion_mean,
        min=0.0, max=4.0,
    )

    need = torch.clamp(
        0.22 + torch.mean(state.damage_trace, dim=1)[:, None] * 3.0,
        min=0.10, max=1.2,
    ).expand(C, K)
    energy_need = torch.clamp(
        (0.34 - pools[:, a3.POOL_ATP]) * 3.0 + 0.35, min=0.12, max=1.8,
    )[:, None]
    membrane_need = torch.clamp(
        (1.02 - closure) * 4.0 + tension * 2.2 + 0.20,
        min=0.10, max=2.0,
    )[:, None]
    transporter_need = torch.clamp(
        0.35 + torch.clamp(0.25 - pools[:, a3.POOL_FUEL], min=0.0)
        + torch.clamp(0.23 - pools[:, a3.POOL_MINERAL], min=0.0)
        + torch.clamp(0.18 - pools[:, a3.POOL_ALT], min=0.0)
        + pools[:, a3.POOL_WASTE], min=0.10, max=1.8,
    )[:, None]
    replicase_need = torch.where(
        state.genome_count < 2,
        torch.full((C,), 1.6, dtype=dtype, device=device),
        torch.full((C,), 0.28, dtype=dtype, device=device),
    )[:, None]
    translator_need = torch.clamp(
        1.2 - endogenous_translator * 0.22, min=0.22, max=1.2,
    )[:, None]
    nucleotide_need = torch.clamp(
        (0.22 - pools[:, a3.POOL_NUCLEOTIDE]) * 5.0 + 0.18,
        min=0.08, max=1.7,
    )[:, None]
    reaction_source = torch.tensor(
        list(g2.REACTION_SOURCE), dtype=torch.int64, device=device,
    )[torch.remainder(parameter, len(g2.REACTION_SOURCE))]
    generic_pool = torch.gather(
        pools[:, None, :].expand(C, K, a3.POOL_COUNT),
        2, reaction_source[:, :, None],
    )[:, :, 0]
    generic_need = torch.clamp(0.18 + generic_pool * 2.2,
                               min=0.10, max=1.5)
    for role_value, role_need in (
        (a3.ROLE_ENERGY, energy_need),
        (a3.ROLE_MEMBRANE, membrane_need),
        (a3.ROLE_TRANSPORTER, transporter_need),
        (a3.ROLE_REPLICASE, replicase_need),
        (a3.ROLE_TRANSLATOR, translator_need),
        (a3.ROLE_NUCLEOTIDE, nucleotide_need),
        (a3.ROLE_GENERIC, generic_need),
    ):
        need = torch.where(role == int(role_value), role_need, need)

    repair_kind = torch.remainder(parameter, int(a3.REPAIR_COUNT))
    repair_need = 0.022 + 2.2 * torch.mean(
        state.membrane_oxidation, dim=1,
    )[:, None] + 1.4 * (1.0 - closure[:, None])
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_ANTIOXIDANT),
        0.030 + 2.8 * reactive[:, None] + 0.6 * state.current_stress[:, None],
        repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_CHAPERONE),
        0.025 + 3.0 * pools[:, a3.POOL_DAMAGED_PROTEIN][:, None], repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_PROTEASE),
        0.022 + 2.1 * pools[:, a3.POOL_DAMAGED_PROTEIN][:, None]
        + 3.3 * pools[:, a3.POOL_AGGREGATE][:, None], repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_GENOME),
        0.022 + 1.5 * state.genome_lesion_mean[:, None], repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_SEGREGATION),
        0.016 + burden[:, None] * (0.4 + 1.4 * state.division_progress[:, None]),
        repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_PROOFREADING),
        0.022 + 0.5 * state.genome_lesion_mean[:, None]
        + 0.6 * state.replication_active.to(dtype)[:, None], repair_need,
    )
    repair_need = torch.where(
        repair_kind == int(a3.REPAIR_QUIESCENCE),
        0.016 + 1.5 * torch.clamp(burden[:, None] - 0.20, min=0.0)
        + 0.7 * state.current_stress[:, None], repair_need,
    )
    repair_need = torch.clamp(repair_need, min=0.012, max=2.2)

    sensor_index = torch.remainder(parameter, int(s4.LIGAND_COUNT))
    sensor_signal = torch.gather(state.last_receptor_activity, 1, sensor_index)
    sensor_need = torch.clamp(
        0.15 + 1.4 * sensor_signal + 0.25 * (1.0 - closure[:, None]),
        min=0.10, max=1.55,
    )
    control_index = torch.remainder(regulator, int(s4.CONTROL_COUNT))
    control_vectors = torch.gather(
        state.last_control_vectors, 1,
        control_index[:, :, None].expand(C, K, 2),
    )
    control_scalars = torch.gather(state.last_control_scalars, 1, control_index)
    drive = torch.linalg.vector_norm(control_vectors, dim=2) + torch.abs(control_scalars)
    effector_need = torch.clamp(
        0.16 + 1.2 * drive + 0.15 * burden[:, None], min=0.10, max=1.55,
    )
    edna = state.last_edna_signal[:, None]
    ecology_kind = torch.remainder(parameter, 8)
    ecology_need = 0.001 + 0.4 * edna
    ecology_need = torch.where(ecology_kind == int(s5.ECO_COMPETENCE),
                               0.003 + 1.4 * edna, ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_RECOMBINASE),
                               0.002 + 0.9 * edna
                               + 0.3 * state.genome_lesion_mean[:, None], ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_NUCLEASE),
                               0.002 + 0.8 * edna, ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_RESTRICTION),
                               0.002 + 0.6 * edna
                               + 0.25 * state.current_stress[:, None], ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_NECROPHAGE),
                               0.003 + 1.2 * state.last_corpse_signal[:, None], ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_DETOX),
                               0.003 + 1.5 * state.last_necrotoxin_signal[:, None]
                               + 0.7 * reactive[:, None], ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_VESICLE),
                               0.0015 + 0.2 * edna, ecology_need)
    ecology_need = torch.where(ecology_kind == int(s5.ECO_MOBILE),
                               0.012 + 0.55 * edna, ecology_need)
    ecology_need = torch.clamp(ecology_need, min=0.001, max=1.8)
    regulator_need = repair_need
    regulator_need = torch.where(localisation == int(s4.LOC_SENSOR),
                                  sensor_need, regulator_need)
    regulator_need = torch.where(localisation == int(s4.LOC_EFFECTOR),
                                  effector_need, regulator_need)
    regulator_need = torch.where(localisation == int(s5.LOC_ECOLOGY),
                                  ecology_need, regulator_need)
    need = torch.where(role == int(a3.ROLE_REGULATOR), regulator_need, need)

    gene_weights = torch.where(
        valid, promoter * copies * need, torch.zeros((C, K), dtype=dtype, device=device),
    )
    total_weight = _torch_ordered_row_sum(gene_weights)
    gate = (
        state.cell_mask & bool(state.gene_expression)
        & (state.genome_count > 0) & (translator > 1e-5)
        & (counts > 0) & (total_weight > 0.0)
    )
    if state.quiescence:
        q_need = torch.clamp(burden - 0.12, min=0.0) + 0.35 * state.current_stress
        quiescence = torch.clamp(
            (quiescence_signal / (0.8 + quiescence_signal)) * q_need * 1.45,
            min=0.0, max=0.82,
        )
    else:
        quiescence = torch.zeros((C,), dtype=dtype, device=device)
    if state.quiescence_effector:
        quiescence = torch.clamp(torch.maximum(
            quiescence, state.behavioural_quiescence,
        ), min=0.0, max=0.86)
    out.last_quiescence = torch.where(gate, quiescence, state.last_quiescence)
    if not state.protein_repair:
        chaperone = torch.zeros_like(chaperone)
    misfold = torch.clamp(
        0.010 + 0.055 * reactive + 0.040 * state.current_stress
        + 0.020 * aggregate - 0.010 * chaperone,
        min=0.004, max=0.42,
    )
    capacity = dt * 0.0060 * translator * (1.0 - 0.72 * quiescence)
    safe_total = torch.where(total_weight > 0.0, total_weight,
                             torch.ones_like(total_weight))
    requested = capacity[:, None] * gene_weights / safe_total[:, None]
    requested = torch.where(valid & gate[:, None], requested,
                            torch.zeros_like(requested))
    # Preserve the frozen per-gene payment recurrence literally.  A closed
    # prefix-budget formula is algebraically equivalent in real arithmetic,
    # but it can leave a tiny negative pool at an exhausted fp64 boundary and
    # amplifies that difference through last_translation for very small dt.
    # This fixed host-known K loop remains fully resident and batches cells;
    # it is intentionally a correctness implementation, not a speed claim.
    fuel_remaining = pools[:, a3.POOL_FUEL].clone()
    mineral_remaining = pools[:, a3.POOL_MINERAL].clone()
    atp_remaining = pools[:, a3.POOL_ATP].clone()
    translation_rate = torch.zeros((C,), dtype=dtype, device=device)
    actual_columns = []
    denominator = max(dt, 1e-9)
    for entry_index in range(K):
        actual_entry = torch.minimum(
            requested[:, entry_index], fuel_remaining / 0.64,
        )
        actual_entry = torch.minimum(
            actual_entry, mineral_remaining / 0.36,
        )
        actual_entry = torch.minimum(
            actual_entry,
            torch.clamp(atp_remaining - 0.042, min=0.0) / 0.52,
        )
        actual_entry = torch.where(
            valid[:, entry_index] & gate & (actual_entry > 0.0),
            actual_entry, torch.zeros_like(actual_entry),
        )
        fuel_remaining = fuel_remaining - 0.64 * actual_entry
        mineral_remaining = mineral_remaining - 0.36 * actual_entry
        atp_remaining = atp_remaining - 0.52 * actual_entry
        translation_rate = translation_rate + actual_entry / denominator
        actual_columns.append(actual_entry)
    actual = torch.stack(actual_columns, dim=1)
    out.last_translation = torch.where(
        state.cell_mask, translation_rate,
        state.last_translation,
    )
    out.pools[:, a3.POOL_FUEL] = fuel_remaining
    out.pools[:, a3.POOL_MINERAL] = mineral_remaining
    out.pools[:, a3.POOL_ATP] = atp_remaining

    def add_products(prefix_name, delta, threshold):
        fingerprints_name = prefix_name + '_fingerprints'
        mass_name = prefix_name + '_mass'
        count_name = prefix_name + '_count'
        source_fingerprints = getattr(state, fingerprints_name)
        source_mass = getattr(state, mass_name)
        source_count = getattr(state, count_name)
        position = torch.arange(P, dtype=torch.int64, device=device)
        used = position[None, :] < source_count[:, None]
        match = (
            valid[:, :, None] & used[:, None, :]
            & (fingerprints[:, :, None] == source_fingerprints[:, None, :])
        )
        exists = torch.any(match, dim=2)
        existing_position = torch.amin(torch.where(
            match, position[None, None, :],
            torch.full((C, K, P), P, dtype=torch.int64, device=device),
        ), dim=2)
        new_entry = valid & (~exists) & (delta > threshold)
        new_rank = torch.cumsum(new_entry.to(torch.int64), dim=1) - 1
        target = torch.where(
            exists, existing_position, source_count[:, None] + new_rank,
        )
        safe_target = torch.clamp(target, min=0, max=P - 1)
        write = exists | new_entry
        flat_target = (
            torch.arange(C, dtype=torch.int64, device=device)[:, None] * P
            + safe_target
        ).reshape(-1)
        flat_mass = getattr(out, mass_name).reshape(-1)
        flat_mass.scatter_add_(
            0, flat_target,
            torch.where(write, delta, torch.zeros_like(delta)).reshape(-1),
        )
        flat_fingerprints = getattr(out, fingerprints_name).reshape(-1)
        flat_fingerprints.scatter_reduce_(
            0, flat_target,
            torch.where(new_entry, fingerprints,
                        torch.full_like(fingerprints, -1)).reshape(-1),
            reduce='amax', include_self=True,
        )
        new_count = source_count + torch.sum(new_entry.to(torch.int64), dim=1)
        setattr(out, count_name, new_count)
        valid_rows = position[None, :] < new_count[:, None]
        total = _torch_ordered_row_sum(torch.where(
            valid_rows, getattr(out, mass_name),
            torch.zeros_like(getattr(out, mass_name)),
        ))
        return total

    active_delta = actual * (1.0 - misfold[:, None])
    damaged_delta = actual * misfold[:, None]
    active_total = add_products('active', active_delta, 1e-10)
    damaged_total = add_products('damaged', damaged_delta, 1e-11)
    out.pools[:, a3.POOL_CATALYST] = torch.where(
        gate, active_total, pools[:, a3.POOL_CATALYST],
    )
    out.pools[:, a3.POOL_DAMAGED_PROTEIN] = torch.where(
        gate, damaged_total, pools[:, a3.POOL_DAMAGED_PROTEIN],
    )
    return out


def paid_translation_plan(binding, dt):
    _require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return paid_translation_plan_torch(binding, dt)
    return paid_translation_plan_numpy(binding, dt)


def _strict_symbols(value, label):
    array = np.asarray(value)
    if array.ndim != 1:
        raise A4SchemaError('%s must be one-dimensional' % label)
    if array.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise A4SchemaError('%s must contain integer symbols' % label)
    if np.any(array < 0) or np.any(array >= ALPHABET_SIZE):
        raise A4SchemaError('%s contains symbol outside [0,%d]' %
                            (label, ALPHABET_SIZE - 1))
    return array.astype(np.uint8, copy=True)


def _strict_float_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise A4SchemaError('%s must be a numeric scalar' % label)
    result = float(value)
    if not math.isfinite(result):
        raise A4SchemaError('%s must be finite' % label)
    return result


def _gene_cache_from_genomes(genomes):
    """Reproduce frozen `_refresh_gene_cache` insertion semantics."""
    cache = {}
    for genome in genomes:
        for spec in g2.parse_genes(genome):
            fingerprint = spec['fingerprint']
            if fingerprint not in cache:
                item = dict(spec)
                item['copy_number'] = 1
                cache[fingerprint] = item
            else:
                cache[fingerprint]['copy_number'] += 1
    return cache


class FullFidelityA4GenomeAdapter:
    """Lossless one-world adapter for A4.1 structural genome ownership."""

    def __init__(self, config=None):
        self.config = GPU068A4Config.from_state(config or {})

    def _prepare_cell(self, cell, index):
        required = (
            'cell_id', 'genomes', 'genome_lesions',
            'replication_template', 'replication_copy',
            'replication_template_lesion', 'replication_fractional',
        )
        missing = [name for name in required if not hasattr(cell, name)]
        if missing:
            raise A4SchemaError(
                'cell[%d] lacks required fields: %s' % (index, missing)
            )
        raw_cell_id = cell.cell_id
        if isinstance(raw_cell_id, (bool, np.bool_)) or not isinstance(
                raw_cell_id, (int, np.integer)):
            raise A4SchemaError('cell[%d] cell_id must be an integer' % index)
        cell_id = int(raw_cell_id)
        if cell_id < 0:
            raise A4SchemaError('cell[%d] has negative cell_id' % index)
        genomes = [
            _strict_symbols(genome, 'cell[%d].genomes[%d]' % (index, gi))
            for gi, genome in enumerate(cell.genomes)
        ]
        lesion_source = np.asarray(cell.genome_lesions)
        if (lesion_source.ndim != 1
                or (lesion_source.size
                    and lesion_source.dtype.kind not in 'iuf')):
            raise A4SchemaError('cell[%d] lesions must be a numeric vector' % index)
        lesions = lesion_source.astype(np.float64, copy=True)
        if len(lesions) > len(genomes):
            raise A4SchemaError('cell[%d] has invalid genome lesion count' % index)
        if not np.isfinite(lesions).all() or np.any(lesions < 0.0):
            raise A4SchemaError('cell[%d] has invalid genome lesion value' % index)

        template_value = cell.replication_template
        active = template_value is not None
        template = (
            _strict_symbols(template_value, 'cell[%d].replication_template' % index)
            if active else None
        )
        copy_symbols = _strict_symbols(
            cell.replication_copy,
            'cell[%d].replication_copy' % index,
        )
        template_lesion = _strict_float_scalar(
            cell.replication_template_lesion,
            'cell[%d].replication_template_lesion' % index,
        )
        fractional = _strict_float_scalar(
            cell.replication_fractional,
            'cell[%d].replication_fractional' % index,
        )
        if template_lesion < 0.0:
            raise A4SchemaError('cell[%d] has invalid template lesion' % index)
        if fractional < 0.0 or fractional >= 1.0:
            raise A4SchemaError('cell[%d] has invalid replication_fractional' % index)
        if active:
            if len(template) == 0:
                raise A4SchemaError('cell[%d] active template is empty' % index)
            if len(copy_symbols) > len(template):
                raise A4SchemaError('cell[%d] copy exceeds template' % index)
        elif (len(copy_symbols) != 0 or template_lesion != 0.0
              or fractional != 0.0):
            raise A4SchemaError(
                'cell[%d] inactive replication carries template/copy/fractional state' % index
            )
        sequences = list(genomes)
        if active:
            sequences.extend((template, copy_symbols))
        for si, sequence in enumerate(sequences):
            _require_capacity(
                len(sequence), self.config.max_sequence_symbols,
                'cell[%d] sequence[%d] symbols' % (index, si),
            )
        return {
            'cell_id': cell_id,
            'genomes': genomes,
            'lesions': lesions.copy(),
            'active': active,
            'template': template,
            'copy': copy_symbols,
            'template_lesion': template_lesion,
            'fractional': fractional,
            'sequences': sequences,
        }

    def pack_cells(self, cells):
        """Pack after validating every source and all aggregate capacities."""
        cells = list(cells)
        prepared = [self._prepare_cell(cell, index)
                    for index, cell in enumerate(cells)]
        ids = [item['cell_id'] for item in prepared]
        if len(ids) != len(set(ids)):
            raise A4SchemaError('duplicate cell IDs are forbidden')
        sequence_count = sum(len(item['sequences']) for item in prepared)
        symbol_count = sum(
            len(sequence) for item in prepared for sequence in item['sequences']
        )
        lesion_count = sum(len(item['lesions']) for item in prepared)
        _require_capacity(len(prepared), self.config.max_cells, 'cells')
        _require_capacity(sequence_count, self.config.max_sequences, 'sequences')
        _require_capacity(symbol_count, self.config.max_symbols, 'symbols')
        _require_capacity(lesion_count, self.config.max_sequences, 'lesions')

        C = int(self.config.max_cells)
        Q = int(self.config.max_sequences)
        S = int(self.config.max_symbols)
        symbols = np.zeros((S,), dtype=np.uint8)
        sequence_offsets = np.full((Q + 1,), -1, dtype=np.int64)
        cell_sequence_offsets = np.full((C + 1,), -1, dtype=np.int64)
        cell_ids = np.full((C,), -1, dtype=np.int64)
        cell_mask = np.zeros((C,), dtype=bool)
        genome_counts = np.zeros((C,), dtype=np.int64)
        lesion_offsets = np.full((C + 1,), -1, dtype=np.int64)
        genome_lesions = np.zeros((Q,), dtype=np.float64)
        replication_active = np.zeros((C,), dtype=bool)
        replication_template_lesions = np.zeros((C,), dtype=np.float64)
        replication_fractional = np.zeros((C,), dtype=np.float64)

        sequence_offsets[0] = 0
        cell_sequence_offsets[0] = 0
        lesion_offsets[0] = 0
        symbol_cursor = 0
        sequence_cursor = 0
        lesion_cursor = 0
        for ci, item in enumerate(prepared):
            cell_ids[ci] = item['cell_id']
            cell_mask[ci] = True
            genome_counts[ci] = len(item['genomes'])
            replication_active[ci] = item['active']
            replication_template_lesions[ci] = item['template_lesion']
            replication_fractional[ci] = item['fractional']
            for sequence in item['sequences']:
                length = len(sequence)
                if length:
                    symbols[symbol_cursor:symbol_cursor + length] = sequence
                symbol_cursor += length
                sequence_cursor += 1
                sequence_offsets[sequence_cursor] = symbol_cursor
            cell_sequence_offsets[ci + 1] = sequence_cursor
            length = len(item['lesions'])
            if length:
                genome_lesions[lesion_cursor:lesion_cursor + length] = item['lesions']
            lesion_cursor += length
            lesion_offsets[ci + 1] = lesion_cursor

        batch = A4RaggedGenomeBatch(
            schema_version=SCHEMA_VERSION,
            cell_capacity=C,
            sequence_capacity=Q,
            symbol_capacity=S,
            max_sequence_symbols=int(self.config.max_sequence_symbols),
            cell_count=len(prepared),
            sequence_count=sequence_count,
            symbol_count=symbol_count,
            lesion_count=lesion_count,
            symbols=symbols,
            sequence_offsets=sequence_offsets,
            cell_sequence_offsets=cell_sequence_offsets,
            cell_ids=cell_ids,
            cell_mask=cell_mask,
            genome_counts=genome_counts,
            lesion_offsets=lesion_offsets,
            genome_lesions=genome_lesions,
            replication_active=replication_active,
            replication_template_lesions=replication_template_lesions,
            replication_fractional=replication_fractional,
        )
        return validate_a4_ragged(batch)

    def unpack_cells(self, batch, cells):
        """Atomically reconstruct structural genome state after full checks."""
        if _is_tensor(batch.symbols):
            # One explicit readback; the resulting NumPy object is fully
            # validated by to_numpy before it is returned.
            host = batch.to_numpy()
        else:
            validate_a4_ragged(batch)
            host = batch.clone()
        cells = list(cells)
        if len(cells) != host.cell_count:
            raise A4SchemaError('target cell count differs from packed batch')
        target_ids = []
        for index, cell in enumerate(cells):
            if not hasattr(cell, 'cell_id'):
                raise A4SchemaError('target cell[%d] lacks cell_id' % index)
            raw_cell_id = cell.cell_id
            if isinstance(raw_cell_id, (bool, np.bool_)) or not isinstance(
                    raw_cell_id, (int, np.integer)):
                raise A4SchemaError(
                    'target cell[%d] cell_id must be an integer' % index
                )
            target_ids.append(int(raw_cell_id))
            for name in ('genomes', 'genome_lesions', 'replication_template',
                         'replication_copy', 'replication_template_lesion',
                         'replication_fractional', 'gene_specs'):
                if not hasattr(cell, name):
                    raise A4SchemaError(
                        'target cell[%d] lacks %s' % (index, name)
                    )
        packed_ids = [int(value) for value in host.cell_ids[:host.cell_count]]
        if target_ids != packed_ids:
            raise A4SchemaError('target cell ID/order differs from packed batch')

        prepared = []
        for ci in range(host.cell_count):
            sequence_first = int(host.cell_sequence_offsets[ci])
            sequence_last = int(host.cell_sequence_offsets[ci + 1])
            sequences = []
            for si in range(sequence_first, sequence_last):
                left = int(host.sequence_offsets[si])
                right = int(host.sequence_offsets[si + 1])
                sequences.append(host.symbols[left:right].copy())
            genome_count = int(host.genome_counts[ci])
            genomes = [item.copy() for item in sequences[:genome_count]]
            active = bool(host.replication_active[ci])
            template = sequences[genome_count].copy() if active else None
            copy_symbols = (
                [int(value) for value in sequences[genome_count + 1]]
                if active else []
            )
            lesion_first = int(host.lesion_offsets[ci])
            lesion_last = int(host.lesion_offsets[ci + 1])
            lesions = [float(value) for value in
                       host.genome_lesions[lesion_first:lesion_last]]
            prepared.append({
                'genomes': genomes,
                'lesions': lesions,
                'template': template,
                'copy': copy_symbols,
                'template_lesion': float(
                    host.replication_template_lesions[ci]
                ),
                'fractional': float(host.replication_fractional[ci]),
                'gene_specs': _gene_cache_from_genomes(genomes),
            })

        # Every conversion and target check above completes before this point.
        for cell, item in zip(cells, prepared):
            cell.genomes = [genome.copy() for genome in item['genomes']]
            cell.genome_lesions = list(item['lesions'])
            cell.replication_template = (
                None if item['template'] is None else item['template'].copy()
            )
            cell.replication_copy = list(item['copy'])
            cell.replication_template_lesion = item['template_lesion']
            cell.replication_fractional = item['fractional']
            cell.gene_specs = copy.deepcopy(item['gene_specs'])
        return cells


def pack_a4_cells(cells, config=None):
    return FullFidelityA4GenomeAdapter(config).pack_cells(cells)


def unpack_a4_cells(batch, cells, config=None):
    return FullFidelityA4GenomeAdapter(config).unpack_cells(batch, cells)


PORT_STATUS = dict(a3.PORT_STATUS)
PORT_STATUS.update({
    'ragged_genome_buffers': 'a4.1-lossless-device-resident-foundation',
    'gene_cache_decode': 'a4.2-batched-resident-derived-cache',
    'translation': 'a4.3-full-formal066-paid-plan-not-integrated-cpu-authoritative',
    'genome_replication': 'cpu-authoritative-next-a4-slice',
    'material_mutation': 'cpu-authoritative-next-a4-slice',
    'full_gpu_world_step': False,
})
