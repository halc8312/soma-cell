# coding: utf-8
"""SOMA-CELL 0.6.8-GPU A4.1 resident ragged-genome foundation.

This development slice intentionally does not replace the promoted A3 world
or any biological phase.  It provides the smallest fail-closed representation
boundary needed to keep variable-length complete genomes, an in-progress
replication template/copy, and lesion state resident on an explicit Torch
device.  Frozen 0.6.6 CPU behavior remains the semantic authority.

No collection is sorted, clipped, or automatically resized.  The adapter
validates the complete batch before allocation and before CPU commit.
"""
from __future__ import division

import copy
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

BUILD = 'SOMA-CELL 0.6.8-GPU A4.1'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | device-resident lossless ragged-genome foundation'
SCHEMA_VERSION = '0.6.8-GPU-A4.1-ragged-genome'
FULL_GPU_WORLD_STEP = False

ALPHABET_SIZE = int(g2.ALPHABET_SIZE)
MAX_FROZEN_GENOME_SYMBOLS = int(g2.MAX_GENOME_LENGTH)


class A4Error(RuntimeError):
    """Base class for fail-closed A4 errors."""


class A4CapacityError(A4Error):
    """A complete source batch cannot fit the declared fixed capacities."""


class A4SchemaError(A4Error):
    """A source or packed ragged state violates the A4.1 schema."""


@dataclass(frozen=True)
class GPU068A4Config:
    """Only the fixed capacities required by the A4.1 slice."""

    max_cells: int = 64
    max_sequences: int = 256
    max_symbols: int = 65536
    max_sequence_symbols: int = MAX_FROZEN_GENOME_SYMBOLS

    def __post_init__(self):
        for name in ('max_cells', 'max_sequences', 'max_symbols',
                     'max_sequence_symbols'):
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
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value) if item.name in _ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        return A4RaggedGenomeBatch(**values)

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
    'translation': 'cpu-authoritative-next-a4-slice',
    'genome_replication': 'cpu-authoritative-next-a4-slice',
    'material_mutation': 'cpu-authoritative-next-a4-slice',
    'full_gpu_world_step': False,
})
