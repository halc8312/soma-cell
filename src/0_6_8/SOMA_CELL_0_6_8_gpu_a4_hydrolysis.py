# coding: utf-8
"""A4.7 genome-symbol hydrolysis RNG tape and pure deletion descriptor.

The A4.7a boundary records the literal frozen 0.3 high-level PCG64 calls for
one cell at the post-lesion-gain, pre-hydrolysis event boundary.  A4.7b applies
that trusted tape to a row-padded polymer/ledger descriptor.  It does not
mutate the ragged arena or gene cache, advance a live world RNG, update a CPU
cell, or replace A3 scheduler authority.

The frozen gate is important: every complete genome with lesion > 0.75 and
length > MIN_GENOME_LENGTH consumes ``Generator.random()`` even when ``dt``
is zero and therefore the hit probability is zero.  A bounded integer draw is
consumed only after a successful comparison, in the same genome-list order.
"""
from __future__ import division

import copy
import hashlib
import json
import math
from dataclasses import dataclass, fields

import numpy as np

import SOMA_CELL_0_6_8_gpu_a4 as a4

try:
    import torch
except Exception:  # pragma: no cover - NumPy reference remains importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.7b'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | resident genome hydrolysis deletion/ledger plan'
SCHEMA_VERSION = '0.6.8-GPU-A4.7a-genome-hydrolysis-rng-tape'
DELETION_PLAN_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.7b-genome-hydrolysis-deletion-plan'
)
FULL_GPU_WORLD_STEP = False

_TAPE_ARRAY_FIELDS = (
    'genome_slot_mask', 'draw_mask', 'uniform_draws', 'hit_mask',
    'deletion_positions',
)
_TAPE_BOOL_FIELDS = ('genome_slot_mask', 'draw_mask', 'hit_mask')
_TAPE_FLOAT64_FIELDS = ('uniform_draws',)
_TAPE_INT64_FIELDS = ('deletion_positions',)
_TAPE_FACTORY_TOKEN = object()

DELETION_SCOPE_OK = 0
DELETION_SCOPE_TAPE_MISMATCH = 1

_DELETION_PLAN_ARRAY_FIELDS = (
    'scope_valid', 'scope_error_code', 'final_symbols', 'final_lengths',
    'genome_lesions_after', 'pools_after', 'symbol_count_after',
    'topology_symbol_delta', 'genome_damage_event_delta',
    'gene_cache_dirty', 'gene_cache_refresh_count',
    'genome_material_symbols_after', 'genome_lesion_mean_after',
)
_DELETION_PLAN_UINT8_FIELDS = ('final_symbols',)
_DELETION_PLAN_BOOL_FIELDS = ('scope_valid', 'gene_cache_dirty')
_DELETION_PLAN_INT64_FIELDS = (
    'scope_error_code', 'final_lengths', 'symbol_count_after',
    'topology_symbol_delta', 'genome_damage_event_delta',
    'gene_cache_refresh_count', 'genome_material_symbols_after',
)
_DELETION_PLAN_FLOAT64_FIELDS = (
    'genome_lesions_after', 'pools_after', 'genome_lesion_mean_after',
)


class A4HydrolysisError(a4.A4Error):
    """Base class for this bounded hydrolysis RNG slice."""


class A4HydrolysisScopeError(A4HydrolysisError):
    """The source is not the single-cell post-gain boundary of A4.7a."""


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _clone_array(value):
    return value.clone() if _is_tensor(value) else np.asarray(value).copy()


def _explicit_host_array(value):
    """Explicit diagnostic/readback boundary; never used by resident require."""
    if _is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _is_int_scalar(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer)))


def _is_lower_hex_digest(value):
    return (
        isinstance(value, str) and len(value) == 64
        and value == value.lower()
        and all(character in '0123456789abcdef' for character in value)
    )


def _strict_dt(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise A4HydrolysisScopeError('hydrolysis dt must be a real scalar')
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise A4HydrolysisScopeError(
            'hydrolysis dt must be finite and nonnegative'
        )
    return result


def _sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    ).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _canonical_pcg64_state(value, label):
    """Return an exact JSON-like NumPy PCG64 state or fail closed."""
    if not isinstance(value, dict) or set(value) != {
            'bit_generator', 'state', 'has_uint32', 'uinteger'}:
        raise a4.A4SchemaError('%s has noncanonical PCG64 keys' % label)
    if value.get('bit_generator') != 'PCG64':
        raise a4.A4SchemaError('%s is not a PCG64 state' % label)
    nested = value.get('state')
    if not isinstance(nested, dict) or set(nested) != {'state', 'inc'}:
        raise a4.A4SchemaError('%s has noncanonical inner state' % label)
    result = {
        'bit_generator': 'PCG64',
        'state': {},
        'has_uint32': None,
        'uinteger': None,
    }
    for name in ('state', 'inc'):
        item = nested[name]
        if not _is_int_scalar(item):
            raise a4.A4SchemaError(
                '%s.%s must be integer' % (label, name)
            )
        item = int(item)
        if item < 0 or item >= (1 << 128):
            raise a4.A4SchemaError(
                '%s.%s outside uint128' % (label, name)
            )
        result['state'][name] = item
    for name, upper in (('has_uint32', 2), ('uinteger', 1 << 32)):
        item = value[name]
        if not _is_int_scalar(item):
            raise a4.A4SchemaError(
                '%s.%s must be integer' % (label, name)
            )
        item = int(item)
        if item < 0 or item >= upper:
            raise a4.A4SchemaError(
                '%s.%s outside range' % (label, name)
            )
        result[name] = item
    try:
        bit_generator = np.random.PCG64()
        bit_generator.state = copy.deepcopy(result)
    except Exception as exc:
        raise a4.A4SchemaError(
            '%s is not accepted by NumPy PCG64' % label
        ) from exc
    if bit_generator.state != result:
        raise a4.A4SchemaError(
            '%s is not a canonical PCG64 state' % label
        )
    return result


def _explicit_array_digest_from_values(values):
    """Hash host values, performing D2H only at an explicit boundary."""
    digest = hashlib.sha256()
    for name in _TAPE_ARRAY_FIELDS:
        array = np.ascontiguousarray(_explicit_host_array(values[name]))
        digest.update(name.encode('ascii'))
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(repr(tuple(array.shape)).encode('ascii'))
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _array_digest(tape):
    if _is_tensor(tape.uniform_draws):
        raise a4.A4SchemaError(
            'resident hydrolysis content requires explicit to_numpy readback'
        )
    return _explicit_array_digest_from_values({
        name: getattr(tape, name) for name in _TAPE_ARRAY_FIELDS
    })


def _schedule_digest_from_values(values):
    before = _canonical_pcg64_state(
        values['rng_before_state'], 'rng_before_state',
    )
    after = _canonical_pcg64_state(
        values['rng_after_state'], 'rng_after_state',
    )
    payload = {
        'schema_version': str(values['schema_version']),
        'sequence_capacity': int(values['sequence_capacity']),
        'sequence_count': int(values['sequence_count']),
        'genome_count': int(values['genome_count']),
        'cell_id': int(values['cell_id']),
        'source_provenance': str(values['source_provenance']),
        'dt_hex': str(values['dt_hex']),
        'draw_count': int(values['draw_count']),
        'hit_count': int(values['hit_count']),
        'rng_before_sha256': _sha256_json(before),
        'rng_after_sha256': _sha256_json(after),
        'array_sha256': _explicit_array_digest_from_values(values),
    }
    return _sha256_json(payload)


def _schedule_digest(tape):
    values = {
        name: getattr(tape, name)
        for name in (
            'schema_version', 'sequence_capacity', 'sequence_count',
            'genome_count', 'cell_id', 'source_provenance', 'dt_hex',
            'draw_count', 'hit_count', 'rng_before_state',
            'rng_after_state',
        ) + _TAPE_ARRAY_FIELDS
    }
    return _schedule_digest_from_values(values)


@dataclass
class A4HydrolysisRngTape:
    """Private-factory one-cell semantic PCG64 tape, never live authority."""

    _factory_token: object
    schema_version: str
    sequence_capacity: int
    sequence_count: int
    genome_count: int
    cell_id: int
    source_provenance: str
    dt_hex: str
    schedule_sha256: str
    rng_before_state: object
    rng_after_state: object
    draw_count: int
    hit_count: int
    genome_slot_mask: object
    draw_mask: object
    uniform_draws: object
    hit_mask: object
    deletion_positions: object

    def clone(self):
        _require_hydrolysis_rng_tape(self)
        values = self._public_values()
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        if _is_tensor(self.uniform_draws):
            values['_resident_expected_arrays'] = {
                name: self._resident_expected_arrays[name].clone()
                for name in _TAPE_ARRAY_FIELDS
            }
        return _make_hydrolysis_rng_tape(**values)

    def to_torch(self, binding, dt, device='cpu'):
        """Validate host replay before the explicit CPU-to-device boundary."""
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        validate_a4_hydrolysis_rng_tape(self, binding, dt)
        requested = torch.device(device)
        if requested.type not in ('cpu', 'cuda'):
            raise A4HydrolysisScopeError(
                'hydrolysis RNG tape device must be cpu or cuda'
            )
        if requested.type == 'cuda' and not torch.cuda.is_available():
            raise A4HydrolysisScopeError('CUDA requested but unavailable')
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'sequence_capacity', 'sequence_count',
                'genome_count', 'cell_id', 'source_provenance', 'dt_hex',
                'schedule_sha256', 'rng_before_state', 'rng_after_state',
                'draw_count', 'hit_count',
            )
        }
        for name in _TAPE_ARRAY_FIELDS:
            source = np.asarray(getattr(self, name))
            dtype = (
                torch.bool if name in _TAPE_BOOL_FIELDS else
                torch.int64 if name in _TAPE_INT64_FIELDS else
                torch.float64
            )
            values[name] = torch.as_tensor(
                source, dtype=dtype, device=requested,
            ).clone()
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        return _make_hydrolysis_rng_tape(**values)

    def to_numpy(self):
        """Read back values; source-aware replay validation stays explicit."""
        _require_hydrolysis_rng_tape(self)
        if not _is_tensor(self.uniform_draws):
            return self.clone()
        if _explicit_array_digest_from_values(
                self._resident_expected_arrays) != (
                self._expected_host_array_sha256):
            raise a4.A4SchemaError(
                'resident hydrolysis expected content changed'
            )
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'sequence_capacity', 'sequence_count',
                'genome_count', 'cell_id', 'source_provenance', 'dt_hex',
                'schedule_sha256', 'rng_before_state', 'rng_after_state',
                'draw_count', 'hit_count',
            )
        }
        for name in _TAPE_ARRAY_FIELDS:
            value = _explicit_host_array(getattr(self, name))
            dtype = (
                bool if name in _TAPE_BOOL_FIELDS else
                np.int64 if name in _TAPE_INT64_FIELDS else
                np.float64
            )
            values[name] = value.astype(dtype, copy=False)
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        return _make_hydrolysis_rng_tape(**values)

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _TAPE_ARRAY_FIELDS):
            raise TypeError(
                'data_ptrs requires a Torch-backed hydrolysis RNG tape'
            )
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _TAPE_ARRAY_FIELDS
        }

    def state_dict(self):
        _require_hydrolysis_rng_tape(self)
        return self._public_values()

    def _public_values(self):
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'sequence_capacity', 'sequence_count',
                'genome_count', 'cell_id', 'source_provenance', 'dt_hex',
                'schedule_sha256', 'rng_before_state', 'rng_after_state',
                'draw_count', 'hit_count',
            )
        }
        values.update({
            name: _clone_array(getattr(self, name))
            for name in _TAPE_ARRAY_FIELDS
        })
        return values


@dataclass
class A4HydrolysisDeletionPlan:
    """Pure row-padded deletion/ledger result, never live authority."""

    schema_version: str
    sequence_capacity: int
    max_sequence_symbols: int
    sequence_count: int
    genome_count: int
    source_symbol_count: int
    cell_id: int
    source_provenance: str
    tape_schedule_sha256: str
    scope_valid: object
    scope_error_code: object
    final_symbols: object
    final_lengths: object
    genome_lesions_after: object
    pools_after: object
    symbol_count_after: object
    topology_symbol_delta: object
    genome_damage_event_delta: object
    gene_cache_dirty: object
    gene_cache_refresh_count: object
    genome_material_symbols_after: object
    genome_lesion_mean_after: object

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value)
                if item.name in _DELETION_PLAN_ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        return A4HydrolysisDeletionPlan(**values)

    def to_numpy(self):
        """Explicitly read back a plan; source replay remains separate."""
        backend = _validate_deletion_plan_metadata(self)
        if backend == 'numpy':
            return _validate_deletion_plan_values(self).clone()
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _DELETION_PLAN_ARRAY_FIELDS:
                value = _explicit_host_array(value)
                if item.name in _DELETION_PLAN_UINT8_FIELDS:
                    value = value.astype(np.uint8, copy=False)
                elif item.name in _DELETION_PLAN_INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                elif item.name in _DELETION_PLAN_BOOL_FIELDS:
                    value = value.astype(bool, copy=False)
                else:
                    value = value.astype(np.float64, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        return _validate_deletion_plan_values(
            A4HydrolysisDeletionPlan(**values)
        )

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _DELETION_PLAN_ARRAY_FIELDS):
            raise TypeError(
                'data_ptrs requires a Torch-backed hydrolysis deletion plan'
            )
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _DELETION_PLAN_ARRAY_FIELDS
        }

    def state_dict(self):
        return {
            item.name: (
                _clone_array(getattr(self, item.name))
                if item.name in _DELETION_PLAN_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, item.name))
            )
            for item in fields(self)
        }


def _validate_tape_metadata(tape):
    if (not isinstance(tape, A4HydrolysisRngTape)
            or tape._factory_token is not _TAPE_FACTORY_TOKEN):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape must come from the private factory'
        )
    if tape.schema_version != SCHEMA_VERSION:
        raise a4.A4SchemaError('hydrolysis RNG tape schema mismatch')
    for name in (
            'sequence_capacity', 'sequence_count', 'genome_count',
            'cell_id', 'draw_count', 'hit_count'):
        if not _is_int_scalar(getattr(tape, name)):
            raise a4.A4SchemaError('%s must be integer' % name)
    capacity = int(tape.sequence_capacity)
    sequence_count = int(tape.sequence_count)
    genome_count = int(tape.genome_count)
    if (capacity <= 0 or sequence_count < 0
            or sequence_count > capacity or genome_count < 0
            or genome_count > sequence_count or int(tape.cell_id) < 0):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape identity/count is invalid'
        )
    if (int(tape.draw_count) < 0 or int(tape.draw_count) > genome_count
            or int(tape.hit_count) < 0
            or int(tape.hit_count) > int(tape.draw_count)):
        raise a4.A4SchemaError('hydrolysis RNG tape counts are invalid')
    if (not _is_lower_hex_digest(tape.source_provenance)
            or not _is_lower_hex_digest(tape.schedule_sha256)):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape provenance digest is invalid'
        )
    if not isinstance(tape.dt_hex, str):
        raise a4.A4SchemaError('hydrolysis RNG tape dt hex is invalid')
    try:
        parsed_dt = float.fromhex(tape.dt_hex)
    except Exception as exc:
        raise a4.A4SchemaError(
            'hydrolysis RNG tape dt hex is invalid'
        ) from exc
    if (not math.isfinite(parsed_dt) or parsed_dt < 0.0
            or parsed_dt.hex() != tape.dt_hex):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape dt is outside supported range'
        )
    _canonical_pcg64_state(tape.rng_before_state, 'rng_before_state')
    _canonical_pcg64_state(tape.rng_after_state, 'rng_after_state')

    kinds = set()
    devices = set()
    for name in _TAPE_ARRAY_FIELDS:
        value = getattr(tape, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise a4.A4SchemaError('%s is not an array/tensor' % name)
    if len(kinds) != 1 or len(devices) > 1:
        raise a4.A4SchemaError(
            'mixed hydrolysis tape backend/device is forbidden'
        )
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_TAPE_BOOL_FIELDS, np.dtype(bool), getattr(torch, 'bool', None)),
        (_TAPE_INT64_FIELDS, np.dtype(np.int64),
         getattr(torch, 'int64', None)),
        (_TAPE_FLOAT64_FIELDS, np.dtype(np.float64),
         getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(tape, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    for name in _TAPE_ARRAY_FIELDS:
        if tuple(getattr(tape, name).shape) != (capacity,):
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return backend


def _validate_semantic_arrays(tape):
    if _is_tensor(tape.uniform_draws):
        raise a4.A4SchemaError(
            'resident hydrolysis semantics require explicit to_numpy readback'
        )
    raw = {name: np.asarray(getattr(tape, name))
           for name in _TAPE_ARRAY_FIELDS}
    capacity = int(tape.sequence_capacity)
    genomes = int(tape.genome_count)
    expected_slots = np.arange(capacity, dtype=np.int64) < genomes
    if not np.array_equal(raw['genome_slot_mask'], expected_slots):
        raise a4.A4SchemaError(
            'hydrolysis genome slots must be the complete-genome prefix'
        )
    if np.any(raw['draw_mask'] & ~raw['genome_slot_mask']):
        raise a4.A4SchemaError('hydrolysis draw exists outside genome slots')
    if np.any(raw['hit_mask'] & ~raw['draw_mask']):
        raise a4.A4SchemaError('hydrolysis hit exists without a draw')
    uniforms = raw['uniform_draws']
    if not np.isfinite(uniforms).all():
        raise a4.A4SchemaError('hydrolysis uniform draws must be finite')
    if np.any(uniforms[~raw['draw_mask']] != 0.0):
        raise a4.A4SchemaError('unused hydrolysis uniform slots must be zero')
    used_uniforms = uniforms[raw['draw_mask']]
    if np.any(used_uniforms < 0.0) or np.any(used_uniforms >= 1.0):
        raise a4.A4SchemaError('hydrolysis uniform draw outside [0,1)')
    positions = raw['deletion_positions']
    if np.any(positions[~raw['hit_mask']] != -1):
        raise a4.A4SchemaError(
            'non-hit hydrolysis deletion positions must be -1'
        )
    if np.any(positions[raw['hit_mask']] < 0):
        raise a4.A4SchemaError('hydrolysis deletion position is negative')
    if int(np.count_nonzero(raw['draw_mask'])) != int(tape.draw_count):
        raise a4.A4SchemaError('hydrolysis draw count differs from mask')
    if int(np.count_nonzero(raw['hit_mask'])) != int(tape.hit_count):
        raise a4.A4SchemaError('hydrolysis hit count differs from mask')
    return raw


def _tape_scalar_metadata(tape):
    before = _canonical_pcg64_state(
        tape.rng_before_state, 'rng_before_state',
    )
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    return (
        str(tape.schema_version), int(tape.sequence_capacity),
        int(tape.sequence_count), int(tape.genome_count), int(tape.cell_id),
        str(tape.source_provenance), str(tape.dt_hex),
        str(tape.schedule_sha256), int(tape.draw_count), int(tape.hit_count),
        _sha256_json(before), _sha256_json(after),
        str(tape._expected_host_array_sha256),
    )


def _make_hydrolysis_rng_tape(**values):
    expected_digest = values.pop('_expected_host_array_sha256', None)
    resident_expected = values.pop('_resident_expected_arrays', None)
    tape = A4HydrolysisRngTape(
        _factory_token=_TAPE_FACTORY_TOKEN, **values
    )
    backend = _validate_tape_metadata(tape)
    if backend == 'numpy':
        _validate_semantic_arrays(tape)
        actual_digest = _array_digest(tape)
        if expected_digest is not None and expected_digest != actual_digest:
            raise a4.A4SchemaError(
                'hydrolysis RNG tape content changed during readback'
            )
        tape._host_array_sha256 = actual_digest
        tape._expected_host_array_sha256 = actual_digest
    else:
        if not _is_lower_hex_digest(expected_digest):
            raise a4.A4SchemaError(
                'resident hydrolysis tape lacks host content digest'
            )
        tape._expected_host_array_sha256 = expected_digest
        tape._resident_data_ptrs = tape.data_ptrs()
        tape._resident_versions = {
            name: int(getattr(tape, name)._version)
            for name in _TAPE_ARRAY_FIELDS
        }
        if resident_expected is None:
            resident_expected = {
                name: getattr(tape, name).clone()
                for name in _TAPE_ARRAY_FIELDS
            }
        if set(resident_expected) != set(_TAPE_ARRAY_FIELDS):
            raise a4.A4SchemaError(
                'resident hydrolysis expected-array set is invalid'
            )
        tape._resident_expected_arrays = {
            name: resident_expected[name].clone()
            for name in _TAPE_ARRAY_FIELDS
        }
        for name in _TAPE_ARRAY_FIELDS:
            expected = tape._resident_expected_arrays[name]
            actual = getattr(tape, name)
            if (not _is_tensor(expected) or expected.device != actual.device
                    or expected.dtype != actual.dtype
                    or tuple(expected.shape) != tuple(actual.shape)):
                raise a4.A4SchemaError(
                    'resident hydrolysis expected %s is invalid' % name
                )
        tape._resident_expected_data_ptrs = {
            name: int(value.data_ptr())
            for name, value in tape._resident_expected_arrays.items()
        }
        tape._resident_expected_versions = {
            name: int(value._version)
            for name, value in tape._resident_expected_arrays.items()
        }
    tape._scalar_metadata = _tape_scalar_metadata(tape)
    return _require_hydrolysis_rng_tape(tape)


def _require_hydrolysis_rng_tape(tape):
    backend = _validate_tape_metadata(tape)
    if getattr(tape, '_scalar_metadata', None) != _tape_scalar_metadata(tape):
        raise a4.A4SchemaError('hydrolysis RNG tape scalar metadata changed')
    expected_digest = getattr(tape, '_expected_host_array_sha256', None)
    if not _is_lower_hex_digest(expected_digest):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape content digest is invalid'
        )
    if backend == 'numpy':
        _validate_semantic_arrays(tape)
        if tape.schedule_sha256 != _schedule_digest(tape):
            raise a4.A4SchemaError(
                'hydrolysis RNG tape schedule provenance differs'
            )
        actual_digest = _array_digest(tape)
        if (getattr(tape, '_host_array_sha256', None) != actual_digest
                or expected_digest != actual_digest):
            raise a4.A4SchemaError(
                'host hydrolysis RNG tape changed after creation'
            )
        return tape

    if (getattr(tape, '_resident_data_ptrs', None) != tape.data_ptrs()
            or getattr(tape, '_resident_versions', None) != {
                name: int(getattr(tape, name)._version)
                for name in _TAPE_ARRAY_FIELDS
            }):
        raise a4.A4SchemaError(
            'resident hydrolysis RNG tape changed after upload'
        )
    expected = getattr(tape, '_resident_expected_arrays', None)
    if (not isinstance(expected, dict)
            or set(expected) != set(_TAPE_ARRAY_FIELDS)
            or getattr(tape, '_resident_expected_data_ptrs', None) != {
                name: int(expected[name].data_ptr())
                for name in _TAPE_ARRAY_FIELDS
            }
            or getattr(tape, '_resident_expected_versions', None) != {
                name: int(expected[name]._version)
                for name in _TAPE_ARRAY_FIELDS
            }):
        raise a4.A4SchemaError(
            'resident hydrolysis expected values changed'
        )
    # No D2H or implicit synchronization belongs here.  ``to_numpy`` is the
    # explicit readback boundary and compares actual bytes with the attested
    # host digest, including Torch ``.data`` version-bypass mutations.
    return tape


def _validate_deletion_plan_metadata(plan):
    if not isinstance(plan, A4HydrolysisDeletionPlan):
        raise a4.A4SchemaError('expected A4HydrolysisDeletionPlan')
    if plan.schema_version != DELETION_PLAN_SCHEMA_VERSION:
        raise a4.A4SchemaError('hydrolysis deletion plan schema mismatch')
    for name in (
            'sequence_capacity', 'max_sequence_symbols', 'sequence_count',
            'genome_count', 'source_symbol_count', 'cell_id'):
        if not _is_int_scalar(getattr(plan, name)):
            raise a4.A4SchemaError('%s must be integer' % name)
    capacity = int(plan.sequence_capacity)
    width = int(plan.max_sequence_symbols)
    sequences = int(plan.sequence_count)
    genomes = int(plan.genome_count)
    source_symbols = int(plan.source_symbol_count)
    if (capacity <= 0 or width <= 0
            or width > int(a4.g2.MAX_GENOME_LENGTH)
            or sequences < 0 or sequences > capacity
            or genomes < 0 or genomes > sequences
            or sequences - genomes not in (0, 2)
            or source_symbols < 0
            or source_symbols > capacity * width
            or int(plan.cell_id) < 0):
        raise a4.A4SchemaError(
            'hydrolysis deletion plan identity/capacity is invalid'
        )
    if (not _is_lower_hex_digest(plan.source_provenance)
            or not _is_lower_hex_digest(plan.tape_schedule_sha256)):
        raise a4.A4SchemaError(
            'hydrolysis deletion plan provenance is invalid'
        )

    kinds = set()
    devices = set()
    for name in _DELETION_PLAN_ARRAY_FIELDS:
        value = getattr(plan, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise a4.A4SchemaError('%s is not an array/tensor' % name)
    if len(kinds) != 1 or len(devices) > 1:
        raise a4.A4SchemaError(
            'mixed hydrolysis deletion backend/device is forbidden'
        )
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_DELETION_PLAN_UINT8_FIELDS, np.dtype(np.uint8),
         getattr(torch, 'uint8', None)),
        (_DELETION_PLAN_INT64_FIELDS, np.dtype(np.int64),
         getattr(torch, 'int64', None)),
        (_DELETION_PLAN_BOOL_FIELDS, np.dtype(bool),
         getattr(torch, 'bool', None)),
        (_DELETION_PLAN_FLOAT64_FIELDS, np.dtype(np.float64),
         getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(plan, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    shapes = {
        'scope_valid': (1,), 'scope_error_code': (1,),
        'final_symbols': (capacity, width),
        'final_lengths': (capacity,),
        'genome_lesions_after': (capacity,),
        'pools_after': (1, int(a4.a3.POOL_COUNT)),
        'symbol_count_after': (1,), 'topology_symbol_delta': (1,),
        'genome_damage_event_delta': (1,),
        'gene_cache_dirty': (1,), 'gene_cache_refresh_count': (1,),
        'genome_material_symbols_after': (1,),
        'genome_lesion_mean_after': (1,),
    }
    for name, shape in shapes.items():
        if tuple(getattr(plan, name).shape) != shape:
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return backend


def _validate_deletion_plan_values(plan):
    """Validate canonical host values without granting commit authority."""
    if _validate_deletion_plan_metadata(plan) != 'numpy':
        raise a4.A4SchemaError(
            'full hydrolysis deletion plan validation requires NumPy'
        )
    raw = {
        name: np.asarray(getattr(plan, name))
        for name in _DELETION_PLAN_ARRAY_FIELDS
    }
    if (not bool(raw['scope_valid'][0])
            or int(raw['scope_error_code'][0]) != DELETION_SCOPE_OK):
        raise A4HydrolysisScopeError(
            'hydrolysis deletion plan contains a resident trust failure'
        )
    sequences = int(plan.sequence_count)
    genomes = int(plan.genome_count)
    width = int(plan.max_sequence_symbols)
    lengths = raw['final_lengths']
    if (np.any(lengths[:sequences] < 0)
            or np.any(lengths[:sequences] > width)
            or np.any(lengths[sequences:] != 0)):
        raise a4.A4SchemaError(
            'hydrolysis deletion plan length vector is invalid'
        )
    symbols = raw['final_symbols']
    for sequence_index in range(sequences):
        length = int(lengths[sequence_index])
        if (np.any(symbols[sequence_index, :length]
                   >= int(a4.g2.ALPHABET_SIZE))
                or np.any(symbols[sequence_index, length:] != 0)):
            raise a4.A4SchemaError(
                'hydrolysis deletion plan polymer is invalid'
            )
    if symbols[sequences:].size and np.any(symbols[sequences:] != 0):
        raise a4.A4SchemaError(
            'hydrolysis deletion plan unused sequence rows are not zero'
        )

    lesions = raw['genome_lesions_after']
    pools = raw['pools_after']
    mean_after = raw['genome_lesion_mean_after']
    if (not np.isfinite(lesions).all()
            or np.any(lesions[:genomes] < 0.0)
            or np.any(lesions[genomes:] != 0.0)
            or not np.isfinite(pools).all()
            or not np.isfinite(mean_after).all()
            or float(mean_after[0]) < 0.0):
        raise a4.A4SchemaError(
            'hydrolysis deletion chemistry contains invalid values'
        )
    paid = set(int(value) for value in a4.TRANSLATION_PAID_POOL_INDICES)
    for pool_index in range(int(a4.a3.POOL_COUNT)):
        minimum = (
            -float(a4.TRANSLATION_LEDGER_ATOL)
            if pool_index in paid else 0.0
        )
        if float(pools[0, pool_index]) < minimum:
            raise a4.A4SchemaError(
                'hydrolysis deletion pool lies outside ledger bounds'
            )

    events = int(raw['genome_damage_event_delta'][0])
    refreshes = int(raw['gene_cache_refresh_count'][0])
    symbol_after = int(raw['symbol_count_after'][0])
    topology_delta = int(raw['topology_symbol_delta'][0])
    material_after = int(raw['genome_material_symbols_after'][0])
    if (events < 0 or events > genomes or refreshes != events
            or bool(raw['gene_cache_dirty'][0]) != bool(events)
            or topology_delta != -events
            or symbol_after != int(plan.source_symbol_count) - events
            or symbol_after < 0
            or symbol_after != int(np.sum(
                lengths[:sequences], dtype=np.int64,
            ))):
        raise a4.A4SchemaError(
            'hydrolysis deletion event/topology ledger is invalid'
        )
    material_expected = int(np.sum(
        lengths[:genomes], dtype=np.int64,
    ))
    if sequences - genomes == 2:
        material_expected += int(lengths[genomes + 1])
    if material_after != material_expected or material_after < 0:
        raise a4.A4SchemaError(
            'hydrolysis deletion material-symbol ledger is invalid'
        )
    expected_mean = (
        float(np.mean(lesions[:genomes], dtype=np.float64))
        if genomes else 1.0
    )
    if not _nonnegative_float64_within_one_ulp(
            mean_after,
            np.asarray([expected_mean], dtype=np.float64)):
        raise a4.A4SchemaError(
            'hydrolysis deletion lesion mean grouping is invalid'
        )
    return plan


def _require_numpy_single_cell_binding(binding):
    a4._require_translation_binding(binding)
    if _is_tensor(binding.ragged.symbols):
        raise a4.A4SchemaError(
            'hydrolysis tape replay requires a NumPy binding'
        )
    ragged = binding.ragged
    state = binding.state
    if int(ragged.cell_count) != 1 or int(state.cell_count) != 1:
        raise A4HydrolysisScopeError(
            'A4.7a hydrolysis tape requires exactly one cell'
        )
    genomes = int(ragged.genome_counts[0])
    lesion_first = int(ragged.lesion_offsets[0])
    lesion_last = int(ragged.lesion_offsets[1])
    if (lesion_first != 0 or lesion_last - lesion_first != genomes
            or int(ragged.lesion_count) != genomes):
        raise A4HydrolysisScopeError(
            'A4.7a source must be post-gain with one lesion per genome'
        )
    if int(ragged.cell_sequence_offsets[0]) != 0:
        raise a4.A4SchemaError('single-cell sequence arena must start at zero')
    return ragged


def _replay_hydrolysis_rng(binding, dt, rng_state_before):
    """Replay literal frozen calls without touching source or a live RNG."""
    ragged = _require_numpy_single_cell_binding(binding)
    dt = _strict_dt(dt)
    before = _canonical_pcg64_state(
        rng_state_before, 'rng_state_before',
    )
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(before)
    generator = np.random.Generator(bit_generator)

    capacity = int(ragged.sequence_capacity)
    sequence_count = int(ragged.sequence_count)
    genome_count = int(ragged.genome_counts[0])
    first_sequence = int(ragged.cell_sequence_offsets[0])
    first_lesion = int(ragged.lesion_offsets[0])
    arrays = {
        'genome_slot_mask': np.zeros((capacity,), dtype=bool),
        'draw_mask': np.zeros((capacity,), dtype=bool),
        'uniform_draws': np.zeros((capacity,), dtype=np.float64),
        'hit_mask': np.zeros((capacity,), dtype=bool),
        'deletion_positions': np.full((capacity,), -1, dtype=np.int64),
    }
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    for genome_index in range(genome_count):
        sequence_index = first_sequence + genome_index
        lesion_index = first_lesion + genome_index
        length = int(
            ragged.sequence_offsets[sequence_index + 1]
            - ragged.sequence_offsets[sequence_index]
        )
        lesion = float(ragged.genome_lesions[lesion_index])
        arrays['genome_slot_mask'][sequence_index] = True
        # Preserve the literal short-circuit gate.  In particular, dt=0 does
        # not suppress this random draw after lesion/length eligibility passes.
        if lesion > 0.75 and length > minimum:
            arrays['draw_mask'][sequence_index] = True
            draw = float(generator.random())
            arrays['uniform_draws'][sequence_index] = draw
            if draw < dt * 0.00065 * lesion:
                arrays['hit_mask'][sequence_index] = True
                arrays['deletion_positions'][sequence_index] = int(
                    generator.integers(0, length)
                )
    return {
        'sequence_capacity': capacity,
        'sequence_count': sequence_count,
        'genome_count': genome_count,
        'cell_id': int(ragged.cell_ids[0]),
        'source_provenance': str(binding.state.source_provenance),
        'dt_hex': dt.hex(),
        'rng_before_state': copy.deepcopy(before),
        'rng_after_state': copy.deepcopy(generator.bit_generator.state),
        'draw_count': int(np.count_nonzero(arrays['draw_mask'])),
        'hit_count': int(np.count_nonzero(arrays['hit_mask'])),
        **arrays
    }


def prepare_genome_hydrolysis_rng_tape(
        binding, dt, rng_state_before):
    """Create an immutable event-local tape without advancing a live RNG."""
    replay = _replay_hydrolysis_rng(binding, dt, rng_state_before)
    values = {
        'schema_version': SCHEMA_VERSION,
        'schedule_sha256': '0' * 64,
        **replay
    }
    values['schedule_sha256'] = _schedule_digest_from_values(values)
    tape = _make_hydrolysis_rng_tape(**values)
    return validate_a4_hydrolysis_rng_tape(tape, binding, dt)


def validate_a4_hydrolysis_rng_tape(tape, binding, dt):
    """Re-derive every high-level call from the attested NumPy source."""
    _require_hydrolysis_rng_tape(tape)
    if _is_tensor(tape.uniform_draws):
        raise a4.A4SchemaError(
            'full hydrolysis tape validation requires a NumPy tape'
        )
    replay = _replay_hydrolysis_rng(binding, dt, tape.rng_before_state)
    for name in (
            'sequence_capacity', 'sequence_count', 'genome_count',
            'cell_id', 'source_provenance', 'dt_hex', 'draw_count',
            'hit_count'):
        if getattr(tape, name) != replay[name]:
            raise A4HydrolysisScopeError(
                'hydrolysis tape does not bind this source/dt'
            )
    for name in _TAPE_ARRAY_FIELDS:
        if not np.array_equal(np.asarray(getattr(tape, name)), replay[name]):
            raise a4.A4SchemaError(
                'hydrolysis RNG tape %s differs from replay' % name
            )
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    if after != replay['rng_after_state']:
        raise a4.A4SchemaError(
            'hydrolysis RNG tape after-state differs from replay'
        )
    if tape.schedule_sha256 != _schedule_digest(tape):
        raise a4.A4SchemaError(
            'hydrolysis RNG tape schedule provenance differs'
        )
    return tape


def _require_hydrolysis_tape_binding(tape, binding, dt):
    """Check host-known source identity without resident readback."""
    a4._require_translation_binding(binding)
    _require_hydrolysis_rng_tape(tape)
    dt = _strict_dt(dt)
    ragged = binding.ragged
    state = binding.state
    if (int(ragged.cell_count) != 1 or int(state.cell_count) != 1
            or int(tape.sequence_capacity)
            != int(ragged.sequence_capacity)
            or int(tape.sequence_count) != int(ragged.sequence_count)
            or int(tape.genome_count) != int(ragged.lesion_count)
            or tape.source_provenance != str(state.source_provenance)
            or tape.dt_hex != dt.hex()):
        raise A4HydrolysisScopeError(
            'hydrolysis tape does not bind this resident source/dt'
        )
    tape_tensor = _is_tensor(tape.uniform_draws)
    source_tensor = _is_tensor(ragged.symbols)
    if tape_tensor != source_tensor:
        raise a4.A4SchemaError(
            'hydrolysis tape and source must share one backend'
        )
    if tape_tensor and tape.uniform_draws.device != ragged.symbols.device:
        raise a4.A4SchemaError(
            'hydrolysis tape and source must share one device'
        )
    return tape


def _resident_hydrolysis_tape_unchanged(tape):
    """Return a device scalar guarding Torch ``.data`` bypass writes."""
    _require_hydrolysis_rng_tape(tape)
    if not _is_tensor(tape.uniform_draws):
        raise a4.A4SchemaError(
            'resident hydrolysis content comparison requires Torch'
        )
    unchanged = torch.ones(
        (), dtype=torch.bool, device=tape.uniform_draws.device,
    )
    for name in _TAPE_ARRAY_FIELDS:
        unchanged = unchanged & torch.all(
            getattr(tape, name) == tape._resident_expected_arrays[name]
        )
    return unchanged


def _torch_numpy_pairwise_sum_rows(values):
    """Reproduce NumPy's contiguous fp64 pairwise sum for fixed rows."""
    rows = int(values.shape[0])

    def pairwise(start, width):
        if width < 8:
            result = torch.full(
                (rows,), -0.0, dtype=values.dtype, device=values.device,
            )
            for position in range(width):
                result = result + values[:, start + position]
            return result
        if width <= 128:
            accumulators = [values[:, start + index] for index in range(8)]
            stop = width - (width % 8)
            for position in range(8, stop, 8):
                accumulators = [
                    accumulators[index]
                    + values[:, start + position + index]
                    for index in range(8)
                ]
            result = (
                (accumulators[0] + accumulators[1])
                + (accumulators[2] + accumulators[3])
            ) + (
                (accumulators[4] + accumulators[5])
                + (accumulators[6] + accumulators[7])
            )
            for position in range(stop, width):
                result = result + values[:, start + position]
            return result
        left_width = width // 2
        left_width -= left_width % 8
        return pairwise(start, left_width) + pairwise(
            start + left_width, width - left_width,
        )

    return pairwise(0, int(values.shape[1]))


def _numpy_hydrolysis_deletion_plan(binding, tape):
    """Apply the attested positions in literal genome order on host arrays."""
    ragged = binding.ragged
    state = binding.state
    capacity = int(ragged.sequence_capacity)
    width = int(ragged.max_sequence_symbols)
    sequences = int(ragged.sequence_count)
    genomes = int(tape.genome_count)
    final_symbols = np.zeros((capacity, width), dtype=np.uint8)
    final_lengths = np.zeros((capacity,), dtype=np.int64)
    lesions_after = np.asarray(
        ragged.genome_lesions, dtype=np.float64,
    ).copy()
    pools_after = np.asarray(state.pools[:1], dtype=np.float64).copy()
    events = 0
    for sequence_index in range(sequences):
        left = int(ragged.sequence_offsets[sequence_index])
        right = int(ragged.sequence_offsets[sequence_index + 1])
        sequence = np.asarray(
            ragged.symbols[left:right], dtype=np.uint8,
        ).copy()
        if (sequence_index < genomes
                and bool(tape.hit_mask[sequence_index])):
            position = int(tape.deletion_positions[sequence_index])
            if position < 0 or position >= len(sequence):
                raise a4.A4SchemaError(
                    'hydrolysis deletion position lies outside its genome'
                )
            sequence = np.concatenate((
                sequence[:position], sequence[position + 1:],
            ))
            # Preserve the CPU's per-hit fp64 grouping.  Multiplication by
            # hit_count is not equivalent at adversarial waste values.
            pools_after[0, a4.a3.POOL_WASTE] += float(a4.g2.MONOMER_MASS)
            lesions_after[sequence_index] *= 0.80
            events += 1
            # Frozen CPU refreshes the whole cache here.  No later operation
            # in this loop reads it, so the pure plan records the literal
            # refresh count/dirty bit and leaves actual decode to commit.
        final_lengths[sequence_index] = len(sequence)
        if len(sequence):
            final_symbols[sequence_index, :len(sequence)] = sequence
    lesion_mean_after = (
        float(np.mean(lesions_after[:genomes], dtype=np.float64))
        if genomes else 1.0
    )
    plan = A4HydrolysisDeletionPlan(
        schema_version=DELETION_PLAN_SCHEMA_VERSION,
        sequence_capacity=capacity,
        max_sequence_symbols=width,
        sequence_count=sequences,
        genome_count=genomes,
        source_symbol_count=int(ragged.symbol_count),
        cell_id=int(tape.cell_id),
        source_provenance=str(tape.source_provenance),
        tape_schedule_sha256=str(tape.schedule_sha256),
        scope_valid=np.ones((1,), dtype=bool),
        scope_error_code=np.zeros((1,), dtype=np.int64),
        final_symbols=final_symbols,
        final_lengths=final_lengths,
        genome_lesions_after=lesions_after,
        pools_after=pools_after,
        symbol_count_after=np.asarray(
            [int(ragged.symbol_count) - events], dtype=np.int64,
        ),
        topology_symbol_delta=np.asarray([-events], dtype=np.int64),
        genome_damage_event_delta=np.asarray([events], dtype=np.int64),
        gene_cache_dirty=np.asarray([events > 0], dtype=bool),
        gene_cache_refresh_count=np.asarray([events], dtype=np.int64),
        genome_material_symbols_after=np.asarray([
            int(state.genome_material_symbols[0]) - events
        ], dtype=np.int64),
        genome_lesion_mean_after=np.asarray(
            [lesion_mean_after], dtype=np.float64,
        ),
    )
    return _validate_deletion_plan_values(plan)


def _host_arrays_bit_exact(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype == np.dtype(np.float64):
        return np.array_equal(left.view(np.uint64), right.view(np.uint64))
    return np.array_equal(left, right)


def _nonnegative_float64_within_one_ulp(left, right):
    """Allow only the measured CPU/CUDA final-division variance."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (left.shape != right.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all() or np.any(left < 0.0)
            or np.any(right < 0.0)):
        return False
    left_bits = left.view(np.uint64).reshape(-1)
    right_bits = right.view(np.uint64).reshape(-1)
    return all(
        abs(int(left_value) - int(right_value)) <= 1
        for left_value, right_value in zip(left_bits, right_bits)
    )


def validate_a4_hydrolysis_deletion_plan(plan, binding, dt, tape):
    """Re-derive a host plan from the exact source and attested RNG tape."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.ragged.symbols):
        raise a4.A4SchemaError(
            'full hydrolysis deletion validation requires NumPy source'
        )
    if _is_tensor(getattr(plan, 'final_symbols', None)):
        raise a4.A4SchemaError(
            'full hydrolysis deletion validation requires NumPy plan'
        )
    if _is_tensor(getattr(tape, 'uniform_draws', None)):
        raise a4.A4SchemaError(
            'full hydrolysis deletion validation requires NumPy tape'
        )
    _require_numpy_single_cell_binding(binding)
    tape = validate_a4_hydrolysis_rng_tape(tape, binding, dt)
    _require_hydrolysis_tape_binding(tape, binding, dt)
    _validate_deletion_plan_values(plan)
    ragged = binding.ragged
    scalar_expected = {
        'schema_version': DELETION_PLAN_SCHEMA_VERSION,
        'sequence_capacity': int(ragged.sequence_capacity),
        'max_sequence_symbols': int(ragged.max_sequence_symbols),
        'sequence_count': int(ragged.sequence_count),
        'genome_count': int(tape.genome_count),
        'source_symbol_count': int(ragged.symbol_count),
        'cell_id': int(tape.cell_id),
        'source_provenance': str(tape.source_provenance),
        'tape_schedule_sha256': str(tape.schedule_sha256),
    }
    for name, expected_value in scalar_expected.items():
        if getattr(plan, name) != expected_value:
            raise A4HydrolysisScopeError(
                'hydrolysis deletion plan does not bind this source/tape'
            )
    expected = _numpy_hydrolysis_deletion_plan(binding, tape)
    for name in _DELETION_PLAN_ARRAY_FIELDS:
        if name == 'genome_lesion_mean_after':
            equal = _nonnegative_float64_within_one_ulp(
                getattr(plan, name), getattr(expected, name),
            )
        else:
            equal = _host_arrays_bit_exact(
                getattr(plan, name), getattr(expected, name),
            )
        if not equal:
            raise a4.A4SchemaError(
                'hydrolysis deletion plan %s differs from replay' % name
            )
    return plan


def genome_hydrolysis_deletion_numpy(binding, dt, tape):
    """Produce one pure NumPy deletion/ledger descriptor."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.ragged.symbols):
        raise a4.A4SchemaError(
            'NumPy hydrolysis deletion requires a NumPy binding'
        )
    _require_numpy_single_cell_binding(binding)
    tape = validate_a4_hydrolysis_rng_tape(tape, binding, dt)
    _require_hydrolysis_tape_binding(tape, binding, dt)
    plan = _numpy_hydrolysis_deletion_plan(binding, tape)
    return validate_a4_hydrolysis_deletion_plan(
        plan, binding, dt, tape,
    )


def genome_hydrolysis_deletion_torch(binding, dt, tape):
    """Apply one trusted tape with fixed resident CPU/CUDA operations."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not _is_tensor(binding.ragged.symbols):
        raise a4.A4SchemaError(
            'Torch hydrolysis deletion requires a Torch binding'
        )
    tape = _require_hydrolysis_tape_binding(tape, binding, dt)
    ragged = binding.ragged
    state = binding.state
    device = ragged.symbols.device
    dtype = state.pools.dtype
    capacity = int(ragged.sequence_capacity)
    width = int(ragged.max_sequence_symbols)
    sequences = int(ragged.sequence_count)
    genomes = int(tape.genome_count)
    symbol_capacity = int(ragged.symbol_capacity)
    source_symbol_count = int(ragged.symbol_count)

    tape_unchanged = _resident_hydrolysis_tape_unchanged(tape)
    sequence_rank = torch.arange(
        capacity, dtype=torch.int64, device=device,
    )
    symbol_rank = torch.arange(
        width, dtype=torch.int64, device=device,
    )
    used_sequence = sequence_rank < sequences
    genome_slot = sequence_rank < genomes
    starts = torch.where(
        used_sequence, ragged.sequence_offsets[:capacity],
        torch.zeros((capacity,), dtype=torch.int64, device=device),
    )
    ends = torch.where(
        used_sequence, ragged.sequence_offsets[1:capacity + 1], starts,
    )
    source_lengths = ends - starts
    hit = tape.hit_mask & genome_slot
    positions = tape.deletion_positions

    source_relation_ok = (
        (ragged.cell_ids[0] == int(tape.cell_id))
        & (ragged.genome_counts[0] == genomes)
        & (ragged.cell_sequence_offsets[0] == 0)
        & (ragged.cell_sequence_offsets[1] == sequences)
        & (ragged.lesion_offsets[0] == 0)
        & (ragged.lesion_offsets[1] == genomes)
        & (ragged.sequence_offsets[sequences] == source_symbol_count)
        & torch.all(tape.genome_slot_mask == genome_slot)
        & torch.all(torch.where(
            used_sequence,
            (source_lengths >= 0) & (source_lengths <= width),
            source_lengths == 0,
        ))
        & torch.all(torch.where(
            hit,
            (positions >= 0) & (positions < source_lengths)
            & (source_lengths > int(a4.g2.MIN_GENOME_LENGTH)),
            positions == -1,
        ))
        & (torch.sum(hit.to(torch.int64)) == int(tape.hit_count))
    )
    semantic_ok = tape_unchanged & source_relation_ok

    source_indices = starts[:, None] + symbol_rank[None, :]
    safe_source_indices = torch.clamp(
        source_indices, min=0, max=symbol_capacity - 1,
    )
    source_rows = ragged.symbols[safe_source_indices]
    source_rows = torch.where(
        used_sequence[:, None]
        & (symbol_rank[None, :] < source_lengths[:, None]),
        source_rows, torch.zeros_like(source_rows),
    )

    shifted_rank = symbol_rank[None, :] + (
        hit[:, None]
        & (symbol_rank[None, :] >= positions[:, None])
    ).to(torch.int64)
    deleted_indices = starts[:, None] + shifted_rank
    safe_deleted_indices = torch.clamp(
        deleted_indices, min=0, max=symbol_capacity - 1,
    )
    deleted_lengths = source_lengths - hit.to(torch.int64)
    deleted_rows = ragged.symbols[safe_deleted_indices]
    deleted_rows = torch.where(
        used_sequence[:, None]
        & (symbol_rank[None, :] < deleted_lengths[:, None]),
        deleted_rows, torch.zeros_like(deleted_rows),
    )

    lesions_source = ragged.genome_lesions
    lesions_candidate = torch.where(
        hit, lesions_source * 0.80, lesions_source,
    )
    event_count = torch.sum(hit.to(torch.int64)).reshape(1)
    pools_candidate = state.pools[:1].clone()
    waste_after = state.pools[0, a4.a3.POOL_WASTE]
    for sequence_index in range(capacity):
        waste_after = torch.where(
            hit[sequence_index],
            waste_after + float(a4.g2.MONOMER_MASS), waste_after,
        )
    pools_candidate[0, a4.a3.POOL_WASTE] = waste_after

    if genomes:
        lesion_total = _torch_numpy_pairwise_sum_rows(
            lesions_candidate[:genomes].reshape(1, genomes)
        )
        lesion_mean_candidate = lesion_total / float(genomes)
    else:
        lesion_mean_candidate = torch.ones(
            (1,), dtype=dtype, device=device,
        )
    symbol_count_candidate = torch.full(
        (1,), source_symbol_count, dtype=torch.int64, device=device,
    ) - event_count
    topology_candidate = -event_count
    material_candidate = state.genome_material_symbols[:1] - event_count
    dirty_candidate = event_count > 0

    success = semantic_ok.reshape(1)
    final_symbols = torch.where(
        success[:, None, None], deleted_rows[None, :, :],
        source_rows[None, :, :],
    )[0]
    final_lengths = torch.where(
        success, deleted_lengths[None, :], source_lengths[None, :],
    )[0]
    lesions_after = torch.where(
        success, lesions_candidate[None, :], lesions_source[None, :],
    )[0]
    pools_after = torch.where(
        success[:, None], pools_candidate, state.pools[:1],
    )
    symbol_count_after = torch.where(
        success, symbol_count_candidate,
        torch.full_like(symbol_count_candidate, source_symbol_count),
    )
    topology_delta = torch.where(
        success, topology_candidate, torch.zeros_like(topology_candidate),
    )
    damage_delta = torch.where(
        success, event_count, torch.zeros_like(event_count),
    )
    cache_dirty = torch.where(
        success, dirty_candidate, torch.zeros_like(dirty_candidate),
    )
    refresh_count = torch.where(
        success, event_count, torch.zeros_like(event_count),
    )
    material_after = torch.where(
        success, material_candidate, state.genome_material_symbols[:1],
    )
    lesion_mean_after = torch.where(
        success, lesion_mean_candidate, state.genome_lesion_mean[:1],
    )
    scope_error = torch.where(
        success,
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.full(
            (1,), DELETION_SCOPE_TAPE_MISMATCH,
            dtype=torch.int64, device=device,
        ),
    )
    plan = A4HydrolysisDeletionPlan(
        schema_version=DELETION_PLAN_SCHEMA_VERSION,
        sequence_capacity=capacity,
        max_sequence_symbols=width,
        sequence_count=sequences,
        genome_count=genomes,
        source_symbol_count=source_symbol_count,
        cell_id=int(tape.cell_id),
        source_provenance=str(tape.source_provenance),
        tape_schedule_sha256=str(tape.schedule_sha256),
        scope_valid=success,
        scope_error_code=scope_error,
        final_symbols=final_symbols,
        final_lengths=final_lengths,
        genome_lesions_after=lesions_after,
        pools_after=pools_after,
        symbol_count_after=symbol_count_after,
        topology_symbol_delta=topology_delta,
        genome_damage_event_delta=damage_delta,
        gene_cache_dirty=cache_dirty,
        gene_cache_refresh_count=refresh_count,
        genome_material_symbols_after=material_after,
        genome_lesion_mean_after=lesion_mean_after,
    )
    _validate_deletion_plan_metadata(plan)
    return plan


def genome_hydrolysis_deletion_plan(binding, dt, tape):
    """Dispatch the pure A4.7b descriptor without committing any state."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.ragged.symbols):
        return genome_hydrolysis_deletion_torch(binding, dt, tape)
    return genome_hydrolysis_deletion_numpy(binding, dt, tape)


PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'genome_symbol_hydrolysis_rng': (
        'a4.7a-single-cell-post-gain-literal-pcg64-event-tape-'
        'not-applied-not-live-rng-committed-not-integrated-cpu-authoritative'
    ),
    'genome_symbol_hydrolysis': (
        'a4.7b-single-cell-row-padded-deletion-ledger-plan-'
        'not-arena-cache-live-rng-committed-not-integrated-'
        'cpu-authoritative'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION',
    'DELETION_PLAN_SCHEMA_VERSION',
    'FULL_GPU_WORLD_STEP', 'PORT_STATUS',
    'DELETION_SCOPE_OK', 'DELETION_SCOPE_TAPE_MISMATCH',
    'A4HydrolysisError', 'A4HydrolysisScopeError',
    'A4HydrolysisRngTape', 'prepare_genome_hydrolysis_rng_tape',
    'validate_a4_hydrolysis_rng_tape',
    'A4HydrolysisDeletionPlan',
    'validate_a4_hydrolysis_deletion_plan',
    'genome_hydrolysis_deletion_numpy',
    'genome_hydrolysis_deletion_torch',
    'genome_hydrolysis_deletion_plan',
)
