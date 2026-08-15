# coding: utf-8
"""A4.7a event-local genome-symbol hydrolysis RNG tape.

This deliberately small development slice records the literal frozen 0.3
high-level PCG64 calls for one cell at the post-lesion-gain, pre-hydrolysis
event boundary.  It does not edit a polymer, mutate the ragged arena or gene
cache, advance a live world RNG, or replace A3 scheduler authority.

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
from dataclasses import dataclass

import numpy as np

import SOMA_CELL_0_6_8_gpu_a4 as a4

try:
    import torch
except Exception:  # pragma: no cover - NumPy reference remains importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.7a'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | event-local genome hydrolysis PCG64 tape'
SCHEMA_VERSION = '0.6.8-GPU-A4.7a-genome-hydrolysis-rng-tape'
FULL_GPU_WORLD_STEP = False

_TAPE_ARRAY_FIELDS = (
    'genome_slot_mask', 'draw_mask', 'uniform_draws', 'hit_mask',
    'deletion_positions',
)
_TAPE_BOOL_FIELDS = ('genome_slot_mask', 'draw_mask', 'hit_mask')
_TAPE_FLOAT64_FIELDS = ('uniform_draws',)
_TAPE_INT64_FIELDS = ('deletion_positions',)
_TAPE_FACTORY_TOKEN = object()


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


PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'genome_symbol_hydrolysis_rng': (
        'a4.7a-single-cell-post-gain-literal-pcg64-event-tape-'
        'not-applied-not-live-rng-committed-not-integrated-cpu-authoritative'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION',
    'FULL_GPU_WORLD_STEP', 'PORT_STATUS',
    'A4HydrolysisError', 'A4HydrolysisScopeError',
    'A4HydrolysisRngTape', 'prepare_genome_hydrolysis_rng_tape',
    'validate_a4_hydrolysis_rng_tape',
)
