# coding: utf-8
"""A4.5b pure resident template-start and substitution-RNG plan.

This deliberately narrow development slice handles a pre-existing active
replication template or, only in its mutation-enabled tape path, the frozen
index-zero start from exactly one complete genome.  The partial copy must
remain incomplete in this call.  It retains deterministic proofreading,
inherited/behavioural quiescence, an external-replicase contribution, and an
event-local PCG64 substitution tape.  It does not commit the
ragged topology, complete a genome, perform structural mutation, advance the
live world RNG, or replace the A3 scheduler.  Frozen Formal066 CPU behavior
remains authority.
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


BUILD = 'SOMA-CELL 0.6.8-GPU A4.5b'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | template-start and PCG64 substitution-tape plan'
SCHEMA_VERSION = '0.6.8-GPU-A4.5b-template-start-substitution-plan'
RNG_TAPE_SCHEMA_VERSION = '0.6.8-GPU-A4.5b-template-start-rng-tape'
FULL_GPU_WORLD_STEP = False

SCOPE_OK = 0
SCOPE_INACTIVE_TEMPLATE = 1
SCOPE_REPLICASE_GATE = 2
SCOPE_COMPLETION = 3
SCOPE_CAPACITY = 4
SCOPE_NEGATIVE_ATP = 5
SCOPE_FP64_DISCRETE_BOUNDARY = 6
SCOPE_RNG_TAPE_MISMATCH = 7

# CPU and CUDA fp64 division may differ by a few ulps.  Do not turn that
# continuous discrepancy into a different integer symbol request.  The
# deliberately conservative band is an engineering scope boundary, not a
# replacement arithmetic model; ambiguous rows remain CPU-authoritative.
FP64_DISCRETE_GUARD_EPS = 4096.0
FP64_EPSILON = float(np.finfo(np.float64).eps)

_PLAN_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code',
    'requested_symbols', 'append_symbols', 'append_count', 'pools_after',
    'replication_fractional_after', 'last_replication_symbols',
    'last_effective_error_rate', 'cumulative_proofreading_atp_after',
    'substitution_events', 'template_start_events',
    'selected_template_indices', 'template_storage_symbols',
)
_PLAN_UINT8_FIELDS = ('append_symbols',)
_PLAN_INT64_FIELDS = (
    'cell_ids', 'scope_error_code', 'requested_symbols', 'append_count',
    'last_replication_symbols',
    'substitution_events', 'selected_template_indices',
    'template_storage_symbols',
)
_PLAN_BOOL_FIELDS = ('cell_mask', 'scope_valid', 'template_start_events')
_PLAN_FLOAT64_FIELDS = (
    'pools_after', 'replication_fractional_after',
    'last_effective_error_rate', 'cumulative_proofreading_atp_after',
)

_RNG_TAPE_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'draw_mask', 'uniform_draws',
    'replacement_raw', 'replacement_mask', 'draw_count',
    'substitution_count', 'effective_error', 'template_start_mask',
    'template_selection_indices',
)
_RNG_TAPE_UINT8_FIELDS = ('replacement_raw',)
_RNG_TAPE_INT64_FIELDS = (
    'cell_ids', 'draw_count', 'substitution_count',
    'template_selection_indices',
)
_RNG_TAPE_BOOL_FIELDS = (
    'cell_mask', 'draw_mask', 'replacement_mask', 'template_start_mask',
)
_RNG_TAPE_FLOAT64_FIELDS = ('uniform_draws', 'effective_error')
_RNG_TAPE_FACTORY_TOKEN = object()


class A4ReplicationError(a4.A4Error):
    """Base class for this bounded replication slice."""


class A4ReplicationScopeError(A4ReplicationError):
    """The requested row requires a later replication slice."""


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _clone_array(value):
    return value.clone() if _is_tensor(value) else np.asarray(value).copy()


def _host_array(value):
    if _is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _strict_real_scalar(value, label, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise A4ReplicationScopeError('%s must be a real scalar' % label)
    result = float(value)
    if not math.isfinite(result):
        raise A4ReplicationScopeError('%s must be finite' % label)
    if minimum is not None and result < float(minimum):
        raise A4ReplicationScopeError('%s is below supported range' % label)
    return result


def _strict_bool(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise A4ReplicationScopeError('%s must be boolean' % label)
    return bool(value)


def _supported_config(config):
    """Extract exactly the deterministic host flags used by A4.5b."""
    required = {
        'genome_replication': True,
        'mutation': False,
    }
    for name, expected in required.items():
        if not hasattr(config, name):
            raise A4ReplicationScopeError('config missing %s' % name)
        actual = _strict_bool(getattr(config, name), 'config.%s' % name)
        if actual is not expected:
            raise A4ReplicationScopeError(
                'deterministic elongation requires config.%s=%s' % (
                    name, expected,
                )
            )
    flags = {}
    for name in (
            'proofreading', 'external_replicase', 'quiescence',
            'quiescence_effector'):
        if not hasattr(config, name):
            raise A4ReplicationScopeError('config missing %s' % name)
        flags[name] = _strict_bool(
            getattr(config, name), 'config.%s' % name,
        )
    mutation_rate = _strict_real_scalar(
        getattr(config, 'mutation_rate', None), 'config.mutation_rate',
    )
    scale = _strict_real_scalar(
        getattr(config, 'eco66_replication_rate_scale', 1.0),
        'config.eco66_replication_rate_scale',
    )
    return mutation_rate, max(0.25, scale), flags


def _substitution_config(config):
    """Return the mutation-free schedule config and an exact host digest."""
    if not hasattr(config, 'mutation') or not _strict_bool(
            getattr(config, 'mutation'), 'config.mutation'):
        raise A4ReplicationScopeError(
            'A4.5b substitution planning requires config.mutation=True'
        )
    deterministic = copy.deepcopy(config)
    deterministic.mutation = False
    mutation_rate, scale, flags = _supported_config(deterministic)
    payload = {
        'genome_replication': True,
        'mutation': True,
        'mutation_rate_hex': float(mutation_rate).hex(),
        'eco66_replication_rate_scale_hex': float(scale).hex(),
        'proofreading': bool(flags['proofreading']),
        'external_replicase': bool(flags['external_replicase']),
        'quiescence': bool(flags['quiescence']),
        'quiescence_effector': bool(flags['quiescence_effector']),
    }
    return deterministic, _sha256_json(payload)


def _strict_dt(value):
    return _strict_real_scalar(value, 'replication dt', minimum=0.0)


def _numpy_fp64_integer_boundary(fractional_total, increment):
    """Return true when fp64 backend variance can change truncation."""
    fractional_total = float(fractional_total)
    increment = float(increment)
    if not (math.isfinite(fractional_total) and math.isfinite(increment)):
        return True
    if not increment > 0.0:
        return False
    nearest = float(np.rint(np.float64(fractional_total)))
    if nearest < 1.0:
        return False
    tolerance = (
        FP64_DISCRETE_GUARD_EPS * FP64_EPSILON
        * max(1.0, abs(fractional_total))
    )
    return abs(fractional_total - nearest) <= tolerance


def _torch_fp64_integer_boundary(fractional_total, increment):
    """Fixed-shape resident counterpart of the NumPy ambiguity guard."""
    finite = torch.isfinite(fractional_total)
    increment_finite = torch.isfinite(increment)
    safe_total = torch.where(
        finite, fractional_total, torch.zeros_like(fractional_total),
    )
    nearest = torch.round(safe_total)
    tolerance = (
        FP64_DISCRETE_GUARD_EPS * FP64_EPSILON
        * torch.clamp(torch.abs(safe_total), min=1.0)
    )
    return (~finite) | (~increment_finite) | (
        (increment > 0.0)
        & (nearest >= 1.0)
        & (torch.abs(safe_total - nearest) <= tolerance)
    )


def _numpy_fp64_comparison_boundary(left, right):
    """Guard a derived fp64 value before it controls a discrete branch."""
    left = float(left)
    right = float(right)
    if not (math.isfinite(left) and math.isfinite(right)):
        return True
    scale = max(abs(left), abs(right))
    tolerance = FP64_DISCRETE_GUARD_EPS * FP64_EPSILON * scale
    return abs(left - right) <= tolerance


def _torch_fp64_comparison_boundary(left, right):
    """Resident comparison guard with the same registered relative band."""
    if not _is_tensor(right):
        right = torch.full_like(left, float(right))
    finite = torch.isfinite(left) & torch.isfinite(right)
    scale = torch.maximum(torch.abs(left), torch.abs(right))
    tolerance = FP64_DISCRETE_GUARD_EPS * FP64_EPSILON * scale
    return (~finite) | (torch.abs(left - right) <= tolerance)


def _require_binding_scope_flags(state, flags):
    # These flags are carried as trusted host metadata in the A4.3 snapshot.
    # Require them to agree with this slice instead of accepting a state packed
    # under a different quiescence policy and a separately supplied config.
    if (bool(state.quiescence) != bool(flags['quiescence'])
            or bool(state.quiescence_effector)
            != bool(flags['quiescence_effector'])):
        raise A4ReplicationScopeError(
            'A4.5b config quiescence flags differ from the packed snapshot'
        )


@dataclass
class A4PaidElongationPlan:
    """Fixed-shape pure result; it is not world or save authority."""

    schema_version: str
    cell_capacity: int
    append_capacity: int
    cell_count: int
    source_provenance: str
    cell_ids: object
    cell_mask: object
    scope_valid: object
    scope_error_code: object
    requested_symbols: object
    append_symbols: object
    append_count: object
    pools_after: object
    replication_fractional_after: object
    last_replication_symbols: object
    last_effective_error_rate: object
    cumulative_proofreading_atp_after: object
    substitution_events: object
    template_start_events: object
    selected_template_indices: object
    template_storage_symbols: object

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value)
                if item.name in _PLAN_ARRAY_FIELDS else copy.deepcopy(value)
            )
        return A4PaidElongationPlan(**values)

    def to_numpy(self):
        if _is_tensor(self.pools_after):
            _validate_plan_metadata(self)
        else:
            validate_a4_paid_elongation_plan(self)
            return self.clone()
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _PLAN_ARRAY_FIELDS:
                value = _host_array(value)
                if item.name in _PLAN_UINT8_FIELDS:
                    value = value.astype(np.uint8, copy=False)
                elif item.name in _PLAN_INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                elif item.name in _PLAN_BOOL_FIELDS:
                    value = value.astype(bool, copy=False)
                else:
                    value = value.astype(np.float64, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        out = A4PaidElongationPlan(**values)
        return validate_a4_paid_elongation_plan(out)

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _PLAN_ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch-backed plan')
        return {name: int(getattr(self, name).data_ptr())
                for name in _PLAN_ARRAY_FIELDS}

    def state_dict(self):
        return {item.name: (
            _clone_array(getattr(self, item.name))
            if item.name in _PLAN_ARRAY_FIELDS
            else copy.deepcopy(getattr(self, item.name))
        ) for item in fields(self)}


@dataclass
class A4SubstitutionRngTape:
    """Ephemeral event tape; it is neither save nor live-RNG authority."""

    _factory_token: object
    schema_version: str
    cell_capacity: int
    append_capacity: int
    cell_count: int
    source_provenance: str
    dt_hex: str
    config_sha256: str
    schedule_sha256: str
    rng_before_state: object
    rng_after_state: object
    cell_ids: object
    cell_mask: object
    draw_mask: object
    uniform_draws: object
    replacement_raw: object
    replacement_mask: object
    draw_count: object
    substitution_count: object
    effective_error: object
    template_start_mask: object
    template_selection_indices: object

    def clone(self):
        _require_rng_tape(self)
        values = {
            name: (
                _clone_array(getattr(self, name))
                if name in _RNG_TAPE_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, name))
            )
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'cell_count', 'source_provenance', 'dt_hex',
                'config_sha256', 'schedule_sha256', 'rng_before_state',
                'rng_after_state',
            ) + _RNG_TAPE_ARRAY_FIELDS
        }
        return _make_rng_tape(**values)

    def to_torch(self, device='cpu'):
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        validate_a4_substitution_rng_tape(self)
        requested = torch.device(device)
        if requested.type not in ('cpu', 'cuda'):
            raise a4.A4DeviceError('RNG tape device must be cpu or cuda')
        if requested.type == 'cuda' and not torch.cuda.is_available():
            raise a4.A4DeviceError('CUDA requested but unavailable')
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'cell_count', 'source_provenance', 'dt_hex',
                'config_sha256', 'schedule_sha256', 'rng_before_state',
                'rng_after_state',
            )
        }
        for name in _RNG_TAPE_ARRAY_FIELDS:
            value = np.asarray(getattr(self, name))
            dtype = (
                torch.uint8 if name in _RNG_TAPE_UINT8_FIELDS else
                torch.int64 if name in _RNG_TAPE_INT64_FIELDS else
                torch.bool if name in _RNG_TAPE_BOOL_FIELDS else
                torch.float64
            )
            values[name] = torch.as_tensor(
                value, dtype=dtype, device=requested,
            ).clone()
        return _make_rng_tape(**values)

    def to_numpy(self):
        _require_rng_tape(self)
        if not _is_tensor(self.uniform_draws):
            return validate_a4_substitution_rng_tape(self.clone())
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'cell_count', 'source_provenance', 'dt_hex',
                'config_sha256', 'schedule_sha256', 'rng_before_state',
                'rng_after_state',
            )
        }
        for name in _RNG_TAPE_ARRAY_FIELDS:
            value = _host_array(getattr(self, name))
            dtype = (
                np.uint8 if name in _RNG_TAPE_UINT8_FIELDS else
                np.int64 if name in _RNG_TAPE_INT64_FIELDS else
                bool if name in _RNG_TAPE_BOOL_FIELDS else
                np.float64
            )
            values[name] = value.astype(dtype, copy=False)
        return validate_a4_substitution_rng_tape(_make_rng_tape(**values))

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _RNG_TAPE_ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch-backed RNG tape')
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _RNG_TAPE_ARRAY_FIELDS
        }

    def state_dict(self):
        return {
            name: (
                _clone_array(getattr(self, name))
                if name in _RNG_TAPE_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, name))
            )
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'cell_count', 'source_provenance', 'dt_hex',
                'config_sha256', 'schedule_sha256', 'rng_before_state',
                'rng_after_state',
            ) + _RNG_TAPE_ARRAY_FIELDS
        }


def _make_rng_tape(**values):
    tape = A4SubstitutionRngTape(
        _factory_token=_RNG_TAPE_FACTORY_TOKEN, **values
    )
    tape._scalar_metadata = _rng_tape_scalar_metadata(tape)
    if _is_tensor(tape.uniform_draws):
        tape._resident_data_ptrs = tape.data_ptrs()
        tape._resident_versions = {
            name: int(getattr(tape, name)._version)
            for name in _RNG_TAPE_ARRAY_FIELDS
        }
    return tape


def _is_lower_hex_digest(value):
    return (
        isinstance(value, str) and len(value) == 64
        and value == value.lower()
        and all(ch in '0123456789abcdef' for ch in value)
    )


def _sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    ).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _canonical_pcg64_state(value, label):
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
        if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)):
            raise a4.A4SchemaError('%s.%s must be integer' % (label, name))
        item = int(item)
        if item < 0 or item >= (1 << 128):
            raise a4.A4SchemaError('%s.%s outside uint128' % (label, name))
        result['state'][name] = item
    for name, upper in (('has_uint32', 2), ('uinteger', 1 << 32)):
        item = value[name]
        if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)):
            raise a4.A4SchemaError('%s.%s must be integer' % (label, name))
        item = int(item)
        if item < 0 or item >= upper:
            raise a4.A4SchemaError('%s.%s outside range' % (label, name))
        result[name] = item
    try:
        bit_generator = np.random.PCG64()
        bit_generator.state = copy.deepcopy(result)
    except Exception as exc:
        raise a4.A4SchemaError('%s is not accepted by NumPy PCG64' % label) from exc
    if bit_generator.state != result:
        raise a4.A4SchemaError('%s is not a canonical PCG64 state' % label)
    return result


def _rng_tape_scalar_metadata(tape):
    """Immutable identity for an uploaded one-event PCG64 tape."""
    before = _canonical_pcg64_state(
        tape.rng_before_state, 'rng_before_state',
    )
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    return (
        str(tape.schema_version), int(tape.cell_capacity),
        int(tape.append_capacity), int(tape.cell_count),
        str(tape.source_provenance), str(tape.dt_hex),
        str(tape.config_sha256), str(tape.schedule_sha256),
        _sha256_json(before), _sha256_json(after),
    )


def _rng_schedule_digest(tape):
    N = int(tape.cell_count)
    payload = {
        'schema': str(tape.schema_version),
        'source': str(tape.source_provenance),
        'dt_hex': str(tape.dt_hex),
        'config_sha256': str(tape.config_sha256),
        'cell_ids': [int(value) for value in np.asarray(tape.cell_ids)[:N]],
        'draw_count': [
            int(value) for value in np.asarray(tape.draw_count)[:N]
        ],
        'template_start_mask': [
            bool(value) for value in np.asarray(tape.template_start_mask)[:N]
        ],
        'template_selection_indices': [
            int(value)
            for value in np.asarray(tape.template_selection_indices)[:N]
        ],
        'effective_error_hex': [
            float(value).hex()
            for value in np.asarray(tape.effective_error)[:N]
        ],
    }
    return _sha256_json(payload)


def _validate_rng_tape_metadata(tape):
    if (not isinstance(tape, A4SubstitutionRngTape)
            or tape._factory_token is not _RNG_TAPE_FACTORY_TOKEN):
        raise a4.A4SchemaError('RNG tape must come from the private factory')
    if tape.schema_version != RNG_TAPE_SCHEMA_VERSION:
        raise a4.A4SchemaError('RNG tape schema mismatch')
    for name in ('cell_capacity', 'append_capacity', 'cell_count'):
        value = getattr(tape, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise a4.A4SchemaError('%s must be integer' % name)
    C = int(tape.cell_capacity)
    W = int(tape.append_capacity)
    N = int(tape.cell_count)
    if C <= 0 or W <= 0 or N < 0 or N > C:
        raise a4.A4SchemaError('RNG tape capacities/count are invalid')
    if (not _is_lower_hex_digest(tape.source_provenance)
            or not _is_lower_hex_digest(tape.config_sha256)
            or not _is_lower_hex_digest(tape.schedule_sha256)):
        raise a4.A4SchemaError('RNG tape provenance digest is invalid')
    try:
        parsed_dt = float.fromhex(tape.dt_hex)
    except Exception as exc:
        raise a4.A4SchemaError('RNG tape dt hex is invalid') from exc
    if not math.isfinite(parsed_dt) or parsed_dt < 0.0:
        raise a4.A4SchemaError('RNG tape dt is outside supported range')
    _canonical_pcg64_state(tape.rng_before_state, 'rng_before_state')
    _canonical_pcg64_state(tape.rng_after_state, 'rng_after_state')
    kinds = set()
    devices = set()
    for name in _RNG_TAPE_ARRAY_FIELDS:
        value = getattr(tape, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise a4.A4SchemaError('%s is not an array/tensor' % name)
    if len(kinds) != 1 or len(devices) > 1:
        raise a4.A4SchemaError('mixed RNG tape backend/device is forbidden')
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_RNG_TAPE_UINT8_FIELDS, np.dtype(np.uint8),
         getattr(torch, 'uint8', None)),
        (_RNG_TAPE_INT64_FIELDS, np.dtype(np.int64),
         getattr(torch, 'int64', None)),
        (_RNG_TAPE_BOOL_FIELDS, np.dtype(bool), getattr(torch, 'bool', None)),
        (_RNG_TAPE_FLOAT64_FIELDS, np.dtype(np.float64),
         getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(tape, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    shapes = {
        'cell_ids': (C,), 'cell_mask': (C,),
        'draw_mask': (C, W), 'uniform_draws': (C, W),
        'replacement_raw': (C, W), 'replacement_mask': (C, W),
        'draw_count': (C,), 'substitution_count': (C,),
        'effective_error': (C,),
        'template_start_mask': (C,),
        'template_selection_indices': (C,),
    }
    for name, shape in shapes.items():
        if tuple(getattr(tape, name).shape) != shape:
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return backend


def _require_rng_tape(tape):
    backend = _validate_rng_tape_metadata(tape)
    if getattr(tape, '_scalar_metadata', None) != (
            _rng_tape_scalar_metadata(tape)):
        raise a4.A4SchemaError('RNG tape scalar metadata changed after creation')
    if backend == 'torch':
        if (getattr(tape, '_resident_data_ptrs', None) != tape.data_ptrs()
                or getattr(tape, '_resident_versions', None) != {
                    name: int(getattr(tape, name)._version)
                    for name in _RNG_TAPE_ARRAY_FIELDS
                }):
            raise a4.A4SchemaError('resident RNG tape changed after upload')
    return tape


def validate_a4_substitution_rng_tape(tape):
    """Replay a host tape and prove PCG64 before/after state exactly."""
    _require_rng_tape(tape)
    backend = _validate_rng_tape_metadata(tape)
    if backend != 'numpy':
        raise a4.A4SchemaError('full RNG tape validation requires NumPy')
    raw = {
        name: np.asarray(getattr(tape, name))
        for name in _RNG_TAPE_ARRAY_FIELDS
    }
    C = int(tape.cell_capacity)
    W = int(tape.append_capacity)
    N = int(tape.cell_count)
    prefix = np.arange(W, dtype=np.int64)[None, :] < raw['draw_count'][:, None]
    if (not np.array_equal(raw['cell_mask'], np.arange(C) < N)
            or not np.array_equal(raw['draw_mask'], prefix)
            or np.any(raw['draw_count'][:N] < 0)
            or np.any(raw['draw_count'][:N] > W)
            or np.any(raw['draw_count'][N:] != 0)):
        raise a4.A4SchemaError('RNG tape draw prefix/count is invalid')
    if (np.any(raw['cell_ids'][:N] < 0)
            or len(set(int(v) for v in raw['cell_ids'][:N])) != N
            or np.any(raw['cell_ids'][N:] != -1)):
        raise a4.A4SchemaError('RNG tape cell identity is invalid')
    if (not np.isfinite(raw['uniform_draws']).all()
            or np.any(raw['uniform_draws'] < 0.0)
            or np.any(raw['uniform_draws'] >= 1.0)
            or not np.isfinite(raw['effective_error']).all()
            or np.any(raw['effective_error'][:N] < 0.0)
            or np.any(raw['effective_error'][N:] != 0.0)):
        raise a4.A4SchemaError('RNG tape floating values are invalid')
    if (np.any(raw['replacement_mask'] & ~raw['draw_mask'])
            or np.any(raw['replacement_raw'][raw['replacement_mask']] >= 7)
            or np.any(raw['replacement_raw'][~raw['replacement_mask']] != 0)
            or np.any(raw['uniform_draws'][~raw['draw_mask']] != 0.0)):
        raise a4.A4SchemaError('RNG tape replacement/tail is invalid')
    if (np.any(raw['template_start_mask'] & ~raw['cell_mask'])
            or np.any(raw['template_selection_indices'][
                raw['template_start_mask']] != 0)
            or np.any(raw['template_selection_indices'][
                ~raw['template_start_mask']] != -1)):
        raise a4.A4SchemaError('RNG tape template selection is invalid')
    expected_counts = np.sum(
        raw['replacement_mask'], axis=1, dtype=np.int64,
    )
    if (not np.array_equal(expected_counts, raw['substitution_count'])
            or np.any(raw['substitution_count'] > raw['draw_count'])):
        raise a4.A4SchemaError('RNG tape substitution count is invalid')
    if tape.schedule_sha256 != _rng_schedule_digest(tape):
        raise a4.A4SchemaError('RNG tape schedule provenance differs')
    before = _canonical_pcg64_state(
        tape.rng_before_state, 'rng_before_state',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    for ci in range(N):
        if bool(raw['template_start_mask'][ci]):
            selected = int(generator.integers(0, 1))
            if selected != int(raw['template_selection_indices'][ci]):
                raise a4.A4SchemaError(
                    'RNG tape template selection replay differs'
                )
        for rank in range(int(raw['draw_count'][ci])):
            uniform = float(generator.random())
            if np.float64(uniform).view(np.uint64) != np.float64(
                    raw['uniform_draws'][ci, rank]).view(np.uint64):
                raise a4.A4SchemaError('RNG tape uniform replay differs')
            expected_hit = uniform < float(raw['effective_error'][ci])
            if bool(raw['replacement_mask'][ci, rank]) != expected_hit:
                raise a4.A4SchemaError('RNG tape mutation decision differs')
            if expected_hit:
                replacement = int(generator.integers(0, 7))
                if replacement != int(raw['replacement_raw'][ci, rank]):
                    raise a4.A4SchemaError('RNG tape integer replay differs')
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    if generator.bit_generator.state != after:
        raise a4.A4SchemaError('RNG tape after-state differs from replay')
    return tape


def _validate_plan_backend(plan):
    kinds = set()
    devices = set()
    for name in _PLAN_ARRAY_FIELDS:
        value = getattr(plan, name)
        if _is_tensor(value):
            kinds.add('torch')
            devices.add(str(value.device))
        elif isinstance(value, np.ndarray):
            kinds.add('numpy')
        else:
            raise a4.A4SchemaError('%s is not an array/tensor' % name)
    if len(kinds) != 1 or len(devices) > 1:
        raise a4.A4SchemaError('mixed plan backend/device is forbidden')
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_PLAN_UINT8_FIELDS, np.dtype(np.uint8), getattr(torch, 'uint8', None)),
        (_PLAN_INT64_FIELDS, np.dtype(np.int64), getattr(torch, 'int64', None)),
        (_PLAN_BOOL_FIELDS, np.dtype(bool), getattr(torch, 'bool', None)),
        (_PLAN_FLOAT64_FIELDS, np.dtype(np.float64), getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(plan, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    return backend


def _validate_plan_metadata(plan):
    if not isinstance(plan, A4PaidElongationPlan):
        raise a4.A4SchemaError('expected A4PaidElongationPlan')
    if plan.schema_version != SCHEMA_VERSION:
        raise a4.A4SchemaError('replication plan schema mismatch')
    for name in ('cell_capacity', 'append_capacity', 'cell_count'):
        value = getattr(plan, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise a4.A4SchemaError('%s must be an integer scalar' % name)
    if plan.cell_capacity <= 0 or plan.append_capacity <= 0:
        raise a4.A4SchemaError('replication plan capacities must be positive')
    if plan.cell_count < 0 or plan.cell_count > plan.cell_capacity:
        raise a4.A4SchemaError('replication plan cell count outside capacity')
    if (not isinstance(plan.source_provenance, str)
            or len(plan.source_provenance) != 64
            or plan.source_provenance != plan.source_provenance.lower()
            or any(ch not in '0123456789abcdef'
                   for ch in plan.source_provenance)):
        raise a4.A4SchemaError('replication source provenance is invalid')
    _validate_plan_backend(plan)
    C = int(plan.cell_capacity)
    W = int(plan.append_capacity)
    shapes = {
        'cell_ids': (C,), 'cell_mask': (C,), 'scope_valid': (C,),
        'scope_error_code': (C,), 'requested_symbols': (C,),
        'append_symbols': (C, W), 'append_count': (C,),
        'pools_after': (C, int(a4.a3.POOL_COUNT)),
        'replication_fractional_after': (C,),
        'last_replication_symbols': (C,),
        'last_effective_error_rate': (C,),
        'cumulative_proofreading_atp_after': (C,),
        'substitution_events': (C,),
        'template_start_events': (C,),
        'selected_template_indices': (C,),
        'template_storage_symbols': (C,),
    }
    for name, shape in shapes.items():
        if tuple(getattr(plan, name).shape) != shape:
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return plan


def validate_a4_paid_elongation_plan(plan):
    """Validate a host plan and reject every unsupported/capacity result."""
    _validate_plan_metadata(plan)
    raw = {name: np.asarray(getattr(plan, name))
           for name in _PLAN_ARRAY_FIELDS}
    C = int(plan.cell_capacity)
    W = int(plan.append_capacity)
    N = int(plan.cell_count)
    if not np.array_equal(raw['cell_mask'], np.arange(C) < N):
        raise a4.A4SchemaError('replication plan mask is not a true prefix')
    if (np.any(raw['cell_ids'][:N] < 0)
            or len(set(int(v) for v in raw['cell_ids'][:N])) != N
            or np.any(raw['cell_ids'][N:] != -1)):
        raise a4.A4SchemaError('replication plan cell identity is invalid')
    error_codes = raw['scope_error_code'][:N]
    if np.any(error_codes == SCOPE_CAPACITY):
        raise a4.A4CapacityError('paid elongation exceeds symbol capacity')
    if np.any(error_codes != SCOPE_OK) or not np.all(raw['scope_valid'][:N]):
        raise A4ReplicationScopeError(
            'paid elongation row is outside A4.5b scope: %s' %
            [int(value) for value in error_codes]
        )
    if np.any(raw['scope_valid'][N:]) or np.any(
            raw['scope_error_code'][N:] != SCOPE_OK):
        raise a4.A4SchemaError('unused scope tail is not canonical')
    if (np.any(raw['requested_symbols'][:N] < 0)
            or np.any(raw['append_count'][:N] < 0)
            or np.any(raw['append_count'][:N] > W)
            or np.any(raw['append_count'][:N] > raw['requested_symbols'][:N])
            or np.any(raw['substitution_events'][:N] < 0)
            or np.any(raw['substitution_events'][:N]
                      > raw['append_count'][:N])):
        raise a4.A4SchemaError('replication counts are invalid')
    start = raw['template_start_events']
    if (np.any(start & ~raw['scope_valid'])
            or np.any(raw['selected_template_indices'][start] != 0)
            or np.any(raw['selected_template_indices'][~start] != -1)
            or np.any(raw['template_storage_symbols'][start] <= 0)
            or np.any(raw['template_storage_symbols'][~start] != 0)
            or np.any(raw['template_storage_symbols'][:N] > W)):
        raise a4.A4SchemaError('template-start topology metadata is invalid')
    if not np.array_equal(
            raw['last_replication_symbols'], raw['append_count']):
        raise a4.A4SchemaError('last replication count differs from append count')
    if (not np.isfinite(raw['pools_after']).all()
            or not np.isfinite(raw['replication_fractional_after']).all()
            or not np.isfinite(raw['last_effective_error_rate']).all()
            or not np.isfinite(
                raw['cumulative_proofreading_atp_after']).all()):
        raise a4.A4SchemaError('replication plan contains nonfinite values')
    if (np.any(raw['replication_fractional_after'][:N] < 0.0)
            or np.any(raw['replication_fractional_after'][:N] >= 1.0)
            or np.any(raw['last_effective_error_rate'][:N] < 0.0)
            or np.any(raw['cumulative_proofreading_atp_after'][:N] < 0.0)):
        raise a4.A4SchemaError('replication telemetry outside range')
    # Only unchanged fuel/mineral may carry the inherited A4.3 sequential
    # fp64 residual.  A supported row rejects negative entry ATP and keeps the
    # frozen 0.022 reserve, while nucleotide admission uses the exact monomer
    # constant; both replication-paid pools therefore remain nonnegative.
    allowed_negative = set(a4.TRANSLATION_PAID_POOL_INDICES)
    allowed_negative.discard(int(a4.a3.POOL_ATP))
    for pool_index in range(int(a4.a3.POOL_COUNT)):
        minimum = -a4.TRANSLATION_LEDGER_ATOL if pool_index in allowed_negative else 0.0
        if np.any(raw['pools_after'][:N, pool_index] < minimum):
            raise a4.A4SchemaError('replication pool outside ledger bounds')
    for ci in range(N):
        count = int(raw['append_count'][ci])
        used = raw['append_symbols'][ci, :count]
        if np.any(used >= a4.ALPHABET_SIZE):
            raise a4.A4SchemaError('append symbol outside frozen alphabet')
        if np.any(raw['append_symbols'][ci, count:] != 0):
            raise a4.A4SchemaError('append tail is not zero')
    for name in _PLAN_ARRAY_FIELDS:
        if name in (
                'cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code',
                'selected_template_indices'):
            continue
        tail = raw[name][N:]
        if tail.size and np.any(tail != 0):
            raise a4.A4SchemaError('%s unused cell tail is not zero' % name)
    if np.any(raw['selected_template_indices'][N:] != -1):
        raise a4.A4SchemaError('selected-template tail is not -1')
    return plan


def _numpy_replicase(binding, ci, specs):
    state = binding.state
    total = 0.0
    for position in range(int(state.active_count[ci])):
        fingerprint = int(state.active_fingerprints[ci, position])
        spec = specs.get(fingerprint)
        if spec is not None and int(spec['role']) == int(a4.a3.ROLE_REPLICASE):
            total += float(state.active_mass[ci, position]) * float(spec['efficiency'])
    pools = state.pools[ci]
    radius = float(state.radius[ci])
    volume = max(0.20, (radius / float(a4.s5.BASE_RADIUS)) ** 2)
    aggregate_concentration = float(
        pools[a4.a3.POOL_AGGREGATE] / volume
    )
    active = max(0.0, float(pools[a4.a3.POOL_CATALYST]))
    damaged = float(pools[a4.a3.POOL_DAMAGED_PROTEIN])
    aggregate = float(pools[a4.a3.POOL_AGGREGATE])
    functional = active / max(1e-9, active + damaged + aggregate)
    toxicity = 1.0 / (1.0 + 3.6 * aggregate_concentration)
    proteostasis = float(np.clip(functional * toxicity, 0.02, 1.0))
    genome_factor = 1.0 / (
        1.0 + 0.85 * float(state.genome_lesion_mean[ci])
    )
    return float((total / 0.040) * proteostasis * genome_factor)


def _numpy_raw_repair(binding, ci, specs, kind, aggregate):
    """Literal 0.4 LOC_REPAIR activity in active-dictionary order."""
    state = binding.state
    total = 0.0
    for position in range(int(state.active_count[ci])):
        fingerprint = int(state.active_fingerprints[ci, position])
        spec = specs.get(fingerprint)
        if (
                spec is not None
                and int(spec['role']) == int(a4.a3.ROLE_REGULATOR)
                and int(spec['localisation']) == int(a4.s4.LOC_REPAIR)
                and int(spec['parameter']) % int(a4.a3.REPAIR_COUNT)
                == int(kind)):
            total += (
                float(state.active_mass[ci, position])
                * float(spec['efficiency']) * float(spec['promoter'])
            )
    inhibition = 1.0 / (1.0 + 2.6 * float(aggregate))
    return float((total / 0.014) * inhibition)


def _numpy_template_copy(binding, ci, allow_template_start=False):
    ragged = binding.ragged
    if bool(ragged.replication_active[ci]):
        sequence_end = int(ragged.cell_sequence_offsets[ci + 1])
        template_index = sequence_end - 2
        copy_index = sequence_end - 1
        template_lesion = float(ragged.replication_template_lesions[ci])
        start_event = False
    elif allow_template_start and int(ragged.genome_counts[ci]) == 1:
        template_index = int(ragged.cell_sequence_offsets[ci])
        copy_index = None
        lesion_first = int(ragged.lesion_offsets[ci])
        lesion_last = int(ragged.lesion_offsets[ci + 1])
        template_lesion = (
            float(ragged.genome_lesions[lesion_first])
            if lesion_last > lesion_first else 0.0
        )
        start_event = True
    else:
        return None, None, False, -1, 0.0
    template_start = int(ragged.sequence_offsets[template_index])
    template_end = int(ragged.sequence_offsets[template_index + 1])
    if copy_index is None:
        partial = np.zeros((0,), dtype=np.uint8)
    else:
        copy_start = int(ragged.sequence_offsets[copy_index])
        copy_end = int(ragged.sequence_offsets[copy_index + 1])
        partial = np.asarray(
            ragged.symbols[copy_start:copy_end], dtype=np.uint8,
        )
    return (
        np.asarray(ragged.symbols[template_start:template_end], dtype=np.uint8),
        partial, start_event, 0 if start_event else -1, template_lesion,
    )


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


def _empty_numpy_plan(binding):
    state = binding.state
    ragged = binding.ragged
    C = int(state.cell_capacity)
    W = int(ragged.max_sequence_symbols)
    return A4PaidElongationPlan(
        schema_version=SCHEMA_VERSION,
        cell_capacity=C,
        append_capacity=W,
        cell_count=int(state.cell_count),
        source_provenance=str(state.source_provenance),
        cell_ids=np.asarray(state.cell_ids, dtype=np.int64).copy(),
        cell_mask=np.asarray(state.cell_mask, dtype=bool).copy(),
        scope_valid=np.zeros((C,), dtype=bool),
        scope_error_code=np.zeros((C,), dtype=np.int64),
        requested_symbols=np.zeros((C,), dtype=np.int64),
        append_symbols=np.zeros((C, W), dtype=np.uint8),
        append_count=np.zeros((C,), dtype=np.int64),
        pools_after=np.asarray(state.pools, dtype=np.float64).copy(),
        replication_fractional_after=np.zeros((C,), dtype=np.float64),
        last_replication_symbols=np.zeros((C,), dtype=np.int64),
        last_effective_error_rate=np.zeros((C,), dtype=np.float64),
        cumulative_proofreading_atp_after=np.asarray(
            state.cumulative_proofreading_atp, dtype=np.float64,
        ).copy(),
        substitution_events=np.zeros((C,), dtype=np.int64),
        template_start_events=np.zeros((C,), dtype=bool),
        selected_template_indices=np.full((C,), -1, dtype=np.int64),
        template_storage_symbols=np.zeros((C,), dtype=np.int64),
    )


def _paid_replication_elongation_numpy(
        binding, dt, config, allow_template_start=False):
    """Literal NumPy reference for the bounded Formal066 continuation."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('NumPy elongation requires a NumPy binding')
    a4.validate_a4_ragged(binding.ragged)
    a4.validate_a4_translation_state(binding.state)
    a4.validate_a4_gene_cache(binding.cache)
    mutation_rate, scale, flags = _supported_config(config)
    _require_binding_scope_flags(binding.state, flags)
    dt = _strict_dt(dt)
    result = _empty_numpy_plan(binding)
    ragged = binding.ragged
    state = binding.state
    N = int(state.cell_count)
    specs_by_cell = binding.cache.materialize_gene_specs_host()
    total_appended = 0
    total_template_storage = 0
    total_start_events = 0
    for ci in range(N):
        (
            template, partial, start_event, selected_index, template_lesion,
        ) = _numpy_template_copy(
            binding, ci, allow_template_start=allow_template_start,
        )
        if (int(state.genome_count[ci]) <= 0 or template is None
                or len(template) == 0 or len(partial) >= len(template)):
            result.scope_error_code[ci] = SCOPE_INACTIVE_TEMPLATE
            continue
        pools = result.pools_after[ci]
        if pools[a4.a3.POOL_ATP] < 0.0:
            result.scope_error_code[ci] = SCOPE_NEGATIVE_ATP
            continue
        specs = specs_by_cell[ci]
        replicase = _numpy_replicase(binding, ci, specs)
        if flags['external_replicase']:
            replicase += 0.85
        if _numpy_fp64_comparison_boundary(replicase, 1e-6):
            result.scope_error_code[ci] = SCOPE_FP64_DISCRETE_BOUNDARY
            continue
        if replicase <= 1e-6:
            result.scope_error_code[ci] = SCOPE_REPLICASE_GATE
            continue
        metrics = a4._numpy_translation_cell_metrics(state, ci)
        proofreading = (
            _numpy_raw_repair(
                binding, ci, specs, a4.a3.REPAIR_PROOFREADING,
                metrics['aggregate'],
            )
            if flags['proofreading'] else 0.0
        )
        proof_fraction = proofreading / (0.75 + proofreading)
        if flags['quiescence']:
            signal = _numpy_raw_repair(
                binding, ci, specs, a4.a3.REPAIR_QUIESCENCE,
                metrics['aggregate'],
            )
            need = (
                max(0.0, float(metrics['burden']) - 0.12)
                + 0.35 * float(state.current_stress[ci])
            )
            quiescence = float(np.clip(
                (signal / (0.8 + signal)) * need * 1.45,
                0.0, 0.82,
            ))
        else:
            quiescence = 0.0
        if flags['quiescence_effector']:
            quiescence = float(np.clip(
                max(quiescence, float(state.behavioural_quiescence[ci])),
                0.0, 0.86,
            ))
        sat_nucleotide = pools[a4.a3.POOL_NUCLEOTIDE] / (
            0.055 + pools[a4.a3.POOL_NUCLEOTIDE]
        )
        sat_atp = pools[a4.a3.POOL_ATP] / (0.10 + pools[a4.a3.POOL_ATP])
        speed = 10.0 * replicase * sat_nucleotide * sat_atp
        speed *= (
            (1.0 - 0.34 * proof_fraction)
            * (1.0 - 0.78 * quiescence)
        )
        # Formal066 first forms the scaled dt argument, then the inherited
        # 0.4 method multiplies speed by that one fp64 value.
        increment = speed * (dt * scale)
        fractional = (
            float(ragged.replication_fractional[ci]) + increment
        )
        if _numpy_fp64_integer_boundary(fractional, increment):
            result.scope_error_code[ci] = SCOPE_FP64_DISCRETE_BOUNDARY
            continue
        requested = int(fractional)
        result.requested_symbols[ci] = requested
        result.replication_fractional_after[ci] = fractional - requested
        volume = max(
            0.20,
            (float(state.radius[ci]) / float(a4.s5.BASE_RADIUS)) ** 2,
        )
        reactive = float(pools[a4.a3.POOL_REACTIVE] / volume)
        raw_error = max(
            0.0,
            mutation_rate
            + 0.0012 * float(template_lesion)
            + 0.0010 * reactive,
        )
        result.last_effective_error_rate[ci] = (
            raw_error * (1.0 - 0.82 * proof_fraction)
        )
        extra_atp = 0.00075 * proof_fraction
        initial_pools = np.asarray(state.pools[ci], dtype=np.float64).copy()
        initial_cumulative = float(state.cumulative_proofreading_atp[ci])
        copied = 0
        comparison_boundary = False
        for _ in range(requested):
            index = len(partial) + copied
            if index >= len(template):
                break
            atp_per_symbol = (
                float(a4.g2.REPLICATION_ATP_PER_SYMBOL) + extra_atp
            )
            if pools[a4.a3.POOL_NUCLEOTIDE] < float(a4.g2.MONOMER_MASS):
                break
            atp_gate = atp_per_symbol + 0.022
            if (flags['proofreading']
                    and _numpy_fp64_comparison_boundary(
                        pools[a4.a3.POOL_ATP], atp_gate,
                    )):
                comparison_boundary = True
                break
            if pools[a4.a3.POOL_ATP] < atp_gate:
                break
            result.append_symbols[ci, copied] = int(template[index])
            pools[a4.a3.POOL_NUCLEOTIDE] -= float(a4.g2.MONOMER_MASS)
            pools[a4.a3.POOL_ATP] -= atp_per_symbol
            result.cumulative_proofreading_atp_after[ci] += extra_atp
            copied += 1
        if comparison_boundary:
            result.scope_error_code[ci] = SCOPE_FP64_DISCRETE_BOUNDARY
            result.requested_symbols[ci] = 0
            result.replication_fractional_after[ci] = 0.0
            result.last_effective_error_rate[ci] = 0.0
            result.append_symbols[ci, :] = 0
            pools[:] = initial_pools
            result.cumulative_proofreading_atp_after[ci] = initial_cumulative
            continue
        result.append_count[ci] = copied
        result.last_replication_symbols[ci] = copied
        total_appended += copied
        if len(partial) + copied >= len(template):
            result.scope_error_code[ci] = SCOPE_COMPLETION
        else:
            result.scope_valid[ci] = True
            if start_event:
                result.template_start_events[ci] = True
                result.selected_template_indices[ci] = selected_index
                result.template_storage_symbols[ci] = len(template)
                total_template_storage += len(template)
                total_start_events += 1
    if (int(ragged.sequence_count) + 2 * total_start_events
            > int(ragged.sequence_capacity)):
        raise a4.A4CapacityError(
            'template start exceeds sequence capacity'
        )
    if (int(ragged.symbol_count) + total_template_storage + total_appended
            > int(ragged.symbol_capacity)):
        raise a4.A4CapacityError(
            'template start/elongation exceeds symbol capacity'
        )
    return validate_a4_paid_elongation_plan(result)


def paid_replication_elongation_numpy(binding, dt, config):
    """Public deterministic A4.4b path; inactive start stays CPU authority."""
    return _paid_replication_elongation_numpy(
        binding, dt, config, allow_template_start=False,
    )


def _torch_replicase(
        binding, valid_entries, fingerprints, role, parameter, localisation,
        promoter, efficiency):
    """Return ordered replication and repair signals plus frozen metrics."""
    state = binding.state
    C = int(state.cell_capacity)
    P = int(state.protein_capacity)
    device = state.pools.device
    position = torch.arange(P, dtype=torch.int64, device=device)
    active_valid = position[None, :] < state.active_count[:, None]
    match = (
        valid_entries[:, :, None] & active_valid[:, None, :]
        & (fingerprints[:, :, None] == state.active_fingerprints[:, None, :])
    )
    zero_contribution = torch.zeros_like(
        state.active_mass[:, None, :] * efficiency[:, :, None]
    )
    replicase_contribution = torch.sum(torch.where(
        match & (role[:, :, None] == int(a4.a3.ROLE_REPLICASE)),
        state.active_mass[:, None, :] * efficiency[:, :, None],
        zero_contribution,
    ), dim=1)
    repair_contribution = (
        state.active_mass[:, None, :] * efficiency[:, :, None]
    ) * promoter[:, :, None]
    proof_contribution = torch.sum(torch.where(
        match
        & (role[:, :, None] == int(a4.a3.ROLE_REGULATOR))
        & (localisation[:, :, None] == int(a4.s4.LOC_REPAIR))
        & (torch.remainder(
            parameter[:, :, None], int(a4.a3.REPAIR_COUNT),
        ) == int(a4.a3.REPAIR_PROOFREADING)),
        repair_contribution, zero_contribution,
    ), dim=1)
    quiescence_contribution = torch.sum(torch.where(
        match
        & (role[:, :, None] == int(a4.a3.ROLE_REGULATOR))
        & (localisation[:, :, None] == int(a4.s4.LOC_REPAIR))
        & (torch.remainder(
            parameter[:, :, None], int(a4.a3.REPAIR_COUNT),
        ) == int(a4.a3.REPAIR_QUIESCENCE)),
        repair_contribution, zero_contribution,
    ), dim=1)
    total = a4._torch_ordered_row_sum(replicase_contribution) / 0.040
    pools = state.pools
    volume = torch.clamp(
        (state.radius / float(a4.s5.BASE_RADIUS)) ** 2, min=0.20,
    )
    aggregate = pools[:, a4.a3.POOL_AGGREGATE] / volume
    active = torch.clamp(pools[:, a4.a3.POOL_CATALYST], min=0.0)
    damaged = pools[:, a4.a3.POOL_DAMAGED_PROTEIN]
    aggregate_pool = pools[:, a4.a3.POOL_AGGREGATE]
    functional = active / torch.clamp(
        active + damaged + aggregate_pool, min=1e-9,
    )
    toxicity = 1.0 / (1.0 + 3.6 * aggregate)
    proteostasis = torch.clamp(functional * toxicity, min=0.02, max=1.0)
    genome_factor = 1.0 / (1.0 + 0.85 * state.genome_lesion_mean)
    inhibition = 1.0 / (1.0 + 2.6 * aggregate)
    proof_signal = (
        a4._torch_ordered_row_sum(proof_contribution) / 0.014
    ) * inhibition
    quiescence_signal = (
        a4._torch_ordered_row_sum(quiescence_contribution) / 0.014
    ) * inhibition
    reactive = pools[:, a4.a3.POOL_REACTIVE] / volume
    protein_total = torch.clamp(
        active + damaged + aggregate_pool, min=0.08,
    )
    protein_damage = (
        damaged + 1.8 * aggregate_pool
    ) / protein_total
    membrane_weights = torch.clamp(state.membrane, min=1e-9)
    membrane_weighted_damage = (
        torch.clamp(state.membrane_oxidation, min=0.0, max=2.5)
        * membrane_weights
    )
    membrane_damage = _torch_numpy_pairwise_sum_rows(
        membrane_weighted_damage,
    ) / _torch_numpy_pairwise_sum_rows(membrane_weights)
    burden = torch.clamp(
        0.36 * protein_damage + 0.24 * membrane_damage
        + 0.20 * reactive + 0.20 * state.genome_lesion_mean,
        min=0.0, max=4.0,
    )
    return (
        total * proteostasis * genome_factor,
        volume, aggregate, burden, proof_signal, quiescence_signal,
    )


def _paid_replication_elongation_torch(
        binding, dt, config, allow_template_start=False):
    """Fixed-shape resident plan with no scalar readback or dynamic output."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('Torch elongation requires a Torch binding')
    a4._validate_resident_ragged_metadata(binding.ragged)
    a4._validate_translation_resident_metadata(binding.state)
    a4._validate_gene_backend_and_dtypes(binding.cache)
    mutation_rate, scale, flags = _supported_config(config)
    _require_binding_scope_flags(binding.state, flags)
    dt = _strict_dt(dt)
    ragged = binding.ragged
    state = binding.state
    cache = binding.cache
    C = int(state.cell_capacity)
    K = int(cache.entry_capacity)
    W = int(ragged.max_sequence_symbols)
    device = state.pools.device
    dtype = state.pools.dtype

    rank = torch.arange(K, dtype=torch.int64, device=device)
    if K:
        first = torch.clamp(cache.cell_entry_offsets[:C], min=0)
        last = torch.clamp(cache.cell_entry_offsets[1:C + 1], min=0)
        counts = torch.clamp(last - first, min=0, max=K)
        indices = first[:, None] + rank[None, :]
        safe_indices = torch.clamp(indices, min=0, max=K - 1)
        valid_entries = (
            state.cell_mask[:, None] & (rank[None, :] < counts[:, None])
            & cache.entry_mask[safe_indices]
        )
        payload = cache.payloads[safe_indices]
        fingerprints = cache.fingerprints[safe_indices]
        role = torch.remainder(payload[:, :, 0].to(torch.int64), 8)
        parameter = torch.remainder(payload[:, :, 1].to(torch.int64), 8)
        promoter = 0.18 + 1.22 * (payload[:, :, 3].to(dtype) / 7.0)
        efficiency = 0.52 + 0.96 * (payload[:, :, 4].to(dtype) / 7.0)
        localisation = torch.remainder(payload[:, :, 6].to(torch.int64), 4)
    else:
        valid_entries = torch.zeros((C, 0), dtype=torch.bool, device=device)
        fingerprints = torch.zeros((C, 0), dtype=torch.int64, device=device)
        role = torch.zeros((C, 0), dtype=torch.int64, device=device)
        parameter = torch.zeros((C, 0), dtype=torch.int64, device=device)
        promoter = torch.zeros((C, 0), dtype=dtype, device=device)
        efficiency = torch.zeros((C, 0), dtype=dtype, device=device)
        localisation = torch.zeros((C, 0), dtype=torch.int64, device=device)
    (
        replicase, volume, aggregate, burden,
        proofreading_signal, quiescence_signal,
    ) = _torch_replicase(
        binding, valid_entries, fingerprints, role, parameter, localisation,
        promoter, efficiency,
    )
    if flags['external_replicase']:
        replicase = replicase + 0.85
    if flags['proofreading']:
        proof_fraction = proofreading_signal / (0.75 + proofreading_signal)
    else:
        proof_fraction = torch.zeros((C,), dtype=dtype, device=device)
    if flags['quiescence']:
        q_need = (
            torch.clamp(burden - 0.12, min=0.0)
            + 0.35 * state.current_stress
        )
        quiescence = torch.clamp(
            (quiescence_signal / (0.8 + quiescence_signal))
            * q_need * 1.45,
            min=0.0, max=0.82,
        )
    else:
        quiescence = torch.zeros((C,), dtype=dtype, device=device)
    if flags['quiescence_effector']:
        quiescence = torch.clamp(torch.maximum(
            quiescence, state.behavioural_quiescence,
        ), min=0.0, max=0.86)

    sequence_end = ragged.cell_sequence_offsets[1:C + 1]
    active_template_index = torch.clamp(
        sequence_end - 2, min=0, max=int(ragged.sequence_capacity) - 1,
    )
    first_sequence = torch.clamp(
        ragged.cell_sequence_offsets[:C],
        min=0, max=int(ragged.sequence_capacity) - 1,
    )
    start_candidate = (
        state.cell_mask & (~ragged.replication_active)
        & (state.genome_count == 1) & bool(allow_template_start)
    )
    template_index = torch.where(
        start_candidate, first_sequence, active_template_index,
    )
    copy_index = torch.clamp(
        sequence_end - 1, min=0, max=int(ragged.sequence_capacity) - 1,
    )
    template_start = ragged.sequence_offsets[template_index]
    template_end = ragged.sequence_offsets[template_index + 1]
    copy_start = ragged.sequence_offsets[copy_index]
    copy_end = ragged.sequence_offsets[copy_index + 1]
    template_length = torch.clamp(template_end - template_start, min=0)
    copy_length = torch.where(
        start_candidate, torch.zeros_like(copy_end),
        torch.clamp(copy_end - copy_start, min=0),
    )
    existing_structural = (
        state.cell_mask & ragged.replication_active
        & (state.genome_count > 0)
        & (template_length > 0) & (copy_length < template_length)
    )
    structural = existing_structural | (
        start_candidate & (template_length > 0)
    )
    safe_lesion_index = torch.clamp(
        ragged.lesion_offsets[:C],
        min=0, max=int(ragged.sequence_capacity) - 1,
    )
    has_selected_lesion = (
        ragged.lesion_offsets[1:C + 1] > ragged.lesion_offsets[:C]
    )
    selected_lesion = torch.where(
        has_selected_lesion,
        ragged.genome_lesions[safe_lesion_index],
        torch.zeros_like(ragged.replication_template_lesions),
    )
    template_lesion = torch.where(
        start_candidate, selected_lesion,
        ragged.replication_template_lesions,
    )
    replicase_boundary = _torch_fp64_comparison_boundary(replicase, 1e-6)
    replicase_gate = (replicase > 1e-6) & (~replicase_boundary)
    pools_after = state.pools.clone()
    atp_value = pools_after[:, a4.a3.POOL_ATP]
    atp_supported = atp_value >= 0.0
    rate_atp = torch.where(
        atp_supported, atp_value, torch.zeros_like(atp_value),
    )
    kinetics_supported = structural & replicase_gate & atp_supported
    sat_nucleotide = pools_after[:, a4.a3.POOL_NUCLEOTIDE] / (
        0.055 + pools_after[:, a4.a3.POOL_NUCLEOTIDE]
    )
    sat_atp = rate_atp / (0.10 + rate_atp)
    speed = 10.0 * replicase * sat_nucleotide * sat_atp
    speed = speed * (
        (1.0 - 0.34 * proof_fraction)
        * (1.0 - 0.78 * quiescence)
    )
    increment = speed * float(dt * scale)
    fractional_input = torch.where(
        start_candidate, torch.zeros_like(ragged.replication_fractional),
        ragged.replication_fractional,
    )
    fractional_total = fractional_input + increment
    fp64_integer_boundary = (
        kinetics_supported
        & _torch_fp64_integer_boundary(fractional_total, increment)
    )
    unambiguous_kinetics = kinetics_supported & (~fp64_integer_boundary)
    safe_fractional_total = torch.where(
        unambiguous_kinetics, fractional_total,
        torch.zeros_like(fractional_total),
    )
    requested = torch.trunc(safe_fractional_total).to(torch.int64)
    fractional_after = safe_fractional_total - requested.to(dtype)
    reactive = pools_after[:, a4.a3.POOL_REACTIVE] / volume
    raw_error = torch.clamp(
        float(mutation_rate)
        + 0.0012 * template_lesion
        + 0.0010 * reactive,
        min=0.0,
    )
    effective_error = raw_error * (1.0 - 0.82 * proof_fraction)
    extra_atp = 0.00075 * proof_fraction

    symbol_rank = torch.arange(W, dtype=torch.int64, device=device)
    symbol_indices = template_start[:, None] + copy_length[:, None] + symbol_rank[None, :]
    safe_symbols = torch.clamp(
        symbol_indices, min=0, max=int(ragged.symbol_capacity) - 1,
    )
    candidates = ragged.symbols[safe_symbols]
    append_symbols = torch.zeros((C, W), dtype=torch.uint8, device=device)
    append_count = torch.zeros((C,), dtype=torch.int64, device=device)
    payment_boundary = torch.zeros((C,), dtype=torch.bool, device=device)
    cumulative_proofreading_atp_after = torch.where(
        state.cell_mask, state.cumulative_proofreading_atp,
        torch.zeros_like(state.cumulative_proofreading_atp),
    ).clone()
    running = unambiguous_kinetics
    for position in range(W):
        atp_per_symbol = (
            float(a4.g2.REPLICATION_ATP_PER_SYMBOL) + extra_atp
        )
        attempted = (
            running & (position < requested)
            & (copy_length + position < template_length)
        )
        nucleotide_supported = (
            pools_after[:, a4.a3.POOL_NUCLEOTIDE]
            >= float(a4.g2.MONOMER_MASS)
        )
        atp_gate = atp_per_symbol + 0.022
        if flags['proofreading']:
            atp_boundary = (
                attempted & nucleotide_supported
                & _torch_fp64_comparison_boundary(
                    pools_after[:, a4.a3.POOL_ATP], atp_gate,
                )
            )
        else:
            atp_boundary = torch.zeros_like(attempted)
        payment_boundary = payment_boundary | atp_boundary
        accepted = (
            attempted & nucleotide_supported & (~atp_boundary)
            & (pools_after[:, a4.a3.POOL_ATP] >= atp_gate)
        )
        append_symbols[:, position] = torch.where(
            accepted, candidates[:, position], append_symbols[:, position],
        )
        pools_after[:, a4.a3.POOL_NUCLEOTIDE] = (
            pools_after[:, a4.a3.POOL_NUCLEOTIDE]
            - accepted.to(dtype) * float(a4.g2.MONOMER_MASS)
        )
        pools_after[:, a4.a3.POOL_ATP] = (
            pools_after[:, a4.a3.POOL_ATP]
            - accepted.to(dtype) * atp_per_symbol
        )
        cumulative_proofreading_atp_after = torch.where(
            accepted,
            cumulative_proofreading_atp_after + extra_atp,
            cumulative_proofreading_atp_after,
        )
        append_count = append_count + accepted.to(torch.int64)
        running = accepted

    # A late ATP comparison ambiguity invalidates the whole pure event rather
    # than exposing a partially paid prefix as commit authority.
    append_symbols = torch.where(
        payment_boundary[:, None], torch.zeros_like(append_symbols),
        append_symbols,
    )
    append_count = torch.where(
        payment_boundary, torch.zeros_like(append_count), append_count,
    )
    pools_after = torch.where(
        payment_boundary[:, None], state.pools, pools_after,
    )
    cumulative_proofreading_atp_after = torch.where(
        payment_boundary, state.cumulative_proofreading_atp,
        cumulative_proofreading_atp_after,
    )
    completed_kinetics = unambiguous_kinetics & (~payment_boundary)

    error = torch.zeros((C,), dtype=torch.int64, device=device)
    error = torch.where(
        state.cell_mask & (~structural),
        torch.full_like(error, SCOPE_INACTIVE_TEMPLATE), error,
    )
    error = torch.where(
        state.cell_mask & structural & (~atp_supported),
        torch.full_like(error, SCOPE_NEGATIVE_ATP), error,
    )
    error = torch.where(
        state.cell_mask & structural & atp_supported
        & (~replicase_boundary) & (~replicase_gate),
        torch.full_like(error, SCOPE_REPLICASE_GATE), error,
    )
    error = torch.where(
        (state.cell_mask & structural & atp_supported & replicase_boundary)
        | fp64_integer_boundary | payment_boundary,
        torch.full_like(error, SCOPE_FP64_DISCRETE_BOUNDARY), error,
    )
    completion = completed_kinetics & (
        copy_length + append_count >= template_length
    )
    error = torch.where(
        completion, torch.full_like(error, SCOPE_COMPLETION), error,
    )
    start_before_capacity = start_candidate & (error == SCOPE_OK)
    sequence_capacity_overflow = (
        int(ragged.sequence_count)
        + 2 * torch.sum(start_before_capacity.to(torch.int64))
        > int(ragged.sequence_capacity)
    )
    symbol_capacity_overflow = (
        int(ragged.symbol_count) + torch.sum(append_count)
        + torch.sum(torch.where(
            start_before_capacity, template_length,
            torch.zeros_like(template_length),
        )) > int(ragged.symbol_capacity)
    )
    capacity_overflow = sequence_capacity_overflow | symbol_capacity_overflow
    error = torch.where(
        state.cell_mask & capacity_overflow,
        torch.full_like(error, SCOPE_CAPACITY), error,
    )
    scope_valid = state.cell_mask & (error == SCOPE_OK)
    template_start_events = start_candidate & scope_valid

    plan = A4PaidElongationPlan(
        schema_version=SCHEMA_VERSION,
        cell_capacity=C,
        append_capacity=W,
        cell_count=int(state.cell_count),
        source_provenance=str(state.source_provenance),
        cell_ids=state.cell_ids.clone(),
        cell_mask=state.cell_mask.clone(),
        scope_valid=scope_valid,
        scope_error_code=error,
        requested_symbols=torch.where(
            completed_kinetics, requested, torch.zeros_like(requested),
        ),
        append_symbols=append_symbols,
        append_count=append_count,
        pools_after=torch.where(
            state.cell_mask[:, None], pools_after, torch.zeros_like(pools_after),
        ),
        replication_fractional_after=torch.where(
            completed_kinetics, fractional_after,
            torch.zeros_like(fractional_after),
        ),
        last_replication_symbols=append_count.clone(),
        last_effective_error_rate=torch.where(
            completed_kinetics,
            effective_error,
            torch.zeros_like(effective_error),
        ),
        cumulative_proofreading_atp_after=cumulative_proofreading_atp_after,
        substitution_events=torch.zeros_like(append_count),
        template_start_events=template_start_events,
        selected_template_indices=torch.where(
            template_start_events, torch.zeros_like(append_count),
            torch.full_like(append_count, -1),
        ),
        template_storage_symbols=torch.where(
            template_start_events, template_length,
            torch.zeros_like(template_length),
        ),
    )
    _validate_plan_metadata(plan)
    return plan


def paid_replication_elongation_torch(binding, dt, config):
    """Public deterministic A4.4b path; inactive start stays CPU authority."""
    return _paid_replication_elongation_torch(
        binding, dt, config, allow_template_start=False,
    )


def _require_rng_tape_binding(tape, binding, dt, config_sha256):
    _require_rng_tape(tape)
    dt = _strict_dt(dt)
    if (tape.source_provenance != str(binding.state.source_provenance)
            or tape.dt_hex != float(dt).hex()
            or tape.config_sha256 != str(config_sha256)
            or int(tape.cell_capacity) != int(binding.state.cell_capacity)
            or int(tape.append_capacity)
            != int(binding.ragged.max_sequence_symbols)
            or int(tape.cell_count) != int(binding.state.cell_count)):
        raise A4ReplicationScopeError(
            'substitution RNG tape does not bind this source/config/dt'
        )
    return tape


def prepare_substitution_rng_tape(
        binding, dt, config, rng_state_before):
    """Replay Formal066 scalar PCG64 calls on a clone, never the live RNG."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('RNG tape preparation requires a NumPy binding')
    deterministic, config_sha256 = _substitution_config(config)
    dt = _strict_dt(dt)
    plan = _paid_replication_elongation_numpy(
        binding, dt, deterministic, allow_template_start=True,
    )
    before = _canonical_pcg64_state(rng_state_before, 'rng_state_before')
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    C = int(plan.cell_capacity)
    W = int(plan.append_capacity)
    N = int(plan.cell_count)
    draw_count = np.asarray(plan.append_count, dtype=np.int64).copy()
    effective_error = np.asarray(
        plan.last_effective_error_rate, dtype=np.float64,
    ).copy()
    draw_mask = np.arange(W, dtype=np.int64)[None, :] < draw_count[:, None]
    uniform_draws = np.zeros((C, W), dtype=np.float64)
    replacement_raw = np.zeros((C, W), dtype=np.uint8)
    replacement_mask = np.zeros((C, W), dtype=bool)
    template_start_mask = np.asarray(
        plan.template_start_events, dtype=bool,
    ).copy()
    template_selection_indices = np.asarray(
        plan.selected_template_indices, dtype=np.int64,
    ).copy()
    for ci in range(N):
        # Frozen Formal066 performs the scalar high-level selection call at
        # this exact point, before this cell's threshold/integer draws.  The
        # present CPU/NumPy PCG64 returns index zero without advancing its
        # state, but replay the call rather than encoding that implementation
        # detail as an RNG rule.
        if bool(template_start_mask[ci]):
            selected = int(generator.integers(0, 1))
            if selected != int(template_selection_indices[ci]):
                raise A4ReplicationScopeError(
                    'template selection replay differs from its plan'
                )
        error = float(effective_error[ci])
        for rank in range(int(draw_count[ci])):
            uniform = float(generator.random())
            # The frozen direct CPU and the independent NumPy plan can differ
            # by a few ulps in effective_error.  Do not discretise that tiny
            # difference into a different mutation decision.
            if _numpy_fp64_comparison_boundary(uniform, error):
                raise A4ReplicationScopeError(
                    'substitution RNG comparison is fp64-ambiguous'
                )
            uniform_draws[ci, rank] = uniform
            if uniform < error:
                replacement_mask[ci, rank] = True
                replacement_raw[ci, rank] = int(generator.integers(0, 7))
    substitution_count = np.sum(
        replacement_mask, axis=1, dtype=np.int64,
    )
    values = {
        'schema_version': RNG_TAPE_SCHEMA_VERSION,
        'cell_capacity': C,
        'append_capacity': W,
        'cell_count': N,
        'source_provenance': str(plan.source_provenance),
        'dt_hex': float(dt).hex(),
        'config_sha256': str(config_sha256),
        'schedule_sha256': '0' * 64,
        'rng_before_state': copy.deepcopy(before),
        'rng_after_state': copy.deepcopy(generator.bit_generator.state),
        'cell_ids': np.asarray(plan.cell_ids, dtype=np.int64).copy(),
        'cell_mask': np.asarray(plan.cell_mask, dtype=bool).copy(),
        'draw_mask': draw_mask,
        'uniform_draws': uniform_draws,
        'replacement_raw': replacement_raw,
        'replacement_mask': replacement_mask,
        'draw_count': draw_count,
        'substitution_count': substitution_count,
        'effective_error': effective_error,
        'template_start_mask': template_start_mask,
        'template_selection_indices': template_selection_indices,
    }
    draft = _make_rng_tape(**values)
    values['schedule_sha256'] = _rng_schedule_digest(draft)
    tape = _make_rng_tape(**values)
    return validate_a4_substitution_rng_tape(tape)


def _numpy_apply_substitution_tape(plan, tape):
    if (not np.array_equal(plan.cell_ids, tape.cell_ids)
            or not np.array_equal(plan.cell_mask, tape.cell_mask)
            or not np.array_equal(plan.append_count, tape.draw_count)
            or not np.array_equal(
                plan.template_start_events, tape.template_start_mask,
            )
            or not np.array_equal(
                plan.selected_template_indices,
                tape.template_selection_indices,
            )
            or not np.array_equal(
                np.asarray(plan.last_effective_error_rate).view(np.uint64),
                np.asarray(tape.effective_error).view(np.uint64),
            )):
        raise A4ReplicationScopeError(
            'substitution RNG tape schedule differs from elongation plan'
        )
    result = plan.clone()
    N = int(result.cell_count)
    for ci in range(N):
        for rank in range(int(result.append_count[ci])):
            if not bool(tape.replacement_mask[ci, rank]):
                continue
            old = int(result.append_symbols[ci, rank])
            new = int(tape.replacement_raw[ci, rank])
            if new >= old:
                new += 1
            result.append_symbols[ci, rank] = new % int(a4.g2.ALPHABET_SIZE)
    result.substitution_events = np.asarray(
        tape.substitution_count, dtype=np.int64,
    ).copy()
    return validate_a4_paid_elongation_plan(result)


def paid_replication_substitution_numpy(binding, dt, config, tape):
    """Apply a validated CPU-resolved substitution tape to a NumPy plan."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('NumPy substitution requires a NumPy binding')
    deterministic, config_sha256 = _substitution_config(config)
    tape = validate_a4_substitution_rng_tape(tape)
    _require_rng_tape_binding(tape, binding, dt, config_sha256)
    plan = _paid_replication_elongation_numpy(
        binding, dt, deterministic, allow_template_start=True,
    )
    return _numpy_apply_substitution_tape(plan, tape)


def paid_replication_substitution_torch(binding, dt, config, tape):
    """Fixed-shape resident application of an attested PCG64 event tape."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('Torch substitution requires a Torch binding')
    deterministic, config_sha256 = _substitution_config(config)
    _require_rng_tape_binding(tape, binding, dt, config_sha256)
    if (not _is_tensor(tape.uniform_draws)
            or tape.uniform_draws.device != binding.state.pools.device):
        raise a4.A4SchemaError('RNG tape must share the resident device')
    base = _paid_replication_elongation_torch(
        binding, dt, deterministic, allow_template_start=True,
    )
    result = base.clone()
    identity_ok = (
        (tape.cell_ids == base.cell_ids)
        & (tape.cell_mask == base.cell_mask)
        & (tape.draw_count == base.append_count)
        & (tape.template_start_mask == base.template_start_events)
        & (
            tape.template_selection_indices
            == base.selected_template_indices
        )
    )
    effective_error_ok = (
        tape.effective_error == base.last_effective_error_rate
    )
    used_draw = tape.draw_mask & base.cell_mask[:, None]
    device_error = base.last_effective_error_rate[:, None]
    ambiguous_draw = used_draw & _torch_fp64_comparison_boundary(
        tape.uniform_draws, device_error,
    )
    decision_mismatch = used_draw & (
        (tape.uniform_draws < device_error) != tape.replacement_mask
    )
    row_boundary = torch.any(ambiguous_draw, dim=1) & base.scope_valid
    row_mismatch = base.cell_mask & (
        ~identity_ok
        | ~effective_error_ok
        | torch.any(decision_mismatch, dim=1)
    )
    # The PCG64 after-state is one ordered stream for the whole batch.  Any
    # row disagreement invalidates the complete event tape, never a suffix.
    batch_boundary = base.cell_mask & torch.any(row_boundary)
    batch_mismatch = (
        base.cell_mask & torch.any(row_mismatch) & ~torch.any(row_boundary)
    )
    failure = batch_boundary | batch_mismatch
    mapped = tape.replacement_raw.to(torch.int64)
    old = base.append_symbols.to(torch.int64)
    mapped = torch.remainder(mapped + (mapped >= old).to(torch.int64), 8)
    substituted = torch.where(
        tape.replacement_mask, mapped.to(torch.uint8), base.append_symbols,
    )
    good = base.scope_valid & ~failure
    result.append_symbols = torch.where(
        good[:, None], substituted, torch.zeros_like(substituted),
    )
    result.substitution_events = torch.where(
        good, tape.substitution_count,
        torch.zeros_like(tape.substitution_count),
    )
    result.scope_error_code = torch.where(
        batch_mismatch & base.scope_valid,
        torch.full_like(base.scope_error_code, SCOPE_RNG_TAPE_MISMATCH),
        base.scope_error_code,
    )
    result.scope_error_code = torch.where(
        batch_boundary & base.scope_valid,
        torch.full_like(
            base.scope_error_code, SCOPE_FP64_DISCRETE_BOUNDARY,
        ),
        result.scope_error_code,
    )
    result.scope_valid = good
    rollback = failure & base.scope_valid
    result.requested_symbols = torch.where(
        rollback, torch.zeros_like(base.requested_symbols),
        base.requested_symbols,
    )
    result.append_count = torch.where(
        rollback, torch.zeros_like(base.append_count), base.append_count,
    )
    result.pools_after = torch.where(
        rollback[:, None], binding.state.pools, base.pools_after,
    )
    result.replication_fractional_after = torch.where(
        rollback, torch.zeros_like(base.replication_fractional_after),
        base.replication_fractional_after,
    )
    result.last_replication_symbols = torch.where(
        rollback, torch.zeros_like(base.last_replication_symbols),
        base.last_replication_symbols,
    )
    result.last_effective_error_rate = torch.where(
        rollback, torch.zeros_like(base.last_effective_error_rate),
        base.last_effective_error_rate,
    )
    result.cumulative_proofreading_atp_after = torch.where(
        rollback, binding.state.cumulative_proofreading_atp,
        base.cumulative_proofreading_atp_after,
    )
    result.template_start_events = torch.where(
        rollback, torch.zeros_like(base.template_start_events),
        base.template_start_events,
    )
    result.selected_template_indices = torch.where(
        rollback, torch.full_like(base.selected_template_indices, -1),
        base.selected_template_indices,
    )
    result.template_storage_symbols = torch.where(
        rollback, torch.zeros_like(base.template_storage_symbols),
        base.template_storage_symbols,
    )
    _validate_plan_metadata(result)
    return result


def paid_replication_substitution_plan(binding, dt, config, tape):
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return paid_replication_substitution_torch(
            binding, dt, config, tape,
        )
    return paid_replication_substitution_numpy(binding, dt, config, tape)


def paid_replication_elongation_plan(binding, dt, config):
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return paid_replication_elongation_torch(binding, dt, config)
    return paid_replication_elongation_numpy(binding, dt, config)


PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.5b-template-start-active-substitution-rng-tape-deterministic-'
        'proofreading-quiescence-noncompletion-plan-not-integrated-'
        'cpu-authoritative'
    ),
    'material_mutation': 'cpu-authoritative-later-a4-slice',
    'full_gpu_world_step': False,
})
