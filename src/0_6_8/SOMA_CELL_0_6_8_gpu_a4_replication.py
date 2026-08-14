# coding: utf-8
"""A4.4a pure resident paid DNA-elongation plan.

This deliberately narrow development slice handles only a pre-existing active
replication template whose partial copy remains incomplete in this call.  It
does not select a template, proofread, mutate, complete a genome, consume RNG,
or replace the A3 scheduler.  Frozen Formal066 CPU behavior remains authority.
"""
from __future__ import division

import copy
import math
from dataclasses import dataclass, fields

import numpy as np

import SOMA_CELL_0_6_8_gpu_a4 as a4

try:
    import torch
except Exception:  # pragma: no cover - NumPy reference remains importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.4a'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | pre-existing-template paid DNA elongation plan'
SCHEMA_VERSION = '0.6.8-GPU-A4.4a-paid-dna-elongation-plan'
FULL_GPU_WORLD_STEP = False

SCOPE_OK = 0
SCOPE_INACTIVE_TEMPLATE = 1
SCOPE_REPLICASE_GATE = 2
SCOPE_COMPLETION = 3
SCOPE_CAPACITY = 4

_PLAN_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code',
    'requested_symbols', 'append_symbols', 'append_count', 'pools_after',
    'replication_fractional_after', 'last_replication_symbols',
    'last_effective_error_rate',
)
_PLAN_UINT8_FIELDS = ('append_symbols',)
_PLAN_INT64_FIELDS = (
    'cell_ids', 'scope_error_code', 'requested_symbols', 'append_count',
    'last_replication_symbols',
)
_PLAN_BOOL_FIELDS = ('cell_mask', 'scope_valid')
_PLAN_FLOAT64_FIELDS = (
    'pools_after', 'replication_fractional_after',
    'last_effective_error_rate',
)


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
    """Extract exactly the host flags supported by A4.4a."""
    required = {
        'genome_replication': True,
        'mutation': False,
        'proofreading': False,
        'external_replicase': False,
        'quiescence': False,
        'quiescence_effector': False,
    }
    for name, expected in required.items():
        if not hasattr(config, name):
            raise A4ReplicationScopeError('config missing %s' % name)
        actual = _strict_bool(getattr(config, name), 'config.%s' % name)
        if actual is not expected:
            raise A4ReplicationScopeError(
                'A4.4a requires config.%s=%s' % (name, expected)
            )
    mutation_rate = _strict_real_scalar(
        getattr(config, 'mutation_rate', None), 'config.mutation_rate',
    )
    scale = _strict_real_scalar(
        getattr(config, 'eco66_replication_rate_scale', 1.0),
        'config.eco66_replication_rate_scale',
    )
    return mutation_rate, max(0.25, scale)


def _strict_dt(value):
    return _strict_real_scalar(value, 'replication dt', minimum=0.0)


def _require_binding_scope_flags(state):
    # These flags are carried as trusted host metadata in the A4.3 snapshot.
    # Require them to agree with this slice instead of accepting a state packed
    # under a different quiescence policy and a separately supplied config.
    if bool(state.quiescence) or bool(state.quiescence_effector):
        raise A4ReplicationScopeError(
            'A4.4a requires a snapshot packed with both quiescence routes off'
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
            'paid elongation row is outside A4.4a scope: %s' %
            [int(value) for value in error_codes]
        )
    if np.any(raw['scope_valid'][N:]) or np.any(
            raw['scope_error_code'][N:] != SCOPE_OK):
        raise a4.A4SchemaError('unused scope tail is not canonical')
    if (np.any(raw['requested_symbols'][:N] < 0)
            or np.any(raw['append_count'][:N] < 0)
            or np.any(raw['append_count'][:N] > W)
            or np.any(raw['append_count'][:N] > raw['requested_symbols'][:N])):
        raise a4.A4SchemaError('replication counts are invalid')
    if not np.array_equal(
            raw['last_replication_symbols'], raw['append_count']):
        raise a4.A4SchemaError('last replication count differs from append count')
    if (not np.isfinite(raw['pools_after']).all()
            or not np.isfinite(raw['replication_fractional_after']).all()
            or not np.isfinite(raw['last_effective_error_rate']).all()):
        raise a4.A4SchemaError('replication plan contains nonfinite values')
    if (np.any(raw['replication_fractional_after'][:N] < 0.0)
            or np.any(raw['replication_fractional_after'][:N] >= 1.0)
            or np.any(raw['last_effective_error_rate'][:N] < 0.0)):
        raise a4.A4SchemaError('replication telemetry outside range')
    # Only the three inherited A4.3 paid pools may carry its registered
    # sequential fp64 residual.  Replication admits a symbol only when the
    # nucleotide pool is >= the exact monomer constant, so nucleotide remains
    # strictly nonnegative after subtracting that same constant.
    allowed_negative = set(a4.TRANSLATION_PAID_POOL_INDICES)
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
        if name in ('cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code'):
            continue
        tail = raw[name][N:]
        if tail.size and np.any(tail != 0):
            raise a4.A4SchemaError('%s unused cell tail is not zero' % name)
    return plan


def _numpy_replicase(binding, ci):
    state = binding.state
    specs = binding.cache.materialize_gene_specs_host()[ci]
    total = 0.0
    for position in range(int(state.active_count[ci])):
        fingerprint = int(state.active_fingerprints[ci, position])
        spec = specs.get(fingerprint)
        if spec is not None and int(spec['role']) == int(a4.a3.ROLE_REPLICASE):
            total += float(state.active_mass[ci, position]) * float(spec['efficiency'])
    metrics = a4._numpy_translation_cell_metrics(state, ci)
    return float(total / 0.040 * metrics['proteostasis'] * metrics['genome_factor'])


def _numpy_template_copy(binding, ci):
    ragged = binding.ragged
    if not bool(ragged.replication_active[ci]):
        return None, None
    sequence_end = int(ragged.cell_sequence_offsets[ci + 1])
    template_index = sequence_end - 2
    copy_index = sequence_end - 1
    template_start = int(ragged.sequence_offsets[template_index])
    template_end = int(ragged.sequence_offsets[template_index + 1])
    copy_start = int(ragged.sequence_offsets[copy_index])
    copy_end = int(ragged.sequence_offsets[copy_index + 1])
    return (
        np.asarray(ragged.symbols[template_start:template_end], dtype=np.uint8),
        np.asarray(ragged.symbols[copy_start:copy_end], dtype=np.uint8),
    )


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
    )


def paid_replication_elongation_numpy(binding, dt, config):
    """Literal NumPy reference for the bounded Formal066 continuation."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('NumPy elongation requires a NumPy binding')
    a4.validate_a4_ragged(binding.ragged)
    a4.validate_a4_translation_state(binding.state)
    a4.validate_a4_gene_cache(binding.cache)
    mutation_rate, scale = _supported_config(config)
    _require_binding_scope_flags(binding.state)
    dt = _strict_dt(dt)
    result = _empty_numpy_plan(binding)
    ragged = binding.ragged
    state = binding.state
    N = int(state.cell_count)
    total_appended = 0
    for ci in range(N):
        template, partial = _numpy_template_copy(binding, ci)
        if template is None or len(template) == 0 or len(partial) >= len(template):
            result.scope_error_code[ci] = SCOPE_INACTIVE_TEMPLATE
            continue
        replicase = _numpy_replicase(binding, ci)
        if replicase <= 1e-6:
            result.scope_error_code[ci] = SCOPE_REPLICASE_GATE
            continue
        pools = result.pools_after[ci]
        sat_nucleotide = pools[a4.a3.POOL_NUCLEOTIDE] / (
            0.055 + pools[a4.a3.POOL_NUCLEOTIDE]
        )
        sat_atp = pools[a4.a3.POOL_ATP] / (0.10 + pools[a4.a3.POOL_ATP])
        speed = 10.0 * replicase * sat_nucleotide * sat_atp
        # Formal066 first forms the scaled dt argument, then the inherited
        # 0.4 method multiplies speed by that one fp64 value.
        fractional = (
            float(ragged.replication_fractional[ci])
            + speed * (dt * scale)
        )
        requested = int(fractional)
        result.requested_symbols[ci] = requested
        result.replication_fractional_after[ci] = fractional - requested
        volume = max(
            0.20,
            (float(state.radius[ci]) / float(a4.s5.BASE_RADIUS)) ** 2,
        )
        reactive = float(pools[a4.a3.POOL_REACTIVE] / volume)
        result.last_effective_error_rate[ci] = max(
            0.0,
            mutation_rate
            + 0.0012 * float(ragged.replication_template_lesions[ci])
            + 0.0010 * reactive,
        )
        copied = 0
        for _ in range(requested):
            index = len(partial) + copied
            if index >= len(template):
                break
            if (pools[a4.a3.POOL_NUCLEOTIDE] < float(a4.g2.MONOMER_MASS)
                    or pools[a4.a3.POOL_ATP]
                    < float(a4.g2.REPLICATION_ATP_PER_SYMBOL) + 0.022):
                break
            result.append_symbols[ci, copied] = int(template[index])
            pools[a4.a3.POOL_NUCLEOTIDE] -= float(a4.g2.MONOMER_MASS)
            pools[a4.a3.POOL_ATP] -= float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
            copied += 1
        result.append_count[ci] = copied
        result.last_replication_symbols[ci] = copied
        total_appended += copied
        if len(partial) + copied >= len(template):
            result.scope_error_code[ci] = SCOPE_COMPLETION
        else:
            result.scope_valid[ci] = True
    if int(ragged.symbol_count) + total_appended > int(ragged.symbol_capacity):
        raise a4.A4CapacityError('paid elongation exceeds symbol capacity')
    return validate_a4_paid_elongation_plan(result)


def _torch_replicase(binding, valid_entries, fingerprints, role, efficiency):
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
    factor = torch.sum(torch.where(
        match & (role[:, :, None] == int(a4.a3.ROLE_REPLICASE)),
        efficiency[:, :, None], torch.zeros_like(efficiency[:, :, None]),
    ), dim=1)
    total = a4._torch_ordered_row_sum(state.active_mass * factor) / 0.040
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
    proteostasis = torch.clamp(
        functional / (1.0 + 3.6 * aggregate), min=0.02, max=1.0,
    )
    genome_factor = 1.0 / (1.0 + 0.85 * state.genome_lesion_mean)
    return total * proteostasis * genome_factor, volume


def paid_replication_elongation_torch(binding, dt, config):
    """Fixed-shape resident plan with no scalar readback or dynamic output."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not _is_tensor(binding.state.pools):
        raise a4.A4SchemaError('Torch elongation requires a Torch binding')
    a4._validate_resident_ragged_metadata(binding.ragged)
    a4._validate_translation_resident_metadata(binding.state)
    a4._validate_gene_backend_and_dtypes(binding.cache)
    mutation_rate, scale = _supported_config(config)
    _require_binding_scope_flags(binding.state)
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
        efficiency = 0.52 + 0.96 * payload[:, :, 4].to(dtype) / 7.0
    else:
        valid_entries = torch.zeros((C, 0), dtype=torch.bool, device=device)
        fingerprints = torch.zeros((C, 0), dtype=torch.int64, device=device)
        role = torch.zeros((C, 0), dtype=torch.int64, device=device)
        efficiency = torch.zeros((C, 0), dtype=dtype, device=device)
    replicase, volume = _torch_replicase(
        binding, valid_entries, fingerprints, role, efficiency,
    )

    sequence_end = ragged.cell_sequence_offsets[1:C + 1]
    template_index = torch.clamp(
        sequence_end - 2, min=0, max=int(ragged.sequence_capacity) - 1,
    )
    copy_index = torch.clamp(
        sequence_end - 1, min=0, max=int(ragged.sequence_capacity) - 1,
    )
    template_start = ragged.sequence_offsets[template_index]
    template_end = ragged.sequence_offsets[template_index + 1]
    copy_start = ragged.sequence_offsets[copy_index]
    copy_end = ragged.sequence_offsets[copy_index + 1]
    template_length = torch.clamp(template_end - template_start, min=0)
    copy_length = torch.clamp(copy_end - copy_start, min=0)
    structural = (
        state.cell_mask & ragged.replication_active
        & (template_length > 0) & (copy_length < template_length)
    )
    replicase_gate = replicase > 1e-6
    pools_after = state.pools.clone()
    sat_nucleotide = pools_after[:, a4.a3.POOL_NUCLEOTIDE] / (
        0.055 + pools_after[:, a4.a3.POOL_NUCLEOTIDE]
    )
    sat_atp = pools_after[:, a4.a3.POOL_ATP] / (
        0.10 + pools_after[:, a4.a3.POOL_ATP]
    )
    speed = 10.0 * replicase * sat_nucleotide * sat_atp
    fractional_total = ragged.replication_fractional + speed * float(dt * scale)
    requested = torch.floor(fractional_total).to(torch.int64)
    fractional_after = fractional_total - requested.to(dtype)
    reactive = pools_after[:, a4.a3.POOL_REACTIVE] / volume
    effective_error = torch.clamp(
        float(mutation_rate)
        + 0.0012 * ragged.replication_template_lesions
        + 0.0010 * reactive,
        min=0.0,
    )

    symbol_rank = torch.arange(W, dtype=torch.int64, device=device)
    symbol_indices = template_start[:, None] + copy_length[:, None] + symbol_rank[None, :]
    safe_symbols = torch.clamp(
        symbol_indices, min=0, max=int(ragged.symbol_capacity) - 1,
    )
    candidates = ragged.symbols[safe_symbols]
    append_symbols = torch.zeros((C, W), dtype=torch.uint8, device=device)
    append_count = torch.zeros((C,), dtype=torch.int64, device=device)
    running = structural & replicase_gate
    for position in range(W):
        accepted = (
            running & (position < requested)
            & (copy_length + position < template_length)
            & (pools_after[:, a4.a3.POOL_NUCLEOTIDE]
               >= float(a4.g2.MONOMER_MASS))
            & (pools_after[:, a4.a3.POOL_ATP]
               >= float(a4.g2.REPLICATION_ATP_PER_SYMBOL) + 0.022)
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
            - accepted.to(dtype) * float(a4.g2.REPLICATION_ATP_PER_SYMBOL)
        )
        append_count = append_count + accepted.to(torch.int64)
        running = accepted

    error = torch.zeros((C,), dtype=torch.int64, device=device)
    error = torch.where(
        state.cell_mask & (~structural),
        torch.full_like(error, SCOPE_INACTIVE_TEMPLATE), error,
    )
    error = torch.where(
        state.cell_mask & structural & (~replicase_gate),
        torch.full_like(error, SCOPE_REPLICASE_GATE), error,
    )
    completion = structural & replicase_gate & (
        copy_length + append_count >= template_length
    )
    error = torch.where(
        completion, torch.full_like(error, SCOPE_COMPLETION), error,
    )
    capacity_overflow = (
        int(ragged.symbol_count) + torch.sum(append_count)
        > int(ragged.symbol_capacity)
    )
    error = torch.where(
        state.cell_mask & (error == SCOPE_OK) & capacity_overflow,
        torch.full_like(error, SCOPE_CAPACITY), error,
    )
    scope_valid = state.cell_mask & (error == SCOPE_OK)

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
            state.cell_mask, requested, torch.zeros_like(requested),
        ),
        append_symbols=append_symbols,
        append_count=append_count,
        pools_after=torch.where(
            state.cell_mask[:, None], pools_after, torch.zeros_like(pools_after),
        ),
        replication_fractional_after=torch.where(
            state.cell_mask, fractional_after,
            torch.zeros_like(fractional_after),
        ),
        last_replication_symbols=append_count.clone(),
        last_effective_error_rate=torch.where(
            state.cell_mask & structural & replicase_gate, effective_error,
            torch.zeros_like(effective_error),
        ),
    )
    _validate_plan_metadata(plan)
    return plan


def paid_replication_elongation_plan(binding, dt, config):
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return paid_replication_elongation_torch(binding, dt, config)
    return paid_replication_elongation_numpy(binding, dt, config)


PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.4a-active-template-mutation-free-proofreading-free-'
        'noncompletion-paid-plan-not-integrated-cpu-authoritative'
    ),
    'material_mutation': 'cpu-authoritative-later-a4-slice',
    'full_gpu_world_step': False,
})
