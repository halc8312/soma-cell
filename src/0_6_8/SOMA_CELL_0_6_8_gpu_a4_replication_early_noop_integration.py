# coding: utf-8
"""A4.8c8 resident plan and atomic bridge for Formal066 early no-ops.

Only the four ordered returns which change ``last_replication_symbols`` to
zero are represented here.  Empty complete genomes, negative entry ATP,
dead cells, and the guarded replicase comparison boundary fail closed.
Normal replication work remains delegated to the frozen A4.8c1--c7 chain.
"""
from __future__ import division

import copy
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a3_scheduler as a3s
import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_gpu_a4_integration as a48a
import SOMA_CELL_0_6_8_gpu_a4_replication as a44
import SOMA_CELL_0_6_8_gpu_a4_replication_completion_integration as a48c2
import SOMA_CELL_0_6_8_gpu_a4_replication_completion_mutation_integration as a48c3
import SOMA_CELL_0_6_8_gpu_a4_replication_integration as a48c
import SOMA_CELL_0_6_8_gpu_a4_replication_start_integration as a48c4
import SOMA_CELL_0_6_8_gpu_a4_replication_start_mutation_free_integration as a48c5
import SOMA_CELL_0_6_8_gpu_a4_replication_start_completion_integration as a48c6
import SOMA_CELL_0_6_8_gpu_a4_replication_start_completion_mutation_integration as a48c7
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - NumPy oracle remains importable
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c8'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | resident pre-active ordinary early-noop bridge'
SCHEMA_VERSION = '0.6.8-GPU-A4.8c8-early-noop-plan'
INTEGRATION_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c8-pre-active-ordinary-early-noop-atomic-commit'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

SCOPE_OK = 0
SCOPE_NOT_EARLY_NOOP = 1
SCOPE_EMPTY_COMPLETE_GENOME = 2
SCOPE_NEGATIVE_ATP = 3
SCOPE_FP64_COMPARISON_BOUNDARY = 4

BRANCH_NONE = 0
BRANCH_DISABLED = 1
BRANCH_NO_GENOME = 2
BRANCH_REPLICASE_GATE = 3
BRANCH_INACTIVE_MULTI_GENOME = 4

BRANCH_NAMES = {
    BRANCH_DISABLED: 'disabled',
    BRANCH_NO_GENOME: 'no-genome',
    BRANCH_REPLICASE_GATE: 'replicase-gate',
    BRANCH_INACTIVE_MULTI_GENOME: 'inactive-multi-genome',
}

_PLAN_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code',
    'branch_code', 'replicase_activity', 'genome_count',
    'replication_active', 'has_empty_complete_genome', 'negative_atp',
    'last_replication_symbols_after', 'rng_call_count',
)
_PLAN_INT64_FIELDS = (
    'cell_ids', 'scope_error_code', 'branch_code', 'genome_count',
    'last_replication_symbols_after', 'rng_call_count',
)
_PLAN_BOOL_FIELDS = (
    'cell_mask', 'scope_valid', 'replication_active',
    'has_empty_complete_genome', 'negative_atp',
)
_PLAN_FLOAT64_FIELDS = ('replicase_activity',)
_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}


class A4ReplicationEarlyNoopCommitError(a48c7.A4ReplicationStartCompletionMutationCommitError):
    """The bounded A4.8c8 source, resident plan, or commit failed."""


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _strict_bool(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise A4ReplicationEarlyNoopCommitError('%s must be boolean' % label)
    return bool(value)


def _strict_plan_dt(value):
    try:
        return a48c._strict_nonnegative_dt(value)
    except Exception as exc:
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 dt must be finite and nonnegative'
        ) from exc


def _early_noop_config(config):
    flags = {}
    for name in ('genome_replication', 'external_replicase', 'mutation'):
        if not hasattr(config, name):
            raise A4ReplicationEarlyNoopCommitError(
                'config missing %s' % name
            )
        flags[name] = _strict_bool(
            getattr(config, name), 'config.%s' % name,
        )
    digest = a44._sha256_json({
        'genome_replication': flags['genome_replication'],
        'external_replicase': flags['external_replicase'],
        'mutation': flags['mutation'],
    })
    return flags, digest


def _plan_scalar_metadata(plan):
    return (
        str(plan.schema_version), int(plan.cell_capacity),
        int(plan.cell_count), str(plan.source_provenance),
        str(plan.dt_hex), str(plan.config_sha256),
        bool(plan.genome_replication_enabled),
        bool(plan.external_replicase_enabled),
        bool(plan.mutation_enabled),
    )


def _plan_values(plan):
    return {name: getattr(plan, name) for name in _PLAN_ARRAY_FIELDS}


def _register_resident_plan(plan):
    object.__setattr__(plan, '_a48c8_metadata', _plan_scalar_metadata(plan))
    object.__setattr__(plan, '_a48c8_data_ptrs', plan.data_ptrs())
    object.__setattr__(plan, '_a48c8_versions', {
        name: int(getattr(plan, name)._version)
        for name in _PLAN_ARRAY_FIELDS
    })
    return plan


@dataclass
class A4ReplicationEarlyNoopPlan:
    """Fixed-shape pure descriptor; never durable world authority."""

    schema_version: str
    cell_capacity: int
    cell_count: int
    source_provenance: str
    dt_hex: str
    config_sha256: str
    genome_replication_enabled: bool
    external_replicase_enabled: bool
    mutation_enabled: bool
    cell_ids: object
    cell_mask: object
    scope_valid: object
    scope_error_code: object
    branch_code: object
    replicase_activity: object
    genome_count: object
    replication_active: object
    has_empty_complete_genome: object
    negative_atp: object
    last_replication_symbols_after: object
    rng_call_count: object

    def clone(self):
        if _is_tensor(self.cell_ids):
            _validate_resident_plan(self)
        else:
            _validate_host_plan(self)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _PLAN_ARRAY_FIELDS:
                value = value.clone() if _is_tensor(value) else np.asarray(value).copy()
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        result = A4ReplicationEarlyNoopPlan(**values)
        if _is_tensor(result.cell_ids):
            return _register_resident_plan(result)
        return _validate_host_plan(result)

    def to_numpy(self):
        if not _is_tensor(self.cell_ids):
            return self.clone()
        _validate_resident_plan(self)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _PLAN_ARRAY_FIELDS:
                value = value.detach().cpu().numpy().copy()
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        return _validate_host_plan(A4ReplicationEarlyNoopPlan(**values))

    def to_torch(self, device='cpu'):
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        if _is_tensor(self.cell_ids):
            _validate_resident_plan(self)
            values = {}
            for item in fields(self):
                value = getattr(self, item.name)
                values[item.name] = (
                    value.detach().to(device=device).clone()
                    if item.name in _PLAN_ARRAY_FIELDS
                    else copy.deepcopy(value)
                )
        else:
            _validate_host_plan(self)
            values = {}
            for item in fields(self):
                value = getattr(self, item.name)
                values[item.name] = (
                    torch.as_tensor(np.asarray(value), device=device).clone()
                    if item.name in _PLAN_ARRAY_FIELDS
                    else copy.deepcopy(value)
                )
        return _register_resident_plan(A4ReplicationEarlyNoopPlan(**values))

    def state_dict(self):
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _PLAN_ARRAY_FIELDS:
                value = value.clone() if _is_tensor(value) else np.asarray(value).copy()
            else:
                value = copy.deepcopy(value)
            result[item.name] = value
        return result

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name)) for name in _PLAN_ARRAY_FIELDS):
            raise TypeError('data_ptrs requires a Torch-backed early-noop plan')
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _PLAN_ARRAY_FIELDS
        }

    def validate(self):
        return (
            _validate_resident_plan(self)
            if _is_tensor(self.cell_ids) else _validate_host_plan(self)
        )


def _validate_plan_schema(plan, resident):
    if not isinstance(plan, A4ReplicationEarlyNoopPlan):
        raise A4ReplicationEarlyNoopCommitError('untrusted early-noop plan')
    if plan.schema_version != SCHEMA_VERSION:
        raise A4ReplicationEarlyNoopCommitError('early-noop plan schema differs')
    C = int(plan.cell_capacity)
    N = int(plan.cell_count)
    if C < 0 or N < 0 or N > C:
        raise A4ReplicationEarlyNoopCommitError('invalid early-noop plan capacity')
    expected = {
        name: (torch.int64 if resident and name in _PLAN_INT64_FIELDS else
               torch.bool if resident and name in _PLAN_BOOL_FIELDS else
               torch.float64 if resident else
               np.dtype(np.int64) if name in _PLAN_INT64_FIELDS else
               np.dtype(bool) if name in _PLAN_BOOL_FIELDS else
               np.dtype(np.float64))
        for name in _PLAN_ARRAY_FIELDS
    }
    device = None
    for name in _PLAN_ARRAY_FIELDS:
        value = getattr(plan, name)
        if resident:
            if not _is_tensor(value) or value.dtype != expected[name] or tuple(value.shape) != (C,):
                raise A4ReplicationEarlyNoopCommitError('resident plan field differs: ' + name)
            if device is None:
                device = value.device
            elif value.device != device:
                raise A4ReplicationEarlyNoopCommitError('resident plan devices differ')
        else:
            value = np.asarray(value)
            if value.dtype != expected[name] or value.shape != (C,):
                raise A4ReplicationEarlyNoopCommitError('host plan field differs: ' + name)
    return C, N


def _validate_resident_plan(plan):
    _validate_plan_schema(plan, True)
    if getattr(plan, '_a48c8_metadata', None) != _plan_scalar_metadata(plan):
        raise A4ReplicationEarlyNoopCommitError('resident plan metadata changed')
    if getattr(plan, '_a48c8_data_ptrs', None) != plan.data_ptrs():
        raise A4ReplicationEarlyNoopCommitError('resident plan storage changed')
    versions = {name: int(getattr(plan, name)._version) for name in _PLAN_ARRAY_FIELDS}
    if getattr(plan, '_a48c8_versions', None) != versions:
        raise A4ReplicationEarlyNoopCommitError('resident plan values changed')
    return plan


def _validate_host_plan(plan):
    C, N = _validate_plan_schema(plan, False)
    raw = {name: np.asarray(getattr(plan, name)) for name in _PLAN_ARRAY_FIELDS}
    if (not np.array_equal(raw['cell_mask'][:N], np.ones(N, dtype=bool))
            or np.any(raw['cell_mask'][N:])
            or np.any(~np.isfinite(raw['replicase_activity']))
            or np.any(raw['replicase_activity'] < 0.0)
            or np.any(raw['last_replication_symbols_after'] != 0)
            or np.any(raw['rng_call_count'] != 0)):
        raise A4ReplicationEarlyNoopCommitError('early-noop plan values are invalid')
    success_codes = np.asarray(sorted(BRANCH_NAMES), dtype=np.int64)
    for ci in range(N):
        valid = bool(raw['scope_valid'][ci])
        code = int(raw['scope_error_code'][ci])
        branch = int(raw['branch_code'][ci])
        if valid:
            if code != SCOPE_OK or not np.any(success_codes == branch):
                raise A4ReplicationEarlyNoopCommitError('invalid early-noop success row')
        elif code == SCOPE_OK or branch != BRANCH_NONE:
            raise A4ReplicationEarlyNoopCommitError('invalid early-noop failure row')
    for name, value in raw.items():
        if name in ('cell_ids', 'cell_mask'):
            continue
        if value[N:].size and np.any(value[N:] != 0):
            raise A4ReplicationEarlyNoopCommitError('early-noop unused tail is not zero')
    return plan


def _empty_numpy_plan(binding, dt, flags, digest):
    state = binding.state
    C = int(state.cell_capacity)
    return A4ReplicationEarlyNoopPlan(
        schema_version=SCHEMA_VERSION, cell_capacity=C,
        cell_count=int(state.cell_count),
        source_provenance=str(state.source_provenance), dt_hex=dt.hex(),
        config_sha256=digest,
        genome_replication_enabled=flags['genome_replication'],
        external_replicase_enabled=flags['external_replicase'],
        mutation_enabled=flags['mutation'],
        cell_ids=np.asarray(state.cell_ids, np.int64).copy(),
        cell_mask=np.asarray(state.cell_mask, bool).copy(),
        scope_valid=np.zeros(C, bool),
        scope_error_code=np.zeros(C, np.int64),
        branch_code=np.zeros(C, np.int64),
        replicase_activity=np.zeros(C, np.float64),
        genome_count=np.asarray(state.genome_count, np.int64).copy(),
        replication_active=np.asarray(state.replication_active, bool).copy(),
        has_empty_complete_genome=np.zeros(C, bool),
        negative_atp=np.zeros(C, bool),
        last_replication_symbols_after=np.zeros(C, np.int64),
        rng_call_count=np.zeros(C, np.int64),
    )


def _numpy_has_empty_complete_genome(ragged, ci):
    count = int(ragged.genome_counts[ci])
    first_sequence = int(ragged.cell_sequence_offsets[ci])
    for offset in range(count):
        sequence = first_sequence + offset
        if int(ragged.sequence_offsets[sequence + 1]) == int(ragged.sequence_offsets[sequence]):
            return True
    return False


def paid_replication_early_noop_numpy(binding, dt, config):
    """Literal NumPy classification of the four ordered CPU early returns."""
    try:
        a4._require_translation_binding(binding)
        if _is_tensor(binding.state.pools):
            raise a4.A4SchemaError('NumPy early-noop plan requires a NumPy binding')
        a4.validate_a4_ragged(binding.ragged)
        a4.validate_a4_translation_state(binding.state)
        a4.validate_a4_gene_cache(binding.cache)
        dt = _strict_plan_dt(dt)
        flags, digest = _early_noop_config(config)
        plan = _empty_numpy_plan(binding, dt, flags, digest)
        state = binding.state
        specs_by_cell = binding.cache.materialize_gene_specs_host()
        N = int(state.cell_count)
        for ci in range(N):
            empty = _numpy_has_empty_complete_genome(binding.ragged, ci)
            negative = float(state.pools[ci, a4.a3.POOL_ATP]) < 0.0
            plan.has_empty_complete_genome[ci] = empty
            plan.negative_atp[ci] = negative
            replicase = a44._numpy_replicase(binding, ci, specs_by_cell[ci])
            if flags['external_replicase']:
                replicase += 0.85
            plan.replicase_activity[ci] = replicase
            if empty:
                plan.scope_error_code[ci] = SCOPE_EMPTY_COMPLETE_GENOME
            elif negative:
                plan.scope_error_code[ci] = SCOPE_NEGATIVE_ATP
            elif not flags['genome_replication']:
                plan.scope_valid[ci] = True
                plan.branch_code[ci] = BRANCH_DISABLED
            elif int(state.genome_count[ci]) == 0:
                plan.scope_valid[ci] = True
                plan.branch_code[ci] = BRANCH_NO_GENOME
            elif a44._numpy_fp64_comparison_boundary(replicase, 1e-6):
                plan.scope_error_code[ci] = SCOPE_FP64_COMPARISON_BOUNDARY
            elif replicase <= 1e-6:
                plan.scope_valid[ci] = True
                plan.branch_code[ci] = BRANCH_REPLICASE_GATE
            elif (not bool(state.replication_active[ci])
                  and int(state.genome_count[ci]) >= 2):
                plan.scope_valid[ci] = True
                plan.branch_code[ci] = BRANCH_INACTIVE_MULTI_GENOME
            else:
                plan.scope_error_code[ci] = SCOPE_NOT_EARLY_NOOP
        return _validate_host_plan(plan)
    except A4ReplicationEarlyNoopCommitError:
        raise
    except Exception as exc:
        raise A4ReplicationEarlyNoopCommitError(
            'NumPy early-noop planning failed closed'
        ) from exc


def _torch_ordered_replicase(binding):
    state = binding.state
    cache = binding.cache
    C = int(state.cell_capacity)
    K = int(cache.entry_capacity)
    device = state.pools.device
    dtype = state.pools.dtype
    rank = torch.arange(K, dtype=torch.int64, device=device)
    if K:
        first = torch.clamp(cache.cell_entry_offsets[:C], min=0)
        last = torch.clamp(cache.cell_entry_offsets[1:C + 1], min=0)
        counts = torch.clamp(last - first, min=0, max=K)
        indices = first[:, None] + rank[None, :]
        safe = torch.clamp(indices, min=0, max=K - 1)
        valid = (state.cell_mask[:, None]
                 & (rank[None, :] < counts[:, None])
                 & cache.entry_mask[safe])
        payload = cache.payloads[safe]
        fingerprints = cache.fingerprints[safe]
        role = torch.remainder(payload[:, :, 0].to(torch.int64), 8)
        parameter = torch.remainder(payload[:, :, 1].to(torch.int64), 8)
        promoter = 0.18 + 1.22 * (payload[:, :, 3].to(dtype) / 7.0)
        efficiency = 0.52 + 0.96 * (payload[:, :, 4].to(dtype) / 7.0)
        localisation = torch.remainder(payload[:, :, 6].to(torch.int64), 4)
    else:
        valid = torch.zeros((C, 0), dtype=torch.bool, device=device)
        fingerprints = torch.zeros((C, 0), dtype=torch.int64, device=device)
        role = torch.zeros((C, 0), dtype=torch.int64, device=device)
        parameter = torch.zeros((C, 0), dtype=torch.int64, device=device)
        promoter = torch.zeros((C, 0), dtype=dtype, device=device)
        efficiency = torch.zeros((C, 0), dtype=dtype, device=device)
        localisation = torch.zeros((C, 0), dtype=torch.int64, device=device)
    return a44._torch_replicase(
        binding, valid, fingerprints, role, parameter, localisation,
        promoter, efficiency,
    )[0]


def _torch_empty_complete_genome(binding):
    ragged = binding.ragged
    state = binding.state
    C = int(state.cell_capacity)
    Q = int(ragged.sequence_capacity)
    device = state.pools.device
    if Q == 0:
        return torch.zeros(C, dtype=torch.bool, device=device)
    rank = torch.arange(Q, dtype=torch.int64, device=device)
    indices = ragged.cell_sequence_offsets[:C, None] + rank[None, :]
    safe = torch.clamp(indices, min=0, max=Q - 1)
    valid = rank[None, :] < ragged.genome_counts[:, None]
    lengths = ragged.sequence_offsets[safe + 1] - ragged.sequence_offsets[safe]
    return torch.any(valid & (lengths == 0), dim=1)


def paid_replication_early_noop_torch(binding, dt, config):
    """Resident fixed-shape counterpart; classification performs no readback."""
    try:
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        a4._require_translation_binding(binding)
        if not _is_tensor(binding.state.pools):
            raise a4.A4SchemaError('Torch early-noop plan requires a Torch binding')
        a4._validate_resident_ragged_metadata(binding.ragged)
        a4._validate_translation_resident_metadata(binding.state)
        a4._validate_gene_backend_and_dtypes(binding.cache)
        dt = _strict_plan_dt(dt)
        flags, digest = _early_noop_config(config)
        state = binding.state
        C = int(state.cell_capacity)
        device = state.pools.device
        replicase = _torch_ordered_replicase(binding)
        if flags['external_replicase']:
            replicase = replicase + 0.85
        empty = _torch_empty_complete_genome(binding)
        negative = state.pools[:, a4.a3.POOL_ATP] < 0.0
        used = state.cell_mask
        disabled = used & (~empty) & (~negative) & (not flags['genome_replication'])
        no_genome = (used & (~empty) & (~negative)
                     & flags['genome_replication'] & (state.genome_count == 0))
        compare = (used & (~empty) & (~negative)
                   & flags['genome_replication'] & (state.genome_count > 0))
        boundary = compare & a44._torch_fp64_comparison_boundary(replicase, 1e-6)
        gate = compare & (~boundary) & (replicase <= 1e-6)
        inactive_multi = (compare & (~boundary) & (replicase > 1e-6)
                          & (~state.replication_active)
                          & (state.genome_count >= 2))
        branch = torch.zeros(C, dtype=torch.int64, device=device)
        branch = torch.where(disabled, torch.full_like(branch, BRANCH_DISABLED), branch)
        branch = torch.where(no_genome, torch.full_like(branch, BRANCH_NO_GENOME), branch)
        branch = torch.where(gate, torch.full_like(branch, BRANCH_REPLICASE_GATE), branch)
        branch = torch.where(inactive_multi, torch.full_like(branch, BRANCH_INACTIVE_MULTI_GENOME), branch)
        valid = disabled | no_genome | gate | inactive_multi
        error = torch.zeros(C, dtype=torch.int64, device=device)
        not_early = used & (~valid) & (~empty) & (~negative) & (~boundary)
        error = torch.where(not_early, torch.full_like(error, SCOPE_NOT_EARLY_NOOP), error)
        error = torch.where(empty & used, torch.full_like(error, SCOPE_EMPTY_COMPLETE_GENOME), error)
        error = torch.where(
            negative & used & (~empty),
            torch.full_like(error, SCOPE_NEGATIVE_ATP), error,
        )
        error = torch.where(boundary, torch.full_like(error, SCOPE_FP64_COMPARISON_BOUNDARY), error)
        plan = A4ReplicationEarlyNoopPlan(
            schema_version=SCHEMA_VERSION, cell_capacity=C,
            cell_count=int(state.cell_count),
            source_provenance=str(state.source_provenance), dt_hex=dt.hex(),
            config_sha256=digest,
            genome_replication_enabled=flags['genome_replication'],
            external_replicase_enabled=flags['external_replicase'],
            mutation_enabled=flags['mutation'],
            cell_ids=state.cell_ids.clone(), cell_mask=used.clone(),
            scope_valid=valid,
            scope_error_code=error, branch_code=branch,
            replicase_activity=torch.where(used, replicase, torch.zeros_like(replicase)),
            genome_count=state.genome_count.clone(),
            replication_active=state.replication_active.clone(),
            has_empty_complete_genome=empty & used,
            negative_atp=negative & used,
            last_replication_symbols_after=torch.zeros(C, dtype=torch.int64, device=device),
            rng_call_count=torch.zeros(C, dtype=torch.int64, device=device),
        )
        return _register_resident_plan(plan)
    except A4ReplicationEarlyNoopCommitError:
        raise
    except Exception as exc:
        raise A4ReplicationEarlyNoopCommitError(
            'Torch early-noop planning failed closed'
        ) from exc


def _canonical_early_noop_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 save integration state is absent or noncanonical'
        )
    if value.get('schema') != INTEGRATION_SCHEMA_VERSION:
        raise A4ReplicationEarlyNoopCommitError('A4.8c8 save schema differs')
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 saved config or device is invalid'
        ) from exc
    return {
        'schema': INTEGRATION_SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _host_plan_artifact_snapshot(plan):
    _validate_host_plan(plan)
    return {
        'object_id': id(plan),
        'array_ids': {name: id(getattr(plan, name)) for name in _PLAN_ARRAY_FIELDS},
        'data_ptrs': {
            name: int(np.asarray(getattr(plan, name)).__array_interface__['data'][0])
            for name in _PLAN_ARRAY_FIELDS
        },
        'state': plan.state_dict(),
    }


def _require_host_plan_artifact(plan, expected):
    actual = _host_plan_artifact_snapshot(plan)
    state = actual.pop('state')
    frozen = copy.deepcopy(expected)
    expected_state = frozen.pop('state')
    if actual != frozen or not a48a._state_arrays_bit_exact(state, expected_state):
        raise A4ReplicationEarlyNoopCommitError(
            'retained A4.8c8 host plan changed before claim'
        )
    return plan


def _host_binding_content_snapshot(binding):
    a4._require_translation_binding(binding)
    return {
        'ragged': binding.ragged.state_dict(),
        'state': binding.state.state_dict(),
        'cache': binding.cache.state_dict(),
    }


def _binding_content_matches(binding, cache, expected):
    a4._require_translation_binding(binding)
    current = {
        'ragged': binding.ragged.state_dict(),
        'state': binding.state.state_dict(),
        'cache': cache.state_dict(),
    }
    return all(
        a48a._state_arrays_bit_exact(current[name], expected[name])
        for name in ('ragged', 'state', 'cache')
    )


def _resident_artifact_snapshot(binding, plan, device):
    a4._require_translation_binding(binding)
    _validate_resident_plan(plan)
    expected_device = torch.device(str(device))
    batches = (
        ('ragged', binding.ragged, a4._ARRAY_FIELDS),
        ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        ('cache', binding.cache, a4._GENE_ARRAY_FIELDS),
        ('plan', plan, _PLAN_ARRAY_FIELDS),
    )
    result = {'object_ids': {'binding': id(binding)}, 'data_ptrs': {}, 'versions': {}}
    for label, batch, names in batches:
        result['object_ids'][label] = id(batch)
        result['data_ptrs'][label] = {}
        result['versions'][label] = {}
        for name in names:
            value = getattr(batch, name)
            if not _is_tensor(value):
                raise A4ReplicationEarlyNoopCommitError(
                    'A4.8c8 resident artifact escaped Torch authority'
                )
            actual_device = value.device
            if (actual_device.type != expected_device.type
                    or (expected_device.index is not None
                        and actual_device.index != expected_device.index)):
                raise A4ReplicationEarlyNoopCommitError(
                    'A4.8c8 resident artifact moved to another device'
                )
            result['data_ptrs'][label][name] = int(value.data_ptr())
            result['versions'][label][name] = int(value._version)
    return result


def _require_resident_artifact(binding, plan, device, expected):
    if _resident_artifact_snapshot(binding, plan, device) != expected:
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 resident object, device, pointer, or version changed'
        )
    return binding, plan


def _plan_semantic_match(resident, oracle):
    _validate_host_plan(resident)
    _validate_host_plan(oracle)
    for item in fields(A4ReplicationEarlyNoopPlan):
        name = item.name
        left = getattr(resident, name)
        right = getattr(oracle, name)
        if name not in _PLAN_ARRAY_FIELDS:
            if left != right:
                raise A4ReplicationEarlyNoopCommitError(
                    'resident/NumPy early-noop metadata differs: ' + name
                )
            continue
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise A4ReplicationEarlyNoopCommitError(
                'resident/NumPy early-noop schema differs: ' + name
            )
        if left.dtype == np.dtype(np.float64):
            if (not np.isfinite(left).all() or not np.isfinite(right).all()
                    or np.any(np.abs(left - right) > REPLICATION_ORACLE_ATOL)):
                raise A4ReplicationEarlyNoopCommitError(
                    'resident/NumPy early-noop fp64 output differs: ' + name
                )
        elif not np.array_equal(left, right):
            raise A4ReplicationEarlyNoopCommitError(
                'resident/NumPy early-noop discrete output differs: ' + name
            )
    return resident


def _early_noop_source_snapshot(world, cell):
    template = getattr(cell, 'replication_template', None)
    if template is not None and not isinstance(template, np.ndarray):
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 replication template is noncanonical'
        )
    if not isinstance(cell.replication_copy, list):
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 replication copy is noncanonical'
        )
    if not isinstance(cell.gene_specs, dict):
        raise A4ReplicationEarlyNoopCommitError(
            'A4.8c8 gene cache is noncanonical'
        )
    return {
        'cell_state': a48c._cell_state_snapshot(cell),
        'pools_object': cell.pools,
        'pools': np.asarray(cell.pools, np.float64).copy(),
        'genomes_object': cell.genomes,
        'genome_objects': tuple(cell.genomes),
        'genomes': tuple(np.asarray(value, np.uint8).copy() for value in cell.genomes),
        'lesions_object': cell.genome_lesions,
        'lesions': tuple(cell.genome_lesions),
        'template_object': template,
        'template_value': None if template is None else np.asarray(template, np.uint8).copy(),
        'copy_object': cell.replication_copy,
        'copy_values': tuple(cell.replication_copy),
        'mutation_object': cell.mutation_events,
        'mutation_items': a48c._mapping_items_snapshot(cell.mutation_events),
        'proteins_object': cell.proteins,
        'proteins_items': a48c._mapping_items_snapshot(cell.proteins),
        'damaged_object': cell.damaged_proteins,
        'damaged_items': a48c._mapping_items_snapshot(cell.damaged_proteins),
        'gene_specs_object': cell.gene_specs,
        'gene_specs_items': tuple(
            (fingerprint, spec, copy.deepcopy(spec))
            for fingerprint, spec in cell.gene_specs.items()
        ),
        'replication_template_lesion': float(cell.replication_template_lesion),
        'replication_fractional': float(cell.replication_fractional),
        'replication_cycles': int(cell.replication_cycles),
        'last_replication_symbols': int(cell.last_replication_symbols),
        'last_effective_error_rate': float(cell.last_effective_error_rate),
        'cumulative_proofreading_atp': float(cell.cumulative_proofreading_atp),
        'novel_path_first_age': copy.deepcopy(cell.novel_path_first_age),
        'age': float(cell.age), 'alive': bool(cell.alive),
        'rng_object': world.rng,
        'rng_state': copy.deepcopy(world.rng.bit_generator.state),
        'dissipated_energy': float(world.dissipated_energy),
    }


def _early_noop_source_matches(world, cell, snapshot, allow_last_zero=False):
    if (cell.pools is not snapshot['pools_object']
            or cell.genomes is not snapshot['genomes_object']
            or len(cell.genomes) != len(snapshot['genome_objects'])
            or any(
                current is not expected
                for current, expected in zip(
                    cell.genomes, snapshot['genome_objects'],
                )
            )
            or cell.genome_lesions is not snapshot['lesions_object']
            or cell.replication_template is not snapshot['template_object']
            or cell.replication_copy is not snapshot['copy_object']
            or cell.mutation_events is not snapshot['mutation_object']
            or tuple(cell.mutation_events.items()) != snapshot['mutation_items']
            or cell.proteins is not snapshot['proteins_object']
            or tuple(cell.proteins.items()) != snapshot['proteins_items']
            or cell.damaged_proteins is not snapshot['damaged_object']
            or tuple(cell.damaged_proteins.items()) != snapshot['damaged_items']
            or cell.gene_specs is not snapshot['gene_specs_object']
            or tuple(cell.gene_specs) != tuple(item[0] for item in snapshot['gene_specs_items'])
            or any(cell.gene_specs[key] is not spec for key, spec, _ in snapshot['gene_specs_items'])
            or world.rng is not snapshot['rng_object']
            or world.rng.bit_generator.state != snapshot['rng_state']
            or not a48c._float64_bits_equal(
                np.asarray([world.dissipated_energy], np.float64),
                np.asarray([snapshot['dissipated_energy']], np.float64),
            )):
        return False
    if allow_last_zero:
        current_last = cell.last_replication_symbols
        cell.last_replication_symbols = snapshot['last_replication_symbols']
        try:
            same = a48c._cell_state_snapshot(cell) == snapshot['cell_state']
        finally:
            cell.last_replication_symbols = current_last
        return same and int(current_last) == 0
    return a48c._cell_state_snapshot(cell) == snapshot['cell_state']


def _restore_mapping(mapping, items):
    mapping.clear()
    for key, value in items:
        mapping[key] = value
    return mapping


@dataclass
class _A4ReplicationEarlyNoopCommitCandidate:
    _factory_token: object
    scheduler_object_id: int
    world_object_id: int
    cell_object_id: int
    cell_id: int
    generation: int
    dt_hex: str
    branch_code: int
    scope_error_code: int
    source_binding_identity: object
    source_binding: object
    source_artifacts: object
    source_content: object
    source_gene_specs: object
    source_snapshot: object
    host_plan: object
    host_plan_artifacts: object
    resident_binding: object
    resident_plan: object
    resident_artifacts: object
    plan: object
    plan_state: object
    live_rng: object
    rng_state: object
    dissipated_energy: float
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4ReplicationEarlyNoopEventScheduler(
        a48c7.A4ReplicationStartCompletionMutationEventScheduler):
    """A4.8c7 scheduler plus canonical-valid early-noop authority."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_early_noop_commit_active = False
        super(A4ReplicationEarlyNoopEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def replication_early_noop_integration_state(self):
        return {
            'schema': INTEGRATION_SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_early_noop_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c8 early-noop commit'
            )
        state = super(A4ReplicationEarlyNoopEventScheduler, self).state_dict()
        state['a4_replication_early_noop'] = self.replication_early_noop_integration_state()
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        own = _canonical_early_noop_integration_state(
            state.get('a4_replication_early_noop'),
        )
        inherited = a48c7.A4ReplicationStartCompletionMutationEventScheduler.from_state(state)
        if (own['config'] != a48a._config_state(inherited.a4_config)
                or own['device'] != inherited.a4_device):
            raise A4ReplicationEarlyNoopCommitError(
                'saved A4.8c7/A4.8c8 config or device differs'
            )
        scheduler = cls(
            a4_config=own['config'], device=own['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler

    def _prepare_replication_early_noop_candidate(self, world, cell, dt, config):
        dt = _strict_plan_dt(dt)
        if config is not world.config:
            raise A4ReplicationEarlyNoopCommitError(
                'replication config must be the active world config object'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 source must be alive; dead direct replication is excluded'
            )
        generation = a48a._strict_counter(getattr(cell, 'generation', None), 'cell generation')
        flags, _ = _early_noop_config(config)
        source_snapshot = _early_noop_source_snapshot(world, cell)
        live_rng, rng_state = a48c._live_pcg64(world)
        _, source_binding = self._pack_host_binding(world, cell)
        if int(source_binding.state.cell_count) != 1:
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 live commit requires exactly one source cell'
            )
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(cell, source_binding, expected=source_gene_specs)
        source_artifacts = a48c._host_binding_artifact_snapshot(source_binding)
        source_content = _host_binding_content_snapshot(source_binding)
        host_plan = paid_replication_early_noop_numpy(source_binding, dt, config)
        resident_binding = a4.bind_a4_translation(
            source_binding.ragged.to_torch(device=self.a4_device),
            source_binding.state.to_torch(device=self.a4_device),
        )
        resident_plan = paid_replication_early_noop_torch(resident_binding, dt, config)
        resident_artifacts = _resident_artifact_snapshot(
            resident_binding, resident_plan, self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(resident_binding)
        if (a48a._binding_identity(resident_source) != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache) != source_binding._cache_provenance):
            raise A4ReplicationEarlyNoopCommitError(
                'resident A4.8c8 source differs after explicit readback'
            )
        replay = paid_replication_early_noop_numpy(resident_source, dt, config)
        if not a48a._state_arrays_bit_exact(replay.state_dict(), host_plan.state_dict()):
            raise A4ReplicationEarlyNoopCommitError(
                'resident source A4.8c8 replay differs from NumPy oracle'
            )
        plan = resident_plan.to_numpy()
        _plan_semantic_match(plan, host_plan)
        return _A4ReplicationEarlyNoopCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self), world_object_id=id(world),
            cell_object_id=id(cell), cell_id=int(cell.cell_id),
            generation=generation, dt_hex=dt.hex(),
            branch_code=int(plan.branch_code[0]),
            scope_error_code=int(plan.scope_error_code[0]),
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding, source_artifacts=source_artifacts,
            source_content=source_content,
            source_gene_specs=source_gene_specs, source_snapshot=source_snapshot,
            host_plan=host_plan,
            host_plan_artifacts=_host_plan_artifact_snapshot(host_plan),
            resident_binding=resident_binding, resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan, plan_state=plan.state_dict(),
            live_rng=live_rng, rng_state=copy.deepcopy(rng_state),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(self.replication_early_noop_integration_state()),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_replication_early_noop_candidate_unchecked(
            self, world, cell, dt, config, candidate):
        if (not isinstance(candidate, _A4ReplicationEarlyNoopCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 candidate is untrusted or already consumed'
            )
        dt = _strict_plan_dt(dt)
        live_rng, rng_state = a48c._live_pcg64(world)
        if (id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or id(cell) != candidate.cell_object_id
                or int(cell.cell_id) != candidate.cell_id
                or a48a._strict_counter(getattr(cell, 'generation', None), 'cell generation') != candidate.generation
                or not bool(getattr(cell, 'alive', False))
                or config is not world.config
                or dt.hex() != candidate.dt_hex
                or live_rng is not candidate.live_rng
                or rng_state != candidate.rng_state
                or self.replication_early_noop_integration_state() != candidate.integration_state
                or a48a._world_config_snapshot(world) != candidate.world_config_snapshot
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([candidate.dissipated_energy], np.float64),
                )
                or not _early_noop_source_matches(world, cell, candidate.source_snapshot)):
            raise A4ReplicationEarlyNoopCommitError(
                'live A4.8c8 cell/config/dt/RNG changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, current_binding, expected=candidate.source_gene_specs)
        if (a48a._binding_identity(current_binding) != candidate.source_binding_identity
                or a48c._host_binding_artifact_snapshot(candidate.source_binding) != candidate.source_artifacts):
            raise A4ReplicationEarlyNoopCommitError(
                'live or retained A4.8c8 source binding changed'
            )
        if (not _binding_content_matches(
                candidate.source_binding, candidate.source_binding.cache,
                candidate.source_content)
                or not _binding_content_matches(
                    current_binding, current_binding.cache,
                    candidate.source_content)):
            raise A4ReplicationEarlyNoopCommitError(
                'live or retained A4.8c8 host source content changed'
            )
        _require_host_plan_artifact(candidate.host_plan, candidate.host_plan_artifacts)
        _require_resident_artifact(
            candidate.resident_binding, candidate.resident_plan,
            self.a4_device, candidate.resident_artifacts,
        )
        retained_source, retained_cache = a48c._resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(retained_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(retained_cache)
                != candidate.source_binding._cache_provenance
                or not _binding_content_matches(
                    retained_source, retained_cache,
                    candidate.source_content)):
            raise A4ReplicationEarlyNoopCommitError(
                'retained A4.8c8 resident source/cache content changed'
            )
        retained_replay = paid_replication_early_noop_numpy(
            retained_source, dt, config,
        )
        if not a48a._state_arrays_bit_exact(
                retained_replay.state_dict(), candidate.host_plan.state_dict()):
            raise A4ReplicationEarlyNoopCommitError(
                'retained A4.8c8 resident source replay changed'
            )
        retained_plan = candidate.resident_plan.to_numpy()
        if (not a48a._state_arrays_bit_exact(
                retained_plan.state_dict(), candidate.plan_state)
                or not a48a._state_arrays_bit_exact(
                    candidate.plan.state_dict(), candidate.plan_state)):
            raise A4ReplicationEarlyNoopCommitError(
                'retained A4.8c8 resident readback plan changed'
            )
        if (int(retained_plan.cell_count) != 1
                or not bool(retained_plan.cell_mask[0])
                or int(retained_plan.cell_ids[0]) != candidate.cell_id
                or str(retained_plan.source_provenance)
                != str(candidate.source_binding.state.source_provenance)
                or candidate.branch_code
                != int(retained_plan.branch_code[0])
                or candidate.scope_error_code
                != int(retained_plan.scope_error_code[0])):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 candidate scalar association changed'
            )
        fresh_host = paid_replication_early_noop_numpy(current_binding, dt, config)
        if not a48a._state_arrays_bit_exact(fresh_host.state_dict(), candidate.host_plan.state_dict()):
            raise A4ReplicationEarlyNoopCommitError(
                'fresh A4.8c8 NumPy plan changed before claim'
            )
        fresh_resident_binding = a4.bind_a4_translation(
            current_binding.ragged.to_torch(device=self.a4_device),
            current_binding.state.to_torch(device=self.a4_device),
        )
        fresh_resident = paid_replication_early_noop_torch(
            fresh_resident_binding, dt, config,
        ).to_numpy()
        _plan_semantic_match(fresh_resident, fresh_host)
        if not a48a._state_arrays_bit_exact(fresh_resident.state_dict(), candidate.plan_state):
            raise A4ReplicationEarlyNoopCommitError(
                'fresh A4.8c8 resident plan changed before claim'
            )
        return candidate

    def _revalidate_replication_early_noop_candidate(
            self, world, cell, dt, config, candidate):
        try:
            return self._revalidate_replication_early_noop_candidate_unchecked(
                world, cell, dt, config, candidate,
            )
        except A4ReplicationEarlyNoopCommitError:
            raise
        except Exception as exc:
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 candidate trust validation failed closed'
            ) from exc

    def _replication_early_noop_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after prepare and before final trust."""
        return candidate

    def _publish_replication_early_noop_candidate(self, world, cell, candidate):
        cell.last_replication_symbols = int(
            candidate.plan.last_replication_symbols_after[0]
        )

    def _rollback_replication_early_noop_publish(self, world, cell, snapshot):
        snapshot['pools_object'][...] = snapshot['pools']
        cell.pools = snapshot['pools_object']
        for current, saved in zip(snapshot['genome_objects'], snapshot['genomes']):
            current[...] = saved
        snapshot['genomes_object'][:] = list(snapshot['genome_objects'])
        cell.genomes = snapshot['genomes_object']
        snapshot['lesions_object'][:] = list(snapshot['lesions'])
        cell.genome_lesions = snapshot['lesions_object']
        template = snapshot['template_object']
        if template is not None:
            template[...] = snapshot['template_value']
        cell.replication_template = template
        snapshot['copy_object'][:] = list(snapshot['copy_values'])
        cell.replication_copy = snapshot['copy_object']
        cell.mutation_events = _restore_mapping(snapshot['mutation_object'], snapshot['mutation_items'])
        cell.proteins = _restore_mapping(snapshot['proteins_object'], snapshot['proteins_items'])
        cell.damaged_proteins = _restore_mapping(snapshot['damaged_object'], snapshot['damaged_items'])
        specs = snapshot['gene_specs_object']
        for _, spec, saved in snapshot['gene_specs_items']:
            spec.clear(); spec.update(copy.deepcopy(saved))
        specs.clear()
        for fingerprint, spec, _ in snapshot['gene_specs_items']:
            specs[fingerprint] = spec
        cell.gene_specs = specs
        cell.replication_template_lesion = snapshot['replication_template_lesion']
        cell.replication_fractional = snapshot['replication_fractional']
        cell.replication_cycles = snapshot['replication_cycles']
        cell.last_replication_symbols = snapshot['last_replication_symbols']
        cell.last_effective_error_rate = snapshot['last_effective_error_rate']
        cell.cumulative_proofreading_atp = snapshot['cumulative_proofreading_atp']
        cell.novel_path_first_age = copy.deepcopy(snapshot['novel_path_first_age'])
        cell.age = snapshot['age']; cell.alive = snapshot['alive']
        world.dissipated_energy = snapshot['dissipated_energy']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])

    def _commit_replication_early_noop_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_replication_early_noop_candidate(
            world, cell, dt, config, candidate,
        )
        if (candidate.scope_error_code != SCOPE_OK
                or candidate.branch_code not in BRANCH_NAMES):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 candidate is not an owned early-noop row'
            )
        snapshot = _early_noop_source_snapshot(world, cell)
        candidate.consumed = True
        branch = BRANCH_NAMES[candidate.branch_code]
        self.claim(cell, 'replication_cpu', metadata={
            'authority': 'A4.8c8-resident-pre-active-ordinary-early-noop-atomic-commit',
            'branch': branch, 'device': self.a4_device,
            'enabled': bool(candidate.plan.genome_replication_enabled),
            'genome_count': int(candidate.plan.genome_count[0]),
            'source_provenance': str(candidate.plan.source_provenance),
            'classifier_sha256': a44._sha256_json({
                'config': candidate.plan.config_sha256,
                'dt_hex': candidate.plan.dt_hex,
                'branch_code': candidate.branch_code,
                'source_provenance': candidate.plan.source_provenance,
            }),
        })
        try:
            self._publish_replication_early_noop_candidate(world, cell, candidate)
            if not _early_noop_source_matches(world, cell, snapshot, allow_last_zero=True):
                raise A4ReplicationEarlyNoopCommitError(
                    'A4.8c8 publish changed state beyond last-symbol zero'
                )
            _, published = self._pack_host_binding(world, cell)
            a48b._require_cell_gene_specs(cell, published, expected=candidate.source_gene_specs)
            if a48a._binding_identity(published) != candidate.source_binding_identity:
                raise A4ReplicationEarlyNoopCommitError(
                    'published A4.8c8 binding differs from source'
                )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': 0, 'work_performed': False,
                'rng_call_count': 0, 'last_replication_symbols': 0,
            })
        except BaseException:
            self._rollback_replication_early_noop_publish(world, cell, snapshot)
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Commit a canonical-valid c8 no-op or delegate normal work to c7."""
        _, record = self._context(world, cell, dt)
        if 'replication_cpu' in record['claimed']:
            raise a3s.A3DuplicateEventError('duplicate A4.8c8 replication_cpu event')
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationEarlyNoopCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_early_noop_commit_active
                or self._a4_replication_start_completion_mutation_commit_active
                or self._a4_replication_start_completion_commit_active
                or self._a4_replication_mutation_free_start_commit_active
                or self._a4_replication_start_commit_active
                or self._a4_replication_completion_mutation_commit_active
                or self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c8 replication commit is forbidden'
            )
        candidate = self._prepare_replication_early_noop_candidate(
            world, cell, dt, config,
        )
        if candidate.scope_error_code == SCOPE_NOT_EARLY_NOOP:
            return super(A4ReplicationEarlyNoopEventScheduler, self).cpu_replication(
                world, cell, dt, config,
            )
        if candidate.scope_error_code != SCOPE_OK:
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 preclaim scope exclusion code=%d' % candidate.scope_error_code
            )
        self._a4_replication_early_noop_commit_active = True
        try:
            ready = self._replication_early_noop_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationEarlyNoopCommitError(
                    'A4.8c8 ready hook must retain its private candidate'
                )
            return self._commit_replication_early_noop_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_early_noop_commit_active = False


class Hybrid066WorldA4ReplicationEarlyNoop(
        a48c7.Hybrid066WorldA4ReplicationStartCompletionMutation):
    """A4.8c7 world retaining bounded A4.8c8 authority on restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationEarlyNoopEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(scheduler, A4ReplicationEarlyNoopEventScheduler):
            raise TypeError(
                'scheduler must be A4ReplicationEarlyNoopEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationEarlyNoop, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationEarlyNoopCommitError(
                'new A4.8c8 world requires explicit A4 config and device'
            )
        config = (
            world_config
            if isinstance(world_config, a3s.a2.s66.Formal066Config)
            else a3s.a2.s66.Formal066Config.from_state(world_config or {})
        )
        world = a3s.a2.s66.Formal066World(
            seed=seed, initial_cells=initial_cells, config=config,
        )
        return cls(
            world, backend=backend,
            gpu_config=gpu_config if backend is None else None,
            a4_config=a4_config, a4_device=a4_device,
        )

    def summary(self):
        output = super(Hybrid066WorldA4ReplicationEarlyNoop, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': INTEGRATION_SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_early_noop_device': self.scheduler.a4_device,
            'a4_replication_early_noop_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4ReplicationEarlyNoop, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_early_noop': (
                self.scheduler.replication_early_noop_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8c8 save version/build differs'
            )
        own = _canonical_early_noop_integration_state(
            state.get('a4_replication_early_noop'),
        )
        c7 = a48c7._canonical_start_completion_mutation_integration_state(
            state.get('a4_replication_start_completion_mutation'),
        )
        c6 = a48c6._canonical_start_completion_integration_state(
            state.get('a4_replication_start_completion'),
        )
        c5 = a48c5._canonical_mutation_free_start_integration_state(
            state.get('a4_replication_mutation_free_start'),
        )
        c4 = a48c4._canonical_start_integration_state(
            state.get('a4_replication_start'),
        )
        c3 = a48c3._canonical_completion_mutation_integration_state(
            state.get('a4_replication_completion_mutation'),
        )
        c2 = a48c2._canonical_completion_integration_state(
            state.get('a4_replication_completion'),
        )
        c1 = a48c._canonical_replication_integration_state(
            state.get('a4_replication'),
        )
        translation = a48b._canonical_translation_integration_state(
            state.get('a4_translation'),
        )
        hydrolysis = a48a._canonical_integration_state(
            state.get('a4_hydrolysis'),
        )
        world = a3s.a2.s66.Formal066World.from_state(state['cpu_world'])
        if 'aggregate_sidecars' not in state:
            raise a3s.A3SchedulerProtocolError(
                'A4.8c8 save is missing aggregate composition sidecars'
            )
        a3s._restore_aggregate_sidecars(world, state['aggregate_sidecars'])
        config_state = copy.deepcopy(state.get('gpu_config', {}))
        if backend is None:
            backend = (
                backend_factory(config_state)
                if backend_factory is not None
                else a3s._default_backend(config_state)
            )
        scheduler = A4ReplicationEarlyNoopEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if (scheduler.replication_early_noop_integration_state() != own
                or scheduler.replication_start_completion_mutation_integration_state() != c7
                or scheduler.replication_start_completion_integration_state() != c6
                or scheduler.replication_mutation_free_start_integration_state() != c5
                or scheduler.replication_start_integration_state() != c4
                or scheduler.completion_mutation_integration_state() != c3
                or scheduler.completion_integration_state() != c2
                or scheduler.replication_integration_state() != c1
                or scheduler.translation_integration_state() != translation
                or scheduler.integration_state() != hydrolysis):
            raise A4ReplicationEarlyNoopCommitError(
                'A4.8a/b/c1/c2/c3/c4/c5/c6/c7/c8 settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c7.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c8-canonical-valid-alive-nonempty-nonnegative-atp-ordered-'
        'disabled-no-genome-replicase-gate-inactive-multi-genome-'
        'resident-early-noop-last-symbol-only-atomic-commit-plus-a4.8c7-'
        'c6-c5-c4-c3-c2-c1-inherited-normal-work'
    ),
    'full_gpu_world_step': False,
})
