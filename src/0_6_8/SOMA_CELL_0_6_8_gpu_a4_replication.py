# coding: utf-8
"""A4.6b2 resident completion-mutation transformation plan.

This deliberately narrow development slice retains every A4.6a plan and the
A4.5 substitution tape, then adds a separate, binding-aware host replay for a
pre-existing active template that completes with mutation enabled.  One
combined PCG64 stream records each cell's paid append substitutions followed
immediately by that same cell's insertion/deletion/duplication/inversion/
transposition and padding calls.  The tape contains semantic draws and bounded
integer payloads, never a precomputed final genome.  This slice consumes that
tape with fixed-shape NumPy or Torch CPU/CUDA operations and returns a pure
structural/material/lifecycle descriptor.  It still does not mutate the ragged
arena, advance a live RNG, refresh a gene cache, commit a CPU cell, hydrolyse
proteins, or replace the A3 scheduler.  Frozen Formal066 CPU behavior remains
authority.
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


BUILD = 'SOMA-CELL 0.6.8-GPU A4.6b2'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | resident completion mutation transformation plan'
SCHEMA_VERSION = '0.6.8-GPU-A4.6a-replication-completion-plan'
RNG_TAPE_SCHEMA_VERSION = '0.6.8-GPU-A4.5b-template-start-rng-tape'
COMPLETION_MUTATION_RNG_TAPE_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.6b1-completion-mutation-rng-tape'
)
COMPLETION_MUTATION_PLAN_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.6b2-completion-mutation-plan'
)
FULL_GPU_WORLD_STEP = False

SCOPE_OK = 0
SCOPE_INACTIVE_TEMPLATE = 1
SCOPE_REPLICASE_GATE = 2
SCOPE_COMPLETION = 3
SCOPE_CAPACITY = 4
SCOPE_NEGATIVE_ATP = 5
SCOPE_FP64_DISCRETE_BOUNDARY = 6
SCOPE_RNG_TAPE_MISMATCH = 7
SCOPE_NONCOMPLETION = 8

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
    'completion_events', 'completed_symbols', 'completed_lengths',
    'new_genome_lesions', 'replication_cycle_deltas',
    'topology_sequence_deltas', 'topology_symbol_deltas',
)
_PLAN_UINT8_FIELDS = ('append_symbols', 'completed_symbols')
_PLAN_INT64_FIELDS = (
    'cell_ids', 'scope_error_code', 'requested_symbols', 'append_count',
    'last_replication_symbols',
    'substitution_events', 'selected_template_indices',
    'template_storage_symbols',
    'completed_lengths', 'replication_cycle_deltas',
    'topology_sequence_deltas', 'topology_symbol_deltas',
)
_PLAN_BOOL_FIELDS = (
    'cell_mask', 'scope_valid', 'template_start_events',
    'completion_events',
)
_PLAN_FLOAT64_FIELDS = (
    'pools_after', 'replication_fractional_after',
    'last_effective_error_rate', 'cumulative_proofreading_atp_after',
    'new_genome_lesions',
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

STRUCTURAL_EVENT_NAMES = (
    'insertion', 'deletion', 'duplication', 'inversion', 'transposition',
)
STRUCTURAL_EVENT_COUNT = len(STRUCTURAL_EVENT_NAMES)
STRUCTURAL_INSERTION = 0
STRUCTURAL_DELETION = 1
STRUCTURAL_DUPLICATION = 2
STRUCTURAL_INVERSION = 3
STRUCTURAL_TRANSPOSITION = 4
STRUCTURAL_THRESHOLD_FACTORS = (0.55, 0.50, 0.42, 0.28, 0.20)
STRUCTURAL_SHORT_EDIT_MAX = 5

_COMPLETION_TAPE_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask',
    'append_draw_mask', 'append_uniform_draws',
    'replacement_raw', 'replacement_mask',
    'append_count', 'substitution_count', 'effective_error',
    'nucleotide_budget_symbols', 'pre_structural_lengths',
    'post_structural_lengths', 'material_delta_symbols',
    'structural_event_counts',
    'threshold_draw_mask', 'threshold_uniform_draws', 'threshold_hit_mask',
    'insertion_count', 'insertion_position', 'insertion_symbols',
    'deletion_count', 'deletion_position',
    'duplication_gene_ordinal', 'duplication_source_start',
    'duplication_position',
    'inversion_left', 'inversion_right',
    'transposition_count', 'transposition_start', 'transposition_position',
    'padding_count', 'padding_symbols',
)
_COMPLETION_TAPE_UINT8_FIELDS = (
    'replacement_raw', 'insertion_symbols', 'padding_symbols',
)
_COMPLETION_TAPE_INT64_FIELDS = (
    'cell_ids', 'append_count', 'substitution_count',
    'nucleotide_budget_symbols', 'pre_structural_lengths',
    'post_structural_lengths', 'material_delta_symbols',
    'structural_event_counts',
    'insertion_count', 'insertion_position',
    'deletion_count', 'deletion_position',
    'duplication_gene_ordinal', 'duplication_source_start',
    'duplication_position', 'inversion_left', 'inversion_right',
    'transposition_count', 'transposition_start', 'transposition_position',
    'padding_count',
)
_COMPLETION_TAPE_BOOL_FIELDS = (
    'cell_mask', 'append_draw_mask', 'replacement_mask',
    'threshold_draw_mask', 'threshold_hit_mask',
)
_COMPLETION_TAPE_FLOAT64_FIELDS = (
    'append_uniform_draws', 'effective_error', 'threshold_uniform_draws',
)
_COMPLETION_TAPE_FACTORY_TOKEN = object()

_COMPLETION_PLAN_ARRAY_FIELDS = (
    'cell_ids', 'cell_mask', 'scope_valid', 'scope_error_code',
    'completion_events', 'pre_structural_lengths', 'final_symbols',
    'final_lengths', 'pools_after', 'requested_symbols', 'append_count',
    'last_replication_symbols', 'last_effective_error_rate',
    'cumulative_proofreading_atp_after', 'substitution_events',
    'structural_event_counts', 'material_delta_symbols',
    'new_genome_lesions', 'replication_cycle_deltas',
    'topology_sequence_deltas', 'topology_symbol_deltas',
    'replication_active_after', 'replication_template_lesions_after',
    'replication_fractional_after', 'genome_count_after',
    'genome_material_symbols_after', 'genome_lesion_mean_after',
)
_COMPLETION_PLAN_UINT8_FIELDS = ('final_symbols',)
_COMPLETION_PLAN_INT64_FIELDS = (
    'cell_ids', 'scope_error_code', 'pre_structural_lengths',
    'final_lengths', 'requested_symbols', 'append_count',
    'last_replication_symbols', 'substitution_events',
    'structural_event_counts', 'material_delta_symbols',
    'replication_cycle_deltas', 'topology_sequence_deltas',
    'topology_symbol_deltas', 'genome_count_after',
    'genome_material_symbols_after',
)
_COMPLETION_PLAN_BOOL_FIELDS = (
    'cell_mask', 'scope_valid', 'completion_events',
    'replication_active_after',
)
_COMPLETION_PLAN_FLOAT64_FIELDS = (
    'pools_after', 'last_effective_error_rate',
    'cumulative_proofreading_atp_after', 'new_genome_lesions',
    'replication_template_lesions_after',
    'replication_fractional_after', 'genome_lesion_mean_after',
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


def _completion_mutation_config(config):
    """Freeze exactly the Formal066 completion-mutation configuration."""
    if not hasattr(config, 'mutation') or not _strict_bool(
            getattr(config, 'mutation'), 'config.mutation'):
        raise A4ReplicationScopeError(
            'A4.6b1 completion mutation requires config.mutation=True'
        )
    deterministic = copy.deepcopy(config)
    deterministic.mutation = False
    mutation_rate, scale, flags = _supported_config(deterministic)
    for name in ('variable_length', 'gene_duplication'):
        if not hasattr(config, name):
            raise A4ReplicationScopeError('config missing %s' % name)
    variable_length = _strict_bool(
        getattr(config, 'variable_length'), 'config.variable_length',
    )
    gene_duplication = _strict_bool(
        getattr(config, 'gene_duplication'), 'config.gene_duplication',
    )
    structural_rate = _strict_real_scalar(
        getattr(config, 'structural_rate', None), 'config.structural_rate',
    )
    options = {
        'variable_length': variable_length,
        'gene_duplication': gene_duplication,
        'structural_rate': max(0.0, structural_rate),
    }
    payload = {
        'genome_replication': True,
        'mutation': True,
        'mutation_rate_hex': float(mutation_rate).hex(),
        'structural_rate_hex': float(structural_rate).hex(),
        'variable_length': variable_length,
        'gene_duplication': gene_duplication,
        'eco66_replication_rate_scale_hex': float(scale).hex(),
        'proofreading': bool(flags['proofreading']),
        'external_replicase': bool(flags['external_replicase']),
        'quiescence': bool(flags['quiescence']),
        'quiescence_effector': bool(flags['quiescence_effector']),
        'alphabet_size': int(a4.g2.ALPHABET_SIZE),
        'min_genome_length': int(a4.g2.MIN_GENOME_LENGTH),
        'max_genome_length': int(a4.g2.MAX_GENOME_LENGTH),
        'gene_span': int(a4.g2.GENE_SPAN),
    }
    return deterministic, options, _sha256_json(payload)


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
    completion_events: object
    completed_symbols: object
    completed_lengths: object
    new_genome_lesions: object
    replication_cycle_deltas: object
    topology_sequence_deltas: object
    topology_symbol_deltas: object

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


@dataclass
class A4CompletionMutationRngTape:
    """Ephemeral combined PCG64 tape; never final-genome or RNG authority."""

    _factory_token: object
    schema_version: str
    cell_capacity: int
    append_capacity: int
    mutation_capacity: int
    cell_count: int
    source_provenance: str
    dt_hex: str
    config_sha256: str
    schedule_sha256: str
    rng_before_state: object
    rng_after_state: object
    cell_ids: object
    cell_mask: object
    append_draw_mask: object
    append_uniform_draws: object
    replacement_raw: object
    replacement_mask: object
    append_count: object
    substitution_count: object
    effective_error: object
    nucleotide_budget_symbols: object
    pre_structural_lengths: object
    post_structural_lengths: object
    material_delta_symbols: object
    structural_event_counts: object
    threshold_draw_mask: object
    threshold_uniform_draws: object
    threshold_hit_mask: object
    insertion_count: object
    insertion_position: object
    insertion_symbols: object
    deletion_count: object
    deletion_position: object
    duplication_gene_ordinal: object
    duplication_source_start: object
    duplication_position: object
    inversion_left: object
    inversion_right: object
    transposition_count: object
    transposition_start: object
    transposition_position: object
    padding_count: object
    padding_symbols: object

    def clone(self):
        _require_completion_mutation_rng_tape(self)
        values = {
            name: (
                _clone_array(getattr(self, name))
                if name in _COMPLETION_TAPE_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, name))
            )
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'mutation_capacity', 'cell_count', 'source_provenance',
                'dt_hex', 'config_sha256', 'schedule_sha256',
                'rng_before_state', 'rng_after_state',
            ) + _COMPLETION_TAPE_ARRAY_FIELDS
        }
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        if _is_tensor(self.append_uniform_draws):
            values['_resident_expected_arrays'] = {
                name: self._resident_expected_arrays[name].clone()
                for name in _COMPLETION_TAPE_ARRAY_FIELDS
            }
        return _make_completion_mutation_rng_tape(**values)

    def to_torch(self, binding, dt, config, device='cpu'):
        """Validate against a NumPy source before the explicit H2D boundary."""
        if torch is None:
            raise RuntimeError('PyTorch is unavailable')
        validate_a4_completion_mutation_rng_tape(
            self, binding, dt, config,
        )
        requested = torch.device(device)
        if requested.type not in ('cpu', 'cuda'):
            raise a4.A4DeviceError(
                'completion-mutation RNG tape device must be cpu or cuda'
            )
        if requested.type == 'cuda' and not torch.cuda.is_available():
            raise a4.A4DeviceError('CUDA requested but unavailable')
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'mutation_capacity', 'cell_count', 'source_provenance',
                'dt_hex', 'config_sha256', 'schedule_sha256',
                'rng_before_state', 'rng_after_state',
            )
        }
        for name in _COMPLETION_TAPE_ARRAY_FIELDS:
            value = np.asarray(getattr(self, name))
            dtype = (
                torch.uint8 if name in _COMPLETION_TAPE_UINT8_FIELDS else
                torch.int64 if name in _COMPLETION_TAPE_INT64_FIELDS else
                torch.bool if name in _COMPLETION_TAPE_BOOL_FIELDS else
                torch.float64
            )
            values[name] = torch.as_tensor(
                value, dtype=dtype, device=requested,
            ).clone()
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        return _make_completion_mutation_rng_tape(**values)

    def to_numpy(self):
        """Read back values; binding-aware semantic validation remains explicit."""
        _require_completion_mutation_rng_tape(self)
        if not _is_tensor(self.append_uniform_draws):
            return self.clone()
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'mutation_capacity', 'cell_count', 'source_provenance',
                'dt_hex', 'config_sha256', 'schedule_sha256',
                'rng_before_state', 'rng_after_state',
            )
        }
        for name in _COMPLETION_TAPE_ARRAY_FIELDS:
            value = _host_array(getattr(self, name))
            dtype = (
                np.uint8 if name in _COMPLETION_TAPE_UINT8_FIELDS else
                np.int64 if name in _COMPLETION_TAPE_INT64_FIELDS else
                bool if name in _COMPLETION_TAPE_BOOL_FIELDS else
                np.float64
            )
            values[name] = value.astype(dtype, copy=False)
        values['_expected_host_array_sha256'] = (
            self._expected_host_array_sha256
        )
        return _make_completion_mutation_rng_tape(**values)

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _COMPLETION_TAPE_ARRAY_FIELDS):
            raise TypeError(
                'data_ptrs requires a Torch-backed completion RNG tape'
            )
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _COMPLETION_TAPE_ARRAY_FIELDS
        }

    def state_dict(self):
        return {
            name: (
                _clone_array(getattr(self, name))
                if name in _COMPLETION_TAPE_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, name))
            )
            for name in (
                'schema_version', 'cell_capacity', 'append_capacity',
                'mutation_capacity', 'cell_count', 'source_provenance',
                'dt_hex', 'config_sha256', 'schedule_sha256',
                'rng_before_state', 'rng_after_state',
            ) + _COMPLETION_TAPE_ARRAY_FIELDS
        }


@dataclass
class A4CompletionMutationPlan:
    """Pure fixed-shape completion result; never arena or live-RNG authority."""

    schema_version: str
    cell_capacity: int
    symbol_capacity: int
    cell_count: int
    source_provenance: str
    cell_ids: object
    cell_mask: object
    scope_valid: object
    scope_error_code: object
    completion_events: object
    pre_structural_lengths: object
    final_symbols: object
    final_lengths: object
    pools_after: object
    requested_symbols: object
    append_count: object
    last_replication_symbols: object
    last_effective_error_rate: object
    cumulative_proofreading_atp_after: object
    substitution_events: object
    structural_event_counts: object
    material_delta_symbols: object
    new_genome_lesions: object
    replication_cycle_deltas: object
    topology_sequence_deltas: object
    topology_symbol_deltas: object
    replication_active_after: object
    replication_template_lesions_after: object
    replication_fractional_after: object
    genome_count_after: object
    genome_material_symbols_after: object
    genome_lesion_mean_after: object

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = (
                _clone_array(value)
                if item.name in _COMPLETION_PLAN_ARRAY_FIELDS
                else copy.deepcopy(value)
            )
        return A4CompletionMutationPlan(**values)

    def to_numpy(self):
        if _is_tensor(self.final_symbols):
            _validate_completion_mutation_plan_metadata(self)
        else:
            return validate_a4_completion_mutation_plan(self).clone()
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _COMPLETION_PLAN_ARRAY_FIELDS:
                value = _host_array(value)
                if item.name in _COMPLETION_PLAN_UINT8_FIELDS:
                    value = value.astype(np.uint8, copy=False)
                elif item.name in _COMPLETION_PLAN_INT64_FIELDS:
                    value = value.astype(np.int64, copy=False)
                elif item.name in _COMPLETION_PLAN_BOOL_FIELDS:
                    value = value.astype(bool, copy=False)
                else:
                    value = value.astype(np.float64, copy=False)
            else:
                value = copy.deepcopy(value)
            values[item.name] = value
        return validate_a4_completion_mutation_plan(
            A4CompletionMutationPlan(**values)
        )

    def data_ptrs(self):
        if not all(_is_tensor(getattr(self, name))
                   for name in _COMPLETION_PLAN_ARRAY_FIELDS):
            raise TypeError(
                'data_ptrs requires a Torch-backed completion mutation plan'
            )
        return {
            name: int(getattr(self, name).data_ptr())
            for name in _COMPLETION_PLAN_ARRAY_FIELDS
        }

    def state_dict(self):
        return {
            item.name: (
                _clone_array(getattr(self, item.name))
                if item.name in _COMPLETION_PLAN_ARRAY_FIELDS
                else copy.deepcopy(getattr(self, item.name))
            )
            for item in fields(self)
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


def _completion_mutation_tape_array_digest(tape):
    """Hash host arrays without granting serialization or commit authority."""
    digest = hashlib.sha256()
    for name in _COMPLETION_TAPE_ARRAY_FIELDS:
        value = np.asarray(getattr(tape, name))
        digest.update(name.encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _completion_mutation_tape_scalar_metadata(tape):
    before = _canonical_pcg64_state(
        tape.rng_before_state, 'rng_before_state',
    )
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    return (
        str(tape.schema_version), int(tape.cell_capacity),
        int(tape.append_capacity), int(tape.mutation_capacity),
        int(tape.cell_count), str(tape.source_provenance),
        str(tape.dt_hex), str(tape.config_sha256),
        str(tape.schedule_sha256), _sha256_json(before),
        _sha256_json(after), str(tape._expected_host_array_sha256),
    )


def _make_completion_mutation_rng_tape(**values):
    expected_host_array_sha256 = values.pop(
        '_expected_host_array_sha256', None,
    )
    resident_expected_arrays = values.pop(
        '_resident_expected_arrays', None,
    )
    tape = A4CompletionMutationRngTape(
        _factory_token=_COMPLETION_TAPE_FACTORY_TOKEN, **values
    )
    if _is_tensor(tape.append_uniform_draws):
        if not _is_lower_hex_digest(expected_host_array_sha256):
            raise a4.A4SchemaError(
                'resident completion-mutation tape lacks host content digest'
            )
        tape._expected_host_array_sha256 = expected_host_array_sha256
        tape._resident_data_ptrs = tape.data_ptrs()
        tape._resident_versions = {
            name: int(getattr(tape, name)._version)
            for name in _COMPLETION_TAPE_ARRAY_FIELDS
        }
        if resident_expected_arrays is None:
            resident_expected_arrays = {
                name: getattr(tape, name).clone()
                for name in _COMPLETION_TAPE_ARRAY_FIELDS
            }
        elif set(resident_expected_arrays) != set(
                _COMPLETION_TAPE_ARRAY_FIELDS):
            raise a4.A4SchemaError(
                'resident completion tape expected-array set is invalid'
            )
        tape._resident_expected_arrays = {
            name: resident_expected_arrays[name].clone()
            for name in _COMPLETION_TAPE_ARRAY_FIELDS
        }
        for name in _COMPLETION_TAPE_ARRAY_FIELDS:
            expected = tape._resident_expected_arrays[name]
            actual = getattr(tape, name)
            if (not _is_tensor(expected)
                    or expected.device != actual.device
                    or expected.dtype != actual.dtype
                    or tuple(expected.shape) != tuple(actual.shape)):
                raise a4.A4SchemaError(
                    'resident completion tape expected %s is invalid' % name
                )
        tape._resident_expected_data_ptrs = {
            name: int(value.data_ptr())
            for name, value in tape._resident_expected_arrays.items()
        }
        tape._resident_expected_versions = {
            name: int(value._version)
            for name, value in tape._resident_expected_arrays.items()
        }
    else:
        tape._host_array_sha256 = _completion_mutation_tape_array_digest(tape)
        if (expected_host_array_sha256 is not None
                and expected_host_array_sha256
                != tape._host_array_sha256):
            raise a4.A4SchemaError(
                'completion-mutation tape content changed during readback'
            )
        tape._expected_host_array_sha256 = tape._host_array_sha256
    tape._scalar_metadata = _completion_mutation_tape_scalar_metadata(tape)
    return tape


def _validate_completion_mutation_tape_metadata(tape):
    if (not isinstance(tape, A4CompletionMutationRngTape)
            or tape._factory_token is not _COMPLETION_TAPE_FACTORY_TOKEN):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape must come from the private factory'
        )
    if tape.schema_version != COMPLETION_MUTATION_RNG_TAPE_SCHEMA_VERSION:
        raise a4.A4SchemaError('completion-mutation RNG tape schema mismatch')
    for name in (
            'cell_capacity', 'append_capacity', 'mutation_capacity',
            'cell_count'):
        value = getattr(tape, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise a4.A4SchemaError('%s must be integer' % name)
    C = int(tape.cell_capacity)
    W = int(tape.append_capacity)
    F = int(tape.mutation_capacity)
    N = int(tape.cell_count)
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    if (C <= 0 or W <= 0 or F != int(a4.g2.MAX_GENOME_LENGTH)
            or W > F or N < 0 or N > C):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape capacities/count are invalid'
        )
    if (not _is_lower_hex_digest(tape.source_provenance)
            or not _is_lower_hex_digest(tape.config_sha256)
            or not _is_lower_hex_digest(tape.schedule_sha256)):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape provenance digest is invalid'
        )
    try:
        parsed_dt = float.fromhex(tape.dt_hex)
    except Exception as exc:
        raise a4.A4SchemaError(
            'completion-mutation RNG tape dt hex is invalid'
        ) from exc
    if not math.isfinite(parsed_dt) or parsed_dt < 0.0:
        raise a4.A4SchemaError(
            'completion-mutation RNG tape dt is outside supported range'
        )
    _canonical_pcg64_state(tape.rng_before_state, 'rng_before_state')
    _canonical_pcg64_state(tape.rng_after_state, 'rng_after_state')
    kinds = set()
    devices = set()
    for name in _COMPLETION_TAPE_ARRAY_FIELDS:
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
            'mixed completion-mutation tape backend/device is forbidden'
        )
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_COMPLETION_TAPE_UINT8_FIELDS, np.dtype(np.uint8),
         getattr(torch, 'uint8', None)),
        (_COMPLETION_TAPE_INT64_FIELDS, np.dtype(np.int64),
         getattr(torch, 'int64', None)),
        (_COMPLETION_TAPE_BOOL_FIELDS, np.dtype(bool),
         getattr(torch, 'bool', None)),
        (_COMPLETION_TAPE_FLOAT64_FIELDS, np.dtype(np.float64),
         getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(tape, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    shapes = {
        'cell_ids': (C,), 'cell_mask': (C,),
        'append_draw_mask': (C, W),
        'append_uniform_draws': (C, W),
        'replacement_raw': (C, W), 'replacement_mask': (C, W),
        'append_count': (C,), 'substitution_count': (C,),
        'effective_error': (C,), 'nucleotide_budget_symbols': (C,),
        'pre_structural_lengths': (C,),
        'post_structural_lengths': (C,),
        'material_delta_symbols': (C,),
        'structural_event_counts': (C, STRUCTURAL_EVENT_COUNT),
        'threshold_draw_mask': (C, STRUCTURAL_EVENT_COUNT),
        'threshold_uniform_draws': (C, STRUCTURAL_EVENT_COUNT),
        'threshold_hit_mask': (C, STRUCTURAL_EVENT_COUNT),
        'insertion_count': (C,), 'insertion_position': (C,),
        'insertion_symbols': (C, STRUCTURAL_SHORT_EDIT_MAX),
        'deletion_count': (C,), 'deletion_position': (C,),
        'duplication_gene_ordinal': (C,),
        'duplication_source_start': (C,),
        'duplication_position': (C,),
        'inversion_left': (C,), 'inversion_right': (C,),
        'transposition_count': (C,), 'transposition_start': (C,),
        'transposition_position': (C,), 'padding_count': (C,),
        'padding_symbols': (C, minimum),
    }
    for name, shape in shapes.items():
        if tuple(getattr(tape, name).shape) != shape:
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return backend


def _require_completion_mutation_rng_tape(tape):
    backend = _validate_completion_mutation_tape_metadata(tape)
    if getattr(tape, '_scalar_metadata', None) != (
            _completion_mutation_tape_scalar_metadata(tape)):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape scalar metadata changed'
        )
    if not _is_lower_hex_digest(
            getattr(tape, '_expected_host_array_sha256', None)):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape content digest is invalid'
        )
    if backend == 'torch':
        if (getattr(tape, '_resident_data_ptrs', None) != tape.data_ptrs()
                or getattr(tape, '_resident_versions', None) != {
                    name: int(getattr(tape, name)._version)
                    for name in _COMPLETION_TAPE_ARRAY_FIELDS
                }):
            raise a4.A4SchemaError(
                'resident completion-mutation RNG tape changed after upload'
            )
        expected = getattr(tape, '_resident_expected_arrays', None)
        if (not isinstance(expected, dict)
                or set(expected) != set(_COMPLETION_TAPE_ARRAY_FIELDS)
                or getattr(tape, '_resident_expected_data_ptrs', None) != {
                    name: int(expected[name].data_ptr())
                    for name in _COMPLETION_TAPE_ARRAY_FIELDS
                }
                or getattr(tape, '_resident_expected_versions', None) != {
                    name: int(expected[name]._version)
                    for name in _COMPLETION_TAPE_ARRAY_FIELDS
                }):
            raise a4.A4SchemaError(
                'resident completion-mutation expected values changed'
            )
    elif getattr(tape, '_host_array_sha256', None) != (
            _completion_mutation_tape_array_digest(tape)):
        raise a4.A4SchemaError(
            'host completion-mutation RNG tape changed after creation'
        )
    return tape


def _completion_mutation_tape_resident_unchanged(tape):
    """Exact device scalar guarding `.data` writes before semantic use."""
    _require_completion_mutation_rng_tape(tape)
    if not _is_tensor(tape.append_uniform_draws):
        raise a4.A4SchemaError(
            'resident content comparison requires a Torch tape'
        )
    unchanged = torch.ones(
        (), dtype=torch.bool, device=tape.append_uniform_draws.device,
    )
    for name in _COMPLETION_TAPE_ARRAY_FIELDS:
        unchanged = unchanged & torch.all(
            getattr(tape, name) == tape._resident_expected_arrays[name]
        )
    return unchanged


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
        'completion_events': (C,),
        'completed_symbols': (C, W),
        'completed_lengths': (C,),
        'new_genome_lesions': (C,),
        'replication_cycle_deltas': (C,),
        'topology_sequence_deltas': (C,),
        'topology_symbol_deltas': (C,),
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
            'paid elongation row is outside A4.6a scope: %s' %
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
    completion = raw['completion_events']
    if (np.any(completion & ~raw['scope_valid'])
            or np.any(completion & start)
            or np.any(raw['completed_lengths'][completion] <= 0)
            or np.any(raw['completed_lengths'][completion] > W)
            or np.any(raw['completed_lengths'][~completion] != 0)
            or np.any(raw['new_genome_lesions'][completion] < 0.0)
            or np.any(raw['new_genome_lesions'][~completion] != 0.0)
            or not np.array_equal(
                raw['replication_cycle_deltas'],
                completion.astype(np.int64),
            )
            or not np.array_equal(
                raw['topology_sequence_deltas'],
                -completion.astype(np.int64),
            )
            or not np.array_equal(
                raw['topology_symbol_deltas'],
                np.where(
                    completion,
                    -raw['completed_lengths'] + raw['append_count'],
                    0,
                ),
            )
            or np.any(raw['replication_fractional_after'][completion] != 0.0)
            or np.any(raw['append_count'][completion] <= 0)
            or np.any(raw['append_count'][completion]
                      > raw['completed_lengths'][completion])):
        raise a4.A4SchemaError('completion topology metadata is invalid')
    if not np.array_equal(
            raw['last_replication_symbols'], raw['append_count']):
        raise a4.A4SchemaError('last replication count differs from append count')
    if (not np.isfinite(raw['pools_after']).all()
            or not np.isfinite(raw['replication_fractional_after']).all()
            or not np.isfinite(raw['last_effective_error_rate']).all()
            or not np.isfinite(raw['new_genome_lesions']).all()
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
        completed_length = int(raw['completed_lengths'][ci])
        completed = raw['completed_symbols'][ci, :completed_length]
        if np.any(completed >= a4.ALPHABET_SIZE):
            raise a4.A4SchemaError(
                'completed symbol outside frozen alphabet'
            )
        if np.any(raw['completed_symbols'][ci, completed_length:] != 0):
            raise a4.A4SchemaError('completed-symbol tail is not zero')
        if bool(completion[ci]):
            appended = int(raw['append_count'][ci])
            if not np.array_equal(
                    completed[completed_length - appended:],
                    raw['append_symbols'][ci, :appended]):
                raise a4.A4SchemaError(
                    'completed suffix differs from paid append'
                )
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


def _validate_completion_mutation_plan_metadata(plan):
    if not isinstance(plan, A4CompletionMutationPlan):
        raise a4.A4SchemaError('expected A4CompletionMutationPlan')
    if plan.schema_version != COMPLETION_MUTATION_PLAN_SCHEMA_VERSION:
        raise a4.A4SchemaError('completion mutation plan schema mismatch')
    for name in ('cell_capacity', 'symbol_capacity', 'cell_count'):
        value = getattr(plan, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise a4.A4SchemaError('%s must be an integer scalar' % name)
    C = int(plan.cell_capacity)
    W = int(plan.symbol_capacity)
    N = int(plan.cell_count)
    if C <= 0 or W <= 0 or W > int(a4.g2.MAX_GENOME_LENGTH):
        raise a4.A4SchemaError(
            'completion mutation plan capacities are invalid'
        )
    if N < 0 or N > C:
        raise a4.A4SchemaError(
            'completion mutation plan cell count is invalid'
        )
    if not _is_lower_hex_digest(plan.source_provenance):
        raise a4.A4SchemaError(
            'completion mutation plan source provenance is invalid'
        )
    kinds = set()
    devices = set()
    for name in _COMPLETION_PLAN_ARRAY_FIELDS:
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
            'mixed completion mutation plan backend/device is forbidden'
        )
    backend = next(iter(kinds))
    for names, numpy_dtype, torch_dtype in (
        (_COMPLETION_PLAN_UINT8_FIELDS, np.dtype(np.uint8),
         getattr(torch, 'uint8', None)),
        (_COMPLETION_PLAN_INT64_FIELDS, np.dtype(np.int64),
         getattr(torch, 'int64', None)),
        (_COMPLETION_PLAN_BOOL_FIELDS, np.dtype(bool),
         getattr(torch, 'bool', None)),
        (_COMPLETION_PLAN_FLOAT64_FIELDS, np.dtype(np.float64),
         getattr(torch, 'float64', None)),
    ):
        for name in names:
            expected = torch_dtype if backend == 'torch' else numpy_dtype
            if getattr(plan, name).dtype != expected:
                raise a4.A4SchemaError('%s has noncanonical dtype' % name)
    shapes = {
        'cell_ids': (C,), 'cell_mask': (C,), 'scope_valid': (C,),
        'scope_error_code': (C,), 'completion_events': (C,),
        'pre_structural_lengths': (C,), 'final_symbols': (C, W),
        'final_lengths': (C,),
        'pools_after': (C, int(a4.a3.POOL_COUNT)),
        'requested_symbols': (C,), 'append_count': (C,),
        'last_replication_symbols': (C,),
        'last_effective_error_rate': (C,),
        'cumulative_proofreading_atp_after': (C,),
        'substitution_events': (C,),
        'structural_event_counts': (C, STRUCTURAL_EVENT_COUNT),
        'material_delta_symbols': (C,), 'new_genome_lesions': (C,),
        'replication_cycle_deltas': (C,),
        'topology_sequence_deltas': (C,),
        'topology_symbol_deltas': (C,),
        'replication_active_after': (C,),
        'replication_template_lesions_after': (C,),
        'replication_fractional_after': (C,),
        'genome_count_after': (C,),
        'genome_material_symbols_after': (C,),
        'genome_lesion_mean_after': (C,),
    }
    for name, shape in shapes.items():
        if tuple(getattr(plan, name).shape) != shape:
            raise a4.A4SchemaError('%s shape mismatch' % name)
    return backend


def validate_a4_completion_mutation_plan(plan):
    """Validate a host final-polymer descriptor before any future commit."""
    if _validate_completion_mutation_plan_metadata(plan) != 'numpy':
        raise a4.A4SchemaError(
            'full completion mutation plan validation requires NumPy'
        )
    raw = {
        name: np.asarray(getattr(plan, name))
        for name in _COMPLETION_PLAN_ARRAY_FIELDS
    }
    C = int(plan.cell_capacity)
    W = int(plan.symbol_capacity)
    N = int(plan.cell_count)
    used = np.arange(C) < N
    if not np.array_equal(raw['cell_mask'], used):
        raise a4.A4SchemaError(
            'completion mutation plan mask is not a true prefix'
        )
    if (np.any(raw['cell_ids'][:N] < 0)
            or len(set(int(value) for value in raw['cell_ids'][:N])) != N
            or np.any(raw['cell_ids'][N:] != -1)):
        raise a4.A4SchemaError(
            'completion mutation plan cell identity is invalid'
        )
    if (not np.all(raw['scope_valid'][:N])
            or np.any(raw['scope_error_code'][:N] != SCOPE_OK)):
        if np.any(raw['scope_error_code'][:N] == SCOPE_CAPACITY):
            raise a4.A4CapacityError(
                'completion mutation plan exceeds fixed capacity'
            )
        raise A4ReplicationScopeError(
            'completion mutation plan is outside resident scope: %s' %
            [int(value) for value in raw['scope_error_code'][:N]]
        )
    if not np.all(raw['completion_events'][:N]):
        raise a4.A4SchemaError(
            'completion mutation plan lacks a used completion event'
        )
    pre = raw['pre_structural_lengths'][:N]
    final = raw['final_lengths'][:N]
    delta = raw['material_delta_symbols'][:N]
    append = raw['append_count'][:N]
    requested = raw['requested_symbols'][:N]
    minimum_final = np.minimum(
        pre, np.full_like(pre, int(a4.g2.MIN_GENOME_LENGTH)),
    )
    if (np.any(pre <= 0) or np.any(pre > W)
            or np.any(final < minimum_final)
            or np.any(final > W)
            or not np.array_equal(final - pre, delta)
            or np.any(append <= 0) or np.any(append > pre)
            or np.any(append > requested)
            or not np.array_equal(
                raw['last_replication_symbols'][:N], append,
            )
            or np.any(raw['substitution_events'][:N] < 0)
            or np.any(raw['substitution_events'][:N] > append)
            or np.any(raw['structural_event_counts'][:N] < 0)):
        raise a4.A4SchemaError(
            'completion mutation counts or lengths are invalid'
        )
    if (not np.array_equal(
            raw['replication_cycle_deltas'][:N],
            np.ones((N,), dtype=np.int64),
        ) or not np.array_equal(
            raw['topology_sequence_deltas'][:N],
            -np.ones((N,), dtype=np.int64),
        ) or not np.array_equal(
            raw['topology_symbol_deltas'][:N],
            -pre + append + delta,
        ) or np.any(raw['replication_active_after'][:N])
            or np.any(raw['replication_template_lesions_after'][:N] != 0.0)
            or np.any(raw['replication_fractional_after'][:N] != 0.0)
            or np.any(raw['genome_count_after'][:N] <= 0)
            or np.any(raw['genome_material_symbols_after'][:N] <= 0)):
        raise a4.A4SchemaError(
            'completion mutation lifecycle descriptor is invalid'
        )
    if (not np.isfinite(raw['pools_after']).all()
            or not np.isfinite(raw['last_effective_error_rate']).all()
            or not np.isfinite(
                raw['cumulative_proofreading_atp_after']).all()
            or not np.isfinite(raw['new_genome_lesions']).all()
            or not np.isfinite(raw['genome_lesion_mean_after']).all()
            or np.any(raw['pools_after'][:N, a4.a3.POOL_NUCLEOTIDE] < 0.0)
            or np.any(raw['last_effective_error_rate'][:N] < 0.0)
            or np.any(raw['cumulative_proofreading_atp_after'][:N] < 0.0)
            or np.any(raw['new_genome_lesions'][:N] < 0.0)
            or np.any(raw['genome_lesion_mean_after'][:N] < 0.0)):
        raise a4.A4SchemaError(
            'completion mutation chemistry contains invalid values'
        )
    for ci in range(N):
        length = int(final[ci])
        if (np.any(raw['final_symbols'][ci, :length]
                   >= int(a4.g2.ALPHABET_SIZE))
                or np.any(raw['final_symbols'][ci, length:] != 0)):
            raise a4.A4SchemaError(
                'completion mutation final polymer is invalid'
            )
    for name in _COMPLETION_PLAN_ARRAY_FIELDS:
        if name in ('cell_ids', 'cell_mask'):
            continue
        tail = raw[name][N:]
        if tail.size and np.any(tail != 0):
            raise a4.A4SchemaError(
                '%s unused cell tail is not zero' % name
            )
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
        completion_events=np.zeros((C,), dtype=bool),
        completed_symbols=np.zeros((C, W), dtype=np.uint8),
        completed_lengths=np.zeros((C,), dtype=np.int64),
        new_genome_lesions=np.zeros((C,), dtype=np.float64),
        replication_cycle_deltas=np.zeros((C,), dtype=np.int64),
        topology_sequence_deltas=np.zeros((C,), dtype=np.int64),
        topology_symbol_deltas=np.zeros((C,), dtype=np.int64),
    )


def _paid_replication_elongation_numpy(
        binding, dt, config, allow_template_start=False,
        completion_only=False):
    """Literal NumPy reference for the bounded Formal066 continuation."""
    if completion_only and allow_template_start:
        raise A4ReplicationScopeError(
            'A4.6a completion requires a pre-existing active template'
        )
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
    total_completion_events = 0
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
            if completion_only:
                completed_length = int(len(partial) + copied)
                result.completed_symbols[ci, :len(partial)] = partial
                result.completed_symbols[
                    ci, len(partial):completed_length
                ] = result.append_symbols[ci, :copied]
                result.completed_lengths[ci] = completed_length
                result.completion_events[ci] = True
                result.replication_cycle_deltas[ci] = 1
                result.topology_sequence_deltas[ci] = -1
                result.topology_symbol_deltas[ci] = (
                    -completed_length + copied
                )
                # Preserve the exact grouping of the frozen 0.4 completion.
                inherited_lesion = float(template_lesion) * (
                    0.28 + 0.22 * (1.0 - proof_fraction)
                )
                new_lesion = (
                    inherited_lesion
                    + float(result.last_effective_error_rate[ci])
                    * completed_length * 0.06
                )
                result.new_genome_lesions[ci] = float(new_lesion)
                result.replication_fractional_after[ci] = 0.0
                result.scope_valid[ci] = True
                total_completion_events += 1
            else:
                result.scope_error_code[ci] = SCOPE_COMPLETION
        else:
            if completion_only:
                result.scope_error_code[ci] = SCOPE_NONCOMPLETION
            else:
                result.scope_valid[ci] = True
                if start_event:
                    result.template_start_events[ci] = True
                    result.selected_template_indices[ci] = selected_index
                    result.template_storage_symbols[ci] = len(template)
                    total_template_storage += len(template)
                    total_start_events += 1
    future_sequence_count = (
        int(ragged.sequence_count) + 2 * total_start_events
        + int(np.sum(result.topology_sequence_deltas[:N], dtype=np.int64))
    )
    if completion_only:
        future_symbol_count = (
            int(ragged.symbol_count)
            + int(np.sum(
                result.topology_symbol_deltas[:N], dtype=np.int64,
            ))
        )
    else:
        future_symbol_count = (
            int(ragged.symbol_count) + total_template_storage
            + total_appended
        )
    future_lesion_count = int(ragged.lesion_count) + total_completion_events
    if (future_sequence_count
            > int(ragged.sequence_capacity)):
        raise a4.A4CapacityError(
            'replication transaction exceeds sequence capacity'
        )
    if future_symbol_count > int(ragged.symbol_capacity):
        raise a4.A4CapacityError(
            'replication transaction exceeds symbol capacity'
        )
    if future_lesion_count > int(ragged.sequence_capacity):
        raise a4.A4CapacityError(
            'replication completion exceeds lesion capacity'
        )
    return validate_a4_paid_elongation_plan(result)


def paid_replication_elongation_numpy(binding, dt, config):
    """Public deterministic A4.4b path; inactive start stays CPU authority."""
    return _paid_replication_elongation_numpy(
        binding, dt, config, allow_template_start=False,
    )


def _paid_replication_completion_numpy(binding, dt, config):
    """Mutation-free all-row completion descriptor; never a commit."""
    return _paid_replication_elongation_numpy(
        binding, dt, config, allow_template_start=False,
        completion_only=True,
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
        binding, dt, config, allow_template_start=False,
        completion_only=False):
    """Fixed-shape resident plan with no scalar readback or dynamic output."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    if completion_only and allow_template_start:
        raise A4ReplicationScopeError(
            'A4.6a completion requires a pre-existing active template'
        )
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
    if completion_only:
        error = torch.where(
            state.cell_mask & (error == SCOPE_OK) & (~completion),
            torch.full_like(error, SCOPE_NONCOMPLETION), error,
        )
    else:
        error = torch.where(
            completion, torch.full_like(error, SCOPE_COMPLETION), error,
        )
    completion_before_capacity = (
        completion & (error == SCOPE_OK) & bool(completion_only)
    )
    topology_sequence_deltas = torch.where(
        completion_before_capacity,
        torch.full_like(append_count, -1),
        torch.zeros_like(append_count),
    )
    topology_symbol_deltas = torch.where(
        completion_before_capacity,
        -template_length + append_count,
        torch.zeros_like(append_count),
    )
    start_before_capacity = start_candidate & (error == SCOPE_OK)
    sequence_capacity_overflow = (
        int(ragged.sequence_count)
        + 2 * torch.sum(start_before_capacity.to(torch.int64))
        + torch.sum(topology_sequence_deltas)
        > int(ragged.sequence_capacity)
    )
    if completion_only:
        symbol_capacity_overflow = (
            int(ragged.symbol_count) + torch.sum(topology_symbol_deltas)
            > int(ragged.symbol_capacity)
        )
    else:
        symbol_capacity_overflow = (
            int(ragged.symbol_count) + torch.sum(append_count)
            + torch.sum(torch.where(
                start_before_capacity, template_length,
                torch.zeros_like(template_length),
            )) > int(ragged.symbol_capacity)
        )
    lesion_capacity_overflow = (
        int(ragged.lesion_count)
        + torch.sum(completion_before_capacity.to(torch.int64))
        > int(ragged.sequence_capacity)
    )
    capacity_overflow = (
        sequence_capacity_overflow | symbol_capacity_overflow
        | lesion_capacity_overflow
    )
    error = torch.where(
        state.cell_mask & capacity_overflow,
        torch.full_like(error, SCOPE_CAPACITY), error,
    )
    completion_events = completion_before_capacity & (error == SCOPE_OK)
    completion_batch_failure = (
        torch.any(state.cell_mask & (~completion_events))
        if completion_only else
        torch.zeros((), dtype=torch.bool, device=device)
    )
    rollback = state.cell_mask & completion_batch_failure
    error = torch.where(
        rollback & (error == SCOPE_OK),
        torch.full_like(error, SCOPE_NONCOMPLETION), error,
    )
    scope_valid = (
        state.cell_mask & (error == SCOPE_OK) & (~completion_batch_failure)
    )
    completion_events = completion_events & scope_valid
    template_start_events = start_candidate & scope_valid

    completed_rank = symbol_rank[None, :]
    partial_indices = copy_start[:, None] + completed_rank
    safe_partial_indices = torch.clamp(
        partial_indices, min=0, max=int(ragged.symbol_capacity) - 1,
    )
    partial_symbols = ragged.symbols[safe_partial_indices]
    append_rank = torch.clamp(
        completed_rank - copy_length[:, None], min=0, max=W - 1,
    )
    appended_symbols = torch.gather(append_symbols, 1, append_rank)
    completed_symbols = torch.where(
        completed_rank < copy_length[:, None],
        partial_symbols, appended_symbols,
    )
    completed_symbols = torch.where(
        completion_events[:, None]
        & (completed_rank < template_length[:, None]),
        completed_symbols, torch.zeros_like(completed_symbols),
    )
    completed_lengths = torch.where(
        completion_events, template_length, torch.zeros_like(template_length),
    )
    inherited_lesion = template_lesion * (
        0.28 + 0.22 * (1.0 - proof_fraction)
    )
    new_genome_lesions = (
        inherited_lesion
        + effective_error * template_length.to(dtype) * 0.06
    )
    new_genome_lesions = torch.where(
        completion_events, new_genome_lesions,
        torch.zeros_like(new_genome_lesions),
    )
    requested_out = torch.where(
        rollback, torch.zeros_like(requested),
        torch.where(
            completed_kinetics, requested, torch.zeros_like(requested),
        ),
    )
    append_symbols_out = torch.where(
        rollback[:, None], torch.zeros_like(append_symbols), append_symbols,
    )
    append_count_out = torch.where(
        rollback, torch.zeros_like(append_count), append_count,
    )
    pools_after_out = torch.where(
        rollback[:, None], state.pools, pools_after,
    )
    fractional_after_out = torch.where(
        completion_events, torch.zeros_like(fractional_after),
        torch.where(
            completed_kinetics, fractional_after,
            torch.zeros_like(fractional_after),
        ),
    )
    fractional_after_out = torch.where(
        rollback, torch.zeros_like(fractional_after_out),
        fractional_after_out,
    )
    effective_error_out = torch.where(
        rollback, torch.zeros_like(effective_error),
        torch.where(
            completed_kinetics, effective_error,
            torch.zeros_like(effective_error),
        ),
    )
    cumulative_out = torch.where(
        rollback, state.cumulative_proofreading_atp,
        cumulative_proofreading_atp_after,
    )
    topology_sequence_deltas = torch.where(
        completion_events, topology_sequence_deltas,
        torch.zeros_like(topology_sequence_deltas),
    )
    topology_symbol_deltas = torch.where(
        completion_events, topology_symbol_deltas,
        torch.zeros_like(topology_symbol_deltas),
    )

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
        requested_symbols=requested_out,
        append_symbols=append_symbols_out,
        append_count=append_count_out,
        pools_after=torch.where(
            state.cell_mask[:, None], pools_after_out,
            torch.zeros_like(pools_after_out),
        ),
        replication_fractional_after=fractional_after_out,
        last_replication_symbols=append_count_out.clone(),
        last_effective_error_rate=effective_error_out,
        cumulative_proofreading_atp_after=cumulative_out,
        substitution_events=torch.zeros_like(append_count_out),
        template_start_events=template_start_events,
        selected_template_indices=torch.where(
            template_start_events, torch.zeros_like(append_count_out),
            torch.full_like(append_count_out, -1),
        ),
        template_storage_symbols=torch.where(
            template_start_events, template_length,
            torch.zeros_like(template_length),
        ),
        completion_events=completion_events,
        completed_symbols=completed_symbols,
        completed_lengths=completed_lengths,
        new_genome_lesions=new_genome_lesions,
        replication_cycle_deltas=completion_events.to(torch.int64),
        topology_sequence_deltas=topology_sequence_deltas,
        topology_symbol_deltas=topology_symbol_deltas,
    )
    _validate_plan_metadata(plan)
    return plan


def paid_replication_elongation_torch(binding, dt, config):
    """Public deterministic A4.4b path; inactive start stays CPU authority."""
    return _paid_replication_elongation_torch(
        binding, dt, config, allow_template_start=False,
    )


def _paid_replication_completion_torch(binding, dt, config):
    """Resident all-row completion descriptor with no scalar readback."""
    return _paid_replication_elongation_torch(
        binding, dt, config, allow_template_start=False,
        completion_only=True,
    )


def _empty_completion_mutation_arrays(plan):
    C = int(plan.cell_capacity)
    W = int(plan.append_capacity)
    F = int(a4.g2.MAX_GENOME_LENGTH)
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    values = {
        'cell_ids': np.asarray(plan.cell_ids, dtype=np.int64).copy(),
        'cell_mask': np.asarray(plan.cell_mask, dtype=bool).copy(),
        'append_draw_mask': (
            np.arange(W, dtype=np.int64)[None, :]
            < np.asarray(plan.append_count, dtype=np.int64)[:, None]
        ),
        'append_uniform_draws': np.zeros((C, W), dtype=np.float64),
        'replacement_raw': np.zeros((C, W), dtype=np.uint8),
        'replacement_mask': np.zeros((C, W), dtype=bool),
        'append_count': np.asarray(plan.append_count, dtype=np.int64).copy(),
        'substitution_count': np.zeros((C,), dtype=np.int64),
        'effective_error': np.asarray(
            plan.last_effective_error_rate, dtype=np.float64,
        ).copy(),
        'nucleotide_budget_symbols': np.zeros((C,), dtype=np.int64),
        'pre_structural_lengths': np.asarray(
            plan.completed_lengths, dtype=np.int64,
        ).copy(),
        'post_structural_lengths': np.zeros((C,), dtype=np.int64),
        'material_delta_symbols': np.zeros((C,), dtype=np.int64),
        'structural_event_counts': np.zeros(
            (C, STRUCTURAL_EVENT_COUNT), dtype=np.int64,
        ),
        'threshold_draw_mask': np.zeros(
            (C, STRUCTURAL_EVENT_COUNT), dtype=bool,
        ),
        'threshold_uniform_draws': np.zeros(
            (C, STRUCTURAL_EVENT_COUNT), dtype=np.float64,
        ),
        'threshold_hit_mask': np.zeros(
            (C, STRUCTURAL_EVENT_COUNT), dtype=bool,
        ),
        'insertion_count': np.zeros((C,), dtype=np.int64),
        'insertion_position': np.full((C,), -1, dtype=np.int64),
        'insertion_symbols': np.zeros(
            (C, STRUCTURAL_SHORT_EDIT_MAX), dtype=np.uint8,
        ),
        'deletion_count': np.zeros((C,), dtype=np.int64),
        'deletion_position': np.full((C,), -1, dtype=np.int64),
        'duplication_gene_ordinal': np.full((C,), -1, dtype=np.int64),
        'duplication_source_start': np.full((C,), -1, dtype=np.int64),
        'duplication_position': np.full((C,), -1, dtype=np.int64),
        'inversion_left': np.full((C,), -1, dtype=np.int64),
        'inversion_right': np.full((C,), -1, dtype=np.int64),
        'transposition_count': np.zeros((C,), dtype=np.int64),
        'transposition_start': np.full((C,), -1, dtype=np.int64),
        'transposition_position': np.full((C,), -1, dtype=np.int64),
        'padding_count': np.zeros((C,), dtype=np.int64),
        'padding_symbols': np.zeros((C, minimum), dtype=np.uint8),
    }
    if F <= 0:
        raise a4.A4SchemaError('frozen mutation capacity must be positive')
    return values


def _completion_mutation_schedule_digest(tape):
    N = int(tape.cell_count)
    raw = {
        name: np.asarray(getattr(tape, name))
        for name in _COMPLETION_TAPE_ARRAY_FIELDS
    }
    payload = {
        'schema': str(tape.schema_version),
        'source': str(tape.source_provenance),
        'dt_hex': str(tape.dt_hex),
        'config_sha256': str(tape.config_sha256),
        'cell_ids': [int(value) for value in raw['cell_ids'][:N]],
        'append_count': [int(value) for value in raw['append_count'][:N]],
        'effective_error_hex': [
            float(value).hex() for value in raw['effective_error'][:N]
        ],
        'nucleotide_budget_symbols': [
            int(value) for value in raw['nucleotide_budget_symbols'][:N]
        ],
        'pre_structural_lengths': [
            int(value) for value in raw['pre_structural_lengths'][:N]
        ],
        'post_structural_lengths': [
            int(value) for value in raw['post_structural_lengths'][:N]
        ],
        'material_delta_symbols': [
            int(value) for value in raw['material_delta_symbols'][:N]
        ],
        'structural_event_counts': [
            [int(value) for value in row]
            for row in raw['structural_event_counts'][:N]
        ],
        'threshold_draw_mask': [
            [bool(value) for value in row]
            for row in raw['threshold_draw_mask'][:N]
        ],
        'threshold_hit_mask': [
            [bool(value) for value in row]
            for row in raw['threshold_hit_mask'][:N]
        ],
    }
    return _sha256_json(payload)


def _record_structural_mutation(
        sequence, generator, options, nucleotide_budget, arrays, ci):
    """Literal high-level PCG64 calls for frozen ``g2.mutate_sequence``."""
    original = np.asarray(sequence, dtype=np.uint8).copy()
    candidate = original.copy()
    F = int(a4.g2.MAX_GENOME_LENGTH)
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    alphabet = int(a4.g2.ALPHABET_SIZE)
    gene_span = int(a4.g2.GENE_SPAN)
    structural = float(options['structural_rate'])
    events = arrays['structural_event_counts'][ci]

    def threshold(operation):
        uniform = float(generator.random())
        limit = structural * float(STRUCTURAL_THRESHOLD_FACTORS[operation])
        if _numpy_fp64_comparison_boundary(uniform, limit):
            raise A4ReplicationScopeError(
                'structural RNG comparison is fp64-ambiguous'
            )
        arrays['threshold_draw_mask'][ci, operation] = True
        arrays['threshold_uniform_draws'][ci, operation] = uniform
        hit = uniform < limit
        arrays['threshold_hit_mask'][ci, operation] = hit
        return hit

    if options['variable_length'] and len(candidate) > 0:
        if (threshold(STRUCTURAL_INSERTION)
                and len(candidate) < F):
            count = int(generator.integers(
                1, min(STRUCTURAL_SHORT_EDIT_MAX, F - len(candidate)) + 1,
            ))
            position = int(generator.integers(0, len(candidate) + 1))
            inserted = np.asarray(generator.integers(
                0, alphabet, count, dtype=np.uint8,
            ), dtype=np.uint8)
            arrays['insertion_count'][ci] = count
            arrays['insertion_position'][ci] = position
            arrays['insertion_symbols'][ci, :count] = inserted
            candidate = np.concatenate([
                candidate[:position], inserted, candidate[position:],
            ])
            events[STRUCTURAL_INSERTION] += count
        if (threshold(STRUCTURAL_DELETION)
                and len(candidate) > minimum):
            count = int(generator.integers(
                1,
                min(STRUCTURAL_SHORT_EDIT_MAX, len(candidate) - minimum) + 1,
            ))
            position = int(generator.integers(
                0, len(candidate) - count + 1,
            ))
            arrays['deletion_count'][ci] = count
            arrays['deletion_position'][ci] = position
            candidate = np.concatenate([
                candidate[:position], candidate[position + count:],
            ])
            events[STRUCTURAL_DELETION] += count
        if options['gene_duplication']:
            duplication_hit = threshold(STRUCTURAL_DUPLICATION)
            if duplication_hit and len(candidate) + gene_span <= F:
                genes = a4.g2.parse_genes(candidate)
                if genes:
                    ordinal = int(generator.integers(0, len(genes)))
                    start = int(genes[ordinal]['start'])
                    fragment = candidate[start:start + gene_span].copy()
                    position = int(generator.integers(
                        0, len(candidate) + 1,
                    ))
                    arrays['duplication_gene_ordinal'][ci] = ordinal
                    arrays['duplication_source_start'][ci] = start
                    arrays['duplication_position'][ci] = position
                    candidate = np.concatenate([
                        candidate[:position], fragment, candidate[position:],
                    ])
                    events[STRUCTURAL_DUPLICATION] += int(len(fragment))

    if len(candidate) >= 4 and threshold(STRUCTURAL_INVERSION):
        left = int(generator.integers(0, len(candidate) - 2))
        right = int(generator.integers(
            left + 2, min(len(candidate), left + 28) + 1,
        ))
        arrays['inversion_left'][ci] = left
        arrays['inversion_right'][ci] = right
        candidate[left:right] = candidate[left:right][::-1]
        events[STRUCTURAL_INVERSION] += 1

    if len(candidate) >= 8 and threshold(STRUCTURAL_TRANSPOSITION):
        count = int(generator.integers(
            2, min(12, len(candidate) // 3) + 1,
        ))
        start = int(generator.integers(0, len(candidate) - count + 1))
        fragment = candidate[start:start + count].copy()
        remainder = np.concatenate([
            candidate[:start], candidate[start + count:],
        ])
        position = int(generator.integers(0, len(remainder) + 1))
        arrays['transposition_count'][ci] = count
        arrays['transposition_start'][ci] = start
        arrays['transposition_position'][ci] = position
        candidate = np.concatenate([
            remainder[:position], fragment, remainder[position:],
        ])
        events[STRUCTURAL_TRANSPOSITION] += 1

    if len(candidate) < minimum:
        count = minimum - len(candidate)
        padding = np.asarray(generator.integers(
            0, alphabet, count, dtype=np.uint8,
        ), dtype=np.uint8)
        arrays['padding_count'][ci] = count
        arrays['padding_symbols'][ci, :count] = padding
        candidate = np.concatenate([candidate, padding])
    if len(candidate) > F:
        candidate = candidate[:F]
    delta = int(len(candidate) - len(original))
    if delta > int(nucleotide_budget):
        candidate = candidate[:len(original) + int(nucleotide_budget)]
        delta = int(len(candidate) - len(original))
    arrays['post_structural_lengths'][ci] = len(candidate)
    arrays['material_delta_symbols'][ci] = delta
    return candidate.astype(np.uint8), delta


def _replay_completion_mutation_rng(
        binding, plan, config, options, rng_state_before):
    """Build one combined cell-interleaved tape and cross-check g2 directly."""
    arrays = _empty_completion_mutation_arrays(plan)
    before = _canonical_pcg64_state(
        rng_state_before, 'rng_state_before',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    N = int(plan.cell_count)
    structural_config = copy.deepcopy(config)
    structural_config.mutation_rate = 0.0
    mass = float(a4.g2.MONOMER_MASS)
    for ci in range(N):
        count = int(arrays['append_count'][ci])
        error = float(arrays['effective_error'][ci])
        appended = np.asarray(
            plan.append_symbols[ci, :count], dtype=np.uint8,
        ).copy()
        for rank in range(count):
            uniform = float(generator.random())
            if _numpy_fp64_comparison_boundary(uniform, error):
                raise A4ReplicationScopeError(
                    'completion substitution comparison is fp64-ambiguous'
                )
            arrays['append_uniform_draws'][ci, rank] = uniform
            if uniform < error:
                arrays['replacement_mask'][ci, rank] = True
                raw = int(generator.integers(0, 7))
                arrays['replacement_raw'][ci, rank] = raw
                old = int(appended[rank])
                appended[rank] = (
                    raw + (1 if raw >= old else 0)
                ) % int(a4.g2.ALPHABET_SIZE)
                arrays['substitution_count'][ci] += 1
        template, partial, start_event, _, _ = _numpy_template_copy(
            binding, ci, allow_template_start=False,
        )
        if (template is None or bool(start_event)
                or len(partial) + count != int(
                    arrays['pre_structural_lengths'][ci])):
            raise A4ReplicationScopeError(
                'completion-mutation source differs from completion plan'
            )
        pre_structural = np.concatenate([partial, appended]).astype(
            np.uint8, copy=False,
        )
        nucleotide_after = float(
            plan.pools_after[ci, a4.a3.POOL_NUCLEOTIDE]
        )
        if not math.isfinite(nucleotide_after) or nucleotide_after < 0.0:
            raise A4ReplicationScopeError(
                'completion-mutation nucleotide pool is invalid'
            )
        # The frozen mutator can add at most one full frozen sequence, so a
        # larger finite integer budget is outcome-equivalent to that bound.
        # Preserve the frozen fp64 division and Python ``int`` conversion,
        # however: a nonfinite quotient remains outside this bounded slice
        # rather than giving A4 behavior where the CPU reference raises.
        mutation_capacity = int(a4.g2.MAX_GENOME_LENGTH)
        raw_budget = nucleotide_after / mass
        if not math.isfinite(raw_budget):
            raise A4ReplicationScopeError(
                'completion-mutation nucleotide budget is nonfinite'
            )
        budget = min(int(raw_budget), mutation_capacity)
        arrays['nucleotide_budget_symbols'][ci] = budget
        structural_before = copy.deepcopy(generator.bit_generator.state)
        candidate, delta = _record_structural_mutation(
            pre_structural, generator, options, budget, arrays, ci,
        )

        # Independent frozen-source oracle: the literal recorder above must have
        # exactly the same high-level calls, final polymer, counters, and cache.
        oracle = np.random.Generator(np.random.PCG64())
        oracle.bit_generator.state = copy.deepcopy(structural_before)
        expected, expected_delta, expected_events = a4.g2.mutate_sequence(
            pre_structural, oracle, structural_config,
            nucleotide_budget=budget,
        )
        expected_counts = np.asarray([
            int(expected_events[name]) for name in STRUCTURAL_EVENT_NAMES
        ], dtype=np.int64)
        if (not np.array_equal(candidate, expected)
                or int(delta) != int(expected_delta)
                or not np.array_equal(
                    arrays['structural_event_counts'][ci], expected_counts,
                )
                or oracle.bit_generator.state != generator.bit_generator.state):
            raise A4ReplicationScopeError(
                'completion structural RNG replay differs from frozen g2'
            )
        if len(candidate) > int(plan.append_capacity):
            raise a4.A4CapacityError(
                'completion mutation exceeds declared sequence capacity'
            )
    future_symbol_count = (
        int(binding.ragged.symbol_count)
        + int(np.sum(
            np.asarray(plan.topology_symbol_deltas[:N], dtype=np.int64),
            dtype=np.int64,
        ))
        + int(np.sum(
            arrays['material_delta_symbols'][:N], dtype=np.int64,
        ))
    )
    if future_symbol_count > int(binding.ragged.symbol_capacity):
        raise a4.A4CapacityError(
            'completion mutation exceeds aggregate symbol capacity'
        )
    return arrays, copy.deepcopy(generator.bit_generator.state)


def prepare_completion_mutation_rng_tape(
        binding, dt, config, rng_state_before):
    """Prepare an immutable combined tape without advancing a live RNG."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError(
            'completion-mutation tape preparation requires a NumPy binding'
        )
    deterministic, options, config_sha256 = _completion_mutation_config(
        config,
    )
    dt = _strict_dt(dt)
    plan = _paid_replication_completion_numpy(
        binding, dt, deterministic,
    )
    before = _canonical_pcg64_state(
        rng_state_before, 'rng_state_before',
    )
    arrays, after = _replay_completion_mutation_rng(
        binding, plan, config, options, before,
    )
    values = {
        'schema_version': COMPLETION_MUTATION_RNG_TAPE_SCHEMA_VERSION,
        'cell_capacity': int(plan.cell_capacity),
        'append_capacity': int(plan.append_capacity),
        'mutation_capacity': int(a4.g2.MAX_GENOME_LENGTH),
        'cell_count': int(plan.cell_count),
        'source_provenance': str(plan.source_provenance),
        'dt_hex': float(dt).hex(),
        'config_sha256': str(config_sha256),
        'schedule_sha256': '0' * 64,
        'rng_before_state': copy.deepcopy(before),
        'rng_after_state': copy.deepcopy(after),
    }
    values.update(arrays)
    draft = _make_completion_mutation_rng_tape(**values)
    values['schedule_sha256'] = _completion_mutation_schedule_digest(draft)
    tape = _make_completion_mutation_rng_tape(**values)
    return validate_a4_completion_mutation_rng_tape(
        tape, binding, dt, config,
    )


def validate_a4_completion_mutation_rng_tape(
        tape, binding, dt, config):
    """Replay every high-level call against its attested NumPy source."""
    _require_completion_mutation_rng_tape(tape)
    if _validate_completion_mutation_tape_metadata(tape) != 'numpy':
        raise a4.A4SchemaError(
            'full completion-mutation tape validation requires NumPy'
        )
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError(
            'completion-mutation tape validation requires a NumPy binding'
        )
    deterministic, options, config_sha256 = _completion_mutation_config(
        config,
    )
    dt = _strict_dt(dt)
    plan = _paid_replication_completion_numpy(
        binding, dt, deterministic,
    )
    if (tape.source_provenance != str(plan.source_provenance)
            or tape.dt_hex != float(dt).hex()
            or tape.config_sha256 != str(config_sha256)
            or int(tape.cell_capacity) != int(plan.cell_capacity)
            or int(tape.append_capacity) != int(plan.append_capacity)
            or int(tape.mutation_capacity)
            != int(a4.g2.MAX_GENOME_LENGTH)
            or int(tape.cell_count) != int(plan.cell_count)):
        raise A4ReplicationScopeError(
            'completion-mutation tape does not bind this source/config/dt'
        )
    expected_arrays, expected_after = _replay_completion_mutation_rng(
        binding, plan, config, options, tape.rng_before_state,
    )
    for name in _COMPLETION_TAPE_ARRAY_FIELDS:
        if not np.array_equal(
                np.asarray(getattr(tape, name)), expected_arrays[name]):
            raise a4.A4SchemaError(
                'completion-mutation RNG tape %s differs from replay' % name
            )
    after = _canonical_pcg64_state(
        tape.rng_after_state, 'rng_after_state',
    )
    if expected_after != after:
        raise a4.A4SchemaError(
            'completion-mutation RNG tape after-state differs from replay'
        )
    if tape.schedule_sha256 != _completion_mutation_schedule_digest(tape):
        raise a4.A4SchemaError(
            'completion-mutation RNG tape schedule provenance differs'
        )
    return tape


def _require_completion_mutation_tape_binding(
        tape, binding, dt, config_sha256):
    """Check host-known identity without reading any resident tensor."""
    _require_completion_mutation_rng_tape(tape)
    dt = _strict_dt(dt)
    if (tape.source_provenance != str(binding.state.source_provenance)
            or tape.dt_hex != float(dt).hex()
            or tape.config_sha256 != str(config_sha256)
            or int(tape.cell_capacity) != int(binding.state.cell_capacity)
            or int(tape.append_capacity)
            != int(binding.ragged.max_sequence_symbols)
            or int(tape.mutation_capacity)
            != int(a4.g2.MAX_GENOME_LENGTH)
            or int(tape.cell_count) != int(binding.state.cell_count)):
        raise A4ReplicationScopeError(
            'completion-mutation tape does not bind this resident source'
        )
    return tape


def _numpy_apply_completion_mutation_tape(binding, base, tape, config):
    """Apply recorded operations without calling the frozen mutator."""
    C = int(base.cell_capacity)
    W = int(base.append_capacity)
    F = int(tape.mutation_capacity)
    N = int(base.cell_count)
    mass = float(a4.g2.MONOMER_MASS)
    alphabet = int(a4.g2.ALPHABET_SIZE)
    gene_span = int(a4.g2.GENE_SPAN)
    final_symbols = np.zeros((C, W), dtype=np.uint8)
    final_lengths = np.zeros((C,), dtype=np.int64)
    pools_after = np.asarray(base.pools_after, dtype=np.float64).copy()
    new_lesions = np.zeros((C,), dtype=np.float64)
    lesion_means_after = np.zeros((C,), dtype=np.float64)
    _, _, flags = _supported_config(config)
    specs_by_cell = binding.cache.materialize_gene_specs_host()

    for ci in range(N):
        pre_length = int(base.completed_lengths[ci])
        append_count = int(base.append_count[ci])
        candidate = np.asarray(
            base.completed_symbols[ci, :pre_length], dtype=np.uint8,
        ).copy()
        append_start = pre_length - append_count
        for rank in range(append_count):
            if not bool(tape.replacement_mask[ci, rank]):
                continue
            index = append_start + rank
            old = int(candidate[index])
            raw = int(tape.replacement_raw[ci, rank])
            candidate[index] = (
                raw + (1 if raw >= old else 0)
            ) % alphabet

        count = int(tape.insertion_count[ci])
        if count:
            position = int(tape.insertion_position[ci])
            inserted = np.asarray(
                tape.insertion_symbols[ci, :count], dtype=np.uint8,
            )
            candidate = np.concatenate([
                candidate[:position], inserted, candidate[position:],
            ])
        count = int(tape.deletion_count[ci])
        if count:
            position = int(tape.deletion_position[ci])
            candidate = np.concatenate([
                candidate[:position], candidate[position + count:],
            ])
        source = int(tape.duplication_source_start[ci])
        if source >= 0:
            position = int(tape.duplication_position[ci])
            fragment = candidate[source:source + gene_span].copy()
            candidate = np.concatenate([
                candidate[:position], fragment, candidate[position:],
            ])
        left = int(tape.inversion_left[ci])
        if left >= 0:
            right = int(tape.inversion_right[ci])
            candidate[left:right] = candidate[left:right][::-1]
        count = int(tape.transposition_count[ci])
        if count:
            start = int(tape.transposition_start[ci])
            position = int(tape.transposition_position[ci])
            fragment = candidate[start:start + count].copy()
            remainder = np.concatenate([
                candidate[:start], candidate[start + count:],
            ])
            candidate = np.concatenate([
                remainder[:position], fragment, remainder[position:],
            ])
        count = int(tape.padding_count[ci])
        if count:
            candidate = np.concatenate([
                candidate,
                np.asarray(tape.padding_symbols[ci, :count], dtype=np.uint8),
            ])
        if len(candidate) > F:
            candidate = candidate[:F]
        budget = int(tape.nucleotide_budget_symbols[ci])
        delta = int(len(candidate) - pre_length)
        if delta > budget:
            candidate = candidate[:pre_length + budget]
            delta = int(len(candidate) - pre_length)
        if (len(candidate) != int(tape.post_structural_lengths[ci])
                or delta != int(tape.material_delta_symbols[ci])):
            raise a4.A4SchemaError(
                'completion mutation application differs from tape lengths'
            )
        if len(candidate) > W:
            raise a4.A4CapacityError(
                'completion mutation final polymer exceeds output capacity'
            )
        final_symbols[ci, :len(candidate)] = candidate
        final_lengths[ci] = len(candidate)
        if delta > 0:
            pools_after[ci, a4.a3.POOL_NUCLEOTIDE] -= delta * mass
        elif delta < 0:
            pools_after[ci, a4.a3.POOL_NUCLEOTIDE] += (-delta) * mass
        effective_error = float(base.last_effective_error_rate[ci])
        if flags['proofreading']:
            metrics = a4._numpy_translation_cell_metrics(binding.state, ci)
            proofreading = _numpy_raw_repair(
                binding, ci, specs_by_cell[ci],
                a4.a3.REPAIR_PROOFREADING, metrics['aggregate'],
            )
            proof_fraction = proofreading / (0.75 + proofreading)
        else:
            proof_fraction = 0.0
        inherited = float(
            binding.ragged.replication_template_lesions[ci]
        ) * (0.28 + 0.22 * (1.0 - proof_fraction))
        new_lesions[ci] = inherited + (
            effective_error * len(candidate) * 0.06
        )
        lesion_first = int(binding.ragged.lesion_offsets[ci])
        lesion_last = int(binding.ragged.lesion_offsets[ci + 1])
        combined_lesions = np.concatenate((
            np.asarray(
                binding.ragged.genome_lesions[
                    lesion_first:lesion_last
                ],
                dtype=np.float64,
            ),
            np.asarray([new_lesions[ci]], dtype=np.float64),
        ))
        lesion_means_after[ci] = float(np.mean(
            combined_lesions, dtype=np.float64,
        ))

    result = A4CompletionMutationPlan(
        schema_version=COMPLETION_MUTATION_PLAN_SCHEMA_VERSION,
        cell_capacity=C,
        symbol_capacity=W,
        cell_count=N,
        source_provenance=str(base.source_provenance),
        cell_ids=np.asarray(base.cell_ids, dtype=np.int64).copy(),
        cell_mask=np.asarray(base.cell_mask, dtype=bool).copy(),
        scope_valid=np.asarray(base.scope_valid, dtype=bool).copy(),
        scope_error_code=np.asarray(
            base.scope_error_code, dtype=np.int64,
        ).copy(),
        completion_events=np.asarray(
            base.completion_events, dtype=bool,
        ).copy(),
        pre_structural_lengths=np.asarray(
            base.completed_lengths, dtype=np.int64,
        ).copy(),
        final_symbols=final_symbols,
        final_lengths=final_lengths,
        pools_after=pools_after,
        requested_symbols=np.asarray(
            base.requested_symbols, dtype=np.int64,
        ).copy(),
        append_count=np.asarray(base.append_count, dtype=np.int64).copy(),
        last_replication_symbols=np.asarray(
            base.last_replication_symbols, dtype=np.int64,
        ).copy(),
        last_effective_error_rate=np.asarray(
            base.last_effective_error_rate, dtype=np.float64,
        ).copy(),
        cumulative_proofreading_atp_after=np.asarray(
            base.cumulative_proofreading_atp_after, dtype=np.float64,
        ).copy(),
        substitution_events=np.asarray(
            tape.substitution_count, dtype=np.int64,
        ).copy(),
        structural_event_counts=np.asarray(
            tape.structural_event_counts, dtype=np.int64,
        ).copy(),
        material_delta_symbols=np.asarray(
            tape.material_delta_symbols, dtype=np.int64,
        ).copy(),
        new_genome_lesions=new_lesions,
        replication_cycle_deltas=np.asarray(
            base.replication_cycle_deltas, dtype=np.int64,
        ).copy(),
        topology_sequence_deltas=np.asarray(
            base.topology_sequence_deltas, dtype=np.int64,
        ).copy(),
        topology_symbol_deltas=(
            np.asarray(base.topology_symbol_deltas, dtype=np.int64)
            + np.asarray(tape.material_delta_symbols, dtype=np.int64)
        ),
        replication_active_after=np.zeros((C,), dtype=bool),
        replication_template_lesions_after=np.zeros(
            (C,), dtype=np.float64,
        ),
        replication_fractional_after=np.zeros((C,), dtype=np.float64),
        genome_count_after=(
            np.asarray(binding.state.genome_count, dtype=np.int64)
            + np.asarray(base.completion_events, dtype=np.int64)
        ),
        genome_material_symbols_after=(
            np.asarray(
                binding.state.genome_material_symbols, dtype=np.int64,
            )
            + np.asarray(base.append_count, dtype=np.int64)
            + np.asarray(tape.material_delta_symbols, dtype=np.int64)
        ),
        genome_lesion_mean_after=lesion_means_after,
    )
    return validate_a4_completion_mutation_plan(result)


def paid_replication_completion_mutation_numpy(
        binding, dt, config, tape):
    """Apply a fully replayed host tape to a pure NumPy completion plan."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        raise a4.A4SchemaError(
            'NumPy completion mutation requires a NumPy binding'
        )
    deterministic, _, config_sha256 = _completion_mutation_config(config)
    tape = validate_a4_completion_mutation_rng_tape(
        tape, binding, dt, config,
    )
    _require_completion_mutation_tape_binding(
        tape, binding, dt, config_sha256,
    )
    base = _paid_replication_completion_numpy(
        binding, dt, deterministic,
    )
    return _numpy_apply_completion_mutation_tape(
        binding, base, tape, deterministic,
    )


def _torch_completion_proof_fraction(binding, enabled):
    """Recompute the frozen proofreading fraction without host readback."""
    state = binding.state
    cache = binding.cache
    C = int(state.cell_capacity)
    K = int(cache.entry_capacity)
    device = state.pools.device
    dtype = state.pools.dtype
    if not enabled:
        return torch.zeros((C,), dtype=dtype, device=device)
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
    _, _, _, _, proofreading_signal, _ = _torch_replicase(
        binding, valid_entries, fingerprints, role, parameter, localisation,
        promoter, efficiency,
    )
    return proofreading_signal / (0.75 + proofreading_signal)


def paid_replication_completion_mutation_torch(
        binding, dt, config, tape):
    """Apply one attested tape with fixed resident CPU/CUDA operations."""
    if torch is None:
        raise RuntimeError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    if not _is_tensor(binding.state.pools):
        raise a4.A4SchemaError(
            'Torch completion mutation requires a Torch binding'
        )
    deterministic, options, config_sha256 = _completion_mutation_config(
        config,
    )
    _, _, deterministic_flags = _supported_config(deterministic)
    _require_completion_mutation_tape_binding(
        tape, binding, dt, config_sha256,
    )
    device = binding.state.pools.device
    if (not _is_tensor(tape.append_uniform_draws)
            or tape.append_uniform_draws.device != device):
        raise a4.A4SchemaError(
            'completion mutation tape must share the resident device'
        )
    resident_unchanged = _completion_mutation_tape_resident_unchanged(tape)
    base = _paid_replication_completion_torch(
        binding, dt, deterministic,
    )
    C = int(base.cell_capacity)
    W = int(base.append_capacity)
    F = int(tape.mutation_capacity)
    minimum = int(a4.g2.MIN_GENOME_LENGTH)
    alphabet = int(a4.g2.ALPHABET_SIZE)
    gene_span = int(a4.g2.GENE_SPAN)
    dtype = binding.state.pools.dtype
    used = base.cell_mask
    rank_w = torch.arange(W, dtype=torch.int64, device=device)
    rank_f = torch.arange(F, dtype=torch.int64, device=device)
    rank_grid = rank_f[None, :].expand(C, F)
    semantic_ok = (
        (tape.cell_ids == base.cell_ids)
        & (tape.cell_mask == base.cell_mask)
        & (tape.append_count == base.append_count)
        & (tape.pre_structural_lengths == base.completed_lengths)
        & (tape.effective_error == base.last_effective_error_rate)
    )
    base_failure = torch.any(
        used & ((~base.scope_valid) | (~base.completion_events))
    )
    boundary_rows = torch.zeros((C,), dtype=torch.bool, device=device)

    expected_append_mask = (
        rank_w[None, :] < base.append_count[:, None]
    )
    append_uniform_valid = (
        torch.isfinite(tape.append_uniform_draws)
        & (tape.append_uniform_draws >= 0.0)
        & (tape.append_uniform_draws < 1.0)
    )
    append_boundary = expected_append_mask & (
        _torch_fp64_comparison_boundary(
            tape.append_uniform_draws,
            base.last_effective_error_rate[:, None],
        )
    )
    expected_replacement = expected_append_mask & (
        tape.append_uniform_draws
        < base.last_effective_error_rate[:, None]
    )
    append_semantic = (
        torch.all(tape.append_draw_mask == expected_append_mask, dim=1)
        & torch.all(
            torch.where(
                expected_append_mask,
                append_uniform_valid,
                tape.append_uniform_draws == 0.0,
            ),
            dim=1,
        )
        & torch.all(
            tape.replacement_mask == expected_replacement, dim=1,
        )
        & torch.all(
            torch.where(
                expected_replacement,
                tape.replacement_raw < 7,
                tape.replacement_raw == 0,
            ),
            dim=1,
        )
        & (
            tape.substitution_count
            == torch.sum(expected_replacement.to(torch.int64), dim=1)
        )
    )
    semantic_ok = semantic_ok & append_semantic
    boundary_rows = boundary_rows | torch.any(append_boundary, dim=1)

    raw_budget = (
        base.pools_after[:, a4.a3.POOL_NUCLEOTIDE]
        / float(a4.g2.MONOMER_MASS)
    )
    finite_budget = torch.isfinite(raw_budget) & (raw_budget >= 0.0)
    # The binding-aware host tape owns Python's frozen float64 division/int
    # result.  Re-truncating the quotient on CUDA can move an exact CPU integer
    # one ulp below its boundary and invent a different material budget.  Check
    # the resident quotient against the attested integer bucket, allowing only
    # the registered fp64 comparison band, then apply the tape integer.
    computed_budget = tape.nucleotide_budget_symbols
    budget_f64 = computed_budget.to(dtype)
    budget_in_range = (computed_budget >= 0) & (computed_budget <= F)
    lower_matches = (raw_budget >= budget_f64) | (
        _torch_fp64_comparison_boundary(raw_budget, budget_f64)
    )
    upper_f64 = budget_f64 + 1.0
    upper_matches = (computed_budget == F) | (raw_budget < upper_f64) | (
        _torch_fp64_comparison_boundary(raw_budget, upper_f64)
    )
    semantic_ok = (
        semantic_ok & finite_budget & budget_in_range
        & lower_matches & upper_matches
    )

    if F > W:
        candidate = torch.cat([
            base.completed_symbols,
            torch.zeros(
                (C, F - W), dtype=torch.uint8, device=device,
            ),
        ], dim=1)
    else:
        candidate = base.completed_symbols[:, :F].clone()
    pre_length = base.completed_lengths
    append_start = pre_length - base.append_count
    old_append = base.append_symbols.to(torch.int64)
    replacement = tape.replacement_raw.to(torch.int64)
    replacement = torch.remainder(
        replacement + (replacement >= old_append).to(torch.int64),
        alphabet,
    ).to(torch.uint8)
    substituted_append = torch.where(
        tape.replacement_mask, replacement, base.append_symbols,
    )
    append_source = torch.clamp(
        rank_grid - append_start[:, None], min=0, max=W - 1,
    )
    appended_full = torch.gather(substituted_append, 1, append_source)
    candidate = torch.where(
        rank_grid < append_start[:, None], candidate, appended_full,
    )
    candidate = torch.where(
        rank_grid < pre_length[:, None], candidate,
        torch.zeros_like(candidate),
    )
    length = pre_length.clone()
    outer_structural = (
        used & bool(options['variable_length']) & (pre_length > 0)
    )

    def threshold(operation, draw_expected):
        uniform = tape.threshold_uniform_draws[:, operation]
        limit = float(options['structural_rate']) * float(
            STRUCTURAL_THRESHOLD_FACTORS[operation]
        )
        uniform_valid = (
            torch.isfinite(uniform) & (uniform >= 0.0) & (uniform < 1.0)
        )
        expected_hit = draw_expected & (uniform < limit)
        valid = (
            (tape.threshold_draw_mask[:, operation] == draw_expected)
            & (
                tape.threshold_hit_mask[:, operation] == expected_hit
            )
            & torch.where(draw_expected, uniform_valid, uniform == 0.0)
        )
        boundary = draw_expected & _torch_fp64_comparison_boundary(
            uniform, limit,
        )
        return expected_hit, valid, boundary

    insertion_hit, valid, boundary = threshold(
        STRUCTURAL_INSERTION, outer_structural,
    )
    semantic_ok = semantic_ok & valid
    boundary_rows = boundary_rows | boundary
    insertion_execute = insertion_hit & (length < F)
    insertion_limit = torch.clamp(F - length, min=0, max=5)
    insertion_count = tape.insertion_count
    insertion_position = tape.insertion_position
    insertion_fields_ok = torch.where(
        insertion_execute,
        (insertion_count >= 1)
        & (insertion_count <= insertion_limit)
        & (insertion_position >= 0)
        & (insertion_position <= length),
        (insertion_count == 0) & (insertion_position == -1),
    )
    short_rank = torch.arange(
        STRUCTURAL_SHORT_EDIT_MAX, dtype=torch.int64, device=device,
    )
    insertion_payload_ok = torch.all(torch.where(
        short_rank[None, :] < insertion_count[:, None],
        tape.insertion_symbols < alphabet,
        tape.insertion_symbols == 0,
    ), dim=1)
    semantic_ok = semantic_ok & insertion_fields_ok & insertion_payload_ok
    insertion_apply = insertion_execute & insertion_fields_ok
    safe_count = torch.where(
        insertion_apply, insertion_count, torch.zeros_like(insertion_count),
    )
    safe_position = torch.clamp(insertion_position, min=0, max=F)
    old_source = torch.where(
        rank_grid < safe_position[:, None], rank_grid,
        rank_grid - safe_count[:, None],
    )
    old_source = torch.clamp(old_source, min=0, max=F - 1)
    shifted = torch.gather(candidate, 1, old_source)
    inserted_rank = torch.clamp(
        rank_grid - safe_position[:, None],
        min=0, max=STRUCTURAL_SHORT_EDIT_MAX - 1,
    )
    inserted = torch.gather(tape.insertion_symbols, 1, inserted_rank)
    in_insert = (
        (rank_grid >= safe_position[:, None])
        & (rank_grid < safe_position[:, None] + safe_count[:, None])
    )
    inserted_candidate = torch.where(in_insert, inserted, shifted)
    length_after = length + safe_count
    candidate = torch.where(
        insertion_apply[:, None]
        & (rank_grid < length_after[:, None]),
        inserted_candidate, candidate,
    )
    length = length_after

    deletion_hit, valid, boundary = threshold(
        STRUCTURAL_DELETION, outer_structural,
    )
    semantic_ok = semantic_ok & valid
    boundary_rows = boundary_rows | boundary
    deletion_execute = deletion_hit & (length > minimum)
    deletion_limit = torch.clamp(length - minimum, min=0, max=5)
    deletion_count = tape.deletion_count
    deletion_position = tape.deletion_position
    deletion_fields_ok = torch.where(
        deletion_execute,
        (deletion_count >= 1)
        & (deletion_count <= deletion_limit)
        & (deletion_position >= 0)
        & (deletion_position + deletion_count <= length),
        (deletion_count == 0) & (deletion_position == -1),
    )
    semantic_ok = semantic_ok & deletion_fields_ok
    deletion_apply = deletion_execute & deletion_fields_ok
    safe_count_delete = torch.where(
        deletion_apply, deletion_count, torch.zeros_like(deletion_count),
    )
    safe_position_delete = torch.clamp(
        deletion_position, min=0, max=F - 1,
    )
    delete_source = torch.where(
        rank_grid < safe_position_delete[:, None], rank_grid,
        rank_grid + safe_count_delete[:, None],
    )
    delete_source = torch.clamp(delete_source, min=0, max=F - 1)
    deleted_candidate = torch.gather(candidate, 1, delete_source)
    length_after = length - safe_count_delete
    candidate = torch.where(
        deletion_apply[:, None],
        torch.where(
            rank_grid < length_after[:, None], deleted_candidate,
            torch.zeros_like(candidate),
        ),
        candidate,
    )
    length = length_after

    duplication_draw = outer_structural & bool(options['gene_duplication'])
    duplication_hit, valid, boundary = threshold(
        STRUCTURAL_DUPLICATION, duplication_draw,
    )
    semantic_ok = semantic_ok & valid
    boundary_rows = boundary_rows | boundary
    gene_count = torch.zeros((C,), dtype=torch.int64, device=device)
    selected_start = torch.full(
        (C,), -1, dtype=torch.int64, device=device,
    )
    next_allowed = torch.zeros((C,), dtype=torch.int64, device=device)
    ordinal = tape.duplication_gene_ordinal
    start_marker = tuple(int(value) for value in a4.g2.START_MARKER)
    stop_marker = tuple(int(value) for value in a4.g2.STOP_MARKER)
    stop_offset = 2 + int(a4.g2.GENE_PAYLOAD)
    for position in range(max(0, F - gene_span + 1)):
        parse_here = (
            (position >= next_allowed)
            & (position + gene_span <= length)
            & (candidate[:, position] == start_marker[0])
            & (candidate[:, position + 1] == start_marker[1])
            & (candidate[:, position + stop_offset] == stop_marker[0])
            & (candidate[:, position + stop_offset + 1] == stop_marker[1])
        )
        selected_start = torch.where(
            parse_here & (gene_count == ordinal),
            torch.full_like(selected_start, position), selected_start,
        )
        gene_count = gene_count + parse_here.to(torch.int64)
        next_allowed = torch.where(
            parse_here,
            torch.full_like(next_allowed, position + gene_span),
            next_allowed,
        )
    duplication_execute = (
        duplication_hit & (length + gene_span <= F) & (gene_count > 0)
    )
    duplication_position = tape.duplication_position
    duplication_fields_ok = torch.where(
        duplication_execute,
        (ordinal >= 0) & (ordinal < gene_count)
        & (tape.duplication_source_start == selected_start)
        & (duplication_position >= 0)
        & (duplication_position <= length),
        (ordinal == -1)
        & (tape.duplication_source_start == -1)
        & (duplication_position == -1),
    )
    semantic_ok = semantic_ok & duplication_fields_ok
    duplication_apply = duplication_execute & duplication_fields_ok
    safe_dup_position = torch.clamp(
        duplication_position, min=0, max=F,
    )
    safe_dup_source = torch.clamp(selected_start, min=0, max=F - gene_span)
    dup_old_source = torch.where(
        rank_grid < safe_dup_position[:, None], rank_grid,
        rank_grid - gene_span,
    )
    dup_old_source = torch.clamp(dup_old_source, min=0, max=F - 1)
    dup_shifted = torch.gather(candidate, 1, dup_old_source)
    dup_fragment_source = torch.clamp(
        safe_dup_source[:, None]
        + rank_grid - safe_dup_position[:, None],
        min=0, max=F - 1,
    )
    dup_fragment = torch.gather(candidate, 1, dup_fragment_source)
    in_duplication = (
        (rank_grid >= safe_dup_position[:, None])
        & (rank_grid < safe_dup_position[:, None] + gene_span)
    )
    duplicated_candidate = torch.where(
        in_duplication, dup_fragment, dup_shifted,
    )
    length_after = length + duplication_apply.to(torch.int64) * gene_span
    candidate = torch.where(
        duplication_apply[:, None]
        & (rank_grid < length_after[:, None]),
        duplicated_candidate, candidate,
    )
    length = length_after

    inversion_draw = used & (length >= 4)
    inversion_hit, valid, boundary = threshold(
        STRUCTURAL_INVERSION, inversion_draw,
    )
    semantic_ok = semantic_ok & valid
    boundary_rows = boundary_rows | boundary
    inversion_left = tape.inversion_left
    inversion_right = tape.inversion_right
    inversion_fields_ok = torch.where(
        inversion_hit,
        (inversion_left >= 0)
        & (inversion_left <= length - 3)
        & (inversion_right >= inversion_left + 2)
        & (inversion_right <= length)
        & (inversion_right <= inversion_left + 28),
        (inversion_left == -1) & (inversion_right == -1),
    )
    semantic_ok = semantic_ok & inversion_fields_ok
    inversion_apply = inversion_hit & inversion_fields_ok
    safe_left = torch.clamp(inversion_left, min=0, max=F - 1)
    safe_right = torch.clamp(inversion_right, min=0, max=F)
    inversion_source = torch.where(
        (rank_grid >= safe_left[:, None])
        & (rank_grid < safe_right[:, None]),
        safe_left[:, None] + safe_right[:, None] - 1 - rank_grid,
        rank_grid,
    )
    inversion_source = torch.clamp(
        inversion_source, min=0, max=F - 1,
    )
    inverted_candidate = torch.gather(candidate, 1, inversion_source)
    candidate = torch.where(
        inversion_apply[:, None], inverted_candidate, candidate,
    )

    transposition_draw = used & (length >= 8)
    transposition_hit, valid, boundary = threshold(
        STRUCTURAL_TRANSPOSITION, transposition_draw,
    )
    semantic_ok = semantic_ok & valid
    boundary_rows = boundary_rows | boundary
    transposition_count = tape.transposition_count
    transposition_start = tape.transposition_start
    transposition_position = tape.transposition_position
    transposition_limit = torch.minimum(
        torch.full_like(length, 12), torch.div(length, 3, rounding_mode='floor'),
    )
    transposition_fields_ok = torch.where(
        transposition_hit,
        (transposition_count >= 2)
        & (transposition_count <= transposition_limit)
        & (transposition_start >= 0)
        & (transposition_start + transposition_count <= length)
        & (transposition_position >= 0)
        & (transposition_position <= length - transposition_count),
        (transposition_count == 0)
        & (transposition_start == -1)
        & (transposition_position == -1),
    )
    semantic_ok = semantic_ok & transposition_fields_ok
    transposition_apply = transposition_hit & transposition_fields_ok
    safe_trans_count = torch.where(
        transposition_apply, transposition_count,
        torch.zeros_like(transposition_count),
    )
    safe_trans_start = torch.clamp(
        transposition_start, min=0, max=F - 1,
    )
    safe_trans_position = torch.clamp(
        transposition_position, min=0, max=F,
    )
    remainder_source = torch.where(
        rank_grid < safe_trans_start[:, None], rank_grid,
        rank_grid + safe_trans_count[:, None],
    )
    remainder_source = torch.clamp(
        remainder_source, min=0, max=F - 1,
    )
    remainder = torch.gather(candidate, 1, remainder_source)
    final_remainder_source = torch.where(
        rank_grid < safe_trans_position[:, None], rank_grid,
        rank_grid - safe_trans_count[:, None],
    )
    final_remainder_source = torch.clamp(
        final_remainder_source, min=0, max=F - 1,
    )
    moved_remainder = torch.gather(remainder, 1, final_remainder_source)
    fragment_source = torch.clamp(
        safe_trans_start[:, None]
        + rank_grid - safe_trans_position[:, None],
        min=0, max=F - 1,
    )
    moved_fragment = torch.gather(candidate, 1, fragment_source)
    in_fragment = (
        (rank_grid >= safe_trans_position[:, None])
        & (
            rank_grid
            < safe_trans_position[:, None] + safe_trans_count[:, None]
        )
    )
    transposed_candidate = torch.where(
        in_fragment, moved_fragment, moved_remainder,
    )
    candidate = torch.where(
        transposition_apply[:, None], transposed_candidate, candidate,
    )

    expected_padding = torch.clamp(minimum - length, min=0)
    padding_semantic = tape.padding_count == expected_padding
    padding_rank = torch.arange(
        minimum, dtype=torch.int64, device=device,
    )
    padding_payload_ok = torch.all(torch.where(
        padding_rank[None, :] < tape.padding_count[:, None],
        tape.padding_symbols < alphabet,
        tape.padding_symbols == 0,
    ), dim=1)
    semantic_ok = semantic_ok & padding_semantic & padding_payload_ok
    padding_source = torch.clamp(
        rank_grid - length[:, None], min=0, max=minimum - 1,
    )
    padding_values = torch.gather(
        tape.padding_symbols, 1, padding_source,
    )
    in_padding = (
        (rank_grid >= length[:, None])
        & (rank_grid < length[:, None] + expected_padding[:, None])
    )
    candidate = torch.where(in_padding, padding_values, candidate)
    length = length + expected_padding

    raw_delta = length - pre_length
    budget_trim = raw_delta > computed_budget
    final_length = torch.where(
        budget_trim, pre_length + computed_budget, length,
    )
    material_delta = final_length - pre_length
    candidate = torch.where(
        rank_grid < final_length[:, None], candidate,
        torch.zeros_like(candidate),
    )
    semantic_ok = semantic_ok & (
        tape.post_structural_lengths == final_length
    ) & (tape.material_delta_symbols == material_delta)
    per_sequence_capacity = final_length <= W

    expected_events = torch.stack([
        torch.where(
            insertion_execute, insertion_count,
            torch.zeros_like(insertion_count),
        ),
        torch.where(
            deletion_execute, deletion_count,
            torch.zeros_like(deletion_count),
        ),
        duplication_execute.to(torch.int64) * gene_span,
        inversion_hit.to(torch.int64),
        transposition_hit.to(torch.int64),
    ], dim=1)
    semantic_ok = semantic_ok & torch.all(
        tape.structural_event_counts == expected_events, dim=1,
    )

    pools_after = base.pools_after.clone()
    nucleotide_before = base.pools_after[:, a4.a3.POOL_NUCLEOTIDE]
    positive_nucleotide = nucleotide_before - (
        material_delta.to(dtype) * float(a4.g2.MONOMER_MASS)
    )
    negative_nucleotide = nucleotide_before + (
        (-material_delta).to(dtype) * float(a4.g2.MONOMER_MASS)
    )
    nucleotide_after = torch.where(
        material_delta > 0, positive_nucleotide,
        torch.where(
            material_delta < 0, negative_nucleotide, nucleotide_before,
        ),
    )
    pools_after[:, a4.a3.POOL_NUCLEOTIDE] = nucleotide_after
    chemistry_ok = torch.isfinite(nucleotide_after) & (
        nucleotide_after >= 0.0
    )
    semantic_ok = semantic_ok & chemistry_ok

    effective_error = base.last_effective_error_rate
    proof_fraction = _torch_completion_proof_fraction(
        binding, deterministic_flags['proofreading'],
    )
    inherited_lesion = binding.ragged.replication_template_lesions * (
        0.28 + 0.22 * (1.0 - proof_fraction)
    )
    new_lesion = inherited_lesion + (
        effective_error * final_length.to(dtype) * 0.06
    )
    semantic_ok = semantic_ok & torch.isfinite(new_lesion) & (
        new_lesion >= 0.0
    )

    lesion_capacity = int(binding.ragged.sequence_capacity)
    lesion_rank = torch.arange(
        lesion_capacity, dtype=torch.int64, device=device,
    )
    lesion_first = binding.ragged.lesion_offsets[:C]
    lesion_last = binding.ragged.lesion_offsets[1:C + 1]
    lesion_count = lesion_last - lesion_first
    lesion_indices = torch.clamp(
        lesion_first[:, None] + lesion_rank[None, :],
        min=0, max=lesion_capacity - 1,
    )
    lesion_values = binding.ragged.genome_lesions[lesion_indices]
    lesion_values = torch.where(
        lesion_rank[None, :] < lesion_count[:, None],
        lesion_values, torch.zeros_like(lesion_values),
    )
    # NumPy groups the complete post-state lesion prefix, including the new
    # lesion, in one reduction.  Appending it after an old-prefix sum changes
    # fp64 association at widths such as seven -> eight.  Build the combined
    # prefix first, then select the same fixed-width pairwise candidate.
    combined_lesion_values = torch.cat([
        lesion_values,
        torch.zeros((C, 1), dtype=dtype, device=device),
    ], dim=1)
    combined_rank = torch.arange(
        lesion_capacity + 1, dtype=torch.int64, device=device,
    )
    combined_lesion_values = torch.where(
        combined_rank[None, :] == lesion_count[:, None],
        new_lesion[:, None], combined_lesion_values,
    )
    post_lesion_count = lesion_count + 1
    lesion_total = torch.zeros((C,), dtype=dtype, device=device)
    for prefix_count in range(1, lesion_capacity + 2):
        prefix_total = _torch_numpy_pairwise_sum_rows(
            combined_lesion_values[:, :prefix_count]
        )
        lesion_total = torch.where(
            post_lesion_count == prefix_count, prefix_total, lesion_total,
        )
    genome_lesion_mean_after = lesion_total / post_lesion_count.to(dtype)

    aggregate_capacity_overflow = (
        int(binding.ragged.symbol_count)
        + torch.sum(base.topology_symbol_deltas)
        + torch.sum(material_delta)
        > int(binding.ragged.symbol_capacity)
    )
    capacity_overflow = (
        aggregate_capacity_overflow
        | torch.any(used & (~per_sequence_capacity))
    )
    boundary_failure = torch.any(used & boundary_rows)
    semantic_failure = (
        (~resident_unchanged)
        | torch.any(used & (~semantic_ok) & (~boundary_rows))
    ) & (~boundary_failure) & (~base_failure)
    batch_failure = (
        base_failure | boundary_failure | semantic_failure
        | capacity_overflow
    )
    success = used & (~batch_failure)
    error = base.scope_error_code.clone()
    error = torch.where(
        used & semantic_failure & (error == SCOPE_OK),
        torch.full_like(error, SCOPE_RNG_TAPE_MISMATCH), error,
    )
    error = torch.where(
        used & boundary_failure & (error == SCOPE_OK),
        torch.full_like(error, SCOPE_FP64_DISCRETE_BOUNDARY), error,
    )
    error = torch.where(
        used & capacity_overflow & (error == SCOPE_OK),
        torch.full_like(error, SCOPE_CAPACITY), error,
    )

    zero_i64 = torch.zeros((C,), dtype=torch.int64, device=device)
    zero_f64 = torch.zeros((C,), dtype=dtype, device=device)
    output = A4CompletionMutationPlan(
        schema_version=COMPLETION_MUTATION_PLAN_SCHEMA_VERSION,
        cell_capacity=C,
        symbol_capacity=W,
        cell_count=int(base.cell_count),
        source_provenance=str(base.source_provenance),
        cell_ids=base.cell_ids.clone(),
        cell_mask=base.cell_mask.clone(),
        scope_valid=success,
        scope_error_code=error,
        completion_events=success.clone(),
        pre_structural_lengths=torch.where(
            success, pre_length, zero_i64,
        ),
        final_symbols=torch.where(
            success[:, None], candidate[:, :W],
            torch.zeros((C, W), dtype=torch.uint8, device=device),
        ),
        final_lengths=torch.where(success, final_length, zero_i64),
        pools_after=torch.where(
            success[:, None], pools_after,
            torch.where(
                used[:, None], binding.state.pools,
                torch.zeros_like(binding.state.pools),
            ),
        ),
        requested_symbols=torch.where(
            success, base.requested_symbols, zero_i64,
        ),
        append_count=torch.where(success, base.append_count, zero_i64),
        last_replication_symbols=torch.where(
            success, base.last_replication_symbols, zero_i64,
        ),
        last_effective_error_rate=torch.where(
            success, effective_error, zero_f64,
        ),
        cumulative_proofreading_atp_after=torch.where(
            success, base.cumulative_proofreading_atp_after,
            torch.where(
                used, binding.state.cumulative_proofreading_atp, zero_f64,
            ),
        ),
        substitution_events=torch.where(
            success, tape.substitution_count, zero_i64,
        ),
        structural_event_counts=torch.where(
            success[:, None], tape.structural_event_counts,
            torch.zeros_like(tape.structural_event_counts),
        ),
        material_delta_symbols=torch.where(
            success, material_delta, zero_i64,
        ),
        new_genome_lesions=torch.where(success, new_lesion, zero_f64),
        replication_cycle_deltas=success.to(torch.int64),
        topology_sequence_deltas=-success.to(torch.int64),
        topology_symbol_deltas=torch.where(
            success, base.topology_symbol_deltas + material_delta, zero_i64,
        ),
        replication_active_after=torch.where(
            success, torch.zeros_like(binding.ragged.replication_active),
            binding.ragged.replication_active,
        ),
        replication_template_lesions_after=torch.where(
            success, zero_f64, binding.ragged.replication_template_lesions,
        ),
        replication_fractional_after=torch.where(
            success, zero_f64, binding.ragged.replication_fractional,
        ),
        genome_count_after=torch.where(
            success, binding.state.genome_count + 1,
            binding.state.genome_count,
        ),
        genome_material_symbols_after=torch.where(
            success,
            binding.state.genome_material_symbols
            + base.append_count + material_delta,
            binding.state.genome_material_symbols,
        ),
        genome_lesion_mean_after=torch.where(
            success, genome_lesion_mean_after,
            binding.state.genome_lesion_mean,
        ),
    )
    _validate_completion_mutation_plan_metadata(output)
    return output


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


def paid_replication_completion_plan(binding, dt, config):
    """Dispatch the A4.6a pure descriptor without exposing mode flags."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return _paid_replication_completion_torch(binding, dt, config)
    return _paid_replication_completion_numpy(binding, dt, config)


def paid_replication_completion_mutation_plan(
        binding, dt, config, tape):
    """Dispatch the pure A4.6b2 descriptor without committing any state."""
    a4._require_translation_binding(binding)
    if _is_tensor(binding.state.pools):
        return paid_replication_completion_mutation_torch(
            binding, dt, config, tape,
        )
    return paid_replication_completion_mutation_numpy(
        binding, dt, config, tape,
    )


PORT_STATUS = dict(a4.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.6b2-pre-existing-active-all-row-completion-combined-pcg64-'
        'substitution-structural-material-fixed-resident-plan-'
        'not-arena-committed-not-integrated-cpu-authoritative'
    ),
    'material_mutation': (
        'a4.6b2-binding-aware-attested-tape-applied-to-pure-resident-'
        'descriptor-not-live-rng-authority-cpu-authoritative'
    ),
    'full_gpu_world_step': False,
})
