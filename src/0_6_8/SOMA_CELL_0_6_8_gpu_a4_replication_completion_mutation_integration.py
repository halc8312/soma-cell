# coding: utf-8
"""A4.8c3 mutation-enabled active-completion replication commit bridge.

This deliberately bounded bridge adds one branch to A4.8c2: an already
active, non-empty replication template whose incomplete copy completes in
this call while ``config.mutation`` is true.  A binding-aware A4.6b1 PCG64
tape drives the A4.6b2 resident final-polymer plan.  Explicit resident
readback is commit authority; the NumPy plan is only a 2e-12 fp64 oracle.

Mutation-free completion is delegated to A4.8c2 and active noncompletion is
delegated through A4.8c2 to A4.8c1.  Inactive start and pre-active early no-op
branches fail before claim, with no frozen-CPU fallback.
"""
from __future__ import division

import copy
import math
import os
import pickle
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
import SOMA_CELL_0_6_8_gpu_a4_replication_integration as a48c
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration fails closed without it
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c3'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | mutation-enabled active completion atomic commit bridge'
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c3-mutation-enabled-active-completion-atomic-commit'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_COMPLETION_MUTATION = 'active-completion-mutation'
_DISPATCH_COMPLETION_MUTATION = 'completion-mutation'
_DISPATCH_COMPLETION_DETERMINISTIC = 'completion-deterministic'
_DISPATCH_NONCOMPLETION = 'noncompletion'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}
_MUTATION_COUNTER_NAMES = (
    'substitution',
) + tuple(a44.STRUCTURAL_EVENT_NAMES)


class A4ReplicationCompletionMutationCommitError(
        a48c2.A4ReplicationCompletionCommitError):
    """The bounded A4.8c3 source, tape, final plan, or commit failed."""


def _canonical_completion_mutation_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 save schema differs'
        )
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _completion_mutation_plan_states_bit_exact(left, right):
    return a48a._state_arrays_bit_exact(
        left.state_dict(), right.state_dict(),
    )


def _nonnegative_float64_within_one_ulp(left, right):
    """Allow only the registered final lesion-mean division variance."""
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


def _completion_mutation_plan_semantic_match(resident, oracle):
    """Require exact discrete output and the preregistered fp64 band."""
    a44.validate_a4_completion_mutation_plan(resident)
    a44.validate_a4_completion_mutation_plan(oracle)
    for item in fields(a44.A4CompletionMutationPlan):
        name = item.name
        left = getattr(resident, name)
        right = getattr(oracle, name)
        if name not in a44._COMPLETION_PLAN_ARRAY_FIELDS:
            if left != right:
                raise A4ReplicationCompletionMutationCommitError(
                    'resident/NumPy completion-mutation metadata differs: '
                    + name
                )
            continue
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise A4ReplicationCompletionMutationCommitError(
                'resident/NumPy completion-mutation schema differs: ' + name
            )
        if left.dtype == np.dtype(np.float64):
            if name == 'genome_lesion_mean_after':
                equal = _nonnegative_float64_within_one_ulp(left, right)
            else:
                equal = (
                    np.isfinite(left).all()
                    and np.isfinite(right).all()
                    and not np.any(
                        np.abs(left - right) > REPLICATION_ORACLE_ATOL
                    )
                )
            if not equal:
                raise A4ReplicationCompletionMutationCommitError(
                    'resident/NumPy completion-mutation fp64 differs: ' + name
                )
        elif not np.array_equal(left, right):
            raise A4ReplicationCompletionMutationCommitError(
                'resident/NumPy completion-mutation discrete differs: ' + name
            )
    return resident


def _completion_mutation_plan_scope(binding, tape, plan, config):
    """Bind one A4.6b2 final descriptor to its exact active source/tape."""
    a4._require_translation_binding(binding)
    a44.validate_a4_completion_mutation_rng_tape(
        tape, binding, float.fromhex(tape.dt_hex),
        config,
    )
    a44.validate_a4_completion_mutation_plan(plan)
    geometry = a48c2._completion_geometry(binding)
    if int(plan.cell_count) != 1:
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 final plan is not exactly one cell'
        )
    if (str(plan.source_provenance)
            != str(binding.state.source_provenance)
            or str(tape.source_provenance)
            != str(binding.state.source_provenance)
            or int(plan.cell_capacity) != int(binding.state.cell_capacity)
            or int(plan.symbol_capacity)
            != int(binding.ragged.max_sequence_symbols)
            or int(tape.cell_capacity) != int(binding.state.cell_capacity)
            or int(tape.append_capacity)
            != int(binding.ragged.max_sequence_symbols)
            or not np.array_equal(
                np.asarray(plan.cell_ids, dtype=np.int64),
                np.asarray(binding.state.cell_ids, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(plan.cell_mask, dtype=bool),
                np.asarray(binding.state.cell_mask, dtype=bool),
            )):
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 final plan does not bind the live source identity'
        )
    template_length = int(geometry['template'].size)
    append_count = int(plan.append_count[0])
    final_length = int(plan.final_lengths[0])
    material_delta = int(plan.material_delta_symbols[0])
    if (not bool(plan.scope_valid[0])
            or int(plan.scope_error_code[0]) != int(a44.SCOPE_OK)
            or not bool(plan.completion_events[0])
            or int(plan.pre_structural_lengths[0]) != template_length
            or int(tape.pre_structural_lengths[0]) != template_length
            or append_count <= 0
            or int(tape.append_count[0]) != append_count
            or final_length != int(tape.post_structural_lengths[0])
            or material_delta != int(tape.material_delta_symbols[0])
            or int(plan.substitution_events[0])
            != int(tape.substitution_count[0])
            or not np.array_equal(
                np.asarray(plan.structural_event_counts[0], dtype=np.int64),
                np.asarray(tape.structural_event_counts[0], dtype=np.int64),
            )
            or int(plan.last_replication_symbols[0]) != append_count
            or int(plan.replication_cycle_deltas[0]) != 1
            or int(plan.topology_sequence_deltas[0]) != -1
            or int(plan.topology_symbol_deltas[0])
            != -template_length + append_count + material_delta
            or bool(plan.replication_active_after[0])
            or float(plan.replication_template_lesions_after[0]) != 0.0
            or float(plan.replication_fractional_after[0]) != 0.0
            or int(plan.genome_count_after[0])
            != int(binding.state.genome_count[0]) + 1
            or int(plan.genome_material_symbols_after[0])
            != (int(binding.state.genome_material_symbols[0])
                + append_count + material_delta)
            or final_length <= 0
            or not math.isfinite(float(plan.new_genome_lesions[0]))
            or float(plan.new_genome_lesions[0]) < 0.0):
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 final plan crossed its active-completion scope'
        )
    before = np.asarray(binding.state.pools[0], dtype=np.float64)
    after = np.asarray(plan.pools_after[0], dtype=np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _REPLICATION_PAID_POOLS
    ]
    if not a48c._float64_bits_equal(before[unchanged], after[unchanged]):
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 final plan changed an out-of-scope material pool'
        )
    return geometry


def _host_array_artifact_snapshot(label, batch, names):
    result = {
        'batch_id': id(batch), 'member_ids': {}, 'data_ptrs': {},
    }
    for name in names:
        value = getattr(batch, name)
        if not isinstance(value, np.ndarray):
            raise A4ReplicationCompletionMutationCommitError(
                '%s.%s escaped NumPy host authority' % (label, name)
            )
        result['member_ids'][name] = id(value)
        result['data_ptrs'][name] = int(
            value.__array_interface__['data'][0]
        )
    return result


def _host_completion_mutation_artifacts(tape, host_plan, readback_plan):
    a44._require_completion_mutation_rng_tape(tape)
    if a44._validate_completion_mutation_tape_metadata(tape) != 'numpy':
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 retained host tape is not NumPy'
        )
    a44.validate_a4_completion_mutation_plan(host_plan)
    a44.validate_a4_completion_mutation_plan(readback_plan)
    return {
        'tape': _host_array_artifact_snapshot(
            'tape', tape, a44._COMPLETION_TAPE_ARRAY_FIELDS,
        ),
        'host_plan': _host_array_artifact_snapshot(
            'host_plan', host_plan, a44._COMPLETION_PLAN_ARRAY_FIELDS,
        ),
        'readback_plan': _host_array_artifact_snapshot(
            'readback_plan', readback_plan,
            a44._COMPLETION_PLAN_ARRAY_FIELDS,
        ),
    }


def _resident_completion_mutation_artifacts(
        binding, tape, plan, device):
    """Freeze every resident wrapper, member, device, pointer, and version."""
    a4._require_translation_binding(binding)
    a44._require_completion_mutation_rng_tape(tape)
    a44._validate_completion_mutation_plan_metadata(plan)
    expected = torch.device(str(device))
    batches = (
        ('ragged', binding.ragged, a4._ARRAY_FIELDS),
        ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        ('cache', binding.cache, a4._GENE_ARRAY_FIELDS),
        ('tape', tape, a44._COMPLETION_TAPE_ARRAY_FIELDS),
        ('plan', plan, a44._COMPLETION_PLAN_ARRAY_FIELDS),
    )
    result = {
        'wrapper_ids': {
            'binding': id(binding), 'ragged': id(binding.ragged),
            'state': id(binding.state), 'cache': id(binding.cache),
        },
        'batch_ids': {}, 'member_ids': {}, 'data_ptrs': {}, 'versions': {},
    }
    for label, batch, names in batches:
        result['batch_ids'][label] = id(batch)
        result['member_ids'][label] = {}
        result['data_ptrs'][label] = {}
        result['versions'][label] = {}
        for name in names:
            value = getattr(batch, name)
            if not a4._is_tensor(value):
                raise A4ReplicationCompletionMutationCommitError(
                    'A4.8c3 resident artifact escaped Torch authority'
                )
            actual = value.device
            if (actual.type != expected.type
                    or (expected.index is not None
                        and actual.index != expected.index)):
                raise A4ReplicationCompletionMutationCommitError(
                    'A4.8c3 resident artifact moved device'
                )
            result['member_ids'][label][name] = id(value)
            result['data_ptrs'][label][name] = int(value.data_ptr())
            result['versions'][label][name] = int(value._version)
    return result


def _require_resident_completion_mutation_artifacts(
        binding, tape, plan, device, expected):
    actual = _resident_completion_mutation_artifacts(
        binding, tape, plan, device,
    )
    if actual != expected:
        raise A4ReplicationCompletionMutationCommitError(
            'A4.8c3 resident object/member/device/pointer/version changed'
        )
    return binding, tape, plan


def _expected_mutation_events(source, plan):
    expected = dict(source.mutation_events)
    for name in _MUTATION_COUNTER_NAMES:
        if name not in expected:
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 source mutation ledger lacks ' + name
            )
    expected['substitution'] += int(plan.substitution_events[0])
    for index, name in enumerate(a44.STRUCTURAL_EVENT_NAMES):
        expected[name] += int(plan.structural_event_counts[0, index])
    return expected


@dataclass
class _A4ReplicationCompletionMutationCommitCandidate:
    """Private one-shot c3 candidate; never durable authority."""

    _factory_token: object
    scheduler_object_id: int
    world_object_id: int
    cell_object_id: int
    cell_id: int
    generation: int
    dt_hex: str
    branch: str
    source_binding_identity: object
    source_binding: object
    source_artifacts: object
    source_cell_state: bytes
    source_gene_specs: object
    source_objects: object
    tape: object
    tape_state: object
    host_replay: object
    host_replay_state: object
    resident_binding: object
    resident_tape: object
    resident_plan: object
    resident_artifacts: object
    plan: object
    plan_state: object
    host_artifacts: object
    candidate_cell: object
    candidate_cell_state: bytes
    candidate_objects: object
    fresh_binding: object
    fresh_artifacts: object
    live_rng: object
    rng_before_state: object
    rng_after_state: object
    dissipated_energy: float
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4ReplicationCompletionMutationEventScheduler(
        a48c2.A4ReplicationCompletionEventScheduler):
    """A4.8c2 scheduler plus mutation-enabled active completion only."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_completion_mutation_commit_active = False
        super(A4ReplicationCompletionMutationEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def completion_mutation_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_completion_mutation_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c3 completion commit'
            )
        state = super(
            A4ReplicationCompletionMutationEventScheduler, self,
        ).state_dict()
        state['a4_replication_completion_mutation'] = (
            self.completion_mutation_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        mutation = _canonical_completion_mutation_integration_state(
            state.get('a4_replication_completion_mutation'),
        )
        completion = (
            a48c2.A4ReplicationCompletionEventScheduler.from_state(state)
        )
        if (mutation['config']
                != a48a._config_state(completion.a4_config)
                or mutation['device'] != completion.a4_device):
            raise A4ReplicationCompletionMutationCommitError(
                'saved A4.8c2/A4.8c3 config or device differs'
            )
        scheduler = cls(
            a4_config=mutation['config'], device=mutation['device'],
            max_receipts=completion.max_receipts,
        )
        scheduler._next_step_id = int(completion._next_step_id)
        scheduler._backend_name = str(completion._backend_name)
        scheduler._receipts = copy.deepcopy(completion._receipts)
        return scheduler

    def _classify_replication_branch_c3(
            self, world, cell, dt, config):
        """Classify with a resident deterministic probe, never exceptions."""
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 replication source cell must be alive'
            )
        if not isinstance(
                getattr(config, 'genome_replication', None),
                (bool, np.bool_)) or not bool(config.genome_replication):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 disabled replication is outside scope'
            )
        mutation_enabled = a48c._strict_mutation_flag(config)
        _, source_binding = self._pack_host_binding(world, cell)
        a48c2._completion_geometry(source_binding)
        source_state = a48c._cell_state_snapshot(cell)
        source_objects = a48c._source_object_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)
        energy_before = float(world.dissipated_energy)
        deterministic = copy.deepcopy(config)
        deterministic.mutation = False
        resident_binding = a4.bind_a4_translation(
            source_binding.ragged.to_torch(device=self.a4_device),
            source_binding.state.to_torch(device=self.a4_device),
        )
        probe = a44.paid_replication_completion_plan(
            resident_binding, dt, deterministic,
        )
        if not a4._is_tensor(probe.scope_error_code):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 dispatch probe escaped resident authority'
            )
        codes = probe.scope_error_code.detach().cpu().numpy().copy()
        completions = probe.completion_events.detach().cpu().numpy().copy()
        valid = probe.scope_valid.detach().cpu().numpy().copy()
        expected_shape = (int(source_binding.state.cell_capacity),)
        if (codes.shape != expected_shape
                or completions.shape != expected_shape
                or valid.shape != expected_shape):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 dispatch probe schema differs'
            )
        if (a48c._cell_state_snapshot(cell) != source_state
                or not a48c._source_objects_match(cell, source_objects)
                or world.rng is not live_rng
                or world.rng.bit_generator.state != rng_before
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], dtype=np.float64),
                    np.asarray([energy_before], dtype=np.float64),
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 dispatch probe changed live authority'
            )
        code = int(codes[0])
        completed = bool(completions[0])
        supported = bool(valid[0])
        if code == int(a44.SCOPE_OK) and completed and supported:
            return (
                _DISPATCH_COMPLETION_MUTATION
                if mutation_enabled else
                _DISPATCH_COMPLETION_DETERMINISTIC
            )
        if (code == int(a44.SCOPE_NONCOMPLETION)
                and not completed and not supported):
            return _DISPATCH_NONCOMPLETION
        raise A4ReplicationCompletionMutationCommitError(
            'replication branch is outside A4.8c3/c2/c1 scope: %d' % code
        )

    def _build_completion_mutation_candidate(self, cell, plan):
        candidate = copy.deepcopy(cell)
        final_length = int(plan.final_lengths[0])
        final = np.asarray(
            plan.final_symbols[0, :final_length], dtype=np.uint8,
        ).copy()
        candidate.pools = np.asarray(
            plan.pools_after[0], dtype=np.float64,
        ).copy()
        candidate.genomes.append(final)
        candidate.genome_lesions.append(float(plan.new_genome_lesions[0]))
        candidate.mutation_events = _expected_mutation_events(cell, plan)
        candidate.replication_cycles += int(
            plan.replication_cycle_deltas[0]
        )
        candidate.replication_template = None
        candidate.replication_template_lesion = float(
            plan.replication_template_lesions_after[0]
        )
        candidate.replication_copy = []
        candidate.replication_fractional = float(
            plan.replication_fractional_after[0]
        )
        candidate.last_replication_symbols = int(
            plan.last_replication_symbols[0]
        )
        candidate.last_effective_error_rate = float(
            plan.last_effective_error_rate[0]
        )
        candidate.cumulative_proofreading_atp = float(
            plan.cumulative_proofreading_atp_after[0]
        )
        candidate._refresh_gene_cache()
        if (candidate.novel_path_first_age is None
                and a4.g2.sequence_has_novel_path(final)):
            candidate.novel_path_first_age = float(candidate.age)
        return candidate

    def _require_completion_mutation_candidate_scope(
            self, source, candidate, tape, plan, source_binding, config):
        geometry = _completion_mutation_plan_scope(
            source_binding, tape, plan, config,
        )
        if (candidate is source
                or candidate.pools is source.pools
                or candidate.genomes is source.genomes
                or candidate.genome_lesions is source.genome_lesions
                or candidate.replication_copy is source.replication_copy
                or candidate.mutation_events is source.mutation_events
                or candidate.proteins is source.proteins
                or candidate.damaged_proteins is source.damaged_proteins
                or candidate.gene_specs is source.gene_specs
                or any(
                    left is right for left, right in zip(
                        candidate.genomes[:len(source.genomes)],
                        source.genomes,
                    )
                )
                or any(
                    candidate.gene_specs.get(fingerprint)
                    is source.gene_specs.get(fingerprint)
                    for fingerprint in source.gene_specs
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 candidate is not deeply isolated from live biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 candidate cell schema differs from source'
            )
        allowed = {
            'pools', 'genomes', 'genome_lesions', 'replication_template',
            'replication_template_lesion', 'replication_copy',
            'replication_fractional', 'replication_cycles', 'gene_specs',
            'novel_path_first_age', 'last_replication_symbols',
            'last_effective_error_rate', 'cumulative_proofreading_atp',
            'mutation_events',
        }
        for name in source_state:
            if name in allowed:
                continue
            before = pickle.dumps(
                copy.deepcopy(source_state[name]),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            after = pickle.dumps(
                copy.deepcopy(candidate_state[name]),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            if before != after:
                raise A4ReplicationCompletionMutationCommitError(
                    'A4.8c3 candidate changed out-of-scope field: ' + name
                )
        final_length = int(plan.final_lengths[0])
        final = np.asarray(
            plan.final_symbols[0, :final_length], dtype=np.uint8,
        )
        expected_novel = copy.deepcopy(source.novel_path_first_age)
        if expected_novel is None and a4.g2.sequence_has_novel_path(final):
            expected_novel = float(source.age)
        if (len(candidate.genomes) != len(source.genomes) + 1
                or any(
                    not np.array_equal(left, right)
                    for left, right in zip(
                        candidate.genomes[:-1], source.genomes,
                    )
                )
                or not np.array_equal(candidate.genomes[-1], final)
                or list(candidate.genome_lesions[:-1])
                != list(source.genome_lesions)
                or not a48c._float64_bits_equal(
                    np.asarray([candidate.genome_lesions[-1]], np.float64),
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or candidate.replication_template is not None
                or candidate.replication_template_lesion != 0.0
                or candidate.replication_copy != []
                or int(candidate.replication_cycles)
                != int(source.replication_cycles) + 1
                or candidate.novel_path_first_age != expected_novel
                or candidate.mutation_events
                != _expected_mutation_events(source, plan)
                or not a48c._float64_bits_equal(
                    candidate.pools,
                    np.asarray(plan.pools_after[0], dtype=np.float64),
                )
                or not a48c._float64_bits_equal(
                    np.asarray([candidate.replication_fractional], np.float64),
                    np.asarray([plan.replication_fractional_after[0]], np.float64),
                )
                or int(candidate.last_replication_symbols)
                != int(plan.last_replication_symbols[0])
                or not a48c._float64_bits_equal(
                    np.asarray([candidate.last_effective_error_rate], np.float64),
                    np.asarray([plan.last_effective_error_rate[0]], np.float64),
                )
                or not a48c._float64_bits_equal(
                    np.asarray([candidate.cumulative_proofreading_atp], np.float64),
                    np.asarray([
                        plan.cumulative_proofreading_atp_after[0]
                    ], np.float64),
                )
                or final_length != int(tape.post_structural_lengths[0])
                or int(geometry['template'].size)
                != int(plan.pre_structural_lengths[0])):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 candidate differs from resident final plan'
            )
        return candidate

    def _validate_fresh_completion_mutation_candidate(
            self, source, candidate, tape, plan, source_binding,
            fresh_binding, config):
        a4._require_translation_binding(fresh_binding)
        self._require_completion_mutation_candidate_scope(
            source, candidate, tape, plan, source_binding, config,
        )
        a48b._require_cell_gene_specs(candidate, fresh_binding)
        geometry = a48c2._completion_geometry(source_binding)
        source_ragged = source_binding.ragged
        fresh_ragged = fresh_binding.ragged
        genome_count = int(geometry['genome_count'])
        lesion_first = int(source_ragged.lesion_offsets[0])
        lesion_last = int(source_ragged.lesion_offsets[1])
        old_lesion_count = lesion_last - lesion_first
        fresh_lesion_first = int(fresh_ragged.lesion_offsets[0])
        fresh_lesion_last = int(fresh_ragged.lesion_offsets[1])
        final_length = int(plan.final_lengths[0])
        final = np.asarray(
            plan.final_symbols[0, :final_length], dtype=np.uint8,
        )
        complete_end = int(geometry['complete_symbol_end'])
        expected_symbol_count = (
            int(source_ragged.symbol_count)
            + int(plan.topology_symbol_deltas[0])
        )
        if (int(fresh_ragged.sequence_count)
                != int(source_ragged.sequence_count) - 1
                or int(fresh_ragged.symbol_count) != expected_symbol_count
                or int(fresh_ragged.lesion_count)
                != int(source_ragged.lesion_count) + 1
                or fresh_lesion_last - fresh_lesion_first
                != old_lesion_count + 1
                or int(fresh_ragged.genome_counts[0]) != genome_count + 1
                or bool(fresh_ragged.replication_active[0])
                or not np.array_equal(
                    fresh_ragged.sequence_offsets[:genome_count + 1],
                    source_ragged.sequence_offsets[:genome_count + 1],
                )
                or int(fresh_ragged.sequence_offsets[genome_count + 1])
                != complete_end + final_length
                or not np.array_equal(
                    fresh_ragged.symbols[:complete_end],
                    source_ragged.symbols[:complete_end],
                )
                or not np.array_equal(
                    fresh_ragged.symbols[
                        complete_end:complete_end + final_length
                    ], final,
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.genome_lesions[
                        fresh_lesion_first:
                        fresh_lesion_first + old_lesion_count
                    ],
                    source_ragged.genome_lesions[
                        lesion_first:lesion_last
                    ],
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.genome_lesions[
                        fresh_lesion_first + old_lesion_count:
                        fresh_lesion_first + old_lesion_count + 1
                    ],
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or float(fresh_ragged.replication_template_lesions[0])
                != float(plan.replication_template_lesions_after[0])
                or float(fresh_ragged.replication_fractional[0])
                != float(plan.replication_fractional_after[0])):
            raise A4ReplicationCompletionMutationCommitError(
                'fresh A4.8c3 ragged state differs from resident final plan'
            )
        source_state = source_binding.state
        fresh_state = fresh_binding.state
        plan_fields = {
            'pools': 'pools_after',
            'cumulative_proofreading_atp': (
                'cumulative_proofreading_atp_after'
            ),
            'genome_count': 'genome_count_after',
            'replication_active': 'replication_active_after',
            'genome_material_symbols': 'genome_material_symbols_after',
            'genome_lesion_mean': 'genome_lesion_mean_after',
        }
        for item in fields(a4.A4TranslationStateBatch):
            name = item.name
            before = getattr(source_state, name)
            after = getattr(fresh_state, name)
            if name not in a4._TRANSLATION_ARRAY_FIELDS:
                if name == 'source_provenance':
                    if after != fresh_binding._ragged_provenance:
                        raise A4ReplicationCompletionMutationCommitError(
                            'fresh A4.8c3 provenance is not self-consistent'
                        )
                elif before != after:
                    raise A4ReplicationCompletionMutationCommitError(
                        'fresh A4.8c3 metadata changed: ' + name
                    )
                continue
            expected = (
                np.asarray(getattr(plan, plan_fields[name]))
                if name in plan_fields else np.asarray(before)
            )
            after = np.asarray(after)
            if expected.dtype != after.dtype or expected.shape != after.shape:
                raise A4ReplicationCompletionMutationCommitError(
                    'fresh A4.8c3 state schema changed: ' + name
                )
            if name == 'genome_lesion_mean':
                same = _nonnegative_float64_within_one_ulp(expected, after)
            else:
                same = (
                    a48c._float64_bits_equal(expected, after)
                    if expected.dtype == np.dtype(np.float64)
                    else np.array_equal(expected, after)
                )
            if not same:
                raise A4ReplicationCompletionMutationCommitError(
                    'fresh A4.8c3 state differs: ' + name
                )
        return fresh_binding

    def _prepare_completion_mutation_candidate(
            self, world, cell, dt, config):
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if not a48c._strict_mutation_flag(config):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 completion requires mutation=True'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 replication source cell must be alive'
            )
        _, source_binding = self._pack_host_binding(world, cell)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        source_artifacts = a48c._host_binding_artifact_snapshot(
            source_binding,
        )
        source_cell_state = a48c._cell_state_snapshot(cell)
        source_objects = a48c._source_object_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)
        tape = a44.prepare_completion_mutation_rng_tape(
            source_binding, dt, config, rng_before,
        )
        host_replay = a44.paid_replication_completion_mutation_plan(
            source_binding, dt, config, tape,
        )
        _completion_mutation_plan_scope(
            source_binding, tape, host_replay, config,
        )

        resident_binding = a4.bind_a4_translation(
            source_binding.ragged.to_torch(device=self.a4_device),
            source_binding.state.to_torch(device=self.a4_device),
        )
        resident_tape = tape.to_torch(
            source_binding, dt, config, device=self.a4_device,
        )
        resident_plan = a44.paid_replication_completion_mutation_plan(
            resident_binding, dt, config, resident_tape,
        )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 resident final plan escaped Torch authority'
            )
        resident_artifacts = _resident_completion_mutation_artifacts(
            resident_binding, resident_tape, resident_plan, self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache)
                != source_binding._cache_provenance):
            raise A4ReplicationCompletionMutationCommitError(
                'resident A4.8c3 source differs after readback'
            )
        resident_tape_host = resident_tape.to_numpy()
        a44.validate_a4_completion_mutation_rng_tape(
            resident_tape_host, resident_source, dt, config,
        )
        if not a48a._state_arrays_bit_exact(
                resident_tape_host.state_dict(), tape.state_dict()):
            raise A4ReplicationCompletionMutationCommitError(
                'resident A4.8c3 tape differs after readback'
            )
        plan = resident_plan.to_numpy()
        _completion_mutation_plan_scope(
            source_binding, tape, plan, config,
        )
        _completion_mutation_plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_completion_mutation_candidate(cell, plan)
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_completion_mutation_candidate(
            cell, candidate_cell, tape, plan, source_binding,
            fresh_binding, config,
        )
        candidate_cell_state = a48c._cell_state_snapshot(candidate_cell)
        candidate_objects = a48c2._completion_candidate_object_snapshot(
            candidate_cell,
        )
        fresh_artifacts = a48c._host_binding_artifact_snapshot(fresh_binding)
        host_artifacts = _host_completion_mutation_artifacts(
            tape, host_replay, plan,
        )
        return _A4ReplicationCompletionMutationCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self), world_object_id=id(world),
            cell_object_id=id(cell), cell_id=int(cell.cell_id),
            generation=generation, dt_hex=dt.hex(),
            branch=_BRANCH_COMPLETION_MUTATION,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding, source_artifacts=source_artifacts,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects, tape=tape,
            tape_state=tape.state_dict(), host_replay=host_replay,
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_tape=resident_tape, resident_plan=resident_plan,
            resident_artifacts=resident_artifacts, plan=plan,
            plan_state=plan.state_dict(), host_artifacts=host_artifacts,
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding, fresh_artifacts=fresh_artifacts,
            live_rng=live_rng, rng_before_state=copy.deepcopy(rng_before),
            rng_after_state=copy.deepcopy(tape.rng_after_state),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.completion_mutation_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_completion_mutation_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(
                candidate, _A4ReplicationCompletionMutationCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 candidate is untrusted or already consumed'
            )
        dt = a48c._strict_nonnegative_dt(dt)
        live_rng, rng_before = a48c._live_pcg64(world)
        if (id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.completion_mutation_integration_state()
                != candidate.integration_state
                or a48a._world_config_snapshot(world)
                != candidate.world_config_snapshot
                or id(cell) != candidate.cell_object_id
                or int(cell.cell_id) != candidate.cell_id
                or a48a._strict_counter(
                    getattr(cell, 'generation', None), 'cell generation',
                ) != candidate.generation
                or not bool(getattr(cell, 'alive', False))
                or dt.hex() != candidate.dt_hex
                or not a48c._strict_mutation_flag(config)
                or live_rng is not candidate.live_rng
                or rng_before != candidate.rng_before_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], dtype=np.float64),
                    np.asarray([candidate.dissipated_energy], dtype=np.float64),
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.source_cell_state
                or not a48c._source_objects_match(
                    cell, candidate.source_objects,
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'live A4.8c3 cell/config/dt/RNG changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.source_binding,
                ) != candidate.source_artifacts):
            raise A4ReplicationCompletionMutationCommitError(
                'retained A4.8c3 host source changed before claim'
            )
        if (_host_completion_mutation_artifacts(
                candidate.tape, candidate.host_replay, candidate.plan,
                ) != candidate.host_artifacts
                or not a48a._state_arrays_bit_exact(
                    candidate.tape.state_dict(), candidate.tape_state,
                )
                or not a48a._state_arrays_bit_exact(
                    candidate.host_replay.state_dict(),
                    candidate.host_replay_state,
                )
                or not a48a._state_arrays_bit_exact(
                    candidate.plan.state_dict(), candidate.plan_state,
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'retained A4.8c3 tape/oracle/readback changed before claim'
            )
        fresh_tape = a44.prepare_completion_mutation_rng_tape(
            current_binding, dt, config, rng_before,
        )
        if (not a48a._state_arrays_bit_exact(
                fresh_tape.state_dict(), candidate.tape_state,
            ) or fresh_tape.rng_after_state != candidate.rng_after_state):
            raise A4ReplicationCompletionMutationCommitError(
                'fresh A4.8c3 tape or after-state differs before claim'
            )
        current_replay = a44.paid_replication_completion_mutation_plan(
            current_binding, dt, config, fresh_tape,
        )
        _completion_mutation_plan_scope(
            current_binding, fresh_tape, current_replay, config,
        )
        if not _completion_mutation_plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationCompletionMutationCommitError(
                'independent A4.8c3 NumPy replay changed before claim'
            )

        _require_resident_completion_mutation_artifacts(
            candidate.resident_binding, candidate.resident_tape,
            candidate.resident_plan, self.a4_device,
            candidate.resident_artifacts,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.source_binding._cache_provenance):
            raise A4ReplicationCompletionMutationCommitError(
                'resident A4.8c3 source changed before claim'
            )
        resident_tape = candidate.resident_tape.to_numpy()
        a44.validate_a4_completion_mutation_rng_tape(
            resident_tape, resident_source, dt, config,
        )
        if not a48a._state_arrays_bit_exact(
                resident_tape.state_dict(), fresh_tape.state_dict()):
            raise A4ReplicationCompletionMutationCommitError(
                'resident A4.8c3 tape changed before claim'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        if not _completion_mutation_plan_states_bit_exact(
                resident_plan, candidate.plan):
            raise A4ReplicationCompletionMutationCommitError(
                'resident A4.8c3 final plan changed before claim'
            )
        _completion_mutation_plan_scope(
            current_binding, fresh_tape, resident_plan, config,
        )
        _completion_mutation_plan_semantic_match(
            resident_plan, current_replay,
        )

        if (a48c._cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not a48c2._completion_candidate_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'fresh A4.8c3 candidate biology changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_completion_mutation_candidate(
            cell, candidate.candidate_cell, fresh_tape, candidate.plan,
            candidate.source_binding, fresh_now, config,
        )
        if (a48c._host_binding_artifact_snapshot(candidate.fresh_binding)
                != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCompletionMutationCommitError(
                'fresh A4.8c3 candidate changed before claim'
            )
        return candidate

    def _replication_completion_mutation_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after preparation and before validation."""
        return candidate

    def _publish_replication_completion_mutation_candidate(
            self, world, cell, candidate):
        """Selective c3 publish from the resident-readback candidate."""
        prepared = candidate.candidate_cell
        prepared_pools = np.asarray(prepared.pools, dtype=np.float64)
        for pool_index in sorted(_REPLICATION_PAID_POOLS):
            cell.pools[pool_index] = prepared_pools[pool_index]
        cell.genomes.append(np.asarray(
            prepared.genomes[-1], dtype=np.uint8,
        ).copy())
        cell.genome_lesions.append(float(prepared.genome_lesions[-1]))
        cell.replication_template = None
        cell.replication_template_lesion = 0.0
        cell.replication_copy[:] = []
        cell.replication_fractional = float(
            prepared.replication_fractional
        )
        cell.replication_cycles = int(prepared.replication_cycles)
        cell.last_replication_symbols = int(
            prepared.last_replication_symbols
        )
        cell.last_effective_error_rate = float(
            prepared.last_effective_error_rate
        )
        cell.cumulative_proofreading_atp = float(
            prepared.cumulative_proofreading_atp
        )
        for name in _MUTATION_COUNTER_NAMES:
            cell.mutation_events[name] = int(
                prepared.mutation_events[name]
            )
        cell.gene_specs.clear()
        for fingerprint, spec in prepared.gene_specs.items():
            cell.gene_specs[fingerprint] = copy.deepcopy(spec)
        cell.novel_path_first_age = copy.deepcopy(
            prepared.novel_path_first_age
        )
        world.rng.bit_generator.state = copy.deepcopy(
            candidate.rng_after_state
        )

    def _published_completion_mutation_matches(
            self, world, cell, candidate, snapshot):
        source_objects = candidate.source_objects
        prepared = candidate.candidate_cell
        old_count = len(source_objects['genome_objects'])
        if (cell.pools is not source_objects['pools_object']
                or cell.genomes is not source_objects['genomes_object']
                or len(cell.genomes) != old_count + 1
                or any(
                    current is not expected for current, expected in zip(
                        cell.genomes[:old_count],
                        source_objects['genome_objects'],
                    )
                )
                or not np.array_equal(
                    cell.genomes[-1], prepared.genomes[-1],
                )
                or cell.genomes[-1] is prepared.genomes[-1]
                or cell.genome_lesions
                is not source_objects['lesions_object']
                or cell.replication_template is not None
                or cell.replication_copy is not source_objects['copy_object']
                or cell.replication_copy != []
                or cell.mutation_events
                is not source_objects['mutation_events_object']
                or tuple(cell.mutation_events.items())
                != tuple(prepared.mutation_events.items())
                or cell.proteins is not source_objects['proteins_object']
                or cell.damaged_proteins
                is not source_objects['damaged_proteins_object']
                or cell.gene_specs is not source_objects['gene_specs_object']
                or not a4._gene_specs_exact(
                    cell.gene_specs, prepared.gene_specs,
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.candidate_cell_state
                or world.rng is not candidate.live_rng
                or world.rng.bit_generator.state
                != candidate.rng_after_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], dtype=np.float64),
                    np.asarray([snapshot['dissipated_energy']], np.float64),
                )):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCompletionMutationCommitError(
                'published A4.8c3 binding differs from fresh candidate'
            )

    @staticmethod
    def _completion_mutation_rng_call_count(tape):
        count = int(np.sum(
            np.asarray(tape.append_draw_mask[:1], dtype=np.int64),
            dtype=np.int64,
        ))
        count += int(np.sum(
            np.asarray(tape.replacement_mask[:1], dtype=np.int64),
            dtype=np.int64,
        ))
        count += int(np.sum(
            np.asarray(tape.threshold_draw_mask[:1], dtype=np.int64),
            dtype=np.int64,
        ))
        count += 3 * int(int(tape.insertion_count[0]) > 0)
        count += 2 * int(int(tape.deletion_count[0]) > 0)
        count += 2 * int(int(tape.duplication_gene_ordinal[0]) >= 0)
        count += 2 * int(int(tape.inversion_left[0]) >= 0)
        count += 3 * int(int(tape.transposition_count[0]) > 0)
        count += int(int(tape.padding_count[0]) > 0)
        return count

    def _commit_completion_mutation_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_completion_mutation_candidate(
            world, cell, dt, config, candidate,
        )
        snapshot = self._completion_publish_snapshot(world, cell)
        candidate.consumed = True
        plan = candidate.plan
        tape = candidate.tape
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c3-resident-completion-mutation-plan-atomic-'
                'cpu-cell-rng-commit'
            ),
            'branch': candidate.branch,
            'mutation_enabled': True,
            'device': self.a4_device,
            'genome_count': int(candidate.source_binding.state.genome_count[0]),
            'source_provenance': str(
                candidate.source_binding.state.source_provenance
            ),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
        })
        try:
            self._publish_replication_completion_mutation_candidate(
                world, cell, candidate,
            )
            self._published_completion_mutation_matches(
                world, cell, candidate, snapshot,
            )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': int(plan.last_replication_symbols[0]),
                'work_performed': int(plan.last_replication_symbols[0]) > 0,
                'requested_symbols': int(plan.requested_symbols[0]),
                'completion_events': int(plan.completion_events[0]),
                'substitution_events': int(plan.substitution_events[0]),
                'structural_event_counts': [
                    int(value) for value in
                    np.asarray(plan.structural_event_counts[0], np.int64)
                ],
                'material_delta_symbols': int(
                    plan.material_delta_symbols[0]
                ),
                'final_length': int(plan.final_lengths[0]),
                'rng_call_count': self._completion_mutation_rng_call_count(
                    tape,
                ),
                'rng_schedule_sha256': str(tape.schedule_sha256),
            })
        except BaseException:
            self._rollback_replication_publish(world, cell, snapshot)
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Dispatch c1/c2 explicitly or commit one bounded c3 branch."""
        _, record = self._context(world, cell, dt)
        if 'replication_cpu' in record['claimed']:
            raise a3s.A3DuplicateEventError(
                'duplicate A4.8c3 replication_cpu event'
            )
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_completion_mutation_commit_active
                or self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c3 replication commit is forbidden'
            )
        branch = self._classify_replication_branch_c3(
            world, cell, dt, config,
        )
        if branch in (
                _DISPATCH_NONCOMPLETION,
                _DISPATCH_COMPLETION_DETERMINISTIC):
            return super(
                A4ReplicationCompletionMutationEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        if branch != _DISPATCH_COMPLETION_MUTATION:
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 dispatch produced an unknown branch'
            )
        self._a4_replication_completion_mutation_commit_active = True
        try:
            candidate = self._prepare_completion_mutation_candidate(
                world, cell, dt, config,
            )
            ready = self._replication_completion_mutation_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationCompletionMutationCommitError(
                    'A4.8c3 ready hook must retain its private candidate'
                )
            return self._commit_completion_mutation_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_completion_mutation_commit_active = False


class Hybrid066WorldA4ReplicationCompletionMutation(
        a48c2.Hybrid066WorldA4ReplicationCompletion):
    """A4.8c2 world retaining bounded A4.8c3 authority after restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationCompletionMutationEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler,
                A4ReplicationCompletionMutationEventScheduler):
            raise TypeError(
                'scheduler must be '
                'A4ReplicationCompletionMutationEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationCompletionMutation, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationCompletionMutationCommitError(
                'new A4.8c3 world requires explicit A4 config and device'
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
        output = super(
            Hybrid066WorldA4ReplicationCompletionMutation, self,
        ).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_completion_mutation_device': (
                self.scheduler.a4_device
            ),
            'a4_replication_completion_mutation_config': (
                a48a._config_state(self.scheduler.a4_config)
            ),
        })
        return output

    def state_dict(self):
        state = super(
            Hybrid066WorldA4ReplicationCompletionMutation, self,
        ).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_completion_mutation': (
                self.scheduler.completion_mutation_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8c3 save version/build differs'
            )
        mutation = _canonical_completion_mutation_integration_state(
            state.get('a4_replication_completion_mutation'),
        )
        completion = a48c2._canonical_completion_integration_state(
            state.get('a4_replication_completion'),
        )
        replication = a48c._canonical_replication_integration_state(
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
                'A4.8c3 save is missing aggregate composition sidecars'
            )
        a3s._restore_aggregate_sidecars(
            world, state['aggregate_sidecars'],
        )
        config_state = copy.deepcopy(state.get('gpu_config', {}))
        if backend is None:
            backend = (
                backend_factory(config_state)
                if backend_factory is not None
                else a3s._default_backend(config_state)
            )
        scheduler = (
            A4ReplicationCompletionMutationEventScheduler.from_state(
                state.get('scheduler', {}),
            )
        )
        if (scheduler.completion_mutation_integration_state() != mutation
                or scheduler.completion_integration_state() != completion
                or scheduler.replication_integration_state() != replication
                or scheduler.translation_integration_state() != translation
                or scheduler.integration_state() != hydrolysis):
            raise A4ReplicationCompletionMutationCommitError(
                'A4.8a/b/c1/c2/c3 world/scheduler settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c2.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c3-mutation-enabled-pre-existing-active-completion-'
        'resident-final-plan-cpu-cell-pcg64-atomic-commit-'
        'a4.8c2-completion-and-a4.8c1-noncompletion-inherited-'
        'inactive-start-early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c3-substitution-structural-material-final-polymer-'
        'same-live-pcg64-atomic-commit-inactive-start-not-integrated'
    ),
    'event_scheduler': (
        'a4.8c3-completion-mutation-plus-a4.8c2-completion-plus-'
        'a4.8c1-noncompletion-plus-a4.8b-translation-plus-'
        'a4.8a-hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationCompletionMutationCommitError',
    'A4ReplicationCompletionMutationEventScheduler',
    'Hybrid066WorldA4ReplicationCompletionMutation',
)
