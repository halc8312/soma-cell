# coding: utf-8
"""A4.8c2 mutation-free active-completion replication commit bridge.

This module adds exactly one branch to A4.8c1: a pre-existing, non-empty
active template whose already-incomplete copy reaches completion in this call
while ``config.mutation`` is false.  The A4.6a resident completion descriptor
is explicit readback/commit authority and an independently recomputed NumPy
descriptor is only a 2e-12 fp64 oracle.  Existing active noncompletion is
explicitly delegated to A4.8c1.  Inactive start, early no-op, and mutation-on
completion fail before any scheduler claim and never fall back to frozen CPU.
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
import SOMA_CELL_0_6_8_gpu_a4_replication_integration as a48c
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration deliberately fails closed
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c2'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | mutation-free active completion atomic commit bridge'
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c2-mutation-free-active-completion-atomic-commit'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_COMPLETION = 'active-completion-deterministic'
_DISPATCH_COMPLETION = 'completion'
_DISPATCH_NONCOMPLETION = 'noncompletion'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}


class A4ReplicationCompletionCommitError(a48c.A4ReplicationCommitError):
    """The bounded A4.8c2 source, descriptor, or commit failed."""


def _canonical_completion_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 save schema differs'
        )
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _completion_geometry(binding):
    a4._require_translation_binding(binding)
    if int(binding.state.cell_count) != 1:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 requires exactly one replication cell'
        )
    ragged = binding.ragged
    if not bool(ragged.replication_active[0]):
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 inactive-template start is outside scope'
        )
    first = int(ragged.cell_sequence_offsets[0])
    end = int(ragged.cell_sequence_offsets[1])
    genome_count = int(ragged.genome_counts[0])
    if end != first + genome_count + 2:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 active replication topology is invalid'
        )
    template_slot = first + genome_count
    copy_slot = template_slot + 1
    template_first = int(ragged.sequence_offsets[template_slot])
    template_last = int(ragged.sequence_offsets[template_slot + 1])
    copy_first = int(ragged.sequence_offsets[copy_slot])
    copy_last = int(ragged.sequence_offsets[copy_slot + 1])
    template = np.asarray(
        ragged.symbols[template_first:template_last], dtype=np.uint8,
    ).copy()
    partial = np.asarray(
        ragged.symbols[copy_first:copy_last], dtype=np.uint8,
    ).copy()
    if template.size <= 0 or partial.size >= template.size:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 requires an incomplete non-empty active copy'
        )
    return {
        'first': first,
        'genome_count': genome_count,
        'template_slot': template_slot,
        'copy_slot': copy_slot,
        'template': template,
        'partial': partial,
        'complete_symbol_end': template_first,
    }


def _completion_plan_scope(binding, plan):
    """Bind one A4.6a completion descriptor to its exact host source."""
    a4._require_translation_binding(binding)
    a44.validate_a4_paid_elongation_plan(plan)
    if int(plan.cell_count) != 1:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 descriptor is not exactly one cell'
        )
    if (str(plan.source_provenance)
            != str(binding.state.source_provenance)
            or int(plan.cell_capacity) != int(binding.state.cell_capacity)
            or int(plan.append_capacity)
            != int(binding.ragged.max_sequence_symbols)
            or not np.array_equal(
                np.asarray(plan.cell_ids, dtype=np.int64),
                np.asarray(binding.state.cell_ids, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(plan.cell_mask, dtype=bool),
                np.asarray(binding.state.cell_mask, dtype=bool),
            )):
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 descriptor does not bind the live source identity'
        )
    geometry = _completion_geometry(binding)
    template = geometry['template']
    partial = geometry['partial']
    appended = int(plan.append_count[0])
    completed_length = int(plan.completed_lengths[0])
    suffix = np.asarray(
        plan.append_symbols[0, :appended], dtype=np.uint8,
    )
    completed = np.asarray(
        plan.completed_symbols[0, :completed_length], dtype=np.uint8,
    )
    expected = np.concatenate((partial, suffix))
    if (appended <= 0
            or partial.size + appended != template.size
            or completed_length != template.size
            or not np.array_equal(suffix, template[partial.size:])
            or not np.array_equal(completed, expected)
            or not bool(plan.scope_valid[0])
            or int(plan.scope_error_code[0]) != int(a44.SCOPE_OK)
            or not bool(plan.completion_events[0])
            or bool(plan.template_start_events[0])
            or int(plan.selected_template_indices[0]) != -1
            or int(plan.template_storage_symbols[0]) != 0
            or int(plan.substitution_events[0]) != 0
            or int(plan.last_replication_symbols[0]) != appended
            or int(plan.replication_cycle_deltas[0]) != 1
            or int(plan.topology_sequence_deltas[0]) != -1
            or int(plan.topology_symbol_deltas[0])
            != appended - completed_length
            or float(plan.replication_fractional_after[0]) != 0.0
            or not math.isfinite(float(plan.new_genome_lesions[0]))
            or float(plan.new_genome_lesions[0]) < 0.0):
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 descriptor crossed its completion scope'
        )
    before = np.asarray(binding.state.pools[0], dtype=np.float64)
    after = np.asarray(plan.pools_after[0], dtype=np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _REPLICATION_PAID_POOLS
    ]
    if not a48c._float64_bits_equal(before[unchanged], after[unchanged]):
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 changed an out-of-scope material pool'
        )
    return geometry


def _completion_candidate_object_snapshot(cell):
    if cell.replication_template is not None:
        raise A4ReplicationCompletionCommitError(
            'A4.8c2 prepared candidate retained an active template'
        )
    return {
        'cell_object': cell,
        'pools_object': cell.pools,
        'genomes_object': cell.genomes,
        'genome_objects': tuple(cell.genomes),
        'lesions_object': cell.genome_lesions,
        'copy_object': cell.replication_copy,
        'mutation_events_object': cell.mutation_events,
        'mutation_events_items': a48c._mapping_items_snapshot(
            cell.mutation_events,
        ),
        'proteins_object': cell.proteins,
        'damaged_proteins_object': cell.damaged_proteins,
        'gene_specs_object': cell.gene_specs,
        'gene_specs_items': tuple(
            (fingerprint, spec, copy.deepcopy(spec))
            for fingerprint, spec in cell.gene_specs.items()
        ),
    }


def _completion_candidate_objects_match(cell, snapshot):
    return (
        cell is snapshot['cell_object']
        and cell.pools is snapshot['pools_object']
        and cell.genomes is snapshot['genomes_object']
        and len(cell.genomes) == len(snapshot['genome_objects'])
        and all(
            current is expected for current, expected in zip(
                cell.genomes, snapshot['genome_objects'],
            )
        )
        and cell.genome_lesions is snapshot['lesions_object']
        and cell.replication_template is None
        and cell.replication_copy is snapshot['copy_object']
        and a48c._mapping_identity_matches(
            cell.mutation_events,
            snapshot['mutation_events_object'],
            snapshot['mutation_events_items'],
        )
        and cell.proteins is snapshot['proteins_object']
        and cell.damaged_proteins is snapshot['damaged_proteins_object']
        and a48b._gene_specs_snapshot_matches(cell, snapshot)
    )


@dataclass
class _A4ReplicationCompletionCommitCandidate:
    """Private one-shot completion candidate; never durable authority."""

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
    host_replay: object
    host_replay_object_id: int
    host_replay_state: object
    resident_binding: object
    resident_plan: object
    resident_artifacts: object
    plan: object
    plan_object_id: int
    plan_state: object
    candidate_cell: object
    candidate_cell_state: bytes
    candidate_objects: object
    fresh_binding: object
    fresh_artifacts: object
    live_rng: object
    rng_before_state: object
    dissipated_energy: float
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4ReplicationCompletionEventScheduler(
        a48c.A4ReplicationEventScheduler):
    """A4.8c1 scheduler plus mutation-free active completion only."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_completion_commit_active = False
        super(A4ReplicationCompletionEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def completion_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_completion_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c2 completion commit'
            )
        state = super(
            A4ReplicationCompletionEventScheduler, self,
        ).state_dict()
        state['a4_replication_completion'] = (
            self.completion_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        completion = _canonical_completion_integration_state(
            state.get('a4_replication_completion'),
        )
        replication = a48c.A4ReplicationEventScheduler.from_state(state)
        if (completion['config']
                != a48a._config_state(replication.a4_config)
                or completion['device'] != replication.a4_device):
            raise A4ReplicationCompletionCommitError(
                'saved A4.8c1/A4.8c2 config or device differs'
            )
        scheduler = cls(
            a4_config=completion['config'],
            device=completion['device'],
            max_receipts=replication.max_receipts,
        )
        scheduler._next_step_id = int(replication._next_step_id)
        scheduler._backend_name = str(replication._backend_name)
        scheduler._receipts = copy.deepcopy(replication._receipts)
        return scheduler

    def _classify_replication_branch(self, world, cell, dt, config):
        """Read the resident A4.6a scope code; never classify by exception."""
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationCompletionCommitError(
                'replication config must be the active world config object'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 replication source cell must be alive'
            )
        if not isinstance(
                getattr(config, 'genome_replication', None),
                (bool, np.bool_)) or not bool(config.genome_replication):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 disabled replication is outside scope'
            )
        mutation_enabled = a48c._strict_mutation_flag(config)
        _, source_binding = self._pack_host_binding(world, cell)
        _completion_geometry(source_binding)
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
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 dispatch probe escaped resident authority'
            )
        codes = probe.scope_error_code.detach().cpu().numpy().copy()
        completions = probe.completion_events.detach().cpu().numpy().copy()
        valid = probe.scope_valid.detach().cpu().numpy().copy()
        if (codes.shape != (int(source_binding.state.cell_capacity),)
                or completions.shape != codes.shape
                or valid.shape != codes.shape):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 dispatch probe schema differs'
            )
        if (a48c._cell_state_snapshot(cell) != source_state
                or not a48c._source_objects_match(cell, source_objects)
                or world.rng is not live_rng
                or world.rng.bit_generator.state != rng_before
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], dtype=np.float64),
                    np.asarray([energy_before], dtype=np.float64),
                )):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 dispatch probe changed live authority'
            )
        code = int(codes[0])
        completed = bool(completions[0])
        supported = bool(valid[0])
        if code == int(a44.SCOPE_OK) and completed and supported:
            if mutation_enabled:
                raise A4ReplicationCompletionCommitError(
                    'mutation-on completion is outside A4.8c2 scope'
                )
            return _DISPATCH_COMPLETION
        if (code == int(a44.SCOPE_NONCOMPLETION)
                and not completed and not supported):
            return _DISPATCH_NONCOMPLETION
        raise A4ReplicationCompletionCommitError(
            'replication branch is outside A4.8c2/A4.8c1 scope: %d' % code
        )

    def _build_completion_candidate(self, cell, plan):
        candidate = copy.deepcopy(cell)
        completed_length = int(plan.completed_lengths[0])
        completed = np.asarray(
            plan.completed_symbols[0, :completed_length], dtype=np.uint8,
        ).copy()
        candidate.pools = np.asarray(
            plan.pools_after[0], dtype=np.float64,
        ).copy()
        candidate.genomes.append(completed)
        candidate.genome_lesions.append(
            float(plan.new_genome_lesions[0])
        )
        candidate.replication_cycles += int(
            plan.replication_cycle_deltas[0]
        )
        candidate.replication_template = None
        candidate.replication_template_lesion = 0.0
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
                and a4.g2.sequence_has_novel_path(completed)):
            candidate.novel_path_first_age = float(candidate.age)
        return candidate

    def _require_completion_candidate_scope(
            self, source, candidate, plan, source_binding):
        geometry = _completion_plan_scope(source_binding, plan)
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
                )):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 candidate is not deeply isolated from live biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 candidate cell schema differs from source'
            )
        allowed = {
            'pools', 'genomes', 'genome_lesions', 'replication_template',
            'replication_template_lesion', 'replication_copy',
            'replication_fractional', 'replication_cycles', 'gene_specs',
            'novel_path_first_age', 'last_replication_symbols',
            'last_effective_error_rate', 'cumulative_proofreading_atp',
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
                raise A4ReplicationCompletionCommitError(
                    'A4.8c2 candidate changed out-of-scope field: %s' % name
                )
        completed_length = int(plan.completed_lengths[0])
        completed = np.asarray(
            plan.completed_symbols[0, :completed_length], dtype=np.uint8,
        )
        expected_novel = copy.deepcopy(source.novel_path_first_age)
        if expected_novel is None and a4.g2.sequence_has_novel_path(completed):
            expected_novel = float(source.age)
        if (len(candidate.genomes) != len(source.genomes) + 1
                or any(
                    not np.array_equal(left, right)
                    for left, right in zip(
                        candidate.genomes[:-1], source.genomes,
                    )
                )
                or not np.array_equal(candidate.genomes[-1], completed)
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
                or candidate.mutation_events != source.mutation_events
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
                )):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 candidate fields differ from resident descriptor'
            )
        if completed_length != int(geometry['template'].size):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 candidate completion length differs'
            )
        return candidate

    def _validate_fresh_completion_candidate(
            self, source, candidate, plan, source_binding, fresh_binding):
        a4._require_translation_binding(fresh_binding)
        self._require_completion_candidate_scope(
            source, candidate, plan, source_binding,
        )
        a48b._require_cell_gene_specs(candidate, fresh_binding)
        geometry = _completion_geometry(source_binding)
        source_ragged = source_binding.ragged
        fresh_ragged = fresh_binding.ragged
        genome_count = int(geometry['genome_count'])
        completed_length = int(plan.completed_lengths[0])
        completed = np.asarray(
            plan.completed_symbols[0, :completed_length], dtype=np.uint8,
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
                or int(fresh_ragged.genome_counts[0]) != genome_count + 1
                or bool(fresh_ragged.replication_active[0])
                or not np.array_equal(
                    fresh_ragged.symbols[:complete_end],
                    source_ragged.symbols[:complete_end],
                )
                or not np.array_equal(
                    fresh_ragged.symbols[
                        complete_end:complete_end + completed_length
                    ],
                    completed,
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.genome_lesions[:genome_count],
                    source_ragged.genome_lesions[:genome_count],
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.genome_lesions[
                        genome_count:genome_count + 1
                    ],
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or float(fresh_ragged.replication_template_lesions[0]) != 0.0
                or float(fresh_ragged.replication_fractional[0]) != 0.0):
            raise A4ReplicationCompletionCommitError(
                'fresh A4.8c2 ragged state differs from resident descriptor'
            )

        source_state = source_binding.state
        fresh_state = fresh_binding.state
        expected_lesion_mean = float(np.mean(np.asarray(
            candidate.genome_lesions, dtype=np.float64,
        )))
        append_count = int(plan.append_count[0])
        for item in fields(a4.A4TranslationStateBatch):
            name = item.name
            before = getattr(source_state, name)
            after = getattr(fresh_state, name)
            if name not in a4._TRANSLATION_ARRAY_FIELDS:
                if name == 'source_provenance':
                    if after != fresh_binding._ragged_provenance:
                        raise A4ReplicationCompletionCommitError(
                            'fresh A4.8c2 provenance is not self-consistent'
                        )
                elif before != after:
                    raise A4ReplicationCompletionCommitError(
                        'fresh A4.8c2 metadata changed: %s' % name
                    )
                continue
            if name == 'pools':
                expected = np.asarray(plan.pools_after, dtype=np.float64)
            elif name == 'cumulative_proofreading_atp':
                expected = np.asarray(
                    plan.cumulative_proofreading_atp_after,
                    dtype=np.float64,
                )
            elif name == 'genome_count':
                expected = np.asarray(before, dtype=np.int64).copy()
                expected[0] += 1
            elif name == 'replication_active':
                expected = np.asarray(before, dtype=bool).copy()
                expected[0] = False
            elif name == 'genome_material_symbols':
                expected = np.asarray(before, dtype=np.int64).copy()
                expected[0] += append_count
            elif name == 'genome_lesion_mean':
                expected = np.asarray(before, dtype=np.float64).copy()
                expected[0] = expected_lesion_mean
            else:
                expected = np.asarray(before)
            after = np.asarray(after)
            if expected.dtype != after.dtype or expected.shape != after.shape:
                raise A4ReplicationCompletionCommitError(
                    'fresh A4.8c2 state schema changed: %s' % name
                )
            same = (
                a48c._float64_bits_equal(expected, after)
                if expected.dtype == np.dtype(np.float64)
                else np.array_equal(expected, after)
            )
            if not same:
                raise A4ReplicationCompletionCommitError(
                    'fresh A4.8c2 state differs: %s' % name
                )
        return fresh_binding

    def _prepare_completion_candidate(self, world, cell, dt, config):
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationCompletionCommitError(
                'replication config must be the active world config object'
            )
        if a48c._strict_mutation_flag(config):
            raise A4ReplicationCompletionCommitError(
                'mutation-on completion is outside A4.8c2 scope'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 replication source cell must be alive'
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

        host_replay = a44.paid_replication_completion_plan(
            source_binding, dt, config,
        )
        _completion_plan_scope(source_binding, host_replay)

        resident_binding = a4.bind_a4_translation(
            source_binding.ragged.to_torch(device=self.a4_device),
            source_binding.state.to_torch(device=self.a4_device),
        )
        resident_plan = a44.paid_replication_completion_plan(
            resident_binding, dt, config,
        )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 resident descriptor did not remain on Torch'
            )
        resident_artifacts = a48c._resident_artifact_snapshot(
            resident_binding, None, resident_plan, self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache)
                != source_binding._cache_provenance):
            raise A4ReplicationCompletionCommitError(
                'resident A4.8c2 source differs after readback'
            )
        plan = resident_plan.to_numpy()
        _completion_plan_scope(source_binding, plan)
        a48c._plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_completion_candidate(cell, plan)
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_completion_candidate(
            cell, candidate_cell, plan, source_binding, fresh_binding,
        )
        candidate_cell_state = a48c._cell_state_snapshot(candidate_cell)
        candidate_objects = _completion_candidate_object_snapshot(
            candidate_cell,
        )
        fresh_artifacts = a48c._host_binding_artifact_snapshot(fresh_binding)
        return _A4ReplicationCompletionCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self),
            world_object_id=id(world),
            cell_object_id=id(cell),
            cell_id=int(cell.cell_id),
            generation=generation,
            dt_hex=dt.hex(),
            branch=_BRANCH_COMPLETION,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding,
            source_artifacts=source_artifacts,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects,
            host_replay=host_replay,
            host_replay_object_id=id(host_replay),
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan,
            plan_object_id=id(plan),
            plan_state=plan.state_dict(),
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding,
            fresh_artifacts=fresh_artifacts,
            live_rng=live_rng,
            rng_before_state=copy.deepcopy(rng_before),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.completion_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_completion_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(
                candidate, _A4ReplicationCompletionCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 candidate is untrusted or already consumed'
            )
        dt = a48c._strict_nonnegative_dt(dt)
        live_rng, rng_before = a48c._live_pcg64(world)
        if (id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.completion_integration_state()
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
                or a48c._strict_mutation_flag(config)
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
            raise A4ReplicationCompletionCommitError(
                'live A4.8c2 cell/config/dt/RNG changed before claim'
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
            raise A4ReplicationCompletionCommitError(
                'retained A4.8c2 host source changed before claim'
            )
        if (id(candidate.host_replay) != candidate.host_replay_object_id
                or not a48a._state_arrays_bit_exact(
                    candidate.host_replay.state_dict(),
                    candidate.host_replay_state,
                )
                or id(candidate.plan) != candidate.plan_object_id
                or not a48a._state_arrays_bit_exact(
                    candidate.plan.state_dict(), candidate.plan_state,
                )):
            raise A4ReplicationCompletionCommitError(
                'retained A4.8c2 oracle/readback changed before claim'
            )
        current_replay = a44.paid_replication_completion_plan(
            current_binding, dt, config,
        )
        _completion_plan_scope(current_binding, current_replay)
        if not a48c._plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationCompletionCommitError(
                'independent A4.8c2 NumPy replay changed before claim'
            )

        a48c._require_resident_artifact_snapshot(
            candidate.resident_binding, None, candidate.resident_plan,
            self.a4_device, candidate.resident_artifacts,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.source_binding._cache_provenance):
            raise A4ReplicationCompletionCommitError(
                'resident A4.8c2 source changed before claim'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        if not a48c._plan_states_bit_exact(resident_plan, candidate.plan):
            raise A4ReplicationCompletionCommitError(
                'resident A4.8c2 descriptor changed before claim'
            )
        _completion_plan_scope(current_binding, resident_plan)
        a48c._plan_semantic_match(resident_plan, current_replay)

        if (a48c._cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not _completion_candidate_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationCompletionCommitError(
                'fresh A4.8c2 candidate biology changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_completion_candidate(
            cell, candidate.candidate_cell, candidate.plan,
            candidate.source_binding, fresh_now,
        )
        if (a48c._host_binding_artifact_snapshot(candidate.fresh_binding)
                != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCompletionCommitError(
                'fresh A4.8c2 candidate changed before claim'
            )
        return candidate

    def _replication_completion_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after preparation and before validation."""
        return candidate

    def _publish_replication_completion_candidate(
            self, world, cell, candidate):
        """Selective completion publish from resident-readback candidate."""
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
        cell.gene_specs.clear()
        for fingerprint, spec in prepared.gene_specs.items():
            cell.gene_specs[fingerprint] = copy.deepcopy(spec)
        cell.novel_path_first_age = copy.deepcopy(
            prepared.novel_path_first_age
        )

    def _published_completion_matches(
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
                or cell.genome_lesions
                is not source_objects['lesions_object']
                or cell.replication_template is not None
                or cell.replication_copy is not source_objects['copy_object']
                or cell.replication_copy != []
                or cell.mutation_events
                is not source_objects['mutation_events_object']
                or tuple(cell.mutation_events.items())
                != tuple(source_objects['mutation_events_items'])
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
                != candidate.rng_before_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], dtype=np.float64),
                    np.asarray([snapshot['dissipated_energy']], dtype=np.float64),
                )):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCompletionCommitError(
                'published A4.8c2 binding differs from fresh candidate'
            )

    @staticmethod
    def _completion_publish_snapshot(world, cell):
        return {
            'pools_object': cell.pools,
            'pools': np.asarray(cell.pools, dtype=np.float64).copy(),
            'genomes_object': cell.genomes,
            'genome_objects': tuple(cell.genomes),
            'genomes': tuple(
                np.asarray(genome, dtype=np.uint8).copy()
                for genome in cell.genomes
            ),
            'lesions_object': cell.genome_lesions,
            'genome_lesions': tuple(cell.genome_lesions),
            'template_object': cell.replication_template,
            'template': np.asarray(
                cell.replication_template, dtype=np.uint8,
            ).copy(),
            'copy_object': cell.replication_copy,
            'replication_copy': tuple(cell.replication_copy),
            'mutation_events_object': cell.mutation_events,
            'mutation_events_items': a48c._mapping_items_snapshot(
                cell.mutation_events,
            ),
            'proteins_object': cell.proteins,
            'proteins_items': a48c._mapping_items_snapshot(cell.proteins),
            'damaged_proteins_object': cell.damaged_proteins,
            'damaged_proteins_items': a48c._mapping_items_snapshot(
                cell.damaged_proteins,
            ),
            'gene_specs_object': cell.gene_specs,
            'gene_specs_items': tuple(
                (fingerprint, spec, copy.deepcopy(spec))
                for fingerprint, spec in cell.gene_specs.items()
            ),
            'replication_fractional': float(cell.replication_fractional),
            'replication_template_lesion': float(
                cell.replication_template_lesion
            ),
            'replication_cycles': int(cell.replication_cycles),
            'novel_path_first_age': copy.deepcopy(
                cell.novel_path_first_age
            ),
            'age': float(cell.age),
            'alive': bool(cell.alive),
            'last_replication_symbols': int(cell.last_replication_symbols),
            'last_effective_error_rate': float(
                cell.last_effective_error_rate
            ),
            'cumulative_proofreading_atp': float(
                cell.cumulative_proofreading_atp
            ),
            'dissipated_energy': float(world.dissipated_energy),
            'rng_object': world.rng,
            'rng_state': copy.deepcopy(world.rng.bit_generator.state),
        }

    def _commit_completion_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_completion_candidate(
            world, cell, dt, config, candidate,
        )
        snapshot = self._completion_publish_snapshot(world, cell)
        candidate.consumed = True
        plan = candidate.plan
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c2-resident-completion-plan-atomic-cpu-cell-commit'
            ),
            'branch': candidate.branch,
            'mutation_enabled': False,
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
            self._publish_replication_completion_candidate(
                world, cell, candidate,
            )
            self._published_completion_matches(
                world, cell, candidate, snapshot,
            )
            amount = int(plan.last_replication_symbols[0])
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': amount,
                'work_performed': amount > 0,
                'requested_symbols': int(plan.requested_symbols[0]),
                'completion_events': int(plan.completion_events[0]),
                'rng_call_count': 0,
            })
        except BaseException:
            self._rollback_replication_publish(world, cell, snapshot)
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Explicitly dispatch c1 noncompletion or c2 completion."""
        self._context(world, cell, dt)
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationCompletionCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c2 replication commit is forbidden'
            )
        branch = self._classify_replication_branch(
            world, cell, dt, config,
        )
        if branch == _DISPATCH_NONCOMPLETION:
            return super(
                A4ReplicationCompletionEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        if branch != _DISPATCH_COMPLETION:
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 dispatch produced an unknown branch'
            )
        self._a4_replication_completion_commit_active = True
        try:
            candidate = self._prepare_completion_candidate(
                world, cell, dt, config,
            )
            ready = self._replication_completion_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationCompletionCommitError(
                    'A4.8c2 ready hook must retain its private candidate'
                )
            return self._commit_completion_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_completion_commit_active = False


class Hybrid066WorldA4ReplicationCompletion(
        a48c.Hybrid066WorldA4Replication):
    """A4.8c1 world retaining bounded A4.8c2 authority after restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationCompletionEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler, A4ReplicationCompletionEventScheduler):
            raise TypeError(
                'scheduler must be A4ReplicationCompletionEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationCompletion, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationCompletionCommitError(
                'new A4.8c2 world requires explicit A4 config and device'
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
            Hybrid066WorldA4ReplicationCompletion, self,
        ).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_completion_device': self.scheduler.a4_device,
            'a4_replication_completion_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(
            Hybrid066WorldA4ReplicationCompletion, self,
        ).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_completion': (
                self.scheduler.completion_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationCompletionCommitError(
                'A4.8c2 save version/build differs'
            )
        completion = _canonical_completion_integration_state(
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
                'A4.8c2 save is missing aggregate composition sidecars'
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
        scheduler = A4ReplicationCompletionEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if (scheduler.completion_integration_state() != completion
                or scheduler.replication_integration_state() != replication
                or scheduler.translation_integration_state() != translation
                or scheduler.integration_state() != hydrolysis):
            raise A4ReplicationCompletionCommitError(
                'A4.8a/b/c1/c2 world/scheduler save settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c2-mutation-free-pre-existing-active-completion-'
        'resident-descriptor-cpu-cell-atomic-commit-'
        'a4.8c1-active-noncompletion-inherited-'
        'mutation-completion-inactive-early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c2-completion-mutation-disabled-live-pcg64-unchanged-'
        'mutation-enabled-completion-not-integrated'
    ),
    'event_scheduler': (
        'a4.8c2-completion-plus-a4.8c1-noncompletion-plus-'
        'a4.8b-translation-plus-a4.8a-hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationCompletionCommitError',
    'A4ReplicationCompletionEventScheduler',
    'Hybrid066WorldA4ReplicationCompletion',
)
