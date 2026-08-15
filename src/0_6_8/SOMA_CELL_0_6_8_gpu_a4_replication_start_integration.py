# coding: utf-8
"""A4.8c4 mutation-enabled inactive-template-start commit bridge.

This bounded bridge adds exactly one branch above A4.8c3: one alive cell
with one non-empty complete genome, no active replication template, mutation
enabled, and an A4.5b resident plan which starts but does not complete the
template in this call.  The explicit resident readback is commit authority;
the independently recomputed NumPy plan is only a strict semantic oracle.

All active-template events delegate explicitly to A4.8c3 (and therefore to
A4.8c2/A4.8c1 where appropriate).  Mutation-free starts, same-call start
completion, and ordinary pre-active no-op branches fail before claim.  There
is no frozen-CPU replication fallback.
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
import SOMA_CELL_0_6_8_gpu_a4_replication_completion_mutation_integration as a48c3
import SOMA_CELL_0_6_8_gpu_a4_replication_integration as a48c
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration fails closed without it
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c4'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | mutation-enabled inactive template-start atomic commit bridge'
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c4-mutation-enabled-inactive-template-start-'
    'noncompletion-atomic-commit'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_TEMPLATE_START = 'inactive-template-start-substitution'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}


class A4ReplicationStartCommitError(
        a48c3.A4ReplicationCompletionMutationCommitError):
    """The bounded A4.8c4 source, tape, resident plan, or commit failed."""


def _canonical_start_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationStartCommitError(
            'A4.8c4 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationStartCommitError('A4.8c4 save schema differs')
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationStartCommitError(
            'A4.8c4 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationStartCommitError(
            'A4.8c4 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _start_plan_states_bit_exact(left, right):
    return a48a._state_arrays_bit_exact(
        left.state_dict(), right.state_dict(),
    )


def _selected_template_lesion(binding):
    ragged = binding.ragged
    first = int(ragged.lesion_offsets[0])
    last = int(ragged.lesion_offsets[1])
    return (
        float(ragged.genome_lesions[first]) if last > first else 0.0
    )


def _start_plan_scope(binding, tape, plan, dt, config):
    """Prove one mutation-enabled inactive start which remains incomplete."""
    a4._require_translation_binding(binding)
    a44.validate_a4_substitution_rng_tape(tape)
    a44.validate_a4_paid_elongation_plan(plan)
    dt = a48c._strict_nonnegative_dt(dt)
    if not a48c._strict_mutation_flag(config):
        raise A4ReplicationStartCommitError(
            'A4.8c4 template start requires mutation=True'
        )
    _, config_sha256 = a44._substitution_config(config)
    a44._require_rng_tape_binding(
        tape, binding, dt, config_sha256,
    )
    state = binding.state
    ragged = binding.ragged
    if int(state.cell_count) != 1 or int(plan.cell_count) != 1:
        raise A4ReplicationStartCommitError(
            'A4.8c4 requires exactly one replication cell'
        )
    if (str(plan.source_provenance) != str(state.source_provenance)
            or str(tape.source_provenance) != str(state.source_provenance)
            or int(plan.cell_capacity) != int(state.cell_capacity)
            or int(tape.cell_capacity) != int(state.cell_capacity)
            or int(plan.append_capacity)
            != int(ragged.max_sequence_symbols)
            or int(tape.append_capacity)
            != int(ragged.max_sequence_symbols)
            or not np.array_equal(
                np.asarray(plan.cell_ids, dtype=np.int64),
                np.asarray(state.cell_ids, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(plan.cell_mask, dtype=bool),
                np.asarray(state.cell_mask, dtype=bool),
            )
            or not np.array_equal(
                np.asarray(tape.cell_ids, dtype=np.int64),
                np.asarray(state.cell_ids, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(tape.cell_mask, dtype=bool),
                np.asarray(state.cell_mask, dtype=bool),
            )):
        raise A4ReplicationStartCommitError(
            'A4.8c4 plan/tape does not bind the live source identity'
        )
    if (bool(ragged.replication_active[0])
            or bool(state.replication_active[0])
            or int(ragged.genome_counts[0]) != 1
            or int(state.genome_count[0]) != 1
            or int(ragged.sequence_count) != 1):
        raise A4ReplicationStartCommitError(
            'A4.8c4 source is not one inactive complete genome'
        )
    template_start = int(ragged.sequence_offsets[0])
    template_end = int(ragged.sequence_offsets[1])
    template_length = template_end - template_start
    append_count = int(plan.append_count[0])
    if (template_length <= 0
            or template_length > int(ragged.max_sequence_symbols)
            or append_count < 0
            or append_count >= template_length
            or not bool(plan.scope_valid[0])
            or int(plan.scope_error_code[0]) != int(a44.SCOPE_OK)
            or not bool(plan.template_start_events[0])
            or int(plan.selected_template_indices[0]) != 0
            or int(plan.template_storage_symbols[0]) != template_length
            or bool(plan.completion_events[0])
            or int(plan.completed_lengths[0]) != 0
            or int(plan.replication_cycle_deltas[0]) != 0
            or int(plan.topology_sequence_deltas[0]) != 0
            or int(plan.topology_symbol_deltas[0]) != 0
            or int(plan.last_replication_symbols[0]) != append_count
            or not bool(tape.template_start_mask[0])
            or int(tape.template_selection_indices[0]) != 0
            or int(tape.draw_count[0]) != append_count
            or int(tape.substitution_count[0])
            != int(plan.substitution_events[0])):
        raise A4ReplicationStartCommitError(
            'A4.8c4 plan crossed its template-start noncompletion scope'
        )
    future_sequences = int(ragged.sequence_count) + 2
    future_symbols = (
        int(ragged.symbol_count) + template_length + append_count
    )
    if (future_sequences > int(ragged.sequence_capacity)
            or future_symbols > int(ragged.symbol_capacity)):
        raise A4ReplicationStartCommitError(
            'A4.8c4 start exceeds fixed Q/S capacity'
        )
    before = np.asarray(state.pools[0], dtype=np.float64)
    after = np.asarray(plan.pools_after[0], dtype=np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _REPLICATION_PAID_POOLS
    ]
    if not a48c._float64_bits_equal(before[unchanged], after[unchanged]):
        raise A4ReplicationStartCommitError(
            'A4.8c4 plan changed an out-of-scope material pool'
        )
    return {
        'template': np.asarray(
            ragged.symbols[template_start:template_end], dtype=np.uint8,
        ),
        'template_length': template_length,
        'append_count': append_count,
        'template_lesion': _selected_template_lesion(binding),
        'future_sequences': future_sequences,
        'future_symbols': future_symbols,
    }


def _host_batch_artifact(label, batch, names):
    result = {'batch_id': id(batch), 'member_ids': {}, 'data_ptrs': {}}
    for name in names:
        value = getattr(batch, name)
        if not isinstance(value, np.ndarray):
            raise A4ReplicationStartCommitError(
                '%s.%s escaped NumPy host authority' % (label, name)
            )
        result['member_ids'][name] = id(value)
        result['data_ptrs'][name] = int(
            value.__array_interface__['data'][0]
        )
    return result


def _host_start_artifacts(tape, host_replay, plan):
    a44.validate_a4_substitution_rng_tape(tape)
    a44.validate_a4_paid_elongation_plan(host_replay)
    a44.validate_a4_paid_elongation_plan(plan)
    return {
        'tape': _host_batch_artifact(
            'tape', tape, a44._RNG_TAPE_ARRAY_FIELDS,
        ),
        'host_replay': _host_batch_artifact(
            'host_replay', host_replay, a44._PLAN_ARRAY_FIELDS,
        ),
        'plan': _host_batch_artifact(
            'plan', plan, a44._PLAN_ARRAY_FIELDS,
        ),
    }


def _resident_start_artifacts(binding, tape, plan, device):
    """Freeze wrappers, members, storage, device, and tensor versions."""
    if torch is None:
        raise A4ReplicationStartCommitError('PyTorch is unavailable')
    a4._require_translation_binding(binding)
    a44._require_rng_tape(tape)
    a44._validate_plan_metadata(plan)
    expected = torch.device(str(device))
    batches = (
        ('ragged', binding.ragged, a4._ARRAY_FIELDS),
        ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        ('cache', binding.cache, a4._GENE_ARRAY_FIELDS),
        ('tape', tape, a44._RNG_TAPE_ARRAY_FIELDS),
        ('plan', plan, a44._PLAN_ARRAY_FIELDS),
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
                raise A4ReplicationStartCommitError(
                    'A4.8c4 resident artifact escaped Torch authority'
                )
            actual = value.device
            if (actual.type != expected.type
                    or (expected.index is not None
                        and actual.index != expected.index)):
                raise A4ReplicationStartCommitError(
                    'A4.8c4 resident artifact moved device'
                )
            result['member_ids'][label][name] = id(value)
            result['data_ptrs'][label][name] = int(value.data_ptr())
            result['versions'][label][name] = int(value._version)
    return result


def _require_resident_start_artifacts(
        binding, tape, plan, device, expected):
    actual = _resident_start_artifacts(binding, tape, plan, device)
    if actual != expected:
        raise A4ReplicationStartCommitError(
            'A4.8c4 resident object/member/device/pointer/version changed'
        )
    return binding, tape, plan


def _inactive_source_object_snapshot(cell):
    """Snapshot an inactive source without c1's ndarray-template premise."""
    if cell.replication_template is not None:
        raise A4ReplicationStartCommitError(
            'A4.8c4 source template must be None'
        )
    if not isinstance(cell.replication_copy, list):
        raise A4ReplicationStartCommitError(
            'A4.8c4 source copy must be the frozen Python list'
        )
    return {
        'pools_object': cell.pools,
        'genomes_object': cell.genomes,
        'genome_objects': tuple(cell.genomes),
        'lesions_object': cell.genome_lesions,
        'template_object': None,
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


def _inactive_source_objects_match(cell, snapshot):
    return (
        cell.pools is snapshot['pools_object']
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
class _A4ReplicationStartCommitCandidate:
    """Private one-shot c4 candidate; never durable authority."""

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


class A4ReplicationStartEventScheduler(
        a48c3.A4ReplicationCompletionMutationEventScheduler):
    """A4.8c3 scheduler plus mutation-enabled inactive start only."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_start_commit_active = False
        super(A4ReplicationStartEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def replication_start_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_start_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c4 template-start commit'
            )
        state = super(A4ReplicationStartEventScheduler, self).state_dict()
        state['a4_replication_start'] = (
            self.replication_start_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        start = _canonical_start_integration_state(
            state.get('a4_replication_start'),
        )
        inherited = (
            a48c3.A4ReplicationCompletionMutationEventScheduler.from_state(
                state,
            )
        )
        if (start['config'] != a48a._config_state(inherited.a4_config)
                or start['device'] != inherited.a4_device):
            raise A4ReplicationStartCommitError(
                'saved A4.8c3/A4.8c4 config or device differs'
            )
        scheduler = cls(
            a4_config=start['config'], device=start['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler

    def _build_replication_start_candidate(self, cell, plan, geometry):
        candidate = copy.deepcopy(cell)
        count = int(geometry['append_count'])
        suffix = [
            int(value) for value in np.asarray(
                plan.append_symbols[0, :count], dtype=np.uint8,
            )
        ]
        candidate.pools = np.asarray(
            plan.pools_after[0], dtype=np.float64,
        ).copy()
        candidate.replication_template = np.asarray(
            geometry['template'], dtype=np.uint8,
        ).copy()
        candidate.replication_template_lesion = float(
            geometry['template_lesion']
        )
        candidate.replication_copy = list(suffix)
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
        if 'substitution' not in candidate.mutation_events:
            raise A4ReplicationStartCommitError(
                'A4.8c4 source lacks the substitution ledger'
            )
        candidate.mutation_events['substitution'] += int(
            plan.substitution_events[0]
        )
        return candidate

    def _require_replication_start_candidate_scope(
            self, source, candidate, tape, plan, source_binding,
            dt, config):
        geometry = _start_plan_scope(
            source_binding, tape, plan, dt, config,
        )
        if (candidate is source
                or candidate.pools is source.pools
                or candidate.genomes is source.genomes
                or candidate.genome_lesions is source.genome_lesions
                or candidate.replication_template is None
                or candidate.replication_template
                is source.genomes[0]
                or candidate.replication_template
                is candidate.genomes[0]
                or candidate.replication_copy is source.replication_copy
                or candidate.mutation_events is source.mutation_events
                or candidate.proteins is source.proteins
                or candidate.damaged_proteins is source.damaged_proteins
                or candidate.gene_specs is source.gene_specs
                or any(
                    left is right for left, right in zip(
                        candidate.genomes, source.genomes,
                    )
                )
                or any(
                    candidate.gene_specs.get(fingerprint)
                    is source.gene_specs.get(fingerprint)
                    for fingerprint in source.gene_specs
                )):
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate is not deeply isolated from live biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate cell schema differs from source'
            )
        allowed = {
            'pools', 'replication_template',
            'replication_template_lesion', 'replication_copy',
            'replication_fractional', 'last_replication_symbols',
            'last_effective_error_rate',
            'cumulative_proofreading_atp', 'mutation_events',
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
                raise A4ReplicationStartCommitError(
                    'A4.8c4 candidate changed out-of-scope field: ' + name
                )
        count = int(geometry['append_count'])
        suffix = [
            int(value) for value in np.asarray(
                plan.append_symbols[0, :count], dtype=np.uint8,
            )
        ]
        if (not np.array_equal(
                candidate.replication_template, geometry['template'])
                or candidate.replication_copy != suffix
                or len(candidate.replication_copy)
                >= len(candidate.replication_template)
                or not a48c._float64_bits_equal(
                    candidate.pools,
                    np.asarray(plan.pools_after[0], dtype=np.float64),
                )
                or not a48c._float64_bits_equal(
                    np.asarray(
                        [candidate.replication_template_lesion], np.float64,
                    ),
                    np.asarray([geometry['template_lesion']], np.float64),
                )
                or not a48c._float64_bits_equal(
                    np.asarray(
                        [candidate.replication_fractional], np.float64,
                    ),
                    np.asarray(
                        [plan.replication_fractional_after[0]], np.float64,
                    ),
                )
                or int(candidate.last_replication_symbols)
                != int(plan.last_replication_symbols[0])
                or not a48c._float64_bits_equal(
                    np.asarray(
                        [candidate.last_effective_error_rate], np.float64,
                    ),
                    np.asarray(
                        [plan.last_effective_error_rate[0]], np.float64,
                    ),
                )
                or not a48c._float64_bits_equal(
                    np.asarray(
                        [candidate.cumulative_proofreading_atp], np.float64,
                    ),
                    np.asarray([
                        plan.cumulative_proofreading_atp_after[0]
                    ], np.float64),
                )):
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate differs from resident start plan'
            )
        expected_events = dict(source.mutation_events)
        if 'substitution' not in expected_events:
            raise A4ReplicationStartCommitError(
                'A4.8c4 source lacks the substitution ledger'
            )
        expected_events['substitution'] += int(plan.substitution_events[0])
        if candidate.mutation_events != expected_events:
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate mutation ledger differs from plan'
            )
        source_specs = source_binding.cache.materialize_gene_specs_host()[0]
        if (not a4._gene_specs_exact(candidate.gene_specs, source_specs)
                or not a4._gene_specs_exact(
                    candidate.gene_specs, source.gene_specs,
                )):
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate changed the derived gene cache'
            )
        return geometry

    def _validate_fresh_replication_start_candidate(
            self, source, candidate, tape, plan, source_binding,
            fresh_binding, dt, config):
        a4._require_translation_binding(fresh_binding)
        if int(fresh_binding.state.cell_count) != 1:
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 candidate is not exactly one cell'
            )
        geometry = self._require_replication_start_candidate_scope(
            source, candidate, tape, plan, source_binding, dt, config,
        )
        a48b._require_cell_gene_specs(candidate, fresh_binding)
        if (a4._gene_cache_provenance(fresh_binding.cache)
                != source_binding._cache_provenance):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 candidate changed its derived gene cache'
            )
        source_ragged = source_binding.ragged
        fresh_ragged = fresh_binding.ragged
        source_symbols = int(source_ragged.symbol_count)
        template_length = int(geometry['template_length'])
        append_count = int(geometry['append_count'])
        append = np.asarray(
            plan.append_symbols[0, :append_count], dtype=np.uint8,
        )
        if (int(fresh_ragged.cell_capacity)
                != int(source_ragged.cell_capacity)
                or int(fresh_ragged.sequence_capacity)
                != int(source_ragged.sequence_capacity)
                or int(fresh_ragged.symbol_capacity)
                != int(source_ragged.symbol_capacity)
                or int(fresh_ragged.max_sequence_symbols)
                != int(source_ragged.max_sequence_symbols)
                or int(fresh_ragged.sequence_count)
                != int(geometry['future_sequences'])
                or int(fresh_ragged.symbol_count)
                != int(geometry['future_symbols'])
                or int(fresh_ragged.lesion_count)
                != int(source_ragged.lesion_count)
                or int(fresh_ragged.genome_counts[0]) != 1
                or not bool(fresh_ragged.replication_active[0])
                or int(fresh_ragged.cell_sequence_offsets[0]) != 0
                or int(fresh_ragged.cell_sequence_offsets[1]) != 3
                or not np.array_equal(
                    fresh_ragged.sequence_offsets[:2],
                    source_ragged.sequence_offsets[:2],
                )
                or int(fresh_ragged.sequence_offsets[2])
                != source_symbols + template_length
                or int(fresh_ragged.sequence_offsets[3])
                != source_symbols + template_length + append_count
                or not np.array_equal(
                    fresh_ragged.symbols[:source_symbols],
                    source_ragged.symbols[:source_symbols],
                )
                or not np.array_equal(
                    fresh_ragged.symbols[
                        source_symbols:source_symbols + template_length
                    ],
                    geometry['template'],
                )
                or not np.array_equal(
                    fresh_ragged.symbols[
                        source_symbols + template_length:
                        source_symbols + template_length + append_count
                    ],
                    append,
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.genome_lesions,
                    source_ragged.genome_lesions,
                )
                or not np.array_equal(
                    fresh_ragged.lesion_offsets,
                    source_ragged.lesion_offsets,
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.replication_template_lesions[:1],
                    np.asarray([geometry['template_lesion']], np.float64),
                )
                or not a48c._float64_bits_equal(
                    fresh_ragged.replication_fractional[:1],
                    np.asarray(
                        plan.replication_fractional_after[:1], np.float64,
                    ),
                )):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 ragged topology differs from resident plan'
            )
        source_state = source_binding.state
        fresh_state = fresh_binding.state
        if (int(fresh_state.protein_capacity)
                != int(source_state.protein_capacity)):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 protein capacity changed'
            )
        for item in fields(a4.A4TranslationStateBatch):
            name = item.name
            before = getattr(source_state, name)
            after = getattr(fresh_state, name)
            if name not in a4._TRANSLATION_ARRAY_FIELDS:
                if name == 'source_provenance':
                    if after != fresh_binding._ragged_provenance:
                        raise A4ReplicationStartCommitError(
                            'fresh A4.8c4 provenance is not self-consistent'
                        )
                elif before != after:
                    raise A4ReplicationStartCommitError(
                        'fresh A4.8c4 metadata changed: ' + name
                    )
                continue
            if name == 'pools':
                expected = np.asarray(plan.pools_after, np.float64)
            elif name == 'cumulative_proofreading_atp':
                expected = np.asarray(
                    plan.cumulative_proofreading_atp_after, np.float64,
                )
            elif name == 'replication_active':
                expected = np.asarray(before, dtype=bool).copy()
                expected[0] = True
            elif name == 'genome_material_symbols':
                expected = np.asarray(before, dtype=np.int64).copy()
                expected[0] += append_count
            else:
                expected = np.asarray(before)
            after = np.asarray(after)
            if expected.dtype != after.dtype or expected.shape != after.shape:
                raise A4ReplicationStartCommitError(
                    'fresh A4.8c4 state schema changed: ' + name
                )
            same = (
                a48c._float64_bits_equal(expected, after)
                if expected.dtype == np.dtype(np.float64)
                else np.array_equal(expected, after)
            )
            if not same:
                raise A4ReplicationStartCommitError(
                    'fresh A4.8c4 state differs: ' + name
                )
        return fresh_binding

    def _prepare_replication_start_candidate(
            self, world, cell, dt, config):
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationStartCommitError(
                'replication config must be the active world config object'
            )
        if not a48c._strict_mutation_flag(config):
            raise A4ReplicationStartCommitError(
                'A4.8c4 inactive template start requires mutation=True'
            )
        if not isinstance(
                getattr(config, 'genome_replication', None),
                (bool, np.bool_)) or not bool(config.genome_replication):
            raise A4ReplicationStartCommitError(
                'A4.8c4 disabled replication is outside scope'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationStartCommitError(
                'A4.8c4 replication source cell must be alive'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        source_objects = _inactive_source_object_snapshot(cell)
        _, source_binding = self._pack_host_binding(world, cell)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        source_artifacts = a48c._host_binding_artifact_snapshot(
            source_binding,
        )
        source_cell_state = a48c._cell_state_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)
        tape = a44.prepare_substitution_rng_tape(
            source_binding, dt, config, rng_before,
        )
        host_replay = a44.paid_replication_substitution_plan(
            source_binding, dt, config, tape,
        )
        geometry = _start_plan_scope(
            source_binding, tape, host_replay, dt, config,
        )

        resident_binding = a4.bind_a4_translation(
            source_binding.ragged.to_torch(device=self.a4_device),
            source_binding.state.to_torch(device=self.a4_device),
        )
        resident_tape = tape.to_torch(device=self.a4_device)
        resident_plan = a44.paid_replication_substitution_plan(
            resident_binding, dt, config, resident_tape,
        )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationStartCommitError(
                'A4.8c4 resident plan escaped Torch authority'
            )
        resident_artifacts = _resident_start_artifacts(
            resident_binding, resident_tape, resident_plan,
            self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache)
                != source_binding._cache_provenance):
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 source differs after readback'
            )
        resident_tape_host = resident_tape.to_numpy()
        if not a48a._state_arrays_bit_exact(
                resident_tape_host.state_dict(), tape.state_dict()):
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 tape differs after readback'
            )
        a44.paid_replication_substitution_plan(
            resident_source, dt, config, resident_tape_host,
        )
        plan = resident_plan.to_numpy()
        _start_plan_scope(source_binding, tape, plan, dt, config)
        a48c._plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_replication_start_candidate(
            cell, plan, geometry,
        )
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_replication_start_candidate(
            cell, candidate_cell, tape, plan, source_binding,
            fresh_binding, dt, config,
        )
        candidate_cell_state = a48c._cell_state_snapshot(candidate_cell)
        candidate_objects = a48c._source_object_snapshot(candidate_cell)
        fresh_artifacts = a48c._host_binding_artifact_snapshot(
            fresh_binding,
        )
        host_artifacts = _host_start_artifacts(
            tape, host_replay, plan,
        )
        return _A4ReplicationStartCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self), world_object_id=id(world),
            cell_object_id=id(cell), cell_id=int(cell.cell_id),
            generation=generation, dt_hex=dt.hex(),
            branch=_BRANCH_TEMPLATE_START,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding,
            source_artifacts=source_artifacts,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects,
            tape=tape, tape_state=tape.state_dict(),
            host_replay=host_replay,
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_tape=resident_tape,
            resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan, plan_state=plan.state_dict(),
            host_artifacts=host_artifacts,
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding,
            fresh_artifacts=fresh_artifacts,
            live_rng=live_rng,
            rng_before_state=copy.deepcopy(rng_before),
            rng_after_state=copy.deepcopy(tape.rng_after_state),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.replication_start_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_replication_start_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(candidate, _A4ReplicationStartCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationStartCommitError(
                'A4.8c4 candidate is untrusted or already consumed'
            )
        dt = a48c._strict_nonnegative_dt(dt)
        live_rng, rng_before = a48c._live_pcg64(world)
        if (candidate.branch != _BRANCH_TEMPLATE_START
                or id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.replication_start_integration_state()
                != candidate.integration_state
                or a48a._world_config_snapshot(world)
                != candidate.world_config_snapshot
                or id(cell) != candidate.cell_object_id
                or int(cell.cell_id) != candidate.cell_id
                or a48a._strict_counter(
                    getattr(cell, 'generation', None), 'cell generation',
                ) != candidate.generation
                or not bool(getattr(cell, 'alive', False))
                or not a48c._strict_mutation_flag(config)
                or dt.hex() != candidate.dt_hex
                or live_rng is not candidate.live_rng
                or rng_before != candidate.rng_before_state
                or candidate.tape.rng_before_state
                != candidate.rng_before_state
                or candidate.tape.rng_after_state
                != candidate.rng_after_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([candidate.dissipated_energy], np.float64),
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.source_cell_state
                or not _inactive_source_objects_match(
                    cell, candidate.source_objects,
                )):
            raise A4ReplicationStartCommitError(
                'live A4.8c4 cell/config/dt/RNG changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity):
            raise A4ReplicationStartCommitError(
                'live A4.8c4 biological source changed before claim'
            )
        a4._require_translation_binding(candidate.source_binding)
        if (a48a._binding_identity(candidate.source_binding)
                != candidate.source_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.source_binding,
                ) != candidate.source_artifacts):
            raise A4ReplicationStartCommitError(
                'retained A4.8c4 host source changed before claim'
            )
        a44.validate_a4_substitution_rng_tape(candidate.tape)
        a44.validate_a4_paid_elongation_plan(candidate.host_replay)
        a44.validate_a4_paid_elongation_plan(candidate.plan)
        if (not a48a._state_arrays_bit_exact(
                candidate.tape.state_dict(), candidate.tape_state)
                or not a48a._state_arrays_bit_exact(
                    candidate.host_replay.state_dict(),
                    candidate.host_replay_state,
                )
                or not a48a._state_arrays_bit_exact(
                    candidate.plan.state_dict(), candidate.plan_state,
                )
                or _host_start_artifacts(
                    candidate.tape, candidate.host_replay, candidate.plan,
                ) != candidate.host_artifacts):
            raise A4ReplicationStartCommitError(
                'retained A4.8c4 host tape/oracle/readback changed'
            )
        fresh_tape = a44.prepare_substitution_rng_tape(
            current_binding, dt, config, rng_before,
        )
        if (not a48a._state_arrays_bit_exact(
                fresh_tape.state_dict(), candidate.tape_state)
                or fresh_tape.rng_after_state
                != candidate.rng_after_state):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 tape or after-state differs before claim'
            )
        current_replay = a44.paid_replication_substitution_plan(
            current_binding, dt, config, fresh_tape,
        )
        _start_plan_scope(
            current_binding, fresh_tape, current_replay, dt, config,
        )
        if not _start_plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationStartCommitError(
                'independent A4.8c4 NumPy replay changed before claim'
            )

        _require_resident_start_artifacts(
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
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 source changed before claim'
            )
        resident_tape = candidate.resident_tape.to_numpy()
        if not a48a._state_arrays_bit_exact(
                resident_tape.state_dict(), candidate.tape_state):
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 tape changed before claim'
            )
        resident_replay = a44.paid_replication_substitution_plan(
            resident_source, dt, config, resident_tape,
        )
        if not _start_plan_states_bit_exact(
                resident_replay, candidate.host_replay):
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 tape replay differs from NumPy oracle'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        if not _start_plan_states_bit_exact(resident_plan, candidate.plan):
            raise A4ReplicationStartCommitError(
                'resident A4.8c4 plan content changed before claim'
            )
        _start_plan_scope(
            current_binding, fresh_tape, resident_plan, dt, config,
        )
        a48c._plan_semantic_match(resident_plan, current_replay)

        if (a48c._cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not a48c._source_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 candidate biology changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_replication_start_candidate(
            cell, candidate.candidate_cell, candidate.tape,
            candidate.plan, candidate.source_binding, fresh_now,
            dt, config,
        )
        a4._require_translation_binding(candidate.fresh_binding)
        if (a48c._host_binding_artifact_snapshot(candidate.fresh_binding)
                != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCommitError(
                'fresh A4.8c4 candidate changed before claim'
            )
        return candidate

    def _replication_start_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after preparation and before final trust."""
        return candidate

    def _publish_replication_start_candidate(
            self, world, cell, candidate):
        """Publish only the frozen CPU start write-set, without aliasing."""
        prepared = candidate.candidate_cell
        prepared_pools = np.asarray(prepared.pools, dtype=np.float64)
        for pool_index in sorted(_REPLICATION_PAID_POOLS):
            cell.pools[pool_index] = prepared_pools[pool_index]
        cell.replication_template = np.asarray(
            prepared.replication_template, dtype=np.uint8,
        ).copy()
        cell.replication_template_lesion = float(
            prepared.replication_template_lesion
        )
        cell.replication_copy = [
            int(value) for value in prepared.replication_copy
        ]
        cell.replication_fractional = float(
            prepared.replication_fractional
        )
        cell.last_replication_symbols = int(
            prepared.last_replication_symbols
        )
        cell.last_effective_error_rate = float(
            prepared.last_effective_error_rate
        )
        cell.cumulative_proofreading_atp = float(
            prepared.cumulative_proofreading_atp
        )
        cell.mutation_events['substitution'] = int(
            prepared.mutation_events['substitution']
        )
        world.rng.bit_generator.state = copy.deepcopy(
            candidate.rng_after_state,
        )

    def _replication_start_publish_snapshot(self, world, cell):
        if cell.replication_template is not None:
            raise A4ReplicationStartCommitError(
                'A4.8c4 publish snapshot requires an inactive template'
            )
        return {
            'pools_object': cell.pools,
            'pools': np.asarray(cell.pools, dtype=np.float64).copy(),
            'template_object': None,
            'copy_object': cell.replication_copy,
            'replication_copy': tuple(cell.replication_copy),
            'mutation_events_object': cell.mutation_events,
            'mutation_events_items': a48c._mapping_items_snapshot(
                cell.mutation_events,
            ),
            'replication_template_lesion': float(
                cell.replication_template_lesion
            ),
            'replication_fractional': float(cell.replication_fractional),
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

    def _published_replication_start_matches(
            self, world, cell, candidate, snapshot):
        source_objects = candidate.source_objects
        if (cell.pools is not source_objects['pools_object']
                or cell.genomes is not source_objects['genomes_object']
                or len(cell.genomes)
                != len(source_objects['genome_objects'])
                or any(
                    current is not expected for current, expected in zip(
                        cell.genomes, source_objects['genome_objects'],
                    )
                )
                or cell.genome_lesions
                is not source_objects['lesions_object']
                or not isinstance(cell.replication_template, np.ndarray)
                or cell.replication_template
                is candidate.candidate_cell.replication_template
                or any(
                    cell.replication_template is genome
                    for genome in cell.genomes
                )
                or cell.replication_copy
                is source_objects['copy_object']
                or cell.replication_copy
                is candidate.candidate_cell.replication_copy
                or cell.mutation_events
                is not source_objects['mutation_events_object']
                or cell.proteins is not source_objects['proteins_object']
                or cell.damaged_proteins
                is not source_objects['damaged_proteins_object']
                or not a48b._gene_specs_snapshot_matches(
                    cell, source_objects,
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.candidate_cell_state
                or world.rng is not candidate.live_rng
                or world.rng.bit_generator.state
                != candidate.rng_after_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([snapshot['dissipated_energy']], np.float64),
                )):
            raise A4ReplicationStartCommitError(
                'A4.8c4 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCommitError(
                'published A4.8c4 binding differs from fresh candidate'
            )

    def _rollback_replication_start_publish(self, world, cell, snapshot):
        """Restore the inactive ``None`` template and original list object."""
        pools = snapshot['pools_object']
        pools[...] = snapshot['pools']
        cell.pools = pools
        cell.replication_template = None
        copied = snapshot['copy_object']
        copied[:] = list(snapshot['replication_copy'])
        cell.replication_copy = copied
        events = snapshot['mutation_events_object']
        events.clear()
        for key, value in snapshot['mutation_events_items']:
            events[key] = copy.deepcopy(value)
        cell.mutation_events = events
        cell.replication_template_lesion = snapshot[
            'replication_template_lesion'
        ]
        cell.replication_fractional = snapshot['replication_fractional']
        cell.last_replication_symbols = snapshot['last_replication_symbols']
        cell.last_effective_error_rate = snapshot[
            'last_effective_error_rate'
        ]
        cell.cumulative_proofreading_atp = snapshot[
            'cumulative_proofreading_atp'
        ]
        world.dissipated_energy = snapshot['dissipated_energy']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])

    def _commit_replication_start_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_replication_start_candidate(
            world, cell, dt, config, candidate,
        )
        snapshot = self._replication_start_publish_snapshot(world, cell)
        candidate.consumed = True
        plan = candidate.plan
        tape = candidate.tape
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c4-resident-template-start-plan-atomic-'
                'cpu-cell-pcg64-commit'
            ),
            'branch': candidate.branch,
            'mutation_enabled': True,
            'device': self.a4_device,
            'genome_count': int(
                candidate.source_binding.state.genome_count[0]
            ),
            'source_provenance': str(
                candidate.source_binding.state.source_provenance
            ),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
        })
        try:
            self._publish_replication_start_candidate(
                world, cell, candidate,
            )
            self._published_replication_start_matches(
                world, cell, candidate, snapshot,
            )
            amount = int(plan.last_replication_symbols[0])
            rng_calls = (
                1
                + int(np.sum(tape.draw_count[:1], dtype=np.int64))
                + int(np.sum(
                    tape.substitution_count[:1], dtype=np.int64,
                ))
            )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': amount,
                'work_performed': True,
                'template_start_events': 1,
                'requested_symbols': int(plan.requested_symbols[0]),
                'substitution_events': int(plan.substitution_events[0]),
                'rng_call_count': rng_calls,
            })
        except BaseException:
            self._rollback_replication_start_publish(world, cell, snapshot)
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Delegate active sources or commit one bounded inactive start."""
        _, record = self._context(world, cell, dt)
        if 'replication_cpu' in record['claimed']:
            raise a3s.A3DuplicateEventError(
                'duplicate A4.8c4 replication_cpu event'
            )
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationStartCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_start_commit_active
                or self._a4_replication_completion_mutation_commit_active
                or self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c4 replication commit is forbidden'
            )
        if getattr(cell, 'replication_template', None) is not None:
            return super(
                A4ReplicationStartEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        if not a48c._strict_mutation_flag(config):
            raise A4ReplicationStartCommitError(
                'mutation-free inactive template start is outside A4.8c4'
            )
        self._a4_replication_start_commit_active = True
        try:
            candidate = self._prepare_replication_start_candidate(
                world, cell, dt, config,
            )
            ready = self._replication_start_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationStartCommitError(
                    'A4.8c4 ready hook must retain its private candidate'
                )
            return self._commit_replication_start_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_start_commit_active = False


class Hybrid066WorldA4ReplicationStart(
        a48c3.Hybrid066WorldA4ReplicationCompletionMutation):
    """A4.8c3 world retaining bounded A4.8c4 authority after restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationStartEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(scheduler, A4ReplicationStartEventScheduler):
            raise TypeError(
                'scheduler must be A4ReplicationStartEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationStart, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationStartCommitError(
                'new A4.8c4 world requires explicit A4 config and device'
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
        output = super(Hybrid066WorldA4ReplicationStart, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_start_device': self.scheduler.a4_device,
            'a4_replication_start_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4ReplicationStart, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_start': (
                self.scheduler.replication_start_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationStartCommitError(
                'A4.8c4 save version/build differs'
            )
        start = _canonical_start_integration_state(
            state.get('a4_replication_start'),
        )
        mutation = a48c3._canonical_completion_mutation_integration_state(
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
                'A4.8c4 save is missing aggregate composition sidecars'
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
        scheduler = A4ReplicationStartEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if (scheduler.replication_start_integration_state() != start
                or scheduler.completion_mutation_integration_state()
                != mutation
                or scheduler.completion_integration_state() != completion
                or scheduler.replication_integration_state() != replication
                or scheduler.translation_integration_state() != translation
                or scheduler.integration_state() != hydrolysis):
            raise A4ReplicationStartCommitError(
                'A4.8a/b/c1/c2/c3/c4 world/scheduler settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c3.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c4-mutation-enabled-inactive-one-genome-template-start-'
        'noncompletion-resident-plan-cpu-cell-pcg64-atomic-commit-'
        'a4.8c3-c2-c1-active-branches-inherited-other-inactive-'
        'early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c4-template-selection-then-substitution-same-live-pcg64-'
        'atomic-commit-start-completion-not-integrated'
    ),
    'event_scheduler': (
        'a4.8c4-template-start-plus-a4.8c3-completion-mutation-plus-'
        'a4.8c2-completion-plus-a4.8c1-noncompletion-plus-a4.8b-'
        'translation-plus-a4.8a-hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationStartCommitError', 'A4ReplicationStartEventScheduler',
    'Hybrid066WorldA4ReplicationStart',
)
