# coding: utf-8
"""A4.8c1 pre-existing-active noncompletion replication commit bridge.

This deliberately bounded correctness bridge replaces the A3
``replication_cpu`` event only when a live Formal066 cell already owns a
non-empty active template and the call remains incomplete.  Mutation-free
events use the A4.4b resident paid-elongation plan.  Mutation-enabled events
use the A4.5a scalar-PCG64 tape and resident substitution plan.  Explicit
resident readback is commit authority; an independently recomputed NumPy plan
is a semantic oracle only.

Inactive-template start, completion, and frozen early no-op branches fail
before the scheduler claim.  They never fall back to the frozen CPU method.
The A4.8b translation and A4.8a hydrolysis bridges remain inherited unchanged.
All bindings, tapes, plans, and candidates are event-local and disposable.
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
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration deliberately fails closed
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c1'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | active noncompletion replication atomic commit bridge'
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c1-active-noncompletion-replication-atomic-commit'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_DETERMINISTIC = 'active-noncompletion-deterministic'
_BRANCH_SUBSTITUTION = 'active-noncompletion-substitution'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}


class A4ReplicationCommitError(a44.A4ReplicationError):
    """The bounded A4.8c1 source, resident plan, or commit failed."""


def _strict_nonnegative_dt(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise A4ReplicationCommitError(
            'A4.8c1 replication dt must be a real scalar'
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise A4ReplicationCommitError(
            'A4.8c1 replication dt must be finite and nonnegative'
        )
    return result


def _strict_mutation_flag(config):
    value = getattr(config, 'mutation', None)
    if not isinstance(value, (bool, np.bool_)):
        raise A4ReplicationCommitError(
            'A4.8c1 requires a boolean world mutation flag'
        )
    return bool(value)


def _canonical_replication_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationCommitError(
            'A4.8c1 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationCommitError('A4.8c1 save schema differs')
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationCommitError(
            'A4.8c1 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationCommitError(
            'A4.8c1 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _live_pcg64(world):
    rng = getattr(world, 'rng', None)
    if (not isinstance(rng, np.random.Generator)
            or not isinstance(rng.bit_generator, np.random.PCG64)):
        raise A4ReplicationCommitError(
            'A4.8c1 requires the active world NumPy PCG64 Generator'
        )
    try:
        state = a44._canonical_pcg64_state(
            copy.deepcopy(rng.bit_generator.state), 'live_rng_state',
        )
    except Exception as exc:
        raise A4ReplicationCommitError(
            'A4.8c1 live PCG64 state is invalid'
        ) from exc
    return rng, state


def _float64_bits_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and left.shape == right.shape
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _mapping_items_snapshot(mapping):
    if not isinstance(mapping, dict):
        raise A4ReplicationCommitError(
            'A4.8c1 mutation ledger must remain a dict'
        )
    return tuple((key, value) for key, value in mapping.items())


def _mapping_identity_matches(mapping, expected_object, expected_items):
    return (
        mapping is expected_object
        and tuple(mapping.items()) == tuple(expected_items)
    )


def _cell_state_snapshot(cell):
    return a48b._cell_state_snapshot(cell)


def _plan_states_bit_exact(left, right):
    return a48a._state_arrays_bit_exact(
        left.state_dict(), right.state_dict(),
    )


def _plan_semantic_match(resident, oracle):
    """Require exact discrete output and the registered fp64 oracle band."""
    a44.validate_a4_paid_elongation_plan(resident)
    a44.validate_a4_paid_elongation_plan(oracle)
    for item in fields(a44.A4PaidElongationPlan):
        name = item.name
        left = getattr(resident, name)
        right = getattr(oracle, name)
        if name not in a44._PLAN_ARRAY_FIELDS:
            if left != right:
                raise A4ReplicationCommitError(
                    'resident/NumPy replication metadata differs: %s' % name
                )
            continue
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise A4ReplicationCommitError(
                'resident/NumPy replication schema differs: %s' % name
            )
        if left.dtype == np.dtype(np.float64):
            if (not np.isfinite(left).all() or not np.isfinite(right).all()
                    or np.any(
                        np.abs(left - right) > REPLICATION_ORACLE_ATOL
                    )):
                raise A4ReplicationCommitError(
                    'resident/NumPy replication fp64 output differs: %s' % name
                )
        elif not np.array_equal(left, right):
            raise A4ReplicationCommitError(
                'resident/NumPy replication discrete output differs: %s' % name
            )
    return resident


def _plan_output_scope(binding, plan, mutation_enabled):
    """Prove that the plan is exactly the pre-existing partial-copy slice."""
    a4._require_translation_binding(binding)
    a44.validate_a4_paid_elongation_plan(plan)
    if int(binding.state.cell_count) != 1 or int(plan.cell_count) != 1:
        raise A4ReplicationCommitError(
            'A4.8c1 requires exactly one replication cell'
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
        raise A4ReplicationCommitError(
            'A4.8c1 plan does not bind the live source identity'
        )
    ragged = binding.ragged
    if not bool(ragged.replication_active[0]):
        raise A4ReplicationCommitError(
            'A4.8c1 inactive-template start is outside scope'
        )
    sequence_end = int(ragged.cell_sequence_offsets[1])
    if sequence_end < 2:
        raise A4ReplicationCommitError(
            'A4.8c1 active replication topology is invalid'
        )
    template_slot = sequence_end - 2
    copy_slot = sequence_end - 1
    template_length = int(
        ragged.sequence_offsets[template_slot + 1]
        - ragged.sequence_offsets[template_slot]
    )
    copy_length = int(
        ragged.sequence_offsets[copy_slot + 1]
        - ragged.sequence_offsets[copy_slot]
    )
    appended = int(plan.append_count[0])
    if (template_length <= 0 or copy_length >= template_length
            or copy_length + appended >= template_length):
        raise A4ReplicationCommitError(
            'A4.8c1 event is not a pre-existing active noncompletion'
        )
    if (bool(plan.template_start_events[0])
            or int(plan.selected_template_indices[0]) != -1
            or int(plan.template_storage_symbols[0]) != 0
            or bool(plan.completion_events[0])
            or int(plan.completed_lengths[0]) != 0
            or int(plan.replication_cycle_deltas[0]) != 0
            or int(plan.topology_sequence_deltas[0]) != 0
            or int(plan.topology_symbol_deltas[0]) != 0):
        raise A4ReplicationCommitError(
            'A4.8c1 plan crossed an excluded topology branch'
        )
    if (not mutation_enabled
            and int(plan.substitution_events[0]) != 0):
        raise A4ReplicationCommitError(
            'mutation-free A4.8c1 plan contains a substitution'
        )
    if int(plan.last_replication_symbols[0]) != appended:
        raise A4ReplicationCommitError(
            'A4.8c1 copied-symbol telemetry differs'
        )
    before = np.asarray(binding.state.pools[0], dtype=np.float64)
    after = np.asarray(plan.pools_after[0], dtype=np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _REPLICATION_PAID_POOLS
    ]
    if not _float64_bits_equal(before[unchanged], after[unchanged]):
        raise A4ReplicationCommitError(
            'A4.8c1 changed an out-of-scope material pool'
        )
    return plan


def _resident_source_readback(resident_binding):
    a4._require_translation_binding(resident_binding)
    ragged = resident_binding.ragged.to_numpy()
    state = resident_binding.state.to_numpy()
    binding = a4.bind_a4_translation(ragged, state)
    cache = resident_binding.cache.to_numpy()
    if a4._gene_cache_provenance(cache) != binding._cache_provenance:
        raise A4ReplicationCommitError(
            'resident A4.8c1 cache differs from source replay'
        )
    return binding, cache


def _resident_artifact_snapshot(binding, tape, plan, device):
    """Freeze resident object/storage identity without an implicit readback."""
    a4._require_translation_binding(binding)
    a44._validate_plan_metadata(plan)
    if tape is not None:
        a44._require_rng_tape(tape)
    expected_device = torch.device(str(device))
    batches = [
        ('ragged', binding.ragged, a4._ARRAY_FIELDS),
        ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        ('cache', binding.cache, a4._GENE_ARRAY_FIELDS),
        ('plan', plan, a44._PLAN_ARRAY_FIELDS),
    ]
    if tape is not None:
        batches.append(('tape', tape, a44._RNG_TAPE_ARRAY_FIELDS))
    result = {'object_ids': {}, 'data_ptrs': {}, 'versions': {}}
    for label, batch, names in batches:
        result['object_ids'][label] = id(batch)
        pointers = {}
        versions = {}
        for name in names:
            value = getattr(batch, name)
            if not a4._is_tensor(value):
                raise A4ReplicationCommitError(
                    'A4.8c1 resident artifact escaped Torch authority'
                )
            actual = value.device
            if (actual.type != expected_device.type
                    or (expected_device.index is not None
                        and actual.index != expected_device.index)):
                raise A4ReplicationCommitError(
                    'A4.8c1 resident artifact moved to another device'
                )
            pointers[name] = int(value.data_ptr())
            versions[name] = int(value._version)
        result['data_ptrs'][label] = pointers
        result['versions'][label] = versions
    result['object_ids'].update({
        'binding': id(binding),
        'binding_ragged': id(binding.ragged),
        'binding_state': id(binding.state),
        'binding_cache': id(binding.cache),
    })
    return result


def _require_resident_artifact_snapshot(binding, tape, plan, device, expected):
    actual = _resident_artifact_snapshot(binding, tape, plan, device)
    if actual != expected:
        raise A4ReplicationCommitError(
            'A4.8c1 resident object, device, pointer, or version changed'
        )
    return binding, tape, plan


def _host_binding_artifact_snapshot(binding):
    """Freeze host binding member/array identity and NumPy storage pointers."""
    a4._require_translation_binding(binding)
    batches = (
        ('ragged', binding.ragged, a4._ARRAY_FIELDS),
        ('state', binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        ('cache', binding.cache, a4._GENE_ARRAY_FIELDS),
    )
    result = {
        'object_ids': {
            'binding': id(binding),
            'ragged': id(binding.ragged),
            'state': id(binding.state),
            'cache': id(binding.cache),
        },
        'array_ids': {},
        'data_ptrs': {},
        'identity': a48a._binding_identity(binding),
    }
    for label, batch, names in batches:
        result['array_ids'][label] = {
            name: id(getattr(batch, name)) for name in names
        }
        result['data_ptrs'][label] = {
            name: int(np.asarray(getattr(batch, name)).__array_interface__[
                'data'
            ][0])
            for name in names
        }
    return result


def _source_object_snapshot(cell):
    template = getattr(cell, 'replication_template', None)
    if not isinstance(template, np.ndarray):
        raise A4ReplicationCommitError(
            'A4.8c1 requires a pre-existing NumPy replication template'
        )
    return {
        'pools_object': cell.pools,
        'genomes_object': cell.genomes,
        'genome_objects': tuple(cell.genomes),
        'lesions_object': cell.genome_lesions,
        'template_object': template,
        'copy_object': cell.replication_copy,
        'mutation_events_object': cell.mutation_events,
        'mutation_events_items': _mapping_items_snapshot(
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


def _source_objects_match(cell, snapshot):
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
        and cell.replication_template is snapshot['template_object']
        and cell.replication_copy is snapshot['copy_object']
        and _mapping_identity_matches(
            cell.mutation_events,
            snapshot['mutation_events_object'],
            snapshot['mutation_events_items'],
        )
        and cell.proteins is snapshot['proteins_object']
        and cell.damaged_proteins is snapshot['damaged_proteins_object']
        and a48b._gene_specs_snapshot_matches(cell, snapshot)
    )


@dataclass
class _A4ReplicationCommitCandidate:
    """Private one-shot candidate; never durable cell or RNG authority."""

    _factory_token: object
    scheduler_object_id: int
    world_object_id: int
    cell_object_id: int
    cell_id: int
    generation: int
    dt_hex: str
    branch: str
    mutation_enabled: bool
    source_binding_identity: object
    source_binding: object
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


class A4ReplicationEventScheduler(a48b.A4TranslationEventScheduler):
    """A4.8b scheduler with only bounded partial replication added."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_commit_active = False
        super(A4ReplicationEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def replication_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c1 replication commit'
            )
        state = super(A4ReplicationEventScheduler, self).state_dict()
        state['a4_replication'] = self.replication_integration_state()
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        replication = _canonical_replication_integration_state(
            state.get('a4_replication'),
        )
        translation = a48b.A4TranslationEventScheduler.from_state(state)
        if (replication['config']
                != a48a._config_state(translation.a4_config)
                or replication['device'] != translation.a4_device):
            raise A4ReplicationCommitError(
                'saved A4.8b/A4.8c1 config or device differs'
            )
        scheduler = cls(
            a4_config=replication['config'],
            device=replication['device'],
            max_receipts=translation.max_receipts,
        )
        scheduler._next_step_id = int(translation._next_step_id)
        scheduler._backend_name = str(translation._backend_name)
        scheduler._receipts = copy.deepcopy(translation._receipts)
        return scheduler

    def _build_replication_candidate(self, cell, plan):
        candidate = copy.deepcopy(cell)
        count = int(plan.append_count[0])
        suffix = [
            int(value) for value in
            np.asarray(plan.append_symbols[0, :count], dtype=np.uint8)
        ]
        candidate.pools = np.asarray(
            plan.pools_after[0], dtype=np.float64,
        ).copy()
        candidate.replication_copy.extend(suffix)
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
            raise A4ReplicationCommitError(
                'A4.8c1 source lacks the substitution ledger'
            )
        candidate.mutation_events['substitution'] += int(
            plan.substitution_events[0]
        )
        return candidate

    def _require_candidate_cell_scope(
            self, source, candidate, plan, source_binding):
        if (candidate is source
                or candidate.pools is source.pools
                or candidate.genomes is source.genomes
                or candidate.genome_lesions is source.genome_lesions
                or candidate.replication_template
                is source.replication_template
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
            raise A4ReplicationCommitError(
                'A4.8c1 candidate is not deeply isolated from live biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationCommitError(
                'A4.8c1 candidate cell schema differs from source'
            )
        allowed = {
            'pools', 'replication_copy', 'replication_fractional',
            'last_replication_symbols', 'last_effective_error_rate',
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
                raise A4ReplicationCommitError(
                    'A4.8c1 candidate changed out-of-scope field: %s' % name
                )
        count = int(plan.append_count[0])
        suffix = [
            int(value) for value in
            np.asarray(plan.append_symbols[0, :count], dtype=np.uint8)
        ]
        if (candidate.replication_copy
                != list(source.replication_copy) + suffix
                or candidate.replication_template is None
                or not np.array_equal(
                    candidate.replication_template,
                    source.replication_template,
                )
                or len(candidate.replication_copy)
                >= len(candidate.replication_template)
                or not _float64_bits_equal(
                    candidate.pools,
                    np.asarray(plan.pools_after[0], dtype=np.float64),
                )
                or not _float64_bits_equal(
                    np.asarray(
                        [candidate.replication_fractional],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [plan.replication_fractional_after[0]],
                        dtype=np.float64,
                    ),
                )
                or int(candidate.last_replication_symbols)
                != int(plan.last_replication_symbols[0])
                or not _float64_bits_equal(
                    np.asarray(
                        [candidate.last_effective_error_rate],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [plan.last_effective_error_rate[0]],
                        dtype=np.float64,
                    ),
                )
                or not _float64_bits_equal(
                    np.asarray(
                        [candidate.cumulative_proofreading_atp],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [plan.cumulative_proofreading_atp_after[0]],
                        dtype=np.float64,
                    ),
                )):
            raise A4ReplicationCommitError(
                'A4.8c1 candidate fields differ from resident plan'
            )
        expected_events = dict(source.mutation_events)
        expected_events['substitution'] += int(plan.substitution_events[0])
        if candidate.mutation_events != expected_events:
            raise A4ReplicationCommitError(
                'A4.8c1 candidate mutation ledger differs from plan'
            )
        source_specs = source_binding.cache.materialize_gene_specs_host()[0]
        if (not a4._gene_specs_exact(candidate.gene_specs, source_specs)
                or not a4._gene_specs_exact(
                    candidate.gene_specs, source.gene_specs,
                )):
            raise A4ReplicationCommitError(
                'A4.8c1 candidate changed the derived gene cache'
            )
        return candidate

    def _validate_fresh_replication_candidate(
            self, source, candidate, plan, source_binding, fresh_binding):
        a4._require_translation_binding(fresh_binding)
        if int(fresh_binding.state.cell_count) != 1:
            raise A4ReplicationCommitError(
                'fresh A4.8c1 candidate is not exactly one cell'
            )
        self._require_candidate_cell_scope(
            source, candidate, plan, source_binding,
        )
        a48b._require_cell_gene_specs(candidate, fresh_binding)
        if (a4._gene_cache_provenance(fresh_binding.cache)
                != source_binding._cache_provenance):
            raise A4ReplicationCommitError(
                'fresh A4.8c1 candidate changed its gene cache'
            )

        source_ragged = source_binding.ragged
        fresh_ragged = fresh_binding.ragged
        sequence_count = int(source_ragged.sequence_count)
        source_symbol_count = int(source_ragged.symbol_count)
        append_count = int(plan.append_count[0])
        append = np.asarray(
            plan.append_symbols[0, :append_count], dtype=np.uint8,
        )
        if (int(fresh_ragged.sequence_count) != sequence_count
                or int(fresh_ragged.symbol_count)
                != source_symbol_count + append_count
                or int(fresh_ragged.lesion_count)
                != int(source_ragged.lesion_count)
                or int(fresh_ragged.genome_counts[0])
                != int(source_ragged.genome_counts[0])
                or not bool(fresh_ragged.replication_active[0])
                or not np.array_equal(
                    fresh_ragged.sequence_offsets[:sequence_count],
                    source_ragged.sequence_offsets[:sequence_count],
                )
                or int(fresh_ragged.sequence_offsets[sequence_count])
                != int(source_ragged.sequence_offsets[sequence_count])
                + append_count
                or not np.array_equal(
                    fresh_ragged.symbols[:source_symbol_count],
                    source_ragged.symbols[:source_symbol_count],
                )
                or not np.array_equal(
                    fresh_ragged.symbols[
                        source_symbol_count:
                        source_symbol_count + append_count
                    ],
                    append,
                )
                or not _float64_bits_equal(
                    fresh_ragged.genome_lesions,
                    source_ragged.genome_lesions,
                )
                or not _float64_bits_equal(
                    fresh_ragged.replication_template_lesions,
                    source_ragged.replication_template_lesions,
                )
                or not _float64_bits_equal(
                    fresh_ragged.replication_fractional[:1],
                    np.asarray(
                        plan.replication_fractional_after[:1],
                        dtype=np.float64,
                    ),
                )):
            raise A4ReplicationCommitError(
                'fresh A4.8c1 ragged state differs from resident plan'
            )

        source_state = source_binding.state
        fresh_state = fresh_binding.state
        for item in fields(a4.A4TranslationStateBatch):
            name = item.name
            before = getattr(source_state, name)
            after = getattr(fresh_state, name)
            if name not in a4._TRANSLATION_ARRAY_FIELDS:
                if name == 'source_provenance':
                    if after != fresh_binding._ragged_provenance:
                        raise A4ReplicationCommitError(
                            'fresh A4.8c1 provenance is not self-consistent'
                        )
                elif before != after:
                    raise A4ReplicationCommitError(
                        'fresh A4.8c1 metadata changed: %s' % name
                    )
                continue
            if name == 'pools':
                expected = np.asarray(plan.pools_after, dtype=np.float64)
            elif name == 'cumulative_proofreading_atp':
                expected = np.asarray(
                    plan.cumulative_proofreading_atp_after,
                    dtype=np.float64,
                )
            elif name == 'genome_material_symbols':
                expected = np.asarray(
                    source_state.genome_material_symbols,
                    dtype=np.int64,
                ).copy()
                expected[0] += append_count
            else:
                expected = np.asarray(before)
            after = np.asarray(after)
            if expected.dtype != after.dtype or expected.shape != after.shape:
                raise A4ReplicationCommitError(
                    'fresh A4.8c1 state schema changed: %s' % name
                )
            if expected.dtype == np.dtype(np.float64):
                same = _float64_bits_equal(expected, after)
            else:
                same = np.array_equal(expected, after)
            if not same:
                raise A4ReplicationCommitError(
                    'fresh A4.8c1 state differs: %s' % name
                )
        return fresh_binding

    def _prepare_replication_candidate(self, world, cell, dt, config):
        dt = _strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationCommitError(
                'replication config must be the active world config object'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationCommitError(
                'A4.8c1 replication source cell must be alive'
            )
        _, source_binding = self._pack_host_binding(world, cell)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        source_cell_state = _cell_state_snapshot(cell)
        source_objects = _source_object_snapshot(cell)
        live_rng, rng_before = _live_pcg64(world)
        mutation_enabled = _strict_mutation_flag(config)

        if mutation_enabled:
            branch = _BRANCH_SUBSTITUTION
            tape = a44.prepare_substitution_rng_tape(
                source_binding, dt, config, rng_before,
            )
            host_replay = a44.paid_replication_substitution_plan(
                source_binding, dt, config, tape,
            )
            rng_after = a44._canonical_pcg64_state(
                copy.deepcopy(tape.rng_after_state),
                'candidate_rng_after_state',
            )
        else:
            branch = _BRANCH_DETERMINISTIC
            tape = None
            host_replay = a44.paid_replication_elongation_plan(
                source_binding, dt, config,
            )
            rng_after = copy.deepcopy(rng_before)
        _plan_output_scope(source_binding, host_replay, mutation_enabled)

        resident_ragged = source_binding.ragged.to_torch(
            device=self.a4_device,
        )
        resident_state = source_binding.state.to_torch(
            device=self.a4_device,
        )
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        if mutation_enabled:
            resident_tape = tape.to_torch(device=self.a4_device)
            resident_plan = a44.paid_replication_substitution_plan(
                resident_binding, dt, config, resident_tape,
            )
        else:
            resident_tape = None
            resident_plan = a44.paid_replication_elongation_plan(
                resident_binding, dt, config,
            )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationCommitError(
                'A4.8c1 resident plan did not remain on Torch'
            )
        resident_artifacts = _resident_artifact_snapshot(
            resident_binding, resident_tape, resident_plan,
            self.a4_device,
        )

        resident_source, resident_cache = _resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache)
                != source_binding._cache_provenance):
            raise A4ReplicationCommitError(
                'resident A4.8c1 source differs after readback'
            )
        if mutation_enabled:
            resident_tape_host = resident_tape.to_numpy()
            if not a48a._state_arrays_bit_exact(
                    resident_tape_host.state_dict(), tape.state_dict()):
                raise A4ReplicationCommitError(
                    'resident A4.8c1 RNG tape differs after readback'
                )
            a44.paid_replication_substitution_plan(
                resident_source, dt, config, resident_tape_host,
            )
        plan = resident_plan.to_numpy()
        _plan_output_scope(source_binding, plan, mutation_enabled)
        _plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_replication_candidate(cell, plan)
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_replication_candidate(
            cell, candidate_cell, plan, source_binding, fresh_binding,
        )
        candidate_cell_state = _cell_state_snapshot(candidate_cell)
        candidate_objects = _source_object_snapshot(candidate_cell)
        fresh_artifacts = _host_binding_artifact_snapshot(fresh_binding)
        return _A4ReplicationCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self),
            world_object_id=id(world),
            cell_object_id=id(cell),
            cell_id=int(cell.cell_id),
            generation=generation,
            dt_hex=dt.hex(),
            branch=branch,
            mutation_enabled=mutation_enabled,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects,
            tape=tape,
            tape_state=(None if tape is None else tape.state_dict()),
            host_replay=host_replay,
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_tape=resident_tape,
            resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan,
            plan_state=plan.state_dict(),
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding,
            fresh_artifacts=fresh_artifacts,
            live_rng=live_rng,
            rng_before_state=copy.deepcopy(rng_before),
            rng_after_state=copy.deepcopy(rng_after),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.replication_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_replication_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(candidate, _A4ReplicationCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationCommitError(
                'A4.8c1 candidate is untrusted or already consumed'
            )
        dt = _strict_nonnegative_dt(dt)
        live_rng, rng_before = _live_pcg64(world)
        if (id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.replication_integration_state()
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
                or _strict_mutation_flag(config)
                != candidate.mutation_enabled
                or live_rng is not candidate.live_rng
                or rng_before != candidate.rng_before_state
                or not _float64_bits_equal(
                    np.asarray(
                        [world.dissipated_energy], dtype=np.float64,
                    ),
                    np.asarray(
                        [candidate.dissipated_energy], dtype=np.float64,
                    ),
                )
                or _cell_state_snapshot(cell)
                != candidate.source_cell_state
                or not _source_objects_match(
                    cell, candidate.source_objects,
                )):
            raise A4ReplicationCommitError(
                'live A4.8c1 cell/config/dt/RNG identity changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity):
            raise A4ReplicationCommitError(
                'live A4.8c1 biological source changed before claim'
            )
        a4._require_translation_binding(candidate.source_binding)
        if (a48a._binding_identity(candidate.source_binding)
                != candidate.source_binding_identity):
            raise A4ReplicationCommitError(
                'retained A4.8c1 host source changed before claim'
            )

        if not a48a._state_arrays_bit_exact(
                candidate.host_replay.state_dict(),
                candidate.host_replay_state):
            raise A4ReplicationCommitError(
                'retained A4.8c1 NumPy oracle changed before claim'
            )
        if not a48a._state_arrays_bit_exact(
                candidate.plan.state_dict(), candidate.plan_state):
            raise A4ReplicationCommitError(
                'retained A4.8c1 resident readback changed before claim'
            )
        a44.validate_a4_paid_elongation_plan(candidate.host_replay)
        a44.validate_a4_paid_elongation_plan(candidate.plan)

        if candidate.mutation_enabled:
            if (candidate.branch != _BRANCH_SUBSTITUTION
                    or candidate.tape is None
                    or candidate.resident_tape is None
                    or candidate.tape_state is None):
                raise A4ReplicationCommitError(
                    'A4.8c1 substitution candidate has invalid branch state'
                )
            a44.validate_a4_substitution_rng_tape(candidate.tape)
            if not a48a._state_arrays_bit_exact(
                    candidate.tape.state_dict(), candidate.tape_state):
                raise A4ReplicationCommitError(
                    'retained A4.8c1 host RNG tape changed before claim'
                )
            if (candidate.tape.rng_before_state
                    != candidate.rng_before_state
                    or candidate.tape.rng_after_state
                    != candidate.rng_after_state):
                raise A4ReplicationCommitError(
                    'A4.8c1 candidate RNG states differ from its tape'
                )
            fresh_tape = a44.prepare_substitution_rng_tape(
                current_binding, dt, config, rng_before,
            )
            if not a48a._state_arrays_bit_exact(
                    fresh_tape.state_dict(), candidate.tape_state):
                raise A4ReplicationCommitError(
                    'fresh A4.8c1 RNG tape differs before claim'
                )
            if fresh_tape.rng_after_state != candidate.rng_after_state:
                raise A4ReplicationCommitError(
                    'fresh A4.8c1 RNG after-state differs before claim'
                )
            current_replay = a44.paid_replication_substitution_plan(
                current_binding, dt, config, fresh_tape,
            )
        else:
            if (candidate.branch != _BRANCH_DETERMINISTIC
                    or candidate.tape is not None
                    or candidate.resident_tape is not None
                    or candidate.tape_state is not None
                    or candidate.rng_after_state
                    != candidate.rng_before_state):
                raise A4ReplicationCommitError(
                    'A4.8c1 deterministic candidate has invalid branch state'
                )
            current_replay = a44.paid_replication_elongation_plan(
                current_binding, dt, config,
            )
        _plan_output_scope(
            current_binding, current_replay,
            candidate.mutation_enabled,
        )
        if not _plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationCommitError(
                'independent A4.8c1 NumPy replay changed before claim'
            )

        _require_resident_artifact_snapshot(
            candidate.resident_binding, candidate.resident_tape,
            candidate.resident_plan, self.a4_device,
            candidate.resident_artifacts,
        )
        resident_source, resident_cache = _resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.source_binding._cache_provenance):
            raise A4ReplicationCommitError(
                'resident A4.8c1 source changed before claim'
            )
        if candidate.mutation_enabled:
            resident_tape = candidate.resident_tape.to_numpy()
            if not a48a._state_arrays_bit_exact(
                    resident_tape.state_dict(), candidate.tape_state):
                raise A4ReplicationCommitError(
                    'resident A4.8c1 RNG tape changed before claim'
                )
            resident_replay = a44.paid_replication_substitution_plan(
                resident_source, dt, config, resident_tape,
            )
            if not _plan_states_bit_exact(
                    resident_replay, candidate.host_replay):
                raise A4ReplicationCommitError(
                    'resident A4.8c1 tape replay differs from host oracle'
                )
        resident_plan = candidate.resident_plan.to_numpy()
        if not _plan_states_bit_exact(resident_plan, candidate.plan):
            raise A4ReplicationCommitError(
                'resident A4.8c1 plan content changed before claim'
            )
        _plan_output_scope(
            current_binding, resident_plan,
            candidate.mutation_enabled,
        )
        _plan_semantic_match(resident_plan, current_replay)

        if (_cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not _source_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationCommitError(
                'fresh A4.8c1 candidate biology changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_replication_candidate(
            cell, candidate.candidate_cell, candidate.plan,
            candidate.source_binding, fresh_now,
        )
        a4._require_translation_binding(candidate.fresh_binding)
        if (_host_binding_artifact_snapshot(candidate.fresh_binding)
                != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCommitError(
                'fresh A4.8c1 candidate changed before claim'
            )
        return candidate

    def _replication_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after preparation and before final validation."""
        return candidate

    def _publish_replication_candidate(self, world, cell, candidate):
        """Selective active-partial publish retaining frozen CPU identities."""
        prepared = candidate.candidate_cell
        prepared_pools = np.asarray(prepared.pools, dtype=np.float64)
        for pool_index in sorted(_REPLICATION_PAID_POOLS):
            cell.pools[pool_index] = prepared_pools[pool_index]
        cell.replication_copy[:] = [
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
        if candidate.mutation_enabled:
            world.rng.bit_generator.state = copy.deepcopy(
                candidate.rng_after_state,
            )

    def _published_replication_matches(
            self, world, cell, candidate, snapshot):
        source_objects = candidate.source_objects
        if (cell.pools is not source_objects['pools_object']
                or cell.replication_copy
                is not source_objects['copy_object']
                or cell.mutation_events
                is not source_objects['mutation_events_object']
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
                or cell.replication_template
                is not source_objects['template_object']
                or cell.proteins is not source_objects['proteins_object']
                or cell.damaged_proteins
                is not source_objects['damaged_proteins_object']
                or not a48b._gene_specs_snapshot_matches(
                    cell, source_objects,
                )
                or _cell_state_snapshot(cell)
                != candidate.candidate_cell_state
                or world.rng is not candidate.live_rng
                or world.rng.bit_generator.state
                != candidate.rng_after_state
                or not _float64_bits_equal(
                    np.asarray(
                        [world.dissipated_energy], dtype=np.float64,
                    ),
                    np.asarray(
                        [snapshot['dissipated_energy']], dtype=np.float64,
                    ),
                )):
            raise A4ReplicationCommitError(
                'A4.8c1 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationCommitError(
                'published A4.8c1 binding differs from fresh candidate'
            )

    @staticmethod
    def _restore_mapping(mapping_object, items):
        mapping_object.clear()
        for key, value in items:
            mapping_object[key] = copy.deepcopy(value)
        return mapping_object

    def _rollback_replication_publish(self, world, cell, snapshot):
        pools = snapshot['pools_object']
        pools[...] = snapshot['pools']
        cell.pools = pools

        genomes = snapshot['genomes_object']
        for genome_object, saved in zip(
                snapshot['genome_objects'], snapshot['genomes']):
            genome_object[...] = saved
        genomes[:] = list(snapshot['genome_objects'])
        cell.genomes = genomes

        lesions = snapshot['lesions_object']
        lesions[:] = list(snapshot['genome_lesions'])
        cell.genome_lesions = lesions

        template = snapshot['template_object']
        template[...] = snapshot['template']
        cell.replication_template = template
        copied = snapshot['copy_object']
        copied[:] = list(snapshot['replication_copy'])
        cell.replication_copy = copied

        events = self._restore_mapping(
            snapshot['mutation_events_object'],
            snapshot['mutation_events_items'],
        )
        cell.mutation_events = events
        proteins = self._restore_mapping(
            snapshot['proteins_object'], snapshot['proteins_items'],
        )
        cell.proteins = proteins
        damaged = self._restore_mapping(
            snapshot['damaged_proteins_object'],
            snapshot['damaged_proteins_items'],
        )
        cell.damaged_proteins = damaged

        gene_specs = snapshot['gene_specs_object']
        for _, spec_object, saved_spec in snapshot['gene_specs_items']:
            spec_object.clear()
            spec_object.update(copy.deepcopy(saved_spec))
        gene_specs.clear()
        for fingerprint, spec_object, _ in snapshot['gene_specs_items']:
            gene_specs[fingerprint] = spec_object
        cell.gene_specs = gene_specs

        cell.replication_fractional = snapshot['replication_fractional']
        cell.replication_template_lesion = snapshot[
            'replication_template_lesion'
        ]
        cell.replication_cycles = snapshot['replication_cycles']
        cell.novel_path_first_age = snapshot['novel_path_first_age']
        cell.age = snapshot['age']
        cell.alive = snapshot['alive']
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

    def _commit_replication_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_replication_candidate(
            world, cell, dt, config, candidate,
        )
        candidate.consumed = True
        plan = candidate.plan
        tape = candidate.tape
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c1-resident-plan-atomic-cpu-cell-rng-commit'
            ),
            'branch': candidate.branch,
            'mutation_enabled': bool(candidate.mutation_enabled),
            'device': self.a4_device,
            'genome_count': int(candidate.source_binding.state.genome_count[0]),
            'source_provenance': str(
                candidate.source_binding.state.source_provenance
            ),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
        })
        snapshot = {
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
            'mutation_events_items': _mapping_items_snapshot(
                cell.mutation_events,
            ),
            'proteins_object': cell.proteins,
            'proteins_items': _mapping_items_snapshot(cell.proteins),
            'damaged_proteins_object': cell.damaged_proteins,
            'damaged_proteins_items': _mapping_items_snapshot(
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
        try:
            self._publish_replication_candidate(world, cell, candidate)
            self._published_replication_matches(
                world, cell, candidate, snapshot,
            )
            amount = int(plan.last_replication_symbols[0])
            rng_calls = 0
            if tape is not None:
                rng_calls = (
                    int(np.sum(tape.draw_count[:1], dtype=np.int64))
                    + int(np.sum(
                        tape.substitution_count[:1], dtype=np.int64,
                    ))
                )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': amount,
                'work_performed': amount > 0,
                'requested_symbols': int(plan.requested_symbols[0]),
                'substitution_events': int(plan.substitution_events[0]),
                'rng_call_count': rng_calls,
            })
        except BaseException:
            self._rollback_replication_publish(world, cell, snapshot)
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Replace one bounded A3 replication event with no CPU fallback."""
        self._context(world, cell, dt)
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationCommitError(
                'replication config must be the active world config object'
            )
        if self._a4_replication_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c1 replication commit is forbidden'
            )
        self._a4_replication_commit_active = True
        try:
            candidate = self._prepare_replication_candidate(
                world, cell, dt, config,
            )
            ready = self._replication_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationCommitError(
                    'A4.8c1 ready hook must retain its private candidate'
                )
            return self._commit_replication_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_commit_active = False


class Hybrid066WorldA4Replication(a48b.Hybrid066WorldA4Translation):
    """A4.8b world retaining bounded A4.8c1 authority after restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(scheduler, A4ReplicationEventScheduler):
            raise TypeError('scheduler must be A4ReplicationEventScheduler')
        super(Hybrid066WorldA4Replication, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationCommitError(
                'new A4.8c1 world requires explicit A4 config and device'
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
        output = super(Hybrid066WorldA4Replication, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_device': self.scheduler.a4_device,
            'a4_replication_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4Replication, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication': self.scheduler.replication_integration_state(),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationCommitError(
                'A4.8c1 save version/build differs'
            )
        integration = _canonical_replication_integration_state(
            state.get('a4_replication'),
        )
        translation_integration = (
            a48b._canonical_translation_integration_state(
                state.get('a4_translation'),
            )
        )
        hydrolysis_integration = a48a._canonical_integration_state(
            state.get('a4_hydrolysis'),
        )
        world = a3s.a2.s66.Formal066World.from_state(state['cpu_world'])
        if 'aggregate_sidecars' not in state:
            raise a3s.A3SchedulerProtocolError(
                'A4.8c1 save is missing aggregate composition sidecars'
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
        scheduler = A4ReplicationEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if (scheduler.replication_integration_state() != integration
                or scheduler.translation_integration_state()
                != translation_integration
                or scheduler.integration_state()
                != hydrolysis_integration):
            raise A4ReplicationCommitError(
                'A4.8a/A4.8b/A4.8c1 world/scheduler save settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48b.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c1-pre-existing-active-noncompletion-mutation-off-on-'
        'resident-plan-cpu-cell-pcg64-atomic-commit-'
        'inactive-completion-early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c1-substitution-only-live-pcg64-atomic-commit-'
        'structural-completion-not-integrated'
    ),
    'event_scheduler': (
        'a4.8c1-replication-plus-a4.8b-translation-plus-'
        'a4.8a-hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationCommitError', 'A4ReplicationEventScheduler',
    'Hybrid066WorldA4Replication',
)
