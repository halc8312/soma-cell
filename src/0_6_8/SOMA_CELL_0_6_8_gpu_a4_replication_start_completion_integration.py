# coding: utf-8
"""A4.8c6 mutation-free inactive-start same-call completion bridge.

The live source remains the frozen inactive one-genome cell.  An event-local
synthetic active-start cell supplies a non-alias selected template, its lesion
or the frozen zero fallback, an empty copy, and zero fractional carry to the
public A4.6a completion-only plan.  Torch resident readback is commit authority;
the independently rebuilt NumPy plan is only the bounded fp64 oracle.

Only the Formal066 ``integers(0, 1)`` selection is replayed.  Mutation draws
are forbidden and mutation-ledger state is preserved.  Active, mutation-enabled inactive,
and mutation-free start/noncompletion sources delegate to A4.8c5 and earlier
bridges.  Other pre-active branches fail before claim without CPU fallback.
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
import SOMA_CELL_0_6_8_gpu_a4_replication_start_integration as a48c4
import SOMA_CELL_0_6_8_gpu_a4_replication_start_mutation_free_integration as a48c5
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration fails closed without it
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c6'
BUILD_ID = BUILD
BUILD_LONG = (
    BUILD + ' | mutation-free inactive-start same-call-completion bridge'
)
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c6-mutation-free-inactive-template-start-'
    'same-call-completion-atomic-commit'
)
SELECTION_EVIDENCE_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c6-single-template-completion-selection-evidence'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_SELECTION_EVIDENCE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_START_COMPLETION = 'inactive-template-start-completion-deterministic'
_DISPATCH_START_COMPLETION = 'start-completion'
_DISPATCH_START_NONCOMPLETION = 'start-noncompletion'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}


class A4ReplicationStartCompletionCommitError(
        a48c5.A4ReplicationMutationFreeStartCommitError):
    """The bounded A4.8c6 source, synthetic plan, or commit failed."""


def _canonical_start_completion_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 save schema differs'
        )
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _inactive_start_completion_geometry(binding):
    """Prove the exact inactive one-genome source and fixed capacities."""
    a4._require_translation_binding(binding)
    state = binding.state
    ragged = binding.ragged
    if (int(state.cell_count) != 1
            or int(ragged.cell_count) != 1
            or int(state.cell_capacity) != int(ragged.cell_capacity)
            or int(ragged.sequence_count) != 1
            or int(ragged.genome_counts[0]) != 1
            or int(state.genome_count[0]) != 1
            or bool(ragged.replication_active[0])
            or bool(state.replication_active[0])
            or int(ragged.cell_sequence_offsets[0]) != 0
            or int(ragged.cell_sequence_offsets[1]) != 1):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 requires exactly one inactive complete genome'
        )
    first = int(ragged.sequence_offsets[0])
    last = int(ragged.sequence_offsets[1])
    template_length = last - first
    if (first != 0
            or last != int(ragged.symbol_count)
            or template_length <= 0
            or template_length > int(ragged.max_sequence_symbols)):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selected complete genome is empty or noncanonical'
        )
    synthetic_sequences = int(ragged.sequence_count) + 2
    final_sequences = int(ragged.sequence_count) + 1
    synthetic_symbols = int(ragged.symbol_count) + template_length
    final_symbols = int(ragged.symbol_count) + template_length
    if (synthetic_sequences > int(ragged.sequence_capacity)
            or final_sequences > int(ragged.sequence_capacity)
            or synthetic_symbols > int(ragged.symbol_capacity)
            or final_symbols > int(ragged.symbol_capacity)):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 source exceeds exact Q/S transaction capacity'
        )
    protein_count = np.asarray(state.active_count, dtype=np.int64)
    damaged_count = np.asarray(state.damaged_count, dtype=np.int64)
    if (int(protein_count[0]) > int(state.protein_capacity)
            or int(damaged_count[0]) > int(state.protein_capacity)):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 source exceeds exact protein capacity'
        )
    lesion_first = int(ragged.lesion_offsets[0])
    lesion_last = int(ragged.lesion_offsets[1])
    if lesion_first != 0 or lesion_last not in (0, 1):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 one-genome lesion topology is noncanonical'
        )
    return {
        'template': np.asarray(
            ragged.symbols[first:last], dtype=np.uint8,
        ).copy(),
        'template_length': template_length,
        'template_lesion': (
            float(ragged.genome_lesions[lesion_first])
            if lesion_last > lesion_first else 0.0
        ),
        'source_sequences': int(ragged.sequence_count),
        'source_symbols': int(ragged.symbol_count),
        'source_lesions': int(ragged.lesion_count),
        'synthetic_sequences': synthetic_sequences,
        'synthetic_symbols': synthetic_symbols,
        'final_sequences': final_sequences,
        'final_symbols': final_symbols,
    }


@dataclass
class _A4SingleTemplateCompletionSelectionEvidence:
    """Private host proof of one frozen high-level selection call."""

    _factory_token: object
    schema_version: str
    source_binding_identity: object
    cell_id: int
    dt_hex: str
    config_sha256: str
    low: int
    high: int
    selected_index: int
    call_count: int
    rng_before_state: object
    rng_after_state: object
    schedule_sha256: str

    def state_dict(self):
        return {
            'schema_version': str(self.schema_version),
            'source_binding_identity': tuple(self.source_binding_identity),
            'cell_id': int(self.cell_id),
            'dt_hex': str(self.dt_hex),
            'config_sha256': str(self.config_sha256),
            'low': int(self.low),
            'high': int(self.high),
            'selected_index': int(self.selected_index),
            'call_count': int(self.call_count),
            'rng_before_state': copy.deepcopy(self.rng_before_state),
            'rng_after_state': copy.deepcopy(self.rng_after_state),
            'schedule_sha256': str(self.schedule_sha256),
        }


def _completion_selection_schedule_sha256(evidence):
    before = a48c5._canonical_selection_pcg64_state(
        evidence.rng_before_state, 'completion_selection_rng_before_state',
    )
    after = a48c5._canonical_selection_pcg64_state(
        evidence.rng_after_state, 'completion_selection_rng_after_state',
    )
    return a44._sha256_json({
        'schema': str(evidence.schema_version),
        'source_binding_identity': list(evidence.source_binding_identity),
        'cell_id': int(evidence.cell_id),
        'dt_hex': str(evidence.dt_hex),
        'config_sha256': str(evidence.config_sha256),
        'low': int(evidence.low),
        'high': int(evidence.high),
        'selected_index': int(evidence.selected_index),
        'call_count': int(evidence.call_count),
        'rng_before_sha256': a44._sha256_json(before),
        'rng_after_sha256': a44._sha256_json(after),
    })


def _completion_selection_frozen_sha256(evidence):
    return a44._sha256_json(evidence.state_dict())


def _prepare_single_template_completion_selection_evidence(
        binding, dt, config, rng_state_before):
    """Replay exactly one ``integers(0, 1)`` on a cloned PCG64."""
    a4._require_translation_binding(binding)
    dt = a48c._strict_nonnegative_dt(dt)
    _inactive_start_completion_geometry(binding)
    config_sha256 = a48c5._mutation_free_start_config_sha256(config)
    before = a48c5._canonical_selection_pcg64_state(
        rng_state_before, 'completion_selection_rng_before_state',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    selected = a48c5._formal066_single_template_selection(generator)
    if selected != 0:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 single-template selection did not return index zero'
        )
    after = a48c5._canonical_selection_pcg64_state(
        generator.bit_generator.state,
        'completion_selection_rng_after_state',
    )
    evidence = _A4SingleTemplateCompletionSelectionEvidence(
        _factory_token=_SELECTION_EVIDENCE_FACTORY_TOKEN,
        schema_version=SELECTION_EVIDENCE_SCHEMA_VERSION,
        source_binding_identity=a48a._binding_identity(binding),
        cell_id=int(binding.state.cell_ids[0]), dt_hex=dt.hex(),
        config_sha256=config_sha256, low=0, high=1,
        selected_index=selected, call_count=1,
        rng_before_state=before, rng_after_state=after,
        schedule_sha256='0' * 64,
    )
    evidence.schedule_sha256 = _completion_selection_schedule_sha256(
        evidence,
    )
    evidence._frozen_sha256 = _completion_selection_frozen_sha256(evidence)
    return _validate_single_template_completion_selection_evidence(
        evidence, binding, dt, config,
    )


def _validate_single_template_completion_selection_evidence(
        evidence, binding, dt, config):
    a4._require_translation_binding(binding)
    dt = a48c._strict_nonnegative_dt(dt)
    _inactive_start_completion_geometry(binding)
    if (not isinstance(
                evidence, _A4SingleTemplateCompletionSelectionEvidence)
            or evidence._factory_token
            is not _SELECTION_EVIDENCE_FACTORY_TOKEN
            or evidence.schema_version
            != SELECTION_EVIDENCE_SCHEMA_VERSION):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selection evidence is not factory-authenticated'
        )
    for name in ('cell_id', 'low', 'high', 'selected_index', 'call_count'):
        value = getattr(evidence, name)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 selection evidence %s is not an integer' % name
            )
    identity = a48a._binding_identity(binding)
    if (not isinstance(evidence.source_binding_identity, tuple)
            or len(evidence.source_binding_identity) != 3
            or not all(
                a44._is_lower_hex_digest(value)
                for value in evidence.source_binding_identity
            )
            or tuple(evidence.source_binding_identity) != identity
            or int(evidence.cell_id) != int(binding.state.cell_ids[0])
            or evidence.dt_hex != dt.hex()
            or not a44._is_lower_hex_digest(evidence.config_sha256)
            or not a44._is_lower_hex_digest(evidence.schedule_sha256)
            or evidence.config_sha256
            != a48c5._mutation_free_start_config_sha256(config)
            or (int(evidence.low), int(evidence.high),
                int(evidence.selected_index), int(evidence.call_count))
            != (0, 1, 0, 1)):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selection evidence differs from source/config/dt'
        )
    before = a48c5._canonical_selection_pcg64_state(
        evidence.rng_before_state, 'completion_selection_rng_before_state',
    )
    after = a48c5._canonical_selection_pcg64_state(
        evidence.rng_after_state, 'completion_selection_rng_after_state',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    if a48c5._formal066_single_template_selection(generator) != 0:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selection evidence replay returned another index'
        )
    if generator.bit_generator.state != after:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selection evidence after-state differs from replay'
        )
    if (evidence.schedule_sha256
            != _completion_selection_schedule_sha256(evidence)
            or getattr(evidence, '_frozen_sha256', None)
            != _completion_selection_frozen_sha256(evidence)):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 selection evidence changed after creation'
        )
    return evidence


def _build_synthetic_active_start_cell(source, geometry):
    synthetic = copy.deepcopy(source)
    synthetic.replication_template = np.asarray(
        geometry['template'], dtype=np.uint8,
    ).copy()
    synthetic.replication_template_lesion = float(
        geometry['template_lesion']
    )
    synthetic.replication_copy = []
    synthetic.replication_fractional = 0.0
    return synthetic


def _synthetic_active_start_objects(cell):
    try:
        return a48c._source_object_snapshot(cell)
    except Exception as exc:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic active-start objects are noncanonical'
        ) from exc


def _require_synthetic_active_start_binding(
        source, synthetic, source_binding, synthetic_binding, geometry):
    """Bind the event-local active representation to the live source."""
    a4._require_translation_binding(source_binding)
    a4._require_translation_binding(synthetic_binding)
    if (synthetic is source
            or synthetic.pools is source.pools
            or synthetic.genomes is source.genomes
            or synthetic.genome_lesions is source.genome_lesions
            or synthetic.replication_template is None
            or synthetic.replication_template is source.genomes[0]
            or synthetic.replication_copy is source.replication_copy
            or synthetic.mutation_events is source.mutation_events
            or synthetic.proteins is source.proteins
            or synthetic.damaged_proteins is source.damaged_proteins
            or synthetic.gene_specs is source.gene_specs
            or any(
                left is right for left, right in zip(
                    synthetic.genomes, source.genomes,
                )
            )
            or any(
                synthetic.gene_specs.get(fingerprint)
                is source.gene_specs.get(fingerprint)
                for fingerprint in source.gene_specs
            )
            or np.shares_memory(
                synthetic.replication_template, source.genomes[0],
            )
            or np.shares_memory(
                synthetic.replication_template, synthetic.genomes[0],
            )
            or np.shares_memory(
                synthetic.replication_template, geometry['template'],
            )):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic active-start cell aliases live biology'
        )
    before = copy.deepcopy(source.state_dict())
    after = copy.deepcopy(synthetic.state_dict())
    if set(before) != set(after):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic cell schema differs from source'
        )
    allowed = {
        'replication_template', 'replication_template_lesion',
        'replication_copy', 'replication_fractional',
    }
    for name in before:
        if name in allowed:
            continue
        if pickle.dumps(
                before[name], protocol=pickle.HIGHEST_PROTOCOL,
        ) != pickle.dumps(
                after[name], protocol=pickle.HIGHEST_PROTOCOL,
        ):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 synthetic cell changed out-of-scope field: ' + name
            )
    template = np.asarray(geometry['template'], dtype=np.uint8)
    if (not np.array_equal(synthetic.replication_template, template)
            or synthetic.replication_template is template
            or synthetic.replication_copy != []
            or not a48c._float64_bits_equal(
                np.asarray([synthetic.replication_template_lesion], np.float64),
                np.asarray([geometry['template_lesion']], np.float64),
            )
            or float(synthetic.replication_fractional) != 0.0
            or tuple(synthetic.mutation_events.items())
            != tuple(source.mutation_events.items())
            or not a4._gene_specs_exact(
                synthetic.gene_specs, source.gene_specs,
            )):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic cell differs from frozen template start'
        )
    source_ragged = source_binding.ragged
    ragged = synthetic_binding.ragged
    tlen = int(geometry['template_length'])
    if (int(ragged.sequence_count) != int(geometry['synthetic_sequences'])
            or int(ragged.symbol_count) != int(geometry['synthetic_symbols'])
            or int(ragged.lesion_count)
            != int(source_ragged.lesion_count)
            or int(ragged.genome_counts[0]) != 1
            or not bool(ragged.replication_active[0])
            or int(ragged.cell_sequence_offsets[0]) != 0
            or int(ragged.cell_sequence_offsets[1]) != 3
            or not np.array_equal(
                ragged.sequence_offsets[:4],
                np.asarray([0, tlen, 2 * tlen, 2 * tlen], np.int64),
            )
            or not np.array_equal(
                ragged.symbols[:tlen], geometry['template'],
            )
            or not np.array_equal(
                ragged.symbols[tlen:2 * tlen], geometry['template'],
            )
            or not a48c._float64_bits_equal(
                ragged.genome_lesions, source_ragged.genome_lesions,
            )
            or not np.array_equal(
                ragged.lesion_offsets, source_ragged.lesion_offsets,
            )
            or not a48c._float64_bits_equal(
                ragged.replication_template_lesions[:1],
                np.asarray([geometry['template_lesion']], np.float64),
            )
            or float(ragged.replication_fractional[0]) != 0.0):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic ragged topology differs from live source'
        )
    source_state = source_binding.state
    state = synthetic_binding.state
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        left = getattr(source_state, name)
        right = getattr(state, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            if name == 'source_provenance':
                if right != synthetic_binding._ragged_provenance:
                    raise A4ReplicationStartCompletionCommitError(
                        'A4.8c6 synthetic provenance is inconsistent'
                    )
            elif left != right:
                raise A4ReplicationStartCompletionCommitError(
                    'A4.8c6 synthetic metadata changed: ' + name
                )
            continue
        expected = np.asarray(left)
        if name == 'replication_active':
            expected = expected.astype(bool, copy=True)
            expected[0] = True
        actual = np.asarray(right)
        same = (
            a48c._float64_bits_equal(expected, actual)
            if expected.dtype == np.dtype(np.float64)
            else np.array_equal(expected, actual)
        )
        if (expected.dtype != actual.dtype
                or expected.shape != actual.shape or not same):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 synthetic state changed: ' + name
            )
    if (a4._gene_cache_provenance(synthetic_binding.cache)
            != source_binding._cache_provenance):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 synthetic start changed derived gene cache'
        )
    return synthetic_binding


def _start_completion_plan_scope(
        source_binding, synthetic_binding, plan, dt, config, evidence=None):
    """Prove a public completion-only plan over the synthetic source."""
    a4._require_translation_binding(source_binding)
    a4._require_translation_binding(synthetic_binding)
    a44.validate_a4_paid_elongation_plan(plan)
    dt = a48c._strict_nonnegative_dt(dt)
    if a48c._strict_mutation_flag(config):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 completion scope requires mutation disabled'
        )
    a48c5._mutation_free_start_config_sha256(config)
    geometry = _inactive_start_completion_geometry(source_binding)
    try:
        a48c2._completion_plan_scope(synthetic_binding, plan)
    except Exception as exc:
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 public completion plan crossed its synthetic scope'
        ) from exc
    tlen = int(geometry['template_length'])
    completed_length = int(plan.completed_lengths[0])
    completed = np.asarray(
        plan.completed_symbols[0, :completed_length], dtype=np.uint8,
    )
    suffix = np.asarray(
        plan.append_symbols[0, :int(plan.append_count[0])], dtype=np.uint8,
    )
    if (int(plan.append_count[0]) != tlen
            or completed_length != tlen
            or not np.array_equal(suffix, geometry['template'])
            or not np.array_equal(completed, geometry['template'])
            or int(plan.topology_sequence_deltas[0]) != -1
            or int(plan.topology_symbol_deltas[0]) != 0
            or int(plan.replication_cycle_deltas[0]) != 1
            or int(plan.substitution_events[0]) != 0
            or bool(plan.template_start_events[0])
            or not bool(plan.completion_events[0])):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 plan is not exact start-to-completion work'
        )
    before = np.asarray(source_binding.state.pools[0], np.float64)
    after = np.asarray(plan.pools_after[0], np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _REPLICATION_PAID_POOLS
    ]
    if not a48c._float64_bits_equal(before[unchanged], after[unchanged]):
        raise A4ReplicationStartCompletionCommitError(
            'A4.8c6 plan changed an out-of-scope material pool'
        )
    if evidence is not None:
        _validate_single_template_completion_selection_evidence(
            evidence, source_binding, dt, config,
        )
    return geometry


def _host_completion_artifacts(evidence, host_replay, plan):
    a44.validate_a4_paid_elongation_plan(host_replay)
    a44.validate_a4_paid_elongation_plan(plan)
    return {
        'evidence_id': id(evidence),
        'evidence_frozen_sha256': _completion_selection_frozen_sha256(
            evidence,
        ),
        'host_replay': a48c5._host_batch_artifact(
            'host_replay', host_replay, a44._PLAN_ARRAY_FIELDS,
        ),
        'plan': a48c5._host_batch_artifact(
            'plan', plan, a44._PLAN_ARRAY_FIELDS,
        ),
    }


@dataclass
class _A4ReplicationStartCompletionCommitCandidate:
    """Private one-shot c6 candidate; never durable authority."""

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
    evidence: object
    evidence_state: object
    synthetic_cell: object
    synthetic_cell_state: bytes
    synthetic_objects: object
    synthetic_binding_identity: object
    synthetic_binding: object
    synthetic_artifacts: object
    host_replay: object
    host_replay_state: object
    resident_binding: object
    resident_plan: object
    resident_artifacts: object
    plan: object
    plan_state: object
    host_plan_artifacts: object
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


class A4ReplicationStartCompletionEventScheduler(
        a48c5.A4ReplicationMutationFreeStartEventScheduler):
    """A4.8c5 scheduler plus mutation-free start/completion authority."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_start_completion_commit_active = False
        super(A4ReplicationStartCompletionEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def replication_start_completion_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_start_completion_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c6 start/completion commit'
            )
        state = super(
            A4ReplicationStartCompletionEventScheduler, self,
        ).state_dict()
        state['a4_replication_start_completion'] = (
            self.replication_start_completion_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        own = _canonical_start_completion_integration_state(
            state.get('a4_replication_start_completion'),
        )
        inherited = (
            a48c5.A4ReplicationMutationFreeStartEventScheduler.from_state(
                state,
            )
        )
        if (own['config'] != a48a._config_state(inherited.a4_config)
                or own['device'] != inherited.a4_device):
            raise A4ReplicationStartCompletionCommitError(
                'saved A4.8c5/A4.8c6 config or device differs'
            )
        scheduler = cls(
            a4_config=own['config'], device=own['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler

    def _synthetic_active_start(self, world, cell, source_binding):
        geometry = _inactive_start_completion_geometry(source_binding)
        if (getattr(cell, 'replication_template', None) is not None
                or not isinstance(cell.replication_copy, list)
                or cell.replication_copy != []):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 live source is not an inactive empty-copy cell'
            )
        synthetic = _build_synthetic_active_start_cell(cell, geometry)
        _, synthetic_binding = self._pack_host_binding(world, synthetic)
        _require_synthetic_active_start_binding(
            cell, synthetic, source_binding, synthetic_binding, geometry,
        )
        return geometry, synthetic, synthetic_binding

    def _classify_start_completion_branch(self, world, cell, dt, config):
        """Classify by public resident completion scope, never exception."""
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationStartCompletionCommitError(
                'replication config must be the active world config object'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 replication source cell must be alive'
            )
        if a48c._strict_mutation_flag(config):
            raise A4ReplicationStartCompletionCommitError(
                'mutation-enabled source must delegate before c6 classify'
            )
        a48c5._mutation_free_start_config_sha256(config)
        source_state = a48c._cell_state_snapshot(cell)
        source_objects = a48c5._inactive_source_object_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)
        energy_before = float(world.dissipated_energy)
        _, source_binding = self._pack_host_binding(world, cell)
        _, synthetic, synthetic_binding = self._synthetic_active_start(
            world, cell, source_binding,
        )
        resident_binding = a4.bind_a4_translation(
            synthetic_binding.ragged.to_torch(device=self.a4_device),
            synthetic_binding.state.to_torch(device=self.a4_device),
        )
        probe = a44.paid_replication_completion_plan(
            resident_binding, dt, config,
        )
        if not a4._is_tensor(probe.scope_error_code):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 dispatch probe escaped resident authority'
            )
        codes = probe.scope_error_code.detach().cpu().numpy().copy()
        completions = probe.completion_events.detach().cpu().numpy().copy()
        valid = probe.scope_valid.detach().cpu().numpy().copy()
        if (codes.shape != (int(source_binding.state.cell_capacity),)
                or completions.shape != codes.shape
                or valid.shape != codes.shape):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 dispatch probe schema differs'
            )
        if (a48c._cell_state_snapshot(cell) != source_state
                or not a48c5._inactive_source_objects_match(
                    cell, source_objects,
                )
                or world.rng is not live_rng
                or world.rng.bit_generator.state != rng_before
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([energy_before], np.float64),
                )):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 dispatch probe changed live authority'
            )
        # Keep the derivation live through the checks; it is never authority.
        if synthetic.replication_template is None:
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 dispatch lost its synthetic source'
            )
        code = int(codes[0])
        completed = bool(completions[0])
        supported = bool(valid[0])
        if code == int(a44.SCOPE_OK) and completed and supported:
            return _DISPATCH_START_COMPLETION
        if (code == int(a44.SCOPE_NONCOMPLETION)
                and not completed and not supported):
            return _DISPATCH_START_NONCOMPLETION
        raise A4ReplicationStartCompletionCommitError(
            'replication branch is outside A4.8c6/c5 scope: %d' % code
        )

    def _build_start_completion_candidate(
            self, synthetic_cell, plan):
        return (
            a48c2.A4ReplicationCompletionEventScheduler.
            _build_completion_candidate(self, synthetic_cell, plan)
        )

    def _require_final_start_completion_candidate(
            self, source, synthetic, candidate, evidence, plan,
            source_binding, synthetic_binding, fresh_binding, dt, config):
        """Validate the committed net state directly against live inactive."""
        geometry = _start_completion_plan_scope(
            source_binding, synthetic_binding, plan, dt, config, evidence,
        )
        a4._require_translation_binding(fresh_binding)
        if (candidate is source or candidate is synthetic
                or candidate.pools is source.pools
                or candidate.pools is synthetic.pools
                or candidate.genomes is source.genomes
                or candidate.genomes is synthetic.genomes
                or candidate.genome_lesions is source.genome_lesions
                or candidate.replication_copy is source.replication_copy
                or candidate.replication_copy is synthetic.replication_copy
                or candidate.mutation_events is source.mutation_events
                or candidate.proteins is source.proteins
                or candidate.damaged_proteins is source.damaged_proteins
                or candidate.gene_specs is source.gene_specs
                or any(
                    current is old for current in candidate.genomes
                    for old in source.genomes
                )
                or any(
                    candidate.gene_specs.get(fingerprint)
                    is source.gene_specs.get(fingerprint)
                    for fingerprint in source.gene_specs
                )
                or any(
                    np.shares_memory(candidate.genomes[-1], old)
                    for old in source.genomes
                )
                or np.shares_memory(
                    candidate.genomes[-1], geometry['template'],
                )):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 final candidate aliases source or synthetic biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 final candidate schema differs from source'
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
            if pickle.dumps(
                    source_state[name], protocol=pickle.HIGHEST_PROTOCOL,
            ) != pickle.dumps(
                    candidate_state[name], protocol=pickle.HIGHEST_PROTOCOL,
            ):
                raise A4ReplicationStartCompletionCommitError(
                    'A4.8c6 final candidate changed out-of-scope field: '
                    + name
                )
        tlen = int(geometry['template_length'])
        completed = np.asarray(
            plan.completed_symbols[0, :int(plan.completed_lengths[0])],
            dtype=np.uint8,
        )
        expected_novel = copy.deepcopy(source.novel_path_first_age)
        if expected_novel is None and a4.g2.sequence_has_novel_path(completed):
            expected_novel = float(source.age)
        if (len(candidate.genomes) != len(source.genomes) + 1
                or len(candidate.genomes) != 2
                or any(
                    not np.array_equal(left, right)
                    for left, right in zip(
                        candidate.genomes[:-1], source.genomes,
                    )
                )
                or not np.array_equal(candidate.genomes[-1], completed)
                or not np.array_equal(completed, geometry['template'])
                or list(candidate.genome_lesions[:-1])
                != list(source.genome_lesions)
                or not a48c._float64_bits_equal(
                    np.asarray([candidate.genome_lesions[-1]], np.float64),
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or candidate.replication_template is not None
                or float(candidate.replication_template_lesion) != 0.0
                or candidate.replication_copy != []
                or float(candidate.replication_fractional) != 0.0
                or int(candidate.replication_cycles)
                != int(source.replication_cycles) + 1
                or candidate.novel_path_first_age != expected_novel
                or tuple(candidate.mutation_events.items())
                != tuple(source.mutation_events.items())
                or not a48c._float64_bits_equal(
                    candidate.pools,
                    np.asarray(plan.pools_after[0], np.float64),
                )
                or int(candidate.last_replication_symbols) != tlen
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
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 final cell differs from resident completion plan'
            )
        a48b._require_cell_gene_specs(candidate, fresh_binding)
        ragged = fresh_binding.ragged
        source_ragged = source_binding.ragged
        lesion_count = int(source_ragged.lesion_count)
        if (int(ragged.sequence_count) != int(geometry['final_sequences'])
                or int(ragged.symbol_count) != int(geometry['final_symbols'])
                or int(ragged.lesion_count) != lesion_count + 1
                or int(ragged.genome_counts[0]) != 2
                or bool(ragged.replication_active[0])
                or int(ragged.cell_sequence_offsets[0]) != 0
                or int(ragged.cell_sequence_offsets[1]) != 2
                or not np.array_equal(
                    ragged.sequence_offsets[:3],
                    np.asarray([0, tlen, 2 * tlen], np.int64),
                )
                or not np.array_equal(
                    ragged.symbols[:tlen], geometry['template'],
                )
                or not np.array_equal(
                    ragged.symbols[tlen:2 * tlen], completed,
                )
                or not a48c._float64_bits_equal(
                    ragged.genome_lesions[:lesion_count],
                    source_ragged.genome_lesions[:lesion_count],
                )
                or not a48c._float64_bits_equal(
                    ragged.genome_lesions[lesion_count:lesion_count + 1],
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or float(ragged.replication_template_lesions[0]) != 0.0
                or float(ragged.replication_fractional[0]) != 0.0):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 final ragged topology differs from direct net state'
            )
        source_translation = source_binding.state
        final_translation = fresh_binding.state
        expected_lesion_mean = (
            float(np.mean(np.asarray(candidate.genome_lesions, np.float64)))
            if candidate.genome_lesions else 1.0
        )
        for item in fields(a4.A4TranslationStateBatch):
            name = item.name
            before = getattr(source_translation, name)
            after = getattr(final_translation, name)
            if name not in a4._TRANSLATION_ARRAY_FIELDS:
                if name == 'source_provenance':
                    if after != fresh_binding._ragged_provenance:
                        raise A4ReplicationStartCompletionCommitError(
                            'A4.8c6 final provenance is inconsistent'
                        )
                elif before != after:
                    raise A4ReplicationStartCompletionCommitError(
                        'A4.8c6 final metadata changed: ' + name
                    )
                continue
            if name == 'pools':
                expected = np.asarray(plan.pools_after, np.float64)
            elif name == 'cumulative_proofreading_atp':
                expected = np.asarray(
                    plan.cumulative_proofreading_atp_after, np.float64,
                )
            elif name == 'genome_count':
                expected = np.asarray(before, np.int64).copy()
                expected[0] += 1
            elif name == 'genome_material_symbols':
                expected = np.asarray(before, np.int64).copy()
                expected[0] += tlen
            elif name == 'genome_lesion_mean':
                expected = np.asarray(before, np.float64).copy()
                expected[0] = expected_lesion_mean
            else:
                expected = np.asarray(before)
            actual = np.asarray(after)
            same = (
                a48c._float64_bits_equal(expected, actual)
                if expected.dtype == np.dtype(np.float64)
                else np.array_equal(expected, actual)
            )
            if (expected.dtype != actual.dtype
                    or expected.shape != actual.shape or not same):
                raise A4ReplicationStartCompletionCommitError(
                    'A4.8c6 final translation state differs: ' + name
                )
        return geometry

    def _prepare_replication_start_completion_candidate(
            self, world, cell, dt, config):
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationStartCompletionCommitError(
                'replication config must be the active world config object'
            )
        if a48c._strict_mutation_flag(config):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 requires mutation disabled'
            )
        a48c5._mutation_free_start_config_sha256(config)
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 replication source cell must be alive'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        source_objects = a48c5._inactive_source_object_snapshot(cell)
        if cell.replication_copy != []:
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 source copy must be empty'
            )
        _, source_binding = self._pack_host_binding(world, cell)
        geometry = _inactive_start_completion_geometry(source_binding)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        source_artifacts = a48c._host_binding_artifact_snapshot(
            source_binding,
        )
        source_cell_state = a48c._cell_state_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)
        evidence = _prepare_single_template_completion_selection_evidence(
            source_binding, dt, config, rng_before,
        )

        _, synthetic_cell, synthetic_binding = (
            self._synthetic_active_start(world, cell, source_binding)
        )
        synthetic_cell_state = a48c._cell_state_snapshot(synthetic_cell)
        synthetic_objects = _synthetic_active_start_objects(synthetic_cell)
        synthetic_identity = a48a._binding_identity(synthetic_binding)
        synthetic_artifacts = a48c._host_binding_artifact_snapshot(
            synthetic_binding,
        )
        host_replay = a44.paid_replication_completion_plan(
            synthetic_binding, dt, config,
        )
        _start_completion_plan_scope(
            source_binding, synthetic_binding, host_replay,
            dt, config, evidence,
        )

        resident_binding = a4.bind_a4_translation(
            synthetic_binding.ragged.to_torch(device=self.a4_device),
            synthetic_binding.state.to_torch(device=self.a4_device),
        )
        resident_plan = a44.paid_replication_completion_plan(
            resident_binding, dt, config,
        )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 resident plan escaped Torch authority'
            )
        resident_artifacts = a48c._resident_artifact_snapshot(
            resident_binding, None, resident_plan, self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source) != synthetic_identity
                or a4._gene_cache_provenance(resident_cache)
                != synthetic_binding._cache_provenance):
            raise A4ReplicationStartCompletionCommitError(
                'resident A4.8c6 synthetic source differs after readback'
            )
        resident_replay = a44.paid_replication_completion_plan(
            resident_source, dt, config,
        )
        if not a48c._plan_states_bit_exact(resident_replay, host_replay):
            raise A4ReplicationStartCompletionCommitError(
                'resident A4.8c6 source replay differs from NumPy oracle'
            )
        plan = resident_plan.to_numpy()
        _start_completion_plan_scope(
            source_binding, synthetic_binding, plan, dt, config, evidence,
        )
        a48c._plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_start_completion_candidate(
            synthetic_cell, plan,
        )
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._require_final_start_completion_candidate(
            cell, synthetic_cell, candidate_cell, evidence, plan,
            source_binding, synthetic_binding, fresh_binding, dt, config,
        )
        candidate_cell_state = a48c._cell_state_snapshot(candidate_cell)
        candidate_objects = a48c2._completion_candidate_object_snapshot(
            candidate_cell,
        )
        fresh_artifacts = a48c._host_binding_artifact_snapshot(fresh_binding)
        host_plan_artifacts = _host_completion_artifacts(
            evidence, host_replay, plan,
        )
        return _A4ReplicationStartCompletionCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self), world_object_id=id(world),
            cell_object_id=id(cell), cell_id=int(cell.cell_id),
            generation=generation, dt_hex=dt.hex(),
            branch=_BRANCH_START_COMPLETION,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding,
            source_artifacts=source_artifacts,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects,
            evidence=evidence, evidence_state=evidence.state_dict(),
            synthetic_cell=synthetic_cell,
            synthetic_cell_state=synthetic_cell_state,
            synthetic_objects=synthetic_objects,
            synthetic_binding_identity=synthetic_identity,
            synthetic_binding=synthetic_binding,
            synthetic_artifacts=synthetic_artifacts,
            host_replay=host_replay,
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan, plan_state=plan.state_dict(),
            host_plan_artifacts=host_plan_artifacts,
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding,
            fresh_artifacts=fresh_artifacts,
            live_rng=live_rng,
            rng_before_state=copy.deepcopy(rng_before),
            rng_after_state=copy.deepcopy(evidence.rng_after_state),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.replication_start_completion_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_replication_start_completion_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(
                candidate, _A4ReplicationStartCompletionCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 candidate is untrusted or already consumed'
            )
        dt = a48c._strict_nonnegative_dt(dt)
        live_rng, rng_before = a48c._live_pcg64(world)
        if (candidate.branch != _BRANCH_START_COMPLETION
                or id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.replication_start_completion_integration_state()
                != candidate.integration_state
                or a48a._world_config_snapshot(world)
                != candidate.world_config_snapshot
                or id(cell) != candidate.cell_object_id
                or int(cell.cell_id) != candidate.cell_id
                or a48a._strict_counter(
                    getattr(cell, 'generation', None), 'cell generation',
                ) != candidate.generation
                or not bool(getattr(cell, 'alive', False))
                or a48c._strict_mutation_flag(config)
                or dt.hex() != candidate.dt_hex
                or live_rng is not candidate.live_rng
                or rng_before != candidate.rng_before_state
                or candidate.evidence.rng_before_state
                != candidate.rng_before_state
                or candidate.evidence.rng_after_state
                != candidate.rng_after_state
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([candidate.dissipated_energy], np.float64),
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.source_cell_state
                or not a48c5._inactive_source_objects_match(
                    cell, candidate.source_objects,
                )):
            raise A4ReplicationStartCompletionCommitError(
                'live A4.8c6 cell/config/dt/RNG changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity):
            raise A4ReplicationStartCompletionCommitError(
                'live A4.8c6 biological source changed before claim'
            )
        a4._require_translation_binding(candidate.source_binding)
        if (a48a._binding_identity(candidate.source_binding)
                != candidate.source_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.source_binding,
                ) != candidate.source_artifacts):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 source binding changed before claim'
            )
        _validate_single_template_completion_selection_evidence(
            candidate.evidence, candidate.source_binding, dt, config,
        )
        if candidate.evidence.state_dict() != candidate.evidence_state:
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 selection evidence changed'
            )

        if (a48c._cell_state_snapshot(candidate.synthetic_cell)
                != candidate.synthetic_cell_state
                or not a48c._source_objects_match(
                    candidate.synthetic_cell, candidate.synthetic_objects,
                )):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 synthetic cell changed before claim'
            )
        _require_synthetic_active_start_binding(
            cell, candidate.synthetic_cell, candidate.source_binding,
            candidate.synthetic_binding,
            _inactive_start_completion_geometry(candidate.source_binding),
        )
        if (a48a._binding_identity(candidate.synthetic_binding)
                != candidate.synthetic_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.synthetic_binding,
                ) != candidate.synthetic_artifacts):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 synthetic binding changed before claim'
            )
        _, synthetic_now, synthetic_binding_now = (
            self._synthetic_active_start(world, cell, current_binding)
        )
        if (a48c._cell_state_snapshot(synthetic_now)
                != candidate.synthetic_cell_state
                or a48a._binding_identity(synthetic_binding_now)
                != candidate.synthetic_binding_identity):
            raise A4ReplicationStartCompletionCommitError(
                'fresh A4.8c6 synthetic derivation changed before claim'
            )
        fresh_evidence = (
            _prepare_single_template_completion_selection_evidence(
                current_binding, dt, config, rng_before,
            )
        )
        if (fresh_evidence.state_dict() != candidate.evidence_state
                or fresh_evidence.rng_after_state
                != candidate.rng_after_state):
            raise A4ReplicationStartCompletionCommitError(
                'fresh A4.8c6 selection evidence changed before claim'
            )

        a44.validate_a4_paid_elongation_plan(candidate.host_replay)
        a44.validate_a4_paid_elongation_plan(candidate.plan)
        if (not a48a._state_arrays_bit_exact(
                candidate.host_replay.state_dict(),
                candidate.host_replay_state,
            )
                or not a48a._state_arrays_bit_exact(
                    candidate.plan.state_dict(), candidate.plan_state,
                )
                or _host_completion_artifacts(
                    candidate.evidence, candidate.host_replay,
                    candidate.plan,
                ) != candidate.host_plan_artifacts):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 oracle/readback changed before claim'
            )
        current_replay = a44.paid_replication_completion_plan(
            synthetic_binding_now, dt, config,
        )
        _start_completion_plan_scope(
            current_binding, synthetic_binding_now, current_replay,
            dt, config, fresh_evidence,
        )
        if not a48c._plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationStartCompletionCommitError(
                'fresh A4.8c6 NumPy completion plan changed before claim'
            )

        a48c._require_resident_artifact_snapshot(
            candidate.resident_binding, None, candidate.resident_plan,
            self.a4_device, candidate.resident_artifacts,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.synthetic_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.synthetic_binding._cache_provenance):
            raise A4ReplicationStartCompletionCommitError(
                'resident A4.8c6 source changed before claim'
            )
        resident_replay = a44.paid_replication_completion_plan(
            resident_source, dt, config,
        )
        if not a48c._plan_states_bit_exact(
                resident_replay, candidate.host_replay):
            raise A4ReplicationStartCompletionCommitError(
                'resident A4.8c6 replay differs from NumPy oracle'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        if not a48c._plan_states_bit_exact(resident_plan, candidate.plan):
            raise A4ReplicationStartCompletionCommitError(
                'resident A4.8c6 plan changed before claim'
            )
        _start_completion_plan_scope(
            current_binding, synthetic_binding_now, resident_plan,
            dt, config, fresh_evidence,
        )
        a48c._plan_semantic_match(resident_plan, current_replay)

        if (a48c._cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not a48c2._completion_candidate_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 final candidate changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._require_final_start_completion_candidate(
            cell, candidate.synthetic_cell, candidate.candidate_cell,
            candidate.evidence, candidate.plan, candidate.source_binding,
            candidate.synthetic_binding, fresh_now, dt, config,
        )
        if (a48c._host_binding_artifact_snapshot(candidate.fresh_binding)
                != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCompletionCommitError(
                'retained A4.8c6 final binding changed before claim'
            )
        return candidate

    def _replication_start_completion_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after prepare and before final trust."""
        return candidate

    @staticmethod
    def _start_completion_publish_snapshot(world, cell):
        if cell.replication_template is not None:
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 publish snapshot requires inactive source'
            )
        return {
            'pools_object': cell.pools,
            'pools': np.asarray(cell.pools, np.float64).copy(),
            'genomes_object': cell.genomes,
            'genome_objects': tuple(cell.genomes),
            'genomes': tuple(
                np.asarray(genome, np.uint8).copy()
                for genome in cell.genomes
            ),
            'lesions_object': cell.genome_lesions,
            'genome_lesions': tuple(cell.genome_lesions),
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
            'replication_template_lesion': float(
                cell.replication_template_lesion
            ),
            'replication_fractional': float(cell.replication_fractional),
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

    def _publish_replication_start_completion_candidate(
            self, world, cell, candidate):
        """Publish final resident state without touching mutation ledger."""
        prepared = candidate.candidate_cell
        prepared_pools = np.asarray(prepared.pools, np.float64)
        for pool_index in sorted(_REPLICATION_PAID_POOLS):
            cell.pools[pool_index] = prepared_pools[pool_index]
        cell.genomes.append(np.asarray(
            prepared.genomes[-1], np.uint8,
        ).copy())
        cell.genome_lesions.append(float(prepared.genome_lesions[-1]))
        cell.replication_template = None
        cell.replication_template_lesion = 0.0
        # Formal066 assigns a fresh list at start and another fresh list at
        # completion; do not retain the inactive entry list on success.
        cell.replication_copy = []
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
        world.rng.bit_generator.state = copy.deepcopy(
            candidate.rng_after_state,
        )

    def _published_start_completion_matches(
            self, world, cell, candidate, snapshot):
        source_objects = candidate.source_objects
        prepared = candidate.candidate_cell
        old_count = len(source_objects['genome_objects'])
        new_genome = cell.genomes[-1]
        if (cell.pools is not source_objects['pools_object']
                or cell.genomes is not source_objects['genomes_object']
                or len(cell.genomes) != old_count + 1
                or any(
                    current is not expected for current, expected in zip(
                        cell.genomes[:old_count],
                        source_objects['genome_objects'],
                    )
                )
                or new_genome is prepared.genomes[-1]
                or any(new_genome is old for old in cell.genomes[:-1])
                or any(
                    np.shares_memory(new_genome, old)
                    for old in cell.genomes[:-1]
                )
                or not np.array_equal(new_genome, prepared.genomes[-1])
                or cell.genome_lesions
                is not source_objects['lesions_object']
                or cell.replication_template is not None
                or cell.replication_copy
                is source_objects['copy_object']
                or cell.replication_copy
                is prepared.replication_copy
                or cell.replication_copy != []
                or not a48c._mapping_identity_matches(
                    cell.mutation_events,
                    source_objects['mutation_events_object'],
                    source_objects['mutation_events_items'],
                )
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
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([snapshot['dissipated_energy']], np.float64),
                )):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCompletionCommitError(
                'published A4.8c6 binding differs from final candidate'
            )

    def _rollback_replication_start_completion_publish(
            self, world, cell, snapshot):
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

        cell.replication_template = None
        copied = snapshot['copy_object']
        copied[:] = list(snapshot['replication_copy'])
        cell.replication_copy = copied

        cell.mutation_events = self._restore_mapping(
            snapshot['mutation_events_object'],
            snapshot['mutation_events_items'],
        )
        cell.proteins = self._restore_mapping(
            snapshot['proteins_object'], snapshot['proteins_items'],
        )
        cell.damaged_proteins = self._restore_mapping(
            snapshot['damaged_proteins_object'],
            snapshot['damaged_proteins_items'],
        )
        gene_specs = snapshot['gene_specs_object']
        for _, spec_object, saved_spec in snapshot['gene_specs_items']:
            spec_object.clear()
            spec_object.update(copy.deepcopy(saved_spec))
        gene_specs.clear()
        for fingerprint, spec_object, _ in snapshot['gene_specs_items']:
            gene_specs[fingerprint] = spec_object
        cell.gene_specs = gene_specs

        cell.replication_template_lesion = snapshot[
            'replication_template_lesion'
        ]
        cell.replication_fractional = snapshot['replication_fractional']
        cell.replication_cycles = snapshot['replication_cycles']
        cell.novel_path_first_age = snapshot['novel_path_first_age']
        cell.age = snapshot['age']
        cell.alive = snapshot['alive']
        cell.last_replication_symbols = snapshot[
            'last_replication_symbols'
        ]
        cell.last_effective_error_rate = snapshot[
            'last_effective_error_rate'
        ]
        cell.cumulative_proofreading_atp = snapshot[
            'cumulative_proofreading_atp'
        ]
        world.dissipated_energy = snapshot['dissipated_energy']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])

    def _commit_replication_start_completion_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_replication_start_completion_candidate(
            world, cell, dt, config, candidate,
        )
        snapshot = self._start_completion_publish_snapshot(world, cell)
        candidate.consumed = True
        plan = candidate.plan
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c6-resident-mutation-free-start-completion-plan-'
                'selection-only-pcg64-atomic-commit'
            ),
            'branch': candidate.branch,
            'mutation_enabled': False,
            'device': self.a4_device,
            'genome_count': int(
                candidate.source_binding.state.genome_count[0]
            ),
            'source_provenance': str(
                candidate.source_binding.state.source_provenance
            ),
            'synthetic_provenance': str(
                candidate.synthetic_binding.state.source_provenance
            ),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
        })
        try:
            self._publish_replication_start_completion_candidate(
                world, cell, candidate,
            )
            self._published_start_completion_matches(
                world, cell, candidate, snapshot,
            )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': int(plan.last_replication_symbols[0]),
                'work_performed': True,
                'template_start_events': 1,
                'completion_events': 1,
                'requested_symbols': int(plan.requested_symbols[0]),
                'substitution_events': 0,
                'rng_call_count': 1,
            })
        except BaseException:
            self._rollback_replication_start_completion_publish(
                world, cell, snapshot,
            )
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Delegate prior branches or commit one c6 start/completion."""
        _, record = self._context(world, cell, dt)
        if 'replication_cpu' in record['claimed']:
            raise a3s.A3DuplicateEventError(
                'duplicate A4.8c6 replication_cpu event'
            )
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationStartCompletionCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_start_completion_commit_active
                or self._a4_replication_mutation_free_start_commit_active
                or self._a4_replication_start_commit_active
                or self._a4_replication_completion_mutation_commit_active
                or self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c6 replication commit is forbidden'
            )
        if (getattr(cell, 'replication_template', None) is not None
                or a48c._strict_mutation_flag(config)):
            return super(
                A4ReplicationStartCompletionEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        branch = self._classify_start_completion_branch(
            world, cell, dt, config,
        )
        if branch == _DISPATCH_START_NONCOMPLETION:
            return super(
                A4ReplicationStartCompletionEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        if branch != _DISPATCH_START_COMPLETION:
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 dispatch produced an unknown branch'
            )
        self._a4_replication_start_completion_commit_active = True
        try:
            candidate = self._prepare_replication_start_completion_candidate(
                world, cell, dt, config,
            )
            ready = self._replication_start_completion_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4ReplicationStartCompletionCommitError(
                    'A4.8c6 ready hook must retain its private candidate'
                )
            return self._commit_replication_start_completion_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_start_completion_commit_active = False


class Hybrid066WorldA4ReplicationStartCompletion(
        a48c5.Hybrid066WorldA4ReplicationMutationFreeStart):
    """A4.8c5 world retaining bounded A4.8c6 authority on restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationStartCompletionEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler, A4ReplicationStartCompletionEventScheduler):
            raise TypeError(
                'scheduler must be '
                'A4ReplicationStartCompletionEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationStartCompletion, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationStartCompletionCommitError(
                'new A4.8c6 world requires explicit A4 config and device'
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
            Hybrid066WorldA4ReplicationStartCompletion, self,
        ).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_start_completion_device': (
                self.scheduler.a4_device
            ),
            'a4_replication_start_completion_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(
            Hybrid066WorldA4ReplicationStartCompletion, self,
        ).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_start_completion': (
                self.scheduler.
                replication_start_completion_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8c6 save version/build differs'
            )
        own = _canonical_start_completion_integration_state(
            state.get('a4_replication_start_completion'),
        )
        start_free = a48c5._canonical_mutation_free_start_integration_state(
            state.get('a4_replication_mutation_free_start'),
        )
        start = a48c4._canonical_start_integration_state(
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
                'A4.8c6 save is missing aggregate composition sidecars'
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
        scheduler = A4ReplicationStartCompletionEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if (scheduler.replication_start_completion_integration_state()
                != own
                or scheduler.
                replication_mutation_free_start_integration_state()
                != start_free
                or scheduler.replication_start_integration_state() != start
                or scheduler.completion_mutation_integration_state()
                != mutation
                or scheduler.completion_integration_state() != completion
                or scheduler.replication_integration_state() != replication
                or scheduler.translation_integration_state() != translation
                or scheduler.integration_state() != hydrolysis):
            raise A4ReplicationStartCompletionCommitError(
                'A4.8a/b/c1/c2/c3/c4/c5/c6 settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c5.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c6-mutation-free-inactive-one-genome-template-start-'
        'same-call-completion-synthetic-active-public-plan-selection-only-'
        'pcg64-resident-atomic-commit-a4.8c5-c4-c3-c2-c1-inherited-'
        'other-inactive-early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c6-mutation-free-selection-only-ledger-state-unchanged-plus-'
        'a4.8c5-c4-branches-mutation-enabled-start-completion-not-integrated'
    ),
    'event_scheduler': (
        'a4.8c6-start-completion-plus-a4.8c5-mutation-free-start-plus-'
        'a4.8c4-mutation-start-plus-a4.8c3-completion-mutation-plus-'
        'a4.8c2-completion-plus-a4.8c1-noncompletion-plus-a4.8b-'
        'translation-plus-a4.8a-hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationStartCompletionCommitError',
    'A4ReplicationStartCompletionEventScheduler',
    'Hybrid066WorldA4ReplicationStartCompletion',
)
