# coding: utf-8
"""A4.8c7 mutation-enabled inactive-start same-call completion bridge.

This deliberately bounded bridge owns exactly one live Formal066 branch: an
inactive cell with one non-empty genome starts replication and completes it in
the same call while ``config.mutation`` is true.  One cloned-PCG64 selection
call is chained directly into the public A4.6b1 completion-mutation tape; the
public A4.6b2 resident final-polymer descriptor is commit authority.

The synthetic active-start cell is event-local and never published.  A
mutation-disabled config clone is used only to classify completion versus
noncompletion without consuming RNG.  Earlier A4.8c6/c5/c4/c3/c2/c1 branches
remain delegated and no frozen-CPU fallback is introduced.
"""
from __future__ import division

import copy
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
import SOMA_CELL_0_6_8_gpu_a4_replication_start_completion_integration as a48c6
import SOMA_CELL_0_6_8_gpu_a4_translation_integration as a48b

try:
    import torch
except Exception:  # pragma: no cover - integration fails closed without it
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8c7'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | mutation-enabled inactive-start completion bridge'
SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c7-mutation-enabled-inactive-template-start-'
    'same-call-completion-atomic-commit'
)
SELECTION_EVIDENCE_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c7-single-template-completion-mutation-selection-evidence'
)
RNG_CHAIN_SCHEMA_VERSION = (
    '0.6.8-GPU-A4.8c7-selection-completion-mutation-pcg64-chain'
)
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
REPLICATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_SELECTION_EVIDENCE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_BRANCH_START_COMPLETION_MUTATION = 'inactive-template-start-completion-mutation'
_DISPATCH_START_COMPLETION_MUTATION = 'start-completion-mutation'
_DISPATCH_START_NONCOMPLETION = 'start-noncompletion'
_REPLICATION_PAID_POOLS = {
    int(a4.a3.POOL_NUCLEOTIDE), int(a4.a3.POOL_ATP),
}
_MUTATION_COUNTER_NAMES = (
    'substitution',
) + tuple(a44.STRUCTURAL_EVENT_NAMES)


class A4ReplicationStartCompletionMutationCommitError(
        a48c6.A4ReplicationStartCompletionCommitError):
    """The bounded A4.8c7 source, RNG chain, plan, or commit failed."""


def _canonical_start_completion_mutation_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 save schema differs'
        )
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 saved fixed-capacity config is invalid'
        )
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
        device = a48a._canonical_device(value.get('device'))
    except Exception as exc:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 saved config or device is invalid'
        ) from exc
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _completion_mutation_config_sha256(config):
    """Use the exact public A4.6b1 mutation-config digest authority."""
    _, _, digest = a44._completion_mutation_config(config)
    if not a44._is_lower_hex_digest(digest):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 completion-mutation config digest is noncanonical'
        )
    return digest


@dataclass
class _A4SingleTemplateCompletionMutationSelectionEvidence:
    """Private proof that exactly one Formal066 selection call ran first."""

    _factory_token: object
    schema_version: str
    source_binding_identity: object
    synthetic_binding_identity: object
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
            'synthetic_binding_identity': tuple(
                self.synthetic_binding_identity
            ),
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


def _selection_schedule_sha256(evidence):
    before = a48c5._canonical_selection_pcg64_state(
        evidence.rng_before_state, 'start_completion_mutation_rng_before',
    )
    after = a48c5._canonical_selection_pcg64_state(
        evidence.rng_after_state, 'start_completion_mutation_rng_after',
    )
    return a44._sha256_json({
        'schema': str(evidence.schema_version),
        'source_binding_identity': list(evidence.source_binding_identity),
        'synthetic_binding_identity': list(
            evidence.synthetic_binding_identity
        ),
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


def _selection_frozen_sha256(evidence):
    return a44._sha256_json(evidence.state_dict())


def _prepare_single_template_completion_mutation_selection_evidence(
        source_binding, synthetic_binding, dt, config, rng_state_before):
    """Replay only ``integers(0, 1)`` on a cloned PCG64."""
    a4._require_translation_binding(source_binding)
    a4._require_translation_binding(synthetic_binding)
    dt = a48c._strict_nonnegative_dt(dt)
    a48c6._inactive_start_completion_geometry(source_binding)
    config_sha256 = _completion_mutation_config_sha256(config)
    before = a48c5._canonical_selection_pcg64_state(
        rng_state_before, 'start_completion_mutation_rng_before',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    selected = a48c5._formal066_single_template_selection(generator)
    if selected != 0:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 single-template selection did not return zero'
        )
    after = a48c5._canonical_selection_pcg64_state(
        generator.bit_generator.state,
        'start_completion_mutation_rng_after',
    )
    evidence = _A4SingleTemplateCompletionMutationSelectionEvidence(
        _factory_token=_SELECTION_EVIDENCE_FACTORY_TOKEN,
        schema_version=SELECTION_EVIDENCE_SCHEMA_VERSION,
        source_binding_identity=a48a._binding_identity(source_binding),
        synthetic_binding_identity=a48a._binding_identity(
            synthetic_binding
        ),
        cell_id=int(source_binding.state.cell_ids[0]),
        dt_hex=dt.hex(), config_sha256=config_sha256,
        low=0, high=1, selected_index=selected, call_count=1,
        rng_before_state=before, rng_after_state=after,
        schedule_sha256='0' * 64,
    )
    evidence.schedule_sha256 = _selection_schedule_sha256(evidence)
    evidence._frozen_sha256 = _selection_frozen_sha256(evidence)
    return _validate_single_template_completion_mutation_selection_evidence(
        evidence, source_binding, synthetic_binding, dt, config,
    )


def _validate_single_template_completion_mutation_selection_evidence(
        evidence, source_binding, synthetic_binding, dt, config):
    a4._require_translation_binding(source_binding)
    a4._require_translation_binding(synthetic_binding)
    dt = a48c._strict_nonnegative_dt(dt)
    a48c6._inactive_start_completion_geometry(source_binding)
    if (not isinstance(
                evidence,
                _A4SingleTemplateCompletionMutationSelectionEvidence)
            or evidence._factory_token
            is not _SELECTION_EVIDENCE_FACTORY_TOKEN
            or evidence.schema_version
            != SELECTION_EVIDENCE_SCHEMA_VERSION):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection evidence is not factory-authenticated'
        )
    for name in ('cell_id', 'low', 'high', 'selected_index', 'call_count'):
        value = getattr(evidence, name)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 selection evidence %s is not an integer' % name
            )
    source_identity = a48a._binding_identity(source_binding)
    synthetic_identity = a48a._binding_identity(synthetic_binding)
    identities = (
        evidence.source_binding_identity,
        evidence.synthetic_binding_identity,
    )
    if (any(
            not isinstance(identity, tuple) or len(identity) != 3
            or not all(a44._is_lower_hex_digest(item) for item in identity)
            for identity in identities)
            or tuple(evidence.source_binding_identity) != source_identity
            or tuple(evidence.synthetic_binding_identity)
            != synthetic_identity
            or int(evidence.cell_id) != int(source_binding.state.cell_ids[0])
            or evidence.dt_hex != dt.hex()
            or evidence.config_sha256
            != _completion_mutation_config_sha256(config)
            or not a44._is_lower_hex_digest(evidence.schedule_sha256)
            or (int(evidence.low), int(evidence.high),
                int(evidence.selected_index), int(evidence.call_count))
            != (0, 1, 0, 1)):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection evidence differs from source/config/dt'
        )
    before = a48c5._canonical_selection_pcg64_state(
        evidence.rng_before_state, 'start_completion_mutation_rng_before',
    )
    after = a48c5._canonical_selection_pcg64_state(
        evidence.rng_after_state, 'start_completion_mutation_rng_after',
    )
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(before)
    if a48c5._formal066_single_template_selection(generator) != 0:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection replay returned another index'
        )
    if generator.bit_generator.state != after:
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection after-state differs from replay'
        )
    if (evidence.schedule_sha256 != _selection_schedule_sha256(evidence)
            or getattr(evidence, '_frozen_sha256', None)
            != _selection_frozen_sha256(evidence)):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection evidence changed after creation'
        )
    return evidence


def _completion_mutation_rng_chain_sha256(evidence, tape):
    """Digest the indivisible selection -> A4.6b1 PCG64 schedule."""
    if (not isinstance(
                evidence,
                _A4SingleTemplateCompletionMutationSelectionEvidence)
            or evidence._factory_token
            is not _SELECTION_EVIDENCE_FACTORY_TOKEN
            or evidence.schema_version
            != SELECTION_EVIDENCE_SCHEMA_VERSION
            or evidence.schedule_sha256
            != _selection_schedule_sha256(evidence)
            or getattr(evidence, '_frozen_sha256', None)
            != _selection_frozen_sha256(evidence)):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 RNG chain lacks factory-authenticated selection'
        )
    a44._require_completion_mutation_rng_tape(tape)
    before = a48c5._canonical_selection_pcg64_state(
        evidence.rng_before_state, 'rng_chain_before_state',
    )
    selection_after = a48c5._canonical_selection_pcg64_state(
        evidence.rng_after_state, 'rng_chain_selection_after_state',
    )
    tape_before = a48c5._canonical_selection_pcg64_state(
        tape.rng_before_state, 'rng_chain_tape_before_state',
    )
    tape_after = a48c5._canonical_selection_pcg64_state(
        tape.rng_after_state, 'rng_chain_tape_after_state',
    )
    if (selection_after != tape_before
            or evidence.config_sha256 != tape.config_sha256
            or evidence.dt_hex != tape.dt_hex
            or not a44._is_lower_hex_digest(evidence.config_sha256)
            or not a44._is_lower_hex_digest(evidence.schedule_sha256)
            or not a44._is_lower_hex_digest(tape.schedule_sha256)
            or not a44._is_lower_hex_digest(tape.source_provenance)):
        raise A4ReplicationStartCompletionMutationCommitError(
            'A4.8c7 selection/tape RNG schedule is discontinuous'
        )
    return a44._sha256_json({
        'schema': RNG_CHAIN_SCHEMA_VERSION,
        'source_binding_identity': list(
            evidence.source_binding_identity
        ),
        'synthetic_binding_identity': list(
            evidence.synthetic_binding_identity
        ),
        'synthetic_source_provenance': str(tape.source_provenance),
        'cell_id': int(evidence.cell_id),
        'dt_hex': str(evidence.dt_hex),
        'config_sha256': str(evidence.config_sha256),
        'selection_schedule_sha256': str(evidence.schedule_sha256),
        'completion_mutation_schedule_sha256': str(
            tape.schedule_sha256
        ),
        'rng_before_sha256': a44._sha256_json(before),
        'rng_after_selection_sha256': a44._sha256_json(selection_after),
        'tape_rng_before_sha256': a44._sha256_json(tape_before),
        'rng_after_sha256': a44._sha256_json(tape_after),
    })


def _required_protein_rows(cell):
    genes = set(cell.gene_specs)
    return max(
        len(genes.union(cell.proteins)),
        len(genes.union(cell.damaged_proteins)),
    )


def _require_start_completion_mutation_transaction_capacity(
        source, synthetic, final, source_binding, synthetic_binding,
        fresh_binding, plan):
    """Prove exact Q/S/W/P maxima across source, transient, and final."""
    template_length = int(source_binding.ragged.symbol_count)
    final_length = int(plan.final_lengths[0])
    required = {
        'Q': 3,
        'S': max(2 * template_length, template_length + final_length),
        'W': max(template_length, final_length),
        'P': max(
            _required_protein_rows(source),
            _required_protein_rows(synthetic),
            _required_protein_rows(final),
        ),
    }
    capacities = {
        'Q': int(source_binding.ragged.sequence_capacity),
        'S': int(source_binding.ragged.symbol_capacity),
        'W': int(source_binding.ragged.max_sequence_symbols),
        'P': int(source_binding.state.protein_capacity),
    }
    if any(capacities[name] < required[name] for name in required):
        raise a4.A4CapacityError(
            'A4.8c7 exact Q/S/W/P transaction capacity exceeded'
        )
    observed = (
        (source_binding, 1, template_length),
        (synthetic_binding, 3, 2 * template_length),
        (fresh_binding, 2, template_length + final_length),
    )
    for binding, sequences, symbols in observed:
        if (int(binding.ragged.sequence_count) != sequences
                or int(binding.ragged.symbol_count) != symbols
                or int(binding.ragged.sequence_capacity) != capacities['Q']
                or int(binding.ragged.symbol_capacity) != capacities['S']
                or int(binding.ragged.max_sequence_symbols)
                != capacities['W']
                or int(binding.state.protein_capacity) != capacities['P']):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 observed source/transient/final capacity differs'
            )
    return required


def _completion_mutation_rng_call_count(tape):
    """Count the exact high-level PCG64 calls recorded by A4.6b1."""
    calls = 1 + int(np.sum(np.asarray(tape.append_count, np.int64)))
    calls += int(np.sum(np.asarray(tape.substitution_count, np.int64)))
    calls += int(np.sum(np.asarray(tape.threshold_draw_mask, bool)))
    calls += 3 * int(np.count_nonzero(tape.insertion_count))
    calls += 2 * int(np.count_nonzero(tape.deletion_count))
    duplication = np.asarray(tape.structural_event_counts, np.int64)[
        :, int(a44.STRUCTURAL_DUPLICATION)
    ]
    inversion = np.asarray(tape.structural_event_counts, np.int64)[
        :, int(a44.STRUCTURAL_INVERSION)
    ]
    transposition = np.asarray(tape.structural_event_counts, np.int64)[
        :, int(a44.STRUCTURAL_TRANSPOSITION)
    ]
    calls += 2 * int(np.count_nonzero(duplication))
    calls += 2 * int(np.count_nonzero(inversion))
    calls += 3 * int(np.count_nonzero(transposition))
    calls += int(np.count_nonzero(tape.padding_count))
    return calls


@dataclass
class _A4ReplicationStartCompletionMutationCommitCandidate:
    """Private one-shot c7 candidate; never durable authority."""

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
    transaction_capacity: object
    rng_chain_sha256: str
    live_rng: object
    rng_before_state: object
    rng_after_selection_state: object
    rng_after_state: object
    dissipated_energy: float
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4ReplicationStartCompletionMutationEventScheduler(
        a48c6.A4ReplicationStartCompletionEventScheduler):
    """A4.8c6 scheduler plus bounded mutation-enabled start/completion."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_replication_start_completion_mutation_commit_active = False
        super(A4ReplicationStartCompletionMutationEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def replication_start_completion_mutation_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_replication_start_completion_mutation_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8c7 start/completion commit'
            )
        state = super(
            A4ReplicationStartCompletionMutationEventScheduler, self,
        ).state_dict()
        state['a4_replication_start_completion_mutation'] = (
            self.replication_start_completion_mutation_integration_state()
        )
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        own = _canonical_start_completion_mutation_integration_state(
            state.get('a4_replication_start_completion_mutation'),
        )
        inherited = a48c6.A4ReplicationStartCompletionEventScheduler.from_state(
            state,
        )
        if (own['config'] != a48a._config_state(inherited.a4_config)
                or own['device'] != inherited.a4_device):
            raise A4ReplicationStartCompletionMutationCommitError(
                'saved A4.8c6/A4.8c7 config or device differs'
            )
        scheduler = cls(
            a4_config=own['config'], device=own['device'],
            max_receipts=inherited.max_receipts,
        )
        scheduler._next_step_id = int(inherited._next_step_id)
        scheduler._backend_name = str(inherited._backend_name)
        scheduler._receipts = copy.deepcopy(inherited._receipts)
        return scheduler

    def _classify_start_completion_mutation_branch(
            self, world, cell, dt, config):
        """Use a deterministic config clone only as a resident scope probe."""
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationStartCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 replication source cell must be alive'
            )
        deterministic, _, _ = a44._completion_mutation_config(config)
        if a48c._strict_mutation_flag(deterministic):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 deterministic dispatch clone retained mutation'
            )
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
            resident_binding, dt, deterministic,
        )
        if not a4._is_tensor(probe.scope_error_code):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 dispatch probe escaped resident authority'
            )
        codes = probe.scope_error_code.detach().cpu().numpy().copy()
        completions = probe.completion_events.detach().cpu().numpy().copy()
        valid = probe.scope_valid.detach().cpu().numpy().copy()
        expected_shape = (int(source_binding.state.cell_capacity),)
        if (codes.shape != expected_shape
                or completions.shape != expected_shape
                or valid.shape != expected_shape):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 dispatch probe schema differs'
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
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 dispatch probe changed live authority'
            )
        if synthetic.replication_template is None:
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 dispatch lost its synthetic source'
            )
        code = int(codes[0])
        completed = bool(completions[0])
        supported = bool(valid[0])
        if code == int(a44.SCOPE_OK) and completed and supported:
            return _DISPATCH_START_COMPLETION_MUTATION
        if (code == int(a44.SCOPE_NONCOMPLETION)
                and not completed and not supported):
            return _DISPATCH_START_NONCOMPLETION
        raise A4ReplicationStartCompletionMutationCommitError(
            'replication branch is outside A4.8c7/c6 scope: %d' % code
        )

    def _require_final_start_completion_mutation_candidate(
            self, source, synthetic, candidate, evidence, tape, plan,
            source_binding, synthetic_binding, fresh_binding, dt, config):
        """Validate the live-inactive to final net state, not just c3 delta."""
        dt = a48c._strict_nonnegative_dt(dt)
        geometry = a48c6._inactive_start_completion_geometry(source_binding)
        _validate_single_template_completion_mutation_selection_evidence(
            evidence, source_binding, synthetic_binding, dt, config,
        )
        if evidence.rng_after_state != tape.rng_before_state:
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 selection after-state is not A4.6b1 before-state'
            )
        if (evidence.config_sha256 != tape.config_sha256
                or tape.config_sha256
                != _completion_mutation_config_sha256(config)):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 selection/tape config digest chain differs'
            )
        try:
            a48c3._completion_mutation_plan_scope(
                synthetic_binding, tape, plan, config,
            )
            self._validate_fresh_completion_mutation_candidate(
                synthetic, candidate, tape, plan, synthetic_binding,
                fresh_binding, config,
            )
        except (a48c3.A4ReplicationCompletionMutationCommitError,
                a4.A4Error) as exc:
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 synthetic completion-mutation scope differs'
            ) from exc

        if (candidate is source or candidate is synthetic
                or candidate.pools is source.pools
                or candidate.pools is synthetic.pools
                or np.shares_memory(candidate.pools, source.pools)
                or np.shares_memory(candidate.pools, synthetic.pools)
                or candidate.genomes is source.genomes
                or candidate.genomes is synthetic.genomes
                or candidate.genome_lesions is source.genome_lesions
                or candidate.genome_lesions is synthetic.genome_lesions
                or candidate.replication_copy is source.replication_copy
                or candidate.replication_copy is synthetic.replication_copy
                or candidate.mutation_events is source.mutation_events
                or candidate.mutation_events is synthetic.mutation_events
                or candidate.proteins is source.proteins
                or candidate.proteins is synthetic.proteins
                or candidate.damaged_proteins is source.damaged_proteins
                or candidate.damaged_proteins
                is synthetic.damaged_proteins
                or candidate.gene_specs is source.gene_specs
                or candidate.gene_specs is synthetic.gene_specs
                or any(
                    current is old for current in candidate.genomes
                    for old in source.genomes
                )
                or any(
                    current is old for current in candidate.genomes
                    for old in synthetic.genomes
                )
                or any(
                    candidate.gene_specs.get(fingerprint)
                    is source.gene_specs.get(fingerprint)
                    for fingerprint in source.gene_specs
                )
                or any(
                    candidate.gene_specs.get(fingerprint)
                    is synthetic.gene_specs.get(fingerprint)
                    for fingerprint in synthetic.gene_specs
                )
                or any(
                    current is old
                    for current in candidate.gene_specs.values()
                    for old in source.gene_specs.values()
                )
                or any(
                    current is old
                    for current in candidate.gene_specs.values()
                    for old in synthetic.gene_specs.values()
                )
                or any(
                    np.shares_memory(current, old)
                    for current in candidate.genomes
                    for old in source.genomes
                )
                or any(
                    np.shares_memory(current, old)
                    for current in candidate.genomes
                    for old in synthetic.genomes
                )
                or any(
                    np.shares_memory(candidate.genomes[-1], prior)
                    for prior in candidate.genomes[:-1]
                )
                or np.shares_memory(
                    candidate.genomes[-1], geometry['template'],
                )
                or np.shares_memory(
                    candidate.genomes[-1],
                    synthetic.replication_template,
                )):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 final candidate aliases live or synthetic biology'
            )
        source_state = copy.deepcopy(source.state_dict())
        candidate_state = copy.deepcopy(candidate.state_dict())
        if set(source_state) != set(candidate_state):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 final candidate schema differs from live source'
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
            if pickle.dumps(
                    source_state[name], protocol=pickle.HIGHEST_PROTOCOL,
                    ) != pickle.dumps(
                    candidate_state[name], protocol=pickle.HIGHEST_PROTOCOL,
                    ):
                raise A4ReplicationStartCompletionMutationCommitError(
                    'A4.8c7 final candidate changed out-of-scope field: '
                    + name
                )

        template_length = int(geometry['template_length'])
        final_length = int(plan.final_lengths[0])
        material_delta = int(plan.material_delta_symbols[0])
        final = np.asarray(
            plan.final_symbols[0, :final_length], np.uint8,
        )
        expected_novel = copy.deepcopy(source.novel_path_first_age)
        if expected_novel is None and a4.g2.sequence_has_novel_path(final):
            expected_novel = float(source.age)
        expected_mutations = a48c3._expected_mutation_events(source, plan)
        if (int(plan.append_count[0]) != template_length
                or int(plan.pre_structural_lengths[0]) != template_length
                or final_length != template_length + material_delta
                or int(plan.topology_sequence_deltas[0]) != -1
                or int(plan.topology_symbol_deltas[0]) != material_delta
                or int(plan.genome_material_symbols_after[0])
                != template_length + final_length
                or len(candidate.genomes) != 2
                or len(candidate.genomes) != len(source.genomes) + 1
                or not np.array_equal(
                    candidate.genomes[0], source.genomes[0],
                )
                or not np.array_equal(candidate.genomes[-1], final)
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
                != tuple(expected_mutations.items())
                or not a48c._float64_bits_equal(
                    candidate.pools,
                    np.asarray(plan.pools_after[0], np.float64),
                )
                or int(candidate.last_replication_symbols)
                != template_length
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
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 final cell differs from resident net plan'
            )

        a48b._require_cell_gene_specs(candidate, fresh_binding)
        ragged = fresh_binding.ragged
        source_ragged = source_binding.ragged
        old_lesions = int(source_ragged.lesion_count)
        if (int(ragged.sequence_count) != 2
                or int(ragged.symbol_count)
                != template_length + final_length
                or int(ragged.lesion_count) != old_lesions + 1
                or int(ragged.genome_counts[0]) != 2
                or bool(ragged.replication_active[0])
                or int(ragged.cell_sequence_offsets[0]) != 0
                or int(ragged.cell_sequence_offsets[1]) != 2
                or not np.array_equal(
                    ragged.sequence_offsets[:3],
                    np.asarray([
                        0, template_length, template_length + final_length,
                    ], np.int64),
                )
                or not np.array_equal(
                    ragged.symbols[:template_length], geometry['template'],
                )
                or not np.array_equal(
                    ragged.symbols[
                        template_length:template_length + final_length
                    ], final,
                )
                or not a48c._float64_bits_equal(
                    ragged.genome_lesions[:old_lesions],
                    source_ragged.genome_lesions[:old_lesions],
                )
                or not a48c._float64_bits_equal(
                    ragged.genome_lesions[old_lesions:old_lesions + 1],
                    np.asarray([plan.new_genome_lesions[0]], np.float64),
                )
                or float(ragged.replication_template_lesions[0]) != 0.0
                or float(ragged.replication_fractional[0]) != 0.0):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 final ragged topology differs from live net state'
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
                        raise A4ReplicationStartCompletionMutationCommitError(
                            'A4.8c7 final provenance is inconsistent'
                        )
                elif before != after:
                    raise A4ReplicationStartCompletionMutationCommitError(
                        'A4.8c7 final metadata changed: ' + name
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
            elif name == 'replication_active':
                expected = np.asarray(before, bool).copy()
                expected[0] = False
            elif name == 'genome_material_symbols':
                expected = np.asarray(before, np.int64).copy()
                expected[0] += final_length
            elif name == 'genome_lesion_mean':
                expected = np.asarray(before, np.float64).copy()
                expected[0] = expected_lesion_mean
            else:
                expected = np.asarray(before)
            actual = np.asarray(after)
            if name == 'genome_lesion_mean':
                same = a48c3._nonnegative_float64_within_one_ulp(
                    expected, actual,
                )
            else:
                same = (
                    a48c._float64_bits_equal(expected, actual)
                    if expected.dtype == np.dtype(np.float64)
                    else np.array_equal(expected, actual)
                )
            if (expected.dtype != actual.dtype
                    or expected.shape != actual.shape or not same):
                raise A4ReplicationStartCompletionMutationCommitError(
                    'A4.8c7 final translation state differs: ' + name
                )
        return _require_start_completion_mutation_transaction_capacity(
            source, synthetic, candidate, source_binding,
            synthetic_binding, fresh_binding, plan,
        )

    def _prepare_replication_start_completion_mutation_candidate(
            self, world, cell, dt, config):
        dt = a48c._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4ReplicationStartCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if not a48c._strict_mutation_flag(config):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 requires mutation=True'
            )
        _completion_mutation_config_sha256(config)
        if not bool(getattr(cell, 'alive', False)):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 replication source cell must be alive'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        source_objects = a48c5._inactive_source_object_snapshot(cell)
        if cell.replication_copy != []:
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 source copy must be empty'
            )
        _, source_binding = self._pack_host_binding(world, cell)
        a48c6._inactive_start_completion_geometry(source_binding)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        a48b._require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        source_artifacts = a48c._host_binding_artifact_snapshot(
            source_binding,
        )
        source_cell_state = a48c._cell_state_snapshot(cell)
        live_rng, rng_before = a48c._live_pcg64(world)

        _, synthetic_cell, synthetic_binding = self._synthetic_active_start(
            world, cell, source_binding,
        )
        synthetic_cell_state = a48c._cell_state_snapshot(synthetic_cell)
        synthetic_objects = a48c6._synthetic_active_start_objects(
            synthetic_cell,
        )
        synthetic_identity = a48a._binding_identity(synthetic_binding)
        synthetic_artifacts = a48c._host_binding_artifact_snapshot(
            synthetic_binding,
        )
        evidence = (
            _prepare_single_template_completion_mutation_selection_evidence(
                source_binding, synthetic_binding, dt, config, rng_before,
            )
        )
        tape = a44.prepare_completion_mutation_rng_tape(
            synthetic_binding, dt, config, evidence.rng_after_state,
        )
        if (tape.rng_before_state != evidence.rng_after_state
                or tape.config_sha256 != evidence.config_sha256):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 selection-to-tape chain is discontinuous'
            )
        rng_chain_sha256 = _completion_mutation_rng_chain_sha256(
            evidence, tape,
        )
        host_replay = a44.paid_replication_completion_mutation_plan(
            synthetic_binding, dt, config, tape,
        )
        a48c3._completion_mutation_plan_scope(
            synthetic_binding, tape, host_replay, config,
        )

        resident_binding = a4.bind_a4_translation(
            synthetic_binding.ragged.to_torch(device=self.a4_device),
            synthetic_binding.state.to_torch(device=self.a4_device),
        )
        resident_tape = tape.to_torch(
            synthetic_binding, dt, config, device=self.a4_device,
        )
        resident_plan = a44.paid_replication_completion_mutation_plan(
            resident_binding, dt, config, resident_tape,
        )
        if not a4._is_tensor(resident_plan.pools_after):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 resident final plan escaped Torch authority'
            )
        resident_artifacts = a48c3._resident_completion_mutation_artifacts(
            resident_binding, resident_tape, resident_plan, self.a4_device,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source) != synthetic_identity
                or a4._gene_cache_provenance(resident_cache)
                != synthetic_binding._cache_provenance):
            raise A4ReplicationStartCompletionMutationCommitError(
                'resident A4.8c7 synthetic source differs after readback'
            )
        resident_tape_host = resident_tape.to_numpy()
        a44.validate_a4_completion_mutation_rng_tape(
            resident_tape_host, resident_source, dt, config,
        )
        if not a48a._state_arrays_bit_exact(
                resident_tape_host.state_dict(), tape.state_dict()):
            raise A4ReplicationStartCompletionMutationCommitError(
                'resident A4.8c7 tape differs after readback'
            )
        plan = resident_plan.to_numpy()
        a48c3._completion_mutation_plan_scope(
            synthetic_binding, tape, plan, config,
        )
        a48c3._completion_mutation_plan_semantic_match(plan, host_replay)

        candidate_cell = self._build_completion_mutation_candidate(
            synthetic_cell, plan,
        )
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        transaction_capacity = (
            self._require_final_start_completion_mutation_candidate(
                cell, synthetic_cell, candidate_cell, evidence, tape, plan,
                source_binding, synthetic_binding, fresh_binding, dt, config,
            )
        )
        candidate_cell_state = a48c._cell_state_snapshot(candidate_cell)
        candidate_objects = a48c2._completion_candidate_object_snapshot(
            candidate_cell,
        )
        fresh_artifacts = a48c._host_binding_artifact_snapshot(fresh_binding)
        host_artifacts = a48c3._host_completion_mutation_artifacts(
            tape, host_replay, plan,
        )
        return _A4ReplicationStartCompletionMutationCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            scheduler_object_id=id(self), world_object_id=id(world),
            cell_object_id=id(cell), cell_id=int(cell.cell_id),
            generation=generation, dt_hex=dt.hex(),
            branch=_BRANCH_START_COMPLETION_MUTATION,
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding, source_artifacts=source_artifacts,
            source_cell_state=source_cell_state,
            source_gene_specs=source_gene_specs,
            source_objects=source_objects, evidence=evidence,
            evidence_state=evidence.state_dict(),
            synthetic_cell=synthetic_cell,
            synthetic_cell_state=synthetic_cell_state,
            synthetic_objects=synthetic_objects,
            synthetic_binding_identity=synthetic_identity,
            synthetic_binding=synthetic_binding,
            synthetic_artifacts=synthetic_artifacts,
            tape=tape, tape_state=tape.state_dict(),
            host_replay=host_replay,
            host_replay_state=host_replay.state_dict(),
            resident_binding=resident_binding,
            resident_tape=resident_tape, resident_plan=resident_plan,
            resident_artifacts=resident_artifacts, plan=plan,
            plan_state=plan.state_dict(), host_artifacts=host_artifacts,
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            candidate_objects=candidate_objects,
            fresh_binding=fresh_binding, fresh_artifacts=fresh_artifacts,
            transaction_capacity=transaction_capacity,
            rng_chain_sha256=rng_chain_sha256,
            live_rng=live_rng, rng_before_state=copy.deepcopy(rng_before),
            rng_after_selection_state=copy.deepcopy(
                evidence.rng_after_state
            ),
            rng_after_state=copy.deepcopy(tape.rng_after_state),
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.replication_start_completion_mutation_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_replication_start_completion_mutation_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(
                candidate,
                _A4ReplicationStartCompletionMutationCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 candidate is untrusted or already consumed'
            )
        dt = a48c._strict_nonnegative_dt(dt)
        live_rng, rng_before = a48c._live_pcg64(world)
        if (candidate.branch != _BRANCH_START_COMPLETION_MUTATION
                or id(self) != candidate.scheduler_object_id
                or id(world) != candidate.world_object_id
                or config is not world.config
                or self.replication_start_completion_mutation_integration_state()
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
                or candidate.evidence.rng_before_state
                != candidate.rng_before_state
                or candidate.evidence.rng_after_state
                != candidate.rng_after_selection_state
                or candidate.tape.rng_before_state
                != candidate.rng_after_selection_state
                or candidate.tape.rng_after_state
                != candidate.rng_after_state
                or candidate.evidence.config_sha256
                != candidate.tape.config_sha256
                or not a44._is_lower_hex_digest(
                    candidate.rng_chain_sha256
                )
                or _completion_mutation_rng_chain_sha256(
                    candidate.evidence, candidate.tape,
                ) != candidate.rng_chain_sha256
                or not a48c._float64_bits_equal(
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([candidate.dissipated_energy], np.float64),
                )
                or a48c._cell_state_snapshot(cell)
                != candidate.source_cell_state
                or not a48c5._inactive_source_objects_match(
                    cell, candidate.source_objects,
                )):
            raise A4ReplicationStartCompletionMutationCommitError(
                'live A4.8c7 cell/config/dt/RNG changed before claim'
            )

        _, current_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity):
            raise A4ReplicationStartCompletionMutationCommitError(
                'live A4.8c7 biological source changed before claim'
            )
        a4._require_translation_binding(candidate.source_binding)
        if (a48a._binding_identity(candidate.source_binding)
                != candidate.source_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.source_binding,
                ) != candidate.source_artifacts):
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 source binding changed before claim'
            )

        _validate_single_template_completion_mutation_selection_evidence(
            candidate.evidence, candidate.source_binding,
            candidate.synthetic_binding, dt, config,
        )
        if candidate.evidence.state_dict() != candidate.evidence_state:
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 selection evidence changed'
            )
        if (a48c._cell_state_snapshot(candidate.synthetic_cell)
                != candidate.synthetic_cell_state
                or not a48c._source_objects_match(
                    candidate.synthetic_cell, candidate.synthetic_objects,
                )):
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 synthetic cell changed before claim'
            )
        a48c6._require_synthetic_active_start_binding(
            cell, candidate.synthetic_cell, candidate.source_binding,
            candidate.synthetic_binding,
            a48c6._inactive_start_completion_geometry(
                candidate.source_binding
            ),
        )
        if (a48a._binding_identity(candidate.synthetic_binding)
                != candidate.synthetic_binding_identity
                or a48c._host_binding_artifact_snapshot(
                    candidate.synthetic_binding,
                ) != candidate.synthetic_artifacts):
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 synthetic binding changed before claim'
            )
        _, synthetic_now, synthetic_binding_now = (
            self._synthetic_active_start(world, cell, current_binding)
        )
        if (a48c._cell_state_snapshot(synthetic_now)
                != candidate.synthetic_cell_state
                or a48a._binding_identity(synthetic_binding_now)
                != candidate.synthetic_binding_identity):
            raise A4ReplicationStartCompletionMutationCommitError(
                'fresh A4.8c7 synthetic derivation changed before claim'
            )
        fresh_evidence = (
            _prepare_single_template_completion_mutation_selection_evidence(
                current_binding, synthetic_binding_now, dt, config,
                rng_before,
            )
        )
        if (fresh_evidence.state_dict() != candidate.evidence_state
                or fresh_evidence.rng_after_state
                != candidate.rng_after_selection_state):
            raise A4ReplicationStartCompletionMutationCommitError(
                'fresh A4.8c7 selection evidence changed before claim'
            )

        a44.validate_a4_completion_mutation_rng_tape(
            candidate.tape, candidate.synthetic_binding, dt, config,
        )
        a44.validate_a4_completion_mutation_plan(candidate.host_replay)
        a44.validate_a4_completion_mutation_plan(candidate.plan)
        if (a48c3._host_completion_mutation_artifacts(
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
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 tape/oracle/readback changed before claim'
            )
        fresh_tape = a44.prepare_completion_mutation_rng_tape(
            synthetic_binding_now, dt, config,
            fresh_evidence.rng_after_state,
        )
        if (fresh_tape.rng_before_state
                != fresh_evidence.rng_after_state
                or fresh_tape.config_sha256
                != fresh_evidence.config_sha256
                or not a48a._state_arrays_bit_exact(
                    fresh_tape.state_dict(), candidate.tape_state,
                )
                or fresh_tape.rng_after_state
                != candidate.rng_after_state):
            raise A4ReplicationStartCompletionMutationCommitError(
                'fresh A4.8c7 chained tape changed before claim'
            )
        if (_completion_mutation_rng_chain_sha256(
                fresh_evidence, fresh_tape,
                ) != candidate.rng_chain_sha256):
            raise A4ReplicationStartCompletionMutationCommitError(
                'fresh A4.8c7 combined RNG chain digest changed'
            )
        current_replay = a44.paid_replication_completion_mutation_plan(
            synthetic_binding_now, dt, config, fresh_tape,
        )
        a48c3._completion_mutation_plan_scope(
            synthetic_binding_now, fresh_tape, current_replay, config,
        )
        if not a48c3._completion_mutation_plan_states_bit_exact(
                current_replay, candidate.host_replay):
            raise A4ReplicationStartCompletionMutationCommitError(
                'fresh A4.8c7 NumPy final plan changed before claim'
            )

        a48c3._require_resident_completion_mutation_artifacts(
            candidate.resident_binding, candidate.resident_tape,
            candidate.resident_plan, self.a4_device,
            candidate.resident_artifacts,
        )
        resident_source, resident_cache = a48c._resident_source_readback(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.synthetic_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.synthetic_binding._cache_provenance):
            raise A4ReplicationStartCompletionMutationCommitError(
                'resident A4.8c7 source changed before claim'
            )
        resident_tape = candidate.resident_tape.to_numpy()
        a44.validate_a4_completion_mutation_rng_tape(
            resident_tape, resident_source, dt, config,
        )
        if not a48a._state_arrays_bit_exact(
                resident_tape.state_dict(), fresh_tape.state_dict()):
            raise A4ReplicationStartCompletionMutationCommitError(
                'resident A4.8c7 tape changed before claim'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        if not a48c3._completion_mutation_plan_states_bit_exact(
                resident_plan, candidate.plan):
            raise A4ReplicationStartCompletionMutationCommitError(
                'resident A4.8c7 final plan changed before claim'
            )
        a48c3._completion_mutation_plan_scope(
            synthetic_binding_now, fresh_tape, resident_plan, config,
        )
        a48c3._completion_mutation_plan_semantic_match(
            resident_plan, current_replay,
        )

        if (a48c._cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state
                or not a48c2._completion_candidate_objects_match(
                    candidate.candidate_cell, candidate.candidate_objects,
                )):
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 final candidate changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        capacity = self._require_final_start_completion_mutation_candidate(
            cell, candidate.synthetic_cell, candidate.candidate_cell,
            candidate.evidence, candidate.tape, candidate.plan,
            candidate.source_binding, candidate.synthetic_binding,
            fresh_now, dt, config,
        )
        if (capacity != candidate.transaction_capacity
                or a48c._host_binding_artifact_snapshot(
                    candidate.fresh_binding,
                ) != candidate.fresh_artifacts
                or a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCompletionMutationCommitError(
                'retained A4.8c7 final binding/capacity changed'
            )
        return candidate

    def _replication_start_completion_mutation_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after prepare and before final trust."""
        return candidate

    def _publish_replication_start_completion_mutation_candidate(
            self, world, cell, candidate):
        """Publish only the final resident-readback biology and RNG state."""
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
        # Formal066 creates one list at start and another at completion.
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

    def _published_start_completion_mutation_matches(
            self, world, cell, candidate, snapshot):
        source_objects = candidate.source_objects
        prepared = candidate.candidate_cell
        old_count = len(source_objects['genome_objects'])
        new_genome = cell.genomes[-1]
        source_mutation_keys = tuple(
            key for key, _ in source_objects['mutation_events_items']
        )
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
                or cell.replication_copy is source_objects['copy_object']
                or cell.replication_copy is prepared.replication_copy
                or cell.replication_copy != []
                or cell.mutation_events
                is not source_objects['mutation_events_object']
                or tuple(cell.mutation_events) != source_mutation_keys
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
                    np.asarray([world.dissipated_energy], np.float64),
                    np.asarray([snapshot['dissipated_energy']], np.float64),
                )):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        a48b._require_cell_gene_specs(cell, published_binding)
        if (a48a._binding_identity(published_binding)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4ReplicationStartCompletionMutationCommitError(
                'published A4.8c7 binding differs from final candidate'
            )

    def _commit_replication_start_completion_mutation_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_replication_start_completion_mutation_candidate(
            world, cell, dt, config, candidate,
        )
        snapshot = self._start_completion_publish_snapshot(world, cell)
        candidate.consumed = True
        plan = candidate.plan
        structural_counts = [
            int(value) for value in
            np.asarray(plan.structural_event_counts[0], np.int64)
        ]
        self.claim(cell, 'replication_cpu', metadata={
            'authority': (
                'A4.8c7-resident-mutation-enabled-start-completion-final-'
                'plan-pcg64-chain-atomic-commit'
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
            'synthetic_provenance': str(
                candidate.synthetic_binding.state.source_provenance
            ),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
            'rng_schedule_sha256': candidate.rng_chain_sha256,
        })
        try:
            self._publish_replication_start_completion_mutation_candidate(
                world, cell, candidate,
            )
            self._published_start_completion_mutation_matches(
                world, cell, candidate, snapshot,
            )
            self.annotate_claim(cell, 'replication_cpu', {
                'amount': int(plan.last_replication_symbols[0]),
                'work_performed': True,
                'template_start_events': 1,
                'completion_events': 1,
                'requested_symbols': int(plan.requested_symbols[0]),
                'substitution_events': int(plan.substitution_events[0]),
                'structural_event_counts': structural_counts,
                'material_delta_symbols': int(
                    plan.material_delta_symbols[0]
                ),
                'final_length': int(plan.final_lengths[0]),
                'rng_call_count': _completion_mutation_rng_call_count(
                    candidate.tape
                ),
                'rng_schedule_sha256': candidate.rng_chain_sha256,
                'selection_schedule_sha256': str(
                    candidate.evidence.schedule_sha256
                ),
                'completion_mutation_schedule_sha256': str(
                    candidate.tape.schedule_sha256
                ),
                'transaction_capacity': copy.deepcopy(
                    candidate.transaction_capacity
                ),
            })
        except BaseException:
            self._rollback_replication_start_completion_publish(
                world, cell, snapshot,
            )
            raise
        return None

    def cpu_replication(self, world, cell, dt, config=None):
        """Delegate inherited branches or atomically commit one c7 event."""
        _, record = self._context(world, cell, dt)
        if 'replication_cpu' in record['claimed']:
            raise a3s.A3DuplicateEventError(
                'duplicate A4.8c7 replication_cpu event'
            )
        config = world.config if config is None else config
        if config is not world.config:
            raise A4ReplicationStartCompletionMutationCommitError(
                'replication config must be the active world config object'
            )
        if (self._a4_replication_start_completion_mutation_commit_active
                or self._a4_replication_start_completion_commit_active
                or self._a4_replication_mutation_free_start_commit_active
                or self._a4_replication_start_commit_active
                or self._a4_replication_completion_mutation_commit_active
                or self._a4_replication_completion_commit_active
                or self._a4_replication_commit_active):
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8c7 replication commit is forbidden'
            )
        mutation_enabled = a48c._strict_mutation_flag(config)
        if (getattr(cell, 'replication_template', None) is not None
                or not mutation_enabled):
            return super(
                A4ReplicationStartCompletionMutationEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        branch = self._classify_start_completion_mutation_branch(
            world, cell, dt, config,
        )
        if branch == _DISPATCH_START_NONCOMPLETION:
            return super(
                A4ReplicationStartCompletionMutationEventScheduler, self,
            ).cpu_replication(world, cell, dt, config)
        if branch != _DISPATCH_START_COMPLETION_MUTATION:
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 dispatch produced an unknown branch'
            )
        self._a4_replication_start_completion_mutation_commit_active = True
        try:
            candidate = (
                self._prepare_replication_start_completion_mutation_candidate(
                    world, cell, dt, config,
                )
            )
            ready = (
                self._replication_start_completion_mutation_candidate_ready(
                    world, cell, dt, config, candidate,
                )
            )
            if ready is not candidate:
                raise A4ReplicationStartCompletionMutationCommitError(
                    'A4.8c7 ready hook must retain its private candidate'
                )
            return self._commit_replication_start_completion_mutation_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_replication_start_completion_mutation_commit_active = False


class Hybrid066WorldA4ReplicationStartCompletionMutation(
        a48c6.Hybrid066WorldA4ReplicationStartCompletion):
    """A4.8c6 world retaining bounded A4.8c7 authority on restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4ReplicationStartCompletionMutationEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(
                scheduler,
                A4ReplicationStartCompletionMutationEventScheduler):
            raise TypeError(
                'scheduler must be '
                'A4ReplicationStartCompletionMutationEventScheduler'
            )
        super(Hybrid066WorldA4ReplicationStartCompletionMutation, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4ReplicationStartCompletionMutationCommitError(
                'new A4.8c7 world requires explicit A4 config and device'
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
            Hybrid066WorldA4ReplicationStartCompletionMutation, self,
        ).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_replication_start_completion_mutation_device': (
                self.scheduler.a4_device
            ),
            'a4_replication_start_completion_mutation_config': (
                a48a._config_state(self.scheduler.a4_config)
            ),
        })
        return output

    def state_dict(self):
        state = super(
            Hybrid066WorldA4ReplicationStartCompletionMutation, self,
        ).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_replication_start_completion_mutation': (
                self.scheduler.
                replication_start_completion_mutation_integration_state()
            ),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8c7 save version/build differs'
            )
        own = _canonical_start_completion_mutation_integration_state(
            state.get('a4_replication_start_completion_mutation'),
        )
        start_completion = a48c6._canonical_start_completion_integration_state(
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
                'A4.8c7 save is missing aggregate composition sidecars'
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
            A4ReplicationStartCompletionMutationEventScheduler.from_state(
                state.get('scheduler', {}),
            )
        )
        if (scheduler.
                replication_start_completion_mutation_integration_state()
                != own
                or scheduler.replication_start_completion_integration_state()
                != start_completion
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
            raise A4ReplicationStartCompletionMutationCommitError(
                'A4.8a/b/c1/c2/c3/c4/c5/c6/c7 settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48c6.PORT_STATUS)
PORT_STATUS.update({
    'genome_replication': (
        'a4.8c7-mutation-enabled-inactive-one-genome-template-start-'
        'same-call-completion-selection-to-completion-mutation-tape-'
        'resident-final-plan-atomic-commit-a4.8c6-c5-c4-c3-c2-c1-'
        'inherited-other-inactive-early-noop-fail-closed'
    ),
    'material_mutation': (
        'a4.8c7-append-substitution-structural-padding-trim-one-pcg64-'
        'chain-exact-ledger-material-refund-cache-refresh-plus-a4.8c6-'
        'and-earlier-branches'
    ),
    'event_scheduler': (
        'a4.8c7-start-completion-mutation-plus-a4.8c6-start-completion-'
        'plus-a4.8c5-mutation-free-start-plus-a4.8c4-mutation-start-'
        'plus-a4.8c3-completion-mutation-plus-a4.8c2-completion-plus-'
        'a4.8c1-noncompletion-plus-a4.8b-translation-plus-a4.8a-'
        'hydrolysis-bounded-overrides'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'REPLICATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4ReplicationStartCompletionMutationCommitError',
    'A4ReplicationStartCompletionMutationEventScheduler',
    'Hybrid066WorldA4ReplicationStartCompletionMutation',
)
