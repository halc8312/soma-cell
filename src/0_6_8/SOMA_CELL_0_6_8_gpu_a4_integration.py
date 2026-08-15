# coding: utf-8
"""A4.8a single-cell genome-hydrolysis atomic commit bridge.

This bounded correctness bridge replaces only the existing A3
``genome_hydrolysis_cpu_rng`` scheduler event.  The post-lesion-gain CPU cell
is packed as a one-cell A4 source, the literal PCG64 schedule is replayed
without advancing the live generator, and the A4.7b deletion plan is executed
on an explicitly selected Torch device.  Full host replay and a fresh compact
binding complete before the A3 event is claimed.  Only then are the CPU cell
and live PCG64 state published, with local rollback on every publish failure.

The event-local source, tape, plan, and fresh binding are disposable.  They
are neither accepted from callers nor retained as save/clone authority.  A3
source and scheduler files remain the promoted authority for every other
event, and this module makes no full-world-step or performance claim.
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
import SOMA_CELL_0_6_8_gpu_a4_hydrolysis as a47

try:
    import torch
except Exception:  # pragma: no cover - integration deliberately fails closed
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8a'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | single-cell hydrolysis atomic commit bridge'
SCHEMA_VERSION = '0.6.8-GPU-A4.8a-hydrolysis-atomic-commit'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}


class A4HydrolysisCommitError(a47.A4HydrolysisError):
    """The bounded A4.8a source/commit contract failed closed."""


def _strict_nonnegative_dt(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise A4HydrolysisCommitError('hydrolysis dt must be a real scalar')
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise A4HydrolysisCommitError(
            'hydrolysis dt must be finite and nonnegative'
        )
    return result


def _config_state(config):
    return {
        item.name: int(getattr(config, item.name))
        for item in fields(a4.GPU068A4Config)
    }


def _canonical_device(value):
    if torch is None:
        raise A4HydrolysisCommitError(
            'PyTorch is required; A4.8a has no CPU fallback'
        )
    if value is None or str(value) == 'auto':
        raise A4HydrolysisCommitError(
            'A4.8a requires an explicit cpu or cuda device'
        )
    try:
        device = torch.device(str(value))
    except Exception as exc:
        raise A4HydrolysisCommitError(
            'invalid A4.8a device: %s' % value
        ) from exc
    if device.type not in ('cpu', 'cuda'):
        raise A4HydrolysisCommitError(
            'A4.8a supports explicit cpu or cuda devices only'
        )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise A4HydrolysisCommitError(
            'CUDA explicitly requested but unavailable'
        )
    return str(device)


def _canonical_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4HydrolysisCommitError(
            'A4.8a save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4HydrolysisCommitError('A4.8a save schema differs')
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4HydrolysisCommitError('A4.8a saved config is invalid')
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
    except Exception as exc:
        raise A4HydrolysisCommitError(
            'A4.8a saved config is invalid'
        ) from exc
    device = _canonical_device(value.get('device'))
    return {
        'schema': SCHEMA_VERSION,
        'config': _config_state(config),
        'device': device,
    }


def _same_float64_bits(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return (
        left.shape == right.shape
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _within_one_nonnegative_float64_ulp(left, right):
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


def _strict_counter(value, label):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise A4HydrolysisCommitError('%s must be an integer' % label)
    result = int(value)
    if result < 0:
        raise A4HydrolysisCommitError('%s must be nonnegative' % label)
    return result


def _live_pcg64(world):
    rng = getattr(world, 'rng', None)
    if (not isinstance(rng, np.random.Generator)
            or not isinstance(rng.bit_generator, np.random.PCG64)):
        raise A4HydrolysisCommitError(
            'A4.8a requires the active world NumPy PCG64 Generator'
        )
    state = a47._canonical_pcg64_state(
        copy.deepcopy(rng.bit_generator.state), 'live_rng_state',
    )
    return rng, state


def _binding_identity(binding):
    a4._require_translation_binding(binding)
    return (
        str(binding._ragged_provenance),
        str(binding._state_provenance),
        str(binding._cache_provenance),
    )


def _world_config_snapshot(world):
    state_method = getattr(getattr(world, 'config', None), 'state_dict', None)
    if not callable(state_method):
        raise A4HydrolysisCommitError(
            'A4.8a world config must expose state_dict()'
        )
    return pickle.dumps(
        copy.deepcopy(state_method()), protocol=pickle.HIGHEST_PROTOCOL,
    )


def _state_arrays_bit_exact(left, right):
    if set(left) != set(right):
        return False
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if isinstance(left_value, np.ndarray) or isinstance(
                right_value, np.ndarray):
            if (not isinstance(left_value, np.ndarray)
                    or not isinstance(right_value, np.ndarray)
                    or left_value.dtype != right_value.dtype
                    or left_value.shape != right_value.shape):
                return False
            if left_value.dtype == np.dtype(np.float64):
                if not np.array_equal(
                        left_value.view(np.uint64),
                        right_value.view(np.uint64)):
                    return False
            elif not np.array_equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


@dataclass
class _A4HydrolysisCommitCandidate:
    """Private one-shot event candidate; never durable world authority."""

    _factory_token: object
    cell_object_id: int
    cell_id: int
    generation: int
    dt_hex: str
    source_counter: int
    source_binding_identity: object
    source_binding: object
    tape: object
    plan: object
    resident_binding: object
    resident_tape: object
    resident_plan: object
    candidate_cell: object
    fresh_binding: object
    live_rng: object
    rng_before_state: object
    rng_after_state: object
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4HydrolysisEventScheduler(a3s.A3EventScheduler):
    """A3 scheduler with only genome hydrolysis replaced by A4.8a."""

    def __init__(self, a4_config, device, max_receipts=128):
        if a4_config is None:
            raise A4HydrolysisCommitError(
                'A4.8a requires an explicit fixed-capacity config'
            )
        try:
            self.a4_config = a4.GPU068A4Config.from_state(a4_config)
        except Exception as exc:
            raise A4HydrolysisCommitError(
                'invalid A4.8a fixed-capacity config'
            ) from exc
        self.a4_device = _canonical_device(device)
        self._a4_hydrolysis_commit_active = False
        super(A4HydrolysisEventScheduler, self).__init__(
            max_receipts=max_receipts,
        )

    def integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': _config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_hydrolysis_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8a hydrolysis commit'
            )
        state = super(A4HydrolysisEventScheduler, self).state_dict()
        state['a4_hydrolysis'] = self.integration_state()
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        integration = _canonical_integration_state(
            state.get('a4_hydrolysis'),
        )
        base = a3s.A3EventScheduler.from_state(state)
        scheduler = cls(
            a4_config=integration['config'],
            device=integration['device'],
            max_receipts=base.max_receipts,
        )
        scheduler._next_step_id = int(base._next_step_id)
        scheduler._backend_name = str(base._backend_name)
        scheduler._receipts = copy.deepcopy(base._receipts)
        return scheduler

    def _pack_host_binding(self, world, cell):
        adapter = a4.FullFidelityA4GenomeAdapter(self.a4_config)
        ragged = adapter.pack_cells([cell])
        state = a4.pack_a4_translation_state(
            [cell], ragged, world.config, self.a4_config,
        )
        binding = a4.bind_a4_translation(ragged, state)
        return adapter, binding

    def _build_candidate_cell(self, cell, plan):
        genomes = int(plan.genome_count)
        sequences = int(plan.sequence_count)
        lengths = np.asarray(plan.final_lengths, dtype=np.int64)
        symbols = np.asarray(plan.final_symbols, dtype=np.uint8)
        rows = [
            symbols[index, :int(lengths[index])].copy()
            for index in range(sequences)
        ]

        candidate = copy.copy(cell)
        candidate.genomes = [row.copy() for row in rows[:genomes]]
        candidate.genome_lesions = [
            float(value) for value in
            np.asarray(plan.genome_lesions_after, dtype=np.float64)[:genomes]
        ]
        if sequences - genomes == 2:
            candidate.replication_template = rows[genomes].copy()
            candidate.replication_copy = [
                int(value) for value in rows[genomes + 1]
            ]
        elif sequences == genomes:
            candidate.replication_template = None
            candidate.replication_copy = []
        else:  # The plan validator should already make this unreachable.
            raise A4HydrolysisCommitError(
                'A4.8a candidate has invalid replication topology'
            )
        candidate.pools = np.asarray(
            plan.pools_after[0], dtype=np.float64,
        ).copy()
        candidate._refresh_gene_cache()
        source_counter = _strict_counter(
            getattr(cell, 'genome_damage_events', None),
            'genome_damage_events',
        )
        delta = _strict_counter(
            int(plan.genome_damage_event_delta[0]),
            'genome_damage_event_delta',
        )
        candidate.genome_damage_events = source_counter + delta
        return candidate

    def _validate_fresh_candidate(self, candidate, plan, binding):
        ragged = binding.ragged
        state = binding.state
        sequences = int(plan.sequence_count)
        genomes = int(plan.genome_count)
        lengths = np.asarray(plan.final_lengths, dtype=np.int64)
        offsets = np.zeros((sequences + 1,), dtype=np.int64)
        if sequences:
            offsets[1:] = np.cumsum(lengths[:sequences], dtype=np.int64)
        flat = np.concatenate([
            np.asarray(
                plan.final_symbols[index, :int(lengths[index])],
                dtype=np.uint8,
            )
            for index in range(sequences)
        ]) if sequences else np.zeros((0,), dtype=np.uint8)
        if (int(ragged.cell_count) != 1
                or int(ragged.sequence_count) != sequences
                or int(ragged.genome_counts[0]) != genomes
                or int(ragged.symbol_count) != int(plan.symbol_count_after[0])
                or not np.array_equal(
                    ragged.sequence_offsets[:sequences + 1], offsets,
                )
                or not np.array_equal(ragged.symbols[:len(flat)], flat)
                or not np.array_equal(
                    ragged.genome_lesions[:genomes],
                    plan.genome_lesions_after[:genomes],
                )
                or not _same_float64_bits(state.pools[:1], plan.pools_after)
                or int(state.genome_material_symbols[0])
                != int(plan.genome_material_symbols_after[0])
                or not _within_one_nonnegative_float64_ulp(
                    state.genome_lesion_mean[:1],
                    plan.genome_lesion_mean_after,
                )):
            raise A4HydrolysisCommitError(
                'fresh compact A4.8a candidate differs from validated plan'
            )
        expected_counter = _strict_counter(
            getattr(candidate, 'genome_damage_events', None),
            'candidate genome_damage_events',
        )
        if expected_counter < int(plan.genome_damage_event_delta[0]):
            raise A4HydrolysisCommitError(
                'fresh A4.8a damage-event counter is invalid'
            )
        return binding

    def _prepare_candidate(self, world, cell, dt):
        dt = _strict_nonnegative_dt(dt)
        generation = _strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        source_counter = _strict_counter(
            getattr(cell, 'genome_damage_events', None),
            'genome_damage_events',
        )
        _, source_binding = self._pack_host_binding(world, cell)
        live_rng, rng_before = _live_pcg64(world)
        tape = a47.prepare_genome_hydrolysis_rng_tape(
            source_binding, dt, rng_before,
        )

        resident_ragged = source_binding.ragged.to_torch(
            device=self.a4_device,
        )
        resident_state = source_binding.state.to_torch(
            device=self.a4_device,
        )
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        resident_tape = tape.to_torch(
            source_binding, dt, device=self.a4_device,
        )
        resident_plan = a47.genome_hydrolysis_deletion_plan(
            resident_binding, dt, resident_tape,
        )
        plan = resident_plan.to_numpy()
        a47.validate_a4_hydrolysis_deletion_plan(
            plan, source_binding, dt, tape,
        )

        candidate_cell = self._build_candidate_cell(cell, plan)
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_candidate(candidate_cell, plan, fresh_binding)
        rng_after = a47._canonical_pcg64_state(
            copy.deepcopy(tape.rng_after_state), 'candidate_rng_after_state',
        )
        return _A4HydrolysisCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            cell_object_id=id(cell),
            cell_id=int(cell.cell_id),
            generation=generation,
            dt_hex=dt.hex(),
            source_counter=source_counter,
            source_binding_identity=_binding_identity(source_binding),
            source_binding=source_binding,
            tape=tape,
            plan=plan,
            resident_binding=resident_binding,
            resident_tape=resident_tape,
            resident_plan=resident_plan,
            candidate_cell=candidate_cell,
            fresh_binding=fresh_binding,
            live_rng=live_rng,
            rng_before_state=copy.deepcopy(rng_before),
            rng_after_state=copy.deepcopy(rng_after),
            integration_state=copy.deepcopy(self.integration_state()),
            world_config_snapshot=_world_config_snapshot(world),
        )

    def _revalidate_candidate(self, world, cell, dt, candidate):
        if (not isinstance(candidate, _A4HydrolysisCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4HydrolysisCommitError(
                'A4.8a commit candidate is untrusted or already consumed'
            )
        dt = _strict_nonnegative_dt(dt)
        live_rng, rng_state = _live_pcg64(world)
        candidate_rng_before = a47._canonical_pcg64_state(
            copy.deepcopy(candidate.rng_before_state),
            'candidate_rng_before_state',
        )
        candidate_rng_after = a47._canonical_pcg64_state(
            copy.deepcopy(candidate.rng_after_state),
            'candidate_rng_after_state',
        )
        tape_rng_before = a47._canonical_pcg64_state(
            copy.deepcopy(candidate.tape.rng_before_state),
            'candidate_tape_rng_before_state',
        )
        tape_rng_after = a47._canonical_pcg64_state(
            copy.deepcopy(candidate.tape.rng_after_state),
            'candidate_tape_rng_after_state',
        )
        if (self.integration_state() != candidate.integration_state
                or _world_config_snapshot(world)
                != candidate.world_config_snapshot
                or id(cell) != candidate.cell_object_id
                or int(cell.cell_id) != candidate.cell_id
                or _strict_counter(
                    getattr(cell, 'generation', None), 'cell generation',
                ) != candidate.generation
                or dt.hex() != candidate.dt_hex
                or live_rng is not candidate.live_rng
                or rng_state != candidate_rng_before
                or candidate_rng_before != tape_rng_before
                or candidate_rng_after != tape_rng_after
                or _strict_counter(
                    getattr(cell, 'genome_damage_events', None),
                    'genome_damage_events',
                ) != candidate.source_counter):
            raise A4HydrolysisCommitError(
                'live A4.8a cell/dt/RNG identity changed before claim'
            )
        _, current_binding = self._pack_host_binding(world, cell)
        if _binding_identity(current_binding) != candidate.source_binding_identity:
            raise A4HydrolysisCommitError(
                'live A4.8a biological source changed before claim'
            )
        a47.validate_a4_hydrolysis_rng_tape(
            candidate.tape, candidate.source_binding, dt,
        )
        a47.validate_a4_hydrolysis_deletion_plan(
            candidate.plan, candidate.source_binding, dt, candidate.tape,
        )
        # Re-read every retained resident input/output at the final trust
        # boundary.  This catches ordinary writes and Torch ``.data`` version
        # bypasses before the scheduler event is claimed.
        resident_ragged = candidate.resident_binding.ragged.to_numpy()
        resident_state = candidate.resident_binding.state.to_numpy()
        resident_source = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        resident_cache = candidate.resident_binding.cache.to_numpy()
        if (_binding_identity(resident_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.source_binding._cache_provenance):
            raise A4HydrolysisCommitError(
                'resident A4.8a source changed before claim'
            )
        resident_tape = candidate.resident_tape.to_numpy()
        a47.validate_a4_hydrolysis_rng_tape(
            resident_tape, resident_source, dt,
        )
        if not _state_arrays_bit_exact(
                resident_tape.state_dict(), candidate.tape.state_dict()):
            raise A4HydrolysisCommitError(
                'resident A4.8a RNG tape changed before claim'
            )
        resident_plan = candidate.resident_plan.to_numpy()
        a47.validate_a4_hydrolysis_deletion_plan(
            resident_plan, resident_source, dt, resident_tape,
        )
        if not _state_arrays_bit_exact(
                resident_plan.state_dict(), candidate.plan.state_dict()):
            raise A4HydrolysisCommitError(
                'resident A4.8a deletion plan changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_candidate(
            candidate.candidate_cell, candidate.plan, fresh_now,
        )
        if _binding_identity(fresh_now) != _binding_identity(
                candidate.fresh_binding):
            raise A4HydrolysisCommitError(
                'fresh A4.8a candidate changed before claim'
            )
        expected_counter = (
            candidate.source_counter
            + int(candidate.plan.genome_damage_event_delta[0])
        )
        if _strict_counter(
                getattr(candidate.candidate_cell, 'genome_damage_events', None),
                'candidate genome_damage_events',
                ) != expected_counter:
            raise A4HydrolysisCommitError(
                'fresh A4.8a damage-event counter changed before claim'
            )
        return candidate

    def _candidate_ready(self, world, cell, dt, candidate):
        """Protected test seam after preparation and before final validation."""
        return candidate

    def _publish_candidate(self, world, cell, candidate):
        prepared = candidate.candidate_cell
        cell.genomes[:] = [
            np.asarray(genome, dtype=np.uint8).copy()
            for genome in prepared.genomes
        ]
        cell.genome_lesions[:] = [
            float(value) for value in prepared.genome_lesions
        ]
        cell.pools[...] = np.asarray(prepared.pools, dtype=np.float64)
        cell.gene_specs = copy.deepcopy(prepared.gene_specs)
        cell.genome_damage_events = int(prepared.genome_damage_events)
        world.rng.bit_generator.state = copy.deepcopy(
            candidate.rng_after_state,
        )

    def _published_matches(self, world, cell, candidate):
        prepared = candidate.candidate_cell
        if (len(cell.genomes) != len(prepared.genomes)
                or any(not np.array_equal(left, right) for left, right in zip(
                    cell.genomes, prepared.genomes,
                ))
                or list(cell.genome_lesions) != list(prepared.genome_lesions)
                or not _same_float64_bits(cell.pools, prepared.pools)
                or cell.gene_specs != prepared.gene_specs
                or int(cell.genome_damage_events)
                != int(prepared.genome_damage_events)
                or world.rng is not candidate.live_rng
                or world.rng.bit_generator.state != candidate.rng_after_state):
            raise A4HydrolysisCommitError(
                'A4.8a atomic publish differs from prepared candidate'
            )

    def _rollback_publish(self, world, cell, snapshot):
        genomes_object = snapshot['genomes_object']
        genomes_object[:] = [item.copy() for item in snapshot['genomes']]
        cell.genomes = genomes_object
        lesions_object = snapshot['lesions_object']
        lesions_object[:] = list(snapshot['genome_lesions'])
        cell.genome_lesions = lesions_object
        pools_object = snapshot['pools_object']
        pools_object[...] = snapshot['pools']
        cell.pools = pools_object
        cell.gene_specs = snapshot['gene_specs_object']
        cell.genome_damage_events = snapshot['genome_damage_events']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])

    def _commit_candidate(self, world, cell, dt, hazards, candidate):
        self._revalidate_candidate(world, cell, dt, candidate)
        candidate.consumed = True
        tape = candidate.tape
        plan = candidate.plan
        self.claim(cell, 'genome_hydrolysis_cpu_rng', metadata={
            'authority': 'A4.8a-resident-plan-atomic-cpu-rng-commit',
            'enabled': True,
            'device': self.a4_device,
            'legacy_hazard_count': len(hazards),
            'eligible_genome_count': int(tape.draw_count),
            'draw_count': int(tape.draw_count),
            'hit_count': int(tape.hit_count),
            'source_provenance': str(tape.source_provenance),
            'final_provenance': str(
                candidate.fresh_binding.state.source_provenance
            ),
        })
        snapshot = {
            'genomes_object': cell.genomes,
            'genomes': [np.asarray(item, dtype=np.uint8).copy()
                        for item in cell.genomes],
            'lesions_object': cell.genome_lesions,
            'genome_lesions': list(cell.genome_lesions),
            'pools_object': cell.pools,
            'pools': np.asarray(cell.pools, dtype=np.float64).copy(),
            'gene_specs_object': cell.gene_specs,
            'genome_damage_events': int(cell.genome_damage_events),
            'rng_object': world.rng,
            'rng_state': copy.deepcopy(world.rng.bit_generator.state),
        }
        try:
            self._publish_candidate(world, cell, candidate)
            self._published_matches(world, cell, candidate)
            mutations = int(plan.genome_damage_event_delta[0])
            self.annotate_claim(cell, 'genome_hydrolysis_cpu_rng', {
                'mutation_count': mutations,
                'rng_draw_count': int(tape.draw_count) + int(tape.hit_count),
                'gene_cache_refresh_count': int(
                    plan.gene_cache_refresh_count[0]
                ),
                'work_performed': mutations > 0,
            })
        except BaseException:
            self._rollback_publish(world, cell, snapshot)
            raise
        return int(plan.genome_damage_event_delta[0]) > 0

    def cpu_genome_hydrolysis(self, world, cell, hazards, dt):
        """Replace one A3 hydrolysis event without accepting external plans."""
        self._context(world, cell, dt)
        if not isinstance(hazards, tuple):
            raise a3s.A3SchedulerProtocolError(
                'legacy genome hazards must retain ordered-tuple call shape'
            )
        if self._a4_hydrolysis_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8a hydrolysis commit is forbidden'
            )
        self._a4_hydrolysis_commit_active = True
        try:
            candidate = self._prepare_candidate(world, cell, dt)
            ready = self._candidate_ready(world, cell, dt, candidate)
            if ready is not candidate:
                raise A4HydrolysisCommitError(
                    'A4.8a ready hook must retain its private candidate'
                )
            return self._commit_candidate(
                world, cell, dt, hazards, candidate,
            )
        finally:
            self._a4_hydrolysis_commit_active = False


class Hybrid066WorldA4Hydrolysis(a3s.Hybrid066WorldA3):
    """A3 hybrid world retaining A4.8a authority across save and clone."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4HydrolysisEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(scheduler, A4HydrolysisEventScheduler):
            raise TypeError('scheduler must be A4HydrolysisEventScheduler')
        else:
            if (a4_config is not None
                    and _config_state(a4.GPU068A4Config.from_state(a4_config))
                    != _config_state(scheduler.a4_config)):
                raise A4HydrolysisCommitError(
                    'explicit A4 config differs from scheduler config'
                )
            if (a4_device is not None
                    and _canonical_device(a4_device) != scheduler.a4_device):
                raise A4HydrolysisCommitError(
                    'explicit A4 device differs from scheduler device'
                )
        super(Hybrid066WorldA4Hydrolysis, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4HydrolysisCommitError(
                'new A4.8a world requires explicit A4 config and device'
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
        output = super(Hybrid066WorldA4Hydrolysis, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_hydrolysis_device': self.scheduler.a4_device,
            'a4_hydrolysis_config': _config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4Hydrolysis, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_hydrolysis': self.scheduler.integration_state(),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4HydrolysisCommitError(
                'A4.8a save version/build differs'
            )
        integration = _canonical_integration_state(
            state.get('a4_hydrolysis'),
        )
        world = a3s.a2.s66.Formal066World.from_state(state['cpu_world'])
        if 'aggregate_sidecars' not in state:
            raise a3s.A3SchedulerProtocolError(
                'A4.8a save is missing aggregate composition sidecars'
            )
        a3s._restore_aggregate_sidecars(world, state['aggregate_sidecars'])
        config_state = copy.deepcopy(state.get('gpu_config', {}))
        if backend is None:
            backend = (
                backend_factory(config_state)
                if backend_factory is not None
                else a3s._default_backend(config_state)
            )
        scheduler = A4HydrolysisEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if scheduler.integration_state() != integration:
            raise A4HydrolysisCommitError(
                'A4.8a world/scheduler save settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a47.PORT_STATUS)
PORT_STATUS.update({
    'genome_symbol_hydrolysis_rng': (
        'a4.8a-single-cell-literal-pcg64-live-state-atomic-commit'
    ),
    'genome_symbol_hydrolysis': (
        'a4.8a-single-cell-resident-plan-cpu-cell-atomic-commit'
    ),
    'event_scheduler': (
        'a4.8a-bounded-hydrolysis-override-all-other-events-a3-authoritative'
    ),
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'PORT_STATUS',
    'A4HydrolysisCommitError', 'A4HydrolysisEventScheduler',
    'Hybrid066WorldA4Hydrolysis',
)
