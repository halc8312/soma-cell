# coding: utf-8
"""A4.8b paid-translation atomic commit bridge.

This bounded correctness bridge replaces only the existing ``translation_cpu``
event.  A fresh post-maintenance cell snapshot is evaluated by the resident
Torch A4.3 plan on an explicitly selected device.  The explicit resident
readback is the commit authority; an independent NumPy replay is only a
claim-before-commit semantic oracle.  No caller may inject a binding, plan, or
candidate, and the frozen CPU translation bridge is never used as fallback.

The A4.8a hydrolysis bridge remains inherited unchanged.  Replication and every
other event retain their existing A3 authority.  Event-local bindings, plans,
and candidates are disposable and are never save/clone authority.
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
SOURCE_DIR = (
    HERE if os.path.basename(HERE) == '0_6_8'
    else os.path.abspath(os.path.join(HERE, '..', 'src', '0_6_8'))
)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import SOMA_CELL_0_6_8_gpu_a3_scheduler as a3s
import SOMA_CELL_0_6_8_gpu_a4 as a4
import SOMA_CELL_0_6_8_gpu_a4_integration as a48a

try:
    import torch
except Exception:  # pragma: no cover - integration deliberately fails closed
    torch = None


BUILD = 'SOMA-CELL 0.6.8-GPU A4.8b'
BUILD_ID = BUILD
BUILD_LONG = BUILD + ' | paid translation atomic commit bridge'
SCHEMA_VERSION = '0.6.8-GPU-A4.8b-translation-atomic-commit'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False
TRANSLATION_ORACLE_ATOL = 2e-12

_CANDIDATE_FACTORY_TOKEN = object()
_INTEGRATION_STATE_KEYS = {'schema', 'config', 'device'}
_TRANSLATION_MUTABLE_ARRAYS = {
    'pools', 'last_translation', 'last_quiescence',
    'active_fingerprints', 'active_mass', 'active_count',
    'damaged_fingerprints', 'damaged_mass', 'damaged_count',
}
_TRANSLATION_PAID_OR_SYNC_POOLS = {
    int(a4.a3.POOL_FUEL), int(a4.a3.POOL_MINERAL), int(a4.a3.POOL_ATP),
    int(a4.a3.POOL_CATALYST), int(a4.a3.POOL_DAMAGED_PROTEIN),
}


class A4TranslationCommitError(a4.A4Error):
    """The bounded A4.8b source, replay, or commit contract failed."""


def _canonical_translation_integration_state(value):
    if not isinstance(value, Mapping) or set(value) != _INTEGRATION_STATE_KEYS:
        raise A4TranslationCommitError(
            'A4.8b save integration state is absent or noncanonical'
        )
    if value.get('schema') != SCHEMA_VERSION:
        raise A4TranslationCommitError('A4.8b save schema differs')
    raw_config = value.get('config')
    if not isinstance(raw_config, Mapping):
        raise A4TranslationCommitError('A4.8b saved config is invalid')
    try:
        config = a4.GPU068A4Config.from_state(dict(raw_config))
    except Exception as exc:
        raise A4TranslationCommitError(
            'A4.8b saved config is invalid'
        ) from exc
    device = a48a._canonical_device(value.get('device'))
    return {
        'schema': SCHEMA_VERSION,
        'config': a48a._config_state(config),
        'device': device,
    }


def _float64_bits_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and left.shape == right.shape
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _translation_states_bit_exact(left, right):
    return a48a._state_arrays_bit_exact(
        left.state_dict(), right.state_dict(),
    )


def _translation_semantic_match(resident, oracle):
    """Require exact discrete state and the registered fp64 oracle tolerance."""
    a4.validate_a4_translation_state(resident)
    a4.validate_a4_translation_state(oracle)
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        left = getattr(resident, name)
        right = getattr(oracle, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            if left != right:
                raise A4TranslationCommitError(
                    'resident/NumPy translation metadata differs: %s' % name
                )
            continue
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise A4TranslationCommitError(
                'resident/NumPy translation array schema differs: %s' % name
            )
        if left.dtype == np.dtype(np.float64):
            if (not np.isfinite(left).all() or not np.isfinite(right).all()
                    or np.any(np.abs(left - right) > TRANSLATION_ORACLE_ATOL)):
                raise A4TranslationCommitError(
                    'resident/NumPy translation fp64 result differs: %s' % name
                )
        elif not np.array_equal(left, right):
            raise A4TranslationCommitError(
                'resident/NumPy translation discrete result differs: %s' % name
            )
    return resident


def _translation_output_scope(source, result):
    """Prove that the resident plan changed only frozen translate fields."""
    a4.validate_a4_translation_state(source)
    a4.validate_a4_translation_state(result)
    for item in fields(a4.A4TranslationStateBatch):
        name = item.name
        before = getattr(source, name)
        after = getattr(result, name)
        if name not in a4._TRANSLATION_ARRAY_FIELDS:
            if before != after:
                raise A4TranslationCommitError(
                    'translation changed scalar source metadata: %s' % name
                )
            continue
        before = np.asarray(before)
        after = np.asarray(after)
        if before.dtype != after.dtype or before.shape != after.shape:
            raise A4TranslationCommitError(
                'translation changed array schema: %s' % name
            )
        if name in _TRANSLATION_MUTABLE_ARRAYS:
            continue
        if before.dtype == np.dtype(np.float64):
            same = _float64_bits_equal(before, after)
        else:
            same = np.array_equal(before, after)
        if not same:
            raise A4TranslationCommitError(
                'translation changed out-of-scope state: %s' % name
            )
    before_pools = np.asarray(source.pools, dtype=np.float64)
    after_pools = np.asarray(result.pools, dtype=np.float64)
    unchanged = [
        index for index in range(int(a4.a3.POOL_COUNT))
        if index not in _TRANSLATION_PAID_OR_SYNC_POOLS
    ]
    if not _float64_bits_equal(before_pools[:, unchanged],
                              after_pools[:, unchanged]):
        raise A4TranslationCommitError(
            'translation changed an out-of-scope material pool'
        )
    return result


def _translation_sync_performed(binding):
    """Reproduce the frozen gate that decides whether both dicts are synced.

    ``last_translation > 0`` is not this gate: ATP reserve and ``dt == 0``
    can execute both ``_sync_*`` calls while producing no protein.
    """
    a4._require_translation_binding(binding)
    state = binding.state
    if a4._is_tensor(state.pools):
        raise A4TranslationCommitError(
            'translation sync gate requires a host binding'
        )
    if int(state.cell_count) != 1:
        raise A4TranslationCommitError(
            'A4.8b translation commit requires exactly one cell'
        )
    ci = 0
    if not state.gene_expression or int(state.genome_count[ci]) == 0:
        return False
    specs = binding.cache.materialize_gene_specs_host()[ci]
    active = {
        int(state.active_fingerprints[ci, position]):
        float(state.active_mass[ci, position])
        for position in range(int(state.active_count[ci]))
    }
    metrics = a4._numpy_translation_cell_metrics(state, ci)

    def role_activity(role):
        total = 0.0
        for fingerprint, amount in active.items():
            spec = specs.get(fingerprint)
            if spec is not None and int(spec['role']) == int(role):
                total += amount * spec['efficiency']
        total /= 0.040
        if int(role) != int(a4.a3.ROLE_REGULATOR):
            total *= metrics['proteostasis'] * metrics['genome_factor']
        return float(total)

    translator_activity = role_activity(a4.a3.ROLE_TRANSLATOR)
    translator = translator_activity + (0.75 if state.external_translator else 0.0)
    if translator <= 1e-5 or not specs:
        return False
    total_weight = 0.0
    for spec in specs.values():
        total_weight += (
            spec['promoter'] * spec.get('copy_number', 1)
            * a4._numpy_protein_need(
                state, ci, spec, metrics, translator_activity,
            )
        )
    return bool(total_weight > 0.0)


def _dict_from_translation_row(state, prefix):
    count = int(getattr(state, prefix + '_count')[0])
    fingerprints = np.asarray(
        getattr(state, prefix + '_fingerprints'), dtype=np.int64,
    )[0, :count]
    masses = np.asarray(
        getattr(state, prefix + '_mass'), dtype=np.float64,
    )[0, :count]
    return {
        int(fingerprint): float(mass)
        for fingerprint, mass in zip(fingerprints, masses)
    }


def _require_cell_gene_specs(cell, binding, expected=None):
    """Bind the live Python cache to the genome-derived disposable cache."""
    a4._require_translation_binding(binding)
    materialized = binding.cache.materialize_gene_specs_host()
    if len(materialized) != 1:
        raise A4TranslationCommitError(
            'A4.8b gene-cache check requires exactly one cell'
        )
    if not a4._gene_specs_exact(cell.gene_specs, materialized[0]):
        raise A4TranslationCommitError(
            'live A4.8b gene_specs differs from genome-derived cache'
        )
    if expected is not None and not a4._gene_specs_exact(
            cell.gene_specs, expected):
        raise A4TranslationCommitError(
            'live A4.8b gene_specs changed before claim'
        )
    return cell.gene_specs


def _cell_state_snapshot(cell):
    """Freeze canonical cell biology for private-candidate attestation."""
    state_dict = getattr(cell, 'state_dict', None)
    if not callable(state_dict):
        raise A4TranslationCommitError(
            'A4.8b candidate cell must expose state_dict()'
        )
    try:
        return pickle.dumps(
            copy.deepcopy(state_dict()), protocol=pickle.HIGHEST_PROTOCOL,
        )
    except Exception as exc:
        raise A4TranslationCommitError(
            'A4.8b candidate cell state is not snapshot-safe'
        ) from exc


def _gene_specs_snapshot_matches(cell, snapshot):
    """Check outer/nested identity, order, and exact derived-cache values."""
    gene_specs = snapshot['gene_specs_object']
    items = snapshot['gene_specs_items']
    if (cell.gene_specs is not gene_specs
            or list(gene_specs.keys()) != [item[0] for item in items]):
        return False
    expected = {}
    for fingerprint, spec_object, saved_spec in items:
        if gene_specs[fingerprint] is not spec_object:
            return False
        expected[fingerprint] = saved_spec
    return a4._gene_specs_exact(gene_specs, expected)


def _resident_artifact_snapshot(binding, plan, device):
    """Attest the exact Torch objects and storages owning resident authority."""
    a4._require_translation_binding(binding)
    a4._validate_translation_resident_metadata(plan)
    expected_device = torch.device(str(device))
    batches = (
        (binding.ragged, a4._ARRAY_FIELDS),
        (binding.state, a4._TRANSLATION_ARRAY_FIELDS),
        (binding.cache, a4._GENE_ARRAY_FIELDS),
        (plan, a4._TRANSLATION_ARRAY_FIELDS),
    )
    for batch, names in batches:
        for name in names:
            value = getattr(batch, name)
            if not a4._is_tensor(value):
                raise A4TranslationCommitError(
                    'A4.8b resident artifact left Torch authority'
                )
            actual_device = value.device
            if (actual_device.type != expected_device.type
                    or (expected_device.index is not None
                        and actual_device.index != expected_device.index)):
                raise A4TranslationCommitError(
                    'A4.8b resident artifact moved to another device'
                )
    return {
        'object_ids': (
            id(binding), id(binding.ragged), id(binding.state),
            id(binding.cache), id(plan),
        ),
        'ragged_data_ptrs': dict(binding.ragged.data_ptrs()),
        'state_data_ptrs': dict(binding.state.data_ptrs()),
        'cache_data_ptrs': dict(binding.cache.data_ptrs()),
        'plan_data_ptrs': dict(plan.data_ptrs()),
    }


def _require_resident_artifact_snapshot(binding, plan, device, expected):
    actual = _resident_artifact_snapshot(binding, plan, device)
    if actual != expected:
        raise A4TranslationCommitError(
            'A4.8b resident authority object or storage changed before claim'
        )
    return binding, plan


@dataclass
class _A4TranslationCommitCandidate:
    """Private one-shot candidate; never durable world authority."""

    _factory_token: object
    cell_object_id: int
    cell_id: int
    generation: int
    dt_hex: str
    source_binding_identity: object
    source_binding: object
    resident_binding: object
    resident_plan: object
    resident_artifacts: object
    plan: object
    host_replay: object
    sync_performed: bool
    candidate_cell: object
    candidate_cell_state: bytes
    fresh_binding: object
    live_rng: object
    rng_state: object
    source_pools_object: object
    source_proteins_object: object
    source_damaged_proteins_object: object
    source_gene_specs_object: object
    source_gene_specs: object
    dissipated_energy: float
    integration_state: object
    world_config_snapshot: object
    consumed: bool = False


class A4TranslationEventScheduler(a48a.A4HydrolysisEventScheduler):
    """A4.8a scheduler with only paid translation additionally replaced."""

    def __init__(self, a4_config, device, max_receipts=128):
        self._a4_translation_commit_active = False
        super(A4TranslationEventScheduler, self).__init__(
            a4_config=a4_config, device=device, max_receipts=max_receipts,
        )

    def translation_integration_state(self):
        return {
            'schema': SCHEMA_VERSION,
            'config': a48a._config_state(self.a4_config),
            'device': self.a4_device,
        }

    def state_dict(self):
        if self._a4_translation_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'cannot serialize an active A4.8b translation commit'
            )
        state = super(A4TranslationEventScheduler, self).state_dict()
        state['a4_translation'] = self.translation_integration_state()
        return state

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        translation = _canonical_translation_integration_state(
            state.get('a4_translation'),
        )
        hydrolysis = a48a.A4HydrolysisEventScheduler.from_state(state)
        if (translation['config']
                != a48a._config_state(hydrolysis.a4_config)
                or translation['device'] != hydrolysis.a4_device):
            raise A4TranslationCommitError(
                'saved A4.8a/A4.8b config or device differs'
            )
        scheduler = cls(
            a4_config=translation['config'],
            device=translation['device'],
            max_receipts=hydrolysis.max_receipts,
        )
        scheduler._next_step_id = int(hydrolysis._next_step_id)
        scheduler._backend_name = str(hydrolysis._backend_name)
        scheduler._receipts = copy.deepcopy(hydrolysis._receipts)
        return scheduler

    def _build_translation_candidate_cell(self, cell, plan):
        # Isolate every mutable biological field, including fields outside the
        # A4 translation pack (for example membrane and last_repair_flux).
        # A ready-seam mutation must never reach the live source cell.
        candidate = copy.deepcopy(cell)
        candidate.pools = np.asarray(plan.pools[0], dtype=np.float64).copy()
        candidate.proteins = _dict_from_translation_row(plan, 'active')
        candidate.damaged_proteins = _dict_from_translation_row(
            plan, 'damaged',
        )
        candidate.last_translation = float(plan.last_translation[0])
        candidate.last_quiescence = float(plan.last_quiescence[0])
        return candidate

    def _validate_fresh_translation_candidate(
            self, candidate, plan, source_binding, fresh_binding):
        a4._require_translation_binding(fresh_binding)
        if int(fresh_binding.state.cell_count) != 1:
            raise A4TranslationCommitError(
                'fresh translation candidate is not exactly one cell'
            )
        source_identity = a48a._binding_identity(source_binding)
        fresh_identity = a48a._binding_identity(fresh_binding)
        _require_cell_gene_specs(candidate, fresh_binding)
        source_specs = source_binding.cache.materialize_gene_specs_host()[0]
        if not a4._gene_specs_exact(candidate.gene_specs, source_specs):
            raise A4TranslationCommitError(
                'translation candidate changed live gene_specs'
            )
        if (fresh_identity[0] != source_identity[0]
                or fresh_identity[2] != source_identity[2]):
            raise A4TranslationCommitError(
                'translation candidate changed genome or gene cache'
            )
        if not _translation_states_bit_exact(fresh_binding.state, plan):
            raise A4TranslationCommitError(
                'fresh translation candidate differs from resident plan'
            )
        if (candidate.proteins != _dict_from_translation_row(plan, 'active')
                or candidate.damaged_proteins
                != _dict_from_translation_row(plan, 'damaged')
                or not _float64_bits_equal(candidate.pools, plan.pools[0])
                or not _float64_bits_equal(
                    np.asarray([candidate.last_translation], dtype=np.float64),
                    np.asarray([plan.last_translation[0]], dtype=np.float64),
                )
                or not _float64_bits_equal(
                    np.asarray([candidate.last_quiescence], dtype=np.float64),
                    np.asarray([plan.last_quiescence[0]], dtype=np.float64),
                )):
            raise A4TranslationCommitError(
                'translation candidate CPU fields differ from resident plan'
            )
        return fresh_binding

    def _readback_resident_source(self, resident_binding):
        a4._require_translation_binding(resident_binding)
        ragged = resident_binding.ragged.to_numpy()
        state = resident_binding.state.to_numpy()
        binding = a4.bind_a4_translation(ragged, state)
        cache = resident_binding.cache.to_numpy()
        if (a4._gene_cache_provenance(cache)
                != binding._cache_provenance):
            raise A4TranslationCommitError(
                'resident translation cache differs from its source replay'
            )
        return binding, cache

    def _prepare_translation_candidate(self, world, cell, dt, config):
        dt = a48a._strict_nonnegative_dt(dt)
        if config is not world.config:
            raise A4TranslationCommitError(
                'translation config must be the active world config object'
            )
        generation = a48a._strict_counter(
            getattr(cell, 'generation', None), 'cell generation',
        )
        if not bool(getattr(cell, 'alive', False)):
            raise A4TranslationCommitError(
                'A4.8b translation source cell must be alive'
            )
        _, source_binding = self._pack_host_binding(world, cell)
        source_gene_specs = copy.deepcopy(cell.gene_specs)
        _require_cell_gene_specs(
            cell, source_binding, expected=source_gene_specs,
        )
        live_rng, rng_state = a48a._live_pcg64(world)
        sync_performed = _translation_sync_performed(source_binding)

        resident_ragged = source_binding.ragged.to_torch(
            device=self.a4_device,
        )
        resident_state = source_binding.state.to_torch(
            device=self.a4_device,
        )
        resident_binding = a4.bind_a4_translation(
            resident_ragged, resident_state,
        )
        resident_plan = a4.paid_translation_plan(
            resident_binding, dt,
        )
        if not a4._is_tensor(resident_plan.pools):
            raise A4TranslationCommitError(
                'A4.8b resident plan did not remain on Torch'
            )
        resident_artifacts = _resident_artifact_snapshot(
            resident_binding, resident_plan, self.a4_device,
        )

        resident_source, resident_cache = self._readback_resident_source(
            resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != a48a._binding_identity(source_binding)
                or a4._gene_cache_provenance(resident_cache)
                != source_binding._cache_provenance):
            raise A4TranslationCommitError(
                'resident translation source differs after readback'
            )
        plan = resident_plan.to_numpy()
        host_replay = a4.paid_translation_plan_numpy(source_binding, dt)
        _translation_output_scope(source_binding.state, plan)
        _translation_semantic_match(plan, host_replay)

        candidate_cell = self._build_translation_candidate_cell(cell, plan)
        _, fresh_binding = self._pack_host_binding(world, candidate_cell)
        self._validate_fresh_translation_candidate(
            candidate_cell, plan, source_binding, fresh_binding,
        )
        candidate_cell_state = _cell_state_snapshot(candidate_cell)
        return _A4TranslationCommitCandidate(
            _factory_token=_CANDIDATE_FACTORY_TOKEN,
            cell_object_id=id(cell),
            cell_id=int(cell.cell_id),
            generation=generation,
            dt_hex=dt.hex(),
            source_binding_identity=a48a._binding_identity(source_binding),
            source_binding=source_binding,
            resident_binding=resident_binding,
            resident_plan=resident_plan,
            resident_artifacts=resident_artifacts,
            plan=plan,
            host_replay=host_replay,
            sync_performed=bool(sync_performed),
            candidate_cell=candidate_cell,
            candidate_cell_state=candidate_cell_state,
            fresh_binding=fresh_binding,
            live_rng=live_rng,
            rng_state=copy.deepcopy(rng_state),
            source_pools_object=cell.pools,
            source_proteins_object=cell.proteins,
            source_damaged_proteins_object=cell.damaged_proteins,
            source_gene_specs_object=cell.gene_specs,
            source_gene_specs=source_gene_specs,
            dissipated_energy=float(world.dissipated_energy),
            integration_state=copy.deepcopy(
                self.translation_integration_state(),
            ),
            world_config_snapshot=a48a._world_config_snapshot(world),
        )

    def _revalidate_translation_candidate(
            self, world, cell, dt, config, candidate):
        if (not isinstance(candidate, _A4TranslationCommitCandidate)
                or candidate._factory_token is not _CANDIDATE_FACTORY_TOKEN
                or candidate.consumed):
            raise A4TranslationCommitError(
                'A4.8b candidate is untrusted or already consumed'
            )
        dt = a48a._strict_nonnegative_dt(dt)
        live_rng, rng_state = a48a._live_pcg64(world)
        if (config is not world.config
                or self.translation_integration_state()
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
                or live_rng is not candidate.live_rng
                or rng_state != candidate.rng_state
                or cell.pools is not candidate.source_pools_object
                or cell.proteins is not candidate.source_proteins_object
                or cell.damaged_proteins
                is not candidate.source_damaged_proteins_object
                or cell.gene_specs is not candidate.source_gene_specs_object
                or not _float64_bits_equal(
                    np.asarray(
                        [world.dissipated_energy], dtype=np.float64,
                    ),
                    np.asarray(
                        [candidate.dissipated_energy], dtype=np.float64,
                    ),
                )):
            raise A4TranslationCommitError(
                'live A4.8b cell/config/dt/RNG identity changed before claim'
            )

        _require_resident_artifact_snapshot(
            candidate.resident_binding, candidate.resident_plan,
            self.a4_device, candidate.resident_artifacts,
        )

        _, current_binding = self._pack_host_binding(world, cell)
        _require_cell_gene_specs(
            cell, current_binding, expected=candidate.source_gene_specs,
        )
        if (a48a._binding_identity(current_binding)
                != candidate.source_binding_identity):
            raise A4TranslationCommitError(
                'live A4.8b biological source changed before claim'
            )
        a4._require_translation_binding(candidate.source_binding)

        resident_source, resident_cache = self._readback_resident_source(
            candidate.resident_binding,
        )
        if (a48a._binding_identity(resident_source)
                != candidate.source_binding_identity
                or a4._gene_cache_provenance(resident_cache)
                != candidate.source_binding._cache_provenance):
            raise A4TranslationCommitError(
                'resident A4.8b source changed before claim'
            )
        if not _translation_states_bit_exact(
                resident_source.state, candidate.source_binding.state):
            raise A4TranslationCommitError(
                'resident A4.8b physiology changed before claim'
            )

        resident_plan = candidate.resident_plan.to_numpy()
        if not _translation_states_bit_exact(resident_plan, candidate.plan):
            raise A4TranslationCommitError(
                'resident A4.8b plan changed before claim'
            )
        host_replay = a4.paid_translation_plan_numpy(current_binding, dt)
        if not _translation_states_bit_exact(
                host_replay, candidate.host_replay):
            raise A4TranslationCommitError(
                'independent A4.8b host replay changed before claim'
            )
        _translation_output_scope(current_binding.state, resident_plan)
        _translation_semantic_match(resident_plan, host_replay)
        if _translation_sync_performed(current_binding) != bool(
                candidate.sync_performed):
            raise A4TranslationCommitError(
                'A4.8b dictionary sync gate changed before claim'
            )

        if not a4._gene_specs_exact(
                candidate.candidate_cell.gene_specs,
                candidate.source_gene_specs):
            raise A4TranslationCommitError(
                'fresh A4.8b candidate gene_specs changed before claim'
            )
        if (_cell_state_snapshot(candidate.candidate_cell)
                != candidate.candidate_cell_state):
            raise A4TranslationCommitError(
                'fresh A4.8b candidate biology changed before claim'
            )
        _, fresh_now = self._pack_host_binding(
            world, candidate.candidate_cell,
        )
        self._validate_fresh_translation_candidate(
            candidate.candidate_cell, candidate.plan,
            candidate.source_binding, fresh_now,
        )
        if (a48a._binding_identity(fresh_now)
                != a48a._binding_identity(candidate.fresh_binding)):
            raise A4TranslationCommitError(
                'fresh A4.8b candidate changed before claim'
            )
        return candidate

    def _translation_candidate_ready(
            self, world, cell, dt, config, candidate):
        """Protected test seam after preparation and before final validation."""
        return candidate

    def _publish_translation_candidate(self, world, cell, candidate):
        prepared = candidate.candidate_cell
        cell.pools[...] = np.asarray(prepared.pools, dtype=np.float64)
        if candidate.sync_performed:
            # Frozen _sync_* replaces both dict objects after the weight gate,
            # including ATP-reserve and dt==0 paths with zero translated mass.
            cell.proteins = dict(prepared.proteins)
            cell.damaged_proteins = dict(prepared.damaged_proteins)
        cell.last_translation = float(prepared.last_translation)
        cell.last_quiescence = float(prepared.last_quiescence)

    def _published_translation_matches(
            self, world, cell, candidate, snapshot):
        prepared = candidate.candidate_cell
        if (cell.pools is not snapshot['pools_object']
                or not _float64_bits_equal(cell.pools, prepared.pools)
                or not _gene_specs_snapshot_matches(cell, snapshot)
                or cell.proteins != prepared.proteins
                or cell.damaged_proteins != prepared.damaged_proteins
                or (candidate.sync_performed
                    and (cell.proteins is snapshot['proteins_object']
                         or cell.damaged_proteins
                         is snapshot['damaged_proteins_object']))
                or (not candidate.sync_performed
                    and (cell.proteins is not snapshot['proteins_object']
                         or cell.damaged_proteins
                         is not snapshot['damaged_proteins_object']))
                or not _float64_bits_equal(
                    np.asarray([cell.last_translation], dtype=np.float64),
                    np.asarray([prepared.last_translation], dtype=np.float64),
                )
                or not _float64_bits_equal(
                    np.asarray([cell.last_quiescence], dtype=np.float64),
                    np.asarray([prepared.last_quiescence], dtype=np.float64),
                )
                or world.rng is not candidate.live_rng
                or world.rng.bit_generator.state != candidate.rng_state
                or not _float64_bits_equal(
                    np.asarray(
                        [world.dissipated_energy], dtype=np.float64,
                    ),
                    np.asarray(
                        [snapshot['dissipated_energy']], dtype=np.float64,
                    ),
                )):
            raise A4TranslationCommitError(
                'A4.8b atomic publish differs from resident candidate'
            )
        _, published_binding = self._pack_host_binding(world, cell)
        self._validate_fresh_translation_candidate(
            cell, candidate.plan, candidate.source_binding,
            published_binding,
        )

    def _rollback_translation_publish(self, world, cell, snapshot):
        pools = snapshot['pools_object']
        pools[...] = snapshot['pools']
        cell.pools = pools
        proteins = snapshot['proteins_object']
        proteins.clear()
        proteins.update(snapshot['proteins'])
        cell.proteins = proteins
        damaged = snapshot['damaged_proteins_object']
        damaged.clear()
        damaged.update(snapshot['damaged_proteins'])
        cell.damaged_proteins = damaged
        if not _gene_specs_snapshot_matches(cell, snapshot):
            # Only a faulty publish seam may have touched this out-of-scope
            # cache after claim.  Restore the original outer and nested dict
            # objects, their insertion order, and their exact saved values.
            gene_specs = snapshot['gene_specs_object']
            items = snapshot['gene_specs_items']
            for _, spec_object, saved_spec in items:
                spec_object.clear()
                spec_object.update(copy.deepcopy(saved_spec))
            gene_specs.clear()
            for fingerprint, spec_object, _ in items:
                gene_specs[fingerprint] = spec_object
            cell.gene_specs = gene_specs
        cell.last_translation = snapshot['last_translation']
        cell.last_quiescence = snapshot['last_quiescence']
        world.dissipated_energy = snapshot['dissipated_energy']
        world.rng = snapshot['rng_object']
        world.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])

    def _commit_translation_candidate(
            self, world, cell, dt, config, candidate):
        self._revalidate_translation_candidate(
            world, cell, dt, config, candidate,
        )
        candidate.consumed = True
        plan = candidate.plan
        self.claim(cell, 'translation_cpu', metadata={
            'authority': 'A4.8b-resident-plan-atomic-cpu-cell-commit',
            'enabled': bool(plan.gene_expression),
            'device': self.a4_device,
            'genome_count': int(plan.genome_count[0]),
            'source_provenance': str(
                a4._translation_state_provenance(
                    candidate.source_binding.state,
                )
            ),
            'final_provenance': str(
                a4._translation_state_provenance(
                    candidate.fresh_binding.state,
                )
            ),
            'sync_performed': bool(candidate.sync_performed),
        })
        snapshot = {
            'pools_object': cell.pools,
            'pools': np.asarray(cell.pools, dtype=np.float64).copy(),
            'proteins_object': cell.proteins,
            'proteins': dict(cell.proteins),
            'damaged_proteins_object': cell.damaged_proteins,
            'damaged_proteins': dict(cell.damaged_proteins),
            'gene_specs_object': cell.gene_specs,
            'gene_specs_items': tuple(
                (fingerprint, spec, copy.deepcopy(spec))
                for fingerprint, spec in cell.gene_specs.items()
            ),
            'last_translation': float(cell.last_translation),
            'last_quiescence': float(cell.last_quiescence),
            'dissipated_energy': float(world.dissipated_energy),
            'rng_object': world.rng,
            'rng_state': copy.deepcopy(world.rng.bit_generator.state),
        }
        try:
            self._publish_translation_candidate(world, cell, candidate)
            self._published_translation_matches(
                world, cell, candidate, snapshot,
            )
            amount = float(plan.last_translation[0])
            self.annotate_claim(cell, 'translation_cpu', {
                'amount': amount,
                'work_performed': amount > 0.0,
                'sync_performed': bool(candidate.sync_performed),
                'rng_draw_count': 0,
            })
        except BaseException:
            self._rollback_translation_publish(world, cell, snapshot)
            raise
        return None

    def cpu_translation(self, world, cell, dt, config=None):
        """Replace one A3 translation event without external plan authority."""
        self._context(world, cell, dt)
        config = world.config if config is None else config
        if config is not world.config:
            raise A4TranslationCommitError(
                'translation config must be the active world config object'
            )
        if self._a4_translation_commit_active:
            raise a3s.A3SchedulerProtocolError(
                'nested A4.8b translation commit is forbidden'
            )
        self._a4_translation_commit_active = True
        try:
            candidate = self._prepare_translation_candidate(
                world, cell, dt, config,
            )
            ready = self._translation_candidate_ready(
                world, cell, dt, config, candidate,
            )
            if ready is not candidate:
                raise A4TranslationCommitError(
                    'A4.8b ready hook must retain its private candidate'
                )
            return self._commit_translation_candidate(
                world, cell, dt, config, candidate,
            )
        finally:
            self._a4_translation_commit_active = False


class Hybrid066WorldA4Translation(a48a.Hybrid066WorldA4Hydrolysis):
    """A4.8a world retaining A4.8b translation authority after restore."""

    def __init__(self, cpu_world, backend=None, scheduler=None,
                 gpu_config=None, a4_config=None, a4_device=None):
        if scheduler is None:
            scheduler = A4TranslationEventScheduler(
                a4_config=a4_config, device=a4_device,
            )
        elif not isinstance(scheduler, A4TranslationEventScheduler):
            raise TypeError('scheduler must be A4TranslationEventScheduler')
        super(Hybrid066WorldA4Translation, self).__init__(
            cpu_world, backend=backend, scheduler=scheduler,
            gpu_config=gpu_config, a4_config=a4_config,
            a4_device=a4_device,
        )

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, a4_config=None, a4_device=None):
        if a4_config is None or a4_device is None:
            raise A4TranslationCommitError(
                'new A4.8b world requires explicit A4 config and device'
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
        output = super(Hybrid066WorldA4Translation, self).summary()
        output.update({
            'gpu_build': BUILD,
            'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_full_world_step': False,
            'a4_translation_device': self.scheduler.a4_device,
            'a4_translation_config': a48a._config_state(
                self.scheduler.a4_config,
            ),
        })
        return output

    def state_dict(self):
        state = super(Hybrid066WorldA4Translation, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'a4_translation': self.scheduler.translation_integration_state(),
        })
        return state

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        state = dict(state or {})
        if (state.get('save_version') != SAVE_VERSION
                or state.get('build') != BUILD):
            raise A4TranslationCommitError(
                'A4.8b save version/build differs'
            )
        integration = _canonical_translation_integration_state(
            state.get('a4_translation'),
        )
        world = a3s.a2.s66.Formal066World.from_state(state['cpu_world'])
        if 'aggregate_sidecars' not in state:
            raise a3s.A3SchedulerProtocolError(
                'A4.8b save is missing aggregate composition sidecars'
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
        scheduler = A4TranslationEventScheduler.from_state(
            state.get('scheduler', {}),
        )
        if scheduler.translation_integration_state() != integration:
            raise A4TranslationCommitError(
                'A4.8b world/scheduler save settings differ'
            )
        return cls(world, backend=backend, scheduler=scheduler)


PORT_STATUS = dict(a48a.PORT_STATUS)
PORT_STATUS.update({
    'translation': (
        'a4.8b-single-cell-resident-paid-plan-cpu-cell-atomic-commit'
    ),
    'event_scheduler': (
        'a4.8b-translation-plus-a4.8a-hydrolysis-bounded-overrides'
    ),
    'genome_replication': 'cpu-authoritative-scheduler-bridged',
    'full_gpu_world_step': False,
})


__all__ = (
    'BUILD', 'BUILD_ID', 'BUILD_LONG', 'SCHEMA_VERSION', 'SAVE_VERSION',
    'FULL_GPU_WORLD_STEP', 'TRANSLATION_ORACLE_ATOL', 'PORT_STATUS',
    'A4TranslationCommitError', 'A4TranslationEventScheduler',
    'Hybrid066WorldA4Translation',
)
