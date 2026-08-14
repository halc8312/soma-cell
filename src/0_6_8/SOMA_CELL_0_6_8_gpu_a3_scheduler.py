# coding: utf-8
"""Fail-closed A3 event scheduler and frozen-0.6.6 world wrapper.

The A3 scientific backend is supplied through a checked capability boundary;
it is not imported when this module is imported.  The integrated path replaces
the inherited metabolism monolith with one backend call and guards every A2
surface/export/leak/radius/motion call by a (step, cell, event) claim.
"""
from __future__ import division

import copy
import importlib
import math
import os
import pickle
import sys
import types
from collections.abc import Mapping

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import SOMA_CELL_0_6_8_gpu_a2 as a2

BUILD = 'SOMA-CELL 0.6.8-GPU A3 scheduler'
BUILD_ID = BUILD
SCHEMA_VERSION = '0.6.8-GPU-A3-SCHEDULER.0'
SAVE_VERSION = 1
FULL_GPU_WORLD_STEP = False


def _a3_core_module():
    """Lazy core lookup used for aggregate-composition sidecar lifecycle."""
    module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3')
    required = ('aggregate_composition_state', 'apply_aggregate_composition_state',
                'AggregateCompositionState')
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise A3BackendCapabilityError(
            'A3 core missing sidecar capabilities: ' + ', '.join(missing)
        )
    return module

WORLD_EVENT_ORDER = (
    'chemostat', 'pre_p2', 'corpse_edna', 'interaction_surface',
    'interaction_hgt', 'cell_metabolism_loop',
    'collision_death_division', 'post_p2', 'washout',
)

# Public names/order frozen by docs/SOMA_CELL_0_6_8_GPU_A3_EVENT_ORDER_JA.md.
CELL_EVENT_ORDER = (
    'surface_exchange', 'gene_refresh', 'generic_reactions',
    'precursor_synthesis', 'maintenance', 'translation_cpu',
    'replication_cpu', 'surface_assembly', 'protein_damage', 'aggregation',
    'reactive_byproduct', 'membrane_oxidation', 'genome_lesion_gain',
    'genome_hydrolysis_cpu_rng', 'transporter_smoothing',
    'membrane_smoothing', 'ordinary_decay', 'waste_export', 'leak', 'radius',
    'motion', 'division_update_cpu', 'base_viability', 'cell_age',
    'repair_antioxidant', 'repair_chaperone', 'repair_protease',
    'repair_genome', 'repair_membrane', 'damage_viability',
    'sensorimotor_learning_cpu', 'mobile_export_cpu',
    'formal066_supplemental', 'segregation_plan', 'actual_split_cpu',
    'death_release_cpu',
)

_WORLD_RANK = {name: i for i, name in enumerate(WORLD_EVENT_ORDER)}
_CELL_RANK = {name: i for i, name in enumerate(CELL_EVENT_ORDER)}
_METABOLISM_END = _CELL_RANK['formal066_supplemental']
_METABOLISM_EVENTS = CELL_EVENT_ORDER[:_METABOLISM_END + 1]
_DIVISION_EVENTS = ('segregation_plan', 'actual_split_cpu')


class A3SchedulerError(RuntimeError):
    pass


class A3SchedulerProtocolError(A3SchedulerError):
    pass


class A3DuplicateEventError(A3SchedulerError):
    pass


class A3EventOrderError(A3SchedulerError):
    pass


class A3IncompleteCellError(A3SchedulerError):
    pass


class A3BackendCapabilityError(A3SchedulerProtocolError):
    pass


class A3ParentMonolithError(A3SchedulerProtocolError):
    pass


class A3BackendProtocol(object):
    """Runtime-only backend protocol.  It contains no fallback stubs."""

    REQUIRED_METHODS = (
        'diffuse_field_inplace', 'ligand_profiles', 'surface_exchange_inplace',
        'waste_export_inplace', 'leak_inplace', 'radius_inplace',
        'motion_inplace', 'metabolism_inplace',
        'segregation_plan_inplace', 'stats',
    )

    @classmethod
    def validate(cls, backend):
        if backend is None:
            raise A3BackendCapabilityError(
                'A3 backend is required; there is no implicit CPU fallback'
            )
        missing = [name for name in cls.REQUIRED_METHODS
                   if not callable(getattr(backend, name, None))]
        if missing:
            raise A3BackendCapabilityError(
                'A3 backend missing capabilities: ' + ', '.join(missing)
            )
        if not hasattr(backend, 'config'):
            raise A3BackendCapabilityError('A3 backend must expose config')
        return backend


def validate_backend_a3(backend):
    return A3BackendProtocol.validate(backend)


def _cell_id(cell):
    if isinstance(cell, (int, np.integer)):
        return int(cell)
    if not hasattr(cell, 'cell_id'):
        raise A3SchedulerProtocolError('cell must expose stable cell_id')
    return int(cell.cell_id)


def _world_age(world):
    age = getattr(world, 'age', None)
    return None if age is None else float(age)


class A3EventScheduler(object):
    """Strict exact-once scheduler with deterministic JSON-like receipts."""

    def __init__(self, max_receipts=128):
        self.max_receipts = int(max_receipts)
        if self.max_receipts < 1:
            raise ValueError('max_receipts must be >= 1')
        self._next_step_id = 0
        self._receipts = []
        self._active = None
        self._backend_name = 'unbound'

    @property
    def active(self):
        return self._active is not None

    @property
    def current_step_id(self):
        return None if self._active is None else int(self._active['step_id'])

    @property
    def receipts(self):
        return tuple(copy.deepcopy(self._receipts))

    @property
    def last_receipt(self):
        return None if not self._receipts else copy.deepcopy(self._receipts[-1])

    def bind_backend(self, backend):
        validate_backend_a3(backend)
        self._backend_name = type(backend).__name__
        return self

    def begin_step(self, world, cells):
        if self._active is not None:
            raise A3SchedulerProtocolError('another A3 step is already active')
        records = {}
        order = []
        for cell in list(cells):
            cid = _cell_id(cell)
            if cid in records:
                raise A3SchedulerProtocolError('duplicate cell_id: %d' % cid)
            records[cid] = {
                'object': cell,
                'alive_at_begin': bool(getattr(cell, 'alive', True)),
                'events': [], 'claimed': set(), 'last_rank': -1,
                'finished': False, 'metabolism_reserved': False,
                'metabolism_begun': False, 'metabolism_ended': False,
                'split_reserved': False,
            }
            order.append(cid)
        step_id = self._next_step_id
        self._next_step_id += 1
        self._active = {
            'step_id': int(step_id), 'world': world,
            'world_age_before': _world_age(world),
            'world_events': [], 'world_claimed': set(),
            'world_last_rank': -1, 'cells': records, 'cell_order': order,
            'ordinal': 0, 'metabolism_context': None,
            'attached_cell_ids': [], 'removed_cell_ids': [],
        }
        return int(step_id)

    def _require_active(self):
        if self._active is None:
            raise A3SchedulerProtocolError('no active A3 step')
        return self._active

    def _require_world(self, world):
        active = self._require_active()
        if world is not active['world']:
            raise A3SchedulerProtocolError('world is not the active A3 world')
        return active

    def _record(self, cell):
        active = self._require_active()
        cid = _cell_id(cell)
        if cid not in active['cells']:
            raise A3SchedulerProtocolError(
                'cell_id %d did not participate at begin_step' % cid
            )
        record = active['cells'][cid]
        if not isinstance(cell, (int, np.integer)) and record['object'] is not cell:
            raise A3SchedulerProtocolError('cell_id %d was rebound in-step' % cid)
        return cid, record

    def _ordinal(self):
        active = self._require_active()
        value = active['ordinal']
        active['ordinal'] += 1
        return int(value)

    def _detail(self, metadata):
        if metadata is not None and not isinstance(metadata, Mapping):
            raise A3SchedulerProtocolError('metadata must be a mapping or None')
        detail = copy.deepcopy(dict(metadata or {}))
        backend = str(detail.pop('backend', self._backend_name))
        return backend, detail

    def claim_world(self, event, status='executed', metadata=None):
        active = self._require_active()
        event, status = str(event), str(status)
        if event not in _WORLD_RANK:
            raise A3SchedulerProtocolError('unknown world event: %s' % event)
        if status not in ('executed', 'skipped'):
            raise A3SchedulerProtocolError('invalid event status: %s' % status)
        if event in active['world_claimed']:
            raise A3DuplicateEventError('duplicate world event: %s' % event)
        rank = _WORLD_RANK[event]
        missing_predecessors = [name for name in WORLD_EVENT_ORDER[:rank]
                                if name not in active['world_claimed']]
        if missing_predecessors:
            raise A3EventOrderError(
                'world event %s missing predecessors: %s' %
                (event, ', '.join(missing_predecessors))
            )
        if rank < active['world_last_rank']:
            raise A3EventOrderError('out-of-order world event: %s' % event)
        backend, detail = self._detail(metadata)
        entry = {'ordinal': self._ordinal(), 'event': event, 'status': status,
                 'backend': backend, 'metadata': detail}
        active['world_events'].append(entry)
        active['world_claimed'].add(event)
        active['world_last_rank'] = rank
        return copy.deepcopy(entry)

    def ensure_world_event(self, event, status='executed', metadata=None):
        active = self._require_active()
        if event in active['world_claimed']:
            return None
        return self.claim_world(event, status=status, metadata=metadata)

    def claim(self, cell, event, status='executed', metadata=None):
        active = self._require_active()
        cid, record = self._record(cell)
        event, status = str(event), str(status)
        if event not in _CELL_RANK:
            raise A3SchedulerProtocolError('unknown cell event: %s' % event)
        if status not in ('executed', 'skipped'):
            raise A3SchedulerProtocolError('invalid event status: %s' % status)
        if record['finished']:
            raise A3SchedulerProtocolError('cell_id %d is finished' % cid)
        if event in record['claimed']:
            raise A3DuplicateEventError(
                'duplicate (%d, %d, %s)' % (active['step_id'], cid, event)
            )
        rank = _CELL_RANK[event]
        missing_predecessors = [name for name in CELL_EVENT_ORDER[:rank]
                                if name not in record['claimed']]
        if missing_predecessors:
            raise A3EventOrderError(
                'cell_id %d event %s missing predecessors: %s' %
                (cid, event, ', '.join(missing_predecessors))
            )
        if rank < record['last_rank']:
            raise A3EventOrderError('cell_id %d out-of-order event %s' % (cid, event))
        if event == 'actual_split_cpu' and 'segregation_plan' not in record['claimed']:
            raise A3IncompleteCellError('actual split requires segregation plan')
        if event == 'death_release_cpu':
            missing = [name for name in _DIVISION_EVENTS if name not in record['claimed']]
            if missing:
                raise A3IncompleteCellError(
                    'death release before division close: ' + ', '.join(missing)
                )
        backend, detail = self._detail(metadata)
        entry = {'ordinal': self._ordinal(), 'event': event, 'status': status,
                 'backend': backend, 'metadata': detail}
        record['events'].append(entry)
        record['claimed'].add(event)
        record['last_rank'] = rank
        return copy.deepcopy(entry)

    def claim_once(self, cell, event, status='executed', metadata=None):
        return self.claim(cell, event, status=status, metadata=metadata)

    def annotate_claim(self, cell, event, metadata):
        if not isinstance(metadata, Mapping):
            raise A3SchedulerProtocolError('annotation must be a mapping')
        _, record = self._record(cell)
        for entry in record['events']:
            if entry['event'] == str(event):
                entry['metadata'].update(copy.deepcopy(dict(metadata)))
                return copy.deepcopy(entry)
        raise A3SchedulerProtocolError('unclaimed event: %s' % event)

    def _missing(self, record, names):
        return [name for name in names if name not in record['claimed']]

    def _skip_missing(self, cell, names, reason):
        _, record = self._record(cell)
        for name in names:
            if name not in record['claimed']:
                self.claim(cell, name, status='skipped',
                           metadata={'reason': str(reason)})

    def reserve_metabolism_dispatch(self, world, cell, dt):
        """Reserve the sole metabolism entry before calling the A3 backend."""
        self._require_world(world)
        cid, record = self._record(cell)
        if record['metabolism_reserved']:
            raise A3DuplicateEventError('duplicate metabolism dispatch for cell_id %d' % cid)
        if self._active['metabolism_context'] is not None:
            raise A3SchedulerProtocolError('nested A3 metabolism dispatch')
        record['metabolism_reserved'] = True
        record['dispatch_dt'] = float(dt)
        self.ensure_world_event('cell_metabolism_loop', metadata={'first_cell_id': cid})

    def begin_a3_metabolism(self, world, cell, dt):
        """Backend entry; returns False only for an already-dead cell."""
        self._require_world(world)
        cid, record = self._record(cell)
        if not record['metabolism_reserved']:
            raise A3SchedulerProtocolError('backend began metabolism without dispatch reservation')
        if record['metabolism_begun']:
            raise A3DuplicateEventError('metabolism already began for cell_id %d' % cid)
        if self._active['metabolism_context'] is not None:
            raise A3SchedulerProtocolError('nested A3 metabolism context')
        if float(dt) != float(record['dispatch_dt']):
            raise A3SchedulerProtocolError('backend changed metabolism dt')
        record['metabolism_begun'] = True
        self._active['metabolism_context'] = {
            'cell_id': cid, 'world': world, 'dt': float(dt),
            'position_before': np.asarray(cell.pos, dtype=float).copy(),
        }
        cell._defer_damage_viability = True
        if not bool(getattr(cell, 'alive', False)):
            self._skip_missing(cell, _METABOLISM_EVENTS, 'dead_at_metabolism_entry')
            cell._defer_damage_viability = False
            return False
        return True

    def _context(self, world, cell, dt=None):
        self._require_world(world)
        cid, record = self._record(cell)
        context = self._active['metabolism_context']
        if context is None or context['cell_id'] != cid:
            raise A3SchedulerProtocolError('CPU bridge used outside this cell metabolism')
        if dt is not None and float(dt) != context['dt']:
            raise A3SchedulerProtocolError('CPU bridge changed metabolism dt')
        return context, record

    def end_a3_metabolism(self, world, cell, dt=None):
        self._context(world, cell, dt)
        cid, record = self._record(cell)
        missing = self._missing(record, _METABOLISM_EVENTS)
        # Called from the backend's finally block.  Do not mask the original
        # duplicate/order/capacity exception with an incomplete-tail error.
        exceptional_exit = sys.exc_info()[0] is not None
        if missing and not exceptional_exit:
            raise A3IncompleteCellError(
                'cell_id %d metabolism missing: %s' % (cid, ', '.join(missing))
            )
        record['metabolism_ended'] = True
        if missing:
            record['metabolism_abort_missing'] = tuple(missing)
        self._active['metabolism_context'] = None
        return True

    def assert_metabolism_returned(self, cell):
        cid, record = self._record(cell)
        if not record['metabolism_begun'] or not record['metabolism_ended']:
            raise A3SchedulerProtocolError(
                'backend returned without begin/end for cell_id %d' % cid
            )

    def cpu_translation(self, world, cell, dt, config=None):
        self._context(world, cell, dt)
        config = world.config if config is None else config
        enabled = bool(getattr(config, 'gene_expression', True))
        self.claim(cell, 'translation_cpu', metadata={
            'authority': 'frozen-0.3', 'enabled': enabled,
            'genome_count': len(getattr(cell, 'genomes', ())),
        })
        result = _original_cell_method(cell, 'translate')(dt, config)
        amount = float(getattr(cell, 'last_translation', 0.0))
        self.annotate_claim(cell, 'translation_cpu', {
            'amount': amount, 'work_performed': amount > 0.0,
        })
        return result

    def cpu_replication(self, world, cell, dt, config=None):
        self._context(world, cell, dt)
        config = world.config if config is None else config
        enabled = bool(getattr(config, 'genome_replication', True))
        self.claim(cell, 'replication_cpu', metadata={
            'authority': 'frozen-0.6.6-wrapper-to-0.4', 'enabled': enabled,
            'genome_count': len(getattr(cell, 'genomes', ())),
        })
        result = _original_cell_method(cell, '_replicate_genome')(world, dt, config)
        amount = int(getattr(cell, 'last_replication_symbols', 0))
        self.annotate_claim(cell, 'replication_cpu', {
            'amount': amount, 'work_performed': amount > 0,
        })
        return result

    def cpu_genome_hydrolysis(self, world, cell, hazards, dt):
        """Consume frozen 0.3 RNG in genome insertion order.

        A hazard record is validated before the event claim and before any RNG
        or biological state is touched.  Exactly one random draw is consumed
        per record; an integer draw is consumed only when hydrolysis succeeds.
        """
        self._context(world, cell, dt)
        if not isinstance(hazards, tuple):
            raise A3SchedulerProtocolError('genome hazards must be an ordered tuple')
        required = {
            'genome_index', 'probability', 'lesion_after_gain', 'genome_length',
            'minimum_length', 'monomer_mass',
        }
        checked = []
        previous = -1
        for item in hazards:
            if not isinstance(item, Mapping) or not required.issubset(item):
                raise A3SchedulerProtocolError('invalid genome hydrolysis hazard')
            index = int(item['genome_index'])
            probability = float(item['probability'])
            lesion = float(item['lesion_after_gain'])
            length = int(item['genome_length'])
            minimum = int(item['minimum_length'])
            monomer = float(item['monomer_mass'])
            if index <= previous or index < 0 or index >= len(cell.genomes):
                raise A3SchedulerProtocolError('hazard genome order/index is invalid')
            if (not np.isfinite(probability) or probability < 0.0 or
                    probability > 1.0 or not np.isfinite(lesion)):
                raise A3SchedulerProtocolError('non-finite/invalid genome hazard')
            if length != len(cell.genomes[index]) or length <= minimum:
                raise A3SchedulerProtocolError('hazard genome length is stale/ineligible')
            if not np.isfinite(monomer) or monomer < 0.0:
                raise A3SchedulerProtocolError('invalid hazard monomer mass')
            if index >= len(cell.genome_lesions):
                raise A3SchedulerProtocolError('hazard lesion index is absent')
            checked.append((index, probability, length, minimum, monomer))
            previous = index
        self.claim(
            cell, 'genome_hydrolysis_cpu_rng',
            metadata={
                'hazard_count': len(checked), 'authority': 'frozen-0.3-rng',
                'enabled': True,
            },
        )
        changed = False
        mutations = 0
        for index, probability, expected_length, minimum, monomer in checked:
            draw = float(world.rng.random())
            if draw >= probability:
                continue
            sequence = cell.genomes[index]
            if len(sequence) != expected_length or len(sequence) <= minimum:
                raise A3SchedulerProtocolError('genome changed during hazard consumption')
            position = int(world.rng.integers(0, len(sequence)))
            cell.genomes[index] = np.concatenate(
                [sequence[:position], sequence[position + 1:]]
            )
            cell.pools[a2.s5.POOL_WASTE] += monomer
            cell.genome_lesions[index] *= 0.80
            cell.genome_damage_events += 1
            cell._refresh_gene_cache()
            changed = True
            mutations += 1
        self.annotate_claim(cell, 'genome_hydrolysis_cpu_rng', {
            'mutation_count': mutations,
            'rng_draw_count': len(checked) + mutations,
            'work_performed': bool(checked),
        })
        return changed

    def run_post_housekeeping_cpu(self, world, cell, dt, config=None):
        """Run frozen export/leak/radius/motion/division/viability/age order."""
        self._context(world, cell, dt)
        config = world.config if config is None else config
        cell.export_waste(world.field, dt, config)
        cell.leak(world, dt)
        cell.update_radius(dt)
        cell.update_motion(world.rng, dt)
        self.claim(cell, 'division_update_cpu', metadata={
            'authority': 'frozen-0.6.6-wrapper-to-0.4',
        })
        _original_cell_method(cell, 'update_division')(dt, config)
        self.claim(cell, 'base_viability', metadata={'authority': 'frozen-0.3-deferred'})
        cell.update_viability(dt)
        self.claim(cell, 'cell_age', metadata={'dt': float(dt)})
        cell.age += dt
        if not bool(cell.alive):
            cell._defer_damage_viability = False
            remaining = CELL_EVENT_ORDER[
                _CELL_RANK['repair_antioxidant']:_METABOLISM_END + 1
            ]
            self._skip_missing(cell, remaining, 'dead_after_base_viability')
            return False
        return True

    def run_post_repair_cpu(self, world, cell, dt, config=None):
        """Run frozen damage viability, learning and mobile-export wrappers."""
        context, _ = self._context(world, cell, dt)
        config = world.config if config is None else config
        cell._defer_damage_viability = False
        self.claim(cell, 'damage_viability', metadata={'authority': 'frozen-0.3'})
        cell._update_damage_viability(dt)

        # Exact tail of DamageProtoCell.metabolism after damage viability.
        current = cell.functional_age()
        alpha = 1.0 - math.exp(-0.15 * dt)
        old_ema = cell.functional_age_ema
        cell.functional_age_ema += alpha * (current - cell.functional_age_ema)
        slope = (cell.functional_age_ema - old_ema) / max(dt, 1e-9)
        cell.functional_age_slope += alpha * (slope - cell.functional_age_slope)
        cell._previous_functional_age = current

        if not bool(cell.alive):
            self.claim(cell, 'sensorimotor_learning_cpu', status='skipped',
                       metadata={'reason': 'dead_after_damage_viability'})
            self.claim(cell, 'mobile_export_cpu', status='skipped',
                       metadata={'reason': 'dead_after_damage_viability'})
            return False

        # Exact SensorimotorProtoCell tail, including passive-distance ledger.
        self.claim(cell, 'sensorimotor_learning_cpu',
                   metadata={'authority': 'frozen-0.4'})
        displacement = float(np.linalg.norm(
            a2.s4.wrapped_delta(context['position_before'], cell.pos)
        ))
        active_estimate = min(displacement, cell.last_motor_force * dt * 0.15)
        cell.cumulative_passive_distance += max(0.0, displacement - active_estimate)
        _original_cell_method(cell, '_update_sensorimotor_learning')(world, dt, config)

        mobile_enabled = (bool(getattr(config, 'mobile_elements', True)) and
                          bool(getattr(config, 'vesicle_export', True)))
        exports_before = int(getattr(cell, 'mobile_exports', 0))
        self.claim(cell, 'mobile_export_cpu', metadata={
            'authority': 'frozen-0.5', 'enabled': mobile_enabled,
        })
        _original_cell_method(cell, 'export_mobile_element')(world, dt)
        amount = int(getattr(cell, 'mobile_exports', 0)) - exports_before
        self.annotate_claim(cell, 'mobile_export_cpu', {
            'amount': amount, 'work_performed': amount > 0,
        })
        return bool(cell.alive)

    def reserve_split(self, world, cell):
        self._require_world(world)
        cid, record = self._record(cell)
        missing = self._missing(record, _METABOLISM_EVENTS)
        if missing:
            raise A3IncompleteCellError(
                'split cell_id %d before metabolism close: %s' %
                (cid, ', '.join(missing))
            )
        if record['split_reserved']:
            raise A3DuplicateEventError('duplicate split dispatch for cell_id %d' % cid)
        record['split_reserved'] = True

    def finish_split(self, cell, daughters):
        cid, record = self._record(cell)
        if not record['split_reserved']:
            raise A3SchedulerProtocolError('split was not reserved')
        missing = self._missing(record, _DIVISION_EVENTS)
        if missing:
            raise A3IncompleteCellError('split did not claim: ' + ', '.join(missing))
        ids = [] if daughters is None else [_cell_id(item) for item in daughters]
        self.annotate_claim(cell, 'actual_split_cpu', {
            'daughter_cell_ids': ids,
            'result': 'daughters' if daughters is not None else 'none',
        })
        return daughters

    def register_daughters(self, parent, daughters):
        """Register born cells so same-step washout/death remains receipted."""
        active = self._require_active()
        parent_id, _ = self._record(parent)
        for daughter in list(daughters or ()):
            cid = _cell_id(daughter)
            if cid in active['cells']:
                raise A3SchedulerProtocolError('daughter cell_id already registered: %d' % cid)
            active['cells'][cid] = {
                'object': daughter, 'alive_at_begin': False, 'born_this_step': True,
                'parent_cell_id': parent_id, 'events': [], 'claimed': set(),
                'last_rank': -1, 'finished': False,
                'metabolism_reserved': False, 'metabolism_begun': False,
                'metabolism_ended': True, 'split_reserved': False,
            }
            active['cell_order'].append(cid)
            self._skip_missing(daughter, _METABOLISM_EVENTS,
                               'born_after_metabolism_loop')
            self._skip_missing(daughter, _DIVISION_EVENTS,
                               'born_after_division_decision')
        return daughters

    def prepare_death_release(self, world, cell):
        self._require_world(world)
        _, record = self._record(cell)
        if record['metabolism_begun']:
            missing = self._missing(record, _METABOLISM_EVENTS)
            if missing:
                raise A3IncompleteCellError(
                    'death after incomplete metabolism: ' + ', '.join(missing)
                )
        else:
            self._skip_missing(cell, _METABOLISM_EVENTS, 'dead_before_metabolism')
        self._skip_missing(cell, _DIVISION_EVENTS, 'death_not_division')
        self.claim(cell, 'death_release_cpu',
                   metadata={'authority': 'frozen-inherited-death-release'})

    def close_division_pass(self):
        active = self._require_active()
        for cid in active['cell_order']:
            record = active['cells'][cid]
            if 'death_release_cpu' in record['claimed']:
                continue
            missing = self._missing(record, _METABOLISM_EVENTS)
            if missing:
                raise A3IncompleteCellError(
                    'cell_id %d incomplete before division close: %s' %
                    (cid, ', '.join(missing))
                )
            self._skip_missing(record['object'], _DIVISION_EVENTS,
                               'no_admitted_split')

    def note_attachments(self, attached_cell_ids, removed_cell_ids=()):
        active = self._require_active()
        active['attached_cell_ids'] = [int(value) for value in attached_cell_ids]
        active['removed_cell_ids'] = [int(value) for value in removed_cell_ids]

    def finish_cell(self, cell):
        cid, record = self._record(cell)
        if record['finished']:
            raise A3SchedulerProtocolError('finish_cell called twice for %d' % cid)
        if self._active['metabolism_context'] is not None:
            raise A3SchedulerProtocolError('cannot finish a cell inside metabolism')
        missing = self._missing(record, _METABOLISM_EVENTS)
        if missing:
            if not record['metabolism_begun'] and not bool(getattr(cell, 'alive', False)):
                self._skip_missing(cell, _METABOLISM_EVENTS, 'dead_before_metabolism')
            else:
                raise A3IncompleteCellError(
                    'cell_id %d missing: %s' % (cid, ', '.join(missing))
                )
        self._skip_missing(cell, _DIVISION_EVENTS, 'no_admitted_split')
        if 'death_release_cpu' not in record['claimed']:
            self.claim(cell, 'death_release_cpu', status='skipped',
                       metadata={'reason': 'not_released_this_step'})
        record['finished'] = True
        return {
            'cell_id': cid, 'finished': True,
            'events': copy.deepcopy(record['events']),
        }

    def _receipt(self, status, world, error=None):
        active = self._require_active()
        cells = []
        flat_executed, flat_skipped = [], []
        for entry in active['world_events']:
            item = {'scope': 'world', 'event': entry['event']}
            (flat_executed if entry['status'] == 'executed' else flat_skipped).append(item)
        for cid in active['cell_order']:
            record = active['cells'][cid]
            cell_entry = {'cell_id': cid, 'finished': bool(record['finished']),
                          'alive_at_begin': bool(record['alive_at_begin']),
                          'born_this_step': bool(record.get('born_this_step', False)),
                          'events': copy.deepcopy(record['events'])}
            if 'parent_cell_id' in record:
                cell_entry['parent_cell_id'] = int(record['parent_cell_id'])
            if 'metabolism_abort_missing' in record:
                cell_entry['metabolism_abort_missing'] = list(
                    record['metabolism_abort_missing']
                )
            cells.append(cell_entry)
            for entry in record['events']:
                item = {'scope': 'cell', 'cell_id': cid, 'event': entry['event']}
                (flat_executed if entry['status'] == 'executed' else flat_skipped).append(item)
        receipt = {
            'schema': SCHEMA_VERSION, 'step_id': int(active['step_id']),
            'status': str(status),
            'world_age_before': active['world_age_before'],
            'world_age_after': _world_age(world),
            'world_events': copy.deepcopy(active['world_events']),
            'cells': cells, 'executed': flat_executed, 'skipped': flat_skipped,
            'attached_cell_ids': list(active['attached_cell_ids']),
            'removed_cell_ids': list(active['removed_cell_ids']),
        }
        if error is not None:
            receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        return receipt

    def finish_step(self, world=None):
        active = self._require_active()
        world = active['world'] if world is None else world
        self._require_world(world)
        missing_world = [name for name in WORLD_EVENT_ORDER
                         if name not in active['world_claimed']]
        if missing_world:
            raise A3SchedulerProtocolError(
                'world step missing phases: ' + ', '.join(missing_world)
            )
        unfinished = [cid for cid in active['cell_order']
                      if not active['cells'][cid]['finished']]
        if unfinished:
            raise A3SchedulerProtocolError('unfinished cell_ids: %r' % unfinished)
        receipt = self._receipt('complete', world)
        self._receipts.append(copy.deepcopy(receipt))
        self._receipts = self._receipts[-self.max_receipts:]
        self._active = None
        return receipt

    def abort_step(self, error):
        active = self._require_active()
        receipt = self._receipt('aborted', active['world'], error=error)
        self._receipts.append(copy.deepcopy(receipt))
        self._receipts = self._receipts[-self.max_receipts:]
        self._active = None
        return receipt

    def state_dict(self):
        if self._active is not None:
            raise A3SchedulerProtocolError('cannot serialize an active scheduler')
        return {'schema': SCHEMA_VERSION, 'max_receipts': self.max_receipts,
                'next_step_id': self._next_step_id,
                'backend_name': self._backend_name,
                'receipts': copy.deepcopy(self._receipts)}

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        scheduler = cls(max_receipts=int(state.get('max_receipts', 128)))
        scheduler._next_step_id = int(state.get('next_step_id', 0))
        scheduler._backend_name = str(state.get('backend_name', 'unbound'))
        scheduler._receipts = copy.deepcopy(list(state.get('receipts', ())))[
            -scheduler.max_receipts:
        ]
        return scheduler


_ORIGINAL_CELL_ATTR = {
    'metabolism': '_soma068a3_original_metabolism',
    'translate': '_soma068a3_original_translate',
    '_replicate_genome': '_soma068a3_original_replicate_genome',
    'update_division': '_soma068a3_original_update_division',
    '_update_sensorimotor_learning': '_soma068a3_original_learning',
    'export_mobile_element': '_soma068a3_original_mobile_export',
    '_damage_partition_fractions': '_soma068a3_original_damage_partition',
    'split': '_soma068a3_original_split',
    'sense_environment': '_soma068a3_original_sense_environment',
    'process_corpse_contact': '_soma068a3_original_corpse_contact',
    'environmental_damage': '_soma068a3_original_environmental_damage',
}

_A2_ORIGINAL_ATTR = {
    'surface_exchange': '_soma068a2_original_surface_exchange',
    'export_waste': '_soma068a2_original_export_waste',
    'leak': '_soma068a2_original_leak',
    'update_radius': '_soma068a2_original_update_radius',
    'update_motion': '_soma068a2_original_update_motion',
}

_WORLD_ORIGINAL_ATTR = {
    '_chemostat': '_soma068a3_original_chemostat',
    '_pre_p2_step': '_soma068a3_original_pre_p2_step',
    '_step_corpses': '_soma068a3_original_step_corpses',
    '_resolve_collisions': '_soma068a3_original_resolve_collisions',
    '_handle_divisions_and_deaths': '_soma068a3_original_division_death_handler',
    '_post_p2_step': '_soma068a3_original_post_p2_step',
    '_washout': '_soma068a3_original_washout',
    '_release_dead_cell': '_soma068a3_original_release_dead_cell',
}


def _original_cell_method(cell, name):
    attribute = _ORIGINAL_CELL_ATTR.get(name)
    method = None if attribute is None else getattr(cell, attribute, None)
    if not callable(method):
        raise A3SchedulerProtocolError('missing frozen CPU method: %s' % name)
    return method


def _cell_runtime(cell):
    scheduler = getattr(cell, '_soma068a3_scheduler', None)
    backend = getattr(cell, '_soma068a3_backend', None)
    world = getattr(cell, '_soma068a3_world', None)
    if not isinstance(scheduler, A3EventScheduler):
        raise A3SchedulerProtocolError('cell has no A3 scheduler')
    validate_backend_a3(backend)
    scheduler._require_world(world)
    return scheduler, backend, world


def _world_runtime(world):
    scheduler = getattr(world, '_soma068a3_scheduler', None)
    backend = getattr(world, '_soma068a3_backend', None)
    if not isinstance(scheduler, A3EventScheduler):
        raise A3SchedulerProtocolError('world has no A3 scheduler')
    validate_backend_a3(backend)
    scheduler._require_world(world)
    return scheduler, backend


def _a3_metabolism(cell, world, dt, config):
    scheduler, backend, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('metabolism world differs from attached world')
    scheduler.reserve_metabolism_dispatch(world, cell, dt)
    if not bool(cell.alive):
        scheduler.begin_a3_metabolism(world, cell, dt)
        scheduler.end_a3_metabolism(world, cell, dt)
        scheduler.assert_metabolism_returned(cell)
        return None
    # The only authoritative metabolism entry.  Never call the saved monolith.
    result = backend.metabolism_inplace(world, cell, dt, scheduler)
    scheduler.assert_metabolism_returned(cell)
    return result


def _a3_surface_exchange(cell, field, dt, config):
    scheduler, backend, world = _cell_runtime(cell)
    if getattr(field, '_world_ref', world) is not world:
        raise A3SchedulerProtocolError('surface field differs from attached world')
    scheduler.ensure_world_event('interaction_surface',
                                 metadata={'first_cell_id': int(cell.cell_id)})
    has_particles = len(getattr(field, 'amount', ())) > 0
    enabled = bool(getattr(config, 'transport', True))
    scheduler.claim(cell, 'surface_exchange', metadata={
        'authority': 'A2', 'particle_count': len(field.amount),
        'enabled': enabled,
    })
    result = backend.surface_exchange_inplace(field, cell, dt, config)
    uptake = float(np.sum(np.asarray(getattr(cell, 'last_uptake', ()), dtype=float)))
    uptake += float(getattr(cell, 'last_uptake_alt', 0.0))
    scheduler.annotate_claim(cell, 'surface_exchange', {
        'amount': uptake,
        'work_performed': bool(getattr(cell, 'alive', False)) and has_particles,
    })
    return result


def _a3_export_waste(cell, field, dt, config):
    scheduler, backend, world = _cell_runtime(cell)
    if getattr(field, '_world_ref', world) is not world:
        raise A3SchedulerProtocolError('export field differs from attached world')
    enabled = bool(getattr(config, 'waste_export', True))
    waste = float(cell.pools[a2.s5.POOL_WASTE])
    scheduler.claim(cell, 'waste_export', metadata={
        'authority': 'A2', 'waste_before': waste, 'enabled': enabled,
    })
    result = backend.waste_export_inplace(field, cell, dt, config)
    amount = float(getattr(cell, 'last_export', 0.0))
    scheduler.annotate_claim(cell, 'waste_export', {
        'amount': amount, 'work_performed': amount > 0.0,
    })
    return result


def _a3_leak(cell, world, dt):
    scheduler, backend, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('leak world differs from attached world')
    atp_before = float(cell.pools[a2.s5.POOL_ATP])
    copy_before = len(getattr(cell, 'replication_copy', ()))
    scheduler.claim(cell, 'leak', metadata={'authority': 'A2', 'enabled': True})
    result = backend.leak_inplace(world, cell, dt)
    amount = float(getattr(cell, 'last_leak', 0.0))
    work = (amount > 0.0 or float(cell.pools[a2.s5.POOL_ATP]) != atp_before or
            len(getattr(cell, 'replication_copy', ())) != copy_before)
    scheduler.annotate_claim(cell, 'leak', {
        'amount': amount, 'work_performed': bool(work),
    })
    return result


def _a3_radius(cell, dt):
    scheduler, backend, _ = _cell_runtime(cell)
    before = float(cell.radius)
    scheduler.claim(cell, 'radius', metadata={'authority': 'A2', 'enabled': True})
    result = backend.radius_inplace(cell, dt)
    scheduler.annotate_claim(cell, 'radius', {
        'amount': abs(float(cell.radius) - before), 'work_performed': True,
    })
    return result


def _a3_motion(cell, rng, dt):
    scheduler, backend, world = _cell_runtime(cell)
    if rng is not world.rng:
        raise A3SchedulerProtocolError('motion must consume the active world RNG')
    scheduler.claim(cell, 'motion',
                    metadata={'authority': 'A2-plan/frozen-CPU-RNG', 'enabled': True})
    result = backend.motion_inplace(cell, rng, dt)
    scheduler.annotate_claim(cell, 'motion', {
        'work_performed': True, 'rng_draw_count': 2,
    })
    return result


def _forbidden_cpu_entry(name):
    def _blocked(cell, *args, **kwargs):
        raise A3ParentMonolithError(
            '%s must be invoked through the A3 scheduler CPU bridge' % name
        )
    return _blocked


def _a3_damage_partition(cell, world):
    scheduler, backend, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('segregation world differs from attached world')
    _, record = scheduler._record(cell)
    if not record['split_reserved']:
        raise A3SchedulerProtocolError(
            'segregation plan is legal only inside an admitted CPU split call'
        )
    config = world.config
    enabled = (bool(getattr(config, 'damage_segregation', True)) and
               not bool(getattr(config, 'forced_symmetric_damage', False)))
    scheduler.claim(cell, 'segregation_plan', metadata={
        'authority': 'A3', 'actual_split_only': True, 'enabled': enabled,
    })
    result = backend.segregation_plan_inplace(world, cell, scheduler=scheduler)
    cached = getattr(cell, '_soma068a3_last_segregation_plan', {})
    paid = float(cached.get('paid', 0.0))
    effective = float(cached.get('effective', 0.0))
    scheduler.annotate_claim(cell, 'segregation_plan', {
        'amount': paid, 'effective': effective,
        'work_performed': paid > 0.0 or effective > 0.0,
    })
    # This callback is immediately followed by the inherited material split.
    # Claim that CPU event now, before its first mutation.
    scheduler.claim(cell, 'actual_split_cpu', metadata={
        'authority': 'frozen-0.6.6-inherited-split', 'dispatch': 'admitted',
    })
    return result


def _partition_aggregate_sidecar(parent_state, parent_aggregate, daughters):
    """Preserve typed/unresolved aggregate identity through frozen split mass."""
    core = _a3_core_module()
    parent_aggregate = float(parent_aggregate)
    if parent_aggregate < -1e-12:
        raise A3SchedulerProtocolError('negative parent aggregate scalar')
    states = []
    for daughter in daughters:
        daughter_total = float(daughter.pools[a2.s5.POOL_AGGREGATE])
        if daughter_total < -1e-12:
            raise A3SchedulerProtocolError('negative daughter aggregate scalar')
        if parent_aggregate <= 1e-15:
            if daughter_total > 2e-10:
                raise A3SchedulerProtocolError(
                    'daughter aggregate appeared from zero parent sidecar'
                )
            items, unresolved = (), 0.0
        else:
            fraction = daughter_total / parent_aggregate
            items = tuple((int(key), float(value) * fraction)
                          for key, value in parent_state.items)
            unresolved = float(parent_state.unresolved) * fraction
            # Keep exact scalar closure despite the final floating operation.
            residual = daughter_total - (sum(value for _, value in items) + unresolved)
            unresolved += residual
            if unresolved < -2e-10:
                raise A3SchedulerProtocolError('sidecar partition produced negative unresolved mass')
            unresolved = max(0.0, unresolved)
        state = core.AggregateCompositionState(items, unresolved)
        core.apply_aggregate_composition_state(daughter, state)
        states.append(state)
    return tuple(states)


def _a3_split(cell, world):
    scheduler, _, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('split world differs from attached world')
    scheduler.reserve_split(world, cell)
    core = _a3_core_module()
    parent_state = core.aggregate_composition_state(cell)
    parent_aggregate = float(cell.pools[a2.s5.POOL_AGGREGATE])
    daughters = _original_cell_method(cell, 'split')(world)
    if daughters is not None:
        _partition_aggregate_sidecar(parent_state, parent_aggregate, daughters)
    scheduler.finish_split(cell, daughters)
    scheduler.register_daughters(cell, daughters)
    return daughters


def _a3_sense_environment(cell, world, dt, config):
    scheduler, _, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('sense world differs from attached world')
    scheduler.ensure_world_event('interaction_surface',
                                 metadata={'first_cell_id': int(cell.cell_id)})
    return _original_cell_method(cell, 'sense_environment')(world, dt, config)


def _a3_corpse_contact(cell, world, dt):
    scheduler, _, attached_world = _cell_runtime(cell)
    if world is not attached_world:
        raise A3SchedulerProtocolError('HGT world differs from attached world')
    scheduler.ensure_world_event('interaction_hgt',
                                 metadata={'first_cell_id': int(cell.cell_id)})
    return _original_cell_method(cell, 'process_corpse_contact')(world, dt)


def _a3_environmental_damage(cell, rng, dt, enabled=True):
    scheduler, _, world = _cell_runtime(cell)
    if rng is not world.rng:
        raise A3SchedulerProtocolError('environmental damage must use world RNG')
    scheduler.ensure_world_event('cell_metabolism_loop',
                                 metadata={'first_cell_id': int(cell.cell_id)})
    return _original_cell_method(cell, 'environmental_damage')(rng, dt, enabled=enabled)


def _dynamic_diffusion(field, dt):
    backend = getattr(field, '_soma068a3_backend', None)
    validate_backend_a3(backend)
    return backend.diffuse_field_inplace(field, dt)


def _dynamic_profiles(cell, field):
    _, backend, _ = _cell_runtime(cell)
    return backend.ligand_profiles(field, cell)


def _world_original(world, name):
    attribute = _WORLD_ORIGINAL_ATTR[name]
    method = getattr(world, attribute, None)
    if not callable(method):
        raise A3SchedulerProtocolError('missing frozen world method: %s' % name)
    return method


def _a3_world_chemostat(world):
    scheduler, _ = _world_runtime(world)
    enabled = bool(getattr(world.config, 'eco66_chemostat', False))
    scheduler.claim_world('chemostat', metadata={'enabled': enabled})
    return _world_original(world, '_chemostat')()


def _a3_world_pre_p2(world, dt):
    scheduler, _ = _world_runtime(world)
    scheduler.claim_world('pre_p2')
    return _world_original(world, '_pre_p2_step')(dt)


def _a3_world_corpses(world, dt):
    scheduler, _ = _world_runtime(world)
    active = bool(getattr(world, 'corpses', ())) or getattr(world, 'edna', None) is not None
    scheduler.claim_world('corpse_edna', metadata={'work_available': active})
    return _world_original(world, '_step_corpses')(dt)


def _close_empty_interaction_phases(scheduler):
    active = scheduler._require_active()
    any_alive = any(bool(getattr(record['object'], 'alive', False))
                    for record in active['cells'].values())
    for name in ('interaction_surface', 'interaction_hgt', 'cell_metabolism_loop'):
        if name not in active['world_claimed']:
            if any_alive:
                raise A3SchedulerProtocolError(
                    'living cells reached division handler without world phase %s' % name
                )
            scheduler.claim_world(name, status='skipped',
                                  metadata={'reason': 'no_living_cells'})


def _a3_world_division_death(world):
    scheduler, _ = _world_runtime(world)
    _close_empty_interaction_phases(scheduler)
    if 'collision_death_division' not in scheduler._active['world_claimed']:
        raise A3SchedulerProtocolError('division handler reached before collision claim')
    result = _world_original(world, '_handle_divisions_and_deaths')()
    scheduler.close_division_pass()
    return result


def _a3_world_collisions(world):
    scheduler, _ = _world_runtime(world)
    _close_empty_interaction_phases(scheduler)
    scheduler.claim_world('collision_death_division')
    return _world_original(world, '_resolve_collisions')()


def _a3_world_post_p2(world, dt):
    scheduler, _ = _world_runtime(world)
    scheduler.claim_world('post_p2')
    return _world_original(world, '_post_p2_step')(dt)


def _a3_world_washout(world):
    scheduler, _ = _world_runtime(world)
    enabled = bool(getattr(world.config, 'eco66_washout', False))
    scheduler.claim_world('washout', metadata={'enabled': enabled})
    return _world_original(world, '_washout')()


def _a3_world_release_dead(world, cell):
    scheduler, _ = _world_runtime(world)
    scheduler.prepare_death_release(world, cell)
    return _world_original(world, '_release_dead_cell')(cell)


def _attach_backend_to_world_a3(world, backend, scheduler):
    """Attach dynamic, rebind-safe A3 hooks to a frozen Formal066World."""
    if not isinstance(world, a2.s66.Formal066World):
        raise TypeError('A3 integration requires Formal066World')
    validate_backend_a3(backend)
    if not isinstance(scheduler, A3EventScheduler):
        raise TypeError('scheduler must be A3EventScheduler')
    scheduler.bind_backend(backend)

    # Establish A1/A2 originals first.  Their frozen CPU originals are retained
    # solely for explicit detach/raw-CPU operation, never automatic fallback.
    a2._attach_backend_to_world_a2(world, backend)
    world._soma068a3_backend = backend
    world._soma068a3_scheduler = scheduler

    field = world.field
    field._soma068a3_backend = backend
    field._soma068a3_scheduler = scheduler
    field.step_diffusion = types.MethodType(_dynamic_diffusion, field)

    for name, attribute in _WORLD_ORIGINAL_ATTR.items():
        if not hasattr(world, attribute):
            method = getattr(world, name, None)
            if not callable(method):
                raise A3SchedulerProtocolError('world is missing method %s' % name)
            setattr(world, attribute, method)
    world._chemostat = types.MethodType(_a3_world_chemostat, world)
    world._pre_p2_step = types.MethodType(_a3_world_pre_p2, world)
    world._step_corpses = types.MethodType(_a3_world_corpses, world)
    world._resolve_collisions = types.MethodType(_a3_world_collisions, world)
    world._handle_divisions_and_deaths = types.MethodType(
        _a3_world_division_death, world
    )
    world._post_p2_step = types.MethodType(_a3_world_post_p2, world)
    world._washout = types.MethodType(_a3_world_washout, world)
    world._release_dead_cell = types.MethodType(_a3_world_release_dead, world)

    for cell in list(world.cells):
        cell._soma068a3_backend = backend
        cell._soma068a3_scheduler = scheduler
        cell._soma068a3_world = world

        for name, attribute in _ORIGINAL_CELL_ATTR.items():
            if not hasattr(cell, attribute):
                method = getattr(cell, name, None)
                if not callable(method):
                    raise A3SchedulerProtocolError('cell is missing method %s' % name)
                setattr(cell, attribute, method)

        # A1's closure captured the backend at first attachment.  Replace it by
        # an attribute lookup so restored/daughter cells cannot retain a stale
        # backend closure.
        cell._particle_ligand_profiles = types.MethodType(_dynamic_profiles, cell)

        cell.metabolism = types.MethodType(_a3_metabolism, cell)
        cell.surface_exchange = types.MethodType(_a3_surface_exchange, cell)
        cell.export_waste = types.MethodType(_a3_export_waste, cell)
        cell.leak = types.MethodType(_a3_leak, cell)
        cell.update_radius = types.MethodType(_a3_radius, cell)
        cell.update_motion = types.MethodType(_a3_motion, cell)

        # Direct calls would be the usual route by which the inherited parent
        # monolith accidentally runs a second copy.  Only scheduler bridges can
        # invoke the saved bound CPU methods in integrated mode.
        cell.translate = types.MethodType(_forbidden_cpu_entry('translate'), cell)
        cell._replicate_genome = types.MethodType(
            _forbidden_cpu_entry('_replicate_genome'), cell
        )
        cell.update_division = types.MethodType(
            _forbidden_cpu_entry('update_division'), cell
        )
        cell._update_sensorimotor_learning = types.MethodType(
            _forbidden_cpu_entry('_update_sensorimotor_learning'), cell
        )
        cell.export_mobile_element = types.MethodType(
            _forbidden_cpu_entry('export_mobile_element'), cell
        )
        cell._damage_partition_fractions = types.MethodType(
            _a3_damage_partition, cell
        )
        cell.split = types.MethodType(_a3_split, cell)
        cell.sense_environment = types.MethodType(_a3_sense_environment, cell)
        cell.process_corpse_contact = types.MethodType(_a3_corpse_contact, cell)
        cell.environmental_damage = types.MethodType(
            _a3_environmental_damage, cell
        )
    return world


def detach_backend_from_world_a3(world):
    """Explicitly restore the raw CPU world; never called on A3 failure."""
    scheduler = getattr(world, '_soma068a3_scheduler', None)
    if isinstance(scheduler, A3EventScheduler) and scheduler.active:
        raise A3SchedulerProtocolError('cannot detach during an active step')
    for name, attribute in _WORLD_ORIGINAL_ATTR.items():
        original = getattr(world, attribute, None)
        if callable(original):
            setattr(world, name, original)
    field = world.field
    original_diffusion = getattr(field, '_soma068_original_step_diffusion', None)
    if callable(original_diffusion):
        field.step_diffusion = original_diffusion
    for cell in list(world.cells):
        original_profiles = getattr(cell, '_soma068_original_particle_profiles', None)
        if callable(original_profiles):
            cell._particle_ligand_profiles = original_profiles
        for name, attribute in _A2_ORIGINAL_ATTR.items():
            original = getattr(cell, attribute, None)
            if callable(original):
                setattr(cell, name, original)
        for name, attribute in _ORIGINAL_CELL_ATTR.items():
            original = getattr(cell, attribute, None)
            if callable(original):
                setattr(cell, name, original)
    return world


def _default_backend(gpu_config=None):
    """Lazy A3 import: capability errors occur at construction, not import."""
    module = importlib.import_module('SOMA_CELL_0_6_8_gpu_a3')
    backend_type = getattr(module, 'TorchKernelBackendA3', None)
    if backend_type is None:
        raise A3BackendCapabilityError(
            'SOMA_CELL_0_6_8_gpu_a3.TorchKernelBackendA3 is unavailable'
        )
    return validate_backend_a3(backend_type(gpu_config))


def _backend_config_state(backend):
    state_dict = getattr(backend.config, 'state_dict', None)
    if callable(state_dict):
        return copy.deepcopy(state_dict())
    if hasattr(backend.config, '__dict__'):
        return copy.deepcopy(dict(backend.config.__dict__))
    raise A3BackendCapabilityError('backend config is not serializable')


def _aggregate_sidecars_by_cell(world):
    core = _a3_core_module()
    return {
        str(int(cell.cell_id)): core.aggregate_composition_state(cell).state_dict()
        for cell in world.cells
    }


def _restore_aggregate_sidecars(world, payload):
    """Restore by stable cell_id before any A3 hook/backend attachment."""
    core = _a3_core_module()
    payload = dict(payload or {})
    cells = {str(int(cell.cell_id)): cell for cell in world.cells}
    if set(payload) != set(cells):
        raise A3SchedulerProtocolError(
            'aggregate sidecar cell ids differ from restored CPU world'
        )
    for cid, cell in cells.items():
        state = core.AggregateCompositionState.from_state(payload[cid])
        core.apply_aggregate_composition_state(cell, state)
    return world


PORT_STATUS = dict(a2.PORT_STATUS)
PORT_STATUS.update({
    'gene_coded_metabolism_damage_repair': 'A3-hybrid-integrated',
    'damage_segregation_plan': 'A3-actual-split-only',
    'translation_replication_division': 'CPU-authoritative-scheduler-bridged',
    'particle_rng_neural_hgt_death': 'CPU-authoritative',
    'full_gpu_world_step': False,
})


class Hybrid066WorldA3(object):
    """Frozen Formal066World plus one exact-once A3 metabolism authority."""

    def __init__(self, cpu_world, backend=None, scheduler=None, gpu_config=None):
        if not isinstance(cpu_world, a2.s66.Formal066World):
            raise TypeError('Hybrid066WorldA3 requires Formal066World')
        if backend is not None and gpu_config is not None:
            raise ValueError('pass backend or gpu_config, not both')
        self.backend = _default_backend(gpu_config) if backend is None else validate_backend_a3(backend)
        self.scheduler = scheduler if scheduler is not None else A3EventScheduler()
        if not isinstance(self.scheduler, A3EventScheduler):
            raise TypeError('scheduler must be A3EventScheduler')
        self.world = cpu_world
        _attach_backend_to_world_a3(self.world, self.backend, self.scheduler)

    @property
    def receipts(self):
        return self.scheduler.receipts

    @property
    def last_receipt(self):
        return self.scheduler.last_receipt

    @classmethod
    def new(cls, seed=101, initial_cells=3, world_config=None,
            gpu_config=None, backend=None, scheduler=None):
        config = (world_config if isinstance(world_config, a2.s66.Formal066Config)
                  else a2.s66.Formal066Config.from_state(world_config or {}))
        world = a2.s66.Formal066World(
            seed=seed, initial_cells=initial_cells, config=config
        )
        return cls(world, backend=backend, scheduler=scheduler,
                   gpu_config=gpu_config if backend is None else None)

    def step(self, dt):
        _attach_backend_to_world_a3(self.world, self.backend, self.scheduler)
        participants = list(self.world.cells)
        before_ids = [_cell_id(cell) for cell in participants]
        self.scheduler.begin_step(self.world, participants)
        try:
            self.world.step(dt)
            # Reattach daughters after inherited 0.6.6 recast, before exposing
            # them to any later caller.  They participate from the next step.
            _attach_backend_to_world_a3(self.world, self.backend, self.scheduler)
            after_ids = [_cell_id(cell) for cell in self.world.cells]
            before_set, after_set = set(before_ids), set(after_ids)
            self.scheduler.note_attachments(
                [cid for cid in after_ids if cid not in before_set],
                [cid for cid in before_ids if cid not in after_set],
            )
            # Includes derived daughters registered by the split wrapper.  This
            # also closes a daughter selected by Formal066 same-step washout.
            finish_cells = [
                self.scheduler._active['cells'][cid]['object']
                for cid in self.scheduler._active['cell_order']
            ]
            for cell in finish_cells:
                self.scheduler.finish_cell(cell)
            return self.scheduler.finish_step(self.world)
        except BaseException as error:
            if self.scheduler.active:
                self.scheduler.abort_step(error)
            raise

    def summary(self):
        output = self.world.summary()
        output.update({
            'gpu_build': BUILD, 'gpu_schema': SCHEMA_VERSION,
            'gpu_port_status': dict(PORT_STATUS),
            'gpu_backend': self.backend.stats(),
            'gpu_full_world_step': False,
            'a3_scheduler_receipts': len(self.scheduler.receipts),
            'a3_last_step_receipt': self.scheduler.last_receipt,
        })
        return output

    def state_dict(self):
        return {
            'save_version': SAVE_VERSION, 'build': BUILD,
            'gpu_config': _backend_config_state(self.backend),
            'scheduler': self.scheduler.state_dict(),
            'aggregate_sidecars': _aggregate_sidecars_by_cell(self.world),
            'cpu_world': self.world.state_dict(),
        }

    @classmethod
    def from_state(cls, state, backend=None, backend_factory=None):
        world = a2.s66.Formal066World.from_state(state['cpu_world'])
        if 'aggregate_sidecars' not in state:
            raise A3SchedulerProtocolError(
                'A3 save is missing aggregate composition sidecars'
            )
        _restore_aggregate_sidecars(world, state['aggregate_sidecars'])
        config_state = copy.deepcopy(state.get('gpu_config', {}))
        if backend is None:
            backend = (backend_factory(config_state) if backend_factory is not None
                       else _default_backend(config_state))
        scheduler = A3EventScheduler.from_state(state.get('scheduler', {}))
        return cls(world, backend=backend, scheduler=scheduler)

    def clone(self, backend_factory=None):
        state = self.state_dict()
        if backend_factory is None:
            backend_type = type(self.backend)
            backend_factory = lambda config: backend_type(config)
        return type(self).from_state(state, backend_factory=backend_factory)

    def save(self, path):
        temporary = str(path) + '.tmp'
        with open(temporary, 'wb') as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, backend=None, backend_factory=None):
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        return cls.from_state(state, backend=backend,
                              backend_factory=backend_factory)

    def detach_to_raw_cpu(self):
        return detach_backend_from_world_a3(self.world)


def _self_test():
    class DummyWorld(object):
        def __init__(self):
            self.age = 0.0

    class DummyCell(object):
        def __init__(self, cell_id):
            self.cell_id = cell_id
            self.alive = True

    world, cell = DummyWorld(), DummyCell(7)
    scheduler = A3EventScheduler(max_receipts=4)
    assert scheduler.begin_step(world, [cell]) == 0
    for event in WORLD_EVENT_ORDER:
        scheduler.claim_world(event)
    for event in CELL_EVENT_ORDER:
        scheduler.claim(cell, event)
    scheduler.finish_cell(cell)
    receipt = scheduler.finish_step(world)
    assert receipt['status'] == 'complete'
    assert [entry['event'] for entry in receipt['cells'][0]['events']] == list(CELL_EVENT_ORDER)

    # Duplicate and predecessor failures must happen before a second claim.
    scheduler.begin_step(world, [cell])
    scheduler.claim_world('chemostat')
    scheduler.claim(cell, 'surface_exchange')
    try:
        scheduler.claim(cell, 'surface_exchange')
    except A3DuplicateEventError:
        pass
    else:
        raise AssertionError('duplicate claim did not fail')
    scheduler.abort_step(RuntimeError('expected duplicate assay abort'))

    scheduler.begin_step(world, [cell])
    try:
        scheduler.claim(cell, CELL_EVENT_ORDER[5])
    except A3EventOrderError:
        pass
    else:
        raise AssertionError('missing predecessor did not fail')
    scheduler.abort_step(RuntimeError('expected order assay abort'))

    restored = A3EventScheduler.from_state(scheduler.state_dict())
    assert restored.current_step_id is None
    assert len(restored.receipts) == 3
    return True


if __name__ == '__main__':
    _self_test()
    print('SOMA-CELL 0.6.8-GPU A3 scheduler self-test: PASS')
