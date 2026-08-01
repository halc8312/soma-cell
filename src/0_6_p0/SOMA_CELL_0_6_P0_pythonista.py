# coding: utf-8
"""
SOMA-CELL 0.6-P0 — Chemical Body Contract Layer
化学身体API・保存的予算・物理エフェクタ・死組織返却

This is a deliberately pre-neural milestone.  SOMA-CELL 0.5 remains the
frozen chemical body.  P0 adds a fail-closed boundary through which future
material neural tissue may read real chemistry, request finite ATP/material,
ask existing physical effectors to act, and return failed tissue without
silently deleting matter.

No recurrent neuron, learning rule, reward, causal audit, or inherited neural
weight is introduced here.  With no attachment, the simulated 0.5 trajectory
must remain exactly unchanged.  The new classes are infrastructure for P1.
"""
from __future__ import division

import csv
import gc
import hashlib
import json
import math
import os
import pickle
import sys
import time
from types import MappingProxyType

import numpy as np

try:
    import SOMA_CELL_0_5_pythonista as s5
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _BASELINE = os.path.abspath(os.path.join(_HERE, '..', 'baseline'))
    if _BASELINE not in sys.path:
        sys.path.insert(0, _BASELINE)
    import SOMA_CELL_0_5_pythonista as s5

# Existing physics.  P0 does not invent a second body model.
s4 = s5.s4

BUILD = 'SOMA-CELL 0.6-P0.0'
BUILD_LONG = 'SOMA-CELL 0.6-P0.0 Chemical Body Contract Layer'
SAVE_VERSION = 60
PORT_SCHEMA_VERSION = '0.6-P0.1'
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p0.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p0_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p0_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p0_sessions.csv')

SIM_HZ = s5.SIM_HZ
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0

# Escrow categories.  ATP is energy and is excluded from material mass, as in
# the frozen 0.5 body.  The other three are actual matter moved from body pools.
BUDGET_ATP = 0
BUDGET_PROTEIN = 1
BUDGET_MEMBRANE = 2
BUDGET_SIGNAL = 3
BUDGET_NAMES = ('atp', 'protein', 'membrane', 'signal')
BUDGET_COUNT = len(BUDGET_NAMES)

# Tissue matter classes held after budgeted substrate has been assembled.
TISSUE_FUNCTIONAL_PROTEIN = 0
TISSUE_MEMBRANE = 1
TISSUE_SIGNAL = 2
TISSUE_DAMAGED_PROTEIN = 3
TISSUE_AGGREGATE = 4
TISSUE_TOXIC = 5
TISSUE_MATERIAL_NAMES = (
    'functional_protein', 'membrane', 'signal',
    'damaged_protein', 'aggregate', 'toxic',
)
TISSUE_MATERIAL_COUNT = len(TISSUE_MATERIAL_NAMES)

# Raw precursor compositions.  Returning an unused grant reverses these
# mappings exactly, apart from floating-point roundoff.
PROTEIN_FUEL_FRACTION = 0.68
PROTEIN_MINERAL_FRACTION = 0.32
SIGNAL_FUEL_FRACTION = 0.56
SIGNAL_MINERAL_FRACTION = 0.44

# Escrowed feedstock is not yet a neural structure.  Converting it into
# explicitly tracked tissue matter requires finite ATP even in this pre-neural
# milestone.  P1 will additionally require gene-derived assembly machinery.
ASSEMBLY_ATP_PER_PROTEIN = 0.44
ASSEMBLY_ATP_PER_MEMBRANE = 0.18
ASSEMBLY_ATP_PER_SIGNAL = 0.24

# A reserved negative fingerprint is used only to retain material identity in
# the existing damaged-protein dictionary.  It is not a gene and is never
# translated or inherited as information.
NEURAL_DEBRIS_FINGERPRINT = -600600

FORBIDDEN_EFFECTOR_KEYS = frozenset((
    'position', 'pos', 'velocity', 'vel', 'membrane', 'membrane_mass',
    'atp', 'pools', 'dna', 'genome', 'genomes', 'sequence', 'cell_id',
))
ALLOWED_EFFECTOR_KEYS = frozenset((
    'motor', 'transporter_polarity', 'repair_polarity', 'quiescence',
))

clamp = s5.clamp
wrapped_delta = s5.wrapped_delta
finite_array = s5.finite_array
_atomic_pickle = s5._atomic_pickle
_memory_peak_mb_estimate = s5._memory_peak_mb_estimate


def _clean_nonnegative(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return value


def _strict_nonnegative(value, label):
    try:
        value = float(value)
    except Exception as exc:
        raise ValueError('{} must be a finite non-negative number'.format(label)) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError('{} must be a finite non-negative number'.format(label))
    return value


def _strict_dt(value, label='dt'):
    try:
        value = float(value)
    except Exception as exc:
        raise ValueError('{} must be finite and positive'.format(label)) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError('{} must be finite and positive'.format(label))
    return clamp(value, 1e-6, 0.10)


def _readonly_array(value, dtype=float):
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _readonly_mapping(mapping):
    """Recursively freeze mappings while copying mutable arrays.

    ``MappingProxyType`` is available in Pythonista's Python 3 runtime.  The
    sensor contract therefore fails closed even if future tissue code tries to
    mutate a nested sensor frame.  Scalar tuples/strings are already immutable;
    arrays are copied and marked read-only.
    """
    frozen = {}
    for key, value in dict(mapping).items():
        if isinstance(value, dict):
            frozen[key] = _readonly_mapping(value)
        elif isinstance(value, np.ndarray):
            frozen[key] = _readonly_array(value, dtype=value.dtype)
        elif isinstance(value, list):
            frozen[key] = tuple(value)
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


def _normalised_vector(value, limit=1.0):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (2,) or not finite_array(array):
        raise ValueError('effector vector must be a finite two-vector')
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return np.zeros(2, dtype=float), 0.0
    magnitude = min(float(limit), norm)
    return array / norm * magnitude, magnitude


def _state_digest(value):
    """Deterministic diagnostics hash; no simulation state is consumed."""
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


class P0Config(s5.EcologyConfig):
    """0.5 switches plus finite limits for the future neural body port."""

    def __init__(
        self,
        neural_body_port=True,
        neural_strict_effectors=True,
        neural_atp_reserve=0.024,
        neural_fuel_reserve=0.040,
        neural_mineral_reserve=0.040,
        neural_membrane_reserve=0.010,
        neural_atp_rate=0.060,
        neural_protein_rate=0.030,
        neural_membrane_rate=0.018,
        neural_signal_rate=0.016,
        neural_motor_gain_scale=0.80,
        neural_effector_cost_scale=1.0,
        **kwargs
    ):
        super(P0Config, self).__init__(**kwargs)
        self.neural_body_port = bool(neural_body_port)
        self.neural_strict_effectors = bool(neural_strict_effectors)
        self.neural_atp_reserve = float(neural_atp_reserve)
        self.neural_fuel_reserve = float(neural_fuel_reserve)
        self.neural_mineral_reserve = float(neural_mineral_reserve)
        self.neural_membrane_reserve = float(neural_membrane_reserve)
        self.neural_atp_rate = float(neural_atp_rate)
        self.neural_protein_rate = float(neural_protein_rate)
        self.neural_membrane_rate = float(neural_membrane_rate)
        self.neural_signal_rate = float(neural_signal_rate)
        self.neural_motor_gain_scale = float(neural_motor_gain_scale)
        self.neural_effector_cost_scale = float(neural_effector_cost_scale)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))

    def baseline_config(self):
        allowed = set(s5.EcologyConfig().__dict__.keys())
        return s5.EcologyConfig(**{
            key: value for key, value in self.__dict__.items() if key in allowed
        })


class BudgetLedger(object):
    """Accounting only; it is neither reward nor a source of matter."""

    def __init__(
        self,
        requested=None,
        granted=None,
        returned=None,
        spent=None,
        denied=None,
        allocation_calls=0,
        return_calls=0,
        spend_calls=0,
        maximum_material_residual=0.0,
    ):
        self.requested = np.zeros(BUDGET_COUNT, dtype=float) if requested is None else np.asarray(requested, dtype=float).copy()
        self.granted = np.zeros(BUDGET_COUNT, dtype=float) if granted is None else np.asarray(granted, dtype=float).copy()
        self.returned = np.zeros(BUDGET_COUNT, dtype=float) if returned is None else np.asarray(returned, dtype=float).copy()
        self.spent = np.zeros(BUDGET_COUNT, dtype=float) if spent is None else np.asarray(spent, dtype=float).copy()
        self.denied = np.zeros(BUDGET_COUNT, dtype=float) if denied is None else np.asarray(denied, dtype=float).copy()
        self.allocation_calls = int(allocation_calls)
        self.return_calls = int(return_calls)
        self.spend_calls = int(spend_calls)
        self.maximum_material_residual = float(maximum_material_residual)

    def note_allocation(self, requested, granted):
        requested = np.asarray(requested, dtype=float)
        granted = np.asarray(granted, dtype=float)
        self.requested += requested
        self.granted += granted
        self.denied += np.maximum(0.0, requested - granted)
        self.allocation_calls += 1

    def note_return(self, returned):
        self.returned += np.asarray(returned, dtype=float)
        self.return_calls += 1

    def note_spend(self, spent):
        self.spent += np.asarray(spent, dtype=float)
        self.spend_calls += 1

    def finite(self):
        arrays = (self.requested, self.granted, self.returned, self.spent, self.denied)
        return bool(all(finite_array(array) for array in arrays) and np.isfinite(self.maximum_material_residual))

    def state_dict(self):
        return {
            'requested': self.requested.copy(),
            'granted': self.granted.copy(),
            'returned': self.returned.copy(),
            'spent': self.spent.copy(),
            'denied': self.denied.copy(),
            'allocation_calls': self.allocation_calls,
            'return_calls': self.return_calls,
            'spend_calls': self.spend_calls,
            'maximum_material_residual': self.maximum_material_residual,
        }

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class NeuralAttachmentState(object):
    """Material escrow and connection state, intentionally without cognition."""

    def __init__(
        self,
        tissue_id,
        kind='null',
        attached=True,
        active=True,
        created_age=0.0,
        stores=None,
        tissue_material=None,
        cumulative_effects=0,
        cumulative_rejections=0,
        last_effect=None,
        metadata=None,
    ):
        self.tissue_id = str(tissue_id)
        self.kind = str(kind)
        self.attached = bool(attached)
        self.active = bool(active)
        self.created_age = float(created_age)
        self.stores = np.zeros(BUDGET_COUNT, dtype=float) if stores is None else np.asarray(stores, dtype=float).copy()
        self.tissue_material = np.zeros(TISSUE_MATERIAL_COUNT, dtype=float) if tissue_material is None else np.asarray(tissue_material, dtype=float).copy()
        self.cumulative_effects = int(cumulative_effects)
        self.cumulative_rejections = int(cumulative_rejections)
        self.last_effect = {} if last_effect is None else dict(last_effect)
        self.metadata = {} if metadata is None else dict(metadata)

    def material_mass(self):
        return float(
            self.stores[BUDGET_PROTEIN]
            + self.stores[BUDGET_MEMBRANE]
            + self.stores[BUDGET_SIGNAL]
            + np.sum(self.tissue_material)
        )

    def atp(self):
        return float(self.stores[BUDGET_ATP])

    def osmolyte(self):
        return float(
            self.stores[BUDGET_ATP]
            + 0.12 * self.stores[BUDGET_PROTEIN]
            + 0.25 * self.stores[BUDGET_MEMBRANE]
            + 0.35 * self.stores[BUDGET_SIGNAL]
            + 0.10 * self.tissue_material[TISSUE_FUNCTIONAL_PROTEIN]
            + 0.24 * self.tissue_material[TISSUE_SIGNAL]
            + 0.08 * self.tissue_material[TISSUE_DAMAGED_PROTEIN]
            + 0.04 * self.tissue_material[TISSUE_AGGREGATE]
            + 0.75 * self.tissue_material[TISSUE_TOXIC]
        )

    def finite(self):
        return bool(
            finite_array(self.stores)
            and finite_array(self.tissue_material)
            and np.all(self.stores >= -1e-12)
            and np.all(self.tissue_material >= -1e-12)
            and np.isfinite(self.created_age)
        )

    def state_dict(self):
        return {
            'tissue_id': self.tissue_id,
            'kind': self.kind,
            'attached': self.attached,
            'active': self.active,
            'created_age': self.created_age,
            'stores': self.stores.copy(),
            'tissue_material': self.tissue_material.copy(),
            'cumulative_effects': self.cumulative_effects,
            'cumulative_rejections': self.cumulative_rejections,
            'last_effect': dict(self.last_effect),
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class NullNeuralTissue(object):
    """A zero-compute contract probe.  It requests and changes nothing."""

    kind = 'null'

    def __init__(self, tissue_id='null'):
        self.tissue_id = str(tissue_id)

    def step(self, sensors, granted_budget, dt):
        del sensors, granted_budget, dt
        return {}

    def state_dict(self):
        return {'tissue_id': self.tissue_id, 'kind': self.kind}


class ChemicalBodyPort(object):
    """The sole supported boundary between the 0.5 body and future tissue."""

    def __init__(self, world, cell):
        self.world = world
        self.cell = cell

    def _require_enabled(self):
        if not getattr(self.world.config, 'neural_body_port', True):
            raise RuntimeError('chemical body port is disabled')
        if not self.cell.alive:
            raise RuntimeError('host cell is not alive')

    def attachment(self, tissue_id, create=False, kind='null'):
        tissue_id = str(tissue_id)
        state = self.cell.neural_attachments.get(tissue_id)
        if state is None and create:
            state = NeuralAttachmentState(
                tissue_id=tissue_id, kind=kind, created_age=self.world.age,
            )
            self.cell.neural_attachments[tissue_id] = state
            self.cell.neural_attachment_events += 1
        return state

    def attach(self, tissue_id, kind='null'):
        self._require_enabled()
        state = self.attachment(tissue_id, create=True, kind=kind)
        state.attached = True
        state.active = True
        return state

    def raw_sensor_fluxes(self, tissue_id=None):
        """Return copies of physical chemistry only, without RNG or mutation."""
        self._require_enabled()
        cell = self.cell
        if tissue_id is not None and self.attachment(tissue_id) is None:
            raise KeyError('unknown tissue_id: {}'.format(tissue_id))

        # These helpers are deterministic and allocate fresh arrays.  They do
        # not update receptor adaptation, learning traces, or the RNG.
        particle_profiles = cell._particle_ligand_profiles(self.world.field)
        ligand_profiles = cell._all_ligand_profiles(self.world)
        snapshot = {
            'schema': PORT_SCHEMA_VERSION,
            # Routing metadata is deliberately minimal.  Absolute world
            # coordinates and a global clock are not exposed as sensory facts.
            'cell_id': int(cell.cell_id),
            'velocity': _readonly_array(cell.vel),
            'membrane': {
                'material': _readonly_array(cell.membrane),
                'closure': _readonly_array(cell.closure_array()),
                'oxidation': _readonly_array(cell.membrane_oxidation),
                'damage_trace': _readonly_array(cell.damage_trace),
                'transporters': _readonly_array(cell.transporters),
                'tension': float(cell.tension()),
                'radius': float(cell.radius),
                'last_leak': float(cell.last_leak),
            },
            'external': {
                'particle_profiles': _readonly_array(particle_profiles),
                'ligand_profiles': _readonly_array(ligand_profiles),
                'stress_profile': _readonly_array(self.world.stress_profile(cell)),
                'corpse_signal': float(cell.last_corpse_signal),
                'edna_signal': float(cell.last_edna_signal),
                'necrotoxin_signal': float(cell.last_necrotoxin_signal),
            },
            'flux': {
                'last_uptake_by_ligand': _readonly_array(cell.last_uptake_by_ligand),
                'last_surface_flux': _readonly_array(cell.surface_flux),
                'last_export': float(cell.last_export),
                'last_dna_uptake_mass': float(cell.last_dna_uptake_mass),
                'last_necrophagy_mass': float(cell.last_necrophagy_mass),
            },
            'internal': {
                'pools': _readonly_array(cell.pools),
                'atp': float(cell.pools[s5.POOL_ATP]),
                'fuel': float(cell.pools[s5.POOL_FUEL]),
                'mineral': float(cell.pools[s5.POOL_MINERAL]),
                'waste': float(cell.pools[s5.POOL_WASTE]),
                'reactive': float(cell.pools[s5.POOL_REACTIVE]),
                'damaged_protein': float(cell.pools[s5.POOL_DAMAGED_PROTEIN]),
                'aggregate': float(cell.pools[s5.POOL_AGGREGATE]),
                'genome_lesion': float(cell.mean_genome_lesion()),
                'proteostasis': float(cell.proteostasis_factor()),
                'reaction_loop': float(cell.reaction_loop_strength()),
                # No pre-aggregated reward/margin is exposed.  Future tissue
                # must work with the underlying physical bottlenecks.
                'closure_mean': float(cell.closure()),
                'retention': float(math.exp(-3.0 * cell.last_leak)),
            },
        }
        return _readonly_mapping(snapshot)

    def allocate_budget(self, tissue_id, requests, dt):
        """Move finite body resources into tissue escrow without creating any."""
        self._require_enabled()
        state = self.attachment(tissue_id)
        if state is None or not state.attached or not state.active:
            raise KeyError('active attachment required: {}'.format(tissue_id))
        dt = _strict_dt(dt, 'budget.dt')
        if not isinstance(requests, dict):
            raise TypeError('requests must be a mapping')
        unknown = set(requests.keys()) - set(BUDGET_NAMES)
        if unknown:
            raise ValueError('unknown budget categories: {}'.format(sorted(unknown)))
        requested = np.asarray([
            _strict_nonnegative(requests.get(name, 0.0), 'budget.' + name)
            for name in BUDGET_NAMES
        ], dtype=float)
        config = self.world.config
        capped = np.minimum(requested, np.asarray([
            max(0.0, config.neural_atp_rate) * dt,
            max(0.0, config.neural_protein_rate) * dt,
            max(0.0, config.neural_membrane_rate) * dt,
            max(0.0, config.neural_signal_rate) * dt,
        ], dtype=float))

        before_material = self.world.total_material()
        grant = np.zeros(BUDGET_COUNT, dtype=float)

        # ATP is energy escrow, not counted as material in the 0.5 ledger.
        available_atp = max(
            0.0,
            float(self.cell.pools[s5.POOL_ATP]) - max(0.0, config.neural_atp_reserve),
        )
        grant[BUDGET_ATP] = min(capped[BUDGET_ATP], available_atp)
        self.cell.pools[s5.POOL_ATP] -= grant[BUDGET_ATP]

        # Protein substrate and finite signal precursor share fuel/mineral.
        protein = capped[BUDGET_PROTEIN]
        signal = capped[BUDGET_SIGNAL]
        fuel_need = (
            PROTEIN_FUEL_FRACTION * protein
            + SIGNAL_FUEL_FRACTION * signal
        )
        mineral_need = (
            PROTEIN_MINERAL_FRACTION * protein
            + SIGNAL_MINERAL_FRACTION * signal
        )
        fuel_available = max(
            0.0,
            float(self.cell.pools[s5.POOL_FUEL]) - max(0.0, config.neural_fuel_reserve),
        )
        mineral_available = max(
            0.0,
            float(self.cell.pools[s5.POOL_MINERAL]) - max(0.0, config.neural_mineral_reserve),
        )
        scale = 1.0
        if fuel_need > 0.0:
            scale = min(scale, fuel_available / fuel_need)
        if mineral_need > 0.0:
            scale = min(scale, mineral_available / mineral_need)
        scale = clamp(scale, 0.0, 1.0)
        protein *= scale
        signal *= scale
        grant[BUDGET_PROTEIN] = protein
        grant[BUDGET_SIGNAL] = signal
        self.cell.pools[s5.POOL_FUEL] -= (
            PROTEIN_FUEL_FRACTION * protein
            + SIGNAL_FUEL_FRACTION * signal
        )
        self.cell.pools[s5.POOL_MINERAL] -= (
            PROTEIN_MINERAL_FRACTION * protein
            + SIGNAL_MINERAL_FRACTION * signal
        )

        membrane_available = max(
            0.0,
            float(self.cell.pools[s5.POOL_MEM_PRECURSOR])
            - max(0.0, config.neural_membrane_reserve),
        )
        grant[BUDGET_MEMBRANE] = min(capped[BUDGET_MEMBRANE], membrane_available)
        self.cell.pools[s5.POOL_MEM_PRECURSOR] -= grant[BUDGET_MEMBRANE]

        state.stores += grant
        self.cell.neural_budget_ledger.note_allocation(requested, grant)
        self.world.p0_budget_calls += 1
        self.world.p0_allocated_atp += float(grant[BUDGET_ATP])
        self.world.p0_allocated_material += float(np.sum(grant[1:]))
        after_material = self.world.total_material()
        residual = after_material - before_material
        self.cell.neural_budget_ledger.maximum_material_residual = max(
            self.cell.neural_budget_ledger.maximum_material_residual,
            abs(float(residual)),
        )
        return {name: float(grant[index]) for index, name in enumerate(BUDGET_NAMES)}

    def return_unused_budget(self, tissue_id, amounts=None):
        self._require_enabled()
        state = self.attachment(tissue_id)
        if state is None:
            raise KeyError('unknown tissue_id: {}'.format(tissue_id))
        before_material = self.world.total_material()
        if amounts is None:
            returned = state.stores.copy()
        else:
            if not isinstance(amounts, dict):
                raise TypeError('amounts must be a mapping')
            unknown = set(amounts.keys()) - set(BUDGET_NAMES)
            if unknown:
                raise ValueError('unknown budget categories: {}'.format(sorted(unknown)))
            returned = np.asarray([
                min(
                    state.stores[index],
                    _strict_nonnegative(amounts.get(name, 0.0), 'return.' + name),
                )
                for index, name in enumerate(BUDGET_NAMES)
            ], dtype=float)
        state.stores -= returned
        self.cell.pools[s5.POOL_ATP] += returned[BUDGET_ATP]
        self.cell.pools[s5.POOL_FUEL] += (
            PROTEIN_FUEL_FRACTION * returned[BUDGET_PROTEIN]
            + SIGNAL_FUEL_FRACTION * returned[BUDGET_SIGNAL]
        )
        self.cell.pools[s5.POOL_MINERAL] += (
            PROTEIN_MINERAL_FRACTION * returned[BUDGET_PROTEIN]
            + SIGNAL_MINERAL_FRACTION * returned[BUDGET_SIGNAL]
        )
        self.cell.pools[s5.POOL_MEM_PRECURSOR] += returned[BUDGET_MEMBRANE]
        self.cell.neural_budget_ledger.note_return(returned)
        self.world.p0_budget_return_calls += 1
        self.world.p0_returned_atp += float(returned[BUDGET_ATP])
        self.world.p0_returned_material += float(np.sum(returned[1:]))
        residual = self.world.total_material() - before_material
        self.cell.neural_budget_ledger.maximum_material_residual = max(
            self.cell.neural_budget_ledger.maximum_material_residual,
            abs(float(residual)),
        )
        return {name: float(returned[index]) for index, name in enumerate(BUDGET_NAMES)}

    def commit_material(self, tissue_id, protein=0.0, membrane=0.0, signal=0.0,
                        damaged_fraction=0.0, aggregate_fraction=0.0):
        """Assemble escrowed feedstock into tissue matter with finite ATP.

        P0 still does not assign neural function or genetic identity.  This is
        only a conservative assembly boundary so P1 cannot turn raw substrate
        into a free cell membrane/protein stock.
        """
        self._require_enabled()
        state = self.attachment(tissue_id)
        if state is None or not state.attached or not state.active:
            raise KeyError('active attachment required: {}'.format(tissue_id))
        protein = min(state.stores[BUDGET_PROTEIN], _strict_nonnegative(protein, 'commit.protein'))
        membrane = min(state.stores[BUDGET_MEMBRANE], _strict_nonnegative(membrane, 'commit.membrane'))
        signal = min(state.stores[BUDGET_SIGNAL], _strict_nonnegative(signal, 'commit.signal'))
        damaged_fraction = clamp(_strict_nonnegative(damaged_fraction, 'commit.damaged_fraction'), 0.0, 1.0)
        aggregate_fraction = clamp(_strict_nonnegative(aggregate_fraction, 'commit.aggregate_fraction'), 0.0, 1.0)
        aggregate_fraction = min(aggregate_fraction, 1.0 - damaged_fraction)

        atp_required = (
            ASSEMBLY_ATP_PER_PROTEIN * protein
            + ASSEMBLY_ATP_PER_MEMBRANE * membrane
            + ASSEMBLY_ATP_PER_SIGNAL * signal
        )
        scale = 1.0 if atp_required <= 1e-15 else min(
            1.0, float(state.stores[BUDGET_ATP]) / atp_required
        )
        protein *= scale
        membrane *= scale
        signal *= scale
        atp_spent = atp_required * scale

        before_material = self.world.total_material()
        state.stores[BUDGET_ATP] -= atp_spent
        state.stores[BUDGET_PROTEIN] -= protein
        state.stores[BUDGET_MEMBRANE] -= membrane
        state.stores[BUDGET_SIGNAL] -= signal
        functional_fraction = 1.0 - damaged_fraction - aggregate_fraction
        state.tissue_material[TISSUE_FUNCTIONAL_PROTEIN] += protein * functional_fraction
        state.tissue_material[TISSUE_DAMAGED_PROTEIN] += protein * damaged_fraction
        state.tissue_material[TISSUE_AGGREGATE] += protein * aggregate_fraction
        state.tissue_material[TISSUE_MEMBRANE] += membrane
        state.tissue_material[TISSUE_SIGNAL] += signal

        if atp_spent > 0.0:
            self.world.dissipated_energy += atp_spent
            spent = np.zeros(BUDGET_COUNT, dtype=float)
            spent[BUDGET_ATP] = atp_spent
            self.cell.neural_budget_ledger.note_spend(spent)
            self.world.p0_spent_atp += float(atp_spent)
        residual = self.world.total_material() - before_material
        self.cell.neural_budget_ledger.maximum_material_residual = max(
            self.cell.neural_budget_ledger.maximum_material_residual,
            abs(float(residual)),
        )
        return {
            'functional_protein': protein * functional_fraction,
            'damaged_protein': protein * damaged_fraction,
            'aggregate': protein * aggregate_fraction,
            'membrane': membrane,
            'signal': signal,
            'atp_spent': float(atp_spent),
            'material_residual': float(residual),
        }

    def _spend_energy_and_signal(self, state, atp_requested, signal_requested):
        atp_requested = _clean_nonnegative(atp_requested)
        signal_requested = _clean_nonnegative(signal_requested)
        atp_paid = min(float(state.stores[BUDGET_ATP]), atp_requested)
        signal_paid = min(
            float(state.stores[BUDGET_SIGNAL] + state.tissue_material[TISSUE_SIGNAL]),
            signal_requested,
        )
        state.stores[BUDGET_ATP] -= atp_paid
        self.world.dissipated_energy += atp_paid
        # Use free signal escrow first, then assembled finite signal matter.
        free = min(float(state.stores[BUDGET_SIGNAL]), signal_paid)
        state.stores[BUDGET_SIGNAL] -= free
        assembled = signal_paid - free
        if assembled > 0.0:
            state.tissue_material[TISSUE_SIGNAL] -= assembled
        self.cell.pools[s5.POOL_WASTE] += signal_paid
        spent = np.zeros(BUDGET_COUNT, dtype=float)
        spent[BUDGET_ATP] = atp_paid
        spent[BUDGET_SIGNAL] = signal_paid
        self.cell.neural_budget_ledger.note_spend(spent)
        self.world.p0_spent_atp += float(atp_paid)
        self.world.p0_spent_signal += float(signal_paid)
        return atp_paid, signal_paid

    def apply_effector_fluxes(self, tissue_id, fluxes, dt):
        """Route requests through existing membrane physics; never rewrite pose."""
        self._require_enabled()
        state = self.attachment(tissue_id)
        if state is None or not state.active:
            raise KeyError('active attachment required: {}'.format(tissue_id))
        if not isinstance(fluxes, dict):
            raise TypeError('fluxes must be a mapping')
        keys = set(fluxes.keys())
        forbidden = keys & FORBIDDEN_EFFECTOR_KEYS
        if forbidden:
            state.cumulative_rejections += 1
            self.cell.neural_effector_rejections += 1
            self.world.p0_effector_rejections += 1
            raise ValueError('direct body rewrite is forbidden: {}'.format(sorted(forbidden)))
        unknown = keys - ALLOWED_EFFECTOR_KEYS
        if unknown and getattr(self.world.config, 'neural_strict_effectors', True):
            state.cumulative_rejections += 1
            self.cell.neural_effector_rejections += 1
            self.world.p0_effector_rejections += 1
            raise ValueError('unknown effector requests: {}'.format(sorted(unknown)))
        dt = _strict_dt(dt, 'effector.dt')
        before_position = self.cell.pos.copy()
        before_material = self.world.total_material()
        report = {
            'motor_force': 0.0,
            'transporter_movement': 0.0,
            'repair_polarity_gain': 0.0,
            'quiescence': 0.0,
            'atp_spent': 0.0,
            'signal_spent': 0.0,
        }
        config = self.world.config
        cost_scale = max(0.0, config.neural_effector_cost_scale)

        if 'motor' in fluxes:
            command, magnitude = _normalised_vector(fluxes['motor'])
            activity, geometry = self.cell._effector_activity(
                s4.EFFECT_MOTOR, s4.CONTROL_MOTOR
            )
            if activity > 1e-8 and magnitude > 1e-10:
                atp_request = dt * 0.0065 * activity * magnitude ** 2 * cost_scale
                signal_request = dt * 0.00050 * (0.25 + activity) * magnitude
                atp_paid, signal_paid = self._spend_energy_and_signal(
                    state, atp_request, signal_request
                )
                afford = min(
                    atp_paid / max(atp_request, 1e-12),
                    signal_paid / max(signal_request, 1e-12),
                )
                force = (
                    0.0105 * activity * magnitude * afford
                    * config.neural_motor_gain_scale
                    / max(0.72, self.cell.radius / s5.BASE_RADIUS)
                )
                self.cell.surface_flux += command * force
                self.cell.neural_last_motor_command = command.copy()
                self.cell.neural_last_motor_force = force
                converted = min(
                    self.cell.pools[s5.POOL_WASTE],
                    atp_paid * (0.035 + 0.020 * abs(geometry)),
                )
                self.cell.pools[s5.POOL_WASTE] -= converted
                self.cell.pools[s5.POOL_REACTIVE] += converted
                report['motor_force'] = float(force)
                report['atp_spent'] += float(atp_paid)
                report['signal_spent'] += float(signal_paid)

        if 'transporter_polarity' in fluxes:
            direction, magnitude = _normalised_vector(fluxes['transporter_polarity'])
            activity, geometry = self.cell._effector_activity(
                s4.EFFECT_TRANSPORT_POLARITY, s4.CONTROL_MOTOR
            )
            if activity > 1e-8 and magnitude > 1e-10:
                angles = 2.0 * math.pi * (
                    np.arange(s5.MEMBRANE_SEGMENTS) + 0.5
                ) / s5.MEMBRANE_SEGMENTS
                normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                projection = normals.dot(direction)
                sharpness = 1.3 + 1.2 * abs(geometry)
                front = np.exp(sharpness * projection)
                rear = np.exp(-sharpness * projection)
                front /= float(np.sum(front))
                rear /= float(np.sum(rear))
                blend = clamp(dt * 0.040 * activity * magnitude, 0.0, 0.16)
                proposals = self.cell.transporters.copy()
                movement = 0.0
                for channel in range(s4.CHANNEL_COUNT):
                    total = float(np.sum(self.cell.transporters[:, channel]))
                    if total <= 1e-12:
                        continue
                    target = rear if channel == s4.CHANNEL_WASTE else front
                    proposed = (
                        (1.0 - blend) * self.cell.transporters[:, channel]
                        + blend * total * target
                    )
                    movement += 0.5 * float(np.sum(np.abs(
                        proposed - self.cell.transporters[:, channel]
                    )))
                    proposals[:, channel] = proposed
                atp_request = movement * 0.42 * cost_scale
                signal_request = movement * 0.055
                atp_paid, signal_paid = self._spend_energy_and_signal(
                    state, atp_request, signal_request
                )
                afford = min(
                    atp_paid / max(atp_request, 1e-12),
                    signal_paid / max(signal_request, 1e-12),
                ) if movement > 0.0 else 0.0
                self.cell.transporters += afford * (proposals - self.cell.transporters)
                self.cell.transporters = np.maximum(self.cell.transporters, 0.0)
                report['transporter_movement'] = float(movement * afford)
                report['atp_spent'] += float(atp_paid)
                report['signal_spent'] += float(signal_paid)

        if 'repair_polarity' in fluxes:
            direction, magnitude = _normalised_vector(fluxes['repair_polarity'])
            if magnitude > 1e-10:
                angles = 2.0 * math.pi * (
                    np.arange(s5.MEMBRANE_SEGMENTS) + 0.5
                ) / s5.MEMBRANE_SEGMENTS
                normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                target = np.maximum(0.0, normals.dot(direction))
                if float(np.sum(target)) > 0.0:
                    target /= float(np.sum(target))
                atp_request = dt * 0.0012 * magnitude * cost_scale
                signal_request = dt * 0.00020 * magnitude
                atp_paid, signal_paid = self._spend_energy_and_signal(
                    state, atp_request, signal_request
                )
                afford = min(
                    atp_paid / max(atp_request, 1e-12),
                    signal_paid / max(signal_request, 1e-12),
                )
                blend = clamp(dt * 1.4 * magnitude * afford, 0.0, 0.22)
                self.cell.repair_polarity = (
                    (1.0 - blend) * self.cell.repair_polarity + blend * target
                )
                report['repair_polarity_gain'] = float(blend)
                report['atp_spent'] += float(atp_paid)
                report['signal_spent'] += float(signal_paid)

        if 'quiescence' in fluxes:
            scalar = clamp(
                _strict_nonnegative(fluxes['quiescence'], 'effector.quiescence'),
                0.0, 1.0,
            )
            if scalar > 0.0:
                atp_request = dt * 0.00035 * scalar * cost_scale
                signal_request = dt * 0.00012 * scalar
                atp_paid, signal_paid = self._spend_energy_and_signal(
                    state, atp_request, signal_request
                )
                afford = min(
                    atp_paid / max(atp_request, 1e-12),
                    signal_paid / max(signal_request, 1e-12),
                )
                level = clamp(0.72 * scalar * afford, 0.0, 0.72)
                self.cell.behavioural_quiescence = max(
                    float(self.cell.behavioural_quiescence), level
                )
                report['quiescence'] = float(level)
                report['atp_spent'] += float(atp_paid)
                report['signal_spent'] += float(signal_paid)

        if not np.array_equal(before_position, self.cell.pos):
            raise AssertionError('effector port changed position directly')
        residual = self.world.total_material() - before_material
        self.cell.neural_budget_ledger.maximum_material_residual = max(
            self.cell.neural_budget_ledger.maximum_material_residual,
            abs(float(residual)),
        )
        state.cumulative_effects += 1
        state.last_effect = dict(report)
        self.cell.neural_effector_calls += 1
        self.world.p0_effector_calls += 1
        return report

    def return_dead_tissue(self, tissue_id, reason='tissue-failure', remove=True):
        """Return only matter already held by the attachment; never mint products."""
        state = self.attachment(tissue_id)
        if state is None:
            raise KeyError('unknown tissue_id: {}'.format(tissue_id))
        before_material = self.world.total_material()
        returned_material = state.material_mass()
        returned_atp = state.atp()

        protein = (
            state.stores[BUDGET_PROTEIN]
            + state.tissue_material[TISSUE_FUNCTIONAL_PROTEIN]
            + state.tissue_material[TISSUE_DAMAGED_PROTEIN]
        )
        aggregate = state.tissue_material[TISSUE_AGGREGATE]
        membrane = (
            state.stores[BUDGET_MEMBRANE]
            + state.tissue_material[TISSUE_MEMBRANE]
        )
        signal = (
            state.stores[BUDGET_SIGNAL]
            + state.tissue_material[TISSUE_SIGNAL]
        )
        toxic = state.tissue_material[TISSUE_TOXIC]

        if protein > 0.0:
            self.cell.damaged_proteins[NEURAL_DEBRIS_FINGERPRINT] = (
                self.cell.damaged_proteins.get(NEURAL_DEBRIS_FINGERPRINT, 0.0)
                + protein
            )
        self.cell.pools[s5.POOL_AGGREGATE] += aggregate
        self.cell.pools[s5.POOL_MEM_PRECURSOR] += membrane
        self.cell.pools[s5.POOL_WASTE] += signal
        self.cell.pools[s5.POOL_REACTIVE] += toxic
        self.cell._sync_damage_pool()
        self.world.dissipated_energy += returned_atp

        spent = np.zeros(BUDGET_COUNT, dtype=float)
        spent[BUDGET_ATP] = returned_atp
        if returned_atp > 0.0:
            self.cell.neural_budget_ledger.note_spend(spent)
        state.stores[:] = 0.0
        state.tissue_material[:] = 0.0
        state.active = False
        state.attached = False
        state.metadata['return_reason'] = str(reason)
        state.metadata['returned_age'] = float(self.world.age)
        self.cell.neural_tissue_returns += 1
        self.cell.neural_returned_material += returned_material
        self.cell.neural_returned_atp += returned_atp
        self.world.p0_tissue_returns += 1
        self.world.p0_returned_material += float(returned_material)
        self.world.p0_spent_atp += float(returned_atp)
        self.world.p0_tissue_atp_dissipated += float(returned_atp)
        if remove:
            del self.cell.neural_attachments[str(tissue_id)]

        residual = self.world.total_material() - before_material
        self.cell.neural_budget_ledger.maximum_material_residual = max(
            self.cell.neural_budget_ledger.maximum_material_residual,
            abs(float(residual)),
        )
        return {
            'material': float(returned_material),
            'atp_dissipated': float(returned_atp),
            'residual': float(residual),
        }

    def detach(self, tissue_id, return_unused=True):
        state = self.attachment(tissue_id)
        if state is None:
            return False
        if return_unused and np.any(state.stores > 1e-12):
            self.return_unused_budget(tissue_id)
            state = self.attachment(tissue_id)
        # Committed tissue cannot be treated as unused substrate.  It is
        # conservatively dismantled into the host's degradation pools.
        if state is not None and (state.material_mass() > 1e-12 or state.atp() > 1e-12):
            self.return_dead_tissue(tissue_id, reason='forced-detach')
            return True
        self.cell.neural_attachments.pop(str(tissue_id), None)
        return True


class P0ProtoCell(s5.EcologicalProtoCell):
    """Frozen 0.5 chemistry plus empty, material-accounted attachment slots."""

    def __init__(self, *args, **kwargs):
        super(P0ProtoCell, self).__init__(*args, **kwargs)
        self._init_p0_state()

    def _init_p0_state(self):
        self.neural_attachments = {}
        self.neural_budget_ledger = BudgetLedger()
        self.neural_attachment_events = 0
        self.neural_effector_calls = 0
        self.neural_effector_rejections = 0
        self.neural_tissue_returns = 0
        self.neural_returned_material = 0.0
        self.neural_returned_atp = 0.0
        self.neural_last_motor_command = np.zeros(2, dtype=float)
        self.neural_last_motor_force = 0.0

    def body_port(self, world):
        return ChemicalBodyPort(world, self)

    def neural_material_mass(self):
        return float(sum(state.material_mass() for state in self.neural_attachments.values()))

    def neural_atp_escrow(self):
        return float(sum(state.atp() for state in self.neural_attachments.values()))

    def material_mass(self):
        return float(super(P0ProtoCell, self).material_mass() + self.neural_material_mass())

    def osmolyte(self):
        return float(
            super(P0ProtoCell, self).osmolyte()
            + sum(state.osmolyte() for state in self.neural_attachments.values())
        )

    def return_all_neural_tissue(self, world, reason='host-transition'):
        port = self.body_port(world)
        for tissue_id in list(self.neural_attachments.keys()):
            port.return_dead_tissue(tissue_id, reason=reason, remove=True)

    def split(self, world):
        # P0 contains no hereditary neural tissue.  Any attachment escrow is
        # returned before 0.5 performs its already-validated material split.
        if self.neural_attachments:
            self.return_all_neural_tissue(world, reason='pre-division-return')
        daughters = super(P0ProtoCell, self).split(world)
        if daughters is None:
            return None
        for daughter in daughters:
            daughter.__class__ = P0ProtoCell
            daughter._init_p0_state()
        return daughters

    def state_dict(self):
        state = super(P0ProtoCell, self).state_dict()
        state.update({
            'cell_class': 'P0ProtoCell',
            'p0_neural_attachments': {
                key: value.state_dict() for key, value in self.neural_attachments.items()
            },
            'p0_neural_budget_ledger': self.neural_budget_ledger.state_dict(),
            'p0_neural_attachment_events': self.neural_attachment_events,
            'p0_neural_effector_calls': self.neural_effector_calls,
            'p0_neural_effector_rejections': self.neural_effector_rejections,
            'p0_neural_tissue_returns': self.neural_tissue_returns,
            'p0_neural_returned_material': self.neural_returned_material,
            'p0_neural_returned_atp': self.neural_returned_atp,
            'p0_neural_last_motor_command': self.neural_last_motor_command.copy(),
            'p0_neural_last_motor_force': self.neural_last_motor_force,
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = s5.EcologicalProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        cell._init_p0_state()
        # The frozen 0.4/0.5 loader normalises empty controller maps by adding
        # zero-valued entries for every current receptor.  That is physically
        # harmless and gives the same future, but it makes an immediate
        # save/load round trip differ structurally when a pristine cell has not
        # yet executed its first sensor step.  P0's contract promises exact
        # checkpoint state as well as exact continuation, so restore the maps
        # exactly as serialized.  The next ordinary body step will perform the
        # same baseline normalisation on both the original and restored cell.
        for mapping_name in (
            'receptor_baseline', 'controller_phosphorylation',
            'controller_eligibility', 'controller_noise',
            'sensor_activation_by_gene', 'last_gene_perturbation',
        ):
            if mapping_name in state:
                setattr(cell, mapping_name, {
                    int(key): float(value)
                    for key, value in state.get(mapping_name, [])
                })
        cell.neural_attachments = {
            str(key): NeuralAttachmentState.from_state(value)
            for key, value in state.get('p0_neural_attachments', {}).items()
        }
        if 'p0_neural_budget_ledger' in state:
            cell.neural_budget_ledger = BudgetLedger.from_state(
                state['p0_neural_budget_ledger']
            )
        for name in (
            'neural_attachment_events', 'neural_effector_calls',
            'neural_effector_rejections', 'neural_tissue_returns',
        ):
            setattr(cell, name, int(state.get('p0_' + name, getattr(cell, name))))
        for name in ('neural_returned_material', 'neural_returned_atp', 'neural_last_motor_force'):
            setattr(cell, name, float(state.get('p0_' + name, getattr(cell, name))))
        cell.neural_last_motor_command = np.asarray(
            state.get('p0_neural_last_motor_command', np.zeros(2)), dtype=float
        ).copy()
        return cell


class P0World(s5.EcologicalWorld):
    """0.5-compatible world exposing chemical body ports, but no neural logic."""

    def __init__(self, seed=101, initial_cells=3, config=None):
        config = config if config is not None else P0Config()
        if not isinstance(config, P0Config):
            config = P0Config(**config.state_dict())
        super(P0World, self).__init__(
            seed=seed, initial_cells=initial_cells, config=config,
        )
        for cell in self.cells:
            cell.__class__ = P0ProtoCell
            cell._init_p0_state()
        self.config = config
        self._init_p0_world_state()
        # Empty attachment slots add exactly zero material.
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def _init_p0_world_state(self):
        """World-level counters survive host death and are telemetry only."""
        self.p0_port_calls = 0
        self.p0_budget_calls = 0
        self.p0_budget_return_calls = 0
        self.p0_effector_calls = 0
        self.p0_effector_rejections = 0
        self.p0_tissue_returns = 0
        self.p0_allocated_atp = 0.0
        self.p0_allocated_material = 0.0
        self.p0_returned_atp = 0.0
        self.p0_returned_material = 0.0
        self.p0_spent_atp = 0.0
        self.p0_spent_signal = 0.0
        self.p0_tissue_atp_dissipated = 0.0
        self.p0_baseline_lockstep_marker = True

    def port_for(self, cell_id):
        for cell in self.cells:
            if int(cell.cell_id) == int(cell_id):
                self.p0_port_calls += 1
                return cell.body_port(self)
        raise KeyError('cell_id not found: {}'.format(cell_id))

    def attach_null_tissue(self, cell_id, tissue_id='null'):
        port = self.port_for(cell_id)
        port.attach(tissue_id, kind='null')
        return NullNeuralTissue(tissue_id)

    def _release_dead_cell(self, cell):
        if isinstance(cell, P0ProtoCell) and cell.neural_attachments:
            cell.return_all_neural_tissue(self, reason='host-death')
        return super(P0World, self)._release_dead_cell(cell)

    def step(self, dt):
        # Deliberately no neural callbacks in P0.  This single delegation is the
        # lockstep guarantee when attachments are absent.
        super(P0World, self).step(dt)

    def finite(self):
        if not super(P0World, self).finite():
            return False
        for cell in self.cells:
            if not isinstance(cell, P0ProtoCell):
                return False
            if not cell.neural_budget_ledger.finite():
                return False
            if not finite_array(cell.neural_last_motor_command):
                return False
            if not np.isfinite(cell.neural_last_motor_force):
                return False
            if any(not state.finite() for state in cell.neural_attachments.values()):
                return False
        return True

    def summary(self):
        summary = super(P0World, self).summary()
        alive = self.living_cells()
        attachments = [
            state for cell in alive for state in cell.neural_attachments.values()
        ]
        ledgers = [cell.neural_budget_ledger for cell in alive]
        summary.update({
            'build': BUILD,
            'port_schema': PORT_SCHEMA_VERSION,
            'neural_attachments': len(attachments),
            'neural_material_escrow': float(sum(s.material_mass() for s in attachments)),
            'neural_atp_escrow': float(sum(s.atp() for s in attachments)),
            'neural_budget_requested': float(sum(np.sum(l.requested) for l in ledgers)),
            'neural_budget_granted': float(sum(np.sum(l.granted) for l in ledgers)),
            'neural_budget_returned': float(sum(np.sum(l.returned) for l in ledgers)),
            'neural_budget_spent': float(sum(np.sum(l.spent) for l in ledgers)),
            'neural_budget_calls': int(self.p0_budget_calls),
            'neural_budget_return_calls': int(self.p0_budget_return_calls),
            'neural_effector_calls': int(self.p0_effector_calls),
            'neural_effector_rejections': int(self.p0_effector_rejections),
            'neural_tissue_returns': int(self.p0_tissue_returns),
            'neural_allocated_atp': float(self.p0_allocated_atp),
            'neural_allocated_material': float(self.p0_allocated_material),
            'neural_returned_atp': float(self.p0_returned_atp),
            'neural_returned_material': float(self.p0_returned_material),
            'neural_spent_atp': float(self.p0_spent_atp),
            'neural_spent_signal': float(self.p0_spent_signal),
            'neural_tissue_atp_dissipated': float(self.p0_tissue_atp_dissipated),
            'neural_port_calls': int(self.p0_port_calls),
        })
        return summary

    def state_dict(self):
        state = super(P0World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'p0_port_calls': self.p0_port_calls,
            'p0_budget_calls': self.p0_budget_calls,
            'p0_budget_return_calls': self.p0_budget_return_calls,
            'p0_effector_calls': self.p0_effector_calls,
            'p0_effector_rejections': self.p0_effector_rejections,
            'p0_tissue_returns': self.p0_tissue_returns,
            'p0_allocated_atp': self.p0_allocated_atp,
            'p0_allocated_material': self.p0_allocated_material,
            'p0_returned_atp': self.p0_returned_atp,
            'p0_returned_material': self.p0_returned_material,
            'p0_spent_atp': self.p0_spent_atp,
            'p0_spent_signal': self.p0_spent_signal,
            'p0_tissue_atp_dissipated': self.p0_tissue_atp_dissipated,
            'p0_baseline_lockstep_marker': self.p0_baseline_lockstep_marker,
        })
        return state

    @classmethod
    def from_state(cls, state):
        # Let the frozen 0.5 loader reconstruct all existing corpse/eDNA/body
        # state, then add P0-only attachment state without consuming RNG.
        base_state = dict(state)
        base_state['save_version'] = s5.SAVE_VERSION
        base_state['build'] = s5.BUILD
        base_state['config'] = {
            key: value for key, value in dict(state.get('config', {})).items()
            if key in s5.EcologyConfig().__dict__
        }
        world = s5.EcologicalWorld.from_state(base_state)
        world.__class__ = cls
        world._init_p0_world_state()
        world.config = P0Config.from_state(state.get('config', {}))
        world.cells = [P0ProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        world.p0_port_calls = int(state.get('p0_port_calls', 0))
        for name in (
            'p0_budget_calls', 'p0_budget_return_calls', 'p0_effector_calls',
            'p0_effector_rejections', 'p0_tissue_returns',
        ):
            setattr(world, name, int(state.get(name, 0)))
        for name in (
            'p0_allocated_atp', 'p0_allocated_material',
            'p0_returned_atp', 'p0_returned_material',
            'p0_spent_atp', 'p0_spent_signal', 'p0_tissue_atp_dissipated',
        ):
            setattr(world, name, float(state.get(name, 0.0)))
        world.p0_baseline_lockstep_marker = bool(
            state.get('p0_baseline_lockstep_marker', True)
        )
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return P0World.from_state(self.state_dict())


def baseline_projection(world_or_state):
    """Strip P0-only metadata for exact comparison against frozen 0.5."""
    state = world_or_state.state_dict() if hasattr(world_or_state, 'state_dict') else world_or_state

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text.startswith('p0_') or key_text.startswith('neural_'):
                    continue
                if key_text == 'cell_class' and item == 'P0ProtoCell':
                    result[key] = 'EcologicalProtoCell'
                elif key_text == 'build':
                    result[key] = s5.BUILD
                elif key_text == 'save_version':
                    result[key] = s5.SAVE_VERSION
                elif key_text == 'config':
                    allowed = s5.EcologyConfig().__dict__
                    result[key] = clean({k: v for k, v in item.items() if k in allowed})
                else:
                    result[key] = clean(item)
            return result
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(item) for item in value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    return clean(state)


def run_headless_trial(seed=101, seconds=240.0, initial_cells=3, config=None):
    world = P0World(
        seed=seed, initial_cells=initial_cells,
        config=config if config is not None else P0Config(),
    )
    dt = 1.0 / SIM_HZ
    margin_integral = 0.0
    living_time = 0.0
    for _ in range(int(round(float(seconds) * SIM_HZ))):
        world.step(dt)
        summary = world.summary()
        if summary['cells'] > 0:
            margin_integral += summary['mean_autopoietic_margin'] * dt
            living_time += dt
    result = world.summary()
    result['seed'] = int(seed)
    result['seconds'] = float(seconds)
    result['time_mean_margin'] = margin_integral / max(living_time, 1e-12)
    return result


LOG_FIELDS = (
    'session_id', 'reason', 'wall_time', 'sim_age', 'fps', 'sim_rate',
    'cells', 'corpses', 'edna_fragments', 'divisions', 'deaths',
    'mean_autopoietic_margin', 'mean_atp', 'matter_residual',
    'neural_attachments', 'neural_material_escrow', 'neural_atp_escrow',
    'neural_budget_requested', 'neural_budget_granted',
    'neural_budget_returned', 'neural_budget_spent',
    'neural_effector_calls', 'neural_effector_rejections',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), world.seed)
        self.last_log_age = -1e9
        self.status = 'OK'

    def log(self, world, fps=0.0, sim_rate=0.0, reason='interval', force=False):
        if not force and world.age - self.last_log_age < LOG_INTERVAL:
            return
        summary = world.summary()
        row = {name: summary.get(name, '') for name in LOG_FIELDS}
        row.update({
            'session_id': self.session_id,
            'reason': reason,
            'wall_time': time.time(),
            'sim_age': world.age,
            'fps': fps,
            'sim_rate': sim_rate,
        })
        try:
            new_file = not os.path.exists(self.path)
            with open(self.path, 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.last_log_age = world.age
            self.status = 'OK'
        except Exception:
            self.status = 'ERR'


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    rows = []
    with open(log_path, 'r', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 'NO DATA'
    grouped = {}
    for row in rows:
        grouped.setdefault(row['session_id'], []).append(row)
    session_rows = []
    lines = [BUILD_LONG + ' long-run report', '']
    for session_id, items in grouped.items():
        last = items[-1]
        session_rows.append({
            'session_id': session_id,
            'rows': len(items),
            'final_age': last['sim_age'],
            'final_cells': last['cells'],
            'final_residual': last['matter_residual'],
            'final_attachments': last['neural_attachments'],
        })
        lines.append(
            '{}: rows={} age={} cells={} attachments={} residual={}'.format(
                session_id, len(items), last['sim_age'], last['cells'],
                last['neural_attachments'], last['matter_residual'],
            )
        )
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    with open(session_path, 'w', newline='') as handle:
        fields = ('session_id', 'rows', 'final_age', 'final_cells', 'final_residual', 'final_attachments')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(session_rows)
    return 'OK'


try:
    from scene import (
        Scene, run, LANDSCAPE, background, fill, rect, ellipse,
        line, stroke, stroke_weight, text,
    )

    class SomaCellP0Scene(s5.SomaCellEcologyScene):
        def setup(self):
            self.paused = False
            self.accumulator = 0.0
            self.last_wall = time.time()
            self.last_touch_wall = -10.0
            self.last_save_age = 0.0
            self.save_status = 'NEW'
            self.report_status = 'WAIT'
            self.fps = 0.0
            self.sim_rate = 0.0
            self.telemetry_wall = time.time()
            self.telemetry_age = 0.0
            self.telemetry_frames = 0
            try:
                self.world = P0World.load(SAVE_FILE)
                self.save_status = 'LOADED'
            except Exception:
                self.world = P0World(seed=101, initial_cells=3)
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def update(self):
            now = time.time()
            elapsed_wall = clamp(now - self.last_wall, 0.0, 0.25)
            self.last_wall = now
            if not self.paused:
                self.accumulator += elapsed_wall
                fixed = 1.0 / SIM_HZ
                steps = 0
                while self.accumulator >= fixed and steps < 7:
                    self.world.step(fixed)
                    self.accumulator -= fixed
                    steps += 1
                if steps >= 7:
                    self.accumulator = min(self.accumulator, fixed)
            self.telemetry_frames += 1
            elapsed = now - self.telemetry_wall
            if elapsed >= 1.0:
                self.fps = self.telemetry_frames / elapsed
                self.sim_rate = (self.world.age - self.telemetry_age) / elapsed
                self.telemetry_wall = now
                self.telemetry_age = self.world.age
                self.telemetry_frames = 0
            self.logger.log(self.world, self.fps, self.sim_rate)
            if self.world.age - self.last_save_age >= AUTO_SAVE_INTERVAL:
                try:
                    self.world.save(SAVE_FILE)
                    self.save_status = 'OK'
                    self.last_save_age = self.world.age
                    gc.collect()
                except Exception:
                    self.save_status = 'ERR'
            if not self.world.living_cells() and self.report_status == 'WAIT':
                self.logger.log(self.world, self.fps, self.sim_rate, reason='extinct', force=True)
                self.report_status = generate_report()

        def draw(self):
            # Inherit the proven 0.5 world drawing, then replace the header/HUD
            # with P0-specific contract telemetry.
            super(SomaCellP0Scene, self).draw()
            fill(0.010, 0.018, 0.030, 1.0)
            rect(0, self.size.h - 48, self.size.w, 48)
            rect(0, 0, self.size.w, 72)
            summary = self.world.summary()
            fill(0.92, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.64, 0.78, 0.86)
            text('PORT {} | SAVE {} | LOG {} | {:.1f} fps | x{:.2f}'.format(
                PORT_SCHEMA_VERSION, self.save_status, self.logger.status,
                self.fps, self.sim_rate,
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} corpses {} DNA {} div {} deaths {}'.format(
                summary['age'], summary['cells'], summary['corpses'],
                summary['edna_fragments'], summary['divisions'], summary['deaths'],
            ), x=24, y=58, font_size=10, alignment=4)
            text('attachments {} escrow material {:.5f} ATP {:.5f}'.format(
                summary['neural_attachments'], summary['neural_material_escrow'],
                summary['neural_atp_escrow'],
            ), x=24, y=42, font_size=9, alignment=4)
            text('budget requested {:.5f} granted {:.5f} returned {:.5f} spent {:.5f}'.format(
                summary['neural_budget_requested'], summary['neural_budget_granted'],
                summary['neural_budget_returned'], summary['neural_budget_spent'],
            ), x=24, y=26, font_size=9, alignment=4)
            text('effectors {} rejected {} ledger {:+.2e}'.format(
                summary['neural_effector_calls'], summary['neural_effector_rejections'],
                summary['matter_residual'],
            ), x=24, y=10, font_size=9, alignment=4)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = P0World(seed=101, initial_cells=3)
                self.logger = LongRunLogger(self.world)
                self.logger.log(self.world, reason='reset', force=True)
                self.report_status = 'WAIT'
                self.save_status = 'NEW'
                self.accumulator = 0.0
                self.paused = False
                self.last_touch_wall = -10.0
                gc.collect()
                return
            self.last_touch_wall = now
            if touch.location.y > self.size.h - 60:
                self.paused = not self.paused
                self.logger.log(
                    self.world, self.fps, self.sim_rate,
                    reason='pause' if self.paused else 'resume', force=True,
                )
                return
            position = self._unit_position(touch.location)
            if not self.world.puncture_nearest(position):
                self.world.inject_cloud(position)

        def stop(self):
            try:
                self.world.save(SAVE_FILE)
                self.save_status = 'OK'
            except Exception:
                self.save_status = 'ERR'
            self.logger.log(self.world, self.fps, self.sim_rate, reason='stop', force=True)
            self.report_status = generate_report()

except ImportError:
    Scene = None


if __name__ == '__main__':
    if Scene is None:
        print(run_headless_trial(seed=101, seconds=20.0, initial_cells=3))
    else:
        run(SomaCellP0Scene(), LANDSCAPE, show_fps=False)
