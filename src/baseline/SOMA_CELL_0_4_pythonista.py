# coding: utf-8
"""
SOMA-CELL 0.4 — Autopoietic Sensorimotor Semantics
自己生産に根差した原始感覚運動

Pythonista 3 / CPython + NumPy research prototype.

This module extends SOMA-CELL 0.3 without reintroducing a scalar ``health`` or
an externally supplied reward label.  A variable-length material genome now
encodes membrane receptors and ATP-paid effectors.  Receptors sample local
external chemistry around the actual membrane; effectors polarise existing
transporters, contract the surface, and modulate quiescence.  The sign of a
ligand is not fixed in the main condition.  Labile controller phosphorylation
is updated from small endogenous perturbations and the subsequent change in an
"autopoietic margin" derived only from the currently realised boundary,
reaction loop, energy carrier, proteostasis, hereditary integrity and leakage.

The model asks a limited, falsifiable question: can a materially self-producing
protocell discover that a signal is useful or harmful *for its own current
chemistry*, pay for sensing and movement, and reverse that coupling after the
chemistry changes?

This remains a coarse-grained artificial chemistry.  Ligand classes, receptor
and effector grammar, matter types, physics and learning law are human-designed.
It is not a claim of consciousness, biological life, or open-ended evolution.
"""

from __future__ import division

import csv
import gc
import hashlib
import json
import math
import os
import pickle
import time

import numpy as np

import SOMA_CELL_0_1_pythonista as base
import SOMA_CELL_0_2_pythonista as g2
import SOMA_CELL_0_3_pythonista as g3


BUILD = 'SOMA-CELL 0.4.0'
SAVE_VERSION = 4
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_4.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_4_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_4_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_4_sessions.csv')

SIM_HZ = 20.0
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0
MAX_CELLS = 12

# Re-export inherited matter and chemistry constants.
PARTICLE_FUEL = g3.PARTICLE_FUEL
PARTICLE_MINERAL = g3.PARTICLE_MINERAL
PARTICLE_WASTE = g3.PARTICLE_WASTE
PARTICLE_ALT = g3.PARTICLE_ALT
PARTICLE_NAMES = g3.PARTICLE_NAMES

POOL_FUEL = g3.POOL_FUEL
POOL_MINERAL = g3.POOL_MINERAL
POOL_ATP = g3.POOL_ATP
POOL_MEM_PRECURSOR = g3.POOL_MEM_PRECURSOR
POOL_CATALYST = g3.POOL_CATALYST
POOL_TRANSPORTER_PRECURSOR = g3.POOL_TRANSPORTER_PRECURSOR
POOL_WASTE = g3.POOL_WASTE
POOL_ALT = g3.POOL_ALT
POOL_INTERMEDIATE = g3.POOL_INTERMEDIATE
POOL_NUCLEOTIDE = g3.POOL_NUCLEOTIDE
POOL_DAMAGED_PROTEIN = g3.POOL_DAMAGED_PROTEIN
POOL_AGGREGATE = g3.POOL_AGGREGATE
POOL_REACTIVE = g3.POOL_REACTIVE
POOL_COUNT = g3.POOL_COUNT

CHANNEL_FUEL = g3.CHANNEL_FUEL
CHANNEL_MINERAL = g3.CHANNEL_MINERAL
CHANNEL_WASTE = g3.CHANNEL_WASTE
CHANNEL_ALT = g3.CHANNEL_ALT
CHANNEL_COUNT = g3.CHANNEL_COUNT

MEMBRANE_SEGMENTS = g3.MEMBRANE_SEGMENTS
BASE_RADIUS = g3.BASE_RADIUS
INITIAL_MEMBRANE_MASS = g3.INITIAL_MEMBRANE_MASS
MONOMER_MASS = g3.MONOMER_MASS

ROLE_GENERIC = g3.ROLE_GENERIC
ROLE_REGULATOR = g3.ROLE_REGULATOR

# Regulator-gene localisation now defines a physically interpretable class.
# Localisation 0 remains exactly the 0.3 repair/life-history vocabulary.
LOC_REPAIR = 0
LOC_SENSOR = 1
LOC_EFFECTOR = 2
LOC_RESERVED = 3

# Ligands are observations, not built-in rewards.
LIGAND_FUEL = 0
LIGAND_MINERAL = 1
LIGAND_ALT = 2
LIGAND_WASTE = 3
LIGAND_STRESS = 4
LIGAND_TENSION = 5
LIGAND_ATP_DEFICIT = 6
LIGAND_DAMAGE = 7
LIGAND_NAMES = (
    'fuel', 'mineral', 'alt', 'waste',
    'stress', 'tension', 'ATP-deficit', 'damage',
)
LIGAND_COUNT = len(LIGAND_NAMES)

# Controller output channels.  A sensor gene maps one ligand to one channel.
CONTROL_MOTOR = 0
CONTROL_TRANSPORT = 1
CONTROL_REPAIR_POLARITY = 2
CONTROL_QUIESCENCE = 3
CONTROL_TUMBLE = 4
CONTROL_SECRETION = 5
CONTROL_RESERVED_6 = 6
CONTROL_RESERVED_7 = 7
CONTROL_COUNT = 8
CONTROL_NAMES = (
    'motor', 'transport-polarity', 'repair-polarity', 'quiescence',
    'tumble', 'secretion', 'reserved-6', 'reserved-7',
)

# Effector gene parameter.
EFFECT_MOTOR = 0
EFFECT_TRANSPORT_POLARITY = 1
EFFECT_QUIESCENCE = 2
EFFECT_REPAIR_POLARITY = 3
EFFECT_TUMBLE = 4
EFFECT_WASTE_JET = 5
EFFECT_RECEPTOR_RELOCATION = 6
EFFECT_RESERVED = 7
EFFECT_NAMES = (
    'contractile-motor', 'transporter-polariser', 'quiescence-gate',
    'repair-polariser', 'tumble', 'waste-jet',
    'receptor-relocation', 'reserved',
)

clamp = g3.clamp
wrapped_delta = g3.wrapped_delta
torus_distance = g3.torus_distance
unit_vector = g3.unit_vector
finite_array = g3.finite_array
circular_smooth = g3.circular_smooth
_atomic_pickle = g3._atomic_pickle
_memory_peak_mb_estimate = g3._memory_peak_mb_estimate


def _sequence_hash(sequence):
    return hashlib.sha1(np.asarray(sequence, dtype=np.uint8).tobytes()).hexdigest()[:12]


def make_sensor_gene(ligand, control_channel, weight_symbol=4, affinity=5,
                     adaptation=4, plasticity=6, promoter=5, efficiency=5,
                     fidelity=5):
    """Encode a membrane receptor plus a labile controller coupling.

    The four spare symbols encode inherited baseline sign/gain, ligand affinity,
    receptor adaptation rate and lifetime plasticity.  No ligand-specific
    valence table is used by the main condition.
    """
    return g2.make_gene(
        ROLE_REGULATOR,
        parameter=int(ligand),
        regulator=int(control_channel),
        promoter=int(promoter),
        efficiency=int(efficiency),
        fidelity=int(fidelity),
        localisation=LOC_SENSOR,
        spare=(weight_symbol, affinity, adaptation, plasticity),
    )


def make_effector_gene(effect, control_channel, gain=5, rate=5, cost=4,
                       geometry=4, promoter=5, efficiency=5, fidelity=5):
    """Encode one ATP-paid membrane effector listening to a control channel."""
    return g2.make_gene(
        ROLE_REGULATOR,
        parameter=int(effect),
        regulator=int(control_channel),
        promoter=int(promoter),
        efficiency=int(efficiency),
        fidelity=int(fidelity),
        localisation=LOC_EFFECTOR,
        spare=(gain, rate, cost, geometry),
    )


def founding_genome_04():
    """0.3 repair genome plus neutral sensory priors and a complete alt path.

    Every external ligand starts with the *same weak attractive prior*.  Waste
    is not hard-coded as repellent and fuel is not hard-coded as reward.  The
    labile coupling can reverse sign from embodied consequences.  The complete
    alternative-substrate pathway gives the same cell two chemically distinct
    energy opportunities for switch experiments.
    """
    genes = [g3.founding_genome_03()]
    # Complete the alternative pathway without changing the generic grammar.
    genes.append(g2.make_gene(
        g2.ROLE_GENERIC,
        parameter=g2.REACTION_INTERMEDIATE_TO_WASTE,
        promoter=4,
        efficiency=5,
        fidelity=5,
    ))
    # Four externally oriented sensor-controller proteins.  The inherited
    # weight symbol is identical for all four: no built-in good/bad table.
    genes.extend([
        make_sensor_gene(LIGAND_FUEL, CONTROL_MOTOR, weight_symbol=4),
        make_sensor_gene(LIGAND_ALT, CONTROL_MOTOR, weight_symbol=4),
        make_sensor_gene(LIGAND_WASTE, CONTROL_MOTOR, weight_symbol=4),
        make_sensor_gene(LIGAND_ATP_DEFICIT, CONTROL_QUIESCENCE,
                         weight_symbol=4, affinity=4, adaptation=3, plasticity=5),
        make_effector_gene(EFFECT_MOTOR, CONTROL_MOTOR, gain=5, rate=5, cost=4),
        make_effector_gene(EFFECT_TRANSPORT_POLARITY, CONTROL_MOTOR,
                           gain=4, rate=4, cost=4),
        make_effector_gene(EFFECT_QUIESCENCE, CONTROL_QUIESCENCE,
                           gain=4, rate=4, cost=3),
    ])
    sequence = np.concatenate(genes).astype(np.uint8)
    if len(sequence) > g2.MAX_GENOME_LENGTH:
        raise ValueError('0.4 founding genome exceeds material genome limit')
    return sequence


def _sensor_parameters(spec):
    payload = spec['payload']
    genetic_weight = ((float(payload[8]) - 3.5) / 3.5) * 0.80
    affinity = 0.70 + 4.30 * (float(payload[9]) / 7.0)
    adaptation_rate = 0.015 + 0.34 * (float(payload[10]) / 7.0)
    plasticity = 0.0015 + 0.034 * (float(payload[11]) / 7.0)
    exploration = 0.035 + 0.15 * (1.0 - float(spec['fidelity']))
    return genetic_weight, affinity, adaptation_rate, plasticity, exploration


def _effector_parameters(spec):
    payload = spec['payload']
    gain = 0.35 + 1.65 * (float(payload[8]) / 7.0)
    rate = 0.20 + 1.80 * (float(payload[9]) / 7.0)
    cost = 0.50 + 1.10 * (float(payload[10]) / 7.0)
    geometry = (float(payload[11]) - 3.5) / 3.5
    return gain, rate, cost, geometry


def sensorimotor_gene_counts(sequence):
    sensors = np.zeros(LIGAND_COUNT, dtype=int)
    effectors = np.zeros(8, dtype=int)
    for spec in g2.parse_genes(sequence):
        if spec['role'] != ROLE_REGULATOR:
            continue
        if spec['localisation'] == LOC_SENSOR:
            sensors[int(spec['parameter']) % LIGAND_COUNT] += 1
        elif spec['localisation'] == LOC_EFFECTOR:
            effectors[int(spec['parameter']) % 8] += 1
    return sensors, effectors


class SensorimotorConfig(g3.LifeHistoryConfig):
    """Ablations and environmental switches for primitive sensorimotor control."""

    def __init__(
        self,
        sensorimotor=True,
        receptors=True,
        active_motion=True,
        transporter_polarity=True,
        quiescence_effector=True,
        plasticity=True,
        receptor_adaptation=True,
        causal_perturbation=True,
        sensor_cost=True,
        motor_cost=True,
        plasticity_cost=True,
        fixed_reflex=False,
        random_motor=False,
        inherited_controller_state=True,
        environment_mode='patchy',
        switch_age=150.0,
        contaminated_fraction=0.78,
        motor_gain_scale=1.0,
        learning_rate_scale=1.0,
        sensor_cost_scale=1.0,
        motor_cost_scale=1.0,
        **kwargs
    ):
        super(SensorimotorConfig, self).__init__(**kwargs)
        self.sensorimotor = bool(sensorimotor)
        self.receptors = bool(receptors)
        self.active_motion = bool(active_motion)
        self.transporter_polarity = bool(transporter_polarity)
        self.quiescence_effector = bool(quiescence_effector)
        self.plasticity = bool(plasticity)
        self.receptor_adaptation = bool(receptor_adaptation)
        self.causal_perturbation = bool(causal_perturbation)
        self.sensor_cost = bool(sensor_cost)
        self.motor_cost = bool(motor_cost)
        self.plasticity_cost = bool(plasticity_cost)
        self.fixed_reflex = bool(fixed_reflex)
        self.random_motor = bool(random_motor)
        self.inherited_controller_state = bool(inherited_controller_state)
        self.environment_mode = str(environment_mode)
        self.switch_age = float(switch_age)
        self.contaminated_fraction = float(contaminated_fraction)
        self.motor_gain_scale = float(motor_gain_scale)
        self.learning_rate_scale = float(learning_rate_scale)
        self.sensor_cost_scale = float(sensor_cost_scale)
        self.motor_cost_scale = float(motor_cost_scale)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class SensorimotorParticleField(g2.GeneticParticleField):
    """Material particles maintained in spatially distinct geochemical patches."""

    DEFAULT_CENTRES = np.asarray([
        [0.30, 0.66],  # fuel
        [0.70, 0.34],  # mineral
        [0.69, 0.69],  # waste / oxidative debris
        [0.31, 0.31],  # alternative substrate
    ], dtype=float)

    def __init__(self, rng, initial=True, alt_niche=True, environment_mode='patchy'):
        self.environment_mode = str(environment_mode)
        self.patch_centres = self.DEFAULT_CENTRES.copy()
        self.world_age = 0.0
        self.switch_age = 150.0
        super(SensorimotorParticleField, self).__init__(
            rng, initial=False, alt_niche=alt_niche
        )
        if initial:
            self.seed_initial()

    def _spawn_near(self, kind, count, mean_amount, spread=0.060):
        centre = self.patch_centres[int(kind)]
        positions = (centre[None, :] + self.rng.normal(0.0, spread, (count, 2))) % 1.0
        amounts = np.clip(
            self.rng.normal(mean_amount, mean_amount * 0.15, count),
            mean_amount * 0.45,
            mean_amount * 1.55,
        )
        self.add_many(kind, positions, amounts, count_as_injection=False)

    def seed_initial(self):
        if self.environment_mode == 'uniform':
            return super(SensorimotorParticleField, self).seed_initial()
        for kind, count, amount, spread in (
            (PARTICLE_FUEL, 62, 0.029, 0.065),
            (PARTICLE_MINERAL, 66, 0.027, 0.070),
            (PARTICLE_WASTE, 24, 0.018, 0.055),
            (PARTICLE_ALT, 64, 0.029, 0.065),
        ):
            self._spawn_near(kind, count, amount, spread)

    def step_diffusion(self, dt):
        if len(self.amount) == 0:
            return
        if self.environment_mode == 'uniform':
            return super(SensorimotorParticleField, self).step_diffusion(dt)
        diffusion = np.asarray([0.0026, 0.0023, 0.0018, 0.0025], dtype=float)[self.kind]
        noise = self.rng.normal(0.0, 1.0, self.pos.shape)
        self.pos = (self.pos + noise * np.sqrt(2.0 * diffusion[:, None] * dt)) % 1.0
        centres = self.patch_centres[self.kind]
        drift = (centres - self.pos + 0.5) % 1.0 - 0.5
        # A weak geochemical potential maintains gradients without pinning each
        # particle.  This moves existing matter; it does not create it.
        self.pos = (self.pos + drift * (0.020 * dt)) % 1.0

    def external_inflow(self, dt, enabled=True):
        if not enabled:
            return
        if self.environment_mode == 'uniform':
            return super(SensorimotorParticleField, self).external_inflow(dt, enabled)
        switched = self.environment_mode == 'reversal' and self.world_age >= self.switch_age
        rates = {
            PARTICLE_FUEL: 0.015 if not switched else 0.006,
            PARTICLE_MINERAL: 0.006,
            PARTICLE_ALT: 0.008 if not switched else 0.020,
        }
        for kind, rate in rates.items():
            expected = rate * dt
            if self.rng.random() < expected / 0.020:
                position = (
                    self.patch_centres[kind]
                    + self.rng.normal(0.0, 0.050, 2)
                ) % 1.0
                self.add_particle(kind, position, 0.020, count_as_injection=True)

    def add_cloud(self, position, fuel=0.18, mineral=0.15, alt=0.12, spread=0.025):
        position = np.asarray(position, dtype=float)
        for kind, total, count in (
            (PARTICLE_FUEL, float(fuel), 7),
            (PARTICLE_MINERAL, float(mineral), 6),
            (PARTICLE_ALT, float(alt), 5),
        ):
            positions = (position[None, :] + self.rng.normal(0.0, spread, (count, 2))) % 1.0
            self.add_many(
                kind, positions, np.full(count, total / count),
                count_as_injection=True,
            )

    def state_dict(self):
        state = super(SensorimotorParticleField, self).state_dict()
        state.update({
            'environment_mode': self.environment_mode,
            'patch_centres': self.patch_centres.copy(),
            'world_age': float(self.world_age),
            'switch_age': float(self.switch_age),
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        field = cls(
            rng, initial=False,
            alt_niche=state.get('alt_niche', True),
            environment_mode=state.get('environment_mode', 'patchy'),
        )
        # Restore every inherited material ledger field.
        field.pos = np.asarray(state['pos'], dtype=float).copy()
        field.kind = np.asarray(state['kind'], dtype=np.int16).copy()
        field.amount = np.asarray(state['amount'], dtype=float).copy()
        field.recycled_fuel_buffer = float(state.get('recycled_fuel_buffer', 0.0))
        field.recycled_mineral_buffer = float(state.get('recycled_mineral_buffer', 0.0))
        field.injected_material = float(state.get('injected_material', 0.0))
        field.dissipated_material = float(state.get('dissipated_material', 0.0))
        field.recycled_alt_buffer = float(state.get('recycled_alt_buffer', 0.0))
        field.patch_centres = np.asarray(
            state.get('patch_centres', cls.DEFAULT_CENTRES), dtype=float
        ).copy()
        field.world_age = float(state.get('world_age', 0.0))
        field.switch_age = float(state.get('switch_age', 150.0))
        return field


class SensorimotorProtoCell(g3.DamageProtoCell):
    """0.3 protocell with gene-derived receptors, effectors and labile meaning."""

    def __init__(self, cell_id, rng, position=None, generation=0, lineage=0,
                 bootstrap=True):
        super(SensorimotorProtoCell, self).__init__(
            cell_id, rng, position=position, generation=generation,
            lineage=lineage, bootstrap=bootstrap,
        )
        self._init_sensorimotor_state()
        if bootstrap:
            # Existing 0.3 gene fingerprints remain valid; append the 0.4
            # sensory/effector products and make the alt pathway complete.
            self.genomes = [founding_genome_04()]
            self.genome_lesions = [0.0]
            self.replication_template = None
            self.replication_copy = []
            self.replication_fractional = 0.0
            self._refresh_gene_cache()
            for fingerprint, spec in self.gene_specs.items():
                if spec['role'] == ROLE_REGULATOR:
                    if spec['localisation'] == LOC_SENSOR:
                        self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + 0.013
                    elif spec['localisation'] == LOC_EFFECTOR:
                        self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + 0.016
                elif (
                    spec['role'] == ROLE_GENERIC
                    and spec.get('reaction') == g2.REACTION_INTERMEDIATE_TO_WASTE
                ):
                    self.proteins[fingerprint] = self.proteins.get(fingerprint, 0.0) + 0.020
            # The larger hereditary program needs explicit free monomers for a
            # second copy; this is bootstrap matter and enters the initial ledger.
            self.pools[POOL_NUCLEOTIDE] = max(self.pools[POOL_NUCLEOTIDE], 0.37)
            self._sync_protein_pool()
            self.previous_autopoietic_margin = self.autopoietic_margin()

    def _init_sensorimotor_state(self):
        self.receptor_baseline = {}
        self.controller_phosphorylation = {}
        self.controller_eligibility = {}
        self.controller_noise = {}
        self.sensor_activation_by_gene = {}
        self.meaning_mean = np.zeros(LIGAND_COUNT, dtype=float)
        self.meaning_confidence = np.zeros(LIGAND_COUNT, dtype=float)
        self.meaning_trace = np.zeros(LIGAND_COUNT, dtype=float)
        self.uptake_trace = np.zeros(LIGAND_COUNT, dtype=float)
        self.ligand_activation_baseline = np.zeros(LIGAND_COUNT, dtype=float)
        self.last_uptake_by_ligand = np.zeros(LIGAND_COUNT, dtype=float)
        self.cumulative_uptake_by_ligand = np.zeros(LIGAND_COUNT, dtype=float)
        self.last_gene_perturbation = {}
        self.last_ligand_profiles = np.zeros((LIGAND_COUNT, MEMBRANE_SEGMENTS), dtype=float)
        self.last_ligand_gradients = np.zeros((LIGAND_COUNT, 2), dtype=float)
        self.last_receptor_activity = np.zeros(LIGAND_COUNT, dtype=float)
        self.last_control_vectors = np.zeros((CONTROL_COUNT, 2), dtype=float)
        self.last_control_scalars = np.zeros(CONTROL_COUNT, dtype=float)
        self.last_motor_command = np.zeros(2, dtype=float)
        self.last_motor_force = 0.0
        self.last_sensor_atp = 0.0
        self.last_motor_atp = 0.0
        self.last_polarity_atp = 0.0
        self.last_plasticity_atp = 0.0
        self.last_sensorimotor_reactive = 0.0
        self.last_modulator = 0.0
        self.last_margin = 0.0
        self.previous_autopoietic_margin = 0.0
        self.expected_margin_drift = 0.0
        self.behavioural_quiescence = 0.0
        self.repair_polarity = np.zeros(MEMBRANE_SEGMENTS, dtype=float)
        self.cumulative_sensor_atp = 0.0
        self.cumulative_motor_atp = 0.0
        self.cumulative_polarity_atp = 0.0
        self.cumulative_plasticity_atp = 0.0
        self.cumulative_active_distance = 0.0
        self.cumulative_passive_distance = 0.0
        self.controller_updates = 0
        self.meaning_reversals = 0
        self.fuel_contamination_received = 0.0
        self.sensorimotor_steps = 0
        self.last_position_for_distance = self.pos.copy()

    # ------------------------------------------------------------------
    # Gene interpretation and materially paid expression
    # ------------------------------------------------------------------

    def raw_repair_activity(self, kind):
        # 0.4 sensory and effector regulator proteins must not be accidentally
        # counted as one of the eight 0.3 repair proteins.
        total = 0.0
        for fingerprint, amount in self.proteins.items():
            spec = self.gene_specs.get(fingerprint)
            if (
                spec is not None
                and spec['role'] == ROLE_REGULATOR
                and spec['localisation'] == LOC_REPAIR
                and int(spec['parameter']) % 8 == int(kind)
            ):
                total += amount * spec['efficiency'] * spec['promoter']
        inhibition = 1.0 / (1.0 + 2.6 * self.aggregate_concentration())
        return float(total / 0.014 * inhibition)

    def _protein_need(self, spec):
        if spec['role'] == ROLE_REGULATOR:
            if spec['localisation'] == LOC_SENSOR:
                ligand = int(spec['parameter']) % LIGAND_COUNT
                activation = self.last_receptor_activity[ligand]
                return clamp(0.15 + 1.4 * activation + 0.25 * (1.0 - self.closure()), 0.10, 1.55)
            if spec['localisation'] == LOC_EFFECTOR:
                channel = int(spec['regulator']) % CONTROL_COUNT
                drive = float(np.linalg.norm(self.last_control_vectors[channel]))
                drive += abs(float(self.last_control_scalars[channel]))
                return clamp(0.16 + 1.2 * drive + 0.15 * self.damage_burden(), 0.10, 1.55)
        return super(SensorimotorProtoCell, self)._protein_need(spec)

    def sensor_specs(self):
        return [
            (fingerprint, spec)
            for fingerprint, spec in self.gene_specs.items()
            if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_SENSOR
        ]

    def effector_specs(self):
        return [
            (fingerprint, spec)
            for fingerprint, spec in self.gene_specs.items()
            if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_EFFECTOR
        ]

    def _clean_control_state(self):
        valid = set(fingerprint for fingerprint, _ in self.sensor_specs())
        for mapping in (
            self.receptor_baseline, self.controller_phosphorylation,
            self.controller_eligibility, self.controller_noise,
            self.sensor_activation_by_gene, self.last_gene_perturbation,
        ):
            for key in list(mapping.keys()):
                if key not in valid:
                    del mapping[key]
        for fingerprint in valid:
            self.receptor_baseline.setdefault(fingerprint, 0.0)
            self.controller_phosphorylation.setdefault(fingerprint, 0.0)
            self.controller_eligibility.setdefault(fingerprint, 0.0)
            self.controller_noise.setdefault(fingerprint, 0.0)
            self.sensor_activation_by_gene.setdefault(fingerprint, 0.0)
            self.last_gene_perturbation.setdefault(fingerprint, 0.0)

    def _replicate_genome(self, world, dt, config):
        """0.3 replication with structural mutation config filtered safely.

        SOMA-CELL 0.3 internally reconstructed a ``LifeHistoryConfig`` from the
        full config dictionary.  0.4 adds sensorimotor keys, so passing the full
        dictionary would make the parent constructor reject unknown fields.
        The copying chemistry is otherwise identical to 0.3.
        """
        self.last_replication_symbols = 0
        if not config.genome_replication or not self.genomes:
            return
        replicase = self.role_activity(g3.ROLE_REPLICASE)
        if config.external_replicase:
            replicase += 0.85
        if replicase <= 1e-6:
            return
        if self.replication_template is None:
            if len(self.genomes) >= 2:
                return
            template_index = int(world.rng.integers(0, len(self.genomes)))
            self.replication_template = self.genomes[template_index].copy()
            self.replication_template_lesion = float(
                self.genome_lesions[template_index]
                if template_index < len(self.genome_lesions) else 0.0
            )
            self.replication_copy = []
            self.replication_fractional = 0.0

        template = self.replication_template
        if template is None:
            return
        proofreading = (
            self.raw_repair_activity(g3.REPAIR_PROOFREADING)
            if config.proofreading else 0.0
        )
        proof_fraction = proofreading / (0.75 + proofreading)
        quiescence = self.quiescence_level(config)
        sat_nucleotide = self.pools[POOL_NUCLEOTIDE] / (
            0.055 + self.pools[POOL_NUCLEOTIDE]
        )
        sat_atp = self.pools[POOL_ATP] / (0.10 + self.pools[POOL_ATP])
        speed = 10.0 * replicase * sat_nucleotide * sat_atp
        speed *= (1.0 - 0.34 * proof_fraction) * (1.0 - 0.78 * quiescence)
        self.replication_fractional += speed * dt
        requested = int(self.replication_fractional)
        self.replication_fractional -= requested

        lesion_error = 0.0012 * self.replication_template_lesion
        reactive_error = 0.0010 * self.reactive_concentration()
        raw_error = max(0.0, config.mutation_rate + lesion_error + reactive_error)
        effective_error = raw_error * (1.0 - 0.82 * proof_fraction)
        self.last_effective_error_rate = effective_error
        extra_atp = 0.00075 * proof_fraction

        for _ in range(requested):
            index = len(self.replication_copy)
            if index >= len(template):
                break
            atp_per_symbol = g3.REPLICATION_ATP_PER_SYMBOL + extra_atp
            if (
                self.pools[POOL_NUCLEOTIDE] < MONOMER_MASS
                or self.pools[POOL_ATP] < atp_per_symbol + 0.022
            ):
                break
            symbol = int(template[index])
            if config.mutation and world.rng.random() < effective_error:
                new = int(world.rng.integers(0, g2.ALPHABET_SIZE - 1))
                if new >= symbol:
                    new += 1
                symbol = new % g2.ALPHABET_SIZE
                self.mutation_events['substitution'] += 1
            self.replication_copy.append(symbol)
            self.pools[POOL_NUCLEOTIDE] -= MONOMER_MASS
            self.pools[POOL_ATP] -= atp_per_symbol
            self.cumulative_proofreading_atp += extra_atp
            self.last_replication_symbols += 1

        if len(self.replication_copy) < len(template):
            return

        copied = np.asarray(self.replication_copy, dtype=np.uint8)
        base_keys = set(g3.LifeHistoryConfig().__dict__.keys())
        structural_state = {
            key: getattr(config, key)
            for key in base_keys if hasattr(config, key)
        }
        structural_config = g3.LifeHistoryConfig(**structural_state)
        structural_config.mutation_rate = 0.0
        budget_symbols = int(self.pools[POOL_NUCLEOTIDE] / MONOMER_MASS)
        copied, delta_symbols, events = g2.mutate_sequence(
            copied, world.rng, structural_config,
            nucleotide_budget=budget_symbols,
        )
        if delta_symbols > 0:
            self.pools[POOL_NUCLEOTIDE] -= delta_symbols * MONOMER_MASS
        elif delta_symbols < 0:
            self.pools[POOL_NUCLEOTIDE] += (-delta_symbols) * MONOMER_MASS
        for name, count in events.items():
            self.mutation_events[name] += int(count)
        self.genomes.append(copied)
        inherited_lesion = self.replication_template_lesion * (
            0.28 + 0.22 * (1.0 - proof_fraction)
        )
        new_lesion = inherited_lesion + effective_error * len(copied) * 0.06
        self.genome_lesions.append(float(new_lesion))
        self.replication_cycles += 1
        self.replication_template = None
        self.replication_template_lesion = 0.0
        self.replication_copy = []
        self.replication_fractional = 0.0
        self._refresh_gene_cache()
        if self.novel_path_first_age is None and g2.sequence_has_novel_path(copied):
            self.novel_path_first_age = float(self.age)

    # ------------------------------------------------------------------
    # Local membrane sensing and self-production-grounded meaning
    # ------------------------------------------------------------------

    def _particle_ligand_profiles(self, field):
        profiles = np.zeros((4, MEMBRANE_SEGMENTS), dtype=float)
        if len(field.amount) == 0:
            return profiles
        deltas = (field.pos - self.pos[None, :] + 0.5) % 1.0 - 0.5
        distances = np.linalg.norm(deltas, axis=1)
        sense_radius = self.radius + 0.20
        indices = np.where(distances < sense_radius)[0]
        if not len(indices):
            return profiles
        for index in indices:
            distance = max(1e-7, float(distances[index]))
            kind = int(field.kind[index])
            if kind < 0 or kind > 3:
                continue
            segment = self.segment_for_delta(deltas[index])
            amount = float(field.amount[index])
            weight = amount * math.exp(-distance / 0.070) / (0.020 + distance)
            profiles[kind, segment] += weight
        for ligand in range(4):
            profile = profiles[ligand]
            # Membrane-local diffusion of receptor occupancy.
            profile = circular_smooth(profile, 0.34)
            profile = circular_smooth(profile, 0.22)
            profiles[ligand] = profile
        return profiles

    def _all_ligand_profiles(self, world):
        profiles = np.zeros((LIGAND_COUNT, MEMBRANE_SEGMENTS), dtype=float)
        particle_profiles = self._particle_ligand_profiles(world.field)
        profiles[LIGAND_FUEL] = particle_profiles[PARTICLE_FUEL]
        profiles[LIGAND_MINERAL] = particle_profiles[PARTICLE_MINERAL]
        profiles[LIGAND_ALT] = particle_profiles[PARTICLE_ALT]
        profiles[LIGAND_WASTE] = particle_profiles[PARTICLE_WASTE]
        profiles[LIGAND_STRESS] = world.stress_profile(self)
        profiles[LIGAND_TENSION] = self.tension()
        profiles[LIGAND_ATP_DEFICIT] = 1.0 - self.pools[POOL_ATP] / (0.16 + self.pools[POOL_ATP])
        profiles[LIGAND_DAMAGE] = clamp(self.damage_burden(), 0.0, 2.0)
        return np.maximum(profiles, 0.0)

    @staticmethod
    def _fixed_reflex_weight(ligand, has_alt_path=True):
        # An explicitly labelled control used only as an ablation/comparator.
        table = {
            LIGAND_FUEL: 0.82,
            LIGAND_MINERAL: 0.64,
            LIGAND_ALT: 0.62 if has_alt_path else -0.18,
            LIGAND_WASTE: -0.82,
            LIGAND_STRESS: -0.76,
            LIGAND_TENSION: 0.0,
            LIGAND_ATP_DEFICIT: 0.70,
            LIGAND_DAMAGE: 0.55,
        }
        return float(table.get(int(ligand), 0.0))

    def sense_environment(self, world, dt, config):
        self.last_control_vectors[:] = 0.0
        self.last_control_scalars[:] = 0.0
        self.last_receptor_activity[:] = 0.0
        self.last_sensor_atp = 0.0
        self.last_ligand_profiles = self._all_ligand_profiles(world)
        self._clean_control_state()
        if not config.sensorimotor or not config.receptors:
            return

        angles = 2.0 * math.pi * (np.arange(MEMBRANE_SEGMENTS) + 0.5) / MEMBRANE_SEGMENTS
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        closure = self.closure_array()
        total_signal_cost = 0.0
        ligand_activation = np.zeros(LIGAND_COUNT, dtype=float)

        for fingerprint, spec in self.sensor_specs():
            protein = max(0.0, float(self.proteins.get(fingerprint, 0.0)))
            if protein <= 1e-8:
                continue
            ligand = int(spec['parameter']) % LIGAND_COUNT
            channel = int(spec['regulator']) % CONTROL_COUNT
            genetic_weight, affinity, adaptation_rate, plasticity, exploration = _sensor_parameters(spec)
            raw = self.last_ligand_profiles[ligand]
            receptor_capacity = protein * spec['promoter'] * spec['efficiency'] / 0.016
            occupied = raw / (0.018 / max(0.45, affinity) + raw)
            occupied = np.clip(occupied * closure, 0.0, 1.5)
            mean_occupied = float(np.mean(occupied))
            baseline = float(self.receptor_baseline.get(fingerprint, 0.0))
            if config.receptor_adaptation:
                alpha = 1.0 - math.exp(-adaptation_rate * dt)
                baseline += alpha * (mean_occupied - baseline)
                self.receptor_baseline[fingerprint] = baseline
            centred = occupied - baseline
            gradient = np.sum(centred[:, None] * normals, axis=0) / MEMBRANE_SEGMENTS
            gradient *= receptor_capacity
            scalar = float(np.mean(centred) * receptor_capacity)
            gradient_strength = float(np.linalg.norm(gradient))
            activation = gradient_strength + abs(scalar)
            self.sensor_activation_by_gene[fingerprint] = activation
            self.last_receptor_activity[ligand] += activation
            ligand_activation[ligand] += activation

            noise = float(self.controller_noise.get(fingerprint, 0.0))
            if config.causal_perturbation and config.plasticity:
                noise += (-1.35 * noise * dt + world.rng.normal(0.0, 0.55) * math.sqrt(dt))
                noise = clamp(noise, -2.5, 2.5)
            else:
                noise *= math.exp(-1.35 * dt)
            self.controller_noise[fingerprint] = noise

            if config.fixed_reflex:
                weight = self._fixed_reflex_weight(ligand, self.has_novel_path())
                perturbation = 0.0
            else:
                modification = float(self.controller_phosphorylation.get(fingerprint, 0.0))
                perturbation = exploration * noise if config.causal_perturbation else 0.0
                weight = clamp(genetic_weight + modification + perturbation, -2.2, 2.2)

            self.last_control_vectors[channel] += weight * gradient
            self.last_control_scalars[channel] += weight * scalar

            eligibility = float(self.controller_eligibility.get(fingerprint, 0.0))
            eligibility *= math.exp(-0.18 * dt)
            eligibility += perturbation * (gradient_strength + 0.45 * abs(scalar))
            self.controller_eligibility[fingerprint] = clamp(eligibility, -3.0, 3.0)
            self.last_gene_perturbation[fingerprint] = float(perturbation)
            total_signal_cost += dt * (
                0.00016 * protein
                + 0.00044 * activation
                + 0.00010 * abs(noise) * protein
            )

        # Clamp control without changing its direction.
        for channel in range(CONTROL_COUNT):
            norm = float(np.linalg.norm(self.last_control_vectors[channel]))
            if norm > 1.5:
                self.last_control_vectors[channel] *= 1.5 / norm
            self.last_control_scalars[channel] = clamp(
                self.last_control_scalars[channel], -1.5, 1.5
            )
        for ligand in range(LIGAND_COUNT):
            profile = self.last_ligand_profiles[ligand]
            self.last_ligand_gradients[ligand] = (
                np.sum((profile - np.mean(profile))[:, None] * normals, axis=0)
                / MEMBRANE_SEGMENTS
            )
            # Meaning is based on covariance, not on a permanently positive
            # occupancy trace.  A slow ligand-specific baseline separates a
            # newly informative encounter from the fact that all patches exist.
            baseline = float(self.ligand_activation_baseline[ligand])
            deviation = float(ligand_activation[ligand] - baseline)
            baseline += (1.0 - math.exp(-0.035 * dt)) * deviation
            self.ligand_activation_baseline[ligand] = baseline
            self.meaning_trace[ligand] = (
                self.meaning_trace[ligand] * math.exp(-0.12 * dt)
                + clamp(deviation, -2.0, 2.0) * dt * 0.75
            )

        if config.sensor_cost:
            requested = total_signal_cost * config.sensor_cost_scale
            paid = min(max(0.0, self.pools[POOL_ATP] - 0.018), requested)
            self.pools[POOL_ATP] -= paid
            world.dissipated_energy += paid
            self.last_sensor_atp = paid
            self.cumulative_sensor_atp += paid
        self.sensorimotor_steps += 1

    # ------------------------------------------------------------------
    # ATP-paid membrane effectors
    # ------------------------------------------------------------------

    def _effector_activity(self, effect, channel=None):
        total = 0.0
        weighted_geometry = 0.0
        for fingerprint, spec in self.effector_specs():
            if int(spec['parameter']) % 8 != int(effect):
                continue
            if channel is not None and int(spec['regulator']) % CONTROL_COUNT != int(channel):
                continue
            amount = max(0.0, float(self.proteins.get(fingerprint, 0.0)))
            gain, rate, cost, geometry = _effector_parameters(spec)
            activity = amount * spec['promoter'] * spec['efficiency'] * gain / 0.018
            total += activity
            weighted_geometry += activity * geometry
        geometry = weighted_geometry / total if total > 1e-10 else 0.0
        return float(total), float(geometry)

    def _pay_atp(self, world, requested, reserve=0.018):
        paid = min(max(0.0, self.pools[POOL_ATP] - reserve), max(0.0, requested))
        self.pools[POOL_ATP] -= paid
        world.dissipated_energy += paid
        return float(paid)

    def _apply_transporter_polarity(self, world, dt, config, direction):
        self.last_polarity_atp = 0.0
        if not config.transporter_polarity:
            return
        activity, geometry = self._effector_activity(EFFECT_TRANSPORT_POLARITY, CONTROL_MOTOR)
        norm = float(np.linalg.norm(direction))
        if activity <= 1e-7 or norm <= 1e-8:
            return
        direction = direction / norm
        angles = 2.0 * math.pi * (np.arange(MEMBRANE_SEGMENTS) + 0.5) / MEMBRANE_SEGMENTS
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        projection = normals.dot(direction)
        sharpness = 1.3 + 1.2 * abs(geometry)
        front = np.exp(sharpness * projection)
        rear = np.exp(-sharpness * projection)
        front /= float(np.sum(front))
        rear /= float(np.sum(rear))
        blend_request = clamp(dt * 0.040 * activity * norm, 0.0, 0.16)
        movement_measure = 0.0
        proposals = self.transporters.copy()
        for channel in range(CHANNEL_COUNT):
            total = float(np.sum(self.transporters[:, channel]))
            if total <= 1e-12:
                continue
            target = rear if channel == CHANNEL_WASTE else front
            proposed = (1.0 - blend_request) * self.transporters[:, channel] + blend_request * total * target
            movement_measure += 0.5 * float(np.sum(np.abs(proposed - self.transporters[:, channel])))
            proposals[:, channel] = proposed
        requested_atp = movement_measure * 0.42 * config.motor_cost_scale
        if requested_atp <= 0.0:
            return
        paid = self._pay_atp(world, requested_atp)
        fraction = paid / requested_atp if requested_atp > 0.0 else 0.0
        self.transporters += fraction * (proposals - self.transporters)
        self.transporters = np.maximum(self.transporters, 0.0)
        self.last_polarity_atp = paid
        self.cumulative_polarity_atp += paid

    def apply_effectors(self, world, dt, config):
        self.last_motor_atp = 0.0
        self.last_motor_force = 0.0
        self.last_motor_command[:] = 0.0
        self.behavioural_quiescence = 0.0
        if not config.sensorimotor:
            return

        command = self.last_control_vectors[CONTROL_MOTOR].copy()
        if config.random_motor:
            command = world.rng.normal(0.0, 1.0, 2)
        norm = float(np.linalg.norm(command))
        if norm > 1e-10:
            command /= max(1.0, norm)
        self.last_motor_command = command.copy()

        # Existing transporter protein is physically relocated before uptake.
        self._apply_transporter_polarity(world, dt, config, command)

        if config.active_motion:
            activity, geometry = self._effector_activity(EFFECT_MOTOR, CONTROL_MOTOR)
            if activity > 1e-7 and norm > 1e-8:
                command_norm = min(1.0, norm)
                requested = (
                    dt * 0.0065 * activity * command_norm ** 2
                    * config.motor_cost_scale
                ) if config.motor_cost else 0.0
                if config.motor_cost:
                    paid = self._pay_atp(world, requested)
                    affordability = paid / requested if requested > 1e-12 else 0.0
                else:
                    paid = 0.0
                    affordability = 1.0
                force = (
                    0.0105 * activity * command_norm * affordability
                    * config.motor_gain_scale
                    / max(0.72, self.radius / BASE_RADIUS)
                )
                # ATP-driven cortical flow / membrane contraction.  Energy is
                # dissipated above; no material is created or teleported.
                self.surface_flux += command * force
                self.last_motor_force = force
                self.last_motor_atp = paid
                self.cumulative_motor_atp += paid
                self.cumulative_active_distance += force * dt
                # Sensorimotor work increases chemical wear by converting a
                # small amount of existing inert waste into reactive matter.
                converted = min(
                    self.pools[POOL_WASTE],
                    paid * (0.035 + 0.020 * abs(geometry)),
                )
                self.pools[POOL_WASTE] -= converted
                self.pools[POOL_REACTIVE] += converted
                self.last_sensorimotor_reactive = converted

        if config.quiescence_effector:
            activity, _ = self._effector_activity(EFFECT_QUIESCENCE, CONTROL_QUIESCENCE)
            scalar = max(0.0, float(self.last_control_scalars[CONTROL_QUIESCENCE]))
            self.behavioural_quiescence = clamp(
                scalar * activity / (0.8 + activity), 0.0, 0.72
            )

    def quiescence_level(self, config):
        inherited = super(SensorimotorProtoCell, self).quiescence_level(config)
        if not getattr(config, 'quiescence_effector', False):
            return inherited
        return float(clamp(max(inherited, self.behavioural_quiescence), 0.0, 0.86))

    # ------------------------------------------------------------------
    # No external reward: viability change modulates labile couplings
    # ------------------------------------------------------------------

    def autopoietic_components(self):
        closure = clamp(self.closure(), 0.015, 1.0)
        energy = clamp(self.pools[POOL_ATP] / (0.10 + self.pools[POOL_ATP]), 0.015, 1.0)
        loop = clamp(self.reaction_loop_strength() / 0.50, 0.015, 1.0)
        proteostasis = clamp(self.proteostasis_factor(), 0.015, 1.0)
        heredity = clamp(self.genome_function_factor(), 0.015, 1.0)
        boundary_retention = clamp(
            math.exp(-3.0 * self.last_leak) * math.exp(-0.55 * self.tension()),
            0.015, 1.0,
        )
        return np.asarray([
            closure, energy, loop, proteostasis, heredity, boundary_retention,
        ], dtype=float)

    def autopoietic_margin(self):
        components = self.autopoietic_components()
        # Geometric aggregation behaves like a soft bottleneck: improving an
        # already abundant component cannot fully hide collapse elsewhere.
        return float(math.exp(float(np.mean(np.log(np.clip(components, 1e-6, 1.0))))))

    def _update_sensorimotor_learning(self, world, dt, config):
        current = self.autopoietic_margin()
        if self.previous_autopoietic_margin <= 0.0:
            self.previous_autopoietic_margin = current
        raw_drift = (current - self.previous_autopoietic_margin) / max(dt, 1e-9)
        surprise = raw_drift - self.expected_margin_drift
        self.expected_margin_drift += (1.0 - math.exp(-0.025 * dt)) * (
            raw_drift - self.expected_margin_drift
        )
        modulator = math.tanh(9.0 * surprise)
        self.last_modulator = float(modulator)
        self.last_margin = float(current)
        self.previous_autopoietic_margin = current
        self.last_plasticity_atp = 0.0

        # Functional meaning is an empirical covariance between a sensed
        # ligand and later self-production change; it is not a ligand name table.
        old_sign = np.sign(self.meaning_mean.copy())
        for ligand in range(LIGAND_COUNT):
            evidence = clamp(
                self.meaning_trace[ligand] + 1.20 * self.uptake_trace[ligand],
                -2.5, 2.5,
            )
            alpha = 1.0 - math.exp(-0.08 * abs(evidence) * dt)
            observation = clamp(modulator * evidence, -1.0, 1.0)
            self.meaning_mean[ligand] += alpha * (observation - self.meaning_mean[ligand])
            self.meaning_confidence[ligand] = clamp(
                self.meaning_confidence[ligand]
                + dt * 0.035 * abs(evidence) * (1.0 - self.meaning_confidence[ligand])
                - dt * 0.0025 * self.meaning_confidence[ligand],
                0.0, 1.0,
            )
        new_sign = np.sign(self.meaning_mean)
        sign_changed = (
            (old_sign != 0.0) & (new_sign != 0.0) & (old_sign != new_sign)
            & (self.meaning_confidence > 0.18)
        )
        self.meaning_reversals += int(np.sum(sign_changed))

        if (
            not config.sensorimotor or not config.plasticity
            or config.fixed_reflex
        ):
            return
        updates = []
        requested_atp = 0.0
        for fingerprint, spec in self.sensor_specs():
            _, _, _, plasticity, _ = _sensor_parameters(spec)
            eligibility = float(self.controller_eligibility.get(fingerprint, 0.0))
            delta = (
                config.learning_rate_scale * plasticity
                * modulator * eligibility * dt * 2.8
            )
            delta = clamp(delta, -0.022, 0.022)
            if abs(delta) <= 1e-10:
                continue
            updates.append((fingerprint, delta))
            requested_atp += abs(delta) * 0.022
        if not updates:
            return
        if config.plasticity_cost:
            paid = self._pay_atp(world, requested_atp, reserve=0.018)
            fraction = paid / requested_atp if requested_atp > 1e-12 else 0.0
        else:
            paid = 0.0
            fraction = 1.0
        for fingerprint, delta in updates:
            old = float(self.controller_phosphorylation.get(fingerprint, 0.0))
            self.controller_phosphorylation[fingerprint] = clamp(
                old + fraction * delta, -1.65, 1.65
            )
            if fraction > 0.0:
                self.controller_updates += 1
        self.last_plasticity_atp = paid
        self.cumulative_plasticity_atp += paid

    def surface_exchange(self, field, dt, config):
        before_fuel = float(self.pools[POOL_FUEL])
        before_waste = float(self.pools[POOL_WASTE])
        super(SensorimotorProtoCell, self).surface_exchange(field, dt, config)
        world = getattr(field, '_world_ref', None)
        if world is None:
            return
        acquired = max(0.0, float(self.pools[POOL_FUEL]) - before_fuel)
        uptake = np.zeros(LIGAND_COUNT, dtype=float)
        uptake[LIGAND_FUEL] = max(0.0, float(self.last_uptake[0]))
        uptake[LIGAND_MINERAL] = max(0.0, float(self.last_uptake[1]))
        uptake[LIGAND_ALT] = max(0.0, float(self.last_uptake_alt))
        uptake[LIGAND_WASTE] = max(0.0, float(self.pools[POOL_WASTE]) - before_waste)
        self.last_uptake_by_ligand = uptake
        self.cumulative_uptake_by_ligand += uptake
        self.uptake_trace *= math.exp(-0.13 * dt)
        # Actual membrane crossing carries stronger temporal credit than merely
        # detecting a distant patch.  The scale is unit conversion, not valence.
        self.uptake_trace += np.clip(uptake * 70.0, 0.0, 1.5)
        for fingerprint, spec in self.sensor_specs():
            ligand = int(spec['parameter']) % LIGAND_COUNT
            perturbation = float(self.last_gene_perturbation.get(fingerprint, 0.0))
            if perturbation != 0.0 and uptake[ligand] > 0.0:
                eligibility = float(self.controller_eligibility.get(fingerprint, 0.0))
                eligibility += perturbation * min(1.2, uptake[ligand] * 95.0)
                self.controller_eligibility[fingerprint] = clamp(eligibility, -3.0, 3.0)
        if (
            acquired > 0.0
            and world.config.environment_mode == 'reversal'
            and world.age >= world.config.switch_age
        ):
            contaminated = min(
                self.pools[POOL_FUEL],
                acquired * clamp(world.config.contaminated_fraction, 0.0, 1.0),
            )
            self.pools[POOL_FUEL] -= contaminated
            self.pools[POOL_REACTIVE] += contaminated
            self.damage_trace += contaminated / MEMBRANE_SEGMENTS * 1.8
            self.fuel_contamination_received += contaminated
            world.fuel_contamination_total += contaminated

    def metabolism(self, world, dt, config):
        if not self.alive:
            return
        position_before = self.pos.copy()
        super(SensorimotorProtoCell, self).metabolism(world, dt, config)
        if not self.alive:
            return
        displacement = float(np.linalg.norm(wrapped_delta(position_before, self.pos)))
        active_estimate = min(displacement, self.last_motor_force * dt * 0.15)
        self.cumulative_passive_distance += max(0.0, displacement - active_estimate)
        self._update_sensorimotor_learning(world, dt, config)

    def update_division(self, dt, config):
        """Material septum assembly rebalanced for receptor/effector proteome.

        The checkpoint remains inherited from 0.3 and still requires two
        complete material genomes, membrane surplus, catalyst, precursor and
        total matter.  Only the ATP reserve and assembly kinetics are adjusted
        so the added 0.4 apparatus is not an irreversible sterility tax.
        """
        target_septum = 0.108
        if not config.division:
            returned = min(self.septum_mass, 0.006 * dt)
            self.septum_mass -= returned
            self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / target_septum, 0.0, 1.0)
            return
        if self.ready_for_division() or self.division_progress > 0.0:
            reserve = 0.008
            atp_cost = 0.28
            precursor_use = min(
                self.pools[POOL_MEM_PRECURSOR],
                0.018 * dt,
                max(0.0, self.pools[POOL_ATP] - reserve) / atp_cost,
            )
            if precursor_use > 0.0 and self.closure() > 0.89:
                self.pools[POOL_MEM_PRECURSOR] -= precursor_use
                self.pools[POOL_ATP] -= atp_cost * precursor_use
                self.septum_mass += precursor_use
            else:
                returned = min(self.septum_mass, 0.0032 * dt)
                self.septum_mass -= returned
                self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / target_septum, 0.0, 1.0)
        else:
            returned = min(self.septum_mass, 0.0030 * dt)
            self.septum_mass -= returned
            self.pools[POOL_MEM_PRECURSOR] += returned
            self.division_progress = clamp(self.septum_mass / target_septum, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Division and material-state inheritance
    # ------------------------------------------------------------------

    def split(self, world):
        parent_baseline = dict(self.receptor_baseline)
        parent_modification = dict(self.controller_phosphorylation)
        parent_meaning = self.meaning_mean.copy()
        parent_confidence = self.meaning_confidence.copy()
        daughters = super(SensorimotorProtoCell, self).split(world)
        if daughters is None:
            return None
        for index, daughter in enumerate(daughters):
            daughter.__class__ = SensorimotorProtoCell
            daughter._init_sensorimotor_state()
            daughter._refresh_gene_cache()
            valid = set(
                fingerprint for fingerprint, spec in daughter.gene_specs.items()
                if spec['role'] == ROLE_REGULATOR and spec['localisation'] == LOC_SENSOR
            )
            if world.config.inherited_controller_state:
                for fingerprint in valid:
                    if fingerprint in parent_baseline:
                        daughter.receptor_baseline[fingerprint] = parent_baseline[fingerprint]
                    if fingerprint in parent_modification:
                        # Phosphorylation is a labile state of inherited protein,
                        # not a free copy of a neural weight.
                        daughter.controller_phosphorylation[fingerprint] = (
                            0.58 * parent_modification[fingerprint]
                        )
                daughter.meaning_mean = parent_meaning * 0.42
                daughter.meaning_confidence = parent_confidence * 0.36
            daughter.previous_autopoietic_margin = daughter.autopoietic_margin()
            daughter.last_margin = daughter.previous_autopoietic_margin
            daughter.last_position_for_distance = daughter.pos.copy()
        return daughters

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self):
        state = super(SensorimotorProtoCell, self).state_dict()
        # Preserve the live insertion order of protein-state maps.  The
        # inherited serializers sort these maps for readability, but repair
        # fluxes use floating-point reductions whose final bits depend on
        # iteration order.  Keeping the material map order gives exact
        # save/restore continuation rather than merely numerical closeness.
        state['proteins'] = [
            (int(key), float(value)) for key, value in self.proteins.items()
        ]
        state['damaged_proteins'] = [
            (int(key), float(value)) for key, value in self.damaged_proteins.items()
        ]
        state.update({
            'cell_class': 'SensorimotorProtoCell',
            'receptor_baseline': [(int(k), float(v)) for k, v in self.receptor_baseline.items()],
            'controller_phosphorylation': [
                (int(k), float(v)) for k, v in self.controller_phosphorylation.items()
            ],
            'controller_eligibility': [
                (int(k), float(v)) for k, v in self.controller_eligibility.items()
            ],
            'controller_noise': [(int(k), float(v)) for k, v in self.controller_noise.items()],
            'sensor_activation_by_gene': [
                (int(k), float(v)) for k, v in self.sensor_activation_by_gene.items()
            ],
            'last_gene_perturbation': [
                (int(k), float(v)) for k, v in self.last_gene_perturbation.items()
            ],
            'meaning_mean': self.meaning_mean.copy(),
            'meaning_confidence': self.meaning_confidence.copy(),
            'meaning_trace': self.meaning_trace.copy(),
            'uptake_trace': self.uptake_trace.copy(),
            'ligand_activation_baseline': self.ligand_activation_baseline.copy(),
            'last_uptake_by_ligand': self.last_uptake_by_ligand.copy(),
            'cumulative_uptake_by_ligand': self.cumulative_uptake_by_ligand.copy(),
            'last_ligand_profiles': self.last_ligand_profiles.copy(),
            'last_ligand_gradients': self.last_ligand_gradients.copy(),
            'last_receptor_activity': self.last_receptor_activity.copy(),
            'last_control_vectors': self.last_control_vectors.copy(),
            'last_control_scalars': self.last_control_scalars.copy(),
            'last_motor_command': self.last_motor_command.copy(),
            'last_motor_force': float(self.last_motor_force),
            'last_sensor_atp': float(self.last_sensor_atp),
            'last_motor_atp': float(self.last_motor_atp),
            'last_polarity_atp': float(self.last_polarity_atp),
            'last_plasticity_atp': float(self.last_plasticity_atp),
            'last_sensorimotor_reactive': float(self.last_sensorimotor_reactive),
            'last_modulator': float(self.last_modulator),
            'last_margin': float(self.last_margin),
            'previous_autopoietic_margin': float(self.previous_autopoietic_margin),
            'expected_margin_drift': float(self.expected_margin_drift),
            'behavioural_quiescence': float(self.behavioural_quiescence),
            'repair_polarity': self.repair_polarity.copy(),
            'cumulative_sensor_atp': float(self.cumulative_sensor_atp),
            'cumulative_motor_atp': float(self.cumulative_motor_atp),
            'cumulative_polarity_atp': float(self.cumulative_polarity_atp),
            'cumulative_plasticity_atp': float(self.cumulative_plasticity_atp),
            'cumulative_active_distance': float(self.cumulative_active_distance),
            'cumulative_passive_distance': float(self.cumulative_passive_distance),
            'controller_updates': int(self.controller_updates),
            'meaning_reversals': int(self.meaning_reversals),
            'fuel_contamination_received': float(self.fuel_contamination_received),
            'sensorimotor_steps': int(self.sensorimotor_steps),
            'last_position_for_distance': self.last_position_for_distance.copy(),
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = g3.DamageProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        cell._init_sensorimotor_state()
        for name in (
            'receptor_baseline', 'controller_phosphorylation',
            'controller_eligibility', 'controller_noise',
            'sensor_activation_by_gene', 'last_gene_perturbation',
        ):
            setattr(cell, name, {
                int(k): float(v) for k, v in state.get(name, [])
            })
        for name, shape in (
            ('meaning_mean', (LIGAND_COUNT,)),
            ('meaning_confidence', (LIGAND_COUNT,)),
            ('meaning_trace', (LIGAND_COUNT,)),
            ('uptake_trace', (LIGAND_COUNT,)),
            ('ligand_activation_baseline', (LIGAND_COUNT,)),
            ('last_uptake_by_ligand', (LIGAND_COUNT,)),
            ('cumulative_uptake_by_ligand', (LIGAND_COUNT,)),
            ('last_ligand_profiles', (LIGAND_COUNT, MEMBRANE_SEGMENTS)),
            ('last_ligand_gradients', (LIGAND_COUNT, 2)),
            ('last_receptor_activity', (LIGAND_COUNT,)),
            ('last_control_vectors', (CONTROL_COUNT, 2)),
            ('last_control_scalars', (CONTROL_COUNT,)),
            ('last_motor_command', (2,)),
            ('repair_polarity', (MEMBRANE_SEGMENTS,)),
            ('last_position_for_distance', (2,)),
        ):
            value = np.asarray(state.get(name, np.zeros(shape)), dtype=float)
            setattr(cell, name, value.reshape(shape).copy())
        for name in (
            'last_motor_force', 'last_sensor_atp', 'last_motor_atp',
            'last_polarity_atp', 'last_plasticity_atp',
            'last_sensorimotor_reactive', 'last_modulator', 'last_margin',
            'previous_autopoietic_margin', 'expected_margin_drift',
            'behavioural_quiescence', 'cumulative_sensor_atp',
            'cumulative_motor_atp', 'cumulative_polarity_atp',
            'cumulative_plasticity_atp', 'cumulative_active_distance',
            'cumulative_passive_distance', 'fuel_contamination_received',
        ):
            setattr(cell, name, float(state.get(name, getattr(cell, name))))
        for name in ('controller_updates', 'meaning_reversals', 'sensorimotor_steps'):
            setattr(cell, name, int(state.get(name, getattr(cell, name))))
        cell._clean_control_state()
        return cell


class SensorimotorWorld(g3.DamageWorld):
    """Patchy material world with an optional chemistry reversal."""

    def __init__(self, seed=101, initial_cells=1, config=None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = config if config is not None else SensorimotorConfig()
        self.age = 0.0
        self.field = SensorimotorParticleField(
            self.rng, initial=True, alt_niche=True,
            environment_mode=self.config.environment_mode,
        )
        self.field.switch_age = self.config.switch_age
        self.field._world_ref = self
        self.cells = []
        self.next_cell_id = 0
        for index in range(int(initial_cells)):
            angle = 2.0 * math.pi * index / max(1, int(initial_cells))
            radial = 0.0 if int(initial_cells) == 1 else 0.08
            position = np.array([
                0.50 + radial * math.cos(angle),
                0.50 + radial * math.sin(angle),
            ]) % 1.0
            self.cells.append(SensorimotorProtoCell(
                self.next_cell_id, self.rng, position=position,
                generation=0, lineage=index, bootstrap=True,
            ))
            self.next_cell_id += 1
        self.births = 0
        self.divisions = 0
        self.deaths = 0
        self.last_deaths = []
        self.last_births = []
        self.dissipated_energy = 0.0
        self.division_parent_material = 0.0
        self.division_daughter_material = 0.0
        self.division_shed_material = 0.0
        self.division_residual = 0.0
        self.released_dead_material = 0.0
        self.manual_injections = 0
        self.manual_punctures = 0
        self.damage_events = 0
        self.external_protein_assistance = 0.0
        self.first_novel_path_age = None
        self.novel_path_lineages = set()
        self.current_stress = 0.0
        self.stress_pulses = 0
        self._last_stress_high = False
        self.damage_segregation_events = 0
        self.rejuvenation_events = 0
        self.last_damage_partition = None
        self.damage_partition_log = []
        self.fuel_contamination_total = 0.0
        self.switch_events = 0
        self._switch_recorded = False
        self.last_patch_distances = np.zeros(4, dtype=float)
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def stress_profile(self, cell):
        angles = 2.0 * math.pi * (np.arange(MEMBRANE_SEGMENTS) + 0.5) / MEMBRANE_SEGMENTS
        points = (
            cell.pos[None, :]
            + np.stack([np.cos(angles), np.sin(angles)], axis=1)
            * (cell.radius + 0.045)
        ) % 1.0
        hazard = self.field.patch_centres[PARTICLE_WASTE]
        delta = (hazard[None, :] - points + 0.5) % 1.0 - 0.5
        distance = np.linalg.norm(delta, axis=1)
        spatial = np.exp(-(distance / 0.17) ** 2)
        return np.clip(0.15 * self.current_stress + 0.95 * spatial, 0.0, 1.5)

    def _particle_interactions(self, dt):
        # Sensing and physical protein relocation happen before uptake.
        self.field._world_ref = self
        for cell in self.living_cells():
            cell.sense_environment(self, dt, self.config)
            cell.apply_effectors(self, dt, self.config)
            cell.surface_exchange(self.field, dt, self.config)

    def step(self, dt):
        self.field.world_age = self.age
        self.field.switch_age = self.config.switch_age
        if (
            self.config.environment_mode == 'reversal'
            and self.age >= self.config.switch_age
            and not self._switch_recorded
        ):
            self.switch_events += 1
            self._switch_recorded = True
        super(SensorimotorWorld, self).step(dt)
        self.field._world_ref = self
        alive = self.living_cells()
        if alive:
            cell = alive[0]
            self.last_patch_distances = np.asarray([
                torus_distance(cell.pos, self.field.patch_centres[kind])
                for kind in range(4)
            ], dtype=float)

    def inject_cloud(self, position):
        self.field.add_cloud(position)
        self.manual_injections += 1

    def finite(self):
        if not super(SensorimotorWorld, self).finite():
            return False
        if not finite_array(self.field.patch_centres):
            return False
        for cell in self.cells:
            arrays = (
                cell.meaning_mean, cell.meaning_confidence, cell.meaning_trace,
                cell.uptake_trace, cell.ligand_activation_baseline,
                cell.last_uptake_by_ligand, cell.cumulative_uptake_by_ligand,
                cell.last_ligand_profiles, cell.last_ligand_gradients,
                cell.last_receptor_activity, cell.last_control_vectors,
                cell.last_control_scalars, cell.last_motor_command,
                cell.repair_polarity,
            )
            if any(not finite_array(value) for value in arrays):
                return False
            mappings = (
                cell.receptor_baseline, cell.controller_phosphorylation,
                cell.controller_eligibility, cell.controller_noise,
                cell.last_gene_perturbation,
            )
            if any(
                not np.isfinite(value)
                for mapping in mappings for value in mapping.values()
            ):
                return False
        return True

    def summary(self):
        summary = super(SensorimotorWorld, self).summary()
        alive = self.living_cells()
        if alive:
            summary.update({
                'mean_autopoietic_margin': float(np.mean([
                    cell.autopoietic_margin() for cell in alive
                ])),
                'mean_modulator': float(np.mean([cell.last_modulator for cell in alive])),
                'mean_motor_force': float(np.mean([cell.last_motor_force for cell in alive])),
                'sensor_atp_total': float(sum(cell.cumulative_sensor_atp for cell in alive)),
                'motor_atp_total': float(sum(cell.cumulative_motor_atp for cell in alive)),
                'polarity_atp_total': float(sum(cell.cumulative_polarity_atp for cell in alive)),
                'plasticity_atp_total': float(sum(cell.cumulative_plasticity_atp for cell in alive)),
                'controller_updates': int(sum(cell.controller_updates for cell in alive)),
                'meaning_reversals': int(sum(cell.meaning_reversals for cell in alive)),
                'meaning_fuel': float(np.mean([cell.meaning_mean[LIGAND_FUEL] for cell in alive])),
                'meaning_alt': float(np.mean([cell.meaning_mean[LIGAND_ALT] for cell in alive])),
                'meaning_waste': float(np.mean([cell.meaning_mean[LIGAND_WASTE] for cell in alive])),
                'confidence_fuel': float(np.mean([
                    cell.meaning_confidence[LIGAND_FUEL] for cell in alive
                ])),
                'confidence_alt': float(np.mean([
                    cell.meaning_confidence[LIGAND_ALT] for cell in alive
                ])),
                'control_fuel': float(np.mean([
                    cell._mean_controller_weight_for_ligand(LIGAND_FUEL, self.config)
                    for cell in alive
                ])),
                'control_alt': float(np.mean([
                    cell._mean_controller_weight_for_ligand(LIGAND_ALT, self.config)
                    for cell in alive
                ])),
                'control_waste': float(np.mean([
                    cell._mean_controller_weight_for_ligand(LIGAND_WASTE, self.config)
                    for cell in alive
                ])),
                'active_distance_total': float(sum(
                    cell.cumulative_active_distance for cell in alive
                )),
                'fuel_uptake_total': float(sum(
                    cell.cumulative_uptake_by_ligand[LIGAND_FUEL] for cell in alive
                )),
                'mineral_uptake_total': float(sum(
                    cell.cumulative_uptake_by_ligand[LIGAND_MINERAL] for cell in alive
                )),
                'alt_uptake_total': float(sum(
                    cell.cumulative_uptake_by_ligand[LIGAND_ALT] for cell in alive
                )),
                'waste_ingress_total': float(sum(
                    cell.cumulative_uptake_by_ligand[LIGAND_WASTE] for cell in alive
                )),
                'mean_receptor_genes': float(np.mean([
                    len(cell.sensor_specs()) for cell in alive
                ])),
                'mean_effector_genes': float(np.mean([
                    len(cell.effector_specs()) for cell in alive
                ])),
            })
        else:
            summary.update({
                'mean_autopoietic_margin': 0.0, 'mean_modulator': 0.0,
                'mean_motor_force': 0.0, 'sensor_atp_total': 0.0,
                'motor_atp_total': 0.0, 'polarity_atp_total': 0.0,
                'plasticity_atp_total': 0.0, 'controller_updates': 0,
                'meaning_reversals': 0, 'meaning_fuel': 0.0,
                'meaning_alt': 0.0, 'meaning_waste': 0.0,
                'confidence_fuel': 0.0, 'confidence_alt': 0.0,
                'control_fuel': 0.0, 'control_alt': 0.0,
                'control_waste': 0.0, 'active_distance_total': 0.0,
                'fuel_uptake_total': 0.0, 'mineral_uptake_total': 0.0,
                'alt_uptake_total': 0.0, 'waste_ingress_total': 0.0,
                'mean_receptor_genes': 0.0, 'mean_effector_genes': 0.0,
            })
        summary.update({
            'build': BUILD,
            'environment_mode': self.config.environment_mode,
            'switch_age': float(self.config.switch_age),
            'switch_events': int(self.switch_events),
            'fuel_contamination_total': float(self.fuel_contamination_total),
            'distance_fuel_patch': float(self.last_patch_distances[PARTICLE_FUEL]),
            'distance_mineral_patch': float(self.last_patch_distances[PARTICLE_MINERAL]),
            'distance_waste_patch': float(self.last_patch_distances[PARTICLE_WASTE]),
            'distance_alt_patch': float(self.last_patch_distances[PARTICLE_ALT]),
        })
        return summary

    def state_dict(self):
        state = super(SensorimotorWorld, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'field': self.field.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'fuel_contamination_total': float(self.fuel_contamination_total),
            'switch_events': int(self.switch_events),
            '_switch_recorded': bool(self._switch_recorded),
            'last_patch_distances': self.last_patch_distances.copy(),
        })
        return state

    @classmethod
    def from_state(cls, state):
        world = cls(
            seed=int(state['seed']), initial_cells=0,
            config=SensorimotorConfig.from_state(state.get('config', {})),
        )
        world.rng.bit_generator.state = state['rng_state']
        world.age = float(state['age'])
        world.field = SensorimotorParticleField.from_state(world.rng, state['field'])
        world.field._world_ref = world
        world.cells = [SensorimotorProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        for name in (
            'next_cell_id', 'births', 'divisions', 'deaths', 'manual_injections',
            'manual_punctures', 'damage_events', 'stress_pulses',
            'damage_segregation_events', 'rejuvenation_events', 'switch_events',
        ):
            setattr(world, name, int(state.get(name, getattr(world, name))))
        for name in (
            'dissipated_energy', 'division_parent_material', 'division_daughter_material',
            'division_shed_material', 'division_residual', 'released_dead_material',
            'external_protein_assistance', 'initial_total_material',
            'last_step_material_residual', 'current_stress',
            'fuel_contamination_total',
        ):
            setattr(world, name, float(state.get(name, getattr(world, name))))
        world.first_novel_path_age = state.get('first_novel_path_age')
        world.novel_path_lineages = set(state.get('novel_path_lineages', []))
        world.last_deaths = list(state.get('last_deaths', []))
        world.last_births = list(state.get('last_births', []))
        world._last_stress_high = bool(state.get('_last_stress_high', False))
        pair = state.get('last_damage_partition')
        world.last_damage_partition = None if pair is None else tuple(float(v) for v in pair)
        world.damage_partition_log = list(state.get('damage_partition_log', []))
        world._switch_recorded = bool(state.get('_switch_recorded', False))
        world.last_patch_distances = np.asarray(
            state.get('last_patch_distances', np.zeros(4)), dtype=float
        ).copy()
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return SensorimotorWorld.from_state(self.state_dict())


# Method defined after class for compactness in summary calculations.
def _mean_controller_weight_for_ligand(self, ligand, config=None):
    weights = []
    for fingerprint, spec in self.sensor_specs():
        if int(spec['parameter']) % LIGAND_COUNT != int(ligand):
            continue
        genetic, _, _, _, _ = _sensor_parameters(spec)
        if config is not None and config.fixed_reflex:
            weight = self._fixed_reflex_weight(ligand, self.has_novel_path())
        else:
            weight = genetic + self.controller_phosphorylation.get(fingerprint, 0.0)
        weights.append(weight)
    return float(np.mean(weights)) if weights else 0.0


SensorimotorProtoCell._mean_controller_weight_for_ligand = _mean_controller_weight_for_ligand


# ---------------------------------------------------------------------------
# Headless trial, long-run logging and report
# ---------------------------------------------------------------------------


def run_headless_trial(seed=101, seconds=300.0, initial_cells=1, config=None):
    world = SensorimotorWorld(
        seed=seed, initial_cells=initial_cells,
        config=config if config is not None else SensorimotorConfig(),
    )
    dt = 1.0 / SIM_HZ
    margin_integral = 0.0
    post_switch_margin = 0.0
    post_switch_steps = 0
    fuel_distance_integral = 0.0
    alt_distance_integral = 0.0
    living_steps = 0
    for _ in range(int(round(float(seconds) * SIM_HZ))):
        world.step(dt)
        summary = world.summary()
        if summary['cells'] <= 0:
            break
        margin_integral += summary['mean_autopoietic_margin'] * dt
        fuel_distance_integral += summary['distance_fuel_patch'] * dt
        alt_distance_integral += summary['distance_alt_patch'] * dt
        living_steps += 1
        if world.age >= world.config.switch_age:
            post_switch_margin += summary['mean_autopoietic_margin']
            post_switch_steps += 1
    result = world.summary()
    duration = max(dt, living_steps * dt)
    result.update({
        'requested_seconds': float(seconds),
        'completed_seconds': float(world.age),
        'margin_integral': float(margin_integral),
        'mean_margin_over_life': float(margin_integral / duration),
        'mean_fuel_patch_distance': float(fuel_distance_integral / duration),
        'mean_alt_patch_distance': float(alt_distance_integral / duration),
        'post_switch_mean_margin': float(
            post_switch_margin / max(1, post_switch_steps)
        ),
    })
    return result


LOG_FIELDS = (
    'wall_time', 'session_id', 'reason', 'build', 'seed', 'age', 'cells',
    'births', 'divisions', 'deaths', 'max_generation', 'environment_mode',
    'switch_events', 'stress', 'mean_autopoietic_margin', 'mean_atp',
    'mean_closure', 'mean_loop', 'mean_damage', 'mean_proteostasis',
    'meaning_fuel', 'meaning_alt', 'meaning_waste', 'confidence_fuel',
    'confidence_alt', 'control_fuel', 'control_alt', 'control_waste',
    'mean_motor_force', 'sensor_atp_total', 'motor_atp_total',
    'polarity_atp_total', 'plasticity_atp_total', 'controller_updates',
    'meaning_reversals', 'fuel_contamination_total', 'fuel_uptake_total',
    'mineral_uptake_total', 'alt_uptake_total', 'waste_ingress_total',
    'distance_fuel_patch',
    'distance_alt_patch', 'distance_waste_patch', 'total_material',
    'matter_residual', 'division_residual', 'fps', 'sim_rate', 'peak_mb',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), int(world.seed))
        self.last_age = -1e9
        self.status = 'READY'

    def log(self, world, fps=0.0, sim_rate=0.0, reason='interval', force=False):
        if not force and world.age - self.last_age < LOG_INTERVAL:
            return
        row = world.summary()
        row.update({
            'wall_time': time.time(), 'session_id': self.session_id,
            'reason': reason, 'fps': float(fps), 'sim_rate': float(sim_rate),
            'peak_mb': float(_memory_peak_mb_estimate()),
        })
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        try:
            with open(self.path, 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction='ignore')
                if not exists:
                    writer.writeheader()
                writer.writerow({name: row.get(name, '') for name in LOG_FIELDS})
            self.last_age = float(world.age)
            self.status = 'OK'
        except Exception:
            self.status = 'ERR'


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    try:
        with open(log_path, 'r', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return 'EMPTY'
        grouped = {}
        for row in rows:
            grouped.setdefault(row.get('session_id', 'unknown'), []).append(row)
        session_rows = []
        for session_id, items in grouped.items():
            last = items[-1]
            session_rows.append({
                'session_id': session_id,
                'start_age': items[0].get('age', ''),
                'end_age': last.get('age', ''),
                'final_cells': last.get('cells', ''),
                'final_generation': last.get('max_generation', ''),
                'environment_mode': last.get('environment_mode', ''),
                'final_margin': last.get('mean_autopoietic_margin', ''),
                'fuel_meaning': last.get('meaning_fuel', ''),
                'alt_meaning': last.get('meaning_alt', ''),
                'fuel_control': last.get('control_fuel', ''),
                'alt_control': last.get('control_alt', ''),
                'controller_updates': last.get('controller_updates', ''),
                'sensor_atp_total': last.get('sensor_atp_total', ''),
                'motor_atp_total': last.get('motor_atp_total', ''),
                'matter_residual': last.get('matter_residual', ''),
            })
        with open(session_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(session_rows[0].keys()))
            writer.writeheader()
            writer.writerows(session_rows)
        last = rows[-1]
        text_lines = [
            BUILD + ' long-run report',
            'sessions: {}'.format(len(grouped)),
            'last age / cells / generation: {} / {} / {}'.format(
                last.get('age'), last.get('cells'), last.get('max_generation')
            ),
            'environment / switches: {} / {}'.format(
                last.get('environment_mode'), last.get('switch_events')
            ),
            'autopoietic margin: {}'.format(last.get('mean_autopoietic_margin')),
            'functional meaning fuel / alt / waste: {} / {} / {}'.format(
                last.get('meaning_fuel'), last.get('meaning_alt'), last.get('meaning_waste')
            ),
            'controller fuel / alt / waste: {} / {} / {}'.format(
                last.get('control_fuel'), last.get('control_alt'), last.get('control_waste')
            ),
            'sensor / motor / polarity / plasticity ATP: {} / {} / {} / {}'.format(
                last.get('sensor_atp_total'), last.get('motor_atp_total'),
                last.get('polarity_atp_total'), last.get('plasticity_atp_total')
            ),
            'matter residual: {}'.format(last.get('matter_residual')),
            'Meaning values are learned covariances with self-production change, not labels.',
        ]
        with open(report_path, 'w') as handle:
            handle.write('\n'.join(text_lines) + '\n')
        return 'OK'
    except Exception:
        return 'WRITE ERR'


# ---------------------------------------------------------------------------
# Pythonista Scene
# ---------------------------------------------------------------------------


try:
    from scene import Scene, run, LANDSCAPE
    from scene import background, fill, stroke, stroke_weight, ellipse, line, rect, text

    class SomaCellSensorimotorScene(Scene):
        def setup(self):
            self.paused = False
            self.accumulator = 0.0
            self.last_wall = time.time()
            self.last_save_age = 0.0
            self.last_touch_wall = -10.0
            self.fps = 0.0
            self.sim_rate = 0.0
            self.telemetry_wall = time.time()
            self.telemetry_age = 0.0
            self.telemetry_frames = 0
            self.save_status = 'NEW'
            self.report_status = 'WAIT'
            try:
                if os.path.exists(SAVE_FILE):
                    self.world = SensorimotorWorld.load(SAVE_FILE)
                    self.save_status = 'LOAD'
                else:
                    self.world = SensorimotorWorld(seed=101, initial_cells=1)
            except Exception:
                self.world = SensorimotorWorld(seed=101, initial_cells=1)
                self.save_status = 'RECOVER'
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)

        def _world_rect(self):
            width, height = float(self.size.w), float(self.size.h)
            return 34.0, 78.0, width - 68.0, height - 164.0

        def _screen(self, position):
            left, bottom, width, height = self._world_rect()
            position = np.asarray(position, dtype=float) % 1.0
            return left + position[0] * width, bottom + position[1] * height

        def _unit_position(self, point):
            left, bottom, width, height = self._world_rect()
            return np.array([
                clamp((point.x - left) / max(width, 1.0), 0.0, 1.0),
                clamp((point.y - bottom) / max(height, 1.0), 0.0, 1.0),
            ])

        def update(self):
            now = time.time()
            wall_dt = clamp(now - self.last_wall, 0.0, 0.20)
            self.last_wall = now
            if not self.paused:
                self.accumulator += wall_dt
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

        def _draw_particles(self):
            colours = (
                (0.28, 0.90, 0.46),
                (0.96, 0.69, 0.20),
                (0.92, 0.25, 0.29),
                (0.20, 0.72, 1.00),
            )
            for index in range(len(self.world.field.amount)):
                x, y = self._screen(self.world.field.pos[index])
                amount = float(self.world.field.amount[index])
                radius = 1.4 + 7.2 * math.sqrt(clamp(amount / 0.045, 0.0, 1.7))
                colour = colours[int(self.world.field.kind[index])]
                fill(colour[0], colour[1], colour[2], 0.75)
                ellipse(x - radius, y - radius, 2 * radius, 2 * radius)

        def _draw_patch_halos(self):
            colours = (
                (0.22, 0.88, 0.42, 0.08),
                (0.96, 0.66, 0.18, 0.07),
                (0.92, 0.22, 0.25, 0.08),
                (0.18, 0.68, 1.00, 0.08),
            )
            for kind, centre in enumerate(self.world.field.patch_centres):
                x, y = self._screen(centre)
                fill(*colours[kind])
                ellipse(x - 34, y - 34, 68, 68)

        def _draw_cell(self, cell):
            left, bottom, width, height = self._world_rect()
            cx, cy = self._screen(cell.pos)
            scale = min(width, height)
            sr = cell.radius * scale
            closure = cell.closure_array()
            damage = clamp(cell.damage_burden(), 0.0, 1.0)
            margin = clamp(cell.autopoietic_margin(), 0.0, 1.0)
            fill(0.08 + 0.30 * damage, 0.20 + 0.48 * margin, 0.42 - 0.20 * damage, 0.32)
            ellipse(cx - sr, cy - sr, 2 * sr, 2 * sr)

            points = cell.boundary_points()
            receptor_activity = cell.last_receptor_activity
            dominant = int(np.argmax(receptor_activity)) if np.max(receptor_activity) > 1e-8 else -1
            receptor_colours = (
                (0.30, 1.00, 0.48), (1.00, 0.72, 0.22),
                (0.22, 0.76, 1.00), (1.00, 0.28, 0.30),
                (0.95, 0.40, 0.95), (0.75, 0.75, 0.82),
                (1.00, 0.92, 0.36), (1.00, 0.46, 0.22),
            )
            for index in range(MEMBRANE_SEGMENTS):
                nxt = (index + 1) % MEMBRANE_SEGMENTS
                if closure[index] < 0.055 and closure[nxt] < 0.055:
                    continue
                p0 = points[index]
                p1 = p0 + wrapped_delta(p0, points[nxt])
                x0, y0 = self._screen(p0)
                x1 = left + p1[0] * width
                y1 = bottom + p1[1] * height
                local = 0.5 * (closure[index] + closure[nxt])
                oxidation = clamp(0.5 * (
                    cell.membrane_oxidation[index] + cell.membrane_oxidation[nxt]
                ), 0.0, 1.0)
                if dominant >= 0:
                    rc = receptor_colours[dominant]
                    mix = clamp(cell.last_ligand_profiles[dominant, index], 0.0, 1.0) * 0.35
                else:
                    rc = (0.35, 0.80, 0.95)
                    mix = 0.0
                stroke(
                    (0.34 + 0.60 * oxidation) * (1.0 - mix) + rc[0] * mix,
                    (0.78 - 0.50 * oxidation) * (1.0 - mix) + rc[1] * mix,
                    (0.95 - 0.58 * oxidation) * (1.0 - mix) + rc[2] * mix,
                    0.30 + 0.70 * local,
                )
                stroke_weight(0.6 + 3.2 * local)
                line(x0, y0, x1, y1)

            command = cell.last_motor_command
            if float(np.linalg.norm(command)) > 1e-5:
                stroke(0.98, 0.94, 0.42, 0.90)
                stroke_weight(2.0)
                line(cx, cy, cx + command[0] * sr * 1.7, cy + command[1] * sr * 1.7)

            # Meaning petals: green/blue/red correspond to fuel/alt/waste.
            for offset, ligand, colour in (
                (-0.55, LIGAND_FUEL, (0.28, 0.95, 0.44)),
                (0.00, LIGAND_ALT, (0.20, 0.72, 1.00)),
                (0.55, LIGAND_WASTE, (0.96, 0.28, 0.30)),
            ):
                value = clamp(cell.meaning_mean[ligand], -1.0, 1.0)
                confidence = cell.meaning_confidence[ligand]
                x = cx + offset * sr
                y = cy + sr * 0.42
                r = 1.5 + 3.5 * confidence
                if value >= 0.0:
                    fill(colour[0], colour[1], colour[2], 0.35 + 0.55 * confidence)
                else:
                    fill(1.0, 0.35, 0.85, 0.35 + 0.55 * confidence)
                ellipse(x - r, y - r, 2 * r, 2 * r)

            fill(0.91, 0.97, 1.0)
            text('#{} G{} M{:.2f} D{:.2f}'.format(
                cell.cell_id, cell.generation,
                cell.autopoietic_margin(), cell.damage_burden()
            ), x=cx, y=cy - sr - 10, font_size=8, alignment=5)

        def draw(self):
            background(0.010, 0.020, 0.034)
            left, bottom, width, height = self._world_rect()
            fill(0.020, 0.042, 0.058)
            rect(left, bottom, width, height)
            self._draw_patch_halos()
            self._draw_particles()
            for cell in self.world.living_cells():
                self._draw_cell(cell)
            s = self.world.summary()
            fill(0.92, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.64, 0.78, 0.86)
            text('SCENE ACTIVE | SAVE {} | LOG {} | REPORT {} | {:.1f} fps | x{:.2f}'.format(
                self.save_status, self.logger.status, self.report_status,
                self.fps, self.sim_rate
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} div {} deaths {} G{} mode {} switch {}'.format(
                s['age'], s['cells'], s['divisions'], s['deaths'],
                s['max_generation'], s['environment_mode'], s['switch_events']
            ), x=24, y=56, font_size=10, alignment=4)
            text('margin {:.3f} ATP {:.3f} seal {:.3f} loop {:.3f} damage {:.3f}'.format(
                s['mean_autopoietic_margin'], s['mean_atp'], s['mean_closure'],
                s['mean_loop'], s['mean_damage']
            ), x=24, y=40, font_size=9, alignment=4)
            text('meaning F/A/W {:+.2f}/{:+.2f}/{:+.2f}  control {:+.2f}/{:+.2f}/{:+.2f}'.format(
                s['meaning_fuel'], s['meaning_alt'], s['meaning_waste'],
                s['control_fuel'], s['control_alt'], s['control_waste']
            ), x=24, y=24, font_size=9, alignment=4)
            text('sensor/motor/polarity/plastic ATP {:.3f}/{:.3f}/{:.3f}/{:.3f} updates {} ledger {:+.2e}'.format(
                s['sensor_atp_total'], s['motor_atp_total'], s['polarity_atp_total'],
                s['plasticity_atp_total'], s['controller_updates'], s['matter_residual']
            ), x=24, y=9, font_size=8, alignment=4)
            if s['cells'] == 0:
                fill(1.0, 0.38, 0.32)
                text('SENSORIMOTOR AUTOPOIESIS EXTINCT — double tap to reseed',
                     x=self.size.w * 0.5, y=self.size.h * 0.52,
                     font_size=16, alignment=5)
            if self.paused:
                fill(1.0, 0.92, 0.45)
                text('PAUSED', x=self.size.w * 0.5,
                     y=self.size.h - 26, font_size=13, alignment=5)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = SensorimotorWorld(seed=101, initial_cells=1)
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
                    reason='pause' if self.paused else 'resume', force=True
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
        print(run_headless_trial(seed=101, seconds=20.0, initial_cells=1))
    else:
        run(SomaCellSensorimotorScene(), LANDSCAPE, show_fps=False)
