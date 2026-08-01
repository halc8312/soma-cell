# coding: utf-8
"""
SOMA-CELL 0.6-P1 — One Gene-Built Material Neuron
物質ゲノム由来の1神経区画と等費用ダミー対照

P1 is deliberately small.  It keeps the frozen SOMA-CELL 0.6-P0 chemical
body port and adds exactly one material neural compartment.  The compartment
is assembled from finite body ATP/protein/membrane/signal, reads only immutable
P0 sensor frames, and can act only through the paid P0 effector API.

The primary control is not a free or absent brain.  It is an equal-material,
equal-maintenance dummy compartment that executes the same update schedule and
issues the same-magnitude paid effector requests, but chooses direction from an
internal oscillator instead of environmental chemistry.

P1 does not yet implement recurrent connections, three-factor sensorimotor
plasticity, causal audit, or inherited learned weights.  Those remain blocked
until P1 establishes whether one materially paid information-processing cell
can outperform its equal-cost dummy in at least one non-stationary environment.
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
    import SOMA_CELL_0_6_P0_pythonista as p0
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _P0 = os.path.abspath(os.path.join(_HERE, '..', '0_6_p0'))
    _BASE = os.path.abspath(os.path.join(_HERE, '..', 'baseline'))
    for _candidate in (_P0, _BASE):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    import SOMA_CELL_0_6_P0_pythonista as p0

s5 = p0.s5
s4 = p0.s4
g2 = s5.g2

BUILD = 'SOMA-CELL 0.6-P1.0'
BUILD_LONG = 'SOMA-CELL 0.6-P1.0 One Gene-Built Material Neuron'
SAVE_VERSION = 61
P1_SCHEMA_VERSION = '0.6-P1.1'
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p1.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p1_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p1_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p1_sessions.csv')

SIM_HZ = p0.SIM_HZ
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0

TISSUE_NONE = 'none'
TISSUE_NEURON = 'neuron'
TISSUE_DUMMY = 'dummy'
TISSUE_NO_PREDICTION = 'no_prediction'
TISSUE_NO_EFFECTOR = 'no_effector'
TISSUE_MODES = frozenset((
    TISSUE_NONE, TISSUE_NEURON, TISSUE_DUMMY,
    TISSUE_NO_PREDICTION, TISSUE_NO_EFFECTOR,
))

ENV_NATIVE = 'native'
ENV_STABLE_PATCH = 'stable_patch'
ENV_MOVING_PATCH = 'moving_patch'
ENVIRONMENTS = frozenset((ENV_NATIVE, ENV_STABLE_PATCH, ENV_MOVING_PATCH))

P1_TISSUE_ID = 'p1-one-neuron'
P1_NEURAL_KIND = 'one-material-neuron'
P1_DUMMY_KIND = 'equal-cost-noninformational-dummy'

# The neural-development marker uses an already reserved 0.4 effector slot.
# Existing 0.4/0.5 body code translates and maintains the protein but has no
# physical effector bound to EFFECT_RESERVED, so P1 can interpret the product
# without stealing an existing body function or extending the gene alphabet.
NEURAL_GENE_EFFECT = s4.EFFECT_RESERVED
NEURAL_GENE_CHANNEL = s4.CONTROL_RESERVED_7
NEURAL_GENE_LOCALISATION = s4.LOC_EFFECTOR

# Explicit tissue target matter.  Both neuron and dummy use exactly these
# targets and the same ongoing wear schedule.
TARGET_FUNCTIONAL_PROTEIN = 0.032
TARGET_MEMBRANE = 0.010
TARGET_SIGNAL = 0.010
MATURE_FUNCTIONAL_PROTEIN = 0.026
MATURE_MEMBRANE = 0.0075
MATURE_SIGNAL = 0.0045
TURNOVER_DAMAGE = 0.018
TURNOVER_AGGREGATE = 0.007

SENSOR_FEATURE_COUNT = 8
PREDICT_TARGET_COUNT = 4
PREDICT_FEATURE_COUNT = 7

clamp = p0.clamp
wrapped_delta = p0.wrapped_delta
finite_array = p0.finite_array
_atomic_pickle = p0._atomic_pickle
_memory_peak_mb_estimate = p0._memory_peak_mb_estimate


def _readonly_array(value, dtype=float):
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _readonly_mapping(mapping):
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


def _sequence_hash(sequence):
    return hashlib.sha1(np.asarray(sequence, dtype=np.uint8).tobytes()).hexdigest()[:12]


def make_one_neuron_gene(ligand=s4.LIGAND_FUEL, gain=5, prediction=5,
                         persistence=5, promoter=5, efficiency=5,
                         fidelity=6):
    """Encode a neural-development cassette in the reserved effector grammar.

    spare symbols:
      0: selected physical ligand profile
      1: membrane-input gain
      2: local predictor plasticity
      3: predicted-gradient persistence
    """
    return g2.make_gene(
        g2.ROLE_REGULATOR,
        parameter=NEURAL_GENE_EFFECT,
        regulator=NEURAL_GENE_CHANNEL,
        promoter=int(promoter),
        efficiency=int(efficiency),
        fidelity=int(fidelity),
        localisation=NEURAL_GENE_LOCALISATION,
        spare=(int(ligand), int(gain), int(prediction), int(persistence)),
    )


def is_one_neuron_spec(spec):
    return bool(
        spec.get('role') == g2.ROLE_REGULATOR
        and int(spec.get('localisation', -1)) == int(NEURAL_GENE_LOCALISATION)
        and int(spec.get('parameter', -1)) == int(NEURAL_GENE_EFFECT)
        and int(spec.get('regulator', -1)) == int(NEURAL_GENE_CHANNEL)
    )


def one_neuron_specs(cell):
    return [
        (int(fingerprint), spec)
        for fingerprint, spec in cell.gene_specs.items()
        if is_one_neuron_spec(spec)
    ]


def one_neuron_gene_activity(cell):
    total = 0.0
    for fingerprint, spec in one_neuron_specs(cell):
        amount = max(0.0, float(cell.proteins.get(fingerprint, 0.0)))
        total += amount * float(spec['promoter']) * float(spec['efficiency']) / 0.012
    return float(total)


def decode_one_neuron_gene(spec):
    payload = tuple(int(v) for v in spec['payload'])
    ligand = int(payload[8]) % s4.LIGAND_COUNT
    gain = 0.55 + 1.65 * (float(payload[9]) / 7.0)
    prediction_lr = 0.0025 + 0.020 * (float(payload[10]) / 7.0)
    persistence = 0.10 + 0.78 * (float(payload[11]) / 7.0)
    return {
        'ligand': ligand,
        'gain': float(gain),
        'prediction_lr': float(prediction_lr),
        'persistence': float(persistence),
        'fidelity': float(spec['fidelity']),
        'promoter': float(spec['promoter']),
        'efficiency': float(spec['efficiency']),
        'fingerprint': int(spec['fingerprint']),
    }


def install_one_neuron_cassette(cell, bootstrap_protein=True):
    """Install one material gene in a founder; never called during inheritance.

    The extra genome polymer and bootstrap protein are part of the founder's
    initial material inventory.  Daughters receive the gene only through the
    ordinary 0.5 material genome replication and split machinery.
    """
    existing = one_neuron_specs(cell)
    if existing:
        return int(existing[0][0]), False
    if not cell.genomes:
        return None, False
    gene = make_one_neuron_gene()
    if len(cell.genomes[0]) + len(gene) > g2.MAX_GENOME_LENGTH:
        raise ValueError('no physical room for P1 neural-development cassette')
    cell.genomes[0] = np.concatenate([cell.genomes[0], gene]).astype(np.uint8)
    # A larger hereditary program needs matching free monomers before a second
    # copy can be completed.  This is explicit founder matter, not a free copy.
    cell.pools[s5.POOL_NUCLEOTIDE] += len(gene) * s5.MONOMER_MASS
    cell._refresh_gene_cache()
    matches = one_neuron_specs(cell)
    if not matches:
        raise AssertionError('installed P1 gene was not decoded')
    fingerprint = int(matches[0][0])
    if bootstrap_protein:
        cell.proteins[fingerprint] = cell.proteins.get(fingerprint, 0.0) + 0.012
    cell._sync_protein_pool()
    return fingerprint, True


def _membrane_normals():
    angles = 2.0 * math.pi * (
        np.arange(s5.MEMBRANE_SEGMENTS, dtype=float) + 0.5
    ) / s5.MEMBRANE_SEGMENTS
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


MEMBRANE_NORMALS = _membrane_normals()


class P1Config(p0.P0Config):
    """P0 body contract plus one-neuron engineering conditions."""

    def __init__(
        self,
        p1_tissue_mode=TISSUE_NEURON,
        p1_install_gene=True,
        p1_bootstrap_neural_protein=True,
        p1_environment=ENV_NATIVE,
        p1_patch_period=16.0,
        p1_patch_hold=7.0,
        p1_motor_magnitude=0.70,
        p1_transporter_magnitude=0.52,
        p1_material_wear=True,
        p1_predictor=True,
        p1_effectors=True,
        p1_neural_cost=True,
        p1_tissue_turnover=True,
        **kwargs
    ):
        if 'sensorimotor' not in kwargs and p1_environment != ENV_NATIVE:
            kwargs['sensorimotor'] = False
        if 'external_inflow' not in kwargs and p1_environment != ENV_NATIVE:
            kwargs['external_inflow'] = False
        super(P1Config, self).__init__(**kwargs)
        mode = str(p1_tissue_mode)
        if mode not in TISSUE_MODES:
            raise ValueError('unknown P1 tissue mode: {}'.format(mode))
        environment = str(p1_environment)
        if environment not in ENVIRONMENTS:
            raise ValueError('unknown P1 environment: {}'.format(environment))
        self.p1_tissue_mode = mode
        self.p1_install_gene = bool(p1_install_gene)
        self.p1_bootstrap_neural_protein = bool(p1_bootstrap_neural_protein)
        self.p1_environment = environment
        self.p1_patch_period = float(p1_patch_period)
        self.p1_patch_hold = float(p1_patch_hold)
        self.p1_motor_magnitude = float(p1_motor_magnitude)
        self.p1_transporter_magnitude = float(p1_transporter_magnitude)
        self.p1_material_wear = bool(p1_material_wear)
        self.p1_predictor = bool(p1_predictor)
        self.p1_effectors = bool(p1_effectors)
        if not bool(p1_neural_cost):
            raise ValueError(
                'P1 forbids unmetered neural tissue; a costless upper bound must '
                'use an explicit external-assistance ledger in a later milestone'
            )
        self.p1_neural_cost = True
        self.p1_tissue_turnover = bool(p1_tissue_turnover)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class OneMaterialNeuron(object):
    """One paid material compartment with local prediction and no recurrence.

    Numerical variables represent labile channel/protein modification states.
    Their physical substrate is the P0 attachment's finite protein, membrane and
    signal matter.  The object is never copied to daughters; P0 recycles the
    attachment before division and daughters must re-develop from their genome.
    """

    def __init__(self, tissue_id, mode, gene_parameters, rng_seed=0):
        self.tissue_id = str(tissue_id)
        self.mode = str(mode)
        self.gene_parameters = dict(gene_parameters)
        self.rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
        self.development = 0.0
        self.mature = False
        self.failed = False
        self.rebuilds = 0
        self.activation = 0.0
        self.previous_activation = 0.0
        self.phase = 0.0
        self.noise_state = np.zeros(2, dtype=float)
        self.last_direction = np.array([1.0, 0.0], dtype=float)
        self.last_sensor = np.zeros(SENSOR_FEATURE_COUNT, dtype=float)
        self.last_target = np.zeros(PREDICT_TARGET_COUNT, dtype=float)
        self.last_prediction = np.zeros(PREDICT_TARGET_COUNT, dtype=float)
        self.last_predict_features = np.zeros(PREDICT_FEATURE_COUNT, dtype=float)
        self.predict_w = np.zeros((PREDICT_TARGET_COUNT, PREDICT_FEATURE_COUNT), dtype=float)
        self.prediction_error = 0.65
        self.last_prediction_rms = 0.0
        self.predictor_updates = 0
        self.sensor_steps = 0
        self.activity_steps = 0
        self.effector_steps = 0
        self.turnovers = 0
        self.cumulative_motor_force = 0.0
        self.cumulative_atp_spent = 0.0
        self.cumulative_signal_spent = 0.0
        self.cumulative_wear_material = 0.0
        self.last_effector_report = {}
        self.last_material_status = np.zeros(p0.TISSUE_MATERIAL_COUNT, dtype=float)
        self.last_store_status = np.zeros(p0.BUDGET_COUNT, dtype=float)
        self.last_gene_activity = 0.0
        self.last_selected_ligand = int(self.gene_parameters.get('ligand', 0))

    @property
    def kind(self):
        if self.mode == TISSUE_DUMMY:
            return P1_DUMMY_KIND
        return P1_NEURAL_KIND

    @property
    def information_processing(self):
        return self.mode != TISSUE_DUMMY

    @property
    def predictor_enabled(self):
        return self.mode != TISSUE_NO_PREDICTION

    @property
    def effectors_enabled(self):
        return self.mode != TISSUE_NO_EFFECTOR

    def _status(self, port):
        status = port.attachment_status(self.tissue_id)
        self.last_material_status = np.asarray(status['tissue_material'], dtype=float).copy()
        self.last_store_status = np.asarray(status['stores'], dtype=float).copy()
        return status

    def _material_development(self, port, dt, costs=True):
        status = self._status(port)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        functional = float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])
        membrane = float(tissue[p0.TISSUE_MEMBRANE])
        signal = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        protein_need = max(0.0, TARGET_FUNCTIONAL_PROTEIN - functional)
        membrane_need = max(0.0, TARGET_MEMBRANE - membrane)
        signal_need = max(0.0, TARGET_SIGNAL - signal)
        if protein_need + membrane_need + signal_need > 1e-12:
            request = {
                'atp': 0.003 if costs else 0.0,
                'protein': min(protein_need, 0.0015),
                'membrane': min(membrane_need, 0.0009),
                'signal': min(signal_need, 0.0008),
            }
            port.allocate_budget(self.tissue_id, request, dt)
            status = self._status(port)
            stores = np.asarray(status['stores'], dtype=float)
            port.commit_material(
                self.tissue_id,
                protein=min(protein_need, float(stores[p0.BUDGET_PROTEIN])),
                membrane=min(membrane_need, float(stores[p0.BUDGET_MEMBRANE])),
                signal=min(signal_need, float(stores[p0.BUDGET_SIGNAL])),
            )
            status = self._status(port)
            tissue = np.asarray(status['tissue_material'], dtype=float)
            stores = np.asarray(status['stores'], dtype=float)
        functional = float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])
        membrane = float(tissue[p0.TISSUE_MEMBRANE])
        signal = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        ratios = np.asarray([
            functional / TARGET_FUNCTIONAL_PROTEIN,
            membrane / TARGET_MEMBRANE,
            signal / TARGET_SIGNAL,
        ], dtype=float)
        self.development = float(clamp(np.min(ratios), 0.0, 1.0))
        self.mature = bool(
            functional >= MATURE_FUNCTIONAL_PROTEIN
            and membrane >= MATURE_MEMBRANE
            and signal >= MATURE_SIGNAL
        )
        return status

    def _maintenance(self, port, dt, costs=True, wear=True, turnover=True):
        status = self._status(port)
        stores = np.asarray(status['stores'], dtype=float)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        target_atp = 0.006
        target_free_signal = 0.0012
        request = {
            'atp': max(0.0, target_atp - float(stores[p0.BUDGET_ATP])) if costs else 0.0,
            'signal': max(0.0, target_free_signal - float(stores[p0.BUDGET_SIGNAL])) if costs else 0.0,
            'protein': 0.00010 * dt if (costs and wear) else 0.0,
            'membrane': 0.0,
        }
        if any(value > 1e-14 for value in request.values()):
            port.allocate_budget(self.tissue_id, request, dt)
        if costs and wear:
            status = self._status(port)
            stores = np.asarray(status['stores'], dtype=float)
            wear_material = min(float(stores[p0.BUDGET_PROTEIN]), 0.00010 * dt)
            if wear_material > 0.0:
                built = port.commit_material(
                    self.tissue_id,
                    protein=wear_material,
                    damaged_fraction=0.72,
                    aggregate_fraction=0.18,
                )
                self.cumulative_wear_material += float(
                    built['damaged_protein'] + built['aggregate']
                )
        status = self._status(port)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        damage = float(tissue[p0.TISSUE_DAMAGED_PROTEIN])
        aggregate = float(tissue[p0.TISSUE_AGGREGATE])
        if turnover and (damage >= TURNOVER_DAMAGE or aggregate >= TURNOVER_AGGREGATE):
            port.return_dead_tissue(self.tissue_id, reason='p1-activity-turnover')
            self.failed = True
            self.mature = False
            self.turnovers += 1
            return False
        return True

    def _extract_physical_sensor(self, frame):
        ligand = int(self.gene_parameters.get('ligand', 0)) % s4.LIGAND_COUNT
        profiles = np.asarray(frame['external']['ligand_profiles'], dtype=float)
        profile = np.maximum(0.0, profiles[ligand])
        total = float(np.sum(profile))
        if total > 1e-12:
            gradient = np.sum(profile[:, None] * MEMBRANE_NORMALS, axis=0) / total
        else:
            gradient = np.zeros(2, dtype=float)
        concentration = float(np.mean(profile))
        uptake = float(np.asarray(frame['flux']['last_uptake_by_ligand'], dtype=float)[ligand])
        atp = float(frame['internal']['atp'])
        reactive = float(frame['internal']['reactive'])
        closure = float(frame['internal']['closure_mean'])
        retention = float(frame['internal']['retention'])
        features = np.asarray([
            clamp(gradient[0], -1.0, 1.0),
            clamp(gradient[1], -1.0, 1.0),
            math.tanh(2.5 * concentration),
            math.tanh(70.0 * uptake),
            1.0 - clamp(atp / (0.10 + atp), 0.0, 1.0),
            clamp(reactive / (0.08 + reactive), 0.0, 1.0),
            clamp(1.0 - closure, 0.0, 1.0),
            clamp(1.0 - retention, 0.0, 1.0),
        ], dtype=float)
        target = np.asarray([
            features[2], features[0], features[1], features[3],
        ], dtype=float)
        return features, target, gradient

    def _dummy_sensor(self):
        features = np.asarray([
            math.cos(self.phase), math.sin(self.phase),
            0.5 + 0.5 * math.sin(self.phase * 0.37),
            0.5 + 0.5 * math.cos(self.phase * 0.53),
            0.25, 0.20, 0.05, 0.05,
        ], dtype=float)
        target = np.asarray([
            features[2], features[0], features[1], features[3],
        ], dtype=float)
        gradient = features[:2].copy()
        return features, target, gradient

    def _choose_direction(self, gradient):
        if self.mode == TISSUE_DUMMY:
            base = np.asarray([math.cos(self.phase), math.sin(self.phase)], dtype=float)
        else:
            predicted_gradient = self.last_prediction[1:3]
            confidence = math.exp(-2.2 * clamp(self.prediction_error, 0.0, 2.0))
            persistence = float(self.gene_parameters.get('persistence', 0.4))
            if self.predictor_enabled:
                base = gradient + persistence * confidence * predicted_gradient
            else:
                base = gradient.copy()
            if float(np.linalg.norm(base)) < 1e-7:
                base = 0.72 * self.last_direction + 0.28 * self.noise_state
        norm = float(np.linalg.norm(base))
        if norm < 1e-10:
            return np.array([1.0, 0.0], dtype=float)
        return base / norm

    def pre_step(self, port, dt, config, gene_activity):
        self.last_gene_activity = float(gene_activity)
        if self.failed:
            return {'status': 'failed'}
        # Catastrophically damaged tissue must not escape turnover merely by
        # falling below the maturity threshold.  Check the physical attachment
        # before attempting regrowth.
        status = self._status(port)
        existing = np.asarray(status['tissue_material'], dtype=float)
        if config.p1_tissue_turnover and (
            float(existing[p0.TISSUE_DAMAGED_PROTEIN]) >= TURNOVER_DAMAGE
            or float(existing[p0.TISSUE_AGGREGATE]) >= TURNOVER_AGGREGATE
        ):
            port.return_dead_tissue(self.tissue_id, reason='p1-damage-turnover')
            self.failed = True
            self.mature = False
            self.turnovers += 1
            return {'status': 'turnover'}
        status = self._material_development(port, dt, costs=config.p1_neural_cost)
        if not self.mature:
            return {'status': 'developing', 'development': self.development}
        if not self._maintenance(
            port, dt, costs=config.p1_neural_cost,
            wear=config.p1_material_wear,
            turnover=config.p1_tissue_turnover,
        ):
            return {'status': 'turnover'}

        frame = port.raw_sensor_fluxes(self.tissue_id)
        self.phase = (self.phase + dt * 0.92) % (2.0 * math.pi)
        rho = math.exp(-dt / 0.7)
        self.noise_state = (
            rho * self.noise_state
            + math.sqrt(max(0.0, 1.0 - rho * rho))
            * self.rng.normal(0.0, 1.0, 2)
        )
        if self.mode == TISSUE_DUMMY:
            features, target, gradient = self._dummy_sensor()
        else:
            features, target, gradient = self._extract_physical_sensor(frame)
        self.previous_activation = self.activation
        gain = float(self.gene_parameters.get('gain', 1.0))
        drive = gain * (
            0.48 * features[2]
            + 0.34 * float(np.linalg.norm(features[:2]))
            - 0.16 * features[5]
            - 0.08 * features[6]
        )
        if self.mode == TISSUE_DUMMY:
            drive = 0.65 * math.sin(self.phase) + 0.15 * math.cos(0.31 * self.phase)
        target_activation = math.tanh(drive)
        leak = 1.0 - math.exp(-dt / 0.24)
        self.activation += leak * (target_activation - self.activation)

        direction = self._choose_direction(gradient)
        self.last_direction = direction.copy()
        self.last_sensor = features.copy()
        self.last_target = target.copy()
        self.last_predict_features = np.asarray([
            1.0,
            self.activation,
            self.previous_activation,
            direction[0], direction[1],
            target[0],
            clamp(self.prediction_error, 0.0, 1.5),
        ], dtype=float)
        self.last_prediction = np.tanh(self.predict_w.dot(self.last_predict_features))
        self.sensor_steps += 1
        self.activity_steps += 1

        report = {}
        if config.p1_effectors and self.effectors_enabled:
            motor = direction * clamp(config.p1_motor_magnitude, 0.0, 1.0)
            transporter = direction * clamp(config.p1_transporter_magnitude, 0.0, 1.0)
            report = port.apply_effector_fluxes(
                self.tissue_id,
                {'motor': motor, 'transporter_polarity': transporter},
                dt,
            )
            self.effector_steps += 1
            self.cumulative_motor_force += float(report.get('motor_force', 0.0))
            self.cumulative_atp_spent += float(report.get('atp_spent', 0.0))
            self.cumulative_signal_spent += float(report.get('signal_spent', 0.0))
        self.last_effector_report = dict(report)
        return {
            'status': 'active',
            'direction': direction.copy(),
            'prediction': self.last_prediction.copy(),
            'effector': dict(report),
        }

    def post_step(self, port, dt, config):
        if self.failed or not self.mature or self.sensor_steps <= 0:
            return {'status': 'inactive'}
        frame = port.raw_sensor_fluxes(self.tissue_id)
        if self.mode == TISSUE_DUMMY:
            _, next_target, _ = self._dummy_sensor()
        else:
            _, next_target, _ = self._extract_physical_sensor(frame)
        error = next_target - self.last_prediction
        rms = float(math.sqrt(float(np.mean(error ** 2))))
        self.last_prediction_rms = rms
        self.prediction_error = 0.985 * self.prediction_error + 0.015 * rms
        if config.p1_predictor and self.predictor_enabled:
            lr = float(self.gene_parameters.get('prediction_lr', 0.01))
            local_delta = error * (1.0 - self.last_prediction ** 2)
            self.predict_w += lr * dt * np.outer(local_delta, self.last_predict_features)
            np.clip(self.predict_w, -1.8, 1.8, out=self.predict_w)
            self.predictor_updates += 1
        return {'status': 'learned', 'prediction_rms': rms}

    def finite(self):
        arrays = (
            self.noise_state, self.last_direction, self.last_sensor,
            self.last_target, self.last_prediction, self.last_predict_features,
            self.predict_w, self.last_material_status, self.last_store_status,
        )
        scalars = (
            self.development, self.activation, self.previous_activation,
            self.phase, self.prediction_error, self.last_prediction_rms,
            self.cumulative_motor_force, self.cumulative_atp_spent,
            self.cumulative_signal_spent, self.cumulative_wear_material,
            self.last_gene_activity,
        )
        return bool(
            all(finite_array(array) for array in arrays)
            and all(np.isfinite(value) for value in scalars)
        )

    def diagnostics(self):
        return _readonly_mapping({
            'tissue_id': self.tissue_id,
            'mode': self.mode,
            'kind': self.kind,
            'mature': self.mature,
            'failed': self.failed,
            'development': self.development,
            'activation': self.activation,
            'direction': self.last_direction,
            'prediction_error': self.prediction_error,
            'prediction_rms': self.last_prediction_rms,
            'predictor_updates': self.predictor_updates,
            'sensor_steps': self.sensor_steps,
            'effector_steps': self.effector_steps,
            'motor_force_total': self.cumulative_motor_force,
            'atp_spent_total': self.cumulative_atp_spent,
            'signal_spent_total': self.cumulative_signal_spent,
            'wear_material_total': self.cumulative_wear_material,
            'gene_activity': self.last_gene_activity,
            'selected_ligand': self.last_selected_ligand,
            'material': self.last_material_status,
            'stores': self.last_store_status,
        })

    def state_dict(self):
        return {
            'tissue_id': self.tissue_id,
            'mode': self.mode,
            'gene_parameters': dict(self.gene_parameters),
            'rng_state': self.rng.bit_generator.state,
            'development': self.development,
            'mature': self.mature,
            'failed': self.failed,
            'rebuilds': self.rebuilds,
            'activation': self.activation,
            'previous_activation': self.previous_activation,
            'phase': self.phase,
            'noise_state': self.noise_state.copy(),
            'last_direction': self.last_direction.copy(),
            'last_sensor': self.last_sensor.copy(),
            'last_target': self.last_target.copy(),
            'last_prediction': self.last_prediction.copy(),
            'last_predict_features': self.last_predict_features.copy(),
            'predict_w': self.predict_w.copy(),
            'prediction_error': self.prediction_error,
            'last_prediction_rms': self.last_prediction_rms,
            'predictor_updates': self.predictor_updates,
            'sensor_steps': self.sensor_steps,
            'activity_steps': self.activity_steps,
            'effector_steps': self.effector_steps,
            'turnovers': self.turnovers,
            'cumulative_motor_force': self.cumulative_motor_force,
            'cumulative_atp_spent': self.cumulative_atp_spent,
            'cumulative_signal_spent': self.cumulative_signal_spent,
            'cumulative_wear_material': self.cumulative_wear_material,
            'last_effector_report': dict(self.last_effector_report),
            'last_material_status': self.last_material_status.copy(),
            'last_store_status': self.last_store_status.copy(),
            'last_gene_activity': self.last_gene_activity,
            'last_selected_ligand': self.last_selected_ligand,
        }

    @classmethod
    def from_state(cls, state):
        tissue = cls(
            state['tissue_id'], state['mode'], state['gene_parameters'], rng_seed=0,
        )
        tissue.rng.bit_generator.state = state['rng_state']
        for name in (
            'development', 'mature', 'failed', 'rebuilds', 'activation',
            'previous_activation', 'phase', 'prediction_error',
            'last_prediction_rms', 'predictor_updates', 'sensor_steps',
            'activity_steps', 'effector_steps', 'turnovers',
            'cumulative_motor_force', 'cumulative_atp_spent',
            'cumulative_signal_spent', 'cumulative_wear_material',
            'last_gene_activity', 'last_selected_ligand',
        ):
            setattr(tissue, name, state[name])
        for name in (
            'noise_state', 'last_direction', 'last_sensor', 'last_target',
            'last_prediction', 'last_predict_features', 'predict_w',
            'last_material_status', 'last_store_status',
        ):
            setattr(tissue, name, np.asarray(state[name], dtype=float).copy())
        tissue.last_effector_report = dict(state.get('last_effector_report', {}))
        return tissue


class P1ProtoCell(p0.P0ProtoCell):
    """P0 chemical cell plus one non-hereditary material tissue state."""

    def __init__(self, *args, **kwargs):
        super(P1ProtoCell, self).__init__(*args, **kwargs)
        self._init_p1_state()

    def _init_p1_state(self):
        self.p1_tissue = None
        self.p1_tissue_births = 0
        self.p1_tissue_turnovers = 0
        self.p1_neural_gene_fingerprint = -1
        self.p1_neural_gene_installed = False

    def split(self, world):
        # P0 returns all attached neural matter before the validated material
        # split, so the numerical neural state is intentionally not inherited.
        daughters = super(P1ProtoCell, self).split(world)
        if daughters is None:
            return None
        for daughter in daughters:
            daughter.__class__ = P1ProtoCell
            daughter._init_p1_state()
        return daughters

    def state_dict(self):
        state = super(P1ProtoCell, self).state_dict()
        state.update({
            'cell_class': 'P1ProtoCell',
            'p1_tissue': None if self.p1_tissue is None else self.p1_tissue.state_dict(),
            'p1_tissue_births': self.p1_tissue_births,
            'p1_tissue_turnovers': self.p1_tissue_turnovers,
            'p1_neural_gene_fingerprint': self.p1_neural_gene_fingerprint,
            'p1_neural_gene_installed': self.p1_neural_gene_installed,
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = p0.P0ProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        cell._init_p1_state()
        if state.get('p1_tissue') is not None:
            cell.p1_tissue = OneMaterialNeuron.from_state(state['p1_tissue'])
        cell.p1_tissue_births = int(state.get('p1_tissue_births', 0))
        cell.p1_tissue_turnovers = int(state.get('p1_tissue_turnovers', 0))
        cell.p1_neural_gene_fingerprint = int(state.get('p1_neural_gene_fingerprint', -1))
        cell.p1_neural_gene_installed = bool(state.get('p1_neural_gene_installed', False))
        return cell


class P1World(p0.P0World):
    """P0 world orchestrating one material tissue per genetically capable cell."""

    PATCH_CENTRES = (
        np.asarray([0.78, 0.50]), np.asarray([0.22, 0.50]),
        np.asarray([0.50, 0.78]), np.asarray([0.50, 0.22]),
    )

    def __init__(self, seed=101, initial_cells=1, config=None):
        config = config if config is not None else P1Config()
        if not isinstance(config, P1Config):
            config = P1Config(**config.state_dict())
        super(P1World, self).__init__(seed=seed, initial_cells=initial_cells, config=config)
        self.config = config
        self.p1_seed = int(seed)
        self.p1_tissue_creations = 0
        self.p1_tissue_turnovers = 0
        self.p1_pre_steps = 0
        self.p1_post_steps = 0
        self.p1_patch_switches = 0
        self.p1_last_patch_index = -1
        self.p1_last_patch_center = np.asarray([0.5, 0.5], dtype=float)
        for cell in self.cells:
            cell.__class__ = P1ProtoCell
            cell._init_p1_state()
            if self.config.p1_install_gene:
                fingerprint, installed = install_one_neuron_cassette(
                    cell, bootstrap_protein=self.config.p1_bootstrap_neural_protein,
                )
                cell.p1_neural_gene_fingerprint = int(fingerprint if fingerprint is not None else -1)
                cell.p1_neural_gene_installed = bool(installed)
        if self.config.p1_environment != ENV_NATIVE:
            self._initialise_controlled_environment()
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0
        self._ensure_all_tissues()

    def _initialise_controlled_environment(self):
        field = self.field
        # Deterministic compact initial chemistry; this is an initial condition,
        # not an unlogged in-life material injection.
        positions = []
        kinds = []
        amounts = []
        centre = self.PATCH_CENTRES[0]
        for kind, count, total, radius in (
            (s5.PARTICLE_FUEL, 28, 1.40, 0.020),
            (s5.PARTICLE_MINERAL, 22, 0.88, 0.024),
            (s5.PARTICLE_WASTE, 8, 0.10, 0.060),
        ):
            for index in range(count):
                angle = 2.0 * math.pi * (index + 0.5) / count
                ring = radius * (0.35 + 0.65 * ((index % 5) / 4.0))
                local_centre = centre if kind != s5.PARTICLE_WASTE else np.asarray([0.5, 0.5])
                positions.append((local_centre + ring * np.asarray([
                    math.cos(angle), math.sin(angle),
                ])) % 1.0)
                kinds.append(kind)
                amounts.append(total / count)
        field.pos = np.asarray(positions, dtype=float)
        field.kind = np.asarray(kinds, dtype=np.int16)
        field.amount = np.asarray(amounts, dtype=float)
        field.recycled_fuel_buffer = 0.0
        field.recycled_mineral_buffer = 0.0
        field.injected_material = 0.0
        field.dissipated_material = 0.0
        for index, cell in enumerate(self.cells):
            cell.pos = np.asarray([0.50, 0.50 + 0.018 * index], dtype=float) % 1.0
            cell.vel[:] = 0.0
        self.p1_last_patch_center = centre.copy()
        self.p1_last_patch_index = 0

    def _current_patch_index(self):
        if self.config.p1_environment == ENV_STABLE_PATCH:
            return 0
        if self.config.p1_environment == ENV_MOVING_PATCH:
            period = max(2.0, float(self.config.p1_patch_period))
            return int(math.floor(self.age / period)) % len(self.PATCH_CENTRES)
        return -1

    def _maintain_patch(self, dt):
        index = self._current_patch_index()
        if index < 0:
            return
        centre = self.PATCH_CENTRES[index]
        if index != self.p1_last_patch_index:
            self.p1_patch_switches += 1
            self.p1_last_patch_index = index
        self.p1_last_patch_center = centre.copy()
        kinds = self.field.kind
        for kind, strength, radius in (
            (s5.PARTICLE_FUEL, 0.62, 0.022),
            (s5.PARTICLE_MINERAL, 0.55, 0.027),
        ):
            indices = np.where(kinds == kind)[0]
            count = len(indices)
            if count == 0:
                continue
            for local, particle_index in enumerate(indices):
                angle = 2.0 * math.pi * (local + 0.5) / count
                ring = radius * (0.35 + 0.65 * ((local % 5) / 4.0))
                target = (centre + ring * np.asarray([
                    math.cos(angle), math.sin(angle),
                ])) % 1.0
                delta = wrapped_delta(self.field.pos[particle_index], target)
                self.field.pos[particle_index] = (
                    self.field.pos[particle_index]
                    + clamp(dt * strength, 0.0, 0.12) * delta
                ) % 1.0

    def _gene_parameters_for(self, cell):
        specs = one_neuron_specs(cell)
        if not specs:
            return None
        # Highest realised protein activity wins; deterministic tie by fingerprint.
        ranked = sorted(
            specs,
            key=lambda item: (
                -float(cell.proteins.get(item[0], 0.0))
                * float(item[1]['promoter']) * float(item[1]['efficiency']),
                int(item[0]),
            ),
        )
        return decode_one_neuron_gene(ranked[0][1])

    def _new_tissue(self, cell):
        mode = self.config.p1_tissue_mode
        if mode == TISSUE_NONE:
            return None
        parameters = self._gene_parameters_for(cell)
        activity = one_neuron_gene_activity(cell)
        if parameters is None or activity < 0.020:
            return None
        port = self.port_for(cell.cell_id)
        port.attach(P1_TISSUE_ID, kind=(
            P1_DUMMY_KIND if mode == TISSUE_DUMMY else P1_NEURAL_KIND
        ))
        seed = (
            (self.p1_seed * 1000003)
            ^ (int(cell.cell_id) * 9176)
            ^ (int(cell.generation) * 7919)
        ) & 0xFFFFFFFF
        tissue = OneMaterialNeuron(P1_TISSUE_ID, mode, parameters, rng_seed=seed)
        cell.p1_tissue = tissue
        cell.p1_tissue_births += 1
        self.p1_tissue_creations += 1
        return tissue

    def _ensure_tissue(self, cell):
        if not cell.alive or self.config.p1_tissue_mode == TISSUE_NONE:
            return None
        if cell.p1_tissue is not None:
            if cell.p1_tissue.failed:
                cell.p1_tissue_turnovers += 1
                self.p1_tissue_turnovers += 1
                cell.p1_tissue = None
            else:
                return cell.p1_tissue
        if P1_TISSUE_ID in cell.neural_attachments:
            # This can happen only when restoring an inconsistent development
            # checkpoint; fail closed by recycling the orphaned matter.
            self.port_for(cell.cell_id).return_dead_tissue(
                P1_TISSUE_ID, reason='orphaned-p1-attachment'
            )
        return self._new_tissue(cell)

    def _ensure_all_tissues(self):
        for cell in self.living_cells():
            if not isinstance(cell, P1ProtoCell):
                cell.__class__ = P1ProtoCell
                cell._init_p1_state()
            self._ensure_tissue(cell)

    def _pre_neural_step(self, dt):
        for cell in list(self.living_cells()):
            tissue = self._ensure_tissue(cell)
            if tissue is None:
                continue
            port = self.port_for(cell.cell_id)
            tissue.pre_step(
                port, dt, self.config, one_neuron_gene_activity(cell)
            )
            self.p1_pre_steps += 1

    def _post_neural_step(self, dt):
        for cell in list(self.living_cells()):
            tissue = cell.p1_tissue if isinstance(cell, P1ProtoCell) else None
            if tissue is None or tissue.failed:
                continue
            if P1_TISSUE_ID not in cell.neural_attachments:
                tissue.failed = True
                continue
            tissue.post_step(self.port_for(cell.cell_id), dt, self.config)
            self.p1_post_steps += 1
        self._ensure_all_tissues()

    def _release_dead_cell(self, cell):
        if isinstance(cell, P1ProtoCell):
            cell.p1_tissue = None
        return super(P1World, self)._release_dead_cell(cell)

    def step(self, dt):
        dt = clamp(float(dt), 1.0 / 240.0, 0.10)
        self._maintain_patch(dt)
        self._pre_neural_step(dt)
        super(P1World, self).step(dt)
        self._post_neural_step(dt)

    def finite(self):
        if not super(P1World, self).finite():
            return False
        for cell in self.living_cells():
            if not isinstance(cell, P1ProtoCell):
                return False
            if cell.p1_tissue is not None and not cell.p1_tissue.finite():
                return False
        return finite_array(self.p1_last_patch_center)

    def patch_distance(self, cell):
        if self._current_patch_index() < 0:
            return float('nan')
        return float(np.linalg.norm(wrapped_delta(cell.pos, self.p1_last_patch_center)))

    def summary(self):
        summary = super(P1World, self).summary()
        alive = self.living_cells()
        tissues = [
            cell.p1_tissue for cell in alive
            if isinstance(cell, P1ProtoCell) and cell.p1_tissue is not None
        ]
        mature = [t for t in tissues if t.mature and not t.failed]
        distances = [self.patch_distance(cell) for cell in alive]
        distances = [value for value in distances if np.isfinite(value)]
        summary.update({
            'build': BUILD,
            'p1_schema': P1_SCHEMA_VERSION,
            'p1_mode': self.config.p1_tissue_mode,
            'p1_environment': self.config.p1_environment,
            'p1_tissues': len(tissues),
            'p1_mature_tissues': len(mature),
            'p1_tissue_creations': self.p1_tissue_creations,
            'p1_tissue_turnovers': self.p1_tissue_turnovers,
            'p1_predictor_updates': int(sum(t.predictor_updates for t in tissues)),
            'p1_prediction_error': float(np.mean([t.prediction_error for t in tissues])) if tissues else 0.0,
            'p1_motor_force_total': float(sum(t.cumulative_motor_force for t in tissues)),
            'p1_neural_atp_total': float(sum(t.cumulative_atp_spent for t in tissues)),
            'p1_neural_signal_total': float(sum(t.cumulative_signal_spent for t in tissues)),
            'p1_wear_material_total': float(sum(t.cumulative_wear_material for t in tissues)),
            'p1_patch_switches': self.p1_patch_switches,
            'p1_mean_patch_distance': float(np.mean(distances)) if distances else 0.0,
            'p1_pre_steps': self.p1_pre_steps,
            'p1_post_steps': self.p1_post_steps,
        })
        return summary

    def state_dict(self):
        state = super(P1World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'p1_seed': self.p1_seed,
            'p1_tissue_creations': self.p1_tissue_creations,
            'p1_tissue_turnovers': self.p1_tissue_turnovers,
            'p1_pre_steps': self.p1_pre_steps,
            'p1_post_steps': self.p1_post_steps,
            'p1_patch_switches': self.p1_patch_switches,
            'p1_last_patch_index': self.p1_last_patch_index,
            'p1_last_patch_center': self.p1_last_patch_center.copy(),
        })
        return state

    @classmethod
    def from_state(cls, state):
        base_state = dict(state)
        base_state['save_version'] = p0.SAVE_VERSION
        base_state['build'] = p0.BUILD
        # P0's loader knows only P0Config keys; filter P1-only fields there and
        # restore the full P1Config after the chemical body is rebuilt.
        p0_keys = set(p0.P0Config().__dict__.keys())
        base_state['config'] = {
            key: value for key, value in dict(state.get('config', {})).items()
            if key in p0_keys
        }
        world = p0.P0World.from_state(base_state)
        world.__class__ = cls
        world.config = P1Config.from_state(state.get('config', {}))
        world.cells = [P1ProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        world.p1_seed = int(state.get('p1_seed', 101))
        for name in (
            'p1_tissue_creations', 'p1_tissue_turnovers', 'p1_pre_steps',
            'p1_post_steps', 'p1_patch_switches', 'p1_last_patch_index',
        ):
            setattr(world, name, int(state.get(name, 0)))
        world.p1_last_patch_center = np.asarray(
            state.get('p1_last_patch_center', [0.5, 0.5]), dtype=float
        ).copy()
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return P1World.from_state(self.state_dict())


def p0_projection(world_or_state):
    """Strip P1-only metadata for exact regression against frozen P0.

    This helper is deliberately narrow: it is valid only when no P1 cassette or
    tissue has been installed and the native P0 environment is used.
    """
    state = world_or_state.state_dict() if hasattr(world_or_state, 'state_dict') else world_or_state
    p0_config_keys = set(p0.P0Config().__dict__.keys())

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                text = str(key)
                if text.startswith('p1_') or text == 'p1_tissue':
                    continue
                if text == 'cell_class' and item == 'P1ProtoCell':
                    result[key] = 'P0ProtoCell'
                elif text == 'build':
                    result[key] = p0.BUILD
                elif text == 'save_version':
                    result[key] = p0.SAVE_VERSION
                elif text == 'config':
                    result[key] = clean({
                        k: v for k, v in item.items() if k in p0_config_keys
                    })
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


def run_headless_trial(seed=101, seconds=120.0, initial_cells=1, config=None):
    world = P1World(seed=seed, initial_cells=initial_cells, config=config)
    dt = 1.0 / SIM_HZ
    margin_sum = 0.0
    distance_sum = 0.0
    samples = 0
    uptake_start = float(sum(
        np.sum(cell.cumulative_uptake_by_ligand)
        for cell in world.living_cells()
    ))
    for _ in range(int(max(0.0, seconds) * SIM_HZ)):
        world.step(dt)
        alive = world.living_cells()
        if alive:
            margin_sum += float(np.mean([cell.autopoietic_margin() for cell in alive]))
            distances = [world.patch_distance(cell) for cell in alive]
            distances = [value for value in distances if np.isfinite(value)]
            if distances:
                distance_sum += float(np.mean(distances))
            samples += 1
        if not alive:
            break
    summary = world.summary()
    uptake_end = float(sum(
        np.sum(cell.cumulative_uptake_by_ligand)
        for cell in world.living_cells()
    ))
    summary.update({
        'seed': int(seed),
        'requested_seconds': float(seconds),
        'simulated_seconds': float(world.age),
        'mean_margin_over_life': margin_sum / max(1, samples),
        'mean_patch_distance_over_life': distance_sum / max(1, samples),
        'uptake_during_trial': max(0.0, uptake_end - uptake_start),
        'finite': int(world.finite()),
        'final_mass_residual': float(world.matter_ledger_residual()),
    })
    return summary


LOG_FIELDS = (
    'session_id', 'reason', 'wall_time', 'age', 'cells', 'corpses',
    'edna_fragments', 'divisions', 'deaths', 'mean_margin', 'mean_atp',
    'p1_mode', 'p1_environment', 'p1_tissues', 'p1_mature_tissues',
    'p1_predictor_updates', 'p1_prediction_error', 'p1_motor_force_total',
    'p1_neural_atp_total', 'p1_neural_signal_total',
    'p1_mean_patch_distance', 'p1_patch_switches', 'matter_residual',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), int(world.p1_seed))
        self.last_age = -1e9
        self.status = 'WAIT'

    def log(self, world, reason='periodic', force=False):
        if not force and world.age - self.last_age < LOG_INTERVAL:
            return False
        summary = world.summary()
        row = {key: summary.get(key, '') for key in LOG_FIELDS}
        row.update({
            'session_id': self.session_id,
            'reason': reason,
            'wall_time': time.time(),
        })
        exists = os.path.exists(self.path)
        with open(self.path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.last_age = world.age
        self.status = 'OK'
        return True


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE, session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    rows = []
    with open(log_path, 'r', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    sessions = {}
    for row in rows:
        sessions.setdefault(row['session_id'], []).append(row)
    with open(session_path, 'w', newline='', encoding='utf-8') as handle:
        fields = ('session_id', 'rows', 'final_age', 'final_cells', 'mode', 'environment')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session_id, items in sessions.items():
            last = items[-1]
            writer.writerow({
                'session_id': session_id,
                'rows': len(items),
                'final_age': last.get('age', ''),
                'final_cells': last.get('cells', ''),
                'mode': last.get('p1_mode', ''),
                'environment': last.get('p1_environment', ''),
            })
    lines = [
        BUILD_LONG,
        'log: {}'.format(os.path.basename(log_path)),
        'sessions: {}'.format(len(sessions)),
        '',
    ]
    for session_id, items in sessions.items():
        last = items[-1]
        lines.append('{} age={} cells={} mode={} env={} pred_err={} dist={} ledger={}'.format(
            session_id, last.get('age', ''), last.get('cells', ''),
            last.get('p1_mode', ''), last.get('p1_environment', ''),
            last.get('p1_prediction_error', ''),
            last.get('p1_mean_patch_distance', ''),
            last.get('matter_residual', ''),
        ))
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return 'OK'


try:
    from scene import (
        Scene, run, LANDSCAPE, background, fill, rect, ellipse,
        line, stroke, stroke_weight, text,
    )

    class SomaCellP1Scene(p0.SomaCellP0Scene):
        def setup(self):
            background(0.006, 0.012, 0.022)
            try:
                self.world = P1World.load(SAVE_FILE)
                self.save_status = 'LOAD'
            except Exception:
                self.world = P1World(seed=101, initial_cells=3)
                self.save_status = 'NEW'
            self.accumulator = 0.0
            self.last_wall = time.time()
            self.last_save_age = self.world.age
            self.last_touch_wall = -10.0
            self.paused = False
            self.fps = 0.0
            self.sim_rate = 0.0
            self.telemetry_wall = time.time()
            self.telemetry_age = self.world.age
            self.telemetry_frames = 0
            self.logger = LongRunLogger(self.world)
            self.logger.log(self.world, reason='start', force=True)
            self.report_status = 'WAIT'

        def update(self):
            now = time.time()
            elapsed = min(0.20, max(0.0, now - self.last_wall))
            self.last_wall = now
            self.telemetry_frames += 1
            if not self.paused:
                self.accumulator += elapsed
                fixed = 1.0 / SIM_HZ
                steps = 0
                while self.accumulator >= fixed and steps < 5:
                    self.world.step(fixed)
                    self.accumulator -= fixed
                    steps += 1
            wall_delta = now - self.telemetry_wall
            if wall_delta >= 1.0:
                self.fps = self.telemetry_frames / wall_delta
                self.sim_rate = (self.world.age - self.telemetry_age) / wall_delta
                self.telemetry_wall = now
                self.telemetry_age = self.world.age
                self.telemetry_frames = 0
            self.logger.log(self.world)
            if self.world.age - self.last_save_age >= AUTO_SAVE_INTERVAL:
                try:
                    self.world.save(SAVE_FILE)
                    self.save_status = 'OK'
                    self.last_save_age = self.world.age
                    gc.collect()
                except Exception:
                    self.save_status = 'ERR'
            if not self.world.living_cells() and self.report_status == 'WAIT':
                self.logger.log(self.world, reason='extinct', force=True)
                self.report_status = generate_report()

        def draw(self):
            super(SomaCellP1Scene, self).draw()
            # Draw a small material neural compartment inside each host.
            for cell in self.world.living_cells():
                if not isinstance(cell, P1ProtoCell) or cell.p1_tissue is None:
                    continue
                x, y = self._screen(cell.pos)
                radius = max(2.5, cell.radius * min(self.size.w, self.size.h) * 0.26)
                if cell.p1_tissue.mode == TISSUE_DUMMY:
                    colour = (0.65, 0.58, 0.72, 0.92)
                elif cell.p1_tissue.mature:
                    colour = (0.78, 0.30, 0.96, 0.96)
                else:
                    colour = (0.46, 0.25, 0.62, 0.75)
                fill(*colour)
                ellipse(x - radius, y - radius, radius * 2, radius * 2)
                direction = cell.p1_tissue.last_direction
                stroke(0.95, 0.74, 1.0, 0.9)
                stroke_weight(1.3)
                line(x, y, x + direction[0] * radius * 2.4, y + direction[1] * radius * 2.4)

            fill(0.010, 0.018, 0.030, 1.0)
            rect(0, self.size.h - 48, self.size.w, 48)
            rect(0, 0, self.size.w, 88)
            s = self.world.summary()
            fill(0.94, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.65, 0.80, 0.88)
            text('{} | {} | SAVE {} | {:.1f} fps | x{:.2f}'.format(
                s['p1_mode'], s['p1_environment'], self.save_status,
                self.fps, self.sim_rate,
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} div {} deaths {} corpses {} DNA {}'.format(
                s['age'], s['cells'], s['divisions'], s['deaths'],
                s['corpses'], s['edna_fragments'],
            ), x=24, y=74, font_size=10, alignment=4)
            text('tissue {} mature {} created {} turnover {}'.format(
                s['p1_tissues'], s['p1_mature_tissues'],
                s['p1_tissue_creations'], s['p1_tissue_turnovers'],
            ), x=24, y=57, font_size=9, alignment=4)
            text('predict {:.4f} updates {} motor {:.4f} patch d {:.4f}'.format(
                s['p1_prediction_error'], s['p1_predictor_updates'],
                s['p1_motor_force_total'], s['p1_mean_patch_distance'],
            ), x=24, y=40, font_size=9, alignment=4)
            text('neural ATP {:.5f} signal {:.5f} wear {:.5f}'.format(
                s['p1_neural_atp_total'], s['p1_neural_signal_total'],
                s['p1_wear_material_total'],
            ), x=24, y=23, font_size=9, alignment=4)
            text('port residual {:+.2e} world ledger {:+.2e}'.format(
                max([
                    cell.neural_budget_ledger.maximum_material_residual
                    for cell in self.world.living_cells()
                ] or [0.0]), s['matter_residual'],
            ), x=24, y=7, font_size=9, alignment=4)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = P1World(seed=101, initial_cells=3)
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
                    self.world,
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
            self.logger.log(self.world, reason='stop', force=True)
            self.report_status = generate_report()

except ImportError:
    Scene = None


if __name__ == '__main__':
    if Scene is None:
        print(run_headless_trial(
            seed=101,
            seconds=40.0,
            initial_cells=1,
            config=P1Config(p1_environment=ENV_MOVING_PATCH),
        ))
    else:
        run(SomaCellP1Scene(), LANDSCAPE, show_fps=False)
