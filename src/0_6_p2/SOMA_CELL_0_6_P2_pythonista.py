# coding: utf-8
"""
SOMA-CELL 0.6-P2 — Eight-Cell Material Recurrent Tissue
8個の物質神経細胞、有限シグナル再帰、局所予測、三因子可塑性

P2 extends the frozen P1/P0 contracts.  Eight gene-built material neural
compartments are assembled from finite body ATP, protein substrate, membrane
precursor and signal precursor.  They read only immutable physical sensor
frames and may act only through the paid P0 effector port.

The network is deliberately small and local:
- eight directionally arranged material neural cells
- sparse finite-signal recurrent connections
- local next-drive prediction
- perturbation eligibility traces
- four-channel physical-debt consequence, not an external reward
- three-factor sensor/recurrent/motor plasticity
- maturity, re-plasticisation and conservative edge/cell pruning

P2 does not yet implement ABBA/BAAB causal audit, causal calibration, the
change sentinel, falsification programs, cultural transmission, or inherited
learned weights.  Those remain blocked until the paid eight-cell tissue shows
measured value over an equal-material fixed tissue.
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
    import SOMA_CELL_0_6_P1_pythonista as p1
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _P1 = os.path.abspath(os.path.join(_HERE, '..', '0_6_p1'))
    _P0 = os.path.abspath(os.path.join(_HERE, '..', '0_6_p0'))
    _BASE = os.path.abspath(os.path.join(_HERE, '..', 'baseline'))
    for _candidate in (_P1, _P0, _BASE):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    import SOMA_CELL_0_6_P1_pythonista as p1

p0 = p1.p0
s5 = p1.s5
s4 = p1.s4
g2 = p1.g2

BUILD = 'SOMA-CELL 0.6-P2.0'
BUILD_LONG = 'SOMA-CELL 0.6-P2.0 Eight-Cell Material Recurrent Tissue'
SAVE_VERSION = 62
P2_SCHEMA_VERSION = '0.6-P2.2'
BASE_DIR = os.path.dirname(__file__)
SAVE_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p2.pkl')
LOG_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p2_longrun.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p2_report.txt')
SESSION_FILE = os.path.join(BASE_DIR, 'soma_cell_0_6_p2_sessions.csv')

SIM_HZ = p1.SIM_HZ
AUTO_SAVE_INTERVAL = 30.0
LOG_INTERVAL = 10.0

P2_CELL_COUNT = 8
P2_SENSOR_COUNT = 12
P2_PHYSICAL_CHANNELS = 4
P2_PREDICT_FEATURES = 6
P2_EXTERNAL_LIGAND_COUNT = 4

P2_MODE_NONE = 'none'
P2_MODE_FULL = 'full'
P2_MODE_FIXED = 'fixed'
P2_MODE_NO_RECURRENCE = 'no_recurrence'
P2_MODE_NO_PREDICTION = 'no_prediction'
P2_MODE_NO_PLASTICITY = 'no_plasticity'
P2_MODE_NO_EFFECTOR = 'no_effector'
P2_MODES = frozenset((
    P2_MODE_NONE, P2_MODE_FULL, P2_MODE_FIXED,
    P2_MODE_NO_RECURRENCE, P2_MODE_NO_PREDICTION,
    P2_MODE_NO_PLASTICITY, P2_MODE_NO_EFFECTOR,
))

P2_ENV_NATIVE = 'native'
P2_ENV_MOVING_PATCH = 'moving_patch'
P2_ENV_CUE_REVERSAL = 'cue_reversal'
P2_ENV_MEANING_REVERSAL = 'meaning_reversal'
P2_ENVIRONMENTS = frozenset((
    P2_ENV_NATIVE, P2_ENV_MOVING_PATCH,
    P2_ENV_CUE_REVERSAL, P2_ENV_MEANING_REVERSAL,
))

P2_TISSUE_PREFIX = 'p2-material-neuron-'
P2_NEURAL_KIND = 'eight-cell-material-neuron'
P2_FIXED_KIND = 'equal-material-fixed-eight-cell-tissue'

# A separate reserved regulator channel distinguishes the P2 developmental
# cassette from the frozen P1 one-neuron cassette.
P2_GENE_EFFECT = s4.EFFECT_RESERVED
P2_GENE_CHANNEL = s4.CONTROL_RESERVED_6
P2_GENE_LOCALISATION = s4.LOC_EFFECTOR

# Per-neural-cell matter.  Total P2 tissue is about 1.6x the P1 compartment,
# rather than eight full P1 compartments.  The fixed and adaptive modes share
# exactly these targets and the same maintenance/wear schedule.
P2_TARGET_PROTEIN = 0.0075
P2_TARGET_MEMBRANE = 0.0025
P2_TARGET_SIGNAL = 0.0025
P2_MATURE_PROTEIN = 0.0060
P2_MATURE_MEMBRANE = 0.0018
P2_MATURE_SIGNAL = 0.0014
P2_TURNOVER_DAMAGE = 0.0048
P2_TURNOVER_AGGREGATE = 0.0021
P2_BOOTSTRAP_PROTEIN = 0.0040

P2_MAX_SENSOR_WEIGHT = 1.60
P2_MAX_REC_WEIGHT = 0.72
P2_MAX_MOTOR_GAIN = 1.45
P2_MIN_MOTOR_GAIN = -1.45
P2_ELIGIBILITY_TAU = 6.5
P2_REWIRE_INTERVAL = 420
P2_CELL_PRUNE_INTERVAL = 1260
P2_REBUILD_COOLDOWN = 2.5

clamp = p1.clamp
wrapped_delta = p1.wrapped_delta
finite_array = p1.finite_array
_atomic_pickle = p1._atomic_pickle
_memory_peak_mb_estimate = p1._memory_peak_mb_estimate
MEMBRANE_NORMALS = p1.MEMBRANE_NORMALS


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


def _stable_softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=float)
    temperature = max(1e-6, float(temperature))
    shifted = (values - float(np.max(values))) / temperature
    shifted = np.clip(shifted, -60.0, 60.0)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    return exp / total if total > 0.0 else np.full_like(values, 1.0 / len(values))


def p2_tissue_id(index):
    return '{}{}'.format(P2_TISSUE_PREFIX, int(index))


def make_p2_neuron_gene(index, promoter=5, efficiency=5, fidelity=6):
    index = int(index) % P2_CELL_COUNT
    tau_code = (1, 2, 4, 6, 3, 5, 2, 7)[index]
    plasticity_code = (5, 4, 6, 3, 5, 7, 4, 6)[index]
    motor_code = (5, 4, 5, 6, 4, 5, 6, 5)[index]
    return g2.make_gene(
        g2.ROLE_REGULATOR,
        parameter=P2_GENE_EFFECT,
        regulator=P2_GENE_CHANNEL,
        promoter=int(promoter),
        efficiency=int(efficiency),
        fidelity=int(fidelity),
        localisation=P2_GENE_LOCALISATION,
        spare=(index, tau_code, plasticity_code, motor_code),
    )


def is_p2_neuron_spec(spec):
    return bool(
        spec.get('role') == g2.ROLE_REGULATOR
        and int(spec.get('localisation', -1)) == int(P2_GENE_LOCALISATION)
        and int(spec.get('parameter', -1)) == int(P2_GENE_EFFECT)
        and int(spec.get('regulator', -1)) == int(P2_GENE_CHANNEL)
    )


def p2_neuron_specs(cell):
    result = []
    for fingerprint, spec in cell.gene_specs.items():
        if not is_p2_neuron_spec(spec):
            continue
        index = int(spec['payload'][8]) % P2_CELL_COUNT
        result.append((index, int(fingerprint), spec))
    result.sort(key=lambda item: (item[0], item[1]))
    return result


def p2_gene_activity(cell):
    activity = np.zeros(P2_CELL_COUNT, dtype=float)
    for index, fingerprint, spec in p2_neuron_specs(cell):
        amount = max(0.0, float(cell.proteins.get(fingerprint, 0.0)))
        activity[index] += amount * float(spec['promoter']) * float(spec['efficiency']) / 0.012
    return activity


def decode_p2_gene(index, spec):
    payload = tuple(int(v) for v in spec['payload'])
    orientation = int(payload[8]) % P2_CELL_COUNT
    tau = 0.12 + 0.90 * (float(payload[9]) / 7.0)
    plasticity_lr = 0.0030 + 0.020 * (float(payload[10]) / 7.0)
    motor_gain = 0.42 + 0.48 * (float(payload[11]) / 7.0)
    angle = 2.0 * math.pi * orientation / P2_CELL_COUNT
    return {
        'index': int(index),
        'orientation': int(orientation),
        'direction': np.asarray([math.cos(angle), math.sin(angle)], dtype=float),
        'tau': float(tau),
        'plasticity_lr': float(plasticity_lr),
        'motor_gain': float(motor_gain),
        'promoter': float(spec['promoter']),
        'efficiency': float(spec['efficiency']),
        'fidelity': float(spec['fidelity']),
        'fingerprint': int(spec['fingerprint']),
    }


def install_p2_cassette(cell, bootstrap_protein=True):
    existing = p2_neuron_specs(cell)
    existing_indices = {item[0] for item in existing}
    if len(existing_indices) == P2_CELL_COUNT:
        return [item[1] for item in existing], False
    if not cell.genomes:
        return [], False
    genes = []
    for index in range(P2_CELL_COUNT):
        if index not in existing_indices:
            genes.append(make_p2_neuron_gene(index))
    total_symbols = sum(len(gene) for gene in genes)
    if len(cell.genomes[0]) + total_symbols > g2.MAX_GENOME_LENGTH:
        raise ValueError('no physical room for P2 eight-cell developmental cassette')
    if genes:
        cell.genomes[0] = np.concatenate([cell.genomes[0]] + genes).astype(np.uint8)
        cell.pools[s5.POOL_NUCLEOTIDE] += total_symbols * s5.MONOMER_MASS
        cell._refresh_gene_cache()
    specs = p2_neuron_specs(cell)
    by_index = {index: (fingerprint, spec) for index, fingerprint, spec in specs}
    if len(by_index) != P2_CELL_COUNT:
        raise AssertionError('installed P2 cassette did not decode to eight cells')
    if bootstrap_protein:
        for index in range(P2_CELL_COUNT):
            fingerprint = by_index[index][0]
            cell.proteins[fingerprint] = (
                cell.proteins.get(fingerprint, 0.0) + P2_BOOTSTRAP_PROTEIN
            )
    cell._sync_protein_pool()
    return [by_index[i][0] for i in range(P2_CELL_COUNT)], bool(genes)


def p2_gene_parameters(cell):
    selected = {}
    for index, fingerprint, spec in p2_neuron_specs(cell):
        score = (
            float(cell.proteins.get(fingerprint, 0.0))
            * float(spec['promoter']) * float(spec['efficiency'])
        )
        current = selected.get(index)
        if current is None or score > current[0] or (score == current[0] and fingerprint < current[1]):
            selected[index] = (score, fingerprint, spec)
    if len(selected) != P2_CELL_COUNT:
        return None
    return [decode_p2_gene(i, selected[i][2]) for i in range(P2_CELL_COUNT)]


class P2Config(p1.P1Config):
    def __init__(
        self,
        p2_tissue_mode=P2_MODE_FULL,
        p2_install_genes=True,
        p2_bootstrap_neural_protein=True,
        p2_environment=P2_ENV_NATIVE,
        p2_switch_age=54.0,
        p2_patch_period=15.0,
        p2_cue_period=10.0,
        p2_cue_duration=3.5,
        p2_cue_delay=2.5,
        p2_reward_duration=3.0,
        p2_reward_fuel_pulse=0.90,
        p2_reward_mineral_pulse=0.36,
        p2_cue_alt_pulse=0.12,
        p2_relative_patch_radius=0.115,
        p2_cue_toxic_fraction=1.0,
        p2_assay_trap_strength=8.0,
        p2_motor_magnitude=1.0,
        p2_transporter_magnitude=0.48,
        p2_material_wear=True,
        p2_recurrence=True,
        p2_prediction=True,
        p2_plasticity=True,
        p2_effectors=True,
        p2_neural_cost=True,
        p2_tissue_turnover=True,
        p2_pruning=True,
        p2_sensor_learning_rate=0.0015,
        p2_recurrent_learning_rate=0.0004,
        p2_bias_learning_rate=0.0005,
        p2_motor_learning_rate=0.0015,
        p2_ligand_learning_rate=0.10,
        p2_cue_outcome_window=8.0,
        p2_episode_harm_memory=1.25,
        p2_plasticity_warmup=34.0,
        p2_evidence_floor=0.0030,
        p2_event_scale=0.025,
        p2_external_assistance=False,
        **kwargs
    ):
        # P2 is the only neural tissue by default.  P1 remains available as a
        # separately instantiated control in the experiment harness.
        kwargs.setdefault('p1_tissue_mode', p1.TISSUE_NONE)
        kwargs.setdefault('p1_install_gene', False)
        kwargs.setdefault('p1_environment', p1.ENV_NATIVE)
        if p2_environment != P2_ENV_NATIVE:
            kwargs.setdefault('sensorimotor', False)
            kwargs.setdefault('external_inflow', False)
        if p2_environment == P2_ENV_MEANING_REVERSAL:
            kwargs.setdefault('environment_mode', 'reversal')
            kwargs.setdefault('switch_age', float(p2_switch_age))
        super(P2Config, self).__init__(**kwargs)
        mode = str(p2_tissue_mode)
        if mode not in P2_MODES:
            raise ValueError('unknown P2 tissue mode: {}'.format(mode))
        environment = str(p2_environment)
        if environment not in P2_ENVIRONMENTS:
            raise ValueError('unknown P2 environment: {}'.format(environment))
        if not bool(p2_neural_cost):
            raise ValueError(
                'P2 forbids unmetered neural tissue; unfair upper bounds must '
                'be implemented only by the external experiment harness'
            )
        if bool(p2_external_assistance):
            raise ValueError(
                'P2 runtime does not mint external assistance; use the explicit '
                'experiment-harness ledger instead'
            )
        self.p2_tissue_mode = mode
        self.p2_install_genes = bool(p2_install_genes)
        self.p2_bootstrap_neural_protein = bool(p2_bootstrap_neural_protein)
        self.p2_environment = environment
        self.p2_switch_age = float(p2_switch_age)
        self.p2_patch_period = float(p2_patch_period)
        self.p2_cue_period = float(p2_cue_period)
        self.p2_cue_duration = float(p2_cue_duration)
        self.p2_cue_delay = float(p2_cue_delay)
        self.p2_reward_duration = float(p2_reward_duration)
        self.p2_reward_fuel_pulse = float(p2_reward_fuel_pulse)
        self.p2_reward_mineral_pulse = float(p2_reward_mineral_pulse)
        self.p2_cue_alt_pulse = float(p2_cue_alt_pulse)
        self.p2_relative_patch_radius = float(p2_relative_patch_radius)
        self.p2_cue_toxic_fraction = float(p2_cue_toxic_fraction)
        self.p2_assay_trap_strength = float(p2_assay_trap_strength)
        self.p2_motor_magnitude = float(p2_motor_magnitude)
        self.p2_transporter_magnitude = float(p2_transporter_magnitude)
        self.p2_material_wear = bool(p2_material_wear)
        self.p2_recurrence = bool(p2_recurrence)
        self.p2_prediction = bool(p2_prediction)
        self.p2_plasticity = bool(p2_plasticity)
        self.p2_effectors = bool(p2_effectors)
        self.p2_neural_cost = bool(p2_neural_cost)
        self.p2_tissue_turnover = bool(p2_tissue_turnover)
        self.p2_pruning = bool(p2_pruning)
        self.p2_sensor_learning_rate = float(p2_sensor_learning_rate)
        self.p2_recurrent_learning_rate = float(p2_recurrent_learning_rate)
        self.p2_bias_learning_rate = float(p2_bias_learning_rate)
        self.p2_motor_learning_rate = float(p2_motor_learning_rate)
        self.p2_ligand_learning_rate = float(p2_ligand_learning_rate)
        self.p2_cue_outcome_window = float(p2_cue_outcome_window)
        self.p2_episode_harm_memory = float(p2_episode_harm_memory)
        self.p2_plasticity_warmup = float(p2_plasticity_warmup)
        self.p2_evidence_floor = float(p2_evidence_floor)
        self.p2_event_scale = float(p2_event_scale)
        self.p2_external_assistance = False
        if mode == P2_MODE_FIXED:
            self.p2_prediction = False
            self.p2_plasticity = False
        elif mode == P2_MODE_NO_RECURRENCE:
            self.p2_recurrence = False
        elif mode == P2_MODE_NO_PREDICTION:
            self.p2_prediction = False
        elif mode == P2_MODE_NO_PLASTICITY:
            self.p2_plasticity = False
        elif mode == P2_MODE_NO_EFFECTOR:
            self.p2_effectors = False

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class EightCellMaterialTissue(object):
    """Eight materially paid neural cells sharing only finite physical signals."""

    def __init__(self, gene_parameters, mode=P2_MODE_FULL, rng_seed=0):
        if len(gene_parameters) != P2_CELL_COUNT:
            raise ValueError('P2 requires exactly eight decoded gene programs')
        self.mode = str(mode)
        self.gene_parameters = [dict(item) for item in gene_parameters]
        self.rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
        self.tissue_ids = [p2_tissue_id(i) for i in range(P2_CELL_COUNT)]
        self.preferred = np.asarray([item['direction'] for item in gene_parameters], dtype=float)
        self.tau = np.asarray([item['tau'] for item in gene_parameters], dtype=float)
        self.gene_lr = np.asarray([item['plasticity_lr'] for item in gene_parameters], dtype=float)
        self.initial_motor = np.asarray([item['motor_gain'] for item in gene_parameters], dtype=float)

        self.development = np.zeros(P2_CELL_COUNT, dtype=float)
        self.mature = np.zeros(P2_CELL_COUNT, dtype=bool)
        self.present = np.ones(P2_CELL_COUNT, dtype=bool)
        self.cooldown = np.zeros(P2_CELL_COUNT, dtype=float)
        self.cell_generation = np.zeros(P2_CELL_COUNT, dtype=np.int64)
        self.turnovers = np.zeros(P2_CELL_COUNT, dtype=np.int64)

        self.hidden = np.zeros(P2_CELL_COUNT, dtype=float)
        self.prev_hidden = np.zeros(P2_CELL_COUNT, dtype=float)
        # The developmental starting circuit is rotationally symmetric.  The
        # environment seed may perturb experience and chemistry, but it must
        # not secretly hand one lineage a better initial compass.
        self.bias = np.zeros(P2_CELL_COUNT, dtype=float)
        self.homeo_gain = np.ones(P2_CELL_COUNT, dtype=float)
        self.motor_gain = self.initial_motor.copy()

        self.w_sensor = np.zeros((P2_CELL_COUNT, P2_SENSOR_COUNT), dtype=float)
        self.w_sensor[:, :s4.LIGAND_COUNT] = 0.090
        self.w_sensor[:, 8:] = np.asarray([0.030, 0.016, 0.022, 0.022])[None, :]

        self.rec_mask = np.zeros((P2_CELL_COUNT, P2_CELL_COUNT), dtype=bool)
        for i in range(P2_CELL_COUNT):
            self.rec_mask[i, (i - 1) % P2_CELL_COUNT] = True
            self.rec_mask[i, (i + 1) % P2_CELL_COUNT] = True
            if i % 2 == 0:
                self.rec_mask[i, (i + 3) % P2_CELL_COUNT] = True
        np.fill_diagonal(self.rec_mask, False)
        self.w_rec = np.zeros((P2_CELL_COUNT, P2_CELL_COUNT), dtype=float)
        # A deterministic local ring gives every lineage the same short-lived
        # memory substrate.  Sparse cross-links alternate sign to prevent a
        # single global positive attractor while preserving rotational symmetry.
        for i in range(P2_CELL_COUNT):
            self.w_rec[i, (i - 1) % P2_CELL_COUNT] = 0.105
            self.w_rec[i, (i + 1) % P2_CELL_COUNT] = 0.105
            if i % 2 == 0:
                self.w_rec[i, (i + 3) % P2_CELL_COUNT] = 0.028 if (i // 2) % 2 == 0 else -0.028
        self._limit_recurrent_rows(0.52)

        self.probe_state = np.zeros(P2_CELL_COUNT, dtype=float)
        self.motor_probe = np.zeros(P2_CELL_COUNT, dtype=float)
        self.last_probe_scale = np.full(P2_CELL_COUNT, 0.06, dtype=float)
        self.last_motor_sigma = np.full(P2_CELL_COUNT, 0.06, dtype=float)
        self.signal_refractory = np.zeros(P2_CELL_COUNT, dtype=float)

        self.elig_bias = np.zeros(P2_CELL_COUNT, dtype=float)
        self.elig_sensor = np.zeros((P2_CELL_COUNT, P2_SENSOR_COUNT), dtype=float)
        self.elig_rec = np.zeros((P2_CELL_COUNT, P2_CELL_COUNT), dtype=float)
        self.elig_motor = np.zeros(P2_CELL_COUNT, dtype=float)
        self.ligand_trace = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.cue_eligibility = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.cue_episode_timer = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.cue_episode_active = np.zeros(s4.LIGAND_COUNT, dtype=bool)
        self.cue_episode_start_debt = np.zeros((s4.LIGAND_COUNT, P2_PHYSICAL_CHANNELS), dtype=float)
        self.cue_episode_peak_debt = np.zeros((s4.LIGAND_COUNT, P2_PHYSICAL_CHANNELS), dtype=float)
        self.cue_episode_peak = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.cue_episode_outcome = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.cue_episode_updates = 0
        self.uptake_trace = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.last_ligand_concentration = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.last_uptake_signal = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.last_ligand_credit = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.last_ligand_update = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self.ligand_updates = 0

        self.predict_w = np.zeros((P2_CELL_COUNT, P2_PREDICT_FEATURES), dtype=float)
        self.prediction = np.zeros(P2_CELL_COUNT, dtype=float)
        self.prediction_error = np.full(P2_CELL_COUNT, 0.55, dtype=float)
        self.last_predict_features = np.zeros((P2_CELL_COUNT, P2_PREDICT_FEATURES), dtype=float)
        self.last_prediction_rms = 0.0
        self.predictor_updates = 0

        self.local_profiles = np.zeros((P2_CELL_COUNT, P2_PHYSICAL_CHANNELS), dtype=float)
        for i in range(P2_CELL_COUNT):
            angle = 2.0 * math.pi * i / P2_CELL_COUNT
            raw = np.asarray([
                1.0 + 0.18 * math.cos(angle),
                1.0 + 0.18 * math.sin(angle),
                1.0 - 0.18 * math.cos(angle),
                1.0 - 0.18 * math.sin(angle),
            ], dtype=float)
            self.local_profiles[i] = raw / float(np.sum(raw))
        self.causal_estimate = np.zeros((P2_CELL_COUNT, P2_PHYSICAL_CHANNELS), dtype=float)
        self.maturity = np.full(P2_CELL_COUNT, 0.08, dtype=float)
        self.reopen_reserve = np.zeros(P2_CELL_COUNT, dtype=float)
        self.activity_mean = np.zeros(P2_CELL_COUNT, dtype=float)
        self.activity_square = np.zeros(P2_CELL_COUNT, dtype=float)
        self.edge_correlation = np.zeros((P2_CELL_COUNT, P2_CELL_COUNT), dtype=float)

        self.last_features = np.zeros((P2_CELL_COUNT, P2_SENSOR_COUNT), dtype=float)
        self.last_sensor_drive = np.zeros(P2_CELL_COUNT, dtype=float)
        self.last_recurrent_drive = np.zeros(P2_CELL_COUNT, dtype=float)
        self.last_action = np.zeros(2, dtype=float)
        self.last_debt = np.ones(P2_PHYSICAL_CHANNELS, dtype=float) * 0.5
        self.last_debt_valid = False
        self.physical_baseline = np.zeros(P2_PHYSICAL_CHANNELS, dtype=float)
        self.last_physical_improvement = np.zeros(P2_PHYSICAL_CHANNELS, dtype=float)
        self.last_local_modulator = np.zeros(P2_CELL_COUNT, dtype=float)
        self.last_effector_report = {}
        self.last_gateway = 0

        self.material_status = np.zeros((P2_CELL_COUNT, p0.TISSUE_MATERIAL_COUNT), dtype=float)
        self.store_status = np.zeros((P2_CELL_COUNT, p0.BUDGET_COUNT), dtype=float)
        self.gene_activity = np.zeros(P2_CELL_COUNT, dtype=float)

        self.step_count = 0
        self.active_age = 0.0
        self.sensor_steps = 0
        self.activity_steps = 0
        self.plasticity_updates = 0
        self.recurrent_messages = 0
        self.rewire_count = 0
        self.cell_prune_count = 0
        self.tissue_rebuild_count = 0
        self.cumulative_motor_force = 0.0
        self.cumulative_atp_spent = 0.0
        self.cumulative_signal_spent = 0.0
        self.cumulative_wear_material = 0.0
        self.cumulative_external_assistance = 0.0
        self.pending_wear = np.zeros(P2_CELL_COUNT, dtype=float)

    @property
    def recurrence_enabled(self):
        return self.mode != P2_MODE_NO_RECURRENCE

    @property
    def prediction_enabled(self):
        return self.mode not in (P2_MODE_FIXED, P2_MODE_NO_PREDICTION)

    @property
    def plasticity_enabled(self):
        return self.mode not in (P2_MODE_FIXED, P2_MODE_NO_PLASTICITY)

    @property
    def effectors_enabled(self):
        return self.mode != P2_MODE_NO_EFFECTOR

    def _limit_recurrent_rows(self, limit):
        for i in range(P2_CELL_COUNT):
            norm = float(np.sum(np.abs(self.w_rec[i])))
            if norm > limit and norm > 1e-12:
                self.w_rec[i] *= float(limit) / norm
        self.w_rec *= self.rec_mask

    def _reset_numerical_cell(self, index, preserve_generation=True):
        i = int(index)
        generation = int(self.cell_generation[i])
        self.hidden[i] = 0.0
        self.prev_hidden[i] = 0.0
        self.bias[i] = 0.0
        self.homeo_gain[i] = 1.0
        self.motor_gain[i] = self.initial_motor[i]
        self.w_sensor[i] = 0.0
        self.w_sensor[i, :s4.LIGAND_COUNT] = 0.090
        self.w_sensor[i, 8:] = np.asarray([0.030, 0.016, 0.022, 0.022])
        self.w_rec[i] = 0.0
        for j in range(P2_CELL_COUNT):
            if self.rec_mask[j, i]:
                self.w_rec[j, i] = 0.0
        self.w_rec[i, (i - 1) % P2_CELL_COUNT] = 0.095
        self.w_rec[i, (i + 1) % P2_CELL_COUNT] = 0.095
        if i % 2 == 0 and self.rec_mask[i, (i + 3) % P2_CELL_COUNT]:
            self.w_rec[i, (i + 3) % P2_CELL_COUNT] = 0.025 if (i // 2) % 2 == 0 else -0.025
        self.predict_w[i] = 0.0
        self.prediction[i] = 0.0
        self.prediction_error[i] = 0.55
        self.signal_refractory[i] = 0.0
        self.elig_bias[i] = 0.0
        self.elig_sensor[i] = 0.0
        self.elig_rec[i] = 0.0
        self.elig_rec[:, i] = 0.0
        self.elig_motor[i] = 0.0
        self.causal_estimate[i] = 0.0
        self.maturity[i] = 0.05
        self.reopen_reserve[i] = 0.0
        self.activity_mean[i] = 0.0
        self.activity_square[i] = 0.0
        self.edge_correlation[i] = 0.0
        self.edge_correlation[:, i] = 0.0
        self.development[i] = 0.0
        self.mature[i] = False
        if preserve_generation:
            self.cell_generation[i] = generation
        self._limit_recurrent_rows(0.52)

    def _status(self, port, index):
        status = port.attachment_status(self.tissue_ids[int(index)])
        self.material_status[int(index)] = np.asarray(status['tissue_material'], dtype=float)
        self.store_status[int(index)] = np.asarray(status['stores'], dtype=float)
        return status

    def ensure_attachments(self, port, activity):
        self.gene_activity = np.asarray(activity, dtype=float).copy()
        for i in range(P2_CELL_COUNT):
            if self.cooldown[i] > 0.0:
                continue
            if activity[i] < 0.016:
                continue
            tissue_id = self.tissue_ids[i]
            try:
                port.attachment_status(tissue_id)
            except KeyError:
                kind = P2_FIXED_KIND if self.mode == P2_MODE_FIXED else P2_NEURAL_KIND
                port.attach(tissue_id, kind=kind)
                self.present[i] = True
                self._reset_numerical_cell(i)
                self.tissue_rebuild_count += 1

    def _develop_cell(self, port, index, dt, costs=True):
        i = int(index)
        status = self._status(port, i)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        protein_need = max(0.0, P2_TARGET_PROTEIN - float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]))
        membrane_need = max(0.0, P2_TARGET_MEMBRANE - float(tissue[p0.TISSUE_MEMBRANE]))
        signal_total = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        signal_need = max(0.0, P2_TARGET_SIGNAL - signal_total)
        if protein_need + membrane_need + signal_need > 1e-12:
            request = {
                'atp': 0.0012 if costs else 0.0,
                'protein': min(protein_need, 0.00032),
                'membrane': min(membrane_need, 0.00013),
                'signal': min(signal_need, 0.00013),
            }
            port.allocate_budget(self.tissue_ids[i], request, dt)
            status = self._status(port, i)
            stores = np.asarray(status['stores'], dtype=float)
            port.commit_material(
                self.tissue_ids[i],
                protein=min(protein_need, float(stores[p0.BUDGET_PROTEIN])),
                membrane=min(membrane_need, float(stores[p0.BUDGET_MEMBRANE])),
                signal=min(signal_need, float(stores[p0.BUDGET_SIGNAL])),
            )
            status = self._status(port, i)
            tissue = np.asarray(status['tissue_material'], dtype=float)
            stores = np.asarray(status['stores'], dtype=float)
        functional = float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])
        membrane = float(tissue[p0.TISSUE_MEMBRANE])
        signal = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        ratios = np.asarray([
            functional / P2_TARGET_PROTEIN,
            membrane / P2_TARGET_MEMBRANE,
            signal / P2_TARGET_SIGNAL,
        ], dtype=float)
        self.development[i] = float(clamp(np.min(ratios), 0.0, 1.0))
        self.mature[i] = bool(
            functional >= P2_MATURE_PROTEIN
            and membrane >= P2_MATURE_MEMBRANE
            and signal >= P2_MATURE_SIGNAL
        )
        return status

    def _turnover_cell(self, port, index, reason):
        i = int(index)
        try:
            port.return_dead_tissue(self.tissue_ids[i], reason=reason)
        except KeyError:
            pass
        self.present[i] = False
        self.mature[i] = False
        self.cooldown[i] = P2_REBUILD_COOLDOWN
        self.turnovers[i] += 1
        self.cell_generation[i] += 1
        self.cell_prune_count += 1
        self._reset_numerical_cell(i)

    def _maintenance_cell(self, port, index, dt, config):
        i = int(index)
        status = self._status(port, i)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        if config.p2_tissue_turnover and (
            float(tissue[p0.TISSUE_DAMAGED_PROTEIN]) >= P2_TURNOVER_DAMAGE
            or float(tissue[p0.TISSUE_AGGREGATE]) >= P2_TURNOVER_AGGREGATE
        ):
            self._turnover_cell(port, i, 'p2-neural-damage-turnover')
            return False
        stores = np.asarray(status['stores'], dtype=float)
        target_atp = 0.0020
        target_signal = 0.00034
        base_wear = 0.000017 * dt
        activity_wear = 0.000020 * dt * (
            abs(float(self.hidden[i]))
            + 0.45 * float(self.signal_refractory[i])
            + 0.35 * abs(float(self.last_local_modulator[i]))
        )
        wear_request = base_wear + activity_wear + float(self.pending_wear[i])
        request = {
            'atp': max(0.0, target_atp - float(stores[p0.BUDGET_ATP])),
            'signal': max(0.0, target_signal - float(stores[p0.BUDGET_SIGNAL])),
            'protein': wear_request if config.p2_material_wear else 0.0,
            'membrane': 0.0,
        }
        if any(v > 1e-14 for v in request.values()):
            port.allocate_budget(self.tissue_ids[i], request, dt)
        if config.p2_material_wear and wear_request > 0.0:
            status = self._status(port, i)
            stores = np.asarray(status['stores'], dtype=float)
            amount = min(float(stores[p0.BUDGET_PROTEIN]), wear_request)
            if amount > 0.0:
                built = port.commit_material(
                    self.tissue_ids[i], protein=amount,
                    damaged_fraction=0.74, aggregate_fraction=0.16,
                )
                wear = float(built['damaged_protein'] + built['aggregate'])
                self.cumulative_wear_material += wear
        self.pending_wear[i] = 0.0
        return True

    def _material_capacity(self):
        functional = self.material_status[:, p0.TISSUE_FUNCTIONAL_PROTEIN]
        membrane = self.material_status[:, p0.TISSUE_MEMBRANE]
        signal = self.material_status[:, p0.TISSUE_SIGNAL] + self.store_status[:, p0.BUDGET_SIGNAL]
        capacity = np.minimum.reduce((
            functional / max(P2_TARGET_PROTEIN, 1e-12),
            membrane / max(P2_TARGET_MEMBRANE, 1e-12),
            signal / max(P2_TARGET_SIGNAL, 1e-12),
        ))
        return np.clip(capacity, 0.0, 1.0) * self.mature.astype(float)

    def _extract_features(self, frame):
        profiles = np.asarray(frame['external']['ligand_profiles'], dtype=float)
        gradients = np.zeros((s4.LIGAND_COUNT, 2), dtype=float)
        concentration = np.zeros(s4.LIGAND_COUNT, dtype=float)
        for ligand in range(s4.LIGAND_COUNT):
            profile = np.maximum(0.0, profiles[ligand])
            total = float(np.sum(profile))
            concentration[ligand] = math.tanh(2.5 * float(np.mean(profile)))
            if total > 1e-12:
                gradients[ligand] = np.sum(profile[:, None] * MEMBRANE_NORMALS, axis=0) / total
        directional = self.preferred.dot(gradients.T)
        features = np.zeros((P2_CELL_COUNT, P2_SENSOR_COUNT), dtype=float)
        features[:, :s4.LIGAND_COUNT] = np.clip(
            directional * concentration[None, :], -1.0, 1.0
        )
        internal = frame['internal']
        atp = float(internal['atp'])
        reactive = float(internal['reactive'])
        closure = float(internal['closure_mean'])
        retention = float(internal['retention'])
        features[:, 8] = 1.0 - clamp(atp / (0.09 + atp), 0.0, 1.0)
        features[:, 9] = clamp(reactive / (0.07 + reactive), 0.0, 1.0)
        features[:, 10] = clamp(1.0 - closure, 0.0, 1.0)
        features[:, 11] = clamp(1.0 - retention, 0.0, 1.0)
        return features, gradients, concentration

    def _physical_debt(self, frame):
        internal = frame['internal']
        atp = max(0.0, float(internal['atp']))
        fuel = max(0.0, float(internal['fuel']))
        energy_capacity = math.sqrt(
            clamp(atp / (0.075 + atp), 0.0, 1.0)
            * clamp(fuel / (0.11 + fuel), 0.0, 1.0)
        )
        closure = clamp(float(internal['closure_mean']), 0.0, 1.0)
        retention = clamp(float(internal['retention']), 0.0, 1.0)
        boundary_capacity = math.sqrt(max(0.0, closure * retention))
        reactive = clamp(float(internal['reactive']) / 0.10, 0.0, 1.0)
        damaged = clamp(float(internal['damaged_protein']) / 0.16, 0.0, 1.0)
        aggregate = clamp(float(internal['aggregate']) / 0.09, 0.0, 1.0)
        lesion = clamp(float(internal['genome_lesion']) / 0.45, 0.0, 1.0)
        damage_debt = clamp(
            0.36 * reactive + 0.28 * damaged + 0.18 * aggregate + 0.18 * lesion,
            0.0, 1.0,
        )
        process_capacity = math.sqrt(max(
            0.0,
            clamp(float(internal['proteostasis']), 0.0, 1.0)
            * clamp(float(internal['reaction_loop']), 0.0, 1.0),
        ))
        return np.asarray([
            1.0 - energy_capacity,
            1.0 - boundary_capacity,
            damage_debt,
            1.0 - process_capacity,
        ], dtype=float)

    def _gateway_index(self, capacity):
        valid = np.flatnonzero(capacity > 0.25)
        if valid.size == 0:
            return -1
        return int(valid[self.step_count % len(valid)])

    def pre_step(self, port, dt, config, gene_activity):
        dt = clamp(float(dt), 1.0 / 240.0, 0.10)
        for i in range(P2_CELL_COUNT):
            self.cooldown[i] = max(0.0, self.cooldown[i] - dt)
        self.ensure_attachments(port, gene_activity)
        for i in range(P2_CELL_COUNT):
            if self.cooldown[i] > 0.0:
                continue
            try:
                status = self._status(port, i)
                tissue_material = np.asarray(status['tissue_material'], dtype=float)
                if config.p2_tissue_turnover and (
                    float(tissue_material[p0.TISSUE_DAMAGED_PROTEIN]) >= P2_TURNOVER_DAMAGE
                    or float(tissue_material[p0.TISSUE_AGGREGATE]) >= P2_TURNOVER_AGGREGATE
                ):
                    self._turnover_cell(port, i, 'p2-neural-damage-turnover')
                    continue
                self._develop_cell(port, i, dt, costs=config.p2_neural_cost)
            except KeyError:
                self.present[i] = False
                self.mature[i] = False
                continue
            if self.mature[i] and not self._maintenance_cell(port, i, dt, config):
                continue
        capacity = self._material_capacity()
        valid = np.flatnonzero(capacity > 0.15)
        if valid.size == 0:
            return {'status': 'developing', 'mature_cells': 0}
        frame = port.raw_sensor_fluxes(self.tissue_ids[int(valid[0])])
        features, _, concentration = self._extract_features(frame)
        debt = self._physical_debt(frame)
        ligand_decay = math.exp(-dt / P2_ELIGIBILITY_TAU)
        current_concentration = np.clip(concentration, 0.0, 1.0)
        onset = np.maximum(0.0, current_concentration - self.last_ligand_concentration)
        self.ligand_trace = np.maximum(
            ligand_decay * self.ligand_trace, current_concentration,
        )
        self.cue_eligibility = np.clip(
            ligand_decay * self.cue_eligibility + onset, 0.0, 1.5
        )
        self.last_ligand_concentration = current_concentration
        for ligand in range(P2_EXTERNAL_LIGAND_COUNT):
            if onset[ligand] > 0.0020 and not self.cue_episode_active[ligand]:
                self.cue_episode_active[ligand] = True
                self.cue_episode_timer[ligand] = max(1.0, config.p2_cue_outcome_window)
                self.cue_episode_start_debt[ligand] = debt
                self.cue_episode_peak_debt[ligand] = debt
                self.cue_episode_peak[ligand] = float(onset[ligand])
        self.prev_hidden = self.hidden.copy()

        rho = math.exp(-dt / 0.36)
        fresh = math.sqrt(max(0.0, 1.0 - rho * rho))
        self.probe_state = rho * self.probe_state + fresh * self.rng.normal(0.0, 1.0, P2_CELL_COUNT)
        self.motor_probe = rho * self.motor_probe + fresh * self.rng.normal(0.0, 1.0, P2_CELL_COUNT)
        plasticity_open = np.clip(
            0.10 + 0.70 * (1.0 - self.maturity)
            + 0.25 * self.prediction_error + 0.35 * self.reopen_reserve,
            0.10, 1.0,
        )
        tissue_stress = float(np.mean(debt))
        probe_scale = 0.018 + 0.052 * plasticity_open + 0.018 * tissue_stress
        motor_sigma = 0.020 + 0.060 * plasticity_open + 0.018 * tissue_stress
        self.last_probe_scale = probe_scale.copy()
        self.last_motor_sigma = motor_sigma.copy()

        sensory_drive = np.sum(self.w_sensor * features, axis=1) / math.sqrt(P2_SENSOR_COUNT * 0.50)
        raw_recurrent = self.w_rec.dot(self.prev_hidden)
        signal_capacity = np.clip(
            (self.material_status[:, p0.TISSUE_SIGNAL] + self.store_status[:, p0.BUDGET_SIGNAL])
            / max(P2_TARGET_SIGNAL, 1e-12), 0.0, 1.0,
        )
        signal_available = signal_capacity * (1.0 - np.clip(self.signal_refractory, 0.0, 0.95))
        recurrent_drive = raw_recurrent * signal_available * capacity
        effective_recurrent = recurrent_drive if config.p2_recurrence and self.recurrence_enabled else np.zeros_like(recurrent_drive)
        drive = sensory_drive + effective_recurrent + probe_scale * self.probe_state + self.bias
        target = np.tanh(self.homeo_gain * drive)
        leak = 1.0 - np.exp(-dt / np.maximum(self.tau, 0.06))
        self.hidden = self.prev_hidden + leak * (target - self.prev_hidden)
        self.hidden *= capacity

        effective_motor = self.motor_gain + motor_sigma * self.motor_probe
        local_motor_scalar = capacity * self.hidden * effective_motor
        local_actions = (
            np.tanh(2.6 * local_motor_scalar)[:, None] * self.preferred
        )
        raw_action = np.sum(local_actions, axis=0) / math.sqrt(P2_CELL_COUNT)
        action = np.tanh(2.8 * raw_action)
        self.last_action = action.copy()

        decay = math.exp(-dt / P2_ELIGIBILITY_TAU)
        normalized_probe = self.probe_state / np.maximum(probe_scale, 0.015)
        normalized_motor = self.motor_probe / np.maximum(motor_sigma, 0.015)
        self.elig_bias = decay * self.elig_bias + normalized_probe * dt
        self.elig_sensor = decay * self.elig_sensor + normalized_probe[:, None] * features * dt
        self.elig_rec = (
            decay * self.elig_rec
            + normalized_probe[:, None] * self.prev_hidden[None, :] * self.rec_mask * dt
        )
        self.elig_motor = decay * self.elig_motor + normalized_motor * self.hidden * dt
        np.clip(self.elig_bias, -12.0, 12.0, out=self.elig_bias)
        np.clip(self.elig_sensor, -12.0, 12.0, out=self.elig_sensor)
        np.clip(self.elig_rec, -8.0, 8.0, out=self.elig_rec)
        np.clip(self.elig_motor, -10.0, 10.0, out=self.elig_motor)

        action_projection = self.preferred.dot(action)
        self.last_predict_features = np.column_stack((
            np.ones(P2_CELL_COUNT),
            self.hidden,
            self.prev_hidden,
            action_projection,
            effective_recurrent,
            np.clip(self.prediction_error, 0.0, 1.5),
        ))
        self.prediction = np.tanh(np.sum(self.predict_w * self.last_predict_features, axis=1))

        use = np.minimum(
            0.95,
            dt * 0.70 * (
                np.sum(np.abs(self.w_rec) * np.abs(self.prev_hidden)[None, :], axis=1)
                + 0.20 * np.abs(self.hidden)
            ),
        )
        recovery = dt * (0.65 + 0.55 * np.clip(self.store_status[:, p0.BUDGET_ATP] / 0.002, 0.0, 1.0))
        self.signal_refractory = np.clip(
            self.signal_refractory + use - recovery * self.signal_refractory,
            0.0, 0.96,
        )
        self.recurrent_messages += int(np.count_nonzero(
            self.rec_mask & (np.abs(self.prev_hidden[None, :]) > 0.08)
        ))

        report = {}
        gateway = self._gateway_index(capacity)
        self.last_gateway = gateway
        if gateway >= 0 and config.p2_effectors and self.effectors_enabled:
            motor = action * clamp(config.p2_motor_magnitude, 0.0, 1.0)
            transporter = action * clamp(config.p2_transporter_magnitude, 0.0, 1.0)
            report = port.apply_effector_fluxes(
                self.tissue_ids[gateway],
                {'motor': motor, 'transporter_polarity': transporter}, dt,
            )
            self.cumulative_motor_force += float(report.get('motor_force', 0.0))
            self.cumulative_atp_spent += float(report.get('atp_spent', 0.0))
            self.cumulative_signal_spent += float(report.get('signal_spent', 0.0))
        self.last_effector_report = dict(report)
        self.last_features = features.copy()
        self.last_sensor_drive = sensory_drive.copy()
        self.last_recurrent_drive = effective_recurrent.copy()
        self.last_debt = debt.copy()
        self.last_debt_valid = True
        self.sensor_steps += 1
        self.activity_steps += 1
        return {
            'status': 'active',
            'mature_cells': int(np.count_nonzero(self.mature)),
            'action': action.copy(),
            'prediction': self.prediction.copy(),
            'effector': dict(report),
        }

    def _ligand_episode_delta(self, config, outcome_scalar, peak):
        # Deterioration opens faster than improvement.  The sign is produced
        # by measured physical debt change; no ligand identity is labelled as
        # good or bad here.
        asymmetry = 2.40 if float(outcome_scalar) < 0.0 else 0.35
        delta = (
            max(0.0, config.p2_ligand_learning_rate)
            * asymmetry
            * math.tanh(8.0 * float(outcome_scalar))
            * clamp(float(peak) / 0.018, 0.25, 1.0)
        )
        return clamp(delta, -0.085, 0.035)

    def _apply_three_factor_update(self, dt, config, local_modulator, event_gate):
        plasticity_open = np.clip(
            0.10 + 0.72 * (1.0 - self.maturity)
            + 0.28 * self.prediction_error + 0.35 * self.reopen_reserve,
            0.08, 1.0,
        )
        if not (config.p2_plasticity and self.plasticity_enabled):
            return np.zeros(P2_CELL_COUNT, dtype=float)
        gene_scale = np.clip(self.gene_lr / 0.015, 0.40, 1.60)
        factor = (
            np.clip(np.asarray(local_modulator, dtype=float), -0.30, 0.30)
            * plasticity_open * float(event_gate)
            * self.mature.astype(float) * gene_scale
        )
        self.w_sensor += (
            dt * max(0.0, config.p2_sensor_learning_rate)
            * factor[:, None] * self.elig_sensor
        )
        if config.p2_recurrence and self.recurrence_enabled:
            self.w_rec += (
                dt * max(0.0, config.p2_recurrent_learning_rate)
                * factor[:, None] * self.elig_rec * self.rec_mask
            )
        self.bias += (
            dt * max(0.0, config.p2_bias_learning_rate)
            * factor * self.elig_bias
        )
        self.motor_gain += (
            dt * max(0.0, config.p2_motor_learning_rate)
            * factor * self.elig_motor
        )
        np.clip(self.w_sensor, -P2_MAX_SENSOR_WEIGHT, P2_MAX_SENSOR_WEIGHT, out=self.w_sensor)
        np.clip(self.w_rec, -P2_MAX_REC_WEIGHT, P2_MAX_REC_WEIGHT, out=self.w_rec)
        np.clip(self.bias, -1.0, 1.0, out=self.bias)
        np.clip(self.motor_gain, P2_MIN_MOTOR_GAIN, P2_MAX_MOTOR_GAIN, out=self.motor_gain)
        self.w_rec *= self.rec_mask
        self._limit_recurrent_rows(0.72)
        self.plasticity_updates += int(np.count_nonzero(np.abs(factor) > 1e-10))
        self.pending_wear += dt * 0.000020 * np.clip(np.abs(factor), 0.0, 0.30)
        return factor

    def post_step(self, port, dt, config):
        if not self.last_debt_valid:
            return {'status': 'inactive'}
        valid = np.flatnonzero(self.mature)
        if valid.size == 0:
            return {'status': 'inactive'}
        self.active_age += dt
        plasticity_ready = self.active_age >= max(0.0, config.p2_plasticity_warmup)
        frame = port.raw_sensor_fluxes(self.tissue_ids[int(valid[0])])
        next_features, _, next_concentration = self._extract_features(frame)
        next_debt = self._physical_debt(frame)
        flux_uptake = np.asarray(
            frame['flux']['last_uptake_by_ligand'], dtype=float
        )[:s4.LIGAND_COUNT]
        uptake_signal = np.tanh(22.0 * np.maximum(0.0, flux_uptake))
        ligand_decay = math.exp(-dt / P2_ELIGIBILITY_TAU)
        self.uptake_trace = np.maximum(
            ligand_decay * self.uptake_trace, uptake_signal
        )
        self.last_uptake_signal = uptake_signal.copy()
        next_drive = np.sum(self.w_sensor * next_features, axis=1) / math.sqrt(P2_SENSOR_COUNT * 0.50)
        prediction_delta = np.tanh(next_drive) - self.prediction
        rms = float(math.sqrt(float(np.mean(prediction_delta[self.mature] ** 2)))) if np.any(self.mature) else 0.0
        self.last_prediction_rms = rms
        self.prediction_error = (
            0.985 * self.prediction_error + 0.015 * np.abs(prediction_delta)
        )

        predictor_learning = bool(config.p2_prediction and self.prediction_enabled)
        if predictor_learning:
            local_delta = prediction_delta * (1.0 - self.prediction ** 2)
            lr = self.gene_lr * dt
            self.predict_w += lr[:, None] * local_delta[:, None] * self.last_predict_features
            np.clip(self.predict_w, -1.7, 1.7, out=self.predict_w)
            self.predictor_updates += int(np.count_nonzero(self.mature))

        improvement = self.last_debt - next_debt
        scaling = np.asarray([3.4, 3.4, 3.8, 3.2], dtype=float)
        pulse = np.clip(improvement * scaling, -0.32, 0.32)
        advantage = pulse - self.physical_baseline
        self.physical_baseline = 0.997 * self.physical_baseline + 0.003 * pulse
        evidence = advantage.copy()
        evidence[np.abs(evidence) < max(0.0, config.p2_evidence_floor)] = 0.0
        event_strength = float(np.max(np.abs(evidence)))
        event_gate = clamp(
            (event_strength - max(0.0, config.p2_evidence_floor))
            / max(1e-9, config.p2_event_scale), 0.0, 1.0,
        )
        urgency = 0.35 + 0.65 * np.clip(self.last_debt + 0.12, 0.0, 1.0)
        profiles = self.local_profiles * urgency[None, :]
        profiles /= np.maximum(np.sum(profiles, axis=1, keepdims=True), 1e-12)
        local_modulator = profiles.dot(evidence)
        self.last_physical_improvement = improvement.copy()
        self.last_local_modulator = local_modulator.copy()
        # Only molecular onsets create cue eligibility.  Direct membrane
        # uptake suppresses eligibility for that same molecule, preventing a
        # reward substrate from teaching itself.  The remaining trace can span
        # a physical delay without naming any species as an a-priori cue.
        self.cue_eligibility *= np.exp(-5.0 * np.clip(uptake_signal, 0.0, 1.5))
        ligand_credit = np.clip(self.cue_eligibility, 0.0, 1.0)
        self.last_ligand_credit = ligand_credit.copy()
        physical_scalar = float(
            np.sum(urgency * evidence) / max(1e-12, np.sum(urgency))
        )
        self.last_ligand_update[:] = 0.0
        for ligand in range(P2_EXTERNAL_LIGAND_COUNT):
            if self.cue_episode_active[ligand]:
                self.cue_episode_peak_debt[ligand] = np.maximum(
                    self.cue_episode_peak_debt[ligand], next_debt,
                )
        for ligand in range(P2_EXTERNAL_LIGAND_COUNT):
            if not self.cue_episode_active[ligand]:
                continue
            self.cue_episode_timer[ligand] -= dt
            if self.cue_episode_timer[ligand] > 0.0:
                continue
            start_debt = self.cue_episode_start_debt[ligand]
            terminal_improvement = start_debt - next_debt
            acute_harm = np.maximum(
                self.cue_episode_peak_debt[ligand] - start_debt, 0.0,
            )
            outcome_vector = (
                terminal_improvement
                - max(0.0, config.p2_episode_harm_memory) * acute_harm
            )
            episode_urgency = 0.35 + 0.65 * np.clip(start_debt + 0.12, 0.0, 1.0)
            outcome_scalar = float(
                np.sum(episode_urgency * outcome_vector)
                / max(1e-12, np.sum(episode_urgency))
            )
            self.cue_episode_outcome[ligand] = outcome_scalar
            if plasticity_ready and config.p2_plasticity and self.plasticity_enabled:
                delta = self._ligand_episode_delta(
                    config, outcome_scalar, self.cue_episode_peak[ligand],
                )
                self.w_sensor[:, ligand] += delta
                self.last_ligand_update[ligand] = delta
                self.pending_wear += 0.000008 * abs(delta)
                if abs(delta) > 1e-12:
                    self.ligand_updates += 1
                    self.cue_episode_updates += 1
            self.cue_episode_active[ligand] = False
            self.cue_episode_timer[ligand] = 0.0
            self.cue_episode_peak[ligand] = 0.0
            self.cue_episode_peak_debt[ligand] = 0.0
        np.clip(self.w_sensor[:, :s4.LIGAND_COUNT], -1.2, 1.2,
                out=self.w_sensor[:, :s4.LIGAND_COUNT])
        self.cue_eligibility *= max(0.0, 1.0 - 0.12 * event_gate)

        if plasticity_ready:
            self._apply_three_factor_update(
                dt, config, local_modulator, event_gate,
            )

        causal_sample = np.clip(
            self.elig_bias[:, None] * evidence[None, :], -1.6, 1.6
        )
        self.causal_estimate = 0.994 * self.causal_estimate + 0.006 * causal_sample
        competence = np.sum(profiles * np.maximum(self.causal_estimate, 0.0), axis=1)
        prediction_spike = np.maximum(0.0, self.prediction_error - 0.38)
        self.reopen_reserve += dt * (0.14 * prediction_spike - 0.09 * self.reopen_reserve)
        np.clip(self.reopen_reserve, 0.0, 1.2, out=self.reopen_reserve)
        self.maturity += dt * (
            0.012 * competence
            + 0.0012 * (self.prediction_error < 0.20)
            - 0.018 * prediction_spike
        )
        np.clip(self.maturity, 0.0, 1.0, out=self.maturity)

        activity_rate = 1.0 - math.exp(-dt / 2.8)
        self.activity_mean = (
            (1.0 - activity_rate) * self.activity_mean
            + activity_rate * self.hidden
        )
        self.activity_square = (
            (1.0 - activity_rate) * self.activity_square
            + activity_rate * self.hidden ** 2
        )
        self.bias -= dt * 0.008 * self.activity_mean
        self.homeo_gain += dt * 0.006 * (0.15 - self.activity_square)
        np.clip(self.bias, -1.0, 1.0, out=self.bias)
        np.clip(self.homeo_gain, 0.55, 1.85, out=self.homeo_gain)
        self.edge_correlation = (
            0.996 * self.edge_correlation
            + 0.004 * np.outer(self.hidden, self.prev_hidden)
        )

        self.step_count += 1
        if (
            plasticity_ready and config.p2_pruning and config.p2_plasticity and self.plasticity_enabled
            and self.step_count % P2_REWIRE_INTERVAL == 0
        ):
            self._rewire_weak_edge()
        if (
            config.p2_pruning and config.p2_tissue_turnover
            and self.step_count % P2_CELL_PRUNE_INTERVAL == 0
        ):
            self._prune_low_utility_cell(port)
        return {
            'status': 'learned',
            'prediction_rms': rms,
            'physical_improvement': improvement.copy(),
            'local_modulator': local_modulator.copy(),
        }

    def _rewire_weak_edge(self):
        active = np.outer(self.mature, self.mature)
        candidates = np.argwhere(self.rec_mask & active)
        absent = np.argwhere((~self.rec_mask) & active & (~np.eye(P2_CELL_COUNT, dtype=bool)))
        if len(candidates) == 0 or len(absent) == 0:
            return False
        score = []
        for i, j in candidates:
            score.append((
                abs(float(self.w_rec[i, j]))
                * (0.20 + abs(float(self.edge_correlation[i, j]))),
                int(i), int(j),
            ))
        _, old_i, old_j = min(score)
        new_score = []
        for i, j in absent:
            affinity = abs(float(self.edge_correlation[i, j]))
            new_score.append((-affinity, int(i), int(j)))
        _, new_i, new_j = min(new_score)
        old_weight = float(self.w_rec[old_i, old_j])
        self.rec_mask[old_i, old_j] = False
        self.w_rec[old_i, old_j] = 0.0
        self.rec_mask[new_i, new_j] = True
        sign = 1.0 if self.edge_correlation[new_i, new_j] >= 0.0 else -1.0
        self.w_rec[new_i, new_j] = sign * max(0.025, min(0.11, 0.5 * abs(old_weight) + 0.025))
        self.pending_wear[old_i] += 0.000020
        self.pending_wear[new_i] += 0.000030
        self.rewire_count += 1
        self._limit_recurrent_rows(0.72)
        return True

    def _prune_low_utility_cell(self, port):
        mature = np.flatnonzero(self.mature)
        if mature.size < 7:
            return False
        utility = (
            0.45 * self.maturity
            + 0.35 * np.max(np.maximum(self.causal_estimate, 0.0), axis=1)
            + 0.20 * np.clip(1.0 - self.prediction_error, 0.0, 1.0)
        )
        index = int(mature[np.argmin(utility[mature])])
        if utility[index] > 0.16:
            return False
        self._turnover_cell(port, index, 'p2-low-utility-pruning')
        return True

    def force_prune_cell(self, port, index):
        self._turnover_cell(port, int(index), 'p2-validation-pruning')

    def finite(self):
        arrays = (
            self.preferred, self.tau, self.gene_lr, self.initial_motor,
            self.development, self.cooldown, self.hidden, self.prev_hidden,
            self.bias, self.homeo_gain, self.motor_gain, self.w_sensor,
            self.w_rec, self.probe_state, self.motor_probe,
            self.last_probe_scale, self.last_motor_sigma,
            self.signal_refractory, self.elig_bias, self.elig_sensor,
            self.elig_rec, self.elig_motor, self.ligand_trace,
            self.cue_eligibility, self.cue_episode_timer,
            self.cue_episode_start_debt, self.cue_episode_peak_debt,
            self.cue_episode_peak, self.cue_episode_outcome, self.uptake_trace,
            self.last_ligand_concentration,
            self.last_uptake_signal, self.last_ligand_credit,
            self.last_ligand_update, self.predict_w, self.prediction,
            self.prediction_error, self.last_predict_features,
            self.local_profiles, self.causal_estimate, self.maturity,
            self.reopen_reserve, self.activity_mean, self.activity_square,
            self.edge_correlation, self.last_features, self.last_sensor_drive,
            self.last_recurrent_drive, self.last_action, self.last_debt,
            self.physical_baseline, self.last_physical_improvement,
            self.last_local_modulator, self.material_status,
            self.store_status, self.gene_activity, self.pending_wear,
        )
        scalars = (
            self.last_prediction_rms, self.active_age, self.cumulative_motor_force,
            self.cumulative_atp_spent, self.cumulative_signal_spent,
            self.cumulative_wear_material, self.cumulative_external_assistance,
        )
        return bool(
            all(finite_array(array) for array in arrays)
            and all(np.isfinite(value) for value in scalars)
            and np.all(self.development >= -1e-12)
            and np.all(self.maturity >= -1e-12)
        )

    def diagnostics(self):
        return _readonly_mapping({
            'mode': self.mode,
            'active_age': self.active_age,
            'mature_cells': int(np.count_nonzero(self.mature)),
            'development': self.development,
            'hidden': self.hidden,
            'action': self.last_action,
            'prediction_error': self.prediction_error,
            'prediction_rms': self.last_prediction_rms,
            'maturity': self.maturity,
            'reopen_reserve': self.reopen_reserve,
            'causal_estimate': self.causal_estimate,
            'physical_improvement': self.last_physical_improvement,
            'local_modulator': self.last_local_modulator,
            'ligand_trace': self.ligand_trace,
            'cue_eligibility': self.cue_eligibility,
            'cue_episode_timer': self.cue_episode_timer,
            'cue_episode_active': self.cue_episode_active,
            'cue_episode_outcome': self.cue_episode_outcome,
            'cue_episode_peak_debt': self.cue_episode_peak_debt,
            'uptake_trace': self.uptake_trace,
            'ligand_concentration': self.last_ligand_concentration,
            'uptake_signal': self.last_uptake_signal,
            'ligand_credit': self.last_ligand_credit,
            'ligand_update': self.last_ligand_update,
            'ligand_sensor_value': np.mean(
                self.w_sensor[:, :s4.LIGAND_COUNT], axis=0
            ),
            'rec_edges': int(np.count_nonzero(self.rec_mask)),
            'rewire_count': self.rewire_count,
            'cell_prune_count': self.cell_prune_count,
            'turnovers': self.turnovers,
            'material': self.material_status,
            'stores': self.store_status,
            'motor_force_total': self.cumulative_motor_force,
            'atp_spent_total': self.cumulative_atp_spent,
            'signal_spent_total': self.cumulative_signal_spent,
            'wear_material_total': self.cumulative_wear_material,
        })

    def state_dict(self):
        return {
            'mode': self.mode,
            'gene_parameters': [dict(item) for item in self.gene_parameters],
            'rng_state': self.rng.bit_generator.state,
            'tissue_ids': list(self.tissue_ids),
            'preferred': self.preferred.copy(),
            'tau': self.tau.copy(),
            'gene_lr': self.gene_lr.copy(),
            'initial_motor': self.initial_motor.copy(),
            'development': self.development.copy(),
            'mature': self.mature.copy(),
            'present': self.present.copy(),
            'cooldown': self.cooldown.copy(),
            'cell_generation': self.cell_generation.copy(),
            'turnovers': self.turnovers.copy(),
            'hidden': self.hidden.copy(),
            'prev_hidden': self.prev_hidden.copy(),
            'bias': self.bias.copy(),
            'homeo_gain': self.homeo_gain.copy(),
            'motor_gain': self.motor_gain.copy(),
            'w_sensor': self.w_sensor.copy(),
            'rec_mask': self.rec_mask.copy(),
            'w_rec': self.w_rec.copy(),
            'probe_state': self.probe_state.copy(),
            'motor_probe': self.motor_probe.copy(),
            'last_probe_scale': self.last_probe_scale.copy(),
            'last_motor_sigma': self.last_motor_sigma.copy(),
            'signal_refractory': self.signal_refractory.copy(),
            'elig_bias': self.elig_bias.copy(),
            'elig_sensor': self.elig_sensor.copy(),
            'elig_rec': self.elig_rec.copy(),
            'elig_motor': self.elig_motor.copy(),
            'ligand_trace': self.ligand_trace.copy(),
            'cue_eligibility': self.cue_eligibility.copy(),
            'cue_episode_timer': self.cue_episode_timer.copy(),
            'cue_episode_active': self.cue_episode_active.copy(),
            'cue_episode_start_debt': self.cue_episode_start_debt.copy(),
            'cue_episode_peak_debt': self.cue_episode_peak_debt.copy(),
            'cue_episode_peak': self.cue_episode_peak.copy(),
            'cue_episode_outcome': self.cue_episode_outcome.copy(),
            'cue_episode_updates': self.cue_episode_updates,
            'uptake_trace': self.uptake_trace.copy(),
            'last_ligand_concentration': self.last_ligand_concentration.copy(),
            'last_uptake_signal': self.last_uptake_signal.copy(),
            'last_ligand_credit': self.last_ligand_credit.copy(),
            'last_ligand_update': self.last_ligand_update.copy(),
            'ligand_updates': self.ligand_updates,
            'predict_w': self.predict_w.copy(),
            'prediction': self.prediction.copy(),
            'prediction_error': self.prediction_error.copy(),
            'last_predict_features': self.last_predict_features.copy(),
            'last_prediction_rms': self.last_prediction_rms,
            'predictor_updates': self.predictor_updates,
            'local_profiles': self.local_profiles.copy(),
            'causal_estimate': self.causal_estimate.copy(),
            'maturity': self.maturity.copy(),
            'reopen_reserve': self.reopen_reserve.copy(),
            'activity_mean': self.activity_mean.copy(),
            'activity_square': self.activity_square.copy(),
            'edge_correlation': self.edge_correlation.copy(),
            'last_features': self.last_features.copy(),
            'last_sensor_drive': self.last_sensor_drive.copy(),
            'last_recurrent_drive': self.last_recurrent_drive.copy(),
            'last_action': self.last_action.copy(),
            'last_debt': self.last_debt.copy(),
            'last_debt_valid': self.last_debt_valid,
            'physical_baseline': self.physical_baseline.copy(),
            'last_physical_improvement': self.last_physical_improvement.copy(),
            'last_local_modulator': self.last_local_modulator.copy(),
            'last_effector_report': dict(self.last_effector_report),
            'last_gateway': self.last_gateway,
            'material_status': self.material_status.copy(),
            'store_status': self.store_status.copy(),
            'gene_activity': self.gene_activity.copy(),
            'step_count': self.step_count,
            'active_age': self.active_age,
            'sensor_steps': self.sensor_steps,
            'activity_steps': self.activity_steps,
            'plasticity_updates': self.plasticity_updates,
            'recurrent_messages': self.recurrent_messages,
            'rewire_count': self.rewire_count,
            'cell_prune_count': self.cell_prune_count,
            'tissue_rebuild_count': self.tissue_rebuild_count,
            'cumulative_motor_force': self.cumulative_motor_force,
            'cumulative_atp_spent': self.cumulative_atp_spent,
            'cumulative_signal_spent': self.cumulative_signal_spent,
            'cumulative_wear_material': self.cumulative_wear_material,
            'cumulative_external_assistance': self.cumulative_external_assistance,
            'pending_wear': self.pending_wear.copy(),
        }

    @classmethod
    def from_state(cls, state):
        state = dict(state)
        state.setdefault(
            'cue_episode_peak_debt',
            np.zeros((s4.LIGAND_COUNT, P2_PHYSICAL_CHANNELS), dtype=float),
        )
        tissue = cls(state['gene_parameters'], mode=state['mode'], rng_seed=0)
        tissue.rng.bit_generator.state = state['rng_state']
        for name in (
            'preferred', 'tau', 'gene_lr', 'initial_motor', 'development',
            'mature', 'present', 'cooldown', 'cell_generation', 'turnovers',
            'hidden', 'prev_hidden', 'bias', 'homeo_gain', 'motor_gain',
            'w_sensor', 'rec_mask', 'w_rec', 'probe_state', 'motor_probe',
            'last_probe_scale', 'last_motor_sigma', 'signal_refractory',
            'elig_bias', 'elig_sensor', 'elig_rec', 'elig_motor',
            'ligand_trace', 'cue_eligibility', 'cue_episode_timer',
            'cue_episode_active', 'cue_episode_start_debt',
            'cue_episode_peak_debt', 'cue_episode_peak', 'cue_episode_outcome', 'uptake_trace',
            'last_ligand_concentration',
            'last_uptake_signal', 'last_ligand_credit',
            'last_ligand_update', 'predict_w',
            'prediction', 'prediction_error', 'last_predict_features',
            'local_profiles', 'causal_estimate', 'maturity', 'reopen_reserve',
            'activity_mean', 'activity_square', 'edge_correlation',
            'last_features', 'last_sensor_drive', 'last_recurrent_drive',
            'last_action', 'last_debt', 'physical_baseline',
            'last_physical_improvement', 'last_local_modulator',
            'material_status', 'store_status', 'gene_activity', 'pending_wear',
        ):
            setattr(tissue, name, np.asarray(state[name]).copy())
        tissue.tissue_ids = list(state.get('tissue_ids', tissue.tissue_ids))
        for name in (
            'last_prediction_rms', 'predictor_updates', 'ligand_updates',
            'cue_episode_updates',
            'last_debt_valid',
            'last_gateway', 'step_count', 'active_age', 'sensor_steps', 'activity_steps',
            'plasticity_updates', 'recurrent_messages', 'rewire_count',
            'cell_prune_count', 'tissue_rebuild_count',
            'cumulative_motor_force', 'cumulative_atp_spent',
            'cumulative_signal_spent', 'cumulative_wear_material',
            'cumulative_external_assistance',
        ):
            setattr(tissue, name, state[name])
        tissue.last_effector_report = dict(state.get('last_effector_report', {}))
        return tissue


class P2ProtoCell(p1.P1ProtoCell):
    def __init__(self, *args, **kwargs):
        super(P2ProtoCell, self).__init__(*args, **kwargs)
        self._init_p2_state()

    def _init_p2_state(self):
        self.p2_tissue = None
        self.p2_tissue_births = 0
        self.p2_tissue_turnovers = 0
        self.p2_gene_fingerprints = []
        self.p2_genes_installed = False

    def split(self, world):
        daughters = super(P2ProtoCell, self).split(world)
        if daughters is None:
            return None
        for daughter in daughters:
            daughter.__class__ = P2ProtoCell
            daughter._init_p2_state()
        return daughters

    def state_dict(self):
        state = super(P2ProtoCell, self).state_dict()
        state.update({
            'cell_class': 'P2ProtoCell',
            'p2_tissue': None if self.p2_tissue is None else self.p2_tissue.state_dict(),
            'p2_tissue_births': self.p2_tissue_births,
            'p2_tissue_turnovers': self.p2_tissue_turnovers,
            'p2_gene_fingerprints': list(self.p2_gene_fingerprints),
            'p2_genes_installed': self.p2_genes_installed,
        })
        return state

    @classmethod
    def from_state(cls, rng, state):
        parent = p1.P1ProtoCell.from_state(rng, state)
        parent.__class__ = cls
        cell = parent
        cell._init_p2_state()
        if state.get('p2_tissue') is not None:
            cell.p2_tissue = EightCellMaterialTissue.from_state(state['p2_tissue'])
        cell.p2_tissue_births = int(state.get('p2_tissue_births', 0))
        cell.p2_tissue_turnovers = int(state.get('p2_tissue_turnovers', 0))
        cell.p2_gene_fingerprints = [int(v) for v in state.get('p2_gene_fingerprints', [])]
        cell.p2_genes_installed = bool(state.get('p2_genes_installed', False))
        return cell


class P2World(p1.P1World):
    PATCH_CENTRES = (
        np.asarray([0.57, 0.50]), np.asarray([0.50, 0.57]),
        np.asarray([0.43, 0.50]), np.asarray([0.50, 0.43]),
    )

    def __init__(self, seed=101, initial_cells=1, config=None):
        config = config if config is not None else P2Config()
        if not isinstance(config, P2Config):
            config = P2Config(**config.state_dict())
        super(P2World, self).__init__(seed=seed, initial_cells=initial_cells, config=config)
        self.config = config
        self.p2_seed = int(seed)
        self.p2_tissue_creations = 0
        self.p2_tissue_turnovers = 0
        self.p2_pre_steps = 0
        self.p2_post_steps = 0
        self.p2_patch_switches = 0
        self.p2_semantic_switches = 0
        self.p2_last_patch_index = -1
        self.p2_last_target_index = -1
        self.p2_last_cue_index = -1
        self.p2_last_target_center = np.asarray([0.5, 0.5], dtype=float)
        self.p2_last_cue_center = np.asarray([0.5, 0.5], dtype=float)
        self.p2_cycle_origin = np.asarray([0.5, 0.5], dtype=float)
        self.p2_switch_recorded = False
        self.p2_last_stage = 'none'
        self.p2_last_cycle = -1
        self.p2_reward_fuel_indices = np.asarray([], dtype=np.int64)
        self.p2_reward_mineral_indices = np.asarray([], dtype=np.int64)
        self.p2_cue_alt_indices = np.asarray([], dtype=np.int64)
        self.p2_reward_reservoir_fuel = 0.0
        self.p2_reward_reservoir_mineral = 0.0
        self.p2_cue_reservoir_alt = 0.0
        self.p2_reward_releases = 0
        self.p2_cue_releases = 0
        self.p2_reward_uptake_total = 0.0
        self.p2_nonreward_uptake_total = 0.0
        self.p2_cue_toxin_total = 0.0
        self.p2_reward_distance_sum = 0.0
        self.p2_reward_distance_steps = 0
        for cell in self.cells:
            cell.__class__ = P2ProtoCell
            cell._init_p2_state()
            if self.config.p2_install_genes:
                fingerprints, installed = install_p2_cassette(
                    cell, bootstrap_protein=self.config.p2_bootstrap_neural_protein,
                )
                cell.p2_gene_fingerprints = list(fingerprints)
                cell.p2_genes_installed = bool(installed)
        if self.config.p2_environment != P2_ENV_NATIVE:
            self._initialise_p2_environment()
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0
        self._ensure_all_p2_tissues()

    def _initialise_p2_environment(self):
        field = self.field
        positions, kinds, amounts = [], [], []
        neutral = np.asarray([0.5, 0.5], dtype=float)

        def add_group(kind, count, total, radius, centre):
            start_index = len(positions)
            for index in range(count):
                angle = 2.0 * math.pi * (index + 0.5) / count
                ring = radius * (0.35 + 0.65 * ((index % 5) / 4.0))
                positions.append((centre + ring * np.asarray([
                    math.cos(angle), math.sin(angle),
                ])) % 1.0)
                kinds.append(kind)
                amounts.append(float(total) / count if count else 0.0)
            return np.arange(start_index, len(positions), dtype=np.int64)

        if self.config.p2_environment == P2_ENV_CUE_REVERSAL:
            # Physical chemostat assay. Cue and reward molecules live in
            # inaccessible external reservoirs while hidden. They are emitted
            # as ordinary field particles only during their stage, then all
            # unconsumed particles of that assay kind are recaptured. This
            # avoids hidden zero-mass placeholders and keeps matter explicit.
            add_group(s5.PARTICLE_WASTE, 8, 0.08, 0.060, neutral)
            self.p2_reward_reservoir_fuel = 8.00
            self.p2_reward_reservoir_mineral = 3.60
            self.p2_cue_reservoir_alt = 1.10
        else:
            centre = self.PATCH_CENTRES[0]
            add_group(s5.PARTICLE_FUEL, 32, 3.20, 0.024, centre)
            add_group(s5.PARTICLE_MINERAL, 24, 1.60, 0.029, centre)
            add_group(s5.PARTICLE_ALT, 14, 0.24, 0.018, centre)
            add_group(s5.PARTICLE_WASTE, 10, 0.12, 0.060, neutral)

        field.pos = np.asarray(positions, dtype=float)
        field.kind = np.asarray(kinds, dtype=np.int16)
        field.amount = np.asarray(amounts, dtype=float)
        field.recycled_fuel_buffer = 0.0
        field.recycled_mineral_buffer = 0.0
        field.injected_material = 0.0
        field.dissipated_material = 0.0
        for index, cell in enumerate(self.cells):
            cell.pos = np.asarray([0.50, 0.50 + 0.015 * index], dtype=float) % 1.0
            cell.vel[:] = 0.0
        if self.config.p2_environment == P2_ENV_CUE_REVERSAL:
            direction = np.asarray([1.0, 0.0], dtype=float)
            radius = clamp(self.config.p2_relative_patch_radius, 0.075, 0.22)
            self.p2_cycle_origin = np.asarray([0.5, 0.5], dtype=float)
            centre = (self.p2_cycle_origin + radius * direction) % 1.0
        else:
            centre = self.PATCH_CENTRES[0]
            self.p2_cycle_origin = np.asarray([0.5, 0.5], dtype=float)
        self.p2_last_target_center = centre.copy()
        self.p2_last_cue_center = centre.copy()
        self.p2_last_patch_index = 0
        self.p2_last_target_index = 0
        self.p2_last_cue_index = 0

    def _assay_positions(self, count, centre, radius):
        centre = np.asarray(centre, dtype=float)
        result = []
        for local in range(int(count)):
            angle = 2.0 * math.pi * (local + 0.5) / max(1, int(count))
            ring = radius * (0.35 + 0.65 * ((local % 5) / 4.0))
            result.append((centre + ring * np.asarray([
                math.cos(angle), math.sin(angle),
            ])) % 1.0)
        return np.asarray(result, dtype=float)

    def _recapture_kind(self, kind, reservoir_name):
        indices = np.where(self.field.kind == int(kind))[0]
        if indices.size == 0:
            return 0.0
        amount = float(np.sum(np.maximum(0.0, self.field.amount[indices])))
        setattr(self, reservoir_name, float(getattr(self, reservoir_name)) + amount)
        keep = self.field.kind != int(kind)
        self.field.pos = self.field.pos[keep].copy()
        self.field.kind = self.field.kind[keep].copy()
        self.field.amount = self.field.amount[keep].copy()
        return amount

    def _release_kind(self, kind, reservoir_name, requested, centre, count, radius):
        available = max(0.0, float(getattr(self, reservoir_name)))
        amount = min(available, max(0.0, float(requested)))
        if amount <= 1e-12 or int(count) <= 0:
            return 0.0
        positions = self._assay_positions(int(count), centre, radius)
        amounts = np.full(int(count), amount / int(count), dtype=float)
        self.field.add_many(
            int(kind), positions, amounts, count_as_injection=False,
        )
        setattr(self, reservoir_name, available - amount)
        return amount

    def _cue_cycle(self):
        period = max(8.0, self.config.p2_cue_period)
        return int(math.floor(self.age / period))

    def _phase_indices(self):
        env = self.config.p2_environment
        if env == P2_ENV_MOVING_PATCH:
            index = int(math.floor(self.age / max(2.0, self.config.p2_patch_period))) % 4
            return index, index, 'reward'
        if env == P2_ENV_CUE_REVERSAL:
            period = max(8.0, self.config.p2_cue_period)
            cycle = int(math.floor(self.age / period))
            phase = self.age - cycle * period
            cue = cycle % 4
            reversed_rule = self.age >= self.config.p2_switch_age
            target = (cue + 2) % 4 if reversed_rule else cue
            cue_end = self.config.p2_cue_duration
            delay_end = cue_end + self.config.p2_cue_delay
            reward_end = delay_end + self.config.p2_reward_duration
            if phase < cue_end:
                stage = 'cue'
            elif phase < delay_end:
                stage = 'delay'
            elif phase < reward_end:
                stage = 'reward'
            else:
                stage = 'rest'
            return cue, target, stage
        if env == P2_ENV_MEANING_REVERSAL:
            return 0, 0, 'reward'
        return -1, -1, 'native'

    def _move_kind_to(self, kind, centre, dt, strength=1.0, radius=0.025):
        indices = np.where(self.field.kind == int(kind))[0]
        count = len(indices)
        if count == 0:
            return
        centre = np.asarray(centre, dtype=float)
        blend = clamp(dt * float(strength), 0.0, 0.30)
        for local, particle_index in enumerate(indices):
            angle = 2.0 * math.pi * (local + 0.5) / count
            ring = radius * (0.35 + 0.65 * ((local % 5) / 4.0))
            target = (centre + ring * np.asarray([
                math.cos(angle), math.sin(angle),
            ])) % 1.0
            delta = wrapped_delta(self.field.pos[particle_index], target)
            self.field.pos[particle_index] = (
                self.field.pos[particle_index] + blend * delta
            ) % 1.0

    def _maintain_p2_environment(self, dt):
        cue, target, stage = self._phase_indices()
        if cue < 0:
            return
        neutral = np.asarray([0.5, 0.5], dtype=float)
        if self.config.p2_environment == P2_ENV_CUE_REVERSAL:
            cycle = self._cue_cycle()
            if cycle != self.p2_last_cycle:
                alive = self.living_cells()
                if alive:
                    # The assay places each new cue relative to the organism's
                    # actual current position.  The cell is never teleported;
                    # only the chemostat's physical cue/reward reservoirs move.
                    origin = np.mean([cell.pos for cell in alive], axis=0) % 1.0
                else:
                    origin = neutral.copy()
                angle = 2.0 * math.pi * cue / P2_CELL_COUNT
                cue_direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
                reversed_rule = self.age >= self.config.p2_switch_age
                target_direction = -cue_direction if reversed_rule else cue_direction
                radius = clamp(self.config.p2_relative_patch_radius, 0.075, 0.22)
                self.p2_cycle_origin = origin.copy()
                self.p2_last_cue_center = (origin + radius * cue_direction) % 1.0
                self.p2_last_target_center = (origin + radius * target_direction) % 1.0
            cue_center = self.p2_last_cue_center.copy()
            target_center = self.p2_last_target_center.copy()
        else:
            target_center = self.PATCH_CENTRES[target]
            cue_center = self.PATCH_CENTRES[cue]
        if target != self.p2_last_target_index:
            self.p2_patch_switches += 1
        self.p2_last_target_index = target
        self.p2_last_cue_index = cue
        if self.config.p2_environment != P2_ENV_CUE_REVERSAL:
            self.p2_last_target_center = target_center.copy()
            self.p2_last_cue_center = cue_center.copy()
        if (
            self.config.p2_environment in (P2_ENV_CUE_REVERSAL, P2_ENV_MEANING_REVERSAL)
            and self.age >= self.config.p2_switch_age
            and not self.p2_switch_recorded
        ):
            self.p2_semantic_switches += 1
            self.p2_switch_recorded = True

        if self.config.p2_environment == P2_ENV_CUE_REVERSAL:
            cycle = self._cue_cycle()
            changed_stage = stage != self.p2_last_stage or cycle != self.p2_last_cycle
            if changed_stage:
                if self.p2_last_stage == 'cue':
                    self._recapture_kind(
                        s5.PARTICLE_ALT, 'p2_cue_reservoir_alt'
                    )
                if self.p2_last_stage == 'reward':
                    self._recapture_kind(
                        s5.PARTICLE_FUEL, 'p2_reward_reservoir_fuel'
                    )
                    self._recapture_kind(
                        s5.PARTICLE_MINERAL, 'p2_reward_reservoir_mineral'
                    )
                if stage == 'cue':
                    released = self._release_kind(
                        s5.PARTICLE_ALT, 'p2_cue_reservoir_alt',
                        self.config.p2_cue_alt_pulse, cue_center, 14, 0.017,
                    )
                    if released > 0.0:
                        self.p2_cue_releases += 1
                elif stage == 'reward':
                    released_fuel = self._release_kind(
                        s5.PARTICLE_FUEL, 'p2_reward_reservoir_fuel',
                        self.config.p2_reward_fuel_pulse, target_center, 24, 0.024,
                    )
                    released_mineral = self._release_kind(
                        s5.PARTICLE_MINERAL, 'p2_reward_reservoir_mineral',
                        self.config.p2_reward_mineral_pulse, target_center, 18, 0.029,
                    )
                    if released_fuel + released_mineral > 0.0:
                        self.p2_reward_releases += 1
                self.p2_last_stage = stage
                self.p2_last_cycle = cycle
            # The assay is a chemostat with a local physical source/trap.  The
            # ordinary extracellular particles still diffuse, but while a
            # stage is active the trap advects unconsumed molecules back
            # toward the source.  Without this, the baseline diffusion length
            # over a multi-second delay is larger than the cue-target
            # separation and the task ceases to be spatial.
            trap = max(0.0, float(self.config.p2_assay_trap_strength))
            if trap > 0.0:
                if stage == 'cue':
                    self._move_kind_to(
                        s5.PARTICLE_ALT, cue_center, dt,
                        strength=trap, radius=0.017,
                    )
                elif stage == 'reward':
                    self._move_kind_to(
                        s5.PARTICLE_FUEL, target_center, dt,
                        strength=trap, radius=0.024,
                    )
                    self._move_kind_to(
                        s5.PARTICLE_MINERAL, target_center, dt,
                        strength=trap, radius=0.029,
                    )
        elif self.config.p2_environment == P2_ENV_MEANING_REVERSAL:
            self._move_kind_to(s5.PARTICLE_FUEL, self.PATCH_CENTRES[0], dt, strength=2.0, radius=0.024)
            self._move_kind_to(s5.PARTICLE_ALT, self.PATCH_CENTRES[2], dt, strength=2.0, radius=0.024)
            self._move_kind_to(s5.PARTICLE_MINERAL, neutral, dt, strength=1.5, radius=0.035)
        else:
            self._move_kind_to(s5.PARTICLE_FUEL, target_center, dt, strength=2.4, radius=0.024)
            self._move_kind_to(s5.PARTICLE_MINERAL, target_center, dt, strength=2.0, radius=0.029)
            self._move_kind_to(s5.PARTICLE_ALT, neutral, dt, strength=1.4, radius=0.030)

    def total_material(self):
        base = super(P2World, self).total_material()
        reservoir = (
            max(0.0, float(getattr(self, 'p2_reward_reservoir_fuel', 0.0)))
            + max(0.0, float(getattr(self, 'p2_reward_reservoir_mineral', 0.0)))
            + max(0.0, float(getattr(self, 'p2_cue_reservoir_alt', 0.0)))
        )
        return float(base + reservoir)

    def _new_p2_tissue(self, cell):
        if self.config.p2_tissue_mode == P2_MODE_NONE:
            return None
        parameters = p2_gene_parameters(cell)
        activity = p2_gene_activity(cell)
        if parameters is None or np.count_nonzero(activity >= 0.016) < P2_CELL_COUNT:
            return None
        seed = (
            (self.p2_seed * 1000003)
            ^ (int(cell.cell_id) * 9176)
            ^ (int(cell.generation) * 7919)
            ^ 0x6A2B31
        ) & 0xFFFFFFFF
        tissue = EightCellMaterialTissue(
            parameters, mode=self.config.p2_tissue_mode, rng_seed=seed,
        )
        port = self.port_for(cell.cell_id)
        tissue.ensure_attachments(port, activity)
        cell.p2_tissue = tissue
        cell.p2_tissue_births += 1
        self.p2_tissue_creations += 1
        return tissue

    def _ensure_p2_tissue(self, cell):
        if not cell.alive or self.config.p2_tissue_mode == P2_MODE_NONE:
            return None
        if cell.p2_tissue is not None:
            return cell.p2_tissue
        for index in range(P2_CELL_COUNT):
            tissue_id = p2_tissue_id(index)
            if tissue_id in cell.neural_attachments:
                self.port_for(cell.cell_id).return_dead_tissue(
                    tissue_id, reason='orphaned-p2-attachment'
                )
        return self._new_p2_tissue(cell)

    def _ensure_all_p2_tissues(self):
        for cell in self.living_cells():
            if not isinstance(cell, P2ProtoCell):
                cell.__class__ = P2ProtoCell
                cell._init_p2_state()
            self._ensure_p2_tissue(cell)

    def _pre_p2_step(self, dt):
        for cell in list(self.living_cells()):
            tissue = self._ensure_p2_tissue(cell)
            if tissue is None:
                continue
            tissue.pre_step(
                self.port_for(cell.cell_id), dt, self.config, p2_gene_activity(cell)
            )
            self.p2_pre_steps += 1

    def _post_p2_step(self, dt):
        for cell in list(self.living_cells()):
            tissue = cell.p2_tissue if isinstance(cell, P2ProtoCell) else None
            if tissue is None:
                continue
            tissue.post_step(self.port_for(cell.cell_id), dt, self.config)
            self.p2_post_steps += 1
        self._ensure_all_p2_tissues()

    def _release_dead_cell(self, cell):
        if isinstance(cell, P2ProtoCell):
            cell.p2_tissue = None
        return super(P2World, self)._release_dead_cell(cell)

    def _apply_cue_toxicity(self, stage):
        if not (
            self.config.p2_environment == P2_ENV_CUE_REVERSAL
            and self.age >= self.config.p2_switch_age
            and stage == 'cue'
        ):
            return 0.0
        total = 0.0
        fraction = clamp(self.config.p2_cue_toxic_fraction, 0.0, 1.0)
        for cell in self.living_cells():
            uptake = max(0.0, float(getattr(cell, 'last_uptake_alt', 0.0)))
            converted = min(
                max(0.0, float(cell.pools[s4.POOL_ALT])), uptake * fraction,
            )
            if converted <= 0.0:
                continue
            # A changed external chemistry converts the formerly neutral cue
            # substrate into a reactive product after membrane crossing.  This
            # is a mass-preserving molecular consequence, not a reward flag.
            cell.pools[s4.POOL_ALT] -= converted
            cell.pools[s4.POOL_REACTIVE] += converted
            cell.damage_trace += converted / s4.MEMBRANE_SEGMENTS * 1.6
            total += converted
        self.p2_cue_toxin_total += total
        return total

    def step(self, dt):
        dt = clamp(float(dt), 1.0 / 240.0, 0.10)
        self._maintain_p2_environment(dt)
        stage = self._phase_indices()[2]
        before_uptake = float(sum(
            np.sum(cell.cumulative_uptake_by_ligand[:3])
            for cell in self.living_cells()
        ))
        self._pre_p2_step(dt)
        super(P2World, self).step(dt)
        self._apply_cue_toxicity(stage)
        self._post_p2_step(dt)
        after_uptake = float(sum(
            np.sum(cell.cumulative_uptake_by_ligand[:3])
            for cell in self.living_cells()
        ))
        uptake = max(0.0, after_uptake - before_uptake)
        if stage == 'reward':
            self.p2_reward_uptake_total += uptake
            distances = [self.target_distance(cell) for cell in self.living_cells()]
            distances = [value for value in distances if np.isfinite(value)]
            if distances:
                self.p2_reward_distance_sum += float(np.mean(distances))
                self.p2_reward_distance_steps += 1
        else:
            self.p2_nonreward_uptake_total += uptake

    def target_distance(self, cell):
        cue, target, stage = self._phase_indices()
        if target < 0:
            return float('nan')
        if self.config.p2_environment == P2_ENV_CUE_REVERSAL:
            centre = self.p2_last_target_center if stage == 'reward' else self.p2_last_cue_center
        else:
            centre = self.PATCH_CENTRES[target] if stage == 'reward' else self.PATCH_CENTRES[cue]
        return float(np.linalg.norm(wrapped_delta(cell.pos, centre)))

    def finite(self):
        if not super(P2World, self).finite():
            return False
        for cell in self.living_cells():
            if not isinstance(cell, P2ProtoCell):
                return False
            if cell.p2_tissue is not None and not cell.p2_tissue.finite():
                return False
        reservoirs = (
            self.p2_reward_reservoir_fuel, self.p2_reward_reservoir_mineral,
            self.p2_cue_reservoir_alt, self.p2_reward_uptake_total,
            self.p2_nonreward_uptake_total, self.p2_reward_distance_sum,
            self.p2_cue_toxin_total,
        )
        return bool(
            finite_array(self.p2_last_target_center)
            and finite_array(self.p2_last_cue_center)
            and finite_array(self.p2_cycle_origin)
            and all(np.isfinite(value) and value >= -1e-12 for value in reservoirs)
        )

    def summary(self):
        summary = super(P2World, self).summary()
        alive = self.living_cells()
        tissues = [
            cell.p2_tissue for cell in alive
            if isinstance(cell, P2ProtoCell) and cell.p2_tissue is not None
        ]
        distances = [self.target_distance(cell) for cell in alive]
        distances = [v for v in distances if np.isfinite(v)]
        summary.update({
            'build': BUILD,
            'p2_schema': P2_SCHEMA_VERSION,
            'p2_mode': self.config.p2_tissue_mode,
            'p2_environment': self.config.p2_environment,
            'p2_tissues': len(tissues),
            'p2_mature_cells': int(sum(np.count_nonzero(t.mature) for t in tissues)),
            'p2_tissue_creations': self.p2_tissue_creations,
            'p2_tissue_turnovers': int(sum(np.sum(t.turnovers) for t in tissues)),
            'p2_prediction_error': float(np.mean([
                np.mean(t.prediction_error[t.mature]) if np.any(t.mature) else 0.0
                for t in tissues
            ])) if tissues else 0.0,
            'p2_prediction_rms': float(np.mean([t.last_prediction_rms for t in tissues])) if tissues else 0.0,
            'p2_predictor_updates': int(sum(t.predictor_updates for t in tissues)),
            'p2_plasticity_updates': int(sum(t.plasticity_updates for t in tissues)),
            'p2_ligand_updates': int(sum(t.ligand_updates for t in tissues)),
            'p2_cue_episode_updates': int(sum(t.cue_episode_updates for t in tissues)),
            'p2_active_age': float(np.mean([t.active_age for t in tissues])) if tissues else 0.0,
            'p2_plasticity_warmup_remaining': float(np.mean([
                max(0.0, self.config.p2_plasticity_warmup - t.active_age)
                for t in tissues
            ])) if tissues else 0.0,
            'p2_mean_maturity': float(np.mean([
                np.mean(t.maturity[t.mature]) if np.any(t.mature) else 0.0
                for t in tissues
            ])) if tissues else 0.0,
            'p2_mean_abs_causal_estimate': float(np.mean([
                np.mean(np.abs(t.causal_estimate)) for t in tissues
            ])) if tissues else 0.0,
            'p2_ligand_sensor_values': (
                np.mean([
                    np.mean(t.w_sensor[:, :s4.LIGAND_COUNT], axis=0)
                    for t in tissues
                ], axis=0) if tissues else np.zeros(s4.LIGAND_COUNT)
            ),
            'p2_recurrent_messages': int(sum(t.recurrent_messages for t in tissues)),
            'p2_rewire_count': int(sum(t.rewire_count for t in tissues)),
            'p2_cell_prune_count': int(sum(t.cell_prune_count for t in tissues)),
            'p2_motor_force_total': float(sum(t.cumulative_motor_force for t in tissues)),
            'p2_neural_atp_total': float(sum(t.cumulative_atp_spent for t in tissues)),
            'p2_neural_signal_total': float(sum(t.cumulative_signal_spent for t in tissues)),
            'p2_wear_material_total': float(sum(t.cumulative_wear_material for t in tissues)),
            'p2_mean_target_distance': float(np.mean(distances)) if distances else 0.0,
            'p2_patch_switches': self.p2_patch_switches,
            'p2_semantic_switches': self.p2_semantic_switches,
            'p2_reward_releases': self.p2_reward_releases,
            'p2_cue_releases': self.p2_cue_releases,
            'p2_reward_uptake_total': self.p2_reward_uptake_total,
            'p2_nonreward_uptake_total': self.p2_nonreward_uptake_total,
            'p2_cue_toxin_total': self.p2_cue_toxin_total,
            'p2_reward_mean_distance': (
                self.p2_reward_distance_sum / max(1, self.p2_reward_distance_steps)
            ),
            'p2_reward_reservoir_fuel': self.p2_reward_reservoir_fuel,
            'p2_reward_reservoir_mineral': self.p2_reward_reservoir_mineral,
            'p2_cue_reservoir_alt': self.p2_cue_reservoir_alt,
            'p2_pre_steps': self.p2_pre_steps,
            'p2_post_steps': self.p2_post_steps,
        })
        return summary

    def state_dict(self):
        state = super(P2World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'p2_seed': self.p2_seed,
            'p2_tissue_creations': self.p2_tissue_creations,
            'p2_tissue_turnovers': self.p2_tissue_turnovers,
            'p2_pre_steps': self.p2_pre_steps,
            'p2_post_steps': self.p2_post_steps,
            'p2_patch_switches': self.p2_patch_switches,
            'p2_semantic_switches': self.p2_semantic_switches,
            'p2_last_patch_index': self.p2_last_patch_index,
            'p2_last_target_index': self.p2_last_target_index,
            'p2_last_cue_index': self.p2_last_cue_index,
            'p2_last_target_center': self.p2_last_target_center.copy(),
            'p2_last_cue_center': self.p2_last_cue_center.copy(),
            'p2_cycle_origin': self.p2_cycle_origin.copy(),
            'p2_switch_recorded': self.p2_switch_recorded,
            'p2_last_stage': self.p2_last_stage,
            'p2_last_cycle': self.p2_last_cycle,
            'p2_reward_fuel_indices': self.p2_reward_fuel_indices.copy(),
            'p2_reward_mineral_indices': self.p2_reward_mineral_indices.copy(),
            'p2_cue_alt_indices': self.p2_cue_alt_indices.copy(),
            'p2_reward_reservoir_fuel': self.p2_reward_reservoir_fuel,
            'p2_reward_reservoir_mineral': self.p2_reward_reservoir_mineral,
            'p2_cue_reservoir_alt': self.p2_cue_reservoir_alt,
            'p2_reward_releases': self.p2_reward_releases,
            'p2_cue_releases': self.p2_cue_releases,
            'p2_reward_uptake_total': self.p2_reward_uptake_total,
            'p2_nonreward_uptake_total': self.p2_nonreward_uptake_total,
            'p2_cue_toxin_total': self.p2_cue_toxin_total,
            'p2_reward_distance_sum': self.p2_reward_distance_sum,
            'p2_reward_distance_steps': self.p2_reward_distance_steps,
        })
        return state

    @classmethod
    def from_state(cls, state):
        base_state = dict(state)
        base_state['save_version'] = p1.SAVE_VERSION
        base_state['build'] = p1.BUILD
        p1_keys = set(p1.P1Config().__dict__.keys())
        base_state['config'] = {
            key: value for key, value in dict(state.get('config', {})).items()
            if key in p1_keys
        }
        world = p1.P1World.from_state(base_state)
        world.__class__ = cls
        world.config = P2Config.from_state(state.get('config', {}))
        world.cells = [P2ProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        world.p2_seed = int(state.get('p2_seed', 101))
        for name in (
            'p2_tissue_creations', 'p2_tissue_turnovers', 'p2_pre_steps',
            'p2_post_steps', 'p2_patch_switches', 'p2_semantic_switches',
            'p2_last_patch_index', 'p2_last_target_index', 'p2_last_cue_index',
        ):
            setattr(world, name, int(state.get(name, 0)))
        world.p2_last_target_center = np.asarray(
            state.get('p2_last_target_center', [0.5, 0.5]), dtype=float
        ).copy()
        world.p2_last_cue_center = np.asarray(
            state.get('p2_last_cue_center', [0.5, 0.5]), dtype=float
        ).copy()
        world.p2_cycle_origin = np.asarray(
            state.get('p2_cycle_origin', [0.5, 0.5]), dtype=float
        ).copy()
        world.p2_switch_recorded = bool(state.get('p2_switch_recorded', False))
        world.p2_last_stage = str(state.get('p2_last_stage', 'none'))
        world.p2_last_cycle = int(state.get('p2_last_cycle', -1))
        world.p2_reward_fuel_indices = np.asarray(
            state.get('p2_reward_fuel_indices', []), dtype=np.int64
        )
        world.p2_reward_mineral_indices = np.asarray(
            state.get('p2_reward_mineral_indices', []), dtype=np.int64
        )
        world.p2_cue_alt_indices = np.asarray(
            state.get('p2_cue_alt_indices', []), dtype=np.int64
        )
        for name in (
            'p2_reward_reservoir_fuel', 'p2_reward_reservoir_mineral',
            'p2_cue_reservoir_alt', 'p2_reward_uptake_total',
            'p2_nonreward_uptake_total', 'p2_cue_toxin_total',
            'p2_reward_distance_sum',
        ):
            setattr(world, name, float(state.get(name, 0.0)))
        for name in (
            'p2_reward_releases', 'p2_cue_releases',
            'p2_reward_distance_steps',
        ):
            setattr(world, name, int(state.get(name, 0)))
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return P2World.from_state(self.state_dict())


def p1_projection(world_or_state):
    state = world_or_state.state_dict() if hasattr(world_or_state, 'state_dict') else world_or_state
    p1_config_keys = set(p1.P1Config().__dict__.keys())

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                text = str(key)
                if text.startswith('p2_') or text == 'p2_tissue':
                    continue
                if text == 'cell_class' and item == 'P2ProtoCell':
                    result[key] = 'P1ProtoCell'
                elif text == 'build':
                    result[key] = p1.BUILD
                elif text == 'save_version':
                    result[key] = p1.SAVE_VERSION
                elif text == 'config':
                    result[key] = clean({k: v for k, v in item.items() if k in p1_config_keys})
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
    world = P2World(seed=seed, initial_cells=initial_cells, config=config)
    dt = 1.0 / SIM_HZ
    margin_sum = 0.0
    distance_sum = 0.0
    pre_switch_sum = 0.0
    pre_switch_steps = 0
    post_switch_sum = 0.0
    post_switch_steps = 0
    samples = 0
    uptake_start = float(sum(
        np.sum(cell.cumulative_uptake_by_ligand)
        for cell in world.living_cells()
    ))
    for _ in range(int(max(0.0, seconds) * SIM_HZ)):
        world.step(dt)
        alive = world.living_cells()
        if alive:
            margin = float(np.mean([cell.autopoietic_margin() for cell in alive]))
            margin_sum += margin
            distances = [world.target_distance(cell) for cell in alive]
            distances = [v for v in distances if np.isfinite(v)]
            if distances:
                distance_sum += float(np.mean(distances))
            samples += 1
            if world.age < world.config.p2_switch_age:
                pre_switch_sum += margin
                pre_switch_steps += 1
            else:
                post_switch_sum += margin
                post_switch_steps += 1
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
        'mean_target_distance_over_life': distance_sum / max(1, samples),
        'pre_switch_mean_margin': pre_switch_sum / max(1, pre_switch_steps),
        'post_switch_mean_margin': post_switch_sum / max(1, post_switch_steps),
        'uptake_during_trial': max(0.0, uptake_end - uptake_start),
        'finite': int(world.finite()),
        'final_mass_residual': float(world.matter_ledger_residual()),
    })
    return summary


def set_p2_runtime_mode(world, mode):
    """Switch an existing P2 world to an ablation mode without minting matter.

    This helper is intended for exact-state counterfactual experiments.  It
    changes only algorithmic gates; the material attachments, numerical state,
    and chemical body are left untouched.
    """
    mode = str(mode)
    if mode not in P2_MODES:
        raise ValueError('unknown P2 tissue mode: {}'.format(mode))
    world.config.p2_tissue_mode = mode
    world.config.p2_prediction = mode not in (P2_MODE_FIXED, P2_MODE_NO_PREDICTION)
    world.config.p2_plasticity = mode not in (P2_MODE_FIXED, P2_MODE_NO_PLASTICITY)
    world.config.p2_recurrence = mode != P2_MODE_NO_RECURRENCE
    world.config.p2_effectors = mode != P2_MODE_NO_EFFECTOR
    for cell in world.living_cells():
        tissue = getattr(cell, 'p2_tissue', None)
        if tissue is not None:
            tissue.mode = mode
    return world


def _p2_tape_rng_state(seed, step, subsystem, item=0, stream=0):
    sequence = np.random.SeedSequence([
        int(seed), int(step), int(subsystem), int(item), int(stream), 0x6232,
    ])
    return np.random.default_rng(sequence).bit_generator.state


def apply_p2_common_disturbance_tape(world, seed, step, stream=0):
    """Reset stochastic subsystems to a step-indexed common disturbance.

    Treatment and control twins may consume different numbers of random values
    after they diverge.  Re-seeding each subsystem at every step prevents that
    bookkeeping difference from becoming an unrelated future disturbance.
    This function is never called by normal life; it is an evaluation tool.
    """
    world.rng.bit_generator.state = _p2_tape_rng_state(
        seed, step, subsystem=0, stream=stream,
    )
    for cell in world.cells:
        # P0/P1/P2 cells share the world's physical RNG by contract.
        cell.rng = world.rng
        tissue = getattr(cell, 'p2_tissue', None)
        if tissue is not None:
            tissue.rng.bit_generator.state = _p2_tape_rng_state(
                seed, step, subsystem=1, item=int(cell.cell_id), stream=stream,
            )
    return world


LOG_FIELDS = (
    'session_id', 'reason', 'wall_time', 'age', 'cells', 'corpses',
    'edna_fragments', 'divisions', 'deaths', 'mean_margin', 'mean_atp',
    'p2_mode', 'p2_environment', 'p2_tissues', 'p2_mature_cells',
    'p2_prediction_error', 'p2_prediction_rms', 'p2_predictor_updates',
    'p2_plasticity_updates', 'p2_cue_episode_updates', 'p2_active_age',
    'p2_plasticity_warmup_remaining', 'p2_mean_maturity',
    'p2_mean_abs_causal_estimate', 'p2_recurrent_messages', 'p2_rewire_count',
    'p2_cell_prune_count', 'p2_motor_force_total', 'p2_neural_atp_total',
    'p2_neural_signal_total', 'p2_mean_target_distance',
    'p2_patch_switches', 'p2_semantic_switches',
    'p2_reward_uptake_total', 'p2_nonreward_uptake_total',
    'p2_reward_mean_distance', 'matter_residual',
)


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), int(world.p2_seed))
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
                'mode': last.get('p2_mode', ''),
                'environment': last.get('p2_environment', ''),
            })
    lines = [BUILD_LONG, 'log: {}'.format(os.path.basename(log_path)), 'sessions: {}'.format(len(sessions)), '']
    for session_id, items in sessions.items():
        last = items[-1]
        lines.append('{} age={} cells={} mode={} env={} pred={} plastic={} rec={} distance={} ledger={}'.format(
            session_id, last.get('age', ''), last.get('cells', ''),
            last.get('p2_mode', ''), last.get('p2_environment', ''),
            last.get('p2_prediction_error', ''), last.get('p2_plasticity_updates', ''),
            last.get('p2_recurrent_messages', ''), last.get('p2_mean_target_distance', ''),
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

    class SomaCellP2Scene(p1.SomaCellP1Scene):
        def setup(self):
            background(0.006, 0.012, 0.022)
            try:
                self.world = P2World.load(SAVE_FILE)
                self.save_status = 'LOAD'
            except Exception:
                self.world = P2World(seed=101, initial_cells=2)
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
            # Draw the 0.5/P0 body without P1's one-neuron overlay/header.
            p0.SomaCellP0Scene.draw(self)
            for cell in self.world.living_cells():
                tissue = cell.p2_tissue if isinstance(cell, P2ProtoCell) else None
                if tissue is None:
                    continue
                x, y = self._screen(cell.pos)
                base_radius = max(2.0, cell.radius * min(self.size.w, self.size.h) * 0.20)
                for i in range(P2_CELL_COUNT):
                    angle = 2.0 * math.pi * i / P2_CELL_COUNT
                    cx = x + math.cos(angle) * base_radius * 1.65
                    cy = y + math.sin(angle) * base_radius * 1.65
                    radius = base_radius * (0.34 + 0.16 * tissue.development[i])
                    activity = 0.5 + 0.5 * clamp(tissue.hidden[i], -1.0, 1.0)
                    maturity = clamp(tissue.maturity[i], 0.0, 1.0)
                    fill(0.35 + 0.45 * activity, 0.18 + 0.24 * maturity, 0.72 + 0.20 * activity, 0.90)
                    ellipse(cx - radius, cy - radius, radius * 2, radius * 2)
                # Sparse recurrent edges inside the host.
                stroke(0.66, 0.44, 0.94, 0.42)
                stroke_weight(0.7)
                for i, j in np.argwhere(tissue.rec_mask):
                    ai = 2.0 * math.pi * int(i) / P2_CELL_COUNT
                    aj = 2.0 * math.pi * int(j) / P2_CELL_COUNT
                    line(
                        x + math.cos(aj) * base_radius * 1.65,
                        y + math.sin(aj) * base_radius * 1.65,
                        x + math.cos(ai) * base_radius * 1.65,
                        y + math.sin(ai) * base_radius * 1.65,
                    )
                stroke(0.95, 0.76, 1.0, 0.90)
                stroke_weight(1.4)
                line(
                    x, y,
                    x + tissue.last_action[0] * base_radius * 3.2,
                    y + tissue.last_action[1] * base_radius * 3.2,
                )

            fill(0.010, 0.018, 0.030, 1.0)
            rect(0, self.size.h - 48, self.size.w, 48)
            rect(0, 0, self.size.w, 104)
            s = self.world.summary()
            fill(0.94, 0.98, 1.0)
            text(BUILD, x=24, y=self.size.h - 25, font_size=18, alignment=4)
            fill(0.65, 0.80, 0.88)
            text('{} | {} | SAVE {} | {:.1f} fps | x{:.2f}'.format(
                s['p2_mode'], s['p2_environment'], self.save_status,
                self.fps, self.sim_rate,
            ), x=self.size.w - 72, y=self.size.h - 25, font_size=9, alignment=6)
            fill(0.84, 0.92, 0.97)
            text('age {:.1f}s cells {} div {} deaths {} corpses {} DNA {}'.format(
                s['age'], s['cells'], s['divisions'], s['deaths'],
                s['corpses'], s['edna_fragments'],
            ), x=24, y=90, font_size=10, alignment=4)
            text('tissues {} neural cells {} rec msg {} rewires {} prunes {}'.format(
                s['p2_tissues'], s['p2_mature_cells'], s['p2_recurrent_messages'],
                s['p2_rewire_count'], s['p2_cell_prune_count'],
            ), x=24, y=72, font_size=9, alignment=4)
            text('predict {:.4f} RMS {:.4f} updates {} plastic {}'.format(
                s['p2_prediction_error'], s['p2_prediction_rms'],
                s['p2_predictor_updates'], s['p2_plasticity_updates'],
            ), x=24, y=54, font_size=9, alignment=4)
            text('motor {:.4f} ATP {:.5f} signal {:.5f} wear {:.5f}'.format(
                s['p2_motor_force_total'], s['p2_neural_atp_total'],
                s['p2_neural_signal_total'], s['p2_wear_material_total'],
            ), x=24, y=36, font_size=9, alignment=4)
            text('target d {:.4f} switches {}/{} ledger {:+.2e}'.format(
                s['p2_mean_target_distance'], s['p2_patch_switches'],
                s['p2_semantic_switches'], s['matter_residual'],
            ), x=24, y=18, font_size=9, alignment=4)

        def touch_began(self, touch):
            now = time.time()
            if now - self.last_touch_wall < 0.42:
                try:
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                except Exception:
                    pass
                self.world = P2World(seed=101, initial_cells=2)
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
            seed=101, seconds=45.0, initial_cells=1,
            config=P2Config(p2_environment=P2_ENV_CUE_REVERSAL),
        ))
    else:
        run(SomaCellP2Scene(), LANDSCAPE, show_fps=False)
