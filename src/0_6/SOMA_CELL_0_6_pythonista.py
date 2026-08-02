# coding: utf-8
"""SOMA-CELL 0.6 — materially paid causal self-auditing tissue.

This module freezes SOMA-CELL 0.6-P2 as the physical substrate and adds a
small gene-derived controller that can run reversible, counterbalanced stop
experiments on the eight-cell tissue.  The controller pays ATP/material costs,
calibrates its causal claims, opens re-plasticity only when physical change is
plausible, and routes its matter through the ordinary corpse/eDNA chemistry.

No external reward, answer label, free learned-state inheritance, direct pose
rewrite, or free gene transfer is introduced.
"""
from __future__ import division

import csv
import gc
import hashlib
import math
import os
import pickle
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p2'))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, P2_DIR, P1_DIR, P0_DIR, BASE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_P2_pythonista as p2

p1 = p2.p1
p0 = p2.p0
s5 = p2.s5
s4 = p2.s4
g2 = s4.g2

BUILD = 'SOMA-CELL 0.6.0'
BUILD_LONG = BUILD + ' | material causal audit / calibration / change gate'
SAVE_VERSION = 1
FORMAL_SCHEMA_VERSION = '0.6-F2.0'
SAVE_FILE = 'soma_cell_0_6.pkl'
LOG_FILE = 'soma_cell_0_6_longrun.csv'
REPORT_FILE = 'soma_cell_0_6_report.txt'
SESSION_FILE = 'soma_cell_0_6_sessions.csv'
AUTO_SAVE_INTERVAL = 60.0
SIM_HZ = p2.SIM_HZ

clamp = p2.clamp
finite_array = p2.finite_array
_atomic_pickle = p2._atomic_pickle

FORMAL_MODE_FULL = 'full'
FORMAL_MODE_NO_AUDIT = 'no_audit'
FORMAL_MODE_NO_CALIBRATION = 'no_calibration'
FORMAL_MODE_NO_CHANGE_GATE = 'no_change_gate'
FORMAL_MODE_NO_FEEDBACK = 'no_feedback'
FORMAL_MODE_NO_FALSIFICATION = 'no_falsification'
FORMAL_MODE_NO_HGT = 'no_hgt'
FORMAL_MODES = frozenset((
    FORMAL_MODE_FULL, FORMAL_MODE_NO_AUDIT, FORMAL_MODE_NO_CALIBRATION,
    FORMAL_MODE_NO_CHANGE_GATE, FORMAL_MODE_NO_FEEDBACK,
    FORMAL_MODE_NO_FALSIFICATION, FORMAL_MODE_NO_HGT,
))

FORMAL_CONTROLLER_ID = 'formal-material-causal-auditor'
FORMAL_CONTROLLER_KIND = 'material-causal-audit-controller'
FORMAL_GENE_EFFECT = s4.EFFECT_RESERVED
FORMAL_GENE_CHANNEL = s4.CONTROL_RESERVED_7
FORMAL_GENE_LOCALISATION = s4.LOC_EFFECTOR
FORMAL_BOOTSTRAP_PROTEIN = 0.0050
FORMAL_TARGET_PROTEIN = 0.0080
FORMAL_TARGET_MEMBRANE = 0.0022
FORMAL_TARGET_SIGNAL = 0.0060
FORMAL_MATURE_PROTEIN = 0.0062
FORMAL_MATURE_MEMBRANE = 0.0016
FORMAL_MATURE_SIGNAL = 0.0040
FORMAL_DAMAGE_LIMIT = 0.0052
FORMAL_AGGREGATE_LIMIT = 0.0022
FORMAL_GROUP_SIZE = 2
FORMAL_CHANNELS = p2.P2_PHYSICAL_CHANNELS
FORMAL_CONTEXT_DIM = 12
FORMAL_STATE_FEATURE_DIM = 8
FORMAL_MECHANISM_FEATURE_DIM = 8
FORMAL_FEATURE_DIM = FORMAL_STATE_FEATURE_DIM + FORMAL_MECHANISM_FEATURE_DIM
FORMAL_LIGAND_COUNT = p2.P2_EXTERNAL_LIGAND_COUNT
FORMAL_CHANGE_SOURCES = ('state', 'mechanism', 'semantics', 'genome')
FORMAL_MAX_COMMANDS = 2
FORMAL_CHANGE_EVENT_THRESHOLD = 0.55

AUDIT_EVENT_RESOURCE = 1 << 0
AUDIT_EVENT_TOXIN = 1 << 1
AUDIT_EVENT_CORPSE = 1 << 2
AUDIT_EVENT_EDNA = 1 << 3
AUDIT_EVENT_HGT = 1 << 4
AUDIT_EVENT_DIVISION = 1 << 5
AUDIT_EVENT_CONTROLLER_LIMIT = 1 << 6
AUDIT_HARD_EVENT_MASK = AUDIT_EVENT_RESOURCE | AUDIT_EVENT_TOXIN | AUDIT_EVENT_HGT | AUDIT_EVENT_DIVISION


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _sign(value, eps=1e-12):
    value = float(value)
    if value > eps:
        return 1.0
    if value < -eps:
        return -1.0
    return 0.0


def make_formal_controller_gene(promoter=5, efficiency=5, fidelity=6):
    return s4.make_effector_gene(
        FORMAL_GENE_EFFECT, FORMAL_GENE_CHANNEL,
        gain=5, rate=5, cost=5, geometry=4,
        promoter=promoter, efficiency=efficiency, fidelity=fidelity,
    )


def is_formal_controller_spec(spec):
    return bool(
        spec is not None
        and int(spec.get('role', -1)) == int(s4.ROLE_REGULATOR)
        and int(spec.get('localisation', -1)) == int(FORMAL_GENE_LOCALISATION)
        and int(spec.get('parameter', -1)) % 8 == int(FORMAL_GENE_EFFECT)
        and int(spec.get('regulator', -1)) % 8 == int(FORMAL_GENE_CHANNEL)
    )


def formal_controller_specs(cell):
    return [
        (int(fingerprint), spec)
        for fingerprint, spec in cell.gene_specs.items()
        if is_formal_controller_spec(spec)
    ]


def formal_controller_activity(cell):
    total = 0.0
    for fingerprint, spec in formal_controller_specs(cell):
        total += (
            max(0.0, float(cell.proteins.get(fingerprint, 0.0)))
            * float(spec.get('promoter', 0.0))
            * float(spec.get('efficiency', 0.0))
        )
    return float(total / 0.014)


def install_formal_cassette(cell, bootstrap_protein=True):
    existing = formal_controller_specs(cell)
    if existing:
        return int(existing[0][0]), False
    gene = make_formal_controller_gene()
    if len(cell.genomes[0]) + len(gene) > g2.MAX_GENOME_LENGTH:
        raise ValueError('no physical room for formal controller gene')
    cell.genomes[0] = np.concatenate([cell.genomes[0], gene]).astype(np.uint8)
    cell.pools[s5.POOL_NUCLEOTIDE] += len(gene) * s5.MONOMER_MASS
    cell._refresh_gene_cache()
    specs = formal_controller_specs(cell)
    if not specs:
        raise AssertionError('formal controller gene did not decode')
    fingerprint = int(specs[0][0])
    if bootstrap_protein:
        cell.proteins[fingerprint] = cell.proteins.get(fingerprint, 0.0) + FORMAL_BOOTSTRAP_PROTEIN
    cell._sync_protein_pool()
    return fingerprint, True


class Formal06Config(p2.P2Config):
    def __init__(
        self,
        formal_mode=FORMAL_MODE_FULL,
        formal_install_gene=True,
        formal_bootstrap_protein=True,
        audit_enabled=True,
        calibration_enabled=True,
        change_detection_enabled=True,
        change_gated_feedback=True,
        feedback_enabled=True,
        falsification_enabled=True,
        neural_hgt_enabled=True,
        audit_symmetric_order=True,
        audit_group_size=FORMAL_GROUP_SIZE,
        audit_measure_duration=0.24,
        audit_washout_duration=0.05,
        audit_max_retries=1,
        audit_min_active_age=40.0,
        audit_cooldown=14.0,
        audit_evidence_half_life=120.0,
        audit_knockout_strength=0.88,
        audit_safe_debt=0.82,
        audit_safety_structural_delta=0.30,
        audit_safety_structural_absolute=0.93,
        audit_hard_uptake_threshold=0.0015,
        audit_hard_reactive_threshold=0.0010,
        audit_controller_cost=True,
        audit_freeze_learning=True,
        change_warmup=24.0,
        change_fast_tau=1.4,
        change_slow_tau=22.0,
        change_state_threshold=2.75,
        change_mechanism_threshold=1.85,
        change_semantic_threshold=1.35,
        change_semantic_window=3.2,
        change_semantic_harm_memory=0.75,
        change_semantic_min_episodes=6.0,
        change_semantic_min_effect=0.0045,
        change_feedback_threshold=0.38,
        feedback_floor=0.018,
        feedback_weight_decay=0.22,
        feedback_lease_loss=0.46,
        feedback_lease_min=0.22,
        feedback_lease_recovery_tau=48.0,
        falsification_allow_rescue=True,
        falsification_rescue_threshold=0.085,
        formal_auto_start=True,
        **kwargs
    ):
        kwargs.setdefault('p2_tissue_mode', p2.P2_MODE_FULL)
        super(Formal06Config, self).__init__(**kwargs)
        formal_mode = str(formal_mode)
        if formal_mode not in FORMAL_MODES:
            raise ValueError('unknown formal mode: {}'.format(formal_mode))
        if not bool(audit_controller_cost):
            raise ValueError('formal 0.6 forbids costless audit control')
        self.formal_mode = formal_mode
        self.formal_install_gene = bool(formal_install_gene)
        self.formal_bootstrap_protein = bool(formal_bootstrap_protein)
        self.audit_enabled = bool(audit_enabled) and formal_mode != FORMAL_MODE_NO_AUDIT
        self.calibration_enabled = bool(calibration_enabled) and formal_mode != FORMAL_MODE_NO_CALIBRATION
        self.change_detection_enabled = bool(change_detection_enabled)
        self.change_gated_feedback = bool(change_gated_feedback) and formal_mode != FORMAL_MODE_NO_CHANGE_GATE
        self.feedback_enabled = bool(feedback_enabled) and formal_mode != FORMAL_MODE_NO_FEEDBACK
        self.falsification_enabled = bool(falsification_enabled) and formal_mode != FORMAL_MODE_NO_FALSIFICATION
        self.neural_hgt_enabled = bool(neural_hgt_enabled) and formal_mode != FORMAL_MODE_NO_HGT
        self.audit_symmetric_order = bool(audit_symmetric_order)
        self.audit_group_size = int(audit_group_size)
        if self.audit_group_size < 1 or 2 * self.audit_group_size > p2.P2_CELL_COUNT:
            raise ValueError('audit group size incompatible with eight-cell tissue')
        self.audit_measure_duration = float(audit_measure_duration)
        self.audit_washout_duration = float(audit_washout_duration)
        self.audit_max_retries = int(audit_max_retries)
        self.audit_min_active_age = float(audit_min_active_age)
        self.audit_cooldown = float(audit_cooldown)
        self.audit_evidence_half_life = float(audit_evidence_half_life)
        self.audit_knockout_strength = float(audit_knockout_strength)
        self.audit_safe_debt = float(audit_safe_debt)
        self.audit_safety_structural_delta = float(audit_safety_structural_delta)
        self.audit_safety_structural_absolute = float(audit_safety_structural_absolute)
        self.audit_hard_uptake_threshold = float(audit_hard_uptake_threshold)
        self.audit_hard_reactive_threshold = float(audit_hard_reactive_threshold)
        self.audit_controller_cost = True
        self.audit_freeze_learning = bool(audit_freeze_learning)
        self.change_warmup = float(change_warmup)
        self.change_fast_tau = float(change_fast_tau)
        self.change_slow_tau = float(change_slow_tau)
        self.change_state_threshold = float(change_state_threshold)
        self.change_mechanism_threshold = float(change_mechanism_threshold)
        self.change_semantic_threshold = float(change_semantic_threshold)
        self.change_semantic_window = float(change_semantic_window)
        self.change_semantic_harm_memory = float(change_semantic_harm_memory)
        self.change_semantic_min_episodes = float(change_semantic_min_episodes)
        self.change_semantic_min_effect = float(change_semantic_min_effect)
        self.change_feedback_threshold = float(change_feedback_threshold)
        self.feedback_floor = float(feedback_floor)
        self.feedback_weight_decay = float(feedback_weight_decay)
        self.feedback_lease_loss = float(feedback_lease_loss)
        self.feedback_lease_min = float(feedback_lease_min)
        self.feedback_lease_recovery_tau = float(feedback_lease_recovery_tau)
        self.falsification_allow_rescue = bool(falsification_allow_rescue)
        self.falsification_rescue_threshold = float(falsification_rescue_threshold)
        self.formal_auto_start = bool(formal_auto_start)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class CausalCalibrationLedger(object):
    def __init__(self):
        self.weight = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sum_raw = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sum_obs = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sum_raw2 = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sum_raw_obs = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.raw_abs_error = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.cal_abs_error = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sign_correct = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.sign_weight = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.updates = np.zeros(FORMAL_CHANNELS, dtype=np.int64)
        self.cell_trust = np.full((p2.P2_CELL_COUNT, FORMAL_CHANNELS), 0.45, dtype=float)
        self.last_raw = self.last_calibrated = self.last_observed = 0.0
        self.last_reliability = 0.0
        self.last_channel = 0

    def _regression(self, channel):
        k = int(channel)
        w = float(self.weight[k])
        if w <= 1e-9:
            return 0.0, 1.0
        mx = self.sum_raw[k] / w; my = self.sum_obs[k] / w
        vx = max(0.0, self.sum_raw2[k] / w - mx * mx)
        cov = self.sum_raw_obs[k] / w - mx * my
        slope = clamp(cov / (vx + 2.5e-7), -4.0, 4.0)
        intercept = clamp(my - slope * mx, -0.012, 0.012)
        return float(intercept), float(slope)

    def reliability(self, channel):
        k = int(channel); w = float(self.weight[k])
        if w <= 1e-9:
            return 0.08
        coverage = 1.0 - math.exp(-w / 3.0)
        sign_acc = self.sign_correct[k] / max(self.sign_weight[k], 1e-9)
        mae = self.cal_abs_error[k] / max(w, 1e-9)
        return clamp(coverage * (0.35 + 0.65 * sign_acc) * math.exp(-mae / 0.0035), 0.02, 1.0)

    def predict(self, channel, raw_prediction):
        k = int(channel); raw = float(raw_prediction)
        intercept, slope = self._regression(k)
        coverage = 1.0 - math.exp(-float(self.weight[k]) / 2.5)
        calibrated = (1.0 - coverage) * raw + coverage * (intercept + slope * raw)
        return float(clamp(calibrated, -0.018, 0.018)), float(self.reliability(k))

    def update(self, channel, raw_prediction, observed_effect, quality, claim_mask):
        k = int(channel); raw = float(raw_prediction); obs = float(observed_effect)
        q = clamp(float(quality), 0.02, 1.0)
        cal, rel = self.predict(k, raw)
        self.weight[k] += q; self.sum_raw[k] += q * raw; self.sum_obs[k] += q * obs
        self.sum_raw2[k] += q * raw * raw; self.sum_raw_obs[k] += q * raw * obs
        self.raw_abs_error[k] += q * abs(raw - obs); self.cal_abs_error[k] += q * abs(cal - obs)
        cs = _sign(cal, 1e-6); osign = _sign(obs, 1e-6)
        if cs != 0.0 and osign != 0.0:
            self.sign_weight[k] += q; self.sign_correct[k] += q * float(cs == osign)
        self.updates[k] += 1
        target = clamp(0.15 + 0.55 * math.exp(-abs(cal - obs) / 0.004) + 0.30 * float(cs == 0.0 or osign == 0.0 or cs == osign), 0.05, 1.0)
        alpha = 0.05 + 0.18 * q
        for i in np.flatnonzero(np.asarray(claim_mask, dtype=bool)):
            self.cell_trust[i, k] += alpha * (target - self.cell_trust[i, k])
        np.clip(self.cell_trust, 0.04, 1.0, out=self.cell_trust)
        self.last_raw, self.last_calibrated, self.last_observed = raw, cal, obs
        self.last_reliability, self.last_channel = rel, k
        return {'raw_prediction': raw, 'calibrated_prediction': cal, 'observed_effect': obs, 'quality': q, 'reliability': rel}

    def diagnostics(self):
        total = float(np.sum(self.weight))
        return {
            'updates': int(np.sum(self.updates)),
            'raw_mae': float(np.sum(self.raw_abs_error) / max(total, 1e-9)),
            'calibrated_mae': float(np.sum(self.cal_abs_error) / max(total, 1e-9)),
            'sign_accuracy': float(np.sum(self.sign_correct) / max(np.sum(self.sign_weight), 1e-9)),
        }

    def state_dict(self):
        return {name: getattr(self, name).copy() if isinstance(getattr(self, name), np.ndarray) else getattr(self, name) for name in (
            'weight','sum_raw','sum_obs','sum_raw2','sum_raw_obs','raw_abs_error','cal_abs_error','sign_correct','sign_weight','updates','cell_trust','last_raw','last_calibrated','last_observed','last_reliability','last_channel')}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for name, value in state.items():
            setattr(obj, name, np.asarray(value).copy() if isinstance(getattr(obj, name, None), np.ndarray) else value)
        return obj


class MaterialChangeSentinel(object):
    """Source-resolved online change detector.

    Ordinary metabolic state variation is tracked for diagnosis, but it cannot
    by itself open the strong re-plasticity gate.  Strong feedback is driven by
    changes in body/neural mechanics, repeated changes in the consequences of
    an actually transported ligand, or explicit material genome/tissue events.
    """

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.state_fast = np.zeros(FORMAL_STATE_FEATURE_DIM, dtype=float)
        self.state_slow = np.zeros(FORMAL_STATE_FEATURE_DIM, dtype=float)
        self.state_var = np.ones(FORMAL_STATE_FEATURE_DIM, dtype=float) * 0.01
        self.state_count = np.zeros(FORMAL_STATE_FEATURE_DIM, dtype=float)
        self.mech_fast = np.zeros(FORMAL_MECHANISM_FEATURE_DIM, dtype=float)
        self.mech_slow = np.zeros(FORMAL_MECHANISM_FEATURE_DIM, dtype=float)
        self.mech_var = np.ones(FORMAL_MECHANISM_FEATURE_DIM, dtype=float) * 0.01
        self.mech_count = np.zeros(FORMAL_MECHANISM_FEATURE_DIM, dtype=float)
        self.semantic_fast = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.semantic_slow = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.semantic_var = np.ones((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float) * 0.0004
        self.semantic_count = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.last_state_z = np.zeros(FORMAL_STATE_FEATURE_DIM, dtype=float)
        self.last_mechanism_z = np.zeros(FORMAL_MECHANISM_FEATURE_DIM, dtype=float)
        self.last_semantic_z = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.state_probability = 0.0
        self.mechanism_probability = 0.0
        self.semantic_probability = 0.0
        self.genome_probability = 0.0
        self.global_probability = 0.0
        self.channel_probability = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.mechanism_channel_probability = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.semantic_channel_probability = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.source_probability = np.zeros(len(FORMAL_CHANGE_SOURCES), dtype=float)
        self.state_accumulator = 0.0
        self.mechanism_accumulator = 0.0
        self.semantic_accumulator = np.zeros(FORMAL_CHANNELS, dtype=float)
        self.initialised = False
        self.armed = False
        self.events = 0
        self.last_event_age = -1e9
        self.max_probability = 0.0
        self.last_dominant_source = 'state'
        self.semantic_events = 0
        self.mechanism_events = 0
        self.genome_events = 0
        self._state_floor = np.asarray([0.018, 0.020, 0.016, 0.020, 0.055, 0.055, 0.018, 0.018], dtype=float)
        self._mech_floor = np.asarray([0.055, 0.055, 0.035, 0.035, 0.018, 0.018, 0.022, 0.035], dtype=float)
        self._semantic_floor = np.asarray([0.010, 0.010, 0.010, 0.010], dtype=float)
        # feature -> physical debt channel attribution
        self._mech_channel = np.asarray([
            [0.42, 0.06, 0.18, 0.72],  # longitudinal motor transfer
            [0.24, 0.05, 0.24, 0.52],  # lateral slip
            [0.36, 0.18, 0.38, 0.66],  # tissue capacity
            [0.12, 0.10, 0.88, 0.64],  # neural damage fraction
            [0.08, 0.94, 0.48, 0.20],  # membrane closure
            [0.30, 0.72, 0.62, 0.28],  # material retention
            [0.10, 0.12, 0.94, 0.74],  # genome lesion
            [0.34, 0.10, 0.36, 0.70],  # controller capacity
        ], dtype=float)

    @staticmethod
    def _smooth(current, target, dt, rise_tau, fall_tau):
        tau = rise_tau if target > current else fall_tau
        alpha = 1.0 - math.exp(-float(dt) / max(float(tau), 1e-6))
        return float(current + alpha * (target - current))

    @staticmethod
    def _combined_probability(values):
        values = np.clip(np.asarray(values, dtype=float), 0.0, 0.995)
        return float(1.0 - np.prod(1.0 - values))

    def _update_dense(self, x, mask, dt, fast, slow, var, count, floor, config):
        x = np.asarray(x, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        af = 1.0 - math.exp(-dt / max(config.change_fast_tau, 0.1))
        ass = 1.0 - math.exp(-dt / max(config.change_slow_tau, 0.5))
        z = np.zeros_like(x)
        for j in np.flatnonzero(mask & np.isfinite(x)):
            j = int(j); value = float(x[j])
            if count[j] < 1.0:
                fast[j] = value; slow[j] = value; var[j] = floor[j] ** 2; count[j] = 1.0
                continue
            residual = value - slow[j]
            cap = 36.0 * (var[j] + floor[j] ** 2)
            var[j] = (1.0 - ass) * var[j] + ass * min(residual * residual, cap)
            fast[j] += af * (value - fast[j])
            slow[j] += ass * residual
            count[j] += 1.0
            if count[j] >= 5.0:
                z[j] = abs(fast[j] - slow[j]) / (math.sqrt(max(var[j], 0.0)) + floor[j])
        return z

    def _arm_if_ready(self, age, config):
        ready = bool(
            float(age) >= config.change_warmup
            and np.count_nonzero(self.state_count >= 20.0) >= 6
            and np.count_nonzero(self.mech_count >= 20.0) >= 5
        )
        if ready and not self.armed:
            self.armed = True
            self.state_slow[:] = self.state_fast
            self.mech_slow[:] = self.mech_fast
            self.state_accumulator = 0.0
            self.mechanism_accumulator = 0.0
            # Semantic observations collected during tissue development are a
            # baseline, not evidence that the world changed.  Re-anchor every
            # observed ligand at arming time and require fresh post-warmup
            # episodes before a semantic gate can open.
            observed = self.semantic_count > 0.0
            self.semantic_slow[observed] = self.semantic_fast[observed]
            self.semantic_var[:] = self._semantic_floor[None, :] ** 2
            self.semantic_count[observed] = 1.0
            self.last_semantic_z[:] = 0.0
            self.semantic_accumulator[:] = 0.0
            self.state_probability = 0.0
            self.mechanism_probability = 0.0
            self.semantic_probability = 0.0
            self.global_probability = 0.0
            self.channel_probability[:] = 0.0
            self.mechanism_channel_probability[:] = 0.0
            self.semantic_channel_probability[:] = 0.0
        return self.armed

    def _decay_probabilities(self, dt):
        self.state_probability = self._smooth(self.state_probability, 0.0, dt, 0.8, 8.0)
        self.mechanism_probability = self._smooth(self.mechanism_probability, 0.0, dt, 0.6, 10.0)
        self.semantic_probability = self._smooth(self.semantic_probability, 0.0, dt, 0.35, 14.0)
        self.genome_probability = self._smooth(self.genome_probability, 0.0, dt, 0.25, 18.0)
        self.mechanism_channel_probability += (1.0 - math.exp(-dt / 10.0)) * (0.0 - self.mechanism_channel_probability)
        self.semantic_channel_probability += (1.0 - math.exp(-dt / 14.0)) * (0.0 - self.semantic_channel_probability)

    def update(self, state_features, mechanism_features, dt, age, config,
               self_intervention=False, event_flags=0,
               state_mask=None, mechanism_mask=None):
        state = np.asarray(state_features, dtype=float).reshape(FORMAL_STATE_FEATURE_DIM)
        mech = np.asarray(mechanism_features, dtype=float).reshape(FORMAL_MECHANISM_FEATURE_DIM)
        if state_mask is None:
            state_mask = np.ones(FORMAL_STATE_FEATURE_DIM, dtype=bool)
        if mechanism_mask is None:
            mechanism_mask = np.ones(FORMAL_MECHANISM_FEATURE_DIM, dtype=bool)
        dt = clamp(float(dt), 1.0 / 240.0, 0.25)
        if not self.enabled:
            self._decay_probabilities(dt)
            self.global_probability = 0.0; self.channel_probability[:] = 0.0
            return 0.0
        self.last_state_z = self._update_dense(
            state, state_mask, dt, self.state_fast, self.state_slow,
            self.state_var, self.state_count, self._state_floor, config,
        )
        mech_mask = np.asarray(mechanism_mask, dtype=bool).copy()
        if self_intervention:
            # Motor transfer and neural capacity are directly perturbed by a stop
            # experiment.  Persistent membrane/genome evidence remains visible.
            mech_mask[:4] = False
        self.last_mechanism_z = self._update_dense(
            mech, mech_mask, dt, self.mech_fast, self.mech_slow,
            self.mech_var, self.mech_count, self._mech_floor, config,
        )
        self._decay_probabilities(dt)
        if not self._arm_if_ready(age, config):
            self.source_probability[:] = [self.state_probability, self.mechanism_probability,
                                          self.semantic_probability, self.genome_probability]
            return 0.0

        state_evidence = np.clip(
            (self.last_state_z - config.change_state_threshold) / 2.6, 0.0, 1.0,
        )
        mechanism_evidence = np.clip(
            (self.last_mechanism_z - config.change_mechanism_threshold) / 2.0, 0.0, 1.0,
        )
        state_target = float(np.mean(np.sort(state_evidence)[-3:]))
        mechanism_target = self._combined_probability(0.58 * mechanism_evidence)
        self.state_accumulator = max(0.0, math.exp(-dt / 4.5) * self.state_accumulator + dt * (1.9 * state_target - 0.18))
        self.mechanism_accumulator = max(0.0, math.exp(-dt / 6.0) * self.mechanism_accumulator + dt * (2.4 * mechanism_target - 0.11))
        state_target = max(state_target, 1.0 - math.exp(-2.1 * self.state_accumulator))
        mechanism_target = max(mechanism_target, 1.0 - math.exp(-2.8 * self.mechanism_accumulator))
        self.state_probability = self._smooth(self.state_probability, state_target, dt, 0.55, 7.5)
        self.mechanism_probability = self._smooth(self.mechanism_probability, mechanism_target, dt, 0.45, 10.0)

        for k in range(FORMAL_CHANNELS):
            weighted = np.clip(mechanism_evidence * self._mech_channel[:, k], 0.0, 0.98)
            target = self._combined_probability(weighted)
            self.mechanism_channel_probability[k] = self._smooth(
                self.mechanism_channel_probability[k], target, dt, 0.45, 10.0,
            )

        if int(event_flags) & AUDIT_EVENT_HGT:
            self.genome_probability = max(self.genome_probability, 0.86)
            self.genome_events += 1
        if int(event_flags) & AUDIT_EVENT_CONTROLLER_LIMIT:
            self.mechanism_probability = max(self.mechanism_probability, 0.78)
            self.mechanism_channel_probability[:] = np.maximum(
                self.mechanism_channel_probability, np.asarray([0.48, 0.36, 0.82, 0.78]),
            )

        genome_channel = np.asarray([0.22, 0.18, 0.86, 0.72], dtype=float) * self.genome_probability
        self.channel_probability = 1.0 - (
            (1.0 - np.clip(self.mechanism_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(self.semantic_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(genome_channel, 0.0, 0.995))
        )
        self.semantic_probability = float(np.max(self.semantic_channel_probability))
        self.global_probability = self._combined_probability((
            self.mechanism_probability, self.semantic_probability, self.genome_probability,
        ))
        self.source_probability[:] = [
            self.state_probability, self.mechanism_probability,
            self.semantic_probability, self.genome_probability,
        ]
        dominant = int(np.argmax(self.source_probability))
        self.last_dominant_source = FORMAL_CHANGE_SOURCES[dominant]
        self.max_probability = max(self.max_probability, self.global_probability)
        if self.global_probability >= FORMAL_CHANGE_EVENT_THRESHOLD and age - self.last_event_age > 3.0:
            self.events += 1; self.last_event_age = float(age)
            if self.last_dominant_source == 'mechanism': self.mechanism_events += 1
        return self.global_probability

    def observe_semantic(self, ligand, outcome_vector, quality, age, config):
        """Observe a completed material-uptake outcome episode.

        No-contact intervals are never interpreted as zero effect.  A semantic
        change requires an established baseline and repeated sign/large-effect
        disagreement; ordinary noisy metabolic outcomes only widen the baseline
        variance and do not open the strong feedback gate.
        """
        ligand = int(ligand)
        if ligand < 0 or ligand >= FORMAL_LIGAND_COUNT:
            return np.zeros(FORMAL_CHANNELS, dtype=float)
        outcome = np.asarray(outcome_vector, dtype=float).reshape(FORMAL_CHANNELS)
        q = clamp(float(quality), 0.05, 1.0)
        z = np.zeros(FORMAL_CHANNELS, dtype=float)
        evidence = np.zeros(FORMAL_CHANNELS, dtype=float)
        minimum_episodes = max(3.0, float(config.change_semantic_min_episodes))
        minimum_effect = max(5e-4, float(config.change_semantic_min_effect))
        for k in range(FORMAL_CHANNELS):
            value = float(outcome[k])
            count = float(self.semantic_count[ligand, k])
            baseline = float(self.semantic_slow[ligand, k])
            if count <= 1e-12:
                self.semantic_fast[ligand, k] = value
                self.semantic_slow[ligand, k] = value
                self.semantic_var[ligand, k] = max(self._semantic_floor[k] ** 2, 0.25 * minimum_effect ** 2)
                self.semantic_count[ligand, k] = q
                continue
            if count < minimum_episodes:
                new_count = count + q
                alpha = q / max(new_count, 1e-9)
                residual = value - baseline
                self.semantic_slow[ligand, k] += alpha * residual
                self.semantic_fast[ligand, k] = self.semantic_slow[ligand, k]
                self.semantic_var[ligand, k] = max(
                    self._semantic_floor[k] ** 2,
                    (1.0 - alpha) * self.semantic_var[ligand, k] + alpha * residual * residual,
                )
                self.semantic_count[ligand, k] = new_count
                continue

            # Compare a recent episode average with a slowly adapting baseline.
            self.semantic_fast[ligand, k] += 0.42 * q * (value - self.semantic_fast[ligand, k])
            recent = float(self.semantic_fast[ligand, k])
            scale = math.sqrt(max(self.semantic_var[ligand, k], 0.0)) + self._semantic_floor[k]
            shift = recent - baseline
            z[k] = abs(shift) / max(scale, 1e-9)
            sign_flip = bool(
                abs(baseline) >= minimum_effect
                and abs(recent) >= minimum_effect
                and baseline * recent < 0.0
            )
            large_shift = bool(
                abs(shift) >= max(2.0 * minimum_effect, 0.70 * abs(baseline) + minimum_effect)
                and z[k] >= float(config.change_semantic_threshold) + 0.65
            )
            if sign_flip:
                evidence[k] = q * clamp(
                    0.55 + 0.45 * (z[k] - config.change_semantic_threshold) / 2.2,
                    0.48, 1.0,
                )
            elif large_shift:
                evidence[k] = 0.38 * q * clamp(
                    (z[k] - config.change_semantic_threshold) / 2.4, 0.0, 1.0,
                )

            # In the absence of repeatable change evidence, normal outcomes
            # update the baseline and its variability.  During a suspected
            # reversal the old reference is held long enough for confirmation.
            if evidence[k] < 0.12:
                residual = value - self.semantic_slow[ligand, k]
                self.semantic_slow[ligand, k] += 0.035 * q * residual
                cap = 25.0 * (self.semantic_var[ligand, k] + self._semantic_floor[k] ** 2)
                self.semantic_var[ligand, k] = 0.965 * self.semantic_var[ligand, k] + 0.035 * min(residual * residual, cap)
            self.semantic_count[ligand, k] += q

        self.last_semantic_z[ligand] = z
        if not self.enabled or not self.armed:
            return z
        for k in range(FORMAL_CHANNELS):
            self.semantic_accumulator[k] = max(
                0.0,
                0.78 * self.semantic_accumulator[k] + 0.92 * evidence[k] - 0.22,
            )
            if self.semantic_accumulator[k] < 0.20:
                continue
            target = clamp(
                1.0 - math.exp(-2.2 * self.semantic_accumulator[k]), 0.0, 0.995,
            )
            self.semantic_channel_probability[k] = max(
                self.semantic_channel_probability[k], target,
            )
        if float(np.max(evidence)) > 0.20:
            self.semantic_events += 1
        self.semantic_probability = float(np.max(self.semantic_channel_probability))
        self.global_probability = self._combined_probability((
            self.mechanism_probability, self.semantic_probability, self.genome_probability,
        ))
        self.channel_probability = 1.0 - (
            (1.0 - np.clip(self.mechanism_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(self.semantic_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(np.asarray([0.22, 0.18, 0.86, 0.72]) * self.genome_probability, 0.0, 0.995))
        )
        self.source_probability[:] = [
            self.state_probability, self.mechanism_probability,
            self.semantic_probability, self.genome_probability,
        ]
        self.last_dominant_source = FORMAL_CHANGE_SOURCES[int(np.argmax(self.source_probability))]
        self.max_probability = max(self.max_probability, self.global_probability)
        if self.global_probability >= FORMAL_CHANGE_EVENT_THRESHOLD and age - self.last_event_age > 3.0:
            self.events += 1
            self.last_event_age = float(age)
        return z

    def observe_audit_surprise(self, channel, calibrated_prediction, observed_effect,
                               quality, reliability, age, config):
        """Convert a reliable intervention sign reversal into targeted change evidence.

        A first, poorly calibrated prediction error is treated as model learning, not
        proof that the world changed.  Strong evidence requires an already reliable
        causal model, a high-quality stop experiment, and either a sign reversal or
        a large effect-size residual.
        """
        k = int(channel)
        if k < 0 or k >= FORMAL_CHANNELS or not self.enabled or not self.armed:
            return 0.0
        pred = float(calibrated_prediction)
        obs = float(observed_effect)
        q = clamp(float(quality), 0.0, 1.0)
        rel = clamp(float(reliability), 0.0, 1.0)
        rel_gate = clamp((rel - 0.38) / 0.52, 0.0, 1.0)
        ps = _sign(pred, 1.5e-4)
        osign = _sign(obs, 1.5e-4)
        sign_flip = float(ps != 0.0 and osign != 0.0 and ps != osign)
        residual = abs(obs - pred)
        residual_gate = clamp((residual - 4.5e-4) / 0.0032, 0.0, 1.0)
        strength = q * rel_gate * max(sign_flip, 0.34 * residual_gate)
        if strength <= 1e-8:
            return 0.0
        target = clamp(0.18 + 0.79 * strength, 0.0, 0.995)
        self.mechanism_channel_probability[k] = max(
            self.mechanism_channel_probability[k], target,
        )
        self.mechanism_probability = max(
            self.mechanism_probability, clamp(0.12 + 0.70 * strength, 0.0, 0.995),
        )
        genome_channel = np.asarray([0.22, 0.18, 0.86, 0.72], dtype=float) * self.genome_probability
        self.channel_probability = 1.0 - (
            (1.0 - np.clip(self.mechanism_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(self.semantic_channel_probability, 0.0, 0.995))
            * (1.0 - np.clip(genome_channel, 0.0, 0.995))
        )
        self.global_probability = self._combined_probability((
            self.mechanism_probability, self.semantic_probability, self.genome_probability,
        ))
        self.source_probability[:] = [
            self.state_probability, self.mechanism_probability,
            self.semantic_probability, self.genome_probability,
        ]
        self.last_dominant_source = 'mechanism'
        self.max_probability = max(self.max_probability, self.global_probability)
        if self.global_probability >= FORMAL_CHANGE_EVENT_THRESHOLD and age - self.last_event_age > 3.0:
            self.events += 1
            self.mechanism_events += 1
            self.last_event_age = float(age)
        return float(strength)

    def feedback_probability(self, channel):
        k = int(channel)
        if k < 0 or k >= FORMAL_CHANNELS:
            return 0.0
        return clamp(float(self.channel_probability[k]), 0.0, 1.0)

    def feedback_source(self, channel):
        k = int(channel)
        if k < 0 or k >= FORMAL_CHANNELS:
            return 'none'
        genome_channel = float(np.asarray([0.22, 0.18, 0.86, 0.72])[k] * self.genome_probability)
        values = np.asarray([
            self.mechanism_channel_probability[k],
            self.semantic_channel_probability[k],
            genome_channel,
        ], dtype=float)
        return ('mechanism', 'semantics', 'genome')[int(np.argmax(values))]

    def diagnostics(self):
        return {
            'global_probability': float(self.global_probability),
            'state_probability': float(self.state_probability),
            'mechanism_probability': float(self.mechanism_probability),
            'semantic_probability': float(self.semantic_probability),
            'genome_probability': float(self.genome_probability),
            'channel_probability': self.channel_probability.copy(),
            'source_probability': self.source_probability.copy(),
            'dominant_source': self.last_dominant_source,
            'events': int(self.events),
            'max_probability': float(self.max_probability),
            'armed': bool(self.armed),
        }

    def state_dict(self):
        state = dict(self.__dict__)
        for key, value in list(state.items()):
            if isinstance(value, np.ndarray):
                state[key] = value.copy()
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('enabled', True))
        # New schema.
        for key, value in state.items():
            if hasattr(obj, key):
                current = getattr(obj, key)
                if isinstance(current, np.ndarray):
                    value = np.asarray(value, dtype=current.dtype).copy()
                setattr(obj, key, value)
        # Development-save compatibility with the earlier single-vector sentinel.
        if 'fast' in state and 'state_fast' not in state:
            old_fast = np.asarray(state.get('fast', np.zeros(FORMAL_FEATURE_DIM)), dtype=float)
            old_slow = np.asarray(state.get('slow', old_fast), dtype=float)
            old_var = np.asarray(state.get('variance', np.ones(FORMAL_FEATURE_DIM) * 0.01), dtype=float)
            obj.state_fast[:] = old_fast[:FORMAL_STATE_FEATURE_DIM]
            obj.state_slow[:] = old_slow[:FORMAL_STATE_FEATURE_DIM]
            obj.state_var[:] = old_var[:FORMAL_STATE_FEATURE_DIM]
            obj.mech_fast[:] = old_fast[FORMAL_STATE_FEATURE_DIM:FORMAL_FEATURE_DIM]
            obj.mech_slow[:] = old_slow[FORMAL_STATE_FEATURE_DIM:FORMAL_FEATURE_DIM]
            obj.mech_var[:] = old_var[FORMAL_STATE_FEATURE_DIM:FORMAL_FEATURE_DIM]
            obj.global_probability = float(state.get('global_probability', 0.0))
            obj.channel_probability = np.asarray(state.get('channel_probability', np.zeros(FORMAL_CHANNELS)), dtype=float).copy()
        return obj

class MaterialCausalAuditor(object):
    STAGE_IDLE = 'idle'; STAGE_WASH = 'wash'; STAGE_MEASURE = 'measure'

    def __init__(self, enabled=True):
        self.enabled = bool(enabled); self.active = False; self.stage = self.STAGE_IDLE
        self.program = None; self.periods = []; self.period_index = 0; self.timer = 0.0
        self.period_start_debt = np.zeros(FORMAL_CHANNELS); self.period_start_context = np.zeros(FORMAL_CONTEXT_DIM)
        self.period_event = 0; self.period_samples = []; self.records = []; self.retries_for_period = 0
        self.claim_mask = np.zeros(p2.P2_CELL_COUNT, dtype=bool); self.sham_mask = np.zeros(p2.P2_CELL_COUNT, dtype=bool)
        self.period_sequence = []
        self.completed = self.supported = self.contradicted = self.inconclusive = self.aborted = 0
        self.retried = self.contaminated = 0
        self.last_quality = 0.0; self.last_effect = 0.0; self.last_status = 'none'; self.last_age = -1e9
        self.cooldown_until = 0.0
        self.evidence_weight = np.zeros((p2.P2_CELL_COUNT, FORMAL_CHANNELS), dtype=float)
        self.evidence_effect = np.zeros((p2.P2_CELL_COUNT, FORMAL_CHANNELS), dtype=float)
        self.evidence_age = np.full((p2.P2_CELL_COUNT, FORMAL_CHANNELS), -1e9, dtype=float)
        self.cumulative_cost_atp = 0.0; self.cumulative_cost_material = 0.0
        self.cost_start_atp = 0.0; self.cost_start_material = 0.0

    def start(self, program, debt, context, age, rng, config, cost_atp=0.0, cost_material=0.0):
        if not self.enabled or self.active or age < self.cooldown_until:
            return False
        self.program = dict(program)
        self.claim_mask = np.asarray(program['claim_mask'], dtype=bool).copy()
        self.sham_mask = np.asarray(program['sham_mask'], dtype=bool).copy()
        arms = [0, 1]
        if config.audit_symmetric_order and rng.random() < 0.5:
            arms.reverse()
        self.periods = []; self.period_sequence = []
        for arm in arms:
            seq = np.asarray([0,1,1,0] if (not config.audit_symmetric_order or rng.random() < 0.5) else [1,0,0,1], dtype=np.int64)
            self.period_sequence.append(seq.copy())
            for condition in seq:
                self.periods.append((arm, int(condition)))
        self.active = True; self.stage = self.STAGE_WASH; self.period_index = 0; self.timer = 0.0
        self.records = []; self.retries_for_period = 0; self.period_event = 0; self.period_samples = []
        self.period_start_debt = np.asarray(debt, dtype=float).copy(); self.period_start_context = np.asarray(context, dtype=float).copy()
        self.cost_start_atp = float(cost_atp); self.cost_start_material = float(cost_material)
        return True

    def current_arm(self):
        return int(self.periods[self.period_index][0]) if self.active and self.period_index < len(self.periods) else 0

    def current_condition(self):
        return int(self.periods[self.period_index][1]) if self.active and self.period_index < len(self.periods) else 0

    def current_gate(self, config):
        gate = np.ones(p2.P2_CELL_COUNT, dtype=float)
        if self.active and self.stage == self.STAGE_MEASURE and self.current_condition() == 1:
            mask = self.claim_mask if self.current_arm() == 0 else self.sham_mask
            gate[mask] = 1.0 - clamp(config.audit_knockout_strength, 0.0, 0.98)
        return gate

    def _advance(self, debt, context, config):
        self.period_index += 1; self.timer = 0.0; self.period_event = 0; self.period_samples = []
        self.retries_for_period = 0
        if self.period_index >= len(self.periods):
            return False
        self.stage = self.STAGE_WASH
        self.period_start_debt = np.asarray(debt, dtype=float).copy(); self.period_start_context = np.asarray(context, dtype=float).copy()
        return True

    def _finish(self, age, config, cost_atp, cost_material):
        channel = int(self.program['channel'])
        effects = []
        for arm in (0,1):
            ko = [r['slope'][channel] for r in self.records if r['arm'] == arm and r['condition'] == 1]
            intact = [r['slope'][channel] for r in self.records if r['arm'] == arm and r['condition'] == 0]
            effects.append(float(np.median(ko) - np.median(intact)))
        target = effects[0] - effects[1]
        slopes = np.asarray([r['slope'][channel] for r in self.records], dtype=float)
        contexts = np.asarray([r['context_delta'] for r in self.records], dtype=float)
        contamination = float(sum(r['contaminated'] for r in self.records)) / max(1, len(self.records))
        noise = float(np.std(slopes))
        mismatch = float(np.mean(contexts)) if contexts.size else 0.0
        quality = math.exp(-noise / 0.0035) * math.exp(-mismatch / 1.2) * (1.0 - 0.75 * contamination)
        quality = clamp(quality, 0.03, 1.0)
        predicted = float(self.program.get('calibrated_prediction', self.program.get('raw_prediction', 0.0)))
        threshold = 0.00012 + 0.00020 * (1.0 - quality)
        if abs(target) < threshold:
            status = 'inconclusive'; self.inconclusive += 1
        elif _sign(target) == _sign(predicted) or abs(predicted) < threshold:
            status = 'supported'; self.supported += 1
        else:
            status = 'contradicted'; self.contradicted += 1
        self.completed += 1; self.last_quality = quality; self.last_effect = target; self.last_status = status; self.last_age = float(age)
        self.cumulative_cost_atp += max(0.0, float(cost_atp) - self.cost_start_atp)
        self.cumulative_cost_material += max(0.0, float(cost_material) - self.cost_start_material)
        qsign = 1.0 if status == 'supported' else (-1.0 if status == 'contradicted' else 0.0)
        for i in np.flatnonzero(self.claim_mask):
            old = self.evidence_weight[i, channel]
            add = quality * qsign
            self.evidence_weight[i, channel] = clamp(old * 0.82 + add, -3.0, 3.0)
            self.evidence_effect[i, channel] = 0.75 * self.evidence_effect[i, channel] + 0.25 * target
            self.evidence_age[i, channel] = float(age)
        self.active = False; self.stage = self.STAGE_IDLE; self.cooldown_until = float(age) + config.audit_cooldown
        return {'status': status, 'quality': quality, 'target_effect': target, 'claim_effect': effects[0], 'sham_effect': effects[1], 'channel': channel, 'claim_mask': self.claim_mask.copy(), 'sham_mask': self.sham_mask.copy(), 'raw_prediction': float(self.program.get('raw_prediction',0.0)), 'calibrated_prediction': predicted, 'reliability': float(self.program.get('reliability',0.0)), 'program': dict(self.program)}

    def observe(self, debt, context, event_flags, capacity, dt, age, config, cost_atp, cost_material):
        if not self.active:
            return None
        debt = np.asarray(debt, dtype=float); context = np.asarray(context, dtype=float)
        # Structural fail-safe, not ordinary metabolic drift.
        if debt[1] > config.audit_safety_structural_absolute or debt[2] > config.audit_safety_structural_absolute:
            self.active = False; self.stage = self.STAGE_IDLE; self.aborted += 1; self.last_status = 'aborted'; self.cooldown_until = age + config.audit_cooldown
            return {'status':'aborted','quality':0.0,'reason':'structural-safety'}
        self.timer += dt
        self.period_event |= int(event_flags)
        if self.stage == self.STAGE_WASH:
            if self.timer >= config.audit_washout_duration:
                self.stage = self.STAGE_MEASURE; self.timer = 0.0
                self.period_start_debt = debt.copy(); self.period_start_context = context.copy(); self.period_event = int(event_flags); self.period_samples = []
            return None
        self.period_samples.append(debt.copy())
        if self.timer < config.audit_measure_duration:
            return None
        duration = max(self.timer, 1e-9)
        slope = (debt - self.period_start_debt) / duration
        contaminated = int(bool(self.period_event & AUDIT_HARD_EVENT_MASK))
        if contaminated and self.retries_for_period < config.audit_max_retries:
            self.retries_for_period += 1; self.retried += 1; self.contaminated += 1
            self.stage = self.STAGE_WASH; self.timer = 0.0; self.period_event = 0; self.period_samples = []
            return None
        self.records.append({'arm':self.current_arm(),'condition':self.current_condition(),'slope':slope.copy(),'context_delta':float(np.linalg.norm(context-self.period_start_context)),'contaminated':contaminated,'capacity':float(capacity)})
        if contaminated: self.contaminated += 1
        if not self._advance(debt, context, config):
            return self._finish(age, config, cost_atp, cost_material)
        return None

    def evidence_gate(self, age, config):
        elapsed = np.maximum(0.0, float(age) - self.evidence_age)
        decay = np.exp(-math.log(2.0) * elapsed / max(config.audit_evidence_half_life, 1e-9))
        signed = self.evidence_weight * decay
        return np.clip(0.5 + 0.45 * np.tanh(signed), 0.01, 0.99)

    def state_dict(self):
        state = dict(self.__dict__)
        for key, value in list(state.items()):
            if isinstance(value, np.ndarray): state[key] = value.copy()
            elif key == 'program' and value is not None:
                state[key] = dict(value)
                for k in ('claim_mask','sham_mask'):
                    if k in state[key]: state[key][k] = np.asarray(state[key][k]).copy()
            elif key == 'periods': state[key] = [tuple(x) for x in value]
            elif key == 'period_sequence': state[key] = [np.asarray(x).copy() for x in value]
            elif key == 'records':
                state[key] = [{kk:(vv.copy() if isinstance(vv,np.ndarray) else vv) for kk,vv in r.items()} for r in value]
            elif key == 'period_samples': state[key] = [np.asarray(x).copy() for x in value]
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('enabled', True))
        for key, value in state.items():
            if isinstance(getattr(obj,key,None), np.ndarray): value = np.asarray(value).copy()
            setattr(obj,key,value)
        obj.periods = [tuple(x) for x in state.get('periods',[])]
        obj.period_sequence = [np.asarray(x).copy() for x in state.get('period_sequence',[])]
        return obj


class SmallFalsificationCompiler(object):
    def __init__(self, enabled=True):
        self.enabled = bool(enabled); self.compiled = 0; self.started = 0; self.last_program = None

    def _matched_sham(self, tissue, claim_mask, group_size):
        candidates = np.flatnonzero(~claim_mask)
        claim = np.flatnonzero(claim_mask)
        target_activity = float(np.mean(tissue.activity_mean[claim]))
        target_motor = float(np.mean(np.abs(tissue.hidden[claim] * tissue.motor_gain[claim])))
        scores = []
        for i in candidates:
            score = abs(tissue.activity_mean[i]-target_activity) + abs(abs(tissue.hidden[i]*tissue.motor_gain[i])-target_motor)
            scores.append((score,int(i)))
        chosen = [i for _,i in sorted(scores)[:group_size]]
        mask = np.zeros(p2.P2_CELL_COUNT,dtype=bool); mask[chosen]=True
        return mask

    def compile(self, tissue, debt, config):
        if not self.enabled: return None
        trust = tissue.calibrator.cell_trust
        rights = tissue.causal_right_gate
        scores_by_channel = []
        for k in range(FORMAL_CHANNELS):
            score = float(np.max(np.abs(tissue.causal_estimate[:,k]) * trust[:,k] * rights[:,k])) * (0.25 + float(debt[k]))
            scores_by_channel.append(score)
        channel = int(np.argmax(scores_by_channel))
        cell_score = np.abs(tissue.causal_estimate[:,channel]) * trust[:,channel] * rights[:,channel] + 0.05*np.abs(tissue.hidden*tissue.motor_gain)
        order = np.argsort(-cell_score)
        claim = np.zeros(p2.P2_CELL_COUNT,dtype=bool); claim[order[:config.audit_group_size]]=True
        sham = self._matched_sham(tissue,claim,config.audit_group_size)
        raw = float(np.mean(tissue.causal_estimate[claim,channel]) - np.mean(tissue.causal_estimate[sham,channel]))
        cal, reliability = tissue.calibrator.predict(channel,raw)
        commands=[{'op':'gate_cells','count':int(np.sum(claim)),'strength':config.audit_knockout_strength}]
        program={'family':'causal-cell-necessity','claim_mask':claim,'sham_mask':sham,'channel':channel,'raw_prediction':raw,'calibrated_prediction':cal,'reliability':reliability,'commands':commands,'complexity':len(commands)}
        self.compiled += 1; self.last_program = program
        return program

    def random_matched_program(self, tissue, debt, config):
        channel=int(tissue.formal_rng.integers(0,FORMAL_CHANNELS))
        order=tissue.formal_rng.permutation(p2.P2_CELL_COUNT)
        claim=np.zeros(p2.P2_CELL_COUNT,dtype=bool); sham=np.zeros_like(claim)
        claim[order[:config.audit_group_size]]=True; sham[order[config.audit_group_size:2*config.audit_group_size]]=True
        raw=float(np.mean(tissue.causal_estimate[claim,channel])-np.mean(tissue.causal_estimate[sham,channel]))
        cal,rel=tissue.calibrator.predict(channel,raw)
        program={'family':'random-cost-matched','claim_mask':claim,'sham_mask':sham,'channel':channel,'raw_prediction':raw,'calibrated_prediction':cal,'reliability':rel,'commands':[{'op':'gate_cells','count':int(np.sum(claim)),'strength':config.audit_knockout_strength}],'complexity':1}
        self.compiled += 1; self.last_program=program
        return program

    def state_dict(self):
        state={'enabled':self.enabled,'compiled':self.compiled,'started':self.started,'last_program':None}
        if self.last_program is not None:
            state['last_program']=dict(self.last_program)
            for k in ('claim_mask','sham_mask'): state['last_program'][k]=np.asarray(state['last_program'][k]).copy()
        return state

    @classmethod
    def from_state(cls,state):
        obj=cls(state.get('enabled',True)); obj.compiled=int(state.get('compiled',0)); obj.started=int(state.get('started',0)); obj.last_program=state.get('last_program')
        return obj


class FormalMaterialTissue(p2.EightCellMaterialTissue):
    def __init__(self, gene_parameters, mode=p2.P2_MODE_FULL, rng_seed=0, formal_enabled=False):
        super(FormalMaterialTissue,self).__init__(gene_parameters,mode=mode,rng_seed=rng_seed)
        self._init_formal_runtime(rng_seed=rng_seed, formal_enabled=formal_enabled)

    def _init_formal_runtime(self, rng_seed=0, formal_enabled=False):
        """Initialise formal-0.6 state without touching the frozen P2 substrate."""
        self.formal_rng = np.random.default_rng((int(rng_seed) ^ 0xF06A11) & 0xFFFFFFFF)
        self.formal_enabled = bool(formal_enabled)
        self.controller_mature = False
        self.controller_development = 0.0
        self.controller_failed = False
        self.controller_atp_spent = 0.0
        self.controller_material_wear = 0.0
        self.auditor = MaterialCausalAuditor(True)
        self.calibrator = CausalCalibrationLedger()
        self.change_sentinel = MaterialChangeSentinel(True)
        self.compiler = SmallFalsificationCompiler(True)
        self.intervention_gate = np.ones(p2.P2_CELL_COUNT, dtype=float)
        self.resource_lease = np.ones(p2.P2_CELL_COUNT, dtype=float)
        self.causal_right_gate = np.ones((p2.P2_CELL_COUNT, FORMAL_CHANNELS), dtype=float) * 0.5
        self.last_formal_context = np.zeros(FORMAL_CONTEXT_DIM, dtype=float)
        self.last_change_features = np.zeros(FORMAL_FEATURE_DIM, dtype=float)
        self.last_change_state = np.zeros(FORMAL_STATE_FEATURE_DIM, dtype=float)
        self.last_change_mechanism = np.zeros(FORMAL_MECHANISM_FEATURE_DIM, dtype=float)
        self.semantic_episode_active = np.zeros(FORMAL_LIGAND_COUNT, dtype=bool)
        self.semantic_episode_timer = np.zeros(FORMAL_LIGAND_COUNT, dtype=float)
        self.semantic_episode_start_debt = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.semantic_episode_peak_debt = np.zeros((FORMAL_LIGAND_COUNT, FORMAL_CHANNELS), dtype=float)
        self.semantic_episode_uptake = np.zeros(FORMAL_LIGAND_COUNT, dtype=float)
        self.semantic_episode_contaminated = np.zeros(FORMAL_LIGAND_COUNT, dtype=bool)
        self.semantic_episode_updates = 0
        self.last_event_flags = 0
        self.last_reactive = 0.0
        self.last_uptake = 0.0
        self.last_divisions = 0
        self.last_hgt_count = 0
        self.last_audit_result = None
        self.last_feedback_gate = 0.0
        self.last_feedback_source = 'none'
        self.feedback_events = 0
        self.reopened_cells = 0
        self.lease_reductions = 0
        self.lease_recoveries = 0
        self.formal_hgt_activation_age = -1.0
        self.formal_last_start_age = -1e9
        self.formal_no_program_until = 0.0
        self._learning_snapshot = None
        self._gate_backup = None

    def ensure_controller_attachment(self,port):
        try: port.attachment_status(FORMAL_CONTROLLER_ID)
        except KeyError: port.attach(FORMAL_CONTROLLER_ID,kind=FORMAL_CONTROLLER_KIND)

    def _controller_status(self,port):
        self.ensure_controller_attachment(port); return port.attachment_status(FORMAL_CONTROLLER_ID)

    def _develop_controller(self,port,dt):
        status=self._controller_status(port); tissue=np.asarray(status['tissue_material']); stores=np.asarray(status['stores'])
        pn=max(0.0,FORMAL_TARGET_PROTEIN-float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])); mn=max(0.0,FORMAL_TARGET_MEMBRANE-float(tissue[p0.TISSUE_MEMBRANE])); sn=max(0.0,FORMAL_TARGET_SIGNAL-float(tissue[p0.TISSUE_SIGNAL]+stores[p0.BUDGET_SIGNAL]))
        if pn+mn+sn>1e-12:
            port.allocate_budget(FORMAL_CONTROLLER_ID,{'atp':0.0025,'protein':min(pn,0.0015),'membrane':min(mn,0.0007),'signal':min(sn,0.0008)},dt)
            status=self._controller_status(port); stores=np.asarray(status['stores'])
            port.commit_material(FORMAL_CONTROLLER_ID,protein=min(pn,float(stores[p0.BUDGET_PROTEIN])),membrane=min(mn,float(stores[p0.BUDGET_MEMBRANE])),signal=min(sn,float(stores[p0.BUDGET_SIGNAL])))
            status=self._controller_status(port); tissue=np.asarray(status['tissue_material']); stores=np.asarray(status['stores'])
        ratios=[float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])/FORMAL_TARGET_PROTEIN,float(tissue[p0.TISSUE_MEMBRANE])/FORMAL_TARGET_MEMBRANE,float(tissue[p0.TISSUE_SIGNAL]+stores[p0.BUDGET_SIGNAL])/FORMAL_TARGET_SIGNAL]
        self.controller_development=clamp(min(ratios),0.0,1.0)
        self.controller_mature=bool(float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])>=FORMAL_MATURE_PROTEIN and float(tissue[p0.TISSUE_MEMBRANE])>=FORMAL_MATURE_MEMBRANE and float(tissue[p0.TISSUE_SIGNAL]+stores[p0.BUDGET_SIGNAL])>=FORMAL_MATURE_SIGNAL)

    def _maintain_controller(self,port,dt):
        status=self._controller_status(port); tissue=np.asarray(status['tissue_material']); stores=np.asarray(status['stores'])
        if float(tissue[p0.TISSUE_DAMAGED_PROTEIN])>=FORMAL_DAMAGE_LIMIT or float(tissue[p0.TISSUE_AGGREGATE])>=FORMAL_AGGREGATE_LIMIT:
            port.return_dead_tissue(FORMAL_CONTROLLER_ID,reason='formal-controller-turnover'); self.controller_mature=False; self.controller_failed=True; return False
        request={'atp':max(0.0,0.004-float(stores[p0.BUDGET_ATP])),'protein':0.000025*dt,'signal':max(0.0,0.0008-float(stores[p0.BUDGET_SIGNAL])),'membrane':0.0}
        if any(v>1e-14 for v in request.values()): port.allocate_budget(FORMAL_CONTROLLER_ID,request,dt)
        return True

    def _pay_controller_cost(self,port,dt,active=False,group_size=0):
        intensity=(1.0 if active else 0.22)*(1.0+0.18*group_size)
        wear=dt*0.000055*intensity
        if wear<=0.0:return
        port.allocate_budget(FORMAL_CONTROLLER_ID,{'atp':dt*0.0008*intensity,'protein':wear,'membrane':0.0,'signal':0.0},dt)
        status=self._controller_status(port); stores=np.asarray(status['stores'])
        built=port.commit_material(FORMAL_CONTROLLER_ID,protein=min(wear,float(stores[p0.BUDGET_PROTEIN])),damaged_fraction=0.75,aggregate_fraction=0.15)
        self.controller_atp_spent+=float(built.get('atp_spent',0.0)); self.controller_material_wear+=float(built.get('damaged_protein',0.0)+built.get('aggregate',0.0))

    def _context(self,frame,debt):
        internal=frame['internal']; velocity=np.asarray(frame['velocity'],dtype=float)
        return np.asarray([debt[0],debt[1],debt[2],debt[3],float(np.linalg.norm(velocity)),float(internal['atp']),float(internal['reactive']),float(internal['closure_mean']),float(internal['retention']),float(np.mean(self.prediction_error)),float(np.mean(self.maturity)),float(np.mean(np.abs(self.hidden)))],dtype=float)

    def _change_features(self,frame,debt,cell,port):
        internal=frame['internal']; uptake=np.asarray(frame['flux']['last_uptake_by_ligand'],dtype=float)
        atp=max(0.0,float(internal['atp'])); reactive=max(0.0,float(internal['reactive']))
        state=np.asarray(list(debt)+[
            float(np.mean(self.prediction_error)),
            clamp(atp/(0.075+atp),0.0,1.0),
            clamp(reactive/0.10,0.0,1.0),
            math.tanh(20.0*float(np.sum(np.maximum(0.0,uptake[:FORMAL_LIGAND_COUNT])))),
        ],dtype=float)
        command=np.asarray(self.last_action,dtype=float)
        velocity=np.asarray(frame['velocity'],dtype=float)
        command_norm=float(np.linalg.norm(command))
        motor_force=max(0.0,float(self.last_effector_report.get('motor_force',0.0)))
        transduction=motor_force/max(command_norm,0.04) if command_norm>0.02 else 0.0
        velocity_norm=float(np.linalg.norm(velocity))
        alignment=float(np.dot(velocity,command)/(velocity_norm*command_norm+1e-8)) if command_norm>0.02 and velocity_norm>1e-7 else 0.0
        capacity=float(np.mean(self._material_capacity()))
        total=float(np.sum(self.material_status))+1e-12
        neural_damage=float((np.sum(self.material_status[:,p0.TISSUE_DAMAGED_PROTEIN])+np.sum(self.material_status[:,p0.TISSUE_AGGREGATE]))/total)
        lesion=clamp(float(internal.get('genome_lesion',0.0))/0.45,0.0,1.0)
        try:
            status=self._controller_status(port); tissue=np.asarray(status['tissue_material']); stores=np.asarray(status['stores'])
            controller_capacity=clamp(min(
                float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN])/max(FORMAL_TARGET_PROTEIN,1e-12),
                float(tissue[p0.TISSUE_MEMBRANE])/max(FORMAL_TARGET_MEMBRANE,1e-12),
                float(tissue[p0.TISSUE_SIGNAL]+stores[p0.BUDGET_SIGNAL])/max(FORMAL_TARGET_SIGNAL,1e-12),
            ),0.0,1.0)
        except Exception:
            controller_capacity=0.0
        mechanism=np.asarray([
            clamp(transduction/0.010,0.0,2.0),
            clamp(alignment,-1.0,1.0),
            capacity,
            clamp(neural_damage/0.18,0.0,2.0),
            clamp(float(internal['closure_mean']),0.0,1.0),
            clamp(float(internal['retention']),0.0,1.0),
            lesion,
            controller_capacity,
        ],dtype=float)
        state_mask=np.ones(FORMAL_STATE_FEATURE_DIM,dtype=bool)
        mechanism_mask=np.ones(FORMAL_MECHANISM_FEATURE_DIM,dtype=bool)
        if command_norm<=0.02: mechanism_mask[:2]=False
        return state,mechanism,state_mask,mechanism_mask

    def _update_semantic_episodes(self,frame,debt,dt,world,config,contaminated=False):
        uptake=np.maximum(0.0,np.asarray(frame['flux']['last_uptake_by_ligand'],dtype=float)[:FORMAL_LIGAND_COUNT])
        for ligand in range(FORMAL_LIGAND_COUNT):
            signal=math.tanh(28.0*float(uptake[ligand]))
            if (not self.semantic_episode_active[ligand]) and signal>0.018:
                self.semantic_episode_active[ligand]=True
                self.semantic_episode_timer[ligand]=max(0.5,config.change_semantic_window)
                self.semantic_episode_start_debt[ligand]=debt
                self.semantic_episode_peak_debt[ligand]=debt
                self.semantic_episode_uptake[ligand]=0.0
                self.semantic_episode_contaminated[ligand]=bool(contaminated)
            if not self.semantic_episode_active[ligand]:
                continue
            self.semantic_episode_timer[ligand]-=dt
            self.semantic_episode_peak_debt[ligand]=np.maximum(self.semantic_episode_peak_debt[ligand],debt)
            self.semantic_episode_uptake[ligand]+=float(uptake[ligand])
            self.semantic_episode_contaminated[ligand]|=bool(contaminated)
            if self.semantic_episode_timer[ligand]>0.0:
                continue
            start=self.semantic_episode_start_debt[ligand].copy()
            harm=np.maximum(self.semantic_episode_peak_debt[ligand]-start,0.0)
            outcome=start-debt-config.change_semantic_harm_memory*harm
            amount=self.semantic_episode_uptake[ligand]
            quality=clamp(amount/0.0045,0.08,1.0)
            if not self.semantic_episode_contaminated[ligand] and amount>1e-5:
                self.change_sentinel.observe_semantic(ligand,outcome,quality,world.age,config)
                self.semantic_episode_updates+=1
            self.semantic_episode_active[ligand]=False
            self.semantic_episode_timer[ligand]=0.0
            self.semantic_episode_uptake[ligand]=0.0
            self.semantic_episode_contaminated[ligand]=False

    def _event_flags(self,frame,cell,world,config):
        uptake=float(np.sum(np.asarray(frame['flux']['last_uptake_by_ligand'],dtype=float))); reactive=float(frame['internal']['reactive']); flags=0
        if uptake-self.last_uptake>config.audit_hard_uptake_threshold: flags|=AUDIT_EVENT_RESOURCE
        if reactive-self.last_reactive>config.audit_hard_reactive_threshold: flags|=AUDIT_EVENT_TOXIN
        if float(frame['external']['corpse_signal'])>0.10: flags|=AUDIT_EVENT_CORPSE
        if float(frame['external']['edna_signal'])>0.10: flags|=AUDIT_EVENT_EDNA
        if int(getattr(cell,'formal_hgt_activation_count',0))>self.last_hgt_count: flags|=AUDIT_EVENT_HGT
        if int(world.divisions)>self.last_divisions: flags|=AUDIT_EVENT_DIVISION
        if self.controller_failed or float(np.mean(self._material_capacity()))<0.42: flags|=AUDIT_EVENT_CONTROLLER_LIMIT
        self.last_uptake=uptake; self.last_reactive=reactive; self.last_hgt_count=int(getattr(cell,'formal_hgt_activation_count',0)); self.last_divisions=int(world.divisions)
        return flags

    def _snapshot_learning(self):
        names=('w_sensor','w_rec','bias','motor_gain','predict_w','causal_estimate','maturity','reopen_reserve','elig_bias','elig_sensor','elig_rec','elig_motor','cue_eligibility','ligand_trace')
        self._learning_snapshot={n:getattr(self,n).copy() for n in names}

    def _restore_learning(self):
        if self._learning_snapshot is None:return
        for n,v in self._learning_snapshot.items(): setattr(self,n,v.copy())
        self._learning_snapshot=None

    def formal_prepare(self,port,dt,config,cell,world):
        # A vertically inherited/seeded cassette may always express.  A gene
        # acquired through environmental DNA only gains phenotype when the
        # material HGT pathway is enabled; no-HGT is therefore a real
        # phenotype ablation, not merely a counter that stops incrementing.
        gene_authorised = bool(getattr(cell, 'formal_gene_installed', False) or config.neural_hgt_enabled)
        gene_active=bool(gene_authorised and formal_controller_activity(cell)>=0.016)
        self.formal_enabled=bool(gene_active and (config.audit_enabled or config.calibration_enabled or config.change_detection_enabled))
        self.auditor.enabled=bool(self.formal_enabled and config.audit_enabled); self.compiler.enabled=bool(self.formal_enabled and config.falsification_enabled); self.change_sentinel.enabled=bool(self.formal_enabled and config.change_detection_enabled)
        if not self.formal_enabled:
            self.intervention_gate[:]=1.0; return
        self.ensure_controller_attachment(port); self._develop_controller(port,dt); self._maintain_controller(port,dt)
        if not self.controller_mature:
            self.intervention_gate[:]=1.0; return
        recovery=1.0-math.exp(-dt/max(config.feedback_lease_recovery_tau,1e-6))
        before_lease=self.resource_lease.copy(); self.resource_lease+=recovery*(1.0-self.resource_lease)
        np.clip(self.resource_lease,config.feedback_lease_min,1.0,out=self.resource_lease)
        self.lease_recoveries+=int(np.count_nonzero(self.resource_lease-before_lease>1e-10))
        frame=port.raw_sensor_fluxes(self.tissue_ids[0]); debt=self._physical_debt(frame); context=self._context(frame,debt)
        evidence=self.auditor.evidence_gate(world.age,config)
        self.causal_right_gate=np.clip((0.35+0.65*evidence)*self.calibrator.cell_trust*self.resource_lease[:,None],0.01,1.0)
        if config.formal_auto_start and self.auditor.enabled and not self.auditor.active and self.active_age>=config.audit_min_active_age and world.age>=self.auditor.cooldown_until and world.age>=self.formal_no_program_until:
            structurally_safe=debt[1]<config.audit_safe_debt and debt[2]<config.audit_safe_debt and not (debt[0]>0.97 and debt[3]>0.97)
            if structurally_safe:
                program=self.compiler.compile(self,debt,config) if config.falsification_enabled else self.compiler.random_matched_program(self,debt,config)
                if program is not None and self.auditor.start(program,debt,context,world.age,self.formal_rng,config,self.controller_atp_spent,self.controller_material_wear):
                    self.compiler.started+=1; self.formal_last_start_age=world.age
                else:self.formal_no_program_until=world.age+1.5
        self.intervention_gate=self.auditor.current_gate(config)
        if self.auditor.active:
            self._pay_controller_cost(port,dt,active=True,group_size=int(np.sum(self.intervention_gate<1.0)))
            if config.audit_freeze_learning:self._snapshot_learning()
        else:self._pay_controller_cost(port,dt,active=False)

    def pre_step(self,port,dt,config,gene_activity):
        if not self.formal_enabled or not self.controller_mature:
            return super(FormalMaterialTissue,self).pre_step(port,dt,config,gene_activity)
        effective=np.clip(self.intervention_gate*self.resource_lease,0.0,1.0)
        backup_motor=self.motor_gain.copy(); backup_rec=self.w_rec.copy()
        self.motor_gain*=effective
        self.w_rec*=effective[None,:]
        try:return super(FormalMaterialTissue,self).pre_step(port,dt,config,gene_activity)
        finally:
            self.motor_gain=backup_motor; self.w_rec=backup_rec

    def _apply_feedback(self,result,config):
        status=result.get('status')
        mask=np.asarray(result.get('claim_mask',np.zeros(p2.P2_CELL_COUNT)),dtype=bool)
        channel=int(result.get('channel',0))
        if status=='supported':
            restore=0.08*clamp(float(result.get('quality',0.0))*(0.5+0.5*float(result.get('reliability',0.0))),0.0,1.0)
            before=self.resource_lease.copy(); self.resource_lease[mask]+=restore*(1.0-self.resource_lease[mask])
            np.clip(self.resource_lease,config.feedback_lease_min,1.0,out=self.resource_lease)
            self.lease_recoveries+=int(np.count_nonzero(self.resource_lease-before>1e-10))
            return 0.0
        if not config.feedback_enabled or status!='contradicted': return 0.0
        source_probability=self.change_sentinel.feedback_probability(channel)
        if config.change_gated_feedback:
            # A contradicted causal claim is not, by itself, evidence that the
            # body/world mechanism changed.  Calibration may still learn from
            # the audit, but material leases and maturity are protected until
            # source-resolved physical change evidence crosses the threshold.
            if source_probability < config.change_feedback_threshold:
                self.last_feedback_gate = 0.0
                self.last_feedback_source = 'none'
                return 0.0
            scaled=clamp((source_probability-config.change_feedback_threshold)/max(1e-9,1.0-config.change_feedback_threshold),0.0,1.0)
            gate=clamp(config.feedback_floor+(1.0-config.feedback_floor)*scaled,config.feedback_floor,1.0)
        else:
            gate=1.0
        effect_strength=clamp(abs(float(result.get('target_effect',0.0)))/0.0024,0.22,1.0)
        magnitude=clamp(float(result.get('quality',0.0))*gate*(0.45+0.55*float(result.get('reliability',0.0)))*effect_strength,0.0,1.0)
        before=self.maturity.copy(); before_lease=self.resource_lease.copy()
        self.maturity[mask]*=(1.0-0.30*magnitude)
        self.reopen_reserve[mask]=np.maximum(self.reopen_reserve[mask],0.62*magnitude)
        self.prediction_error[mask]+=0.20*magnitude
        self.causal_estimate[mask,channel]*=(1.0-config.feedback_weight_decay*magnitude)
        self.resource_lease[mask]*=(1.0-config.feedback_lease_loss*magnitude)
        np.clip(self.resource_lease,config.feedback_lease_min,1.0,out=self.resource_lease)
        np.clip(self.maturity,0.0,1.0,out=self.maturity); np.clip(self.prediction_error,0.0,2.0,out=self.prediction_error)
        reopened=int(np.count_nonzero(before[mask]-self.maturity[mask]>1e-6))
        reduced=int(np.count_nonzero(before_lease[mask]-self.resource_lease[mask]>1e-8))
        if magnitude>1e-8:self.feedback_events+=1
        self.reopened_cells+=reopened; self.lease_reductions+=reduced; self.last_feedback_gate=gate; self.last_feedback_source=self.change_sentinel.feedback_source(channel)
        return magnitude

    def formal_observe(self,port,dt,config,cell,world):
        if not self.formal_enabled or not self.controller_mature:
            return None
        frame = port.raw_sensor_fluxes(self.tissue_ids[0])
        debt = self._physical_debt(frame)
        context = self._context(frame,debt)
        state_feature, mechanism_feature, state_mask, mechanism_mask = self._change_features(
            frame, debt, cell, port,
        )
        events = self._event_flags(frame,cell,world,config)
        self.last_formal_context = context.copy()
        self.last_change_state = state_feature.copy()
        self.last_change_mechanism = mechanism_feature.copy()
        self.last_change_features = np.concatenate((state_feature, mechanism_feature))
        self.last_event_flags = events
        self.change_sentinel.update(
            state_feature, mechanism_feature, dt, world.age, config,
            self_intervention=self.auditor.active,
            event_flags=events,
            state_mask=state_mask,
            mechanism_mask=mechanism_mask,
        )
        semantic_contamination = bool(
            self.auditor.active
            or (events & (AUDIT_EVENT_CORPSE | AUDIT_EVENT_EDNA | AUDIT_EVENT_HGT | AUDIT_EVENT_DIVISION))
        )
        self._update_semantic_episodes(
            frame, debt, dt, world, config, contaminated=semantic_contamination,
        )
        result = self.auditor.observe(
            debt, context, events, float(np.mean(self._material_capacity())),
            dt, world.age, config, self.controller_atp_spent,
            self.controller_material_wear,
        )
        if config.audit_freeze_learning and self._learning_snapshot is not None:
            self._restore_learning()
        if result is not None and result.get('status') not in ('aborted',):
            if config.calibration_enabled:
                cal = self.calibrator.update(
                    result['channel'], result['raw_prediction'], result['target_effect'],
                    result['quality'], result['claim_mask'],
                )
                result.update(cal)
            else:
                result.update({
                    'calibrated_prediction': float(result.get('raw_prediction',0.0)),
                    'reliability': 0.0,
                })
            result['change_surprise'] = self.change_sentinel.observe_audit_surprise(
                result['channel'],
                result.get('calibrated_prediction', result.get('raw_prediction',0.0)),
                result.get('target_effect',0.0), result.get('quality',0.0),
                result.get('reliability',0.0), world.age, config,
            )
            result['feedback_magnitude'] = self._apply_feedback(result,config)
            self.last_audit_result = result
        return result

    def state_dict(self):
        state = super(FormalMaterialTissue,self).state_dict()
        state.update({
            'formal_schema': FORMAL_SCHEMA_VERSION,
            'formal_rng_state': self.formal_rng.bit_generator.state,
            'formal_enabled': self.formal_enabled,
            'controller_mature': self.controller_mature,
            'controller_development': self.controller_development,
            'controller_failed': self.controller_failed,
            'controller_atp_spent': self.controller_atp_spent,
            'controller_material_wear': self.controller_material_wear,
            'auditor': self.auditor.state_dict(),
            'calibrator': self.calibrator.state_dict(),
            'change_sentinel': self.change_sentinel.state_dict(),
            'compiler': self.compiler.state_dict(),
            'intervention_gate': self.intervention_gate.copy(),
            'resource_lease': self.resource_lease.copy(),
            'causal_right_gate': self.causal_right_gate.copy(),
            'last_formal_context': self.last_formal_context.copy(),
            'last_change_features': self.last_change_features.copy(),
            'last_change_state': self.last_change_state.copy(),
            'last_change_mechanism': self.last_change_mechanism.copy(),
            'semantic_episode_active': self.semantic_episode_active.copy(),
            'semantic_episode_timer': self.semantic_episode_timer.copy(),
            'semantic_episode_start_debt': self.semantic_episode_start_debt.copy(),
            'semantic_episode_peak_debt': self.semantic_episode_peak_debt.copy(),
            'semantic_episode_uptake': self.semantic_episode_uptake.copy(),
            'semantic_episode_contaminated': self.semantic_episode_contaminated.copy(),
            'semantic_episode_updates': self.semantic_episode_updates,
            'last_event_flags': self.last_event_flags,
            'last_reactive': self.last_reactive,
            'last_uptake': self.last_uptake,
            'last_divisions': self.last_divisions,
            'last_hgt_count': self.last_hgt_count,
            'last_audit_result': self.last_audit_result,
            'last_feedback_gate': self.last_feedback_gate,
            'last_feedback_source': self.last_feedback_source,
            'feedback_events': self.feedback_events,
            'reopened_cells': self.reopened_cells,
            'lease_reductions': self.lease_reductions,
            'lease_recoveries': self.lease_recoveries,
            'formal_hgt_activation_age': self.formal_hgt_activation_age,
            'formal_last_start_age': self.formal_last_start_age,
            'formal_no_program_until': self.formal_no_program_until,
        })
        return state

    @classmethod
    def from_state(cls,state):
        base = p2.EightCellMaterialTissue.from_state(state)
        base.__class__ = cls
        obj = base
        obj._init_formal_runtime(rng_seed=0, formal_enabled=state.get('formal_enabled',False))
        obj.formal_rng.bit_generator.state = state.get('formal_rng_state',obj.formal_rng.bit_generator.state)
        obj.formal_enabled = bool(state.get('formal_enabled',False))
        obj.controller_mature = bool(state.get('controller_mature',False))
        obj.controller_development = float(state.get('controller_development',0.0))
        obj.controller_failed = bool(state.get('controller_failed',False))
        obj.controller_atp_spent = float(state.get('controller_atp_spent',0.0))
        obj.controller_material_wear = float(state.get('controller_material_wear',0.0))
        obj.auditor = MaterialCausalAuditor.from_state(state.get('auditor',{}))
        obj.calibrator = CausalCalibrationLedger.from_state(state.get('calibrator',{}))
        obj.change_sentinel = MaterialChangeSentinel.from_state(state.get('change_sentinel',{}))
        obj.compiler = SmallFalsificationCompiler.from_state(state.get('compiler',{}))
        array_defaults = {
            'intervention_gate': np.ones(p2.P2_CELL_COUNT),
            'resource_lease': np.ones(p2.P2_CELL_COUNT),
            'causal_right_gate': np.ones((p2.P2_CELL_COUNT,FORMAL_CHANNELS))*0.5,
            'last_formal_context': np.zeros(FORMAL_CONTEXT_DIM),
            'last_change_features': np.zeros(FORMAL_FEATURE_DIM),
            'last_change_state': np.zeros(FORMAL_STATE_FEATURE_DIM),
            'last_change_mechanism': np.zeros(FORMAL_MECHANISM_FEATURE_DIM),
            'semantic_episode_active': np.zeros(FORMAL_LIGAND_COUNT,dtype=bool),
            'semantic_episode_timer': np.zeros(FORMAL_LIGAND_COUNT),
            'semantic_episode_start_debt': np.zeros((FORMAL_LIGAND_COUNT,FORMAL_CHANNELS)),
            'semantic_episode_peak_debt': np.zeros((FORMAL_LIGAND_COUNT,FORMAL_CHANNELS)),
            'semantic_episode_uptake': np.zeros(FORMAL_LIGAND_COUNT),
            'semantic_episode_contaminated': np.zeros(FORMAL_LIGAND_COUNT,dtype=bool),
        }
        for name, default in array_defaults.items():
            setattr(obj, name, np.asarray(state.get(name,default)).copy())
        scalar_defaults = {
            'semantic_episode_updates': 0,
            'last_event_flags': 0,
            'last_reactive': 0.0,
            'last_uptake': 0.0,
            'last_divisions': 0,
            'last_hgt_count': 0,
            'last_feedback_gate': 0.0,
            'last_feedback_source': 'none',
            'feedback_events': 0,
            'reopened_cells': 0,
            'lease_reductions': 0,
            'lease_recoveries': 0,
            'formal_hgt_activation_age': -1.0,
            'formal_last_start_age': -1e9,
            'formal_no_program_until': 0.0,
        }
        for name, default in scalar_defaults.items():
            setattr(obj, name, state.get(name,default))
        obj.last_audit_result = state.get('last_audit_result')
        obj._learning_snapshot = None
        obj._gate_backup = None
        return obj


class FormalProtoCell(p2.P2ProtoCell):
    def __init__(self,*args,**kwargs):
        super(FormalProtoCell,self).__init__(*args,**kwargs); self._init_formal_state()
    def _init_formal_state(self):
        self.formal_gene_fingerprint=-1; self.formal_gene_installed=False; self.formal_hgt_activation_count=0
    def split(self,world):
        daughters=super(FormalProtoCell,self).split(world)
        if daughters is None:return None
        for d in daughters:
            d.__class__=FormalProtoCell; d._init_formal_state(); specs=formal_controller_specs(d)
            if specs:d.formal_gene_fingerprint=int(specs[0][0]); d.formal_gene_installed=True
        return daughters
    def state_dict(self):
        state=super(FormalProtoCell,self).state_dict(); state.update({'cell_class':'FormalProtoCell','formal_gene_fingerprint':self.formal_gene_fingerprint,'formal_gene_installed':self.formal_gene_installed,'formal_hgt_activation_count':self.formal_hgt_activation_count}); return state
    @classmethod
    def from_state(cls,rng,state):
        cell=p2.P2ProtoCell.from_state(rng,state); cell.__class__=cls; cell._init_formal_state()
        if state.get('p2_tissue') is not None:cell.p2_tissue=FormalMaterialTissue.from_state(state['p2_tissue'])
        cell.formal_gene_fingerprint=int(state.get('formal_gene_fingerprint',-1)); cell.formal_gene_installed=bool(state.get('formal_gene_installed',False)); cell.formal_hgt_activation_count=int(state.get('formal_hgt_activation_count',0)); return cell


class Formal06World(p2.P2World):
    def __init__(self,seed=101,initial_cells=1,config=None):
        config=config if config is not None else Formal06Config()
        if not isinstance(config,Formal06Config):config=Formal06Config(**config.state_dict())
        self.formal_seed=int(seed); self.formal_pre_steps=0; self.formal_post_steps=0; self.formal_tissue_creations=0; self.formal_gene_hgt_activations=0; self.formal_audit_results=0; self.formal_feedback_events=0
        super(Formal06World,self).__init__(seed=seed,initial_cells=initial_cells,config=config); self.config=config
        for cell in self.cells:
            old=cell.p2_tissue; cell.__class__=FormalProtoCell; cell._init_formal_state()
            if config.formal_install_gene:
                fp,installed=install_formal_cassette(cell,bootstrap_protein=config.formal_bootstrap_protein); cell.formal_gene_fingerprint=int(fp); cell.formal_gene_installed=bool(installed)
            if old is not None:
                cell.p2_tissue=FormalMaterialTissue.from_state(old.state_dict()); cell.p2_tissue.formal_enabled=formal_controller_activity(cell)>=0.016
        self.initial_total_material=self.total_material(); self.last_step_material_residual=0.0; self._ensure_all_p2_tissues()

    def _new_p2_tissue(self,cell):
        if self.config.p2_tissue_mode==p2.P2_MODE_NONE:return None
        params=p2.p2_gene_parameters(cell); activity=p2.p2_gene_activity(cell)
        if params is None or np.count_nonzero(activity>=0.016)<p2.P2_CELL_COUNT:return None
        seed=((self.p2_seed*1000003)^(int(cell.cell_id)*9176)^(int(cell.generation)*7919)^0x6A2B31)&0xFFFFFFFF
        tissue=FormalMaterialTissue(params,mode=self.config.p2_tissue_mode,rng_seed=seed,formal_enabled=formal_controller_activity(cell)>=0.016)
        tissue.ensure_attachments(self.port_for(cell.cell_id),activity); cell.p2_tissue=tissue; cell.p2_tissue_births+=1; self.p2_tissue_creations+=1; self.formal_tissue_creations+=1; return tissue

    def _ensure_all_p2_tissues(self):
        for cell in self.living_cells():
            if not isinstance(cell,FormalProtoCell):cell.__class__=FormalProtoCell; cell._init_formal_state()
            tissue=self._ensure_p2_tissue(cell)
            if tissue is not None and not isinstance(tissue,FormalMaterialTissue):cell.p2_tissue=FormalMaterialTissue.from_state(tissue.state_dict()); tissue=cell.p2_tissue
            specs=formal_controller_specs(cell)
            if specs and cell.formal_gene_fingerprint<0:cell.formal_gene_fingerprint=int(specs[0][0])
            if tissue is not None and tissue.formal_hgt_activation_age<0.0 and formal_controller_activity(cell)>=0.016 and not cell.formal_gene_installed and self.config.neural_hgt_enabled:
                tissue.formal_hgt_activation_age=float(self.age); cell.formal_hgt_activation_count+=1; self.formal_gene_hgt_activations+=1

    def _pre_p2_step(self,dt):
        for cell in list(self.living_cells()):
            tissue=self._ensure_p2_tissue(cell)
            if tissue is None:continue
            port=self.port_for(cell.cell_id)
            if isinstance(tissue,FormalMaterialTissue):tissue.formal_prepare(port,dt,self.config,cell,self)
            tissue.pre_step(port,dt,self.config,p2.p2_gene_activity(cell)); self.p2_pre_steps+=1; self.formal_pre_steps+=1

    def _post_p2_step(self,dt):
        for cell in list(self.living_cells()):
            tissue=cell.p2_tissue if isinstance(cell,FormalProtoCell) else None
            if tissue is None:continue
            port=self.port_for(cell.cell_id); tissue.post_step(port,dt,self.config)
            if isinstance(tissue,FormalMaterialTissue):
                result=tissue.formal_observe(port,dt,self.config,cell,self)
                if result is not None and result.get('status')!='aborted':self.formal_audit_results+=1; self.formal_feedback_events+=int(float(result.get('feedback_magnitude',0.0))>0.0)
            self.p2_post_steps+=1; self.formal_post_steps+=1
        self._ensure_all_p2_tissues()

    def finite(self):
        if not super(Formal06World, self).finite():
            return False
        for cell in self.cells:
            if not isinstance(cell, FormalProtoCell):
                return False
            tissue = cell.p2_tissue
            if tissue is None:
                continue
            if not isinstance(tissue, FormalMaterialTissue):
                return False
            arrays = (
                tissue.intervention_gate,
                tissue.resource_lease,
                tissue.causal_right_gate,
                tissue.calibrator.cell_trust,
                tissue.change_sentinel.channel_probability,
                tissue.change_sentinel.mechanism_channel_probability,
                tissue.change_sentinel.semantic_channel_probability,
                tissue.change_sentinel.source_probability,
                tissue.last_change_state,
                tissue.last_change_mechanism,
                tissue.semantic_episode_timer,
                tissue.semantic_episode_start_debt,
                tissue.semantic_episode_peak_debt,
                tissue.semantic_episode_uptake,
            )
            if not all(finite_array(value) for value in arrays):
                return False
            scalars = (
                tissue.change_sentinel.global_probability,
                tissue.change_sentinel.state_probability,
                tissue.change_sentinel.mechanism_probability,
                tissue.change_sentinel.semantic_probability,
                tissue.change_sentinel.genome_probability,
                tissue.last_feedback_gate,
            )
            if not all(np.isfinite(float(value)) for value in scalars):
                return False
        return True

    def summary(self):
        out = super(Formal06World, self).summary()
        tissues = [
            cell.p2_tissue for cell in self.living_cells()
            if isinstance(cell, FormalProtoCell)
            and isinstance(cell.p2_tissue, FormalMaterialTissue)
        ]
        total_weight = sum(float(np.sum(t.calibrator.weight)) for t in tissues)
        raw_error = sum(float(np.sum(t.calibrator.raw_abs_error)) for t in tissues)
        calibrated_error = sum(float(np.sum(t.calibrator.cal_abs_error)) for t in tissues)
        sign_correct = sum(float(np.sum(t.calibrator.sign_correct)) for t in tissues)
        sign_weight = sum(float(np.sum(t.calibrator.sign_weight)) for t in tissues)
        mean = lambda values: float(np.mean(list(values))) if tissues else 0.0
        out.update({
            'build': BUILD,
            'formal_schema': FORMAL_SCHEMA_VERSION,
            'formal_mode': self.config.formal_mode,
            'formal_tissues': len(tissues),
            'formal_controller_mature': int(sum(t.controller_mature for t in tissues)),
            'formal_controller_atp': float(sum(t.controller_atp_spent for t in tissues)),
            'formal_controller_wear': float(sum(t.controller_material_wear for t in tissues)),
            'formal_audits': int(sum(t.auditor.completed for t in tissues)),
            'formal_audit_supported': int(sum(t.auditor.supported for t in tissues)),
            'formal_audit_contradicted': int(sum(t.auditor.contradicted for t in tissues)),
            'formal_audit_inconclusive': int(sum(t.auditor.inconclusive for t in tissues)),
            'formal_audit_aborted': int(sum(t.auditor.aborted for t in tissues)),
            'formal_audit_quality': mean(t.auditor.last_quality for t in tissues),
            'formal_audit_cost_atp': float(sum(t.auditor.cumulative_cost_atp for t in tissues)),
            'formal_audit_cost_material': float(sum(t.auditor.cumulative_cost_material for t in tissues)),
            'formal_calibration_updates': int(sum(np.sum(t.calibrator.updates) for t in tissues)),
            'formal_raw_mae': raw_error / max(total_weight, 1e-9),
            'formal_calibrated_mae': calibrated_error / max(total_weight, 1e-9),
            'formal_calibration_sign_accuracy': sign_correct / max(sign_weight, 1e-9),
            'formal_change_probability': mean(t.change_sentinel.global_probability for t in tissues),
            'formal_change_state_probability': mean(t.change_sentinel.state_probability for t in tissues),
            'formal_change_mechanism_probability': mean(t.change_sentinel.mechanism_probability for t in tissues),
            'formal_change_semantic_probability': mean(t.change_sentinel.semantic_probability for t in tissues),
            'formal_change_genome_probability': mean(t.change_sentinel.genome_probability for t in tissues),
            'formal_change_events': int(sum(t.change_sentinel.events for t in tissues)),
            'formal_change_mechanism_events': int(sum(t.change_sentinel.mechanism_events for t in tissues)),
            'formal_change_semantic_events': int(sum(t.change_sentinel.semantic_events for t in tissues)),
            'formal_feedback_events': int(sum(t.feedback_events for t in tissues)),
            'formal_feedback_gate': mean(t.last_feedback_gate for t in tissues),
            'formal_mean_resource_lease': mean(np.mean(t.resource_lease) for t in tissues),
            'formal_lease_reductions': int(sum(t.lease_reductions for t in tissues)),
            'formal_lease_recoveries': int(sum(t.lease_recoveries for t in tissues)),
            'formal_semantic_episode_updates': int(sum(t.semantic_episode_updates for t in tissues)),
            'formal_reopened_cells': int(sum(t.reopened_cells for t in tissues)),
            'formal_compiled_programs': int(sum(t.compiler.compiled for t in tissues)),
            'formal_started_programs': int(sum(t.compiler.started for t in tissues)),
            'formal_hgt_activations': int(self.formal_gene_hgt_activations),
            'formal_pre_steps': self.formal_pre_steps,
            'formal_post_steps': self.formal_post_steps,
        })
        return out

    def state_dict(self):
        state=super(Formal06World,self).state_dict(); state.update({'save_version':SAVE_VERSION,'build':BUILD,'config':self.config.state_dict(),'cells':[c.state_dict() for c in self.cells],'formal_seed':self.formal_seed,'formal_pre_steps':self.formal_pre_steps,'formal_post_steps':self.formal_post_steps,'formal_tissue_creations':self.formal_tissue_creations,'formal_gene_hgt_activations':self.formal_gene_hgt_activations,'formal_audit_results':self.formal_audit_results,'formal_feedback_events':self.formal_feedback_events}); return state

    @classmethod
    def from_state(cls,state):
        base=dict(state); base['save_version']=p2.SAVE_VERSION; base['build']=p2.BUILD; keys=set(p2.P2Config().__dict__.keys()); base['config']={k:v for k,v in dict(state.get('config',{})).items() if k in keys}
        world=p2.P2World.from_state(base); world.__class__=cls; world.config=Formal06Config.from_state(state.get('config',{})); world.cells=[FormalProtoCell.from_state(world.rng,item) for item in state['cells']]; world.rng.bit_generator.state=state['rng_state']; world.formal_seed=int(state.get('formal_seed',state.get('p2_seed',101)))
        for n in ('formal_pre_steps','formal_post_steps','formal_tissue_creations','formal_gene_hgt_activations','formal_audit_results','formal_feedback_events'):setattr(world,n,int(state.get(n,0)))
        return world
    def save(self,path=SAVE_FILE):_atomic_pickle(path,self.state_dict())
    @classmethod
    def load(cls,path=SAVE_FILE):
        with open(path,'rb') as h:return cls.from_state(pickle.load(h))
    def clone(self):return Formal06World.from_state(self.state_dict())


def set_formal_runtime_mode(world,formal_mode):
    if formal_mode not in FORMAL_MODES:raise ValueError(formal_mode)
    world.config.formal_mode=formal_mode; world.config.audit_enabled=formal_mode!=FORMAL_MODE_NO_AUDIT; world.config.calibration_enabled=formal_mode!=FORMAL_MODE_NO_CALIBRATION; world.config.change_gated_feedback=formal_mode!=FORMAL_MODE_NO_CHANGE_GATE; world.config.feedback_enabled=formal_mode!=FORMAL_MODE_NO_FEEDBACK; world.config.falsification_enabled=formal_mode!=FORMAL_MODE_NO_FALSIFICATION; world.config.neural_hgt_enabled=formal_mode!=FORMAL_MODE_NO_HGT
    for cell in world.living_cells():
        t=getattr(cell,'p2_tissue',None)
        if isinstance(t,FormalMaterialTissue):t.auditor.enabled=world.config.audit_enabled; t.compiler.enabled=world.config.falsification_enabled; t.change_sentinel.enabled=world.config.change_detection_enabled
    return world


def _formal_tape_rng_state(seed,step,subsystem,item=0,stream=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(step),int(subsystem),int(item),int(stream),0x06F06])).bit_generator.state


def apply_formal_common_disturbance_tape(world,seed,step,stream=0):
    p2.apply_p2_common_disturbance_tape(world,seed,step,stream=stream)
    for cell in world.cells:
        t=getattr(cell,'p2_tissue',None)
        if isinstance(t,FormalMaterialTissue):t.formal_rng.bit_generator.state=_formal_tape_rng_state(seed,step,2,int(cell.cell_id),stream)
    return world


def run_headless_trial(seed=101,seconds=120.0,initial_cells=1,config=None):
    world=Formal06World(seed=seed,initial_cells=initial_cells,config=config or Formal06Config()); dt=1.0/SIM_HZ; margin=pre=post=0.0; n=pn=qn=0; start_uptake=world.p2_reward_uptake_total
    for _ in range(int(round(seconds*SIM_HZ))):
        if not world.living_cells():break
        world.step(dt); m=float(np.mean([c.autopoietic_margin() for c in world.living_cells()])) if world.living_cells() else 0.0; margin+=m; n+=1
        if world.age<world.config.p2_switch_age:pre+=m;pn+=1
        else:post+=m;qn+=1
    result=world.summary(); result.update({'mean_margin_over_life':margin/max(n,1),'pre_switch_mean_margin':pre/max(pn,1),'post_switch_mean_margin':post/max(qn,1),'uptake_during_trial':max(0.0,world.p2_reward_uptake_total-start_uptake),'finite':int(world.finite()),'final_mass_residual':world.matter_ledger_residual()}); return result


LOG_FIELDS=tuple(list(p2.LOG_FIELDS)+['formal_mode','formal_tissues','formal_controller_mature','formal_controller_atp','formal_controller_wear','formal_audits','formal_audit_supported','formal_audit_contradicted','formal_audit_inconclusive','formal_audit_aborted','formal_audit_quality','formal_audit_cost_atp','formal_audit_cost_material','formal_calibration_updates','formal_raw_mae','formal_calibrated_mae','formal_calibration_sign_accuracy','formal_change_probability','formal_change_state_probability','formal_change_mechanism_probability','formal_change_semantic_probability','formal_change_genome_probability','formal_change_events','formal_change_mechanism_events','formal_change_semantic_events','formal_feedback_events','formal_feedback_gate','formal_mean_resource_lease','formal_lease_reductions','formal_lease_recoveries','formal_semantic_episode_updates','formal_reopened_cells','formal_compiled_programs','formal_hgt_activations'])


class LongRunLogger(object):
    def __init__(self,world,path=LOG_FILE,interval=10.0):self.path=path;self.interval=float(interval);self.last_age=-1e9;self.session_id='formal-'+uuid.uuid4().hex[:10];self.rows=0;self.status='WAIT'
    def log(self,world,reason='periodic',force=False):
        if not force and world.age-self.last_age<self.interval:return False
        row={k:world.summary().get(k,'') for k in LOG_FIELDS}; row.update({'session_id':self.session_id,'reason':reason,'wall_time':time.time()}); exists=os.path.exists(self.path) and os.path.getsize(self.path)>0
        with open(self.path,'a',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=('session_id','reason','wall_time')+LOG_FIELDS)
            if not exists:w.writeheader()
            w.writerow(row)
        self.last_age=world.age;self.rows+=1;self.status='OK';return True


def generate_report(log_path=LOG_FILE,report_path=REPORT_FILE,session_path=SESSION_FILE):
    if not os.path.exists(log_path):return 'NO LOG'
    with open(log_path,'r',newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
    sessions={}
    for r in rows:sessions.setdefault(r['session_id'],[]).append(r)
    with open(session_path,'w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=('session_id','rows','final_age','final_cells','audits','feedback'));w.writeheader()
        for sid,items in sessions.items():
            last=items[-1];w.writerow({'session_id':sid,'rows':len(items),'final_age':last.get('age',''),'final_cells':last.get('cells',''),'audits':last.get('formal_audits',''),'feedback':last.get('formal_feedback_events','')})
    lines=[BUILD_LONG,'sessions: {}'.format(len(sessions)),'']
    for sid,items in sessions.items():
        last=items[-1];lines.append('{} age={} cells={} audits={} cal={} change={} feedback={} ledger={}'.format(sid,last.get('age',''),last.get('cells',''),last.get('formal_audits',''),last.get('formal_calibration_updates',''),last.get('formal_change_probability',''),last.get('formal_feedback_events',''),last.get('matter_residual','')))
    with open(report_path,'w',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
    return 'OK'


try:
    from scene import Scene,run,LANDSCAPE,background,fill,rect,ellipse,line,stroke,stroke_weight,text
    class SomaCellFormalScene(p2.SomaCellP2Scene):
        def setup(self):
            background(0.006,0.012,0.022)
            try:self.world=Formal06World.load(SAVE_FILE);self.save_status='LOAD'
            except Exception:self.world=Formal06World(seed=101,initial_cells=2,config=Formal06Config(p2_environment=p2.P2_ENV_CUE_REVERSAL));self.save_status='NEW'
            self.accumulator=0.0;self.last_wall=time.time();self.last_save_age=self.world.age;self.last_touch_wall=-10.0;self.paused=False;self.fps=0.0;self.sim_rate=0.0;self.telemetry_wall=time.time();self.telemetry_age=self.world.age;self.telemetry_frames=0;self.logger=LongRunLogger(self.world);self.logger.log(self.world,reason='start',force=True);self.report_status='WAIT'
        def draw(self):
            super(SomaCellFormalScene,self).draw();s=self.world.summary();fill(0.01,0.018,0.03,0.96);rect(0,102,self.size.w,56);fill(0.92,0.80,1.0);text('FORMAL audits {} S/C/I {}/{}/{} q {:.2f} cost {:.5f}'.format(s['formal_audits'],s['formal_audit_supported'],s['formal_audit_contradicted'],s['formal_audit_inconclusive'],s['formal_audit_quality'],s['formal_audit_cost_atp']),x=24,y=143,font_size=9,alignment=4);text('cal {} raw {:.4g} -> {:.4g} change {:.2f} gate {:.2f} reopen {}'.format(s['formal_calibration_updates'],s['formal_raw_mae'],s['formal_calibrated_mae'],s['formal_change_probability'],s['formal_feedback_gate'],s['formal_reopened_cells']),x=24,y=123,font_size=9,alignment=4)
        def stop(self):
            try:self.world.save(SAVE_FILE);self.save_status='OK'
            except Exception:self.save_status='ERR'
            self.logger.log(self.world,reason='stop',force=True);self.report_status=generate_report()
except ImportError:Scene=None


if __name__=='__main__':
    if Scene is None:print(run_headless_trial(seed=101,seconds=70.0,initial_cells=1,config=Formal06Config(p2_environment=p2.P2_ENV_CUE_REVERSAL)))
    else:run(SomaCellFormalScene(),LANDSCAPE,show_fps=False)
