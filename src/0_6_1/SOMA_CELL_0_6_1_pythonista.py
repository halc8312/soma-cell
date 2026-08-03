# coding: utf-8
"""SOMA-CELL 0.6.1 — bounded active causal diagnosis.

This module freezes SOMA-CELL 0.6.0 as the material causal substrate and adds
an evidence-sufficiency ledger plus a physically paid, safety-bounded diagnostic
exposure loop.  The controller distinguishes "no change" from "not enough
relevant experience", actively samples a local molecular cue when evidence is
missing, and prioritises a targeted ABBA/BAAB stop audit when a delayed molecular
outcome reverses.

No reward label, switch-time notification, direct position rewrite, free ATP,
free learned-state inheritance, or free gene transfer is introduced.
"""
from __future__ import division

import csv
import math
import os
import pickle
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F06_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6'))
P2_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p2'))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, F06_DIR, P2_DIR, P1_DIR, P0_DIR, BASE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_pythonista as f06

p2 = f06.p2
p1 = f06.p1
p0 = f06.p0
s5 = f06.s5
s4 = f06.s4
g2 = f06.g2

BUILD = 'SOMA-CELL 0.6.1'
BUILD_LONG = BUILD + ' | evidence sufficiency / bounded active diagnosis'
SAVE_VERSION = 1
SCHEMA_VERSION = '0.6.1-D1.4'
SAVE_FILE = 'soma_cell_0_6_1.pkl'
LOG_FILE = 'soma_cell_0_6_1_longrun.csv'
REPORT_FILE = 'soma_cell_0_6_1_report.txt'
SESSION_FILE = 'soma_cell_0_6_1_sessions.csv'
AUTO_SAVE_INTERVAL = 60.0
SIM_HZ = f06.SIM_HZ

clamp = f06.clamp
finite_array = f06.finite_array
_atomic_pickle = f06._atomic_pickle

DIAGNOSIS_ACTIVE = 'active'
DIAGNOSIS_RANDOM = 'random_cost_matched'
DIAGNOSIS_PASSIVE = 'passive'
DIAGNOSIS_OFF = 'off'
DIAGNOSIS_MODES = frozenset((
    DIAGNOSIS_ACTIVE, DIAGNOSIS_RANDOM, DIAGNOSIS_PASSIVE, DIAGNOSIS_OFF,
))

EVIDENCE_UNKNOWN = 'insufficient_evidence'
EVIDENCE_STABLE = 'supported_stable'
EVIDENCE_SUSPECTED = 'suspected_change'
EVIDENCE_CONFIRMED = 'confirmed_change'

DIAGNOSTIC_IDLE = 'idle'
DIAGNOSTIC_EXPOSE = 'expose'
DIAGNOSTIC_OBSERVE = 'observe'
DIAGNOSTIC_ABORTED = 'aborted'

ALT_LIGAND = int(s4.LIGAND_ALT)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(2, dtype=float)
    return vector / norm


def _copy_array_state(obj):
    state = dict(obj.__dict__)
    for key, value in list(state.items()):
        if isinstance(value, np.ndarray):
            state[key] = value.copy()
    return state


class Formal061Config(f06.Formal06Config):
    """Formal 0.6 plus evidence sufficiency and bounded active diagnosis."""

    def __init__(
        self,
        diagnosis_mode=DIAGNOSIS_ACTIVE,
        diagnosis_enabled=True,
        diagnosis_controller_cost=True,
        diagnosis_min_active_age=24.0,
        diagnosis_cooldown=4.0,
        diagnosis_exposure_duration=0.55,
        diagnosis_observation_duration=4.2,
        diagnosis_max_total_duration=5.75,
        diagnosis_motor_strength=0.05,
        diagnosis_transporter_strength=0.14,
        diagnosis_min_ligand_concentration=0.012,
        diagnosis_safe_energy_debt=0.9999,
        diagnosis_safe_boundary_debt=0.92,
        diagnosis_safe_damage_debt=0.92,
        diagnosis_abort_debt_delta=0.40,
        diagnosis_start_energy_debt=0.82,
        diagnosis_start_boundary_debt=0.78,
        diagnosis_start_damage_debt=0.68,
        diagnosis_min_deficit=0.28,
        diagnosis_stale_age=17.0,
        diagnosis_observation_target=1.20,
        diagnosis_postchange_target=0.78,
        diagnosis_intervention_target=0.62,
        diagnosis_change_threshold=0.40,
        diagnosis_audit_trigger=0.43,
        diagnosis_audit_min_concentration=0.005,
        diagnosis_audit_activation_delay=0.22,
        diagnosis_targeted_audit=True,
        diagnosis_freeze_fast_learning=True,
        diagnosis_atp_planning_rate=0.00048,
        diagnosis_atp_evidence_event=0.000035,
        diagnosis_protein_wear_rate=0.000020,
        diagnosis_evidence_wear=0.0000015,
        diagnosis_evidence_half_life=180.0,
        diagnosis_context_target_span=0.22,
        diagnosis_semantic_peak_target=0.025,
        diagnosis_semantic_dose_target=0.10,
        diagnosis_defer_log_interval=1.0,
        diagnosis_random_stream=0,
        diagnosis_candidate_ligands=(ALT_LIGAND,),
        diagnosis_generic_audit_probability=0.0,
        diagnosis_require_supported_baseline=True,
        diagnosis_min_suspicion=0.14,
        diagnosis_stale_refresh_multiplier=1.8,
        # Low-quality stop experiments may update calibration, but they must
        # not directly rewrite material leases.  A strong single experiment or
        # repeated sign-consistent experiments are required.
        diagnosis_feedback_min_quality=0.20,
        diagnosis_feedback_min_reliability=0.10,
        diagnosis_feedback_required_weight=0.52,
        diagnosis_feedback_replicate_window=42.0,
        # Active calibration of command -> realised motor transduction.
        mechanism_probe_enabled=True,
        mechanism_probe_min_active_age=25.0,
        mechanism_probe_cooldown=0.65,
        mechanism_probe_duration=0.34,
        mechanism_probe_strength=0.18,
        mechanism_probe_min_action=0.055,
        mechanism_probe_min_baseline_weight=1.8,
        mechanism_probe_loss_trigger=0.24,
        mechanism_probe_confirm_ratio=0.72,
        mechanism_probe_required_weight=1.55,
        mechanism_probe_recovery_ratio=0.90,
        mechanism_probe_atp_planning_rate=0.00030,
        mechanism_probe_wear_rate=0.000012,
        # P2's energy-debt coordinate is naturally close to one even in a
        # viable cell.  Safety is therefore a joint crisis test rather than a
        # raw energy-debt cutoff: probing is blocked only when both energy and
        # process capacity are near collapse, or structural channels are unsafe.
        mechanism_probe_crisis_energy_debt=0.9995,
        mechanism_probe_crisis_process_debt=0.97,
        # A paid assist is not granted merely because a fault was detected.
        # It first runs a short counterbalanced lease assay and is then bounded
        # by the measured physical/chemical efficiency of previous bursts.
        mechanism_assist_enabled=False,
        mechanism_assist_feedback_enabled=True,
        mechanism_assist_trial_window=0.30,
        mechanism_assist_washout=0.08,
        mechanism_assist_trial_strength=0.12,
        mechanism_assist_max_strength=0.32,
        mechanism_assist_min_quality=0.45,
        mechanism_assist_min_execution_fraction=0.75,
        mechanism_assist_min_force_rate=1.0e-4,
        mechanism_assist_min_atp_rate=1.0e-6,
        mechanism_assist_min_signal_rate=1.0e-7,
        mechanism_assist_min_replication_fraction=0.75,
        mechanism_assist_min_uptake_effect=1.5e-4,
        mechanism_assist_min_debt_effect=5.0e-4,
        mechanism_assist_max_harm=8.0e-4,
        mechanism_assist_max_energy_harm=3.0e-3,
        mechanism_assist_max_structural_harm=8.0e-4,
        mechanism_assist_max_fatigue_harm=2.5e-2,
        mechanism_assist_lease_duration=3.0,
        mechanism_assist_retest_cooldown=4.0,
        mechanism_assist_rejection_backoff=12.0,
        mechanism_assist_accept_retest_delay=8.0,
        mechanism_assist_max_trials_per_confirmation=2,
        mechanism_assist_lease_recovery_tau=8.0,
        mechanism_assist_efficiency_floor=-0.08,
        mechanism_assist_safe_energy_debt=0.88,
        mechanism_assist_safe_structural_debt=0.78,
        # A confirmed loss of actuator gain can make continued high motor
        # signalling metabolically futile.  The default response is therefore
        # a short, reversible conservation lease on the most motor-dominant
        # cells, not an unbounded boost.
        mechanism_conservation_enabled=True,
        mechanism_conservation_feedback_enabled=True,
        mechanism_conservation_group_size=2,
        mechanism_conservation_loss_gain=0.72,
        mechanism_conservation_scale_floor=0.45,
        mechanism_conservation_scale_ceiling=0.78,
        mechanism_conservation_duration=14.0,
        mechanism_conservation_cooldown=4.0,
        # P2 development historically requested a fixed ATP chunk even for a
        # tiny repair and retained the unspent remainder in neural escrow.
        # 0.6.1 closes that accounting loophole before judging conservation.
        neural_escrow_sanitation_enabled=True,
        neural_escrow_atp_target=0.0020,
        neural_escrow_signal_target=0.00034,
        formal_controller_escrow_atp_target=0.0040,
        formal_controller_escrow_signal_target=0.0008,
        # A conservation lease must suppress actual neural work, not merely
        # shrink the final motor vector.  These lower caps preserve repair
        # matter while returning temporarily unused ATP/signal to the body.
        mechanism_conservation_quiescence_enabled=True,
        mechanism_conservation_atp_store_target=0.0020,
        mechanism_conservation_signal_store_target=0.00034,
        # Controlled physical failure switches.  They are OFF in the normal
        # build and exist only for positive-control/holdout experiments.
        mechanism_fault_enabled=False,
        mechanism_fault_age=36.0,
        mechanism_fault_gain_scale=0.25,
        mechanism_fault_wear=0.00020,
        **kwargs
    ):
        # The existing 0.6 safety gate remains authoritative.  0.6.1 gathers
        # missing evidence; it does not loosen strong feedback by default.
        # In explicit OFF mode, preserve the exact frozen 0.6 defaults so the
        # descendant is a true no-op compatibility layer.
        mode = str(diagnosis_mode)
        if mode != DIAGNOSIS_OFF:
            kwargs.setdefault('change_semantic_min_episodes', 4.0)
            kwargs.setdefault('change_semantic_window', 3.0)
            kwargs.setdefault('audit_min_active_age', 24.0)
            kwargs.setdefault('audit_cooldown', 4.0)
            kwargs.setdefault('audit_measure_duration', 0.28)
            kwargs.setdefault('audit_washout_duration', 0.030)
            kwargs.setdefault('audit_max_retries', 1)
            kwargs.setdefault('formal_auto_start', True)
        super(Formal061Config, self).__init__(**kwargs)
        if mode not in DIAGNOSIS_MODES:
            raise ValueError('unknown diagnosis mode: {}'.format(mode))
        if not bool(diagnosis_controller_cost):
            raise ValueError('SOMA-CELL 0.6.1 forbids costless active diagnosis')
        self.diagnosis_mode = mode
        self.diagnosis_enabled = bool(diagnosis_enabled) and mode != DIAGNOSIS_OFF
        self.diagnosis_controller_cost = True
        self.diagnosis_min_active_age = float(diagnosis_min_active_age)
        self.diagnosis_cooldown = float(diagnosis_cooldown)
        self.diagnosis_exposure_duration = float(diagnosis_exposure_duration)
        self.diagnosis_observation_duration = float(diagnosis_observation_duration)
        self.diagnosis_max_total_duration = float(diagnosis_max_total_duration)
        self.diagnosis_motor_strength = float(diagnosis_motor_strength)
        self.diagnosis_transporter_strength = float(diagnosis_transporter_strength)
        self.diagnosis_min_ligand_concentration = float(diagnosis_min_ligand_concentration)
        self.diagnosis_safe_energy_debt = float(diagnosis_safe_energy_debt)
        self.diagnosis_safe_boundary_debt = float(diagnosis_safe_boundary_debt)
        self.diagnosis_safe_damage_debt = float(diagnosis_safe_damage_debt)
        self.diagnosis_abort_debt_delta = float(diagnosis_abort_debt_delta)
        self.diagnosis_start_energy_debt = float(diagnosis_start_energy_debt)
        self.diagnosis_start_boundary_debt = float(diagnosis_start_boundary_debt)
        self.diagnosis_start_damage_debt = float(diagnosis_start_damage_debt)
        self.diagnosis_min_deficit = float(diagnosis_min_deficit)
        self.diagnosis_stale_age = float(diagnosis_stale_age)
        self.diagnosis_observation_target = float(diagnosis_observation_target)
        self.diagnosis_postchange_target = float(diagnosis_postchange_target)
        self.diagnosis_intervention_target = float(diagnosis_intervention_target)
        self.diagnosis_change_threshold = float(diagnosis_change_threshold)
        self.diagnosis_audit_trigger = float(diagnosis_audit_trigger)
        self.diagnosis_audit_min_concentration = float(diagnosis_audit_min_concentration)
        self.diagnosis_audit_activation_delay = float(diagnosis_audit_activation_delay)
        self.diagnosis_targeted_audit = bool(diagnosis_targeted_audit)
        self.diagnosis_freeze_fast_learning = bool(diagnosis_freeze_fast_learning)
        self.diagnosis_atp_planning_rate = float(diagnosis_atp_planning_rate)
        self.diagnosis_atp_evidence_event = float(diagnosis_atp_evidence_event)
        self.diagnosis_protein_wear_rate = float(diagnosis_protein_wear_rate)
        self.diagnosis_evidence_wear = float(diagnosis_evidence_wear)
        self.diagnosis_evidence_half_life = float(diagnosis_evidence_half_life)
        self.diagnosis_context_target_span = float(diagnosis_context_target_span)
        self.diagnosis_semantic_peak_target = float(diagnosis_semantic_peak_target)
        self.diagnosis_semantic_dose_target = float(diagnosis_semantic_dose_target)
        self.diagnosis_defer_log_interval = float(diagnosis_defer_log_interval)
        self.diagnosis_random_stream = int(diagnosis_random_stream)
        self.diagnosis_candidate_ligands = tuple(int(x) for x in diagnosis_candidate_ligands)
        self.diagnosis_generic_audit_probability = float(diagnosis_generic_audit_probability)
        self.diagnosis_require_supported_baseline = bool(diagnosis_require_supported_baseline)
        self.diagnosis_min_suspicion = float(diagnosis_min_suspicion)
        self.diagnosis_stale_refresh_multiplier = float(diagnosis_stale_refresh_multiplier)
        self.diagnosis_feedback_min_quality = float(diagnosis_feedback_min_quality)
        self.diagnosis_feedback_min_reliability = float(diagnosis_feedback_min_reliability)
        self.diagnosis_feedback_required_weight = float(diagnosis_feedback_required_weight)
        self.diagnosis_feedback_replicate_window = float(diagnosis_feedback_replicate_window)
        self.mechanism_probe_enabled = bool(mechanism_probe_enabled)
        self.mechanism_probe_min_active_age = float(mechanism_probe_min_active_age)
        self.mechanism_probe_cooldown = float(mechanism_probe_cooldown)
        self.mechanism_probe_duration = float(mechanism_probe_duration)
        self.mechanism_probe_strength = float(mechanism_probe_strength)
        self.mechanism_probe_min_action = float(mechanism_probe_min_action)
        self.mechanism_probe_min_baseline_weight = float(mechanism_probe_min_baseline_weight)
        self.mechanism_probe_loss_trigger = float(mechanism_probe_loss_trigger)
        self.mechanism_probe_confirm_ratio = float(mechanism_probe_confirm_ratio)
        self.mechanism_probe_required_weight = float(mechanism_probe_required_weight)
        self.mechanism_probe_recovery_ratio = float(mechanism_probe_recovery_ratio)
        self.mechanism_probe_atp_planning_rate = float(mechanism_probe_atp_planning_rate)
        self.mechanism_probe_wear_rate = float(mechanism_probe_wear_rate)
        self.mechanism_probe_crisis_energy_debt = float(mechanism_probe_crisis_energy_debt)
        self.mechanism_probe_crisis_process_debt = float(mechanism_probe_crisis_process_debt)
        self.mechanism_assist_enabled = bool(mechanism_assist_enabled)
        self.mechanism_assist_feedback_enabled = bool(mechanism_assist_feedback_enabled)
        self.mechanism_assist_trial_window = float(mechanism_assist_trial_window)
        self.mechanism_assist_washout = float(mechanism_assist_washout)
        self.mechanism_assist_trial_strength = float(mechanism_assist_trial_strength)
        self.mechanism_assist_max_strength = float(mechanism_assist_max_strength)
        self.mechanism_assist_min_quality = float(mechanism_assist_min_quality)
        self.mechanism_assist_min_execution_fraction = float(mechanism_assist_min_execution_fraction)
        self.mechanism_assist_min_force_rate = float(mechanism_assist_min_force_rate)
        self.mechanism_assist_min_atp_rate = float(mechanism_assist_min_atp_rate)
        self.mechanism_assist_min_signal_rate = float(mechanism_assist_min_signal_rate)
        self.mechanism_assist_min_replication_fraction = float(mechanism_assist_min_replication_fraction)
        self.mechanism_assist_min_uptake_effect = float(mechanism_assist_min_uptake_effect)
        self.mechanism_assist_min_debt_effect = float(mechanism_assist_min_debt_effect)
        self.mechanism_assist_max_harm = float(mechanism_assist_max_harm)
        self.mechanism_assist_max_energy_harm = float(mechanism_assist_max_energy_harm)
        self.mechanism_assist_max_structural_harm = float(mechanism_assist_max_structural_harm)
        self.mechanism_assist_max_fatigue_harm = float(mechanism_assist_max_fatigue_harm)
        self.mechanism_assist_lease_duration = float(mechanism_assist_lease_duration)
        self.mechanism_assist_retest_cooldown = float(mechanism_assist_retest_cooldown)
        self.mechanism_assist_rejection_backoff = float(mechanism_assist_rejection_backoff)
        self.mechanism_assist_accept_retest_delay = float(mechanism_assist_accept_retest_delay)
        self.mechanism_assist_max_trials_per_confirmation = int(mechanism_assist_max_trials_per_confirmation)
        self.mechanism_assist_lease_recovery_tau = float(mechanism_assist_lease_recovery_tau)
        self.mechanism_assist_efficiency_floor = float(mechanism_assist_efficiency_floor)
        self.mechanism_assist_safe_energy_debt = float(mechanism_assist_safe_energy_debt)
        self.mechanism_assist_safe_structural_debt = float(mechanism_assist_safe_structural_debt)
        self.mechanism_conservation_enabled = bool(mechanism_conservation_enabled)
        self.mechanism_conservation_feedback_enabled = bool(mechanism_conservation_feedback_enabled)
        self.mechanism_conservation_group_size = int(mechanism_conservation_group_size)
        self.mechanism_conservation_loss_gain = float(mechanism_conservation_loss_gain)
        self.mechanism_conservation_scale_floor = float(mechanism_conservation_scale_floor)
        self.mechanism_conservation_scale_ceiling = float(mechanism_conservation_scale_ceiling)
        self.mechanism_conservation_duration = float(mechanism_conservation_duration)
        self.mechanism_conservation_cooldown = float(mechanism_conservation_cooldown)
        self.neural_escrow_sanitation_enabled = bool(neural_escrow_sanitation_enabled)
        self.neural_escrow_atp_target = float(neural_escrow_atp_target)
        self.neural_escrow_signal_target = float(neural_escrow_signal_target)
        self.formal_controller_escrow_atp_target = float(formal_controller_escrow_atp_target)
        self.formal_controller_escrow_signal_target = float(formal_controller_escrow_signal_target)
        self.mechanism_conservation_quiescence_enabled = bool(mechanism_conservation_quiescence_enabled)
        self.mechanism_conservation_atp_store_target = float(mechanism_conservation_atp_store_target)
        self.mechanism_conservation_signal_store_target = float(mechanism_conservation_signal_store_target)
        self.mechanism_fault_enabled = bool(mechanism_fault_enabled)
        self.mechanism_fault_age = float(mechanism_fault_age)
        self.mechanism_fault_gain_scale = float(mechanism_fault_gain_scale)
        self.mechanism_fault_wear = float(mechanism_fault_wear)

    @classmethod
    def from_state(cls, state):
        return cls(**dict(state))


class EvidenceSufficiencyLedger(object):
    """Distinguish evidence absence from evidence for no causal change."""

    def __init__(self, ligand_count=f06.FORMAL_LIGAND_COUNT, channels=f06.FORMAL_CHANNELS):
        self.ligand_count = int(ligand_count)
        self.channels = int(channels)
        shape = (self.ligand_count, self.channels)
        self.observation_weight = np.zeros(shape, dtype=float)
        self.postchange_weight = np.zeros(shape, dtype=float)
        self.baseline_mean = np.zeros(shape, dtype=float)
        self.recent_mean = np.zeros(shape, dtype=float)
        self.variance = np.ones(shape, dtype=float) * 2.5e-5
        self.surprise = np.zeros(shape, dtype=float)
        self.change_evidence = np.zeros(shape, dtype=float)
        self.last_observation_age = np.full(shape, -1e9, dtype=float)
        self.context_min = np.ones((self.ligand_count, self.channels), dtype=float)
        self.context_max = np.zeros((self.ligand_count, self.channels), dtype=float)
        self.intervention_weight = np.zeros(self.channels, dtype=float)
        self.last_intervention_age = np.full(self.channels, -1e9, dtype=float)
        self.semantic_events = 0
        self.audit_events = 0
        self.last_ligand = -1
        self.last_channel = -1
        self.last_status = EVIDENCE_UNKNOWN
        self.max_change_evidence = 0.0
        self.last_decay_age = None
        self.first_observation_sufficient_age = np.full(shape, -1e9, dtype=float)
        self.first_suspected_age = np.full(shape, -1e9, dtype=float)
        self.first_confirmed_age = np.full(shape, -1e9, dtype=float)

    def decay(self, age, config):
        age = float(age)
        if self.last_decay_age is None:
            self.last_decay_age = age
            return
        elapsed_step = max(0.0, age - float(self.last_decay_age))
        self.last_decay_age = age
        if elapsed_step <= 0.0:
            return
        half_life = max(1.0, float(config.diagnosis_evidence_half_life))
        obs_factor = math.exp(-math.log(2.0) * elapsed_step / half_life)
        post_factor = math.exp(-math.log(2.0) * elapsed_step / max(1.0, 0.72 * half_life))
        intervention_factor = math.exp(-math.log(2.0) * elapsed_step / max(1.0, 1.15 * half_life))
        surprise_factor = math.exp(-math.log(2.0) * elapsed_step / max(1.0, 0.55 * half_life))
        self.observation_weight *= obs_factor
        self.postchange_weight *= post_factor
        self.intervention_weight *= intervention_factor
        self.surprise *= surprise_factor
        self.change_evidence *= surprise_factor

    def observe_semantic(self, ligand, outcome, quality, age, context, config):
        ligand = int(ligand)
        if ligand < 0 or ligand >= self.ligand_count:
            return None
        outcome = np.asarray(outcome, dtype=float)
        if outcome.shape != (self.channels,):
            raise ValueError('semantic outcome shape mismatch')
        q = clamp(float(quality), 0.02, 1.0)
        context = np.asarray(context, dtype=float)
        if context.size >= self.channels:
            context4 = np.clip(context[:self.channels], 0.0, 1.0)
        else:
            context4 = np.zeros(self.channels, dtype=float)

        for k in range(self.channels):
            old_weight = float(self.observation_weight[ligand, k])
            value = float(outcome[k])
            if old_weight < max(1.0, 0.72 * config.diagnosis_observation_target):
                alpha = clamp(0.38 * q, 0.08, 0.45)
                if old_weight <= 1e-12:
                    self.baseline_mean[ligand, k] = value
                    self.recent_mean[ligand, k] = value
                else:
                    self.baseline_mean[ligand, k] += alpha * (value - self.baseline_mean[ligand, k])
                    self.recent_mean[ligand, k] += alpha * (value - self.recent_mean[ligand, k])
                residual = value - self.baseline_mean[ligand, k]
                self.variance[ligand, k] = 0.85 * self.variance[ligand, k] + 0.15 * residual * residual
                surprise = 0.0
            else:
                recent_alpha = clamp(0.42 * q, 0.10, 0.48)
                self.recent_mean[ligand, k] += recent_alpha * (value - self.recent_mean[ligand, k])
                scale = math.sqrt(max(1e-8, self.variance[ligand, k] + 2.5e-5))
                delta = self.recent_mean[ligand, k] - self.baseline_mean[ligand, k]
                z = abs(delta) / scale
                baseline_sign = f06._sign(self.baseline_mean[ligand, k], 0.0030)
                recent_sign = f06._sign(self.recent_mean[ligand, k], 0.0030)
                sign_flip = bool(baseline_sign != 0.0 and recent_sign != 0.0 and baseline_sign != recent_sign)
                magnitude_gate = clamp((abs(delta) - 0.0040) / 0.020, 0.0, 1.0)
                surprise = clamp(max(0.78 if sign_flip else 0.0, (z - 1.15) / 3.4, 0.55 * magnitude_gate), 0.0, 1.0)
                if surprise < 0.22:
                    residual = value - self.baseline_mean[ligand, k]
                    self.baseline_mean[ligand, k] += 0.025 * q * residual
                    cap = 25.0 * (self.variance[ligand, k] + 2.5e-5)
                    self.variance[ligand, k] = 0.975 * self.variance[ligand, k] + 0.025 * min(residual * residual, cap)
                else:
                    self.postchange_weight[ligand, k] += q
            self.observation_weight[ligand, k] += q
            self.surprise[ligand, k] = max(0.82 * self.surprise[ligand, k], surprise)
            post_gate = clamp(self.postchange_weight[ligand, k] / max(config.diagnosis_postchange_target, 1e-9), 0.0, 1.0)
            self.change_evidence[ligand, k] = clamp(self.surprise[ligand, k] * (0.30 + 0.70 * post_gate), 0.0, 1.0)
            self.last_observation_age[ligand, k] = float(age)
            self.context_min[ligand, k] = min(self.context_min[ligand, k], float(context4[k]))
            self.context_max[ligand, k] = max(self.context_max[ligand, k], float(context4[k]))

        for k in range(self.channels):
            obs = self.observation_sufficiency(ligand, k, age, config)
            status = self.status(ligand, k, age, config)
            if obs >= 0.42 and self.first_observation_sufficient_age[ligand, k] < -1e8:
                self.first_observation_sufficient_age[ligand, k] = float(age)
            if status in (EVIDENCE_SUSPECTED, EVIDENCE_CONFIRMED) and self.first_suspected_age[ligand, k] < -1e8:
                self.first_suspected_age[ligand, k] = float(age)
            if status == EVIDENCE_CONFIRMED and self.first_confirmed_age[ligand, k] < -1e8:
                self.first_confirmed_age[ligand, k] = float(age)

        self.semantic_events += 1
        index = np.unravel_index(int(np.argmax(self.change_evidence)), self.change_evidence.shape)
        self.last_ligand, self.last_channel = int(index[0]), int(index[1])
        self.last_status = self.status(self.last_ligand, self.last_channel, age, config)
        self.max_change_evidence = max(self.max_change_evidence, float(np.max(self.change_evidence)))
        return self.diagnostics_for(ligand, int(np.argmax(self.change_evidence[ligand])), age, config)

    def observe_audit(self, channel, quality, age, ligand=None, config=None):
        channel = int(channel)
        if channel < 0 or channel >= self.channels:
            return
        q = clamp(float(quality), 0.0, 1.0)
        self.intervention_weight[channel] += q
        self.last_intervention_age[channel] = float(age)
        self.audit_events += 1
        if ligand is not None and config is not None:
            ligand = int(ligand)
            if 0 <= ligand < self.ligand_count:
                status = self.status(ligand, channel, age, config)
                if status == EVIDENCE_CONFIRMED and self.first_confirmed_age[ligand, channel] < -1e8:
                    self.first_confirmed_age[ligand, channel] = float(age)

    def observation_sufficiency(self, ligand, channel, age, config):
        ligand = int(ligand); channel = int(channel)
        count = clamp(self.observation_weight[ligand, channel] / max(config.diagnosis_observation_target, 1e-9), 0.0, 1.0)
        span = max(0.0, self.context_max[ligand, channel] - self.context_min[ligand, channel])
        coverage = clamp(span / max(config.diagnosis_context_target_span, 1e-9), 0.0, 1.0)
        stale = clamp((float(age) - self.last_observation_age[ligand, channel]) / max(config.diagnosis_stale_age, 1e-9), 0.0, 1.0)
        return clamp(count * (0.72 + 0.28 * coverage) * (1.0 - 0.35 * stale), 0.0, 1.0)

    def intervention_sufficiency(self, channel, age, config):
        channel = int(channel)
        count = clamp(self.intervention_weight[channel] / max(config.diagnosis_intervention_target, 1e-9), 0.0, 1.0)
        stale = clamp((float(age) - self.last_intervention_age[channel]) / max(1.5 * config.diagnosis_stale_age, 1e-9), 0.0, 1.0)
        return clamp(count * (1.0 - 0.25 * stale), 0.0, 1.0)

    def deficit(self, ligand, channel, age, config):
        obs = self.observation_sufficiency(ligand, channel, age, config)
        intervention = self.intervention_sufficiency(channel, age, config)
        change = float(self.change_evidence[ligand, channel])
        # Suspected change requires fresh post-change observations and eventually
        # an intervention.  Stable claims mostly need observational refresh.
        required = 0.72 + 0.28 * change
        available = 0.82 * obs + 0.18 * intervention
        return clamp(required - available, 0.0, 1.0)

    def status(self, ligand, channel, age, config):
        obs = self.observation_sufficiency(ligand, channel, age, config)
        change = float(self.change_evidence[ligand, channel])
        post = clamp(self.postchange_weight[ligand, channel] / max(config.diagnosis_postchange_target, 1e-9), 0.0, 1.0)
        intervention = self.intervention_sufficiency(channel, age, config)
        if obs < 0.42:
            return EVIDENCE_UNKNOWN
        if change >= config.diagnosis_change_threshold and post >= 0.55:
            if intervention >= 0.55:
                return EVIDENCE_CONFIRMED
            return EVIDENCE_SUSPECTED
        return EVIDENCE_STABLE

    def best_target(self, concentrations, debt, age, config):
        concentrations = np.asarray(concentrations, dtype=float)
        debt = np.asarray(debt, dtype=float)
        best = None
        candidates = tuple(getattr(config, 'diagnosis_candidate_ligands', tuple(range(min(self.ligand_count, len(concentrations))))))
        for ligand in candidates:
            if ligand < 0 or ligand >= min(self.ligand_count, len(concentrations)):
                continue
            present = clamp(float(concentrations[ligand]) / max(config.diagnosis_min_ligand_concentration, 1e-9), 0.0, 1.0)
            if present <= 0.0:
                continue
            for channel in range(self.channels):
                deficit = self.deficit(ligand, channel, age, config)
                change = float(self.change_evidence[ligand, channel])
                stale = clamp((float(age) - self.last_observation_age[ligand, channel]) / max(config.diagnosis_stale_age, 1e-9), 0.0, 1.0)
                relevance = 0.30 + 0.70 * clamp(float(debt[channel]) + abs(self.baseline_mean[ligand, channel]), 0.0, 1.0)
                score = present * relevance * (0.48 * deficit + 0.42 * change + 0.10 * stale)
                candidate = (score, ligand, channel, deficit, change)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best

    def best_change_target(self, age, config):
        best = None
        for ligand in range(self.ligand_count):
            for channel in range(self.channels):
                change = float(self.change_evidence[ligand, channel])
                obs = self.observation_sufficiency(ligand, channel, age, config)
                score = change * (0.55 + 0.45 * obs)
                candidate = (score, ligand, channel)
                if best is None or score > best[0]:
                    best = candidate
        return best

    def diagnostics_for(self, ligand, channel, age, config):
        return {
            'ligand': int(ligand),
            'channel': int(channel),
            'status': self.status(ligand, channel, age, config),
            'observation_sufficiency': self.observation_sufficiency(ligand, channel, age, config),
            'intervention_sufficiency': self.intervention_sufficiency(channel, age, config),
            'deficit': self.deficit(ligand, channel, age, config),
            'surprise': float(self.surprise[ligand, channel]),
            'change_evidence': float(self.change_evidence[ligand, channel]),
            'baseline': float(self.baseline_mean[ligand, channel]),
            'recent': float(self.recent_mean[ligand, channel]),
            'observation_weight': float(self.observation_weight[ligand, channel]),
            'postchange_weight': float(self.postchange_weight[ligand, channel]),
            'first_observation_sufficient_age': float(self.first_observation_sufficient_age[ligand, channel]),
            'first_suspected_age': float(self.first_suspected_age[ligand, channel]),
            'first_confirmed_age': float(self.first_confirmed_age[ligand, channel]),
        }

    def diagnostics(self, age, config):
        target = self.best_change_target(age, config)
        if target is None:
            return {'status': EVIDENCE_UNKNOWN, 'max_change_evidence': 0.0}
        _, ligand, channel = target
        out = self.diagnostics_for(ligand, channel, age, config)
        out.update({
            'semantic_events': int(self.semantic_events),
            'audit_events': int(self.audit_events),
            'max_change_evidence': float(np.max(self.change_evidence)),
        })
        return out

    def finite(self):
        arrays = (
            self.observation_weight, self.postchange_weight, self.baseline_mean,
            self.recent_mean, self.variance, self.surprise, self.change_evidence,
            self.last_observation_age, self.context_min, self.context_max,
            self.intervention_weight, self.last_intervention_age,
            self.first_observation_sufficient_age, self.first_suspected_age,
            self.first_confirmed_age,
        )
        return bool(all(finite_array(value) for value in arrays))

    def state_dict(self):
        return _copy_array_state(self)

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('ligand_count', f06.FORMAL_LIGAND_COUNT), state.get('channels', f06.FORMAL_CHANNELS))
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=getattr(obj, key).dtype).copy()
            setattr(obj, key, value)
        return obj


class FeedbackReplicationGate(object):
    """Require quality and sign-consistent replication before lease changes.

    Calibration is still allowed to learn from weak audits.  This gate only
    protects the materially consequential feedback path.  A single very clean
    experiment may be sufficient; low-quality results must repeat many times
    and keep the same sign and target group.
    """

    def __init__(self, channels=f06.FORMAL_CHANNELS):
        self.channels = int(channels)
        self.weight = np.zeros(self.channels, dtype=float)
        self.last_sign = np.zeros(self.channels, dtype=np.int8)
        self.last_signature = np.full(self.channels, -1, dtype=np.int64)
        self.last_age = np.full(self.channels, -1e9, dtype=float)
        self.accepted = 0
        self.rejected_quality = 0
        self.rejected_consistency = 0
        self.last_result = None

    @staticmethod
    def _signature(mask):
        signature = 0
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            signature |= (1 << int(i))
        return int(signature)

    def allow(self, result, age, config):
        channel = int(result.get('channel', 0))
        quality = clamp(float(result.get('quality', 0.0)), 0.0, 1.0)
        reliability = clamp(float(result.get('reliability', 0.0)), 0.0, 1.0)
        effect_sign = int(f06._sign(float(result.get('target_effect', 0.0)), 1.5e-4))
        signature = self._signature(result.get('claim_mask', np.zeros(p2.P2_CELL_COUNT)))
        if (
            quality < config.diagnosis_feedback_min_quality
            or reliability < config.diagnosis_feedback_min_reliability
            or effect_sign == 0
            or signature <= 0
        ):
            self.rejected_quality += 1
            self.last_result = {
                'allowed': False, 'reason': 'quality', 'quality': quality,
                'reliability': reliability, 'channel': channel,
            }
            return False

        elapsed = max(0.0, float(age) - float(self.last_age[channel]))
        if elapsed > config.diagnosis_feedback_replicate_window:
            self.weight[channel] = 0.0
        consistent = bool(
            self.last_signature[channel] in (-1, signature)
            and self.last_sign[channel] in (0, effect_sign)
        )
        if not consistent:
            self.weight[channel] = 0.0
            self.rejected_consistency += 1
        else:
            decay = math.exp(-elapsed / max(config.diagnosis_feedback_replicate_window, 1e-6))
            self.weight[channel] *= decay
        contribution = quality * (0.65 + 0.35 * reliability)
        self.weight[channel] += contribution
        self.last_sign[channel] = effect_sign
        self.last_signature[channel] = signature
        self.last_age[channel] = float(age)
        allowed = bool(self.weight[channel] >= config.diagnosis_feedback_required_weight)
        if allowed:
            self.accepted += 1
            # Keep a small memory so the next consistent result is easier to
            # renew, but do not let one old event authorise feedback forever.
            self.weight[channel] *= 0.28
        self.last_result = {
            'allowed': allowed,
            'reason': 'replicated' if allowed else 'waiting-replication',
            'quality': quality, 'reliability': reliability,
            'channel': channel, 'weight': float(self.weight[channel]),
            'signature': signature, 'sign': effect_sign,
        }
        return allowed

    def finite(self):
        return bool(
            finite_array(self.weight) and finite_array(self.last_sign)
            and finite_array(self.last_signature) and finite_array(self.last_age)
        )

    def state_dict(self):
        return _copy_array_state(self)

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('channels', f06.FORMAL_CHANNELS))
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=getattr(obj, key).dtype).copy()
            setattr(obj, key, value)
        return obj


class MechanismGainLedger(object):
    """Materially observed command-to-force calibration history."""

    def __init__(self):
        self.baseline_gain = 0.0
        self.recent_gain = 0.0
        self.baseline_weight = 0.0
        self.recent_weight = 0.0
        self.last_gain = 0.0
        self.gain_ratio = 1.0
        self.loss_fraction = 0.0
        self.confirmation_weight = 0.0
        self.recovery_weight = 0.0
        self.confirmed = False
        self.last_age = -1e9
        self.confirmed_age = -1e9
        self.observations = 0
        self.probe_observations = 0
        self.confirmations = 0
        self.recoveries = 0

    def _update_ratio(self):
        if self.baseline_gain <= 1e-12:
            self.gain_ratio = 1.0
            self.loss_fraction = 0.0
        else:
            self.gain_ratio = clamp(self.recent_gain / self.baseline_gain, 0.0, 2.5)
            self.loss_fraction = clamp(1.0 - self.gain_ratio, 0.0, 1.0)

    def observe(self, gain, quality, age, config, probe=False, mechanism_probability=0.0):
        gain = max(0.0, float(gain))
        quality = clamp(float(quality), 0.0, 1.0)
        if not np.isfinite(gain) or quality <= 1e-9:
            return self.diagnostics()
        self.observations += 1
        self.probe_observations += int(bool(probe))
        self.last_gain = gain
        self.last_age = float(age)
        if self.recent_weight <= 1e-12:
            self.recent_gain = gain
        else:
            alpha = clamp(0.18 + 0.38 * quality, 0.12, 0.58)
            self.recent_gain += alpha * (gain - self.recent_gain)
        self.recent_weight += quality
        self._update_ratio()

        baseline_safe = bool(
            not self.confirmed
            and float(mechanism_probability) < 0.22
            and self.loss_fraction < 0.16
        )
        if self.baseline_weight < config.mechanism_probe_min_baseline_weight or baseline_safe:
            if self.baseline_weight <= 1e-12:
                self.baseline_gain = gain
            else:
                alpha = 0.10 * quality if baseline_safe else 0.30 * quality
                self.baseline_gain += alpha * (gain - self.baseline_gain)
            self.baseline_weight += quality
            self._update_ratio()

        if probe and self.baseline_weight >= config.mechanism_probe_min_baseline_weight:
            if self.gain_ratio <= config.mechanism_probe_confirm_ratio:
                self.confirmation_weight += quality
                self.recovery_weight *= 0.65
            elif self.gain_ratio >= config.mechanism_probe_recovery_ratio:
                self.recovery_weight += quality
                self.confirmation_weight *= 0.55
            else:
                self.confirmation_weight *= 0.88
                self.recovery_weight *= 0.88
            if (not self.confirmed) and self.confirmation_weight >= config.mechanism_probe_required_weight:
                self.confirmed = True
                self.confirmed_age = float(age)
                self.confirmations += 1
            if self.confirmed and self.recovery_weight >= config.mechanism_probe_required_weight:
                self.confirmed = False
                self.recoveries += 1
                self.confirmation_weight = 0.0
                self.recovery_weight = 0.0
                # A recovered actuator becomes the new reference only slowly.
                self.baseline_gain = max(self.baseline_gain, self.recent_gain)
                self._update_ratio()
        return self.diagnostics()

    def suspicion(self, config, mechanism_probability=0.0):
        baseline_ready = clamp(
            self.baseline_weight / max(config.mechanism_probe_min_baseline_weight, 1e-9),
            0.0, 1.0,
        )
        loss = clamp(
            (self.loss_fraction - config.mechanism_probe_loss_trigger)
            / max(1e-9, 1.0 - config.mechanism_probe_loss_trigger), 0.0, 1.0,
        )
        return clamp(baseline_ready * max(loss, float(mechanism_probability)), 0.0, 1.0)

    def diagnostics(self):
        return {
            'baseline_gain': float(self.baseline_gain),
            'recent_gain': float(self.recent_gain),
            'baseline_weight': float(self.baseline_weight),
            'recent_weight': float(self.recent_weight),
            'gain_ratio': float(self.gain_ratio),
            'loss_fraction': float(self.loss_fraction),
            'confirmation_weight': float(self.confirmation_weight),
            'confirmed': bool(self.confirmed),
            'observations': int(self.observations),
            'probe_observations': int(self.probe_observations),
        }

    def finite(self):
        return bool(all(np.isfinite(float(value)) for value in (
            self.baseline_gain, self.recent_gain, self.baseline_weight,
            self.recent_weight, self.last_gain, self.gain_ratio,
            self.loss_fraction, self.confirmation_weight, self.recovery_weight,
            self.last_age, self.confirmed_age,
        )))

    def state_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for key, value in state.items():
            setattr(obj, key, value)
        return obj


class PaidMechanismProbe(object):
    """A short paid pulse that measures realised actuator gain."""

    def __init__(self):
        self.active = False
        self.timer = 0.0
        self.cooldown_until = 0.0
        self.direction = np.zeros(2, dtype=float)
        self.request_strength = 0.0
        self.force_sum = 0.0
        self.request_sum = 0.0
        self.atp_sum = 0.0
        self.signal_sum = 0.0
        self.samples = 0
        self.started = 0
        self.completed = 0
        self.aborted = 0
        self.last_result = None
        self.cumulative_atp = 0.0
        self.cumulative_signal = 0.0
        self.cumulative_force = 0.0

    def start(self, direction, age, config):
        if self.active or float(age) < self.cooldown_until:
            return False
        direction = _unit(direction)
        if float(np.linalg.norm(direction)) <= 1e-10:
            return False
        self.active = True
        self.timer = max(0.12, float(config.mechanism_probe_duration))
        self.direction = direction
        self.request_strength = clamp(config.mechanism_probe_strength, 0.02, 0.65)
        self.force_sum = self.request_sum = self.atp_sum = self.signal_sum = 0.0
        self.samples = 0
        self.started += 1
        self.last_result = None
        return True

    def command(self):
        if not self.active:
            return None
        return self.direction * self.request_strength

    def record(self, report, dt, age, config, ledger, mechanism_probability=0.0):
        if not self.active:
            return None
        report = report or {}
        force = max(0.0, float(report.get('motor_force', 0.0)))
        atp = max(0.0, float(report.get('atp_spent', 0.0)))
        signal = max(0.0, float(report.get('signal_spent', 0.0)))
        self.force_sum += force
        self.request_sum += self.request_strength
        self.atp_sum += atp
        self.signal_sum += signal
        self.samples += 1
        self.cumulative_force += force
        self.cumulative_atp += atp
        self.cumulative_signal += signal
        self.timer -= float(dt)
        if self.timer > 0.0:
            return None
        gain = self.force_sum / max(self.request_sum, 1e-12)
        payment = clamp(
            min(
                self.atp_sum / max(1e-12, self.samples * 1e-7),
                self.signal_sum / max(1e-12, self.samples * 1e-8),
            ), 0.0, 1.0,
        )
        sample_quality = clamp(self.samples / max(2.0, config.mechanism_probe_duration * SIM_HZ), 0.0, 1.0)
        quality = clamp(0.35 + 0.40 * sample_quality + 0.25 * payment, 0.0, 1.0)
        diagnostics = ledger.observe(
            gain, quality, age, config, probe=True,
            mechanism_probability=mechanism_probability,
        )
        self.active = False
        self.cooldown_until = float(age) + config.mechanism_probe_cooldown
        self.completed += 1
        self.last_result = {
            'gain': float(gain), 'quality': float(quality),
            'atp': float(self.atp_sum), 'signal': float(self.signal_sum),
            'force': float(self.force_sum), 'samples': int(self.samples),
            'diagnostics': diagnostics,
        }
        return self.last_result

    def abort(self, age, config, reason):
        self.active = False
        self.cooldown_until = float(age) + config.mechanism_probe_cooldown
        self.aborted += 1
        self.last_result = {'status': 'aborted', 'reason': str(reason), 'age': float(age)}

    def finite(self):
        return bool(
            finite_array(self.direction)
            and all(np.isfinite(float(v)) for v in (
                self.timer, self.cooldown_until, self.request_strength,
                self.force_sum, self.request_sum, self.atp_sum, self.signal_sum,
                self.cumulative_atp, self.cumulative_signal, self.cumulative_force,
            ))
        )

    def state_dict(self):
        return _copy_array_state(self)

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=float).copy()
            setattr(obj, key, value)
        return obj


class MechanismConservationLease(object):
    """Revocable resource conservation after replicated actuator failure.

    A failed actuator can make neural motor expenditure futile.  This lease
    temporarily reduces the resource rights of the most motor-dominant cells.
    It never deletes matter or learned weights and is only issued after paid
    motor probes have confirmed a loss of command-to-force gain.
    """

    def __init__(self, cells=p2.P2_CELL_COUNT):
        self.cells = int(cells)
        self.target_mask = np.zeros(self.cells, dtype=bool)
        self.scale = 1.0
        self.lease_until = 0.0
        self.cooldown_until = 0.0
        self.last_confirmation = 0
        self.issued = 0
        self.expired = 0
        self.revoked = 0
        self.last_issue_age = -1e9
        self.last_result = None
        self.cumulative_active_time = 0.0
        self.cumulative_returned_atp = 0.0
        self.cumulative_returned_signal = 0.0
        self.cumulative_suppressed_activity = 0.0
        self.quiescent_steps = 0
        # Diagnostics for the last target selection.  These are derived from
        # the tissue's own paid motor state, not from an external target.
        self.last_target_score = np.zeros(self.cells, dtype=float)
        self.last_target_alignment = np.zeros(self.cells, dtype=float)
        self.last_target_direction = np.zeros(2, dtype=float)
        self.alignment_fallbacks = 0

    def active(self, age):
        return bool(float(age) < self.lease_until and np.any(self.target_mask) and self.scale < 1.0 - 1e-10)

    def gate(self, age):
        gate = np.ones(self.cells, dtype=float)
        if self.active(age):
            gate[self.target_mask] = self.scale
        return gate

    def maybe_issue(self, tissue, age, debt, config):
        if (
            not config.mechanism_conservation_enabled
            or not config.mechanism_conservation_feedback_enabled
            or not tissue.mechanism_gain.confirmed
            or int(tissue.mechanism_gain.confirmations) <= int(self.last_confirmation)
            or float(age) < self.cooldown_until
        ):
            return False
        debt = np.asarray(debt, dtype=float)
        if not tissue._mechanism_safe(debt, config):
            return False
        capacity = tissue._material_capacity()
        # Suppress cells that are doing positive work along the tissue's own
        # current motor command.  Absolute motor magnitude alone can select an
        # antagonistic cell; quiescing such a cell may *increase* futile drive
        # instead of conserving it.  The target direction is therefore derived
        # exclusively from the current material tissue state.
        scalar = capacity * tissue.hidden * tissue.motor_gain
        local_vectors = np.tanh(2.6 * scalar)[:, None] * tissue.preferred
        direction = _unit(np.asarray(tissue.last_action, dtype=float).reshape(2))
        if float(np.linalg.norm(direction)) <= 1e-10:
            direction = _unit(np.sum(local_vectors, axis=0))
        alignment = local_vectors.dot(direction) if float(np.linalg.norm(direction)) > 1e-10 else np.zeros(self.cells, dtype=float)
        workload = np.abs(scalar)
        positive = np.maximum(0.0, alignment)
        score = positive * (0.55 + 0.45 * capacity) + 0.08 * workload * (positive > 0.0)
        valid = [int(i) for i in np.argsort(-score) if capacity[int(i)] > 0.15 and score[int(i)] > 1e-10]
        required = max(1, config.mechanism_conservation_group_size)
        chosen = valid[:required]
        if len(chosen) < required:
            # A nearly cancelled or quiescent motor command may not provide
            # enough positively aligned cells.  Fall back only for the missing
            # slots, preserving the old conservative workload criterion.
            self.alignment_fallbacks += 1
            fallback = np.argsort(-workload)
            for item in fallback:
                item = int(item)
                if capacity[item] > 0.15 and item not in chosen:
                    chosen.append(item)
                if len(chosen) >= required:
                    break
        if not chosen:
            return False
        mask = np.zeros(self.cells, dtype=bool)
        self.last_target_score = np.asarray(score, dtype=float).copy()
        self.last_target_alignment = np.asarray(alignment, dtype=float).copy()
        self.last_target_direction = np.asarray(direction, dtype=float).copy()
        mask[chosen] = True
        scale = clamp(
            1.0 - config.mechanism_conservation_loss_gain * tissue.mechanism_gain.loss_fraction,
            config.mechanism_conservation_scale_floor,
            config.mechanism_conservation_scale_ceiling,
        )
        self.target_mask = mask
        self.scale = float(scale)
        self.lease_until = float(age) + max(0.25, config.mechanism_conservation_duration)
        self.cooldown_until = self.lease_until + max(0.0, config.mechanism_conservation_cooldown)
        self.last_confirmation = int(tissue.mechanism_gain.confirmations)
        self.last_issue_age = float(age)
        self.issued += 1
        self.last_result = {
            'issued': True, 'age': float(age), 'scale': float(scale),
            'lease_until': float(self.lease_until), 'target_mask': mask.copy(),
            'loss_fraction': float(tissue.mechanism_gain.loss_fraction),
            'confirmation': int(self.last_confirmation),
            'target_alignment': self.last_target_alignment.copy(),
            'target_direction': self.last_target_direction.copy(),
            'alignment_fallbacks': int(self.alignment_fallbacks),
        }
        return True

    def budget_caps(self, tissue, config, age):
        if not self.active(age) or not config.mechanism_conservation_quiescence_enabled:
            return {}
        caps = {}
        for i in np.flatnonzero(self.target_mask):
            caps[str(tissue.tissue_ids[int(i)])] = (
                max(0.0, config.mechanism_conservation_atp_store_target * self.scale),
                max(0.0, config.mechanism_conservation_signal_store_target * self.scale),
            )
        return caps

    def release_excess_escrow(self, port, tissue, config, age):
        caps = self.budget_caps(tissue, config, age)
        if not caps:
            return {'atp': 0.0, 'signal': 0.0}
        total_atp = 0.0
        total_signal = 0.0
        for tissue_id, cap in caps.items():
            status = port.attachment_status(tissue_id)
            stores = np.asarray(status['stores'], dtype=float)
            amounts = {
                'atp': max(0.0, float(stores[p0.BUDGET_ATP]) - float(cap[0])),
                'signal': max(0.0, float(stores[p0.BUDGET_SIGNAL]) - float(cap[1])),
            }
            if amounts['atp'] > 1e-15 or amounts['signal'] > 1e-15:
                returned = port.return_unused_budget(tissue_id, amounts)
                total_atp += float(returned.get('atp', 0.0))
                total_signal += float(returned.get('signal', 0.0))
        self.cumulative_returned_atp += total_atp
        self.cumulative_returned_signal += total_signal
        return {'atp': total_atp, 'signal': total_signal}

    def update(self, age, dt, mechanism_confirmed=True):
        was_active = self.active(age - dt)
        if was_active:
            self.cumulative_active_time += max(0.0, float(dt))
        if not mechanism_confirmed and self.active(age):
            self.lease_until = float(age)
            self.revoked += 1
            self.last_result = dict(self.last_result or {}, revoked=True, revoke_age=float(age))
        if was_active and not self.active(age):
            self.expired += 1

    def finite(self):
        return bool(
            finite_array(self.target_mask)
            and finite_array(self.last_target_score)
            and finite_array(self.last_target_alignment)
            and finite_array(self.last_target_direction)
            and all(np.isfinite(float(v)) for v in (
                self.scale, self.lease_until, self.cooldown_until,
                self.last_issue_age, self.cumulative_active_time,
                self.cumulative_returned_atp, self.cumulative_returned_signal,
                self.cumulative_suppressed_activity,
            ))
        )

    def state_dict(self):
        return _copy_array_state(self)

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('cells', p2.P2_CELL_COUNT))
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=getattr(obj, key).dtype).copy()
            setattr(obj, key, value)
        return obj


class _MechanismQuiescencePort(object):
    """Narrow port proxy that caps refill of targeted neural escrow.

    The proxy never exposes the host body and never refunds spent energy.  It
    only limits *future* ATP/signal grants to the cap established by a verified
    mechanism-conservation lease; all other body-port operations are forwarded
    unchanged.
    """

    def __init__(self, base_port, cap_by_tissue):
        self._base_port = base_port
        self._cap_by_tissue = dict(cap_by_tissue)

    def __getattr__(self, name):
        return getattr(self._base_port, name)

    def allocate_budget(self, tissue_id, requests, dt):
        requests = dict(requests)
        cap = self._cap_by_tissue.get(str(tissue_id))
        if cap is not None:
            status = self._base_port.attachment_status(tissue_id)
            stores = np.asarray(status['stores'], dtype=float)
            atp_cap = max(0.0, float(cap[0]))
            signal_cap = max(0.0, float(cap[1]))
            # Raw material committed in this same call needs assembly ATP in
            # addition to the reusable neural ATP buffer.  Grant exactly that
            # physical need, not P2's historical fixed development chunk.
            assembly_atp = (
                p0.ASSEMBLY_ATP_PER_PROTEIN * max(0.0, float(requests.get('protein', 0.0)))
                + p0.ASSEMBLY_ATP_PER_MEMBRANE * max(0.0, float(requests.get('membrane', 0.0)))
            )
            # Signal requests are ambiguous at the frozen P2 boundary: during
            # development they are assembled, while maintenance uses the same
            # field as a reusable buffer top-up.  The bounded ATP store already
            # pays genuine signal assembly, so adding signal here would create a
            # recurring over-grant during maintenance.
            requests['atp'] = min(
                max(0.0, float(requests.get('atp', 0.0))),
                max(0.0, atp_cap - float(stores[p0.BUDGET_ATP])) + assembly_atp,
            )
            requests['signal'] = min(
                max(0.0, float(requests.get('signal', 0.0))),
                max(0.0, signal_cap - float(stores[p0.BUDGET_SIGNAL])),
            )
        return self._base_port.allocate_budget(tissue_id, requests, dt)


class MechanismAssistLease(object):
    """Counterbalanced paid assist trial followed by a short revocable lease."""

    def __init__(self):
        self.trial_active = False
        self.phase_order = []
        self.phase_index = 0
        self.phase_timer = 0.0
        self.washout_timer = 0.0
        self.phase_start_debt = np.zeros(f06.FORMAL_CHANNELS, dtype=float)
        self.phase_start_age = 0.0
        self.phase_uptake = 0.0
        self.phase_atp = 0.0
        self.phase_signal = 0.0
        self.phase_force = 0.0
        self.phase_concentration = 0.0
        self.phase_samples = 0
        self.phase_requested_samples = 0
        self.phase_executed_samples = 0
        self.records = []
        self.trial_strength = 0.0
        self.trial_direction = np.zeros(2, dtype=float)
        self.lease_strength = 0.0
        self.lease_until = 0.0
        self.cooldown_until = 0.0
        self.last_effect = 0.0
        self.last_quality = 0.0
        self.last_debt_effect = np.zeros(f06.FORMAL_CHANNELS, dtype=float)
        self.last_uptake_effect = 0.0
        self.last_result = None
        self.trials = 0
        self.evidence_accepted = 0
        self.accepted = 0
        self.rejected = 0
        self.last_tested_confirmation = 0
        self.trials_this_confirmation = 0
        self.cumulative_atp = 0.0
        self.cumulative_signal = 0.0
        self.cumulative_force = 0.0
        self.cumulative_uptake = 0.0

    def safe(self, debt, config):
        debt = np.asarray(debt, dtype=float)
        return bool(
            debt[0] < config.mechanism_assist_safe_energy_debt
            and debt[1] < config.mechanism_assist_safe_structural_debt
            and debt[2] < config.mechanism_assist_safe_structural_debt
        )

    def start_trial(self, age, debt, loss_fraction, direction, rng, config, confirmation_index=0):
        if self.trial_active or float(age) < self.cooldown_until:
            return False
        confirmation_index = int(confirmation_index)
        if confirmation_index <= 0:
            return False
        if confirmation_index != int(self.last_tested_confirmation):
            self.last_tested_confirmation = confirmation_index
            self.trials_this_confirmation = 0
        if self.trials_this_confirmation >= int(config.mechanism_assist_max_trials_per_confirmation):
            return False
        if not self.safe(debt, config):
            return False
        direction = _unit(direction)
        if float(np.linalg.norm(direction)) <= 1e-10:
            return False
        self.trial_active = True
        self.phase_order = [False, True, True, False]
        if int(rng.integers(0, 2)):
            self.phase_order = [True, False, False, True]
        self.phase_index = 0
        self.phase_timer = max(0.25, config.mechanism_assist_trial_window)
        self.washout_timer = 0.0
        self.records = []
        self.trial_direction = direction.copy()
        self.trial_strength = clamp(
            max(config.mechanism_assist_trial_strength, 0.45 * loss_fraction),
            0.04, config.mechanism_assist_max_strength,
        )
        self._reset_phase(age, debt)
        self.trials += 1
        self.trials_this_confirmation += 1
        return True

    def _reset_phase(self, age, debt):
        self.phase_start_age = float(age)
        self.phase_start_debt = np.asarray(debt, dtype=float).copy()
        self.phase_uptake = self.phase_atp = self.phase_signal = 0.0
        self.phase_force = self.phase_concentration = 0.0
        self.phase_samples = 0
        self.phase_requested_samples = 0
        self.phase_executed_samples = 0

    def trial_on(self):
        return bool(
            self.trial_active and self.washout_timer <= 0.0
            and self.phase_index < len(self.phase_order)
            and self.phase_order[self.phase_index]
        )

    def active_direction(self, fallback):
        # A lease is valid only for the direction actually tested by the
        # counterbalanced assay.  Following a newly fluctuating network output
        # after the trial would apply evidence from one intervention to another.
        if self.trial_active or self.lease_strength > 1e-8:
            return self.trial_direction.copy()
        return _unit(fallback)

    def active_strength(self, age, loss_fraction, config):
        if self.trial_on():
            return self.trial_strength
        if float(age) < self.lease_until and self.lease_strength > 1e-8:
            return clamp(
                config.mechanism_assist_max_strength * self.lease_strength
                * clamp(loss_fraction / max(config.mechanism_probe_loss_trigger, 1e-6), 0.0, 1.0),
                0.0, config.mechanism_assist_max_strength,
            )
        return 0.0

    def record_step(self, report, debt, uptake, concentration, dt, age, config):
        report = report or {}
        atp = max(0.0, float(report.get('atp_spent', 0.0)))
        signal = max(0.0, float(report.get('signal_spent', 0.0)))
        force = max(0.0, float(report.get('motor_force', 0.0)))
        self.cumulative_atp += atp
        self.cumulative_signal += signal
        self.cumulative_force += force
        self.cumulative_uptake += max(0.0, float(uptake)) * float(dt)
        if not self.trial_active:
            return None
        if self.washout_timer > 0.0:
            self.washout_timer -= float(dt)
            if self.washout_timer <= 0.0:
                self._reset_phase(age, debt)
            return None
        self.phase_uptake += max(0.0, float(uptake)) * float(dt)
        self.phase_atp += atp
        self.phase_signal += signal
        self.phase_force += force
        self.phase_concentration += max(0.0, float(concentration))
        self.phase_samples += 1
        if bool(self.phase_order[self.phase_index]):
            self.phase_requested_samples += 1
            if force > 1e-12 and (atp > 1e-12 or signal > 1e-12):
                self.phase_executed_samples += 1
        self.phase_timer -= float(dt)
        if self.phase_timer > 0.0:
            return None
        duration = max(1e-6, float(age) - self.phase_start_age)
        debt = np.asarray(debt, dtype=float)
        self.records.append({
            'on': bool(self.phase_order[self.phase_index]),
            'duration': duration,
            'debt_improvement': (self.phase_start_debt - debt) / duration,
            'uptake_rate': self.phase_uptake / duration,
            'atp_rate': self.phase_atp / duration,
            'signal_rate': self.phase_signal / duration,
            'force_rate': self.phase_force / duration,
            'concentration': self.phase_concentration / max(self.phase_samples, 1),
            'requested_samples': int(self.phase_requested_samples),
            'executed_samples': int(self.phase_executed_samples),
            'execution_fraction': (
                float(self.phase_executed_samples) / max(self.phase_requested_samples, 1)
                if bool(self.phase_order[self.phase_index]) else 1.0
            ),
            'start_debt': self.phase_start_debt.copy(),
        })
        self.phase_index += 1
        if self.phase_index >= len(self.phase_order):
            return self._finish_trial(age, config)
        self.phase_timer = max(0.25, config.mechanism_assist_trial_window)
        self.washout_timer = max(0.0, config.mechanism_assist_washout)
        if self.washout_timer <= 0.0:
            self._reset_phase(age, debt)
        return None

    def _finish_trial(self, age, config):
        on = [record for record in self.records if record['on']]
        off = [record for record in self.records if not record['on']]
        self.trial_active = False
        if not on or not off:
            self.rejected += 1
            self.cooldown_until = float(age) + config.mechanism_assist_retest_cooldown
            return None
        median = lambda values: float(np.median(np.asarray(values, dtype=float)))
        on_debt = np.median(np.asarray([r['debt_improvement'] for r in on]), axis=0)
        off_debt = np.median(np.asarray([r['debt_improvement'] for r in off]), axis=0)
        debt_effect = np.asarray(on_debt - off_debt, dtype=float)
        uptake_effect = median([r['uptake_rate'] for r in on]) - median([r['uptake_rate'] for r in off])
        on_context = np.mean(np.asarray([np.r_[r['start_debt'], r['concentration']] for r in on]), axis=0)
        off_context = np.mean(np.asarray([np.r_[r['start_debt'], r['concentration']] for r in off]), axis=0)
        context_distance = float(np.linalg.norm(on_context - off_context))
        quality = clamp(math.exp(-2.6 * context_distance), 0.0, 1.0)
        # Every positive result must correspond to a physically executed, paid
        # ON intervention.  Otherwise ordinary temporal drift can masquerade as
        # a useful lease trial (an earlier prototype accepted such a null ON).
        on_execution = median([r.get('execution_fraction', 0.0) for r in on])
        on_force_rate = median([r['force_rate'] for r in on])
        on_atp_rate = median([r['atp_rate'] for r in on])
        on_signal_rate = median([r['signal_rate'] for r in on])
        execution_quality = min(
            clamp(on_execution / max(config.mechanism_assist_min_execution_fraction, 1e-9), 0.0, 1.0),
            clamp(on_force_rate / max(config.mechanism_assist_min_force_rate, 1e-12), 0.0, 1.0),
            clamp(on_atp_rate / max(config.mechanism_assist_min_atp_rate, 1e-12), 0.0, 1.0),
            clamp(on_signal_rate / max(config.mechanism_assist_min_signal_rate, 1e-12), 0.0, 1.0),
        )
        quality *= execution_quality

        off_uptake_median = median([r['uptake_rate'] for r in off])
        uptake_replication = float(np.mean([
            r['uptake_rate'] > off_uptake_median for r in on
        ]))
        channel_replication = [
            float(np.mean([
                r['debt_improvement'][k]
                > median([o['debt_improvement'][k] for o in off])
                for r in on
            ]))
            for k in range(f06.FORMAL_CHANNELS)
        ]
        replication_fraction = max([uptake_replication] + channel_replication)

        # Independent body channels keep veto power, but their acceptable
        # transient scales differ.  Fatigue is rapidly reversible; membrane and
        # damage channels are not.  A resource gain may therefore pay a small
        # fatigue pulse, but never hide structural injury.
        energy_harm = max(0.0, -float(debt_effect[0]))
        boundary_harm = max(0.0, -float(debt_effect[1]))
        damage_harm = max(0.0, -float(debt_effect[2]))
        fatigue_harm = max(0.0, -float(debt_effect[3]))
        structural_harm = max(boundary_harm, damage_harm)
        total_harm = max(energy_harm, structural_harm, fatigue_harm)
        positive_debt = max(0.0, float(np.max(debt_effect[:3])))
        useful = bool(
            uptake_effect >= config.mechanism_assist_min_uptake_effect
            or positive_debt >= config.mechanism_assist_min_debt_effect
        )
        executed = bool(
            on_execution >= config.mechanism_assist_min_execution_fraction
            and on_force_rate >= config.mechanism_assist_min_force_rate
            and on_atp_rate >= config.mechanism_assist_min_atp_rate
            and on_signal_rate >= config.mechanism_assist_min_signal_rate
        )
        replicated = bool(
            replication_fraction >= config.mechanism_assist_min_replication_fraction
        )
        channel_safe = bool(
            energy_harm <= config.mechanism_assist_max_energy_harm
            and structural_harm <= config.mechanism_assist_max_structural_harm
            and fatigue_harm <= config.mechanism_assist_max_fatigue_harm
        )
        evidence_accepted = bool(
            quality >= config.mechanism_assist_min_quality
            and executed and replicated and useful and channel_safe
        )
        feedback_applied = bool(
            evidence_accepted and config.mechanism_assist_feedback_enabled
        )
        harm_score = max(
            energy_harm / max(config.mechanism_assist_max_energy_harm, 1e-9),
            structural_harm / max(config.mechanism_assist_max_structural_harm, 1e-9),
            fatigue_harm / max(config.mechanism_assist_max_fatigue_harm, 1e-9),
        )
        uptake_score = clamp(
            uptake_effect / max(4.0 * config.mechanism_assist_min_uptake_effect, 1e-9),
            -1.0, 2.0,
        )
        debt_score = clamp(
            positive_debt / max(4.0 * config.mechanism_assist_min_debt_effect, 1e-9),
            0.0, 2.0,
        )
        normalised = uptake_score + debt_score - harm_score
        self.last_effect = float(normalised)
        self.last_quality = float(quality)
        self.last_debt_effect = debt_effect.copy()
        self.last_uptake_effect = float(uptake_effect)
        if evidence_accepted:
            self.evidence_accepted += 1
        if feedback_applied:
            self.accepted += 1
            # Keep the lease materially modest.  Strength is bounded by clean
            # execution, context matching and replicated benefit rather than a
            # potentially huge raw debt ratio.
            self.lease_strength = clamp(
                0.12 + 0.26 * quality + 0.18 * max(0.0, normalised),
                0.12, 0.62,
            )
            self.lease_until = float(age) + config.mechanism_assist_lease_duration
        elif evidence_accepted:
            # Cost-matched no-feedback controls run the same physical trial but
            # intentionally withhold the material lease.  This is neither a
            # failed assay nor a free intervention.
            self.lease_strength = 0.0
            self.lease_until = float(age)
        else:
            self.rejected += 1
            self.lease_strength = 0.0
            self.lease_until = float(age)
        if evidence_accepted:
            self.cooldown_until = (
                float(age) + config.mechanism_assist_lease_duration
                + config.mechanism_assist_accept_retest_delay
            )
        else:
            self.cooldown_until = float(age) + max(
                config.mechanism_assist_retest_cooldown,
                config.mechanism_assist_rejection_backoff,
            )
        self.last_result = {
            'accepted': bool(feedback_applied), 'evidence_accepted': bool(evidence_accepted),
            'feedback_applied': bool(feedback_applied), 'quality': float(quality),
            'uptake_effect': float(uptake_effect),
            'debt_effect': debt_effect.copy(),
            'structural_harm': float(structural_harm),
            'energy_harm': float(energy_harm),
            'fatigue_harm': float(fatigue_harm),
            'channel_safe': bool(channel_safe),
            'execution_fraction': float(on_execution),
            'execution_quality': float(execution_quality),
            'force_rate': float(on_force_rate),
            'atp_rate': float(on_atp_rate),
            'signal_rate': float(on_signal_rate),
            'replication_fraction': float(replication_fraction),
            'executed': bool(executed),
            'replicated': bool(replicated),
            'normalised_effect': float(normalised),
            'lease_strength': float(self.lease_strength),
            'lease_until': float(self.lease_until),
        }
        return self.last_result

    def update_lease(self, age, dt, config):
        if float(age) >= self.lease_until and self.lease_strength > 0.0:
            alpha = 1.0 - math.exp(-float(dt) / max(config.mechanism_assist_lease_recovery_tau, 1e-6))
            self.lease_strength += alpha * (0.0 - self.lease_strength)
            if self.lease_strength < 1e-4:
                self.lease_strength = 0.0

    def finite(self):
        return bool(
            finite_array(self.phase_start_debt)
            and finite_array(self.last_debt_effect)
            and finite_array(self.trial_direction)
            and all(np.isfinite(float(v)) for v in (
                self.phase_timer, self.washout_timer, self.phase_start_age,
                self.phase_uptake, self.phase_atp, self.phase_signal,
                self.phase_force, self.phase_concentration, self.trial_strength,
                self.lease_strength, self.lease_until, self.cooldown_until,
                self.last_effect, self.last_quality, self.last_uptake_effect,
                self.cumulative_atp, self.cumulative_signal,
                self.cumulative_force, self.cumulative_uptake,
            ))
        )

    def state_dict(self):
        state = _copy_array_state(self)
        state['records'] = []
        for record in self.records:
            item = dict(record)
            item['debt_improvement'] = np.asarray(item['debt_improvement'], dtype=float).copy()
            item['start_debt'] = np.asarray(item['start_debt'], dtype=float).copy()
            state['records'].append(item)
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=float).copy()
            setattr(obj, key, value)
        obj.records = []
        for record in state.get('records', []):
            item = dict(record)
            item['debt_improvement'] = np.asarray(item['debt_improvement'], dtype=float).copy()
            item['start_debt'] = np.asarray(item['start_debt'], dtype=float).copy()
            obj.records.append(item)
        return obj


class BoundedDiagnosticExposure(object):
    """Physically paid exposure chosen to resolve a specific evidence deficit."""

    def __init__(self):
        self.active = False
        self.phase = DIAGNOSTIC_IDLE
        self.ligand = -1
        self.channel = -1
        self.timer = 0.0
        self.total_timer = 0.0
        self.start_age = -1e9
        self.cooldown_until = 0.0
        self.start_debt = np.zeros(f06.FORMAL_CHANNELS, dtype=float)
        self.start_observation_weight = 0.0
        self.start_change_evidence = 0.0
        self.observe_start_weight = None
        self.random_direction = np.zeros(2, dtype=float)
        self.last_direction = np.zeros(2, dtype=float)
        self.gradient_samples = 0
        self.mode = DIAGNOSIS_ACTIVE
        self.started = 0
        self.completed = 0
        self.aborted = 0
        self.insufficient_gradient = 0
        self.last_reason = 'none'
        self.last_result = None
        self.cumulative_atp = 0.0
        self.cumulative_signal = 0.0
        self.cumulative_motor_force = 0.0
        self.cumulative_material_wear = 0.0
        self.last_effector_report = {}
        self.episode_ligand_exposure = 0.0
        self.episode_directional_samples = 0
        self.episode_exposure_seconds = 0.0
        self.episode_observation_seconds = 0.0
        self.cumulative_ligand_exposure = 0.0
        self.cumulative_directional_samples = 0
        self.cumulative_exposure_seconds = 0.0
        self.cumulative_observation_seconds = 0.0
        self.start_cumulative_atp = 0.0
        self.start_cumulative_signal = 0.0
        self.start_cumulative_motor_force = 0.0
        self.last_evidence_latency = -1.0
        self.cumulative_evidence_latency = 0.0
        self.evidence_latency_count = 0

    def start(self, ligand, channel, debt, age, config, rng):
        if self.active or float(age) < self.cooldown_until:
            return False
        self.active = True
        self.phase = DIAGNOSTIC_EXPOSE
        self.ligand = int(ligand)
        self.channel = int(channel)
        self.timer = max(0.25, float(config.diagnosis_exposure_duration))
        self.total_timer = 0.0
        self.start_age = float(age)
        self.start_debt = np.asarray(debt, dtype=float).copy()
        self.start_observation_weight = 0.0
        self.start_change_evidence = 0.0
        self.observe_start_weight = None
        self.mode = str(config.diagnosis_mode)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        self.random_direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        self.last_direction[:] = 0.0
        self.gradient_samples = 0
        self.episode_ligand_exposure = 0.0
        self.episode_directional_samples = 0
        self.episode_exposure_seconds = 0.0
        self.episode_observation_seconds = 0.0
        self.start_cumulative_atp = float(self.cumulative_atp)
        self.start_cumulative_signal = float(self.cumulative_signal)
        self.start_cumulative_motor_force = float(self.cumulative_motor_force)
        self.started += 1
        self.last_reason = 'started'
        self.last_result = None
        return True

    def safe(self, debt, config):
        debt = np.asarray(debt, dtype=float)
        structural_delta = float(np.max(debt[1:3] - self.start_debt[1:3]))
        return bool(
            debt[0] < config.diagnosis_safe_energy_debt
            and debt[1] < config.diagnosis_safe_boundary_debt
            and debt[2] < config.diagnosis_safe_damage_debt
            and structural_delta < config.diagnosis_abort_debt_delta
        )

    def command(self, frame, debt, dt, age, config):
        if not self.active:
            return None
        self.total_timer += dt
        if self.total_timer >= config.diagnosis_max_total_duration:
            self.abort(age, config, 'timeout')
            return None
        if not self.safe(debt, config):
            self.abort(age, config, 'safety')
            return None
        if self.phase != DIAGNOSTIC_EXPOSE:
            return None
        profiles = np.asarray(frame['external']['ligand_profiles'], dtype=float)
        if self.ligand < 0 or self.ligand >= profiles.shape[0]:
            self.abort(age, config, 'missing-ligand')
            return None
        profile = np.maximum(0.0, profiles[self.ligand])
        # Keep the exposure threshold in the same bounded concentration units
        # used by the evidence target selector and the audit-start gate.  The
        # previous raw-mean comparison rejected weak-but-directional gradients
        # even when the membrane profile contained a clear spatial signal.
        concentration = math.tanh(2.5 * float(np.mean(profile)))
        gradient = np.sum(profile[:, None] * p2.MEMBRANE_NORMALS, axis=0)
        physical_direction = _unit(gradient)
        direction = physical_direction.copy()
        if self.mode == DIAGNOSIS_RANDOM:
            direction = self.random_direction.copy()
        elif self.mode in (DIAGNOSIS_PASSIVE, DIAGNOSIS_OFF):
            direction[:] = 0.0
        self.episode_ligand_exposure += max(0.0, concentration) * dt
        self.cumulative_ligand_exposure += max(0.0, concentration) * dt
        self.episode_exposure_seconds += dt
        self.cumulative_exposure_seconds += dt
        # Passive observation is a real diagnostic arm, not a stalled active
        # episode. Detectability is determined by the physical membrane
        # gradient before intervention direction is zeroed or randomised.
        sensed = bool(
            concentration >= config.diagnosis_min_ligand_concentration
            and float(np.linalg.norm(physical_direction)) >= 1e-8
        )
        if sensed:
            self.last_direction = physical_direction.copy()
            self.gradient_samples += 1
            self.episode_directional_samples += 1
            self.cumulative_directional_samples += 1
        elif self.mode == DIAGNOSIS_ACTIVE and float(np.linalg.norm(self.last_direction)) >= 1e-8:
            # A local membrane profile can briefly drop as the cell crosses a
            # sparse particle ring.  Continue the already-committed bounded
            # exposure along the last physically observed direction instead
            # of stalling until the global timeout.
            direction = self.last_direction.copy()
            self.insufficient_gradient += 1
        elif self.mode == DIAGNOSIS_RANDOM:
            direction = self.random_direction.copy()
        else:
            self.insufficient_gradient += 1
            return None
        self.timer -= dt
        if self.timer <= 0.0:
            self.phase = DIAGNOSTIC_OBSERVE
            self.timer = max(0.5, float(config.diagnosis_observation_duration))
        return {
            'motor': direction * clamp(config.diagnosis_motor_strength, 0.0, 1.0),
            'transporter_polarity': direction * clamp(config.diagnosis_transporter_strength, 0.0, 1.0),
            'direction': direction,
            'concentration': concentration,
        }

    def observe(self, dt, age, ledger, config):
        if not self.active:
            return None
        if self.phase == DIAGNOSTIC_OBSERVE:
            self.episode_observation_seconds += dt
            self.cumulative_observation_seconds += dt
            if self.observe_start_weight is None:
                self.observe_start_weight = float(
                    ledger.observation_weight[self.ligand, self.channel]
                )
                return None
            self.timer -= dt
            if self.ligand >= 0 and self.channel >= 0:
                current_weight = float(ledger.observation_weight[self.ligand, self.channel])
                if current_weight > self.observe_start_weight + 0.20 or self.timer <= 0.0:
                    return self.finish(age, ledger, config)
        return None

    def finish(self, age, ledger, config):
        evidence = ledger.diagnostics_for(self.ligand, self.channel, age, config)
        latency = max(0.0, float(age) - float(self.start_age))
        result = {
            'status': 'completed',
            'ligand': int(self.ligand),
            'channel': int(self.channel),
            'age': float(age),
            'latency': latency,
            'evidence': evidence,
            'evidence_gain': float(evidence.get('observation_weight', 0.0) - self.start_observation_weight),
            'episode_ligand_exposure': float(self.episode_ligand_exposure),
            'episode_directional_samples': int(self.episode_directional_samples),
            'episode_exposure_seconds': float(self.episode_exposure_seconds),
            'episode_observation_seconds': float(self.episode_observation_seconds),
            'episode_atp': float(self.cumulative_atp - self.start_cumulative_atp),
            'episode_signal': float(self.cumulative_signal - self.start_cumulative_signal),
            'episode_motor_force': float(self.cumulative_motor_force - self.start_cumulative_motor_force),
        }
        self.last_evidence_latency = latency
        self.cumulative_evidence_latency += latency
        self.evidence_latency_count += 1
        self.active = False
        self.phase = DIAGNOSTIC_IDLE
        self.cooldown_until = float(age) + config.diagnosis_cooldown
        self.completed += 1
        self.last_reason = 'completed'
        self.last_result = result
        return result

    def abort(self, age, config, reason):
        self.active = False
        self.phase = DIAGNOSTIC_ABORTED
        self.cooldown_until = float(age) + config.diagnosis_cooldown
        self.aborted += 1
        self.last_reason = str(reason)
        self.last_result = {
            'status': 'aborted', 'reason': str(reason), 'age': float(age),
            'latency': max(0.0, float(age) - float(self.start_age)),
            'episode_ligand_exposure': float(self.episode_ligand_exposure),
            'episode_atp': float(self.cumulative_atp - self.start_cumulative_atp),
        }

    def pause_for_audit(self, age, config):
        """Stop diagnostic locomotion while a reversible audit is measuring.

        The diagnostic remains alive as an evidence-acquisition episode, but
        its own motor/transporter intervention is withdrawn so it cannot
        contaminate the ABBA/BAAB comparison it just made possible.
        """
        if not self.active:
            return
        self.phase = DIAGNOSTIC_OBSERVE
        self.timer = max(self.timer, max(0.5, float(config.diagnosis_observation_duration)))
        self.last_reason = 'targeted-audit-started'

    def record_report(self, report):
        if not report:
            return
        self.last_effector_report = dict(report)
        self.cumulative_atp += float(report.get('atp_spent', 0.0))
        self.cumulative_signal += float(report.get('signal_spent', 0.0))
        self.cumulative_motor_force += float(report.get('motor_force', 0.0))

    def state_dict(self):
        state = _copy_array_state(self)
        if self.last_result is not None:
            state['last_result'] = dict(self.last_result)
        state['last_effector_report'] = dict(self.last_effector_report)
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for key, value in state.items():
            if isinstance(getattr(obj, key, None), np.ndarray):
                value = np.asarray(value, dtype=float).copy()
            setattr(obj, key, value)
        return obj


class TargetedDiagnosticCompiler(f06.SmallFalsificationCompiler):
    def __init__(self, enabled=True):
        super(TargetedDiagnosticCompiler, self).__init__(enabled=enabled)
        self.targeted_compiled = 0
        self.last_target_ligand = -1
        self.last_target_channel = -1

    def compile_semantic(self, tissue, frame, ligand, channel, config):
        if not self.enabled:
            return None
        ligand = int(ligand); channel = int(channel)
        profiles = np.asarray(frame['external']['ligand_profiles'], dtype=float)
        if ligand < 0 or ligand >= profiles.shape[0]:
            return None
        profile = np.maximum(0.0, profiles[ligand])
        total = float(np.sum(profile))
        if total <= 1e-12:
            return None
        gradient = _unit(np.sum(profile[:, None] * p2.MEMBRANE_NORMALS, axis=0))
        directional = tissue.preferred.dot(gradient)
        concentration = math.tanh(2.5 * float(np.mean(profile)))
        feature = directional * concentration
        sensory = tissue.w_sensor[:, ligand] * feature
        expected_hidden = np.tanh(sensory + tissue.bias)
        expected_projection = np.tanh(2.6 * expected_hidden * tissue.motor_gain) * directional
        capacity = tissue._material_capacity()
        actual_projection = (
            np.tanh(2.6 * capacity * tissue.hidden * tissue.motor_gain)
            * directional
        )
        # Select cells that currently mediate the cue in either direction.
        # A reversal assay must be able to test both approach and avoidance;
        # restricting the claim to positive projection hid real pathways when
        # the learned sign differed across lineages.
        score = (
            0.55 * np.abs(actual_projection)
            + 0.30 * np.abs(expected_projection)
            + 0.15 * np.abs(sensory)
        ) * capacity
        order = np.argsort(-score)
        claim = np.zeros(p2.P2_CELL_COUNT, dtype=bool)
        chosen = [int(i) for i in order if capacity[int(i)] > 0.15][:config.audit_group_size]
        if len(chosen) < config.audit_group_size:
            chosen = [int(i) for i in np.argsort(-np.abs(sensory) * capacity)[:config.audit_group_size]]
        claim[chosen] = True
        sham = self._matched_sham(tissue, claim, config.audit_group_size)
        # The stop experiment tests the *current neural claim* made by the
        # selected cue-responsive cells.  The delayed semantic ledger decides
        # when this claim deserves re-testing, but its global cue outcome is
        # not the predicted KO effect of a particular cell group.  Using the
        # old cue baseline as the KO prediction conflated those two levels and
        # could label a genuinely contradicted neural claim as "supported".
        raw_prediction = float(
            np.mean(tissue.causal_estimate[claim, channel])
            - np.mean(tissue.causal_estimate[sham, channel])
        )
        semantic_baseline = float(tissue.evidence_ledger.baseline_mean[ligand, channel])
        if abs(raw_prediction) < 1.5e-4:
            # A weak current claim still gets a small, explicitly recorded
            # prior sign from the delayed physical relation.  This is only a
            # fallback; ordinary tests are driven by the neural claim itself.
            raw_prediction = math.copysign(
                2.2e-4,
                semantic_baseline if abs(semantic_baseline) >= 1.5e-4 else 1.0,
            )
        calibrated, reliability = tissue.calibrator.predict(channel, raw_prediction)
        program = {
            'family': 'active-semantic-mediation',
            'claim_mask': claim,
            'sham_mask': sham,
            'channel': channel,
            'target_ligand': ligand,
            'raw_prediction': raw_prediction,
            'semantic_baseline': semantic_baseline,
            'calibrated_prediction': calibrated,
            'reliability': reliability,
            'commands': [
                {'op': 'gate_cells', 'count': int(np.sum(claim)), 'strength': config.audit_knockout_strength},
            ],
            'complexity': 1,
            'evidence_status': tissue.evidence_ledger.status(ligand, channel, tissue._061_last_age, config),
            'change_evidence': float(tissue.evidence_ledger.change_evidence[ligand, channel]),
        }
        self.compiled += 1
        self.targeted_compiled += 1
        self.last_target_ligand = ligand
        self.last_target_channel = channel
        self.last_program = program
        return program

    def state_dict(self):
        state = super(TargetedDiagnosticCompiler, self).state_dict()
        state.update({
            'targeted_compiled': self.targeted_compiled,
            'last_target_ligand': self.last_target_ligand,
            'last_target_channel': self.last_target_channel,
        })
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls(state.get('enabled', True))
        obj.compiled = int(state.get('compiled', 0))
        obj.started = int(state.get('started', 0))
        obj.last_program = state.get('last_program')
        obj.targeted_compiled = int(state.get('targeted_compiled', 0))
        obj.last_target_ligand = int(state.get('last_target_ligand', -1))
        obj.last_target_channel = int(state.get('last_target_channel', -1))
        return obj


class ActiveDiagnosticMaterialTissue(f06.FormalMaterialTissue):
    def __init__(self, gene_parameters, mode=p2.P2_MODE_FULL, rng_seed=0, formal_enabled=False):
        super(ActiveDiagnosticMaterialTissue, self).__init__(gene_parameters, mode=mode, rng_seed=rng_seed, formal_enabled=formal_enabled)
        self._init_061_runtime()

    def _init_061_runtime(self):
        self.evidence_ledger = EvidenceSufficiencyLedger()
        self.diagnostic = BoundedDiagnosticExposure()
        self.compiler = TargetedDiagnosticCompiler(True)
        self.feedback_replication = FeedbackReplicationGate()
        self.mechanism_gain = MechanismGainLedger()
        self.mechanism_probe = PaidMechanismProbe()
        self.mechanism_conservation = MechanismConservationLease()
        self.mechanism_assist = MechanismAssistLease()
        self.pending_targeted_audit = None
        self.pending_audit_visible_since = -1e9
        self.last_diagnosis_result = None
        self.diagnosis_attempts = 0
        self.diagnosis_targeted_audits = 0
        self.diagnosis_feedback_events = 0
        self.diagnosis_planning_atp = 0.0
        self.diagnosis_material_wear = 0.0
        self.diagnosis_false_starts = 0
        self.diagnosis_evidence_refreshes = 0
        self.diagnosis_deferred_unsafe = 0
        self.diagnosis_deferred_no_target = 0
        self.diagnosis_deferred_low_value = 0
        self.diagnosis_deferred_busy = 0
        self._061_last_defer_age = -1e9
        self._061_last_defer_reason = 'none'
        self._diagnosis_learning_snapshot = None
        self._061_prev_cue_active = np.zeros(s4.LIGAND_COUNT, dtype=bool)
        self._061_prev_cue_start = np.zeros((s4.LIGAND_COUNT, f06.FORMAL_CHANNELS), dtype=float)
        self._061_prev_cue_peak = np.zeros((s4.LIGAND_COUNT, f06.FORMAL_CHANNELS), dtype=float)
        self._061_prev_cue_signal_peak = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self._061_episode_max_concentration = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self._061_episode_concentration_integral = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self._061_last_age = 0.0
        self._061_last_concentration = np.zeros(s4.LIGAND_COUNT, dtype=float)
        self._061_last_debt = np.zeros(f06.FORMAL_CHANNELS, dtype=float)
        self._061_last_frame = None
        self._061_last_diagnostic_command = None
        self._061_last_mechanism_probe_report = None
        self._061_last_mechanism_assist_report = None
        self._061_last_total_uptake = 0.0
        self.mechanism_probe_attempts = 0
        self.mechanism_probe_confirmed_events = 0
        self.mechanism_assist_commands = 0
        self.mechanism_assist_feedback_events = 0
        self.mechanism_conservation_feedback_events = 0
        self.mechanism_feedback_blocked_low_quality = 0
        self.mechanism_feedback_blocked_replication = 0
        self.neural_escrow_sanitation_calls = 0
        self.neural_escrow_returned_atp = 0.0
        self.neural_escrow_returned_signal = 0.0

    def _snapshot_diagnosis_learning(self):
        names = (
            'w_sensor', 'w_rec', 'bias', 'motor_gain', 'predict_w',
            'causal_estimate', 'maturity', 'reopen_reserve',
            'elig_bias', 'elig_sensor', 'elig_rec', 'elig_motor',
            'cue_eligibility', 'ligand_trace',
        )
        self._diagnosis_learning_snapshot = {name: getattr(self, name).copy() for name in names}

    def _restore_diagnosis_learning(self):
        if self._diagnosis_learning_snapshot is None:
            return
        for name, value in self._diagnosis_learning_snapshot.items():
            setattr(self, name, value.copy())
        self._diagnosis_learning_snapshot = None

    def _diagnosis_safe(self, debt, config):
        debt = np.asarray(debt, dtype=float)
        return bool(
            debt[0] < config.diagnosis_start_energy_debt
            and debt[1] < config.diagnosis_start_boundary_debt
            and debt[2] < config.diagnosis_start_damage_debt
        )

    def _current_concentrations(self, frame):
        profiles = np.asarray(frame['external']['ligand_profiles'], dtype=float)
        concentration = np.zeros(profiles.shape[0], dtype=float)
        for ligand in range(profiles.shape[0]):
            concentration[ligand] = math.tanh(2.5 * float(np.mean(np.maximum(0.0, profiles[ligand]))))
        return concentration

    def _start_targeted_audit(self, port, frame, debt, context, world, config):
        if self.pending_targeted_audit is None or self.auditor.active:
            return False
        ligand, channel = self.pending_targeted_audit
        concentrations = self._current_concentrations(frame)
        if ligand >= len(concentrations) or concentrations[ligand] < config.diagnosis_audit_min_concentration:
            self.pending_audit_visible_since = -1e9
            return False
        if self.pending_audit_visible_since < -1e8:
            self.pending_audit_visible_since = float(world.age)
            return False
        if float(world.age) - self.pending_audit_visible_since < config.diagnosis_audit_activation_delay:
            return False
        program = self.compiler.compile_semantic(self, frame, ligand, channel, config)
        if program is None:
            return False
        if self.auditor.start(
            program, debt, context, world.age, self.formal_rng, config,
            self.controller_atp_spent, self.controller_material_wear,
        ):
            self.compiler.started += 1
            self.formal_last_start_age = world.age
            self.diagnosis_targeted_audits += 1
            self.pending_targeted_audit = None
            self.pending_audit_visible_since = -1e9
            self.diagnostic.pause_for_audit(world.age, config)
            return True
        return False

    def _record_diagnosis_defer(self, reason, age, config):
        if float(age) - float(self._061_last_defer_age) < max(0.2, config.diagnosis_defer_log_interval):
            return
        self._061_last_defer_age = float(age)
        self._061_last_defer_reason = str(reason)
        if reason == 'unsafe':
            self.diagnosis_deferred_unsafe += 1
        elif reason == 'no-target':
            self.diagnosis_deferred_no_target += 1
        elif reason == 'low-value':
            self.diagnosis_deferred_low_value += 1
        else:
            self.diagnosis_deferred_busy += 1

    def _maybe_start_diagnosis(self, frame, debt, world, config):
        if not config.diagnosis_enabled or config.diagnosis_mode in (DIAGNOSIS_PASSIVE, DIAGNOSIS_OFF):
            return False
        if self.diagnostic.active or self.auditor.active:
            self._record_diagnosis_defer('busy', world.age, config)
            return False
        if self.active_age < config.diagnosis_min_active_age or world.age < self.diagnostic.cooldown_until:
            return False
        if not self._diagnosis_safe(debt, config):
            self._record_diagnosis_defer('unsafe', world.age, config)
            return False
        concentrations = self._current_concentrations(frame)
        best = self.evidence_ledger.best_target(concentrations, debt, world.age, config)
        if best is None:
            self._record_diagnosis_defer('no-target', world.age, config)
            return False
        score, ligand, channel, deficit, change = best
        observation = self.evidence_ledger.observation_sufficiency(ligand, channel, world.age, config)
        status = self.evidence_ledger.status(ligand, channel, world.age, config)
        semantic_hint = max(
            float(change),
            float(self.change_sentinel.semantic_channel_probability[channel]),
        )
        stale_age = float(world.age) - float(self.evidence_ledger.last_observation_age[ligand, channel])
        baseline_ready = bool(
            observation >= 0.42
            and status in (EVIDENCE_STABLE, EVIDENCE_SUSPECTED, EVIDENCE_CONFIRMED)
        )
        stale_supported = bool(
            status == EVIDENCE_STABLE
            and stale_age >= config.diagnosis_stale_refresh_multiplier * config.diagnosis_stale_age
        )
        # Locomotor diagnosis must not invent a relation from an uncalibrated
        # cue. First establish the ordinary relation by passive experience;
        # active exposure is reserved for stale or contradicted evidence.
        if config.diagnosis_require_supported_baseline and not baseline_ready:
            self._record_diagnosis_defer('no-target', world.age, config)
            return False
        if semantic_hint < config.diagnosis_min_suspicion and not stale_supported:
            self._record_diagnosis_defer('low-value', world.age, config)
            return False
        if deficit < config.diagnosis_min_deficit and semantic_hint < config.diagnosis_change_threshold:
            self._record_diagnosis_defer('low-value', world.age, config)
            return False
        score *= (0.35 + 0.65 * max(semantic_hint, 0.35 if stale_supported else 0.0))
        if score < 0.045:
            self._record_diagnosis_defer('low-value', world.age, config)
            return False
        if self.diagnostic.start(ligand, channel, debt, world.age, config, self.formal_rng):
            self.diagnostic.start_observation_weight = float(self.evidence_ledger.observation_weight[ligand, channel])
            self.diagnostic.start_change_evidence = float(self.evidence_ledger.change_evidence[ligand, channel])
            self.diagnosis_attempts += 1
            return True
        return False

    def _action_direction(self):
        action = np.asarray(self.last_action, dtype=float).reshape(2)
        if float(np.linalg.norm(action)) < 1e-10:
            # A probe may be required exactly when the old controller is weak.
            # Fall back to the direction represented by the currently active
            # cells, not to an external target coordinate.
            weighted = np.sum(
                self.preferred * (np.abs(self.hidden * self.motor_gain))[:, None],
                axis=0,
            )
            action = weighted
        return _unit(action)

    def _observe_ordinary_mechanism(self, world, config):
        report = getattr(self, '_061_last_base_effector_report', None)
        if not report:
            return None
        command_norm = float(np.linalg.norm(np.asarray(self.last_action, dtype=float)))
        force = max(0.0, float(report.get('motor_force', 0.0)))
        if command_norm < config.mechanism_probe_min_action or force <= 0.0:
            return None
        gain = force / max(command_norm, 0.04)
        capacity = clamp(float(np.mean(self._material_capacity())), 0.0, 1.0)
        payment = clamp(
            min(
                float(report.get('atp_spent', 0.0)) / 2.0e-7,
                float(report.get('signal_spent', 0.0)) / 2.0e-8,
            ), 0.0, 1.0,
        )
        quality = clamp(0.35 + 0.35 * capacity + 0.30 * payment, 0.0, 1.0)
        return self.mechanism_gain.observe(
            gain, quality, world.age, config, probe=False,
            mechanism_probability=self.change_sentinel.mechanism_probability,
        )

    def _mechanism_safe(self, debt, config):
        debt = np.asarray(debt, dtype=float)
        structurally_safe = bool(
            debt[1] < config.mechanism_assist_safe_structural_debt
            and debt[2] < config.mechanism_assist_safe_structural_debt
        )
        joint_crisis = bool(
            debt[0] >= config.mechanism_probe_crisis_energy_debt
            and debt[3] >= config.mechanism_probe_crisis_process_debt
        )
        return bool(structurally_safe and not joint_crisis)

    def _pay_mechanism_planning_cost(self, port, dt, config, active):
        scale = 1.0 if active else 0.22
        atp = max(0.0, dt * config.mechanism_probe_atp_planning_rate * scale)
        wear = max(0.0, dt * config.mechanism_probe_wear_rate * scale)
        if atp <= 0.0 and wear <= 0.0:
            return
        port.allocate_budget(
            f06.FORMAL_CONTROLLER_ID,
            {'atp': atp, 'protein': wear, 'membrane': 0.0, 'signal': 0.0},
            dt,
        )
        status = self._controller_status(port)
        stores = np.asarray(status['stores'], dtype=float)
        built = port.commit_material(
            f06.FORMAL_CONTROLLER_ID,
            protein=min(wear, float(stores[p0.BUDGET_PROTEIN])),
            damaged_fraction=0.75, aggregate_fraction=0.12,
        )
        self.diagnosis_planning_atp += float(built.get('atp_spent', 0.0))
        self.diagnosis_material_wear += float(
            built.get('damaged_protein', 0.0) + built.get('aggregate', 0.0)
        )

    def _maybe_start_mechanism_probe(self, debt, world, config):
        if (
            not config.mechanism_probe_enabled
            or self.mechanism_probe.active
            or self.mechanism_assist.trial_active
            or self.auditor.active
            or self.diagnostic.active
            or self.active_age < config.mechanism_probe_min_active_age
            or float(world.age) < self.mechanism_probe.cooldown_until
            or not self._mechanism_safe(debt, config)
        ):
            return False
        suspicion = self.mechanism_gain.suspicion(
            config, self.change_sentinel.mechanism_probability,
        )
        if suspicion < 0.20:
            return False
        direction = self._action_direction()
        if float(np.linalg.norm(direction)) <= 1e-10:
            return False
        if self.mechanism_probe.start(direction, world.age, config):
            self.mechanism_probe_attempts += 1
            return True
        return False

    def _maybe_issue_mechanism_conservation(self, debt, world, config):
        if (
            self.auditor.active or self.diagnostic.active
            or self.mechanism_probe.active or self.mechanism_assist.trial_active
        ):
            return False
        issued = self.mechanism_conservation.maybe_issue(
            self, world.age, debt, config,
        )
        if issued:
            self.mechanism_conservation_feedback_events += 1
        return issued

    def _maybe_start_assist_trial(self, debt, world, config):
        if (
            not config.mechanism_assist_enabled
            or not self.mechanism_gain.confirmed
            or self.mechanism_assist.trial_active
            or self.auditor.active
            or self.diagnostic.active
            or self.mechanism_probe.active
            or float(world.age) < self.mechanism_assist.cooldown_until
            or not self._mechanism_safe(debt, config)
        ):
            return False
        if float(world.age) < self.mechanism_assist.lease_until:
            return False
        direction = self._action_direction()
        return self.mechanism_assist.start_trial(
            world.age, debt, self.mechanism_gain.loss_fraction, direction,
            self.formal_rng, config,
            confirmation_index=self.mechanism_gain.confirmations,
        )

    def _apply_feedback(self, result, config):
        # Weak observations may calibrate the model, but materially changing a
        # neural lease requires a clean or sign-consistent replicated result.
        if result.get('status') == 'contradicted':
            allowed = self.feedback_replication.allow(result, self._061_last_age, config)
            if not allowed:
                reason = (self.feedback_replication.last_result or {}).get('reason')
                if reason == 'quality':
                    self.mechanism_feedback_blocked_low_quality += 1
                else:
                    self.mechanism_feedback_blocked_replication += 1
                result['feedback_blocked'] = str(reason)
                return 0.0
        return super(ActiveDiagnosticMaterialTissue, self)._apply_feedback(result, config)

    def formal_prepare(self, port, dt, config, cell, world):
        if config.neural_escrow_sanitation_enabled:
            controller_caps = {
                f06.FORMAL_CONTROLLER_ID: (
                    max(0.0, config.formal_controller_escrow_atp_target),
                    max(0.0, config.formal_controller_escrow_signal_target),
                )
            }
            port = _MechanismQuiescencePort(port, controller_caps)
        if not config.diagnosis_enabled or config.diagnosis_mode == DIAGNOSIS_OFF:
            # Exact formal-0.6 baseline path.  Turning active diagnosis off must
            # not silently suppress the inherited automatic audit controller.
            return super(ActiveDiagnosticMaterialTissue, self).formal_prepare(
                port, dt, config, cell, world,
            )
        # Generic auto-audits are temporarily suppressed while the evidence
        # sufficiency controller decides whether observation, exposure or a
        # targeted intervention is needed.  The inherited material controller,
        # lease recovery and safety logic still run unmodified.
        auto = bool(config.formal_auto_start)
        config.formal_auto_start = False
        try:
            super(ActiveDiagnosticMaterialTissue, self).formal_prepare(port, dt, config, cell, world)
        finally:
            config.formal_auto_start = auto
        self._061_last_age = float(world.age)
        if not self.formal_enabled or not self.controller_mature:
            return
        frame = port.raw_sensor_fluxes(self.tissue_ids[0])
        debt = self._physical_debt(frame)
        context = self._context(frame, debt)
        self._061_last_frame = frame
        self._061_last_debt = debt.copy()
        self._061_last_concentration = self._current_concentrations(frame)
        self.evidence_ledger.decay(world.age, config)
        self.mechanism_assist.update_lease(world.age, dt, config)
        self.mechanism_conservation.update(
            world.age, dt, mechanism_confirmed=self.mechanism_gain.confirmed,
        )
        self._observe_ordinary_mechanism(world, config)
        # Snapshot delayed molecular episodes before P2 post_step resets them.
        self._061_prev_cue_active = self.cue_episode_active.copy()
        self._061_prev_cue_start = self.cue_episode_start_debt.copy()
        self._061_prev_cue_peak = self.cue_episode_peak_debt.copy()
        self._061_prev_cue_signal_peak = self.cue_episode_peak.copy()

        started_audit = False
        mechanism_trial_busy = bool(
            self.mechanism_probe.active or self.mechanism_assist.trial_active
        )
        if (
            not mechanism_trial_busy
            and config.diagnosis_enabled and config.diagnosis_targeted_audit
            and self.pending_targeted_audit is not None
        ):
            started_audit = self._start_targeted_audit(port, frame, debt, context, world, config)
        if not started_audit and not mechanism_trial_busy:
            self._maybe_start_diagnosis(frame, debt, world, config)

        # Motor-transduction diagnosis is independent of semantic exposure.
        # It is only allowed when the semantic/audit controller is idle.
        self._maybe_start_mechanism_probe(debt, world, config)
        self._maybe_issue_mechanism_conservation(debt, world, config)
        self._maybe_start_assist_trial(debt, world, config)

        # Preserve a low-rate generic audit path once there is no urgent missing
        # evidence.  This keeps 0.6 calibration alive without crowding out the
        # diagnostic exposure.
        if (
            auto and self.auditor.enabled and not self.auditor.active
            and not self.diagnostic.active and not self.mechanism_probe.active
            and not self.mechanism_assist.trial_active
            and self.pending_targeted_audit is None
            and self.active_age >= config.audit_min_active_age
            and world.age >= self.auditor.cooldown_until
            and world.age >= self.formal_no_program_until
        ):
            structurally_safe = debt[1] < config.audit_safe_debt and debt[2] < config.audit_safe_debt
            if structurally_safe and self.formal_rng.random() < max(0.0, config.diagnosis_generic_audit_probability):
                program = self.compiler.compile(self, debt, config)
                if program is not None and self.auditor.start(
                    program, debt, context, world.age, self.formal_rng, config,
                    self.controller_atp_spent, self.controller_material_wear,
                ):
                    self.compiler.started += 1
                    self.formal_last_start_age = world.age

        self.intervention_gate = self.auditor.current_gate(config)
        if self.diagnostic.active and config.diagnosis_freeze_fast_learning:
            self._snapshot_diagnosis_learning()
        if self.diagnostic.active:
            exposing = self.diagnostic.phase == DIAGNOSTIC_EXPOSE
            phase_scale = 1.0 if exposing else 0.18
            self._pay_controller_cost(port, dt, active=exposing, group_size=1 if exposing else 0)
            wear = max(0.0, dt * config.diagnosis_protein_wear_rate * phase_scale)
            port.allocate_budget(
                f06.FORMAL_CONTROLLER_ID,
                {'atp': dt * config.diagnosis_atp_planning_rate * phase_scale, 'protein': wear, 'membrane': 0.0, 'signal': 0.0},
                dt,
            )
            status = self._controller_status(port)
            stores = np.asarray(status['stores'], dtype=float)
            built = port.commit_material(
                f06.FORMAL_CONTROLLER_ID,
                protein=min(wear, float(stores[p0.BUDGET_PROTEIN])),
                damaged_fraction=0.75,
                aggregate_fraction=0.12,
            )
            self.diagnosis_planning_atp += float(built.get('atp_spent', 0.0))
            self.diagnosis_material_wear += float(built.get('damaged_protein', 0.0) + built.get('aggregate', 0.0))
        mechanism_active = bool(
            self.mechanism_probe.active
            or self.mechanism_conservation.active(world.age)
            or self.mechanism_assist.trial_active
            or (self.mechanism_assist.lease_strength > 1e-8 and world.age < self.mechanism_assist.lease_until)
        )
        if mechanism_active:
            self._pay_mechanism_planning_cost(port, dt, config, active=True)

    def _normal_neural_budget_caps(self, config):
        if not config.neural_escrow_sanitation_enabled:
            return {}
        caps = {
            str(tissue_id): (
                max(0.0, config.neural_escrow_atp_target),
                max(0.0, config.neural_escrow_signal_target),
            )
            for tissue_id in self.tissue_ids
        }
        caps[f06.FORMAL_CONTROLLER_ID] = (
            max(0.0, config.formal_controller_escrow_atp_target),
            max(0.0, config.formal_controller_escrow_signal_target),
        )
        return caps

    def _sanitize_neural_escrow(self, port, config):
        caps = self._normal_neural_budget_caps(config)
        if not caps:
            return {'atp': 0.0, 'signal': 0.0}
        total_atp = 0.0
        total_signal = 0.0
        for tissue_id, cap in caps.items():
            try:
                status = port.attachment_status(tissue_id)
            except KeyError:
                continue
            stores = np.asarray(status['stores'], dtype=float)
            amounts = {
                'atp': max(0.0, float(stores[p0.BUDGET_ATP]) - float(cap[0])),
                'signal': max(0.0, float(stores[p0.BUDGET_SIGNAL]) - float(cap[1])),
            }
            if amounts['atp'] > 1e-15 or amounts['signal'] > 1e-15:
                returned = port.return_unused_budget(tissue_id, amounts)
                total_atp += float(returned.get('atp', 0.0))
                total_signal += float(returned.get('signal', 0.0))
        self.neural_escrow_sanitation_calls += 1
        self.neural_escrow_returned_atp += total_atp
        self.neural_escrow_returned_signal += total_signal
        return {'atp': total_atp, 'signal': total_signal}

    def pre_step(self, port, dt, config, gene_activity):
        self._sanitize_neural_escrow(port, config)
        conservation_gate = self.mechanism_conservation.gate(self._061_last_age)
        normal_caps = self._normal_neural_budget_caps(config)
        quiescent = bool(
            config.mechanism_conservation_quiescence_enabled
            and np.any(conservation_gate < 1.0 - 1e-12)
        )
        backup_resource_lease = None
        backup_sensor = None
        backup_rec = None
        backup_bias = None
        backup_homeo = None
        work_port = port
        if quiescent:
            mask = conservation_gate < 1.0 - 1e-12
            before_activity = float(np.sum(np.abs(self.hidden[mask])))
            self.mechanism_conservation.release_excess_escrow(
                port, self, config, self._061_last_age,
            )
            caps = dict(normal_caps)
            caps.update(self.mechanism_conservation.budget_caps(
                self, config, self._061_last_age,
            ))
            work_port = _MechanismQuiescencePort(port, caps)
            backup_resource_lease = self.resource_lease.copy()
            backup_sensor = self.w_sensor.copy()
            backup_rec = self.w_rec.copy()
            backup_bias = self.bias.copy()
            backup_homeo = self.homeo_gain.copy()
            # Controller-mediated quiescence reduces receptor drive, incoming
            # and outgoing recurrent gain, motor influence and the currently
            # active state.  Learned parameters are restored after this step;
            # the lower hidden state and lower material expenditure are real.
            self.resource_lease *= conservation_gate
            self.hidden[mask] *= conservation_gate[mask]
            self.prev_hidden[mask] *= conservation_gate[mask]
            self.w_sensor[mask, :] *= conservation_gate[mask, None]
            self.w_rec[mask, :] *= conservation_gate[mask, None]
            self.bias[mask] *= conservation_gate[mask]
            self.homeo_gain[mask] *= conservation_gate[mask]
            self.mechanism_conservation.cumulative_suppressed_activity += max(
                0.0, before_activity - float(np.sum(np.abs(self.hidden[mask])))
            ) * max(0.0, float(dt))
            self.mechanism_conservation.quiescent_steps += 1
        elif np.any(conservation_gate < 1.0 - 1e-12):
            backup_resource_lease = self.resource_lease.copy()
            self.resource_lease *= conservation_gate
            if normal_caps:
                work_port = _MechanismQuiescencePort(port, normal_caps)
        elif normal_caps:
            work_port = _MechanismQuiescencePort(port, normal_caps)
        try:
            report = super(ActiveDiagnosticMaterialTissue, self).pre_step(work_port, dt, config, gene_activity)
        finally:
            if backup_resource_lease is not None:
                self.resource_lease = backup_resource_lease
            if backup_sensor is not None:
                self.w_sensor = backup_sensor
                self.w_rec = backup_rec
                self.bias = backup_bias
                self.homeo_gain = backup_homeo
        # P2 returns a structured report whose physical motor payment lives in
        # the nested ``effector`` field.  Keep the frozen public return value,
        # but calibrate the command-to-force mechanism from the actual paid
        # body-port report rather than the wrapper dictionary.
        if isinstance(report, dict) and isinstance(report.get('effector'), dict):
            self._061_last_base_effector_report = dict(report.get('effector') or {})
        else:
            self._061_last_base_effector_report = dict(report or {})
        self._061_last_diagnostic_command = None
        self._061_last_mechanism_probe_report = None
        self._061_last_mechanism_assist_report = None
        if not self.controller_mature:
            return report
        valid = np.flatnonzero(self._material_capacity() > 0.15)
        if valid.size == 0:
            return report
        frame = port.raw_sensor_fluxes(self.tissue_ids[int(valid[0])])
        debt = self._physical_debt(frame)
        gateway = self.last_gateway if int(self.last_gateway) >= 0 else int(valid[0])

        if self.diagnostic.active and not self.auditor.active:
            command = self.diagnostic.command(frame, debt, dt, self._061_last_age, config)
            self._061_last_diagnostic_command = command
            if command is not None and config.diagnosis_mode not in (DIAGNOSIS_PASSIVE, DIAGNOSIS_OFF):
                extra = port.apply_effector_fluxes(
                    self.tissue_ids[gateway],
                    {
                        'motor': np.asarray(command['motor'], dtype=float),
                        'transporter_polarity': np.asarray(command['transporter_polarity'], dtype=float),
                    },
                    dt,
                )
                self.diagnostic.record_report(extra)
                self.last_effector_report = dict(self.last_effector_report)
                for key in ('motor_force', 'atp_spent', 'signal_spent'):
                    self.last_effector_report[key] = float(self.last_effector_report.get(key, 0.0)) + float(extra.get(key, 0.0))

        # A mechanism probe has priority over assist and never overlaps a stop
        # audit or semantic exposure.  The pulse is paid by the formal
        # controller attachment and experiences the same damaged motor physics.
        if self.mechanism_probe.active and not self.auditor.active and not self.diagnostic.active:
            probe_command = self.mechanism_probe.command()
            if probe_command is not None:
                extra = port.apply_effector_fluxes(
                    f06.FORMAL_CONTROLLER_ID, {'motor': probe_command}, dt,
                )
                self._061_last_mechanism_probe_report = dict(extra)
                before_confirmed = bool(self.mechanism_gain.confirmed)
                result = self.mechanism_probe.record(
                    extra, dt, self._061_last_age, config, self.mechanism_gain,
                    mechanism_probability=self.change_sentinel.mechanism_probability,
                )
                if result is not None and self.mechanism_gain.confirmed and not before_confirmed:
                    self.mechanism_probe_confirmed_events += 1
                    strength = clamp(0.42 + 0.52 * self.mechanism_gain.loss_fraction, 0.0, 0.995)
                    self.change_sentinel.mechanism_probability = max(
                        self.change_sentinel.mechanism_probability, strength,
                    )
                    self.change_sentinel.mechanism_channel_probability[:] = np.maximum(
                        self.change_sentinel.mechanism_channel_probability,
                        np.asarray([0.45, 0.12, 0.22, 0.68]) * strength,
                    )
                    self.change_sentinel.channel_probability[:] = np.maximum(
                        self.change_sentinel.channel_probability,
                        self.change_sentinel.mechanism_channel_probability,
                    )

        elif not self.auditor.active and not self.diagnostic.active:
            direction = self.mechanism_assist.active_direction(self._action_direction())
            strength = self.mechanism_assist.active_strength(
                self._061_last_age, self.mechanism_gain.loss_fraction, config,
            )
            if (
                strength > 1e-10 and float(np.linalg.norm(direction)) > 1e-10
                and self._mechanism_safe(debt, config)
            ):
                extra = port.apply_effector_fluxes(
                    f06.FORMAL_CONTROLLER_ID,
                    {'motor': direction * strength}, dt,
                )
                self._061_last_mechanism_assist_report = dict(extra)
                self.mechanism_assist_commands += 1
                self.last_effector_report = dict(self.last_effector_report)
                for key in ('motor_force', 'atp_spent', 'signal_spent'):
                    self.last_effector_report[key] = float(self.last_effector_report.get(key, 0.0)) + float(extra.get(key, 0.0))
        return report

    def _observe_delayed_cue_outcomes(self, frame, debt, dt, world, config):
        current_concentration = self._current_concentrations(frame)
        active_now = np.asarray(self.cue_episode_active, dtype=bool)
        self._061_episode_max_concentration[active_now] = np.maximum(
            self._061_episode_max_concentration[active_now],
            current_concentration[active_now],
        )
        self._061_episode_concentration_integral[active_now] += (
            current_concentration[active_now] * max(0.0, float(dt))
        )
        ended = self._061_prev_cue_active & (~self.cue_episode_active)
        if not np.any(ended):
            return []
        outcomes = []
        candidate_ligands = set(getattr(config, 'diagnosis_candidate_ligands', (ALT_LIGAND,)))
        for ligand in np.flatnonzero(ended):
            if int(ligand) not in candidate_ligands:
                continue
            start = self._061_prev_cue_start[ligand].copy()
            # Reconstruct the same *channel-resolved* delayed physical outcome
            # used by P2 before it clears the episode peak.  Collapsing this to
            # a single energy channel made audits blind whenever energy debt was
            # saturated, even though the changed molecule was visibly altering
            # damage or fatigue.  No semantic label is added: the vector is only
            # start debt minus terminal debt, with the ordinary P2 acute-harm
            # memory applied to each physical channel.
            peak = self._061_prev_cue_peak[ligand].copy()
            terminal_improvement = start - np.asarray(debt, dtype=float)
            acute_harm = np.maximum(peak - start, 0.0)
            outcome = (
                terminal_improvement
                - max(0.0, config.p2_episode_harm_memory) * acute_harm
            )
            peak_signal = max(
                0.0,
                float(self._061_prev_cue_signal_peak[ligand]),
                float(self._061_episode_max_concentration[ligand]),
            )
            peak_quality = clamp(
                peak_signal / max(config.diagnosis_semantic_peak_target, 1e-9),
                0.0, 1.0,
            )
            dose_quality = clamp(
                float(self._061_episode_concentration_integral[ligand])
                / max(config.diagnosis_semantic_dose_target, 1e-9),
                0.0, 1.0,
            )
            quality = clamp(0.40 * peak_quality + 0.60 * dose_quality, 0.05, 1.0)
            diagnostics = self.evidence_ledger.observe_semantic(
                int(ligand), outcome, quality, world.age, start, config,
            )
            self.change_sentinel.observe_semantic(
                int(ligand), outcome, quality, world.age, config,
            )
            outcomes.append((int(ligand), outcome.copy(), diagnostics))
            self._061_episode_max_concentration[ligand] = 0.0
            self._061_episode_concentration_integral[ligand] = 0.0
        return outcomes

    def _pay_evidence_event_cost(self, port, config):
        atp_request = max(0.0, float(config.diagnosis_atp_evidence_event))
        wear = max(0.0, float(config.diagnosis_evidence_wear))
        if atp_request <= 0.0 and wear <= 0.0:
            return
        port.allocate_budget(
            f06.FORMAL_CONTROLLER_ID,
            {'atp': atp_request, 'protein': wear, 'membrane': 0.0, 'signal': 0.0},
            1.0 / SIM_HZ,
        )
        if wear > 0.0:
            status = self._controller_status(port)
            stores = np.asarray(status['stores'], dtype=float)
            built = port.commit_material(
                f06.FORMAL_CONTROLLER_ID,
                protein=min(wear, float(stores[p0.BUDGET_PROTEIN])),
                damaged_fraction=0.75,
                aggregate_fraction=0.12,
            )
            self.diagnosis_planning_atp += float(built.get('atp_spent', 0.0))
            self.diagnosis_material_wear += float(
                built.get('damaged_protein', 0.0) + built.get('aggregate', 0.0)
            )

    def formal_observe(self, port, dt, config, cell, world):
        result = super(ActiveDiagnosticMaterialTissue, self).formal_observe(port, dt, config, cell, world)
        if not config.diagnosis_enabled or config.diagnosis_mode == DIAGNOSIS_OFF:
            return result
        if not self.formal_enabled or not self.controller_mature:
            return result
        frame = port.raw_sensor_fluxes(self.tissue_ids[0])
        debt = self._physical_debt(frame)
        delayed = self._observe_delayed_cue_outcomes(frame, debt, dt, world, config)
        for ligand, outcome, diagnostics in delayed:
            self._pay_evidence_event_cost(port, config)
            if diagnostics is None:
                continue
            change = float(diagnostics['change_evidence'])
            channel = int(diagnostics['channel'])
            if change >= config.diagnosis_audit_trigger:
                self.change_sentinel.semantic_channel_probability[channel] = max(
                    self.change_sentinel.semantic_channel_probability[channel],
                    clamp(0.20 + 0.78 * change, 0.0, 0.995),
                )
                self.change_sentinel.semantic_probability = float(np.max(self.change_sentinel.semantic_channel_probability))
                self.change_sentinel.channel_probability[channel] = max(
                    self.change_sentinel.channel_probability[channel],
                    self.change_sentinel.semantic_channel_probability[channel],
                )
                self.pending_targeted_audit = (int(ligand), channel)
        if result is not None and result.get('status') not in ('aborted',):
            program = result.get('program', {})
            self.evidence_ledger.observe_audit(
                result.get('channel', 0), result.get('quality', 0.0), world.age,
                ligand=program.get('target_ligand'), config=config,
            )
            if result.get('program', {}).get('family') == 'active-semantic-mediation':
                self.last_diagnosis_result = dict(result)
                if float(result.get('feedback_magnitude', 0.0)) > 0.0:
                    self.diagnosis_feedback_events += 1
        diag_result = self.diagnostic.observe(dt, world.age, self.evidence_ledger, config)
        if diag_result is not None:
            self.last_diagnosis_result = diag_result
            evidence = diag_result.get('evidence', {})
            if evidence.get('change_evidence', 0.0) >= config.diagnosis_audit_trigger:
                self.pending_targeted_audit = (int(diag_result['ligand']), int(diag_result['channel']))
            else:
                self.diagnosis_evidence_refreshes += 1

        uptake = float(np.sum(np.maximum(
            0.0, np.asarray(frame['flux']['last_uptake_by_ligand'], dtype=float)
        )))
        concentration = float(np.max(self._current_concentrations(frame)))
        assist_result = self.mechanism_assist.record_step(
            self._061_last_mechanism_assist_report, debt, uptake,
            concentration, dt, world.age, config,
        )
        if assist_result is not None and assist_result.get('accepted'):
            self.mechanism_assist_feedback_events += 1
        if config.diagnosis_freeze_fast_learning and self._diagnosis_learning_snapshot is not None:
            self._restore_diagnosis_learning()
        return result

    def finite(self):
        return bool(
            super(ActiveDiagnosticMaterialTissue, self).finite()
            and self.evidence_ledger.finite()
            and self.feedback_replication.finite()
            and self.mechanism_gain.finite()
            and self.mechanism_probe.finite()
            and self.mechanism_conservation.finite()
            and self.mechanism_assist.finite()
            and finite_array(self._061_last_concentration)
            and finite_array(self._061_last_debt)
            and np.isfinite(float(self.diagnosis_planning_atp))
            and np.isfinite(float(self.diagnosis_material_wear))
            and np.isfinite(float(self.neural_escrow_returned_atp))
            and np.isfinite(float(self.neural_escrow_returned_signal))
        )

    def diagnostics_061(self, age, config):
        out = self.evidence_ledger.diagnostics(age, config)
        out.update({
            'diagnosis_active': bool(self.diagnostic.active),
            'diagnosis_phase': self.diagnostic.phase,
            'diagnosis_started': int(self.diagnostic.started),
            'diagnosis_completed': int(self.diagnostic.completed),
            'diagnosis_aborted': int(self.diagnostic.aborted),
            'diagnosis_targeted_audits': int(self.diagnosis_targeted_audits),
            'diagnosis_feedback_events': int(self.diagnosis_feedback_events),
            'pending_targeted_audit': self.pending_targeted_audit,
            'pending_audit_visible_since': self.pending_audit_visible_since,
            'diagnosis_deferred_unsafe': int(self.diagnosis_deferred_unsafe),
            'diagnosis_deferred_no_target': int(self.diagnosis_deferred_no_target),
            'diagnosis_deferred_low_value': int(self.diagnosis_deferred_low_value),
            'diagnosis_deferred_busy': int(self.diagnosis_deferred_busy),
            'diagnosis_last_defer_reason': self._061_last_defer_reason,
            'diagnosis_cumulative_exposure': float(self.diagnostic.cumulative_ligand_exposure),
            'diagnosis_mean_evidence_latency': (
                self.diagnostic.cumulative_evidence_latency / max(self.diagnostic.evidence_latency_count, 1)
            ),
            'feedback_replication_accepted': int(self.feedback_replication.accepted),
            'feedback_blocked_low_quality': int(self.mechanism_feedback_blocked_low_quality),
            'feedback_blocked_replication': int(self.mechanism_feedback_blocked_replication),
            'mechanism_baseline_gain': float(self.mechanism_gain.baseline_gain),
            'mechanism_recent_gain': float(self.mechanism_gain.recent_gain),
            'mechanism_gain_ratio': float(self.mechanism_gain.gain_ratio),
            'mechanism_loss_fraction': float(self.mechanism_gain.loss_fraction),
            'mechanism_confirmed': bool(self.mechanism_gain.confirmed),
            'mechanism_probe_started': int(self.mechanism_probe.started),
            'mechanism_probe_completed': int(self.mechanism_probe.completed),
            'mechanism_probe_confirmed_events': int(self.mechanism_probe_confirmed_events),
            'mechanism_probe_atp': float(self.mechanism_probe.cumulative_atp),
            'mechanism_probe_force': float(self.mechanism_probe.cumulative_force),
            'mechanism_conservation_active': bool(self.mechanism_conservation.active(age)),
            'mechanism_conservation_issued': int(self.mechanism_conservation.issued),
            'mechanism_conservation_scale': float(self.mechanism_conservation.scale),
            'mechanism_conservation_active_time': float(self.mechanism_conservation.cumulative_active_time),
            'mechanism_conservation_returned_atp': float(self.mechanism_conservation.cumulative_returned_atp),
            'mechanism_conservation_returned_signal': float(self.mechanism_conservation.cumulative_returned_signal),
            'mechanism_conservation_suppressed_activity': float(self.mechanism_conservation.cumulative_suppressed_activity),
            'mechanism_conservation_quiescent_steps': int(self.mechanism_conservation.quiescent_steps),
            'neural_escrow_sanitation_calls': int(self.neural_escrow_sanitation_calls),
            'neural_escrow_returned_atp': float(self.neural_escrow_returned_atp),
            'neural_escrow_returned_signal': float(self.neural_escrow_returned_signal),
            'mechanism_assist_trials': int(self.mechanism_assist.trials),
            'mechanism_assist_accepted': int(self.mechanism_assist.accepted),
            'mechanism_assist_rejected': int(self.mechanism_assist.rejected),
            'mechanism_assist_lease_strength': float(self.mechanism_assist.lease_strength),
            'mechanism_assist_commands': int(self.mechanism_assist_commands),
            'mechanism_assist_feedback_events': int(self.mechanism_assist_feedback_events),
            'mechanism_assist_atp': float(self.mechanism_assist.cumulative_atp),
            'mechanism_assist_force': float(self.mechanism_assist.cumulative_force),
            'mechanism_assist_last_effect': float(self.mechanism_assist.last_effect),
            'mechanism_assist_last_quality': float(self.mechanism_assist.last_quality),
        })
        return out

    def state_dict(self):
        state = super(ActiveDiagnosticMaterialTissue, self).state_dict()
        state.update({
            'schema_061': SCHEMA_VERSION,
            'evidence_ledger': self.evidence_ledger.state_dict(),
            'diagnostic': self.diagnostic.state_dict(),
            'compiler_061': self.compiler.state_dict(),
            'feedback_replication': self.feedback_replication.state_dict(),
            'mechanism_gain': self.mechanism_gain.state_dict(),
            'mechanism_probe': self.mechanism_probe.state_dict(),
            'mechanism_conservation': self.mechanism_conservation.state_dict(),
            'mechanism_assist': self.mechanism_assist.state_dict(),
            'pending_targeted_audit': self.pending_targeted_audit,
            'last_diagnosis_result': self.last_diagnosis_result,
            'diagnosis_attempts': self.diagnosis_attempts,
            'diagnosis_targeted_audits': self.diagnosis_targeted_audits,
            'diagnosis_feedback_events': self.diagnosis_feedback_events,
            'diagnosis_planning_atp': self.diagnosis_planning_atp,
            'diagnosis_material_wear': self.diagnosis_material_wear,
            'diagnosis_false_starts': self.diagnosis_false_starts,
            'diagnosis_evidence_refreshes': self.diagnosis_evidence_refreshes,
            'diagnosis_deferred_unsafe': self.diagnosis_deferred_unsafe,
            'diagnosis_deferred_no_target': self.diagnosis_deferred_no_target,
            'diagnosis_deferred_low_value': self.diagnosis_deferred_low_value,
            'diagnosis_deferred_busy': self.diagnosis_deferred_busy,
            'mechanism_probe_attempts': self.mechanism_probe_attempts,
            'mechanism_probe_confirmed_events': self.mechanism_probe_confirmed_events,
            'mechanism_assist_commands': self.mechanism_assist_commands,
            'mechanism_assist_feedback_events': self.mechanism_assist_feedback_events,
            'mechanism_conservation_feedback_events': self.mechanism_conservation_feedback_events,
            'neural_escrow_sanitation_calls': self.neural_escrow_sanitation_calls,
            'neural_escrow_returned_atp': self.neural_escrow_returned_atp,
            'neural_escrow_returned_signal': self.neural_escrow_returned_signal,
            'mechanism_feedback_blocked_low_quality': self.mechanism_feedback_blocked_low_quality,
            'mechanism_feedback_blocked_replication': self.mechanism_feedback_blocked_replication,
            '_061_last_defer_age': self._061_last_defer_age,
            '_061_last_defer_reason': self._061_last_defer_reason,
            '_061_prev_cue_active': self._061_prev_cue_active.copy(),
            '_061_prev_cue_start': self._061_prev_cue_start.copy(),
            '_061_prev_cue_peak': self._061_prev_cue_peak.copy(),
            '_061_prev_cue_signal_peak': self._061_prev_cue_signal_peak.copy(),
            '_061_episode_max_concentration': self._061_episode_max_concentration.copy(),
            '_061_episode_concentration_integral': self._061_episode_concentration_integral.copy(),
            '_061_last_age': self._061_last_age,
            '_061_last_concentration': self._061_last_concentration.copy(),
            '_061_last_debt': self._061_last_debt.copy(),
        })
        return state

    @classmethod
    def from_state(cls, state):
        base = f06.FormalMaterialTissue.from_state(state)
        base.__class__ = cls
        obj = base
        obj._init_061_runtime()
        if 'evidence_ledger' in state:
            obj.evidence_ledger = EvidenceSufficiencyLedger.from_state(state['evidence_ledger'])
        if 'diagnostic' in state:
            obj.diagnostic = BoundedDiagnosticExposure.from_state(state['diagnostic'])
        if 'compiler_061' in state:
            obj.compiler = TargetedDiagnosticCompiler.from_state(state['compiler_061'])
        elif isinstance(obj.compiler, f06.SmallFalsificationCompiler):
            obj.compiler = TargetedDiagnosticCompiler.from_state(obj.compiler.state_dict())
        if 'feedback_replication' in state:
            obj.feedback_replication = FeedbackReplicationGate.from_state(state['feedback_replication'])
        if 'mechanism_gain' in state:
            obj.mechanism_gain = MechanismGainLedger.from_state(state['mechanism_gain'])
        if 'mechanism_probe' in state:
            obj.mechanism_probe = PaidMechanismProbe.from_state(state['mechanism_probe'])
        if 'mechanism_conservation' in state:
            obj.mechanism_conservation = MechanismConservationLease.from_state(state['mechanism_conservation'])
        if 'mechanism_assist' in state:
            obj.mechanism_assist = MechanismAssistLease.from_state(state['mechanism_assist'])
        obj.pending_targeted_audit = state.get('pending_targeted_audit')
        obj.pending_audit_visible_since = float(state.get('pending_audit_visible_since', -1e9))
        obj.last_diagnosis_result = state.get('last_diagnosis_result')
        for name in (
            'diagnosis_attempts', 'diagnosis_targeted_audits',
            'diagnosis_feedback_events', 'diagnosis_false_starts',
            'diagnosis_evidence_refreshes', 'diagnosis_deferred_unsafe',
            'diagnosis_deferred_no_target', 'diagnosis_deferred_low_value',
            'diagnosis_deferred_busy',
            'mechanism_probe_attempts', 'mechanism_probe_confirmed_events',
            'mechanism_assist_commands', 'mechanism_assist_feedback_events',
            'mechanism_conservation_feedback_events',
            'neural_escrow_sanitation_calls',
            'mechanism_feedback_blocked_low_quality',
            'mechanism_feedback_blocked_replication',
        ):
            setattr(obj, name, int(state.get(name, getattr(obj, name))))
        for name in (
            'diagnosis_planning_atp', 'diagnosis_material_wear',
            'neural_escrow_returned_atp', 'neural_escrow_returned_signal',
            '_061_last_age', '_061_last_defer_age',
        ):
            setattr(obj, name, float(state.get(name, getattr(obj, name))))
        obj._061_last_defer_reason = str(state.get('_061_last_defer_reason', obj._061_last_defer_reason))
        for name in ('_061_prev_cue_active', '_061_prev_cue_start', '_061_prev_cue_peak', '_061_prev_cue_signal_peak', '_061_episode_max_concentration', '_061_episode_concentration_integral', '_061_last_concentration', '_061_last_debt'):
            if name in state:
                setattr(obj, name, np.asarray(state[name]).copy())
        return obj


class Formal061ProtoCell(f06.FormalProtoCell):
    @classmethod
    def from_state(cls, rng, state):
        cell = f06.FormalProtoCell.from_state(rng, state)
        cell.__class__ = cls
        if state.get('p2_tissue') is not None:
            cell.p2_tissue = ActiveDiagnosticMaterialTissue.from_state(state['p2_tissue'])
        return cell


class Formal061World(f06.Formal06World):
    def __init__(self, seed=101, initial_cells=1, config=None):
        config = config if config is not None else Formal061Config()
        if not isinstance(config, Formal061Config):
            config = Formal061Config(**config.state_dict())
        super(Formal061World, self).__init__(seed=seed, initial_cells=initial_cells, config=config)
        self.config = config
        self.diagnosis_world_steps = 0
        self.mechanism_fault_applied = False
        self.mechanism_fault_events = 0
        self.mechanism_fault_original_gain = float(self.config.neural_motor_gain_scale)
        for cell in self.cells:
            cell.__class__ = Formal061ProtoCell
            if cell.p2_tissue is not None and not isinstance(cell.p2_tissue, ActiveDiagnosticMaterialTissue):
                cell.p2_tissue = ActiveDiagnosticMaterialTissue.from_state(cell.p2_tissue.state_dict())
        self._ensure_all_p2_tissues()
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def _new_p2_tissue(self, cell):
        if self.config.diagnosis_mode == DIAGNOSIS_OFF:
            tissue = super(Formal061World, self)._new_p2_tissue(cell)
            if tissue is not None and not isinstance(tissue, ActiveDiagnosticMaterialTissue):
                tissue = ActiveDiagnosticMaterialTissue.from_state(tissue.state_dict())
                cell.p2_tissue = tissue
            return tissue
        if self.config.p2_tissue_mode == p2.P2_MODE_NONE:
            return None
        params = p2.p2_gene_parameters(cell)
        activity = p2.p2_gene_activity(cell)
        if params is None or np.count_nonzero(activity >= 0.016) < p2.P2_CELL_COUNT:
            return None
        seed = ((self.p2_seed * 1000003) ^ (int(cell.cell_id) * 9176) ^ (int(cell.generation) * 7919) ^ 0x061D1A) & 0xFFFFFFFF
        tissue = ActiveDiagnosticMaterialTissue(
            params, mode=self.config.p2_tissue_mode, rng_seed=seed,
            formal_enabled=f06.formal_controller_activity(cell) >= 0.016,
        )
        tissue.ensure_attachments(self.port_for(cell.cell_id), activity)
        cell.p2_tissue = tissue
        cell.p2_tissue_births += 1
        self.p2_tissue_creations += 1
        self.formal_tissue_creations += 1
        return tissue

    def _ensure_all_p2_tissues(self):
        super(Formal061World, self)._ensure_all_p2_tissues()
        for cell in self.living_cells():
            if not isinstance(cell, Formal061ProtoCell):
                cell.__class__ = Formal061ProtoCell
            tissue = cell.p2_tissue
            if tissue is not None and not isinstance(tissue, ActiveDiagnosticMaterialTissue):
                cell.p2_tissue = ActiveDiagnosticMaterialTissue.from_state(tissue.state_dict())

    def _apply_mechanism_fault_if_due(self):
        if (
            self.mechanism_fault_applied
            or not self.config.mechanism_fault_enabled
            or float(self.age) < self.config.mechanism_fault_age
        ):
            return False
        self.mechanism_fault_applied = True
        self.mechanism_fault_events += 1
        self.mechanism_fault_original_gain = float(self.config.neural_motor_gain_scale)
        self.config.neural_motor_gain_scale = max(
            0.01,
            float(self.config.neural_motor_gain_scale)
            * clamp(self.config.mechanism_fault_gain_scale, 0.02, 1.0),
        )
        wear = max(0.0, float(self.config.mechanism_fault_wear))
        if wear > 0.0:
            # Positive-control tissue injury is a conversion of matter already
            # present in the selected neural compartments.  It must not request
            # fresh body protein merely to manufacture a 'damaged' pool.
            for cell in self.living_cells():
                tissue = getattr(cell, 'p2_tissue', None)
                if not isinstance(tissue, ActiveDiagnosticMaterialTissue):
                    continue
                score = np.abs(tissue.hidden * tissue.motor_gain)
                indices = np.argsort(-score)[:2]
                port = self.port_for(cell.cell_id)
                for index in indices:
                    state = port._attachment(tissue.tissue_ids[int(index)])
                    if state is None:
                        continue
                    functional = float(state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN])
                    converted = min(functional, 0.5 * wear)
                    if converted <= 0.0:
                        continue
                    state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN] -= converted
                    state.tissue_material[p0.TISSUE_DAMAGED_PROTEIN] += 0.84 * converted
                    state.tissue_material[p0.TISSUE_AGGREGATE] += 0.16 * converted
        return True

    def step(self, dt):
        self._apply_mechanism_fault_if_due()
        super(Formal061World, self).step(dt)
        self.diagnosis_world_steps += 1

    def finite(self):
        if not super(Formal061World, self).finite():
            return False
        return all(
            tissue.finite()
            for tissue in (getattr(cell, 'p2_tissue', None) for cell in self.cells)
            if isinstance(tissue, ActiveDiagnosticMaterialTissue)
        )

    def summary(self):
        out = super(Formal061World, self).summary()
        tissues = [
            cell.p2_tissue for cell in self.living_cells()
            if isinstance(getattr(cell, 'p2_tissue', None), ActiveDiagnosticMaterialTissue)
        ]
        mean = lambda values: float(np.mean(list(values))) if tissues else 0.0
        out.update({
            'build': BUILD,
            'diagnosis_schema': SCHEMA_VERSION,
            'diagnosis_mode': self.config.diagnosis_mode,
            'diagnosis_tissues': len(tissues),
            'diagnosis_started': int(sum(t.diagnostic.started for t in tissues)),
            'diagnosis_completed': int(sum(t.diagnostic.completed for t in tissues)),
            'diagnosis_aborted': int(sum(t.diagnostic.aborted for t in tissues)),
            'diagnosis_targeted_audits': int(sum(t.diagnosis_targeted_audits for t in tissues)),
            'diagnosis_feedback_events': int(sum(t.diagnosis_feedback_events for t in tissues)),
            'diagnosis_atp': float(sum(t.diagnostic.cumulative_atp + t.diagnosis_planning_atp for t in tissues)),
            'diagnosis_signal': float(sum(t.diagnostic.cumulative_signal for t in tissues)),
            'diagnosis_motor_force': float(sum(t.diagnostic.cumulative_motor_force for t in tissues)),
            'diagnosis_material_wear': float(sum(t.diagnosis_material_wear for t in tissues)),
            'diagnosis_evidence_events': int(sum(t.evidence_ledger.semantic_events for t in tissues)),
            'diagnosis_audit_evidence': int(sum(t.evidence_ledger.audit_events for t in tissues)),
            'diagnosis_max_change_evidence': mean(np.max(t.evidence_ledger.change_evidence) for t in tissues),
            'diagnosis_mean_observation_sufficiency': mean(
                np.mean([
                    t.evidence_ledger.observation_sufficiency(l, k, self.age, self.config)
                    for l in range(t.evidence_ledger.ligand_count)
                    for k in range(t.evidence_ledger.channels)
                ]) for t in tissues
            ),
            'diagnosis_pending_audits': int(sum(t.pending_targeted_audit is not None for t in tissues)),
            'diagnosis_deferred_unsafe': int(sum(t.diagnosis_deferred_unsafe for t in tissues)),
            'diagnosis_deferred_no_target': int(sum(t.diagnosis_deferred_no_target for t in tissues)),
            'diagnosis_deferred_low_value': int(sum(t.diagnosis_deferred_low_value for t in tissues)),
            'diagnosis_deferred_busy': int(sum(t.diagnosis_deferred_busy for t in tissues)),
            'diagnosis_cumulative_exposure': float(sum(t.diagnostic.cumulative_ligand_exposure for t in tissues)),
            'diagnosis_directional_samples': int(sum(t.diagnostic.cumulative_directional_samples for t in tissues)),
            'diagnosis_mean_evidence_latency': float(np.mean([
                t.diagnostic.cumulative_evidence_latency / max(t.diagnostic.evidence_latency_count, 1)
                for t in tissues if t.diagnostic.evidence_latency_count > 0
            ])) if any(t.diagnostic.evidence_latency_count > 0 for t in tissues) else 0.0,
            'diagnosis_world_steps': int(self.diagnosis_world_steps),
            'mechanism_fault_applied': int(self.mechanism_fault_applied),
            'mechanism_fault_events': int(self.mechanism_fault_events),
            'mechanism_motor_gain_scale': float(self.config.neural_motor_gain_scale),
            'mechanism_baseline_gain': mean(t.mechanism_gain.baseline_gain for t in tissues),
            'mechanism_recent_gain': mean(t.mechanism_gain.recent_gain for t in tissues),
            'mechanism_gain_ratio': mean(t.mechanism_gain.gain_ratio for t in tissues),
            'mechanism_loss_fraction': mean(t.mechanism_gain.loss_fraction for t in tissues),
            'mechanism_confirmed': int(sum(t.mechanism_gain.confirmed for t in tissues)),
            'mechanism_probe_started': int(sum(t.mechanism_probe.started for t in tissues)),
            'mechanism_probe_completed': int(sum(t.mechanism_probe.completed for t in tissues)),
            'mechanism_probe_confirmed_events': int(sum(t.mechanism_probe_confirmed_events for t in tissues)),
            'mechanism_probe_atp': float(sum(t.mechanism_probe.cumulative_atp for t in tissues)),
            'mechanism_probe_force': float(sum(t.mechanism_probe.cumulative_force for t in tissues)),
            'mechanism_conservation_active': int(sum(t.mechanism_conservation.active(self.age) for t in tissues)),
            'mechanism_conservation_issued': int(sum(t.mechanism_conservation.issued for t in tissues)),
            'mechanism_conservation_feedback_events': int(sum(t.mechanism_conservation_feedback_events for t in tissues)),
            'mechanism_conservation_mean_scale': mean(t.mechanism_conservation.scale for t in tissues),
            'mechanism_conservation_active_time': float(sum(t.mechanism_conservation.cumulative_active_time for t in tissues)),
            'mechanism_conservation_returned_atp': float(sum(t.mechanism_conservation.cumulative_returned_atp for t in tissues)),
            'mechanism_conservation_returned_signal': float(sum(t.mechanism_conservation.cumulative_returned_signal for t in tissues)),
            'mechanism_conservation_suppressed_activity': float(sum(t.mechanism_conservation.cumulative_suppressed_activity for t in tissues)),
            'mechanism_conservation_quiescent_steps': int(sum(t.mechanism_conservation.quiescent_steps for t in tissues)),
            'mechanism_conservation_alignment_fallbacks': int(sum(t.mechanism_conservation.alignment_fallbacks for t in tissues)),
            'mechanism_conservation_selected_alignment': mean(
                np.mean(t.mechanism_conservation.last_target_alignment[t.mechanism_conservation.target_mask])
                if np.any(t.mechanism_conservation.target_mask) else 0.0
                for t in tissues
            ),
            'neural_escrow_sanitation_calls': int(sum(t.neural_escrow_sanitation_calls for t in tissues)),
            'neural_escrow_returned_atp': float(sum(t.neural_escrow_returned_atp for t in tissues)),
            'neural_escrow_returned_signal': float(sum(t.neural_escrow_returned_signal for t in tissues)),
            'mechanism_assist_trials': int(sum(t.mechanism_assist.trials for t in tissues)),
            'mechanism_assist_evidence_accepted': int(sum(t.mechanism_assist.evidence_accepted for t in tissues)),
            'mechanism_assist_accepted': int(sum(t.mechanism_assist.accepted for t in tissues)),
            'mechanism_assist_rejected': int(sum(t.mechanism_assist.rejected for t in tissues)),
            'mechanism_assist_commands': int(sum(t.mechanism_assist_commands for t in tissues)),
            'mechanism_assist_feedback_events': int(sum(t.mechanism_assist_feedback_events for t in tissues)),
            'mechanism_assist_atp': float(sum(t.mechanism_assist.cumulative_atp for t in tissues)),
            'mechanism_assist_force': float(sum(t.mechanism_assist.cumulative_force for t in tissues)),
            'mechanism_assist_mean_lease': mean(t.mechanism_assist.lease_strength for t in tissues),
            'mechanism_assist_mean_effect': mean(t.mechanism_assist.last_effect for t in tissues),
            'feedback_blocked_low_quality': int(sum(t.mechanism_feedback_blocked_low_quality for t in tissues)),
            'feedback_blocked_replication': int(sum(t.mechanism_feedback_blocked_replication for t in tissues)),
        })
        return out

    def state_dict(self):
        state = super(Formal061World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'diagnosis_world_steps': self.diagnosis_world_steps,
            'mechanism_fault_applied': self.mechanism_fault_applied,
            'mechanism_fault_events': self.mechanism_fault_events,
            'mechanism_fault_original_gain': self.mechanism_fault_original_gain,
        })
        return state

    @classmethod
    def from_state(cls, state):
        base = dict(state)
        base['save_version'] = f06.SAVE_VERSION
        base['build'] = f06.BUILD
        formal_keys = set(f06.Formal06Config().__dict__.keys())
        base['config'] = {key: value for key, value in dict(state.get('config', {})).items() if key in formal_keys}
        world = f06.Formal06World.from_state(base)
        world.__class__ = cls
        world.config = Formal061Config.from_state(state.get('config', {}))
        world.cells = [Formal061ProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        world.diagnosis_world_steps = int(state.get('diagnosis_world_steps', 0))
        world.mechanism_fault_applied = bool(state.get('mechanism_fault_applied', False))
        world.mechanism_fault_events = int(state.get('mechanism_fault_events', 0))
        world.mechanism_fault_original_gain = float(
            state.get('mechanism_fault_original_gain', world.config.neural_motor_gain_scale)
        )
        return world

    def save(self, path=SAVE_FILE):
        _atomic_pickle(path, self.state_dict())

    @classmethod
    def load(cls, path=SAVE_FILE):
        with open(path, 'rb') as handle:
            return cls.from_state(pickle.load(handle))

    def clone(self):
        return Formal061World.from_state(self.state_dict())


def set_061_runtime_mode(world, diagnosis_mode=None, formal_mode=None):
    if formal_mode is not None:
        f06.set_formal_runtime_mode(world, formal_mode)
    if diagnosis_mode is not None:
        if diagnosis_mode not in DIAGNOSIS_MODES:
            raise ValueError(diagnosis_mode)
        world.config.diagnosis_mode = diagnosis_mode
        world.config.diagnosis_enabled = diagnosis_mode != DIAGNOSIS_OFF
    return world


def apply_061_common_disturbance_tape(world, seed, step, stream=0):
    f06.apply_formal_common_disturbance_tape(world, seed, step, stream=stream)
    # Diagnostic random directions are generated by formal_rng, whose state is
    # already tied to step/cell/stream by the inherited common tape.
    return world


def run_headless_trial(seed=101, seconds=120.0, initial_cells=1, config=None):
    world = Formal061World(seed=seed, initial_cells=initial_cells, config=config or Formal061Config())
    dt = 1.0 / SIM_HZ
    margin = 0.0
    post = 0.0
    count = 0
    post_count = 0
    start_uptake = world.p2_reward_uptake_total
    start_toxin = world.p2_cue_toxin_total
    for _ in range(int(round(seconds * SIM_HZ))):
        if not world.living_cells():
            break
        world.step(dt)
        current = float(np.mean([cell.autopoietic_margin() for cell in world.living_cells()])) if world.living_cells() else 0.0
        margin += current
        count += 1
        if world.age >= world.config.p2_switch_age:
            post += current
            post_count += 1
    result = world.summary()
    result.update({
        'mean_margin_over_life': margin / max(count, 1),
        'post_switch_mean_margin': post / max(post_count, 1),
        'uptake_during_trial': max(0.0, world.p2_reward_uptake_total - start_uptake),
        'toxin_during_trial': max(0.0, world.p2_cue_toxin_total - start_toxin),
        'finite': int(world.finite()),
        'final_mass_residual': world.matter_ledger_residual(),
    })
    return result


LOG_FIELDS = tuple(list(f06.LOG_FIELDS) + [
    'diagnosis_mode', 'diagnosis_tissues', 'diagnosis_started',
    'diagnosis_completed', 'diagnosis_aborted', 'diagnosis_targeted_audits',
    'diagnosis_feedback_events', 'diagnosis_atp', 'diagnosis_signal',
    'diagnosis_motor_force', 'diagnosis_material_wear',
    'diagnosis_evidence_events', 'diagnosis_audit_evidence',
    'diagnosis_max_change_evidence', 'diagnosis_mean_observation_sufficiency',
    'diagnosis_pending_audits', 'diagnosis_deferred_unsafe',
    'diagnosis_deferred_no_target', 'diagnosis_deferred_low_value',
    'diagnosis_deferred_busy', 'diagnosis_cumulative_exposure',
    'diagnosis_directional_samples', 'diagnosis_mean_evidence_latency',
    'mechanism_fault_applied', 'mechanism_confirmed',
    'mechanism_conservation_active', 'mechanism_conservation_issued',
    'mechanism_conservation_feedback_events', 'mechanism_conservation_mean_scale',
    'mechanism_conservation_active_time', 'mechanism_conservation_returned_atp',
    'mechanism_conservation_returned_signal',
    'mechanism_conservation_suppressed_activity',
    'mechanism_conservation_quiescent_steps',
    'mechanism_conservation_alignment_fallbacks',
    'mechanism_conservation_selected_alignment',
    'neural_escrow_sanitation_calls', 'neural_escrow_returned_atp',
    'neural_escrow_returned_signal',
])


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE, interval=10.0):
        self.path = path
        self.interval = float(interval)
        self.last_age = -1e9
        self.session_id = 'diag-' + uuid.uuid4().hex[:10]
        self.rows = 0
        self.status = 'WAIT'

    def log(self, world, reason='periodic', force=False):
        if not force and world.age - self.last_age < self.interval:
            return False
        summary = world.summary()
        row = {key: summary.get(key, '') for key in LOG_FIELDS}
        row.update({'session_id': self.session_id, 'reason': reason, 'wall_time': time.time()})
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        with open(self.path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('session_id', 'reason', 'wall_time') + LOG_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.last_age = world.age
        self.rows += 1
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
        fields = ('session_id', 'rows', 'final_age', 'final_cells', 'diagnosis', 'audits', 'feedback')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session_id, items in sessions.items():
            last = items[-1]
            writer.writerow({
                'session_id': session_id,
                'rows': len(items),
                'final_age': last.get('age', ''),
                'final_cells': last.get('cells', ''),
                'diagnosis': last.get('diagnosis_completed', ''),
                'audits': last.get('formal_audits', ''),
                'feedback': last.get('formal_feedback_events', ''),
            })
    lines = [BUILD_LONG, 'sessions: {}'.format(len(sessions)), '']
    for session_id, items in sessions.items():
        last = items[-1]
        lines.append(
            '{} age={} cells={} diagnosis={} targeted={} audits={} feedback={} change={} ledger={}'.format(
                session_id, last.get('age', ''), last.get('cells', ''),
                last.get('diagnosis_completed', ''), last.get('diagnosis_targeted_audits', ''),
                last.get('formal_audits', ''), last.get('formal_feedback_events', ''),
                last.get('diagnosis_max_change_evidence', ''), last.get('matter_residual', ''),
            )
        )
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return 'OK'


try:
    from scene import Scene, run, LANDSCAPE, background, fill, rect, text

    class SomaCell061Scene(f06.SomaCellFormalScene):
        def setup(self):
            background(0.006, 0.012, 0.022)
            try:
                self.world = Formal061World.load(SAVE_FILE)
                self.save_status = 'LOAD'
            except Exception:
                self.world = Formal061World(
                    seed=101, initial_cells=2,
                    config=Formal061Config(p2_environment=p2.P2_ENV_CUE_REVERSAL),
                )
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

        def draw(self):
            super(SomaCell061Scene, self).draw()
            summary = self.world.summary()
            fill(0.01, 0.018, 0.03, 0.97)
            rect(0, 158, self.size.w, 58)
            fill(0.80, 1.0, 0.93)
            text(
                'DIAG {} done {} abort {} target {} evidence {:.2f}'.format(
                    summary['diagnosis_mode'], summary['diagnosis_completed'],
                    summary['diagnosis_aborted'], summary['diagnosis_targeted_audits'],
                    summary['diagnosis_max_change_evidence'],
                ),
                x=24, y=199, font_size=9, alignment=4,
            )
            text(
                'obs {:.2f} ATP {:.5f} feedback {} pending {}'.format(
                    summary['diagnosis_mean_observation_sufficiency'],
                    summary['diagnosis_atp'], summary['diagnosis_feedback_events'],
                    summary['diagnosis_pending_audits'],
                ),
                x=24, y=179, font_size=9, alignment=4,
            )

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
            seed=101, seconds=110.0, initial_cells=1,
            config=Formal061Config(p2_environment=p2.P2_ENV_CUE_REVERSAL),
        ))
    else:
        run(SomaCell061Scene(), LANDSCAPE, show_fps=False)
