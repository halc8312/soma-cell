# coding: utf-8
"""SOMA-CELL 0.6.3 — demand-driven transient neurogenesis.

The 0.6.2 audit selected no added neural tissue in the tested short task.
This release candidate therefore does not keep a permanent added brain.  A
small gene-authorised, materially paid non-neural sentinel observes physical
ligand/meaning/mechanism evidence.  Only when its estimated information value
exceeds the explicit cost of development, maintenance, and later reabsorption
may it request a temporary 2- or 4-compartment neural organ.

The organ receives a finite lease.  Its contribution is re-tested by a bounded
counterbalanced active/dormant assay.  Non-positive tissue is conservatively
returned to body chemistry.  No external reward, switch-time notification,
correct direction, free tissue, free memory inheritance, or direct position /
ATP / membrane / DNA rewrite is introduced.
"""
from __future__ import division

import csv
import gc
import math
import os
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_2_pythonista as s62

s61 = s62.s61
f06 = s62.f06
p2 = s62.p2
p1 = s62.p1
p0 = s62.p0
s5 = s62.s5
s4 = s62.s4

BUILD = 'SOMA-CELL 0.6.3'
BUILD_LONG = BUILD + ' | demand-driven transient neurogenesis'
SCHEMA_VERSION = '0.6.3-NG1.0'
SAVE_VERSION = 1
SAVE_FILE = 'soma_cell_0_6_3.pkl'
LOG_FILE = 'soma_cell_0_6_3_longrun.csv'
REPORT_FILE = 'soma_cell_0_6_3_report.txt'
SESSION_FILE = 'soma_cell_0_6_3_sessions.csv'
AUTO_SAVE_INTERVAL = 60.0
SIM_HZ = s62.SIM_HZ
clamp = s62.clamp
_atomic_pickle = s62._atomic_pickle

POLICY_DEMAND = 'demand'
POLICY_RANDOM = 'random_cost_matched'
POLICY_ALWAYS_NONE = 'always_no_tissue'
POLICY_PREPARED_NONE = 'prepared_no_tissue'
POLICY_ALWAYS_E2 = 'always_efficient2'
POLICY_ALWAYS_E4 = 'always_efficient4'
POLICY_DEMAND_NO_REABSORB = 'demand_no_reabsorption'
POLICY_DEMAND_NO_PREDICTION = 'demand_no_prediction'
POLICY_DEMAND_NO_PLASTICITY = 'demand_no_plasticity'
POLICIES = frozenset((
    POLICY_DEMAND, POLICY_RANDOM, POLICY_ALWAYS_NONE, POLICY_PREPARED_NONE, POLICY_ALWAYS_E2,
    POLICY_ALWAYS_E4, POLICY_DEMAND_NO_REABSORB,
    POLICY_DEMAND_NO_PREDICTION, POLICY_DEMAND_NO_PLASTICITY,
))
TRANSIENT_POLICIES = frozenset((
    POLICY_DEMAND, POLICY_RANDOM, POLICY_DEMAND_NO_REABSORB,
    POLICY_DEMAND_NO_PREDICTION, POLICY_DEMAND_NO_PLASTICITY,
))

PHASE_NONE = 'none'
PHASE_DEVELOPING = 'developing'
PHASE_SETTLING = 'settling'
PHASE_PROBING = 'probing'
PHASE_LEASED = 'leased'
PHASE_REABSORBING = 'reabsorbing'
PHASE_COOLDOWN = 'cooldown'

SENTINEL_ID = 'neurogenesis-sentinel'
SENTINEL_KIND = 'gene-authorised-nonneural-sentinel'

NG_TARGET_PROTEIN = 0.00320
NG_TARGET_MEMBRANE = 0.00105
NG_TARGET_SIGNAL = 0.00095
NG_MATURE_PROTEIN = 0.00235
NG_MATURE_MEMBRANE = 0.00072
NG_MATURE_SIGNAL = 0.00055
NG_ATP_STORE = 0.00115
NG_SIGNAL_STORE = 0.00018

PROBE_ACTIVE = 1
PROBE_DORMANT = 0
PROBE_SEQUENCE_ABBA = (PROBE_ACTIVE, PROBE_DORMANT, PROBE_DORMANT, PROBE_ACTIVE)
PROBE_SEQUENCE_BAAB = (PROBE_DORMANT, PROBE_ACTIVE, PROBE_ACTIVE, PROBE_DORMANT)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _counter_uniform(seed, cell_id, epoch, stream=0):
    """Stateless pseudo-random control; does not consume the world RNG."""
    x = (int(seed) * 0x9E3779B1 + int(cell_id) * 0x85EBCA77
         + int(epoch) * 0xC2B2AE3D + int(stream) * 0x27D4EB2F) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return float(x & 0xFFFFFFFF) / float(0x100000000)


def _mean_or_zero(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


class Formal063Config(s62.Formal062Config):
    """0.6.2 plus a paid transient-organ policy."""

    def __init__(
        self,
        neurogenesis_policy=POLICY_DEMAND,
        neurogenesis_enabled=True,
        neurogenesis_controller_cost=True,
        sentinel_target_protein=0.00120,
        sentinel_target_signal=0.00024,
        sentinel_target_atp=0.00045,
        sentinel_protein_precursor_reserve=0.00660,
        sentinel_membrane_precursor_reserve=0.00230,
        sentinel_signal_precursor_reserve=0.00198,
        sentinel_mature_protein=0.00088,
        sentinel_mature_signal=0.00015,
        sentinel_maintenance_atp_rate=4.0e-6,
        sentinel_wear_rate=7.0e-8,
        sentinel_update_interval=0.25,
        sentinel_baseline_age=9.0,
        sentinel_min_confidence=0.12,
        sentinel_semantic_change_threshold=0.34,
        sentinel_profile_surprise_threshold=0.30,
        sentinel_motor_loss_threshold=0.32,
        neurogenesis_min_age=5.5,
        neurogenesis_cooldown=10.0,
        neurogenesis_score_threshold=0.035,
        neurogenesis_four_cell_threshold=0.66,
        neurogenesis_safe_margin=0.18,
        neurogenesis_safe_atp=0.16,
        neurogenesis_safe_closure=0.72,
        neurogenesis_development_value_scale=0.78,
        neurogenesis_two_cell_cost_score=0.025,
        neurogenesis_four_cell_cost_score=0.18,
        neurogenesis_random_interval=8.0,
        neurogenesis_random_probability=0.32,
        organ_settle_duration=8.0,
        organ_probe_phase_duration=1.15,
        organ_probe_repetitions=2,
        organ_probe_min_quality=0.45,
        organ_positive_effect=0.00018,
        organ_lease_duration=12.0,
        organ_max_extensions=2,
        organ_reabsorb_nonpositive=True,
        organ_dormant_gate=0.0,
        organ_development_timeout=34.0,
        organ_value_cost_weight=0.35,
        organ_plasticity_warmup=3.0,
        organ_probe_freeze_learning=True,
        external_test_harness=False,
        **kwargs
    ):
        policy = str(neurogenesis_policy)
        if policy not in POLICIES:
            raise ValueError('unknown neurogenesis policy: ' + policy)
        if not bool(neurogenesis_controller_cost) and not bool(external_test_harness):
            raise ValueError('unmetered 0.6.3 sentinel is allowed only in an external test harness')

        # Static controls preserve the exact 0.6.2 profiles.  Transient policies
        # install the existing developmental genes but suppress automatic organ
        # creation until the sentinel authorises a lease.
        if policy == POLICY_ALWAYS_NONE:
            kwargs.setdefault('neural_profile', s62.PROFILE_NO_TISSUE)
        elif policy == POLICY_PREPARED_NONE:
            kwargs.setdefault('neural_profile', s62.PROFILE_EFFICIENT2)
        elif policy == POLICY_ALWAYS_E4:
            kwargs.setdefault('neural_profile', s62.PROFILE_EFFICIENT4)
        else:
            kwargs.setdefault('neural_profile', s62.PROFILE_EFFICIENT2)
        super(Formal063Config, self).__init__(external_test_harness=external_test_harness, **kwargs)

        self.neurogenesis_policy = policy
        self.neurogenesis_enabled = bool(neurogenesis_enabled) and policy not in (POLICY_ALWAYS_NONE, POLICY_PREPARED_NONE)
        self.neurogenesis_controller_cost = bool(neurogenesis_controller_cost)
        self.sentinel_target_protein = float(sentinel_target_protein)
        self.sentinel_target_signal = float(sentinel_target_signal)
        self.sentinel_target_atp = float(sentinel_target_atp)
        self.sentinel_protein_precursor_reserve = max(0.0, float(sentinel_protein_precursor_reserve))
        self.sentinel_membrane_precursor_reserve = max(0.0, float(sentinel_membrane_precursor_reserve))
        self.sentinel_signal_precursor_reserve = max(0.0, float(sentinel_signal_precursor_reserve))
        self.sentinel_mature_protein = float(sentinel_mature_protein)
        self.sentinel_mature_signal = float(sentinel_mature_signal)
        self.sentinel_maintenance_atp_rate = float(sentinel_maintenance_atp_rate)
        self.sentinel_wear_rate = float(sentinel_wear_rate)
        self.sentinel_update_interval = max(0.05, float(sentinel_update_interval))
        self.sentinel_baseline_age = float(sentinel_baseline_age)
        self.sentinel_min_confidence = float(sentinel_min_confidence)
        self.sentinel_semantic_change_threshold = float(sentinel_semantic_change_threshold)
        self.sentinel_profile_surprise_threshold = float(sentinel_profile_surprise_threshold)
        self.sentinel_motor_loss_threshold = float(sentinel_motor_loss_threshold)
        self.neurogenesis_min_age = float(neurogenesis_min_age)
        self.neurogenesis_cooldown = float(neurogenesis_cooldown)
        self.neurogenesis_score_threshold = float(neurogenesis_score_threshold)
        self.neurogenesis_four_cell_threshold = float(neurogenesis_four_cell_threshold)
        self.neurogenesis_safe_margin = float(neurogenesis_safe_margin)
        self.neurogenesis_safe_atp = float(neurogenesis_safe_atp)
        self.neurogenesis_safe_closure = float(neurogenesis_safe_closure)
        self.neurogenesis_development_value_scale = float(neurogenesis_development_value_scale)
        self.neurogenesis_two_cell_cost_score = float(neurogenesis_two_cell_cost_score)
        self.neurogenesis_four_cell_cost_score = float(neurogenesis_four_cell_cost_score)
        self.neurogenesis_random_interval = max(0.5, float(neurogenesis_random_interval))
        self.neurogenesis_random_probability = clamp(float(neurogenesis_random_probability), 0.0, 1.0)
        self.organ_settle_duration = max(0.0, float(organ_settle_duration))
        self.organ_probe_phase_duration = max(0.20, float(organ_probe_phase_duration))
        self.organ_probe_repetitions = max(1, int(organ_probe_repetitions))
        self.organ_probe_min_quality = clamp(float(organ_probe_min_quality), 0.0, 1.0)
        self.organ_positive_effect = float(organ_positive_effect)
        self.organ_lease_duration = max(1.0, float(organ_lease_duration))
        self.organ_max_extensions = max(0, int(organ_max_extensions))
        self.organ_reabsorb_nonpositive = bool(organ_reabsorb_nonpositive)
        self.organ_dormant_gate = clamp(float(organ_dormant_gate), 0.0, 1.0)
        self.organ_development_timeout = max(2.0, float(organ_development_timeout))
        self.organ_value_cost_weight = max(0.0, float(organ_value_cost_weight))
        self.organ_plasticity_warmup = max(0.0, float(organ_plasticity_warmup))
        self.organ_probe_freeze_learning = bool(organ_probe_freeze_learning)
        self.external_test_harness = bool(external_test_harness)

        # Transient profiles use the genes but must not be interpreted as a
        # request for a permanently installed profile.
        if policy in TRANSIENT_POLICIES:
            self.neural_profile = s62.PROFILE_EFFICIENT2
            self.audit_active_cells = 2
            self.p2_tissue_mode = p2.P2_MODE_FULL
            self.p2_install_genes = True
            self.diagnosis_mode = s61.DIAGNOSIS_OFF
            self.diagnosis_enabled = False
            self.p2_recurrence = False
            self.p2_prediction = policy != POLICY_DEMAND_NO_PREDICTION
            self.p2_plasticity = policy != POLICY_DEMAND_NO_PLASTICITY
            self.p2_plasticity_warmup = self.organ_plasticity_warmup

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        state.pop('audit_active_cells', None)
        allowed = set(cls().__dict__.keys())
        allowed.discard('audit_active_cells')
        return cls(**{key: value for key, value in state.items() if key in allowed})


class OrganProbe(object):
    """Counterbalanced active/dormant tissue-value assay."""

    def __init__(self):
        self.active = False
        self.sequence = PROBE_SEQUENCE_ABBA
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self.repetitions_done = 0
        self.margin_sum = {PROBE_ACTIVE: 0.0, PROBE_DORMANT: 0.0}
        self.time_sum = {PROBE_ACTIVE: 0.0, PROBE_DORMANT: 0.0}
        self.atp_start = 0.0
        self.material_start = 0.0
        self.last_effect = 0.0
        self.last_quality = 0.0
        self.completed = 0
        self.positive = 0
        self.nonpositive = 0

    def start(self, seed, cell_id, epoch, tissue):
        self.active = True
        self.sequence = PROBE_SEQUENCE_ABBA if _counter_uniform(seed, cell_id, epoch, 31) < 0.5 else PROBE_SEQUENCE_BAAB
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self.repetitions_done = 0
        self.margin_sum = {PROBE_ACTIVE: 0.0, PROBE_DORMANT: 0.0}
        self.time_sum = {PROBE_ACTIVE: 0.0, PROBE_DORMANT: 0.0}
        self.atp_start = float(tissue.module_ledger.total_atp())
        self.material_start = float(tissue.module_ledger.total_material())
        tissue.organ_gate = float(self.sequence[0])
        tissue.probe_freeze_learning = True

    def current_gate(self):
        if not self.active:
            return 1.0
        return float(self.sequence[self.phase_index])

    def record(self, tissue, margin, dt, config):
        if not self.active:
            return None
        state = int(self.sequence[self.phase_index])
        self.margin_sum[state] += float(margin) * dt
        self.time_sum[state] += dt
        self.phase_elapsed += dt
        if self.phase_elapsed + 1e-12 < config.organ_probe_phase_duration:
            return None
        self.phase_elapsed = 0.0
        self.phase_index += 1
        if self.phase_index < len(self.sequence):
            tissue.organ_gate = float(self.sequence[self.phase_index])
            return None
        self.repetitions_done += 1
        if self.repetitions_done < config.organ_probe_repetitions:
            self.phase_index = 0
            self.sequence = tuple(reversed(self.sequence))
            tissue.organ_gate = float(self.sequence[0])
            tissue.probe_freeze_learning = True
            return None

        active_mean = self.margin_sum[PROBE_ACTIVE] / max(self.time_sum[PROBE_ACTIVE], 1e-12)
        dormant_mean = self.margin_sum[PROBE_DORMANT] / max(self.time_sum[PROBE_DORMANT], 1e-12)
        atp_delta = max(0.0, float(tissue.module_ledger.total_atp()) - self.atp_start)
        material_delta = max(0.0, float(tissue.module_ledger.total_material()) - self.material_start)
        duration = max(self.time_sum[PROBE_ACTIVE] + self.time_sum[PROBE_DORMANT], 1e-12)
        explicit_cost = (atp_delta + 0.75 * material_delta) / duration
        effect = active_mean - dormant_mean - config.organ_value_cost_weight * explicit_cost
        balance = min(self.time_sum.values()) / max(max(self.time_sum.values()), 1e-12)
        quality = clamp(balance * min(1.0, duration / (4.0 * config.organ_probe_phase_duration)), 0.0, 1.0)
        self.last_effect = float(effect)
        self.last_quality = float(quality)
        self.completed += 1
        if quality >= config.organ_probe_min_quality and effect > config.organ_positive_effect:
            self.positive += 1
        else:
            self.nonpositive += 1
        self.active = False
        tissue.organ_gate = 1.0
        tissue.probe_freeze_learning = False
        return {'effect': float(effect), 'quality': float(quality), 'positive': bool(quality >= config.organ_probe_min_quality and effect > config.organ_positive_effect)}

    def state_dict(self):
        return {
            'active': self.active, 'sequence': tuple(self.sequence),
            'phase_index': self.phase_index, 'phase_elapsed': self.phase_elapsed,
            'repetitions_done': self.repetitions_done,
            'margin_sum': dict(self.margin_sum), 'time_sum': dict(self.time_sum),
            'atp_start': self.atp_start, 'material_start': self.material_start,
            'last_effect': self.last_effect, 'last_quality': self.last_quality,
            'completed': self.completed, 'positive': self.positive,
            'nonpositive': self.nonpositive,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls()
        state = dict(state or {})
        obj.active = bool(state.get('active', False))
        obj.sequence = tuple(int(v) for v in state.get('sequence', PROBE_SEQUENCE_ABBA))
        obj.phase_index = int(state.get('phase_index', 0))
        obj.phase_elapsed = float(state.get('phase_elapsed', 0.0))
        obj.repetitions_done = int(state.get('repetitions_done', 0))
        obj.margin_sum = {int(k): float(v) for k, v in state.get('margin_sum', {0: 0.0, 1: 0.0}).items()}
        obj.time_sum = {int(k): float(v) for k, v in state.get('time_sum', {0: 0.0, 1: 0.0}).items()}
        for key in (PROBE_ACTIVE, PROBE_DORMANT):
            obj.margin_sum.setdefault(key, 0.0)
            obj.time_sum.setdefault(key, 0.0)
        obj.atp_start = float(state.get('atp_start', 0.0))
        obj.material_start = float(state.get('material_start', 0.0))
        obj.last_effect = float(state.get('last_effect', 0.0))
        obj.last_quality = float(state.get('last_quality', 0.0))
        obj.completed = int(state.get('completed', 0))
        obj.positive = int(state.get('positive', 0))
        obj.nonpositive = int(state.get('nonpositive', 0))
        return obj


class NeurogenesisState(object):
    def __init__(self, policy=POLICY_DEMAND):
        self.policy = str(policy)
        self.phase = PHASE_NONE
        self.target_profile = s62.PROFILE_EFFICIENT2
        self.active_profile = s62.PROFILE_NO_TISSUE
        self.sentinel_ready = False
        self.sentinel_atp_spent = 0.0
        self.sentinel_material_wear = 0.0
        self.last_update_age = -1e9
        self.previous_profile = None
        self.profile_mean = None
        self.profile_var = 0.0
        self.previous_meaning = None
        self.baseline_meaning = None
        self.baseline_confidence = None
        self.baseline_motor_gain = 0.0
        self.baseline_motor_weight = 0.0
        self.recent_motor_gain = 0.0
        self.baseline_margin = 0.0
        self.baseline_margin_weight = 0.0
        self.body_decline = 0.0
        self.baseline_atp = 0.0
        self.baseline_atp_weight = 0.0
        self.energy_decline = 0.0
        self.novelty = 0.0
        self.information_deficit = 0.0
        self.semantic_change = 0.0
        self.mechanism_change = 0.0
        self.demand_score = 0.0
        self.estimated_value = 0.0
        self.estimated_cost = 0.0
        self.net_value = 0.0
        self.last_trigger_age = -1e9
        self.cooldown_until = 0.0
        self.development_start_age = -1.0
        self.mature_age = -1.0
        self.settle_until = 0.0
        self.lease_until = 0.0
        self.extensions = 0
        self.developments = 0
        self.reabsorptions = 0
        self.development_timeouts = 0
        self.random_epochs = 0
        self.trigger_attempts = 0
        self.trigger_denied_risk = 0
        self.trigger_denied_value = 0
        self.returned_material = 0.0
        self.returned_atp = 0.0
        self.archived_organ_atp = 0.0
        self.archived_organ_material = 0.0
        self.archived_prediction_atp = 0.0
        self.archived_recurrence_atp = 0.0
        self.archived_plasticity_atp = 0.0
        self.precursor_protein_reserved = 0.0
        self.precursor_membrane_reserved = 0.0
        self.precursor_signal_reserved = 0.0
        self.precursor_protein_released = 0.0
        self.precursor_membrane_released = 0.0
        self.precursor_signal_released = 0.0
        self.precursor_protein_used = 0.0
        self.precursor_membrane_used = 0.0
        self.precursor_signal_used = 0.0
        self.precursor_membrane_recovered = 0.0
        self.development_membrane_credit = 0.0
        self.development_membrane_body_before = 0.0
        self.probe = OrganProbe()

    def finite(self):
        values = [
            self.sentinel_atp_spent, self.sentinel_material_wear,
            self.profile_var, self.baseline_motor_gain, self.baseline_motor_weight,
            self.recent_motor_gain, self.baseline_margin, self.baseline_margin_weight,
            self.body_decline, self.baseline_atp, self.baseline_atp_weight,
            self.energy_decline, self.novelty, self.information_deficit,
            self.semantic_change, self.mechanism_change, self.demand_score,
            self.estimated_value, self.estimated_cost, self.net_value,
            self.returned_material, self.returned_atp,
            self.archived_organ_atp, self.archived_organ_material,
            self.archived_prediction_atp, self.archived_recurrence_atp,
            self.archived_plasticity_atp,
            self.precursor_protein_reserved, self.precursor_membrane_reserved,
            self.precursor_signal_reserved, self.precursor_protein_released,
            self.precursor_membrane_released, self.precursor_signal_released,
            self.precursor_protein_used, self.precursor_membrane_used,
            self.precursor_signal_used, self.precursor_membrane_recovered,
            self.development_membrane_credit, self.development_membrane_body_before,
        ]
        return bool(np.all(np.isfinite(values)))

    def state_dict(self):
        return {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in self.__dict__.items() if key != 'probe'
        } | {'probe': self.probe.state_dict()}

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        obj = cls(state.get('policy', POLICY_DEMAND))
        for key, value in state.items():
            if key == 'probe':
                continue
            if key in ('previous_profile', 'profile_mean', 'previous_meaning', 'baseline_meaning', 'baseline_confidence') and value is not None:
                value = np.asarray(value, dtype=float).copy()
            setattr(obj, key, value)
        obj.probe = OrganProbe.from_state(state.get('probe', {}))
        return obj


class TransientNeuralTissue(s62.MetabolicAuditTissue):
    """A 0.6.2 tissue with a reversible whole-organ activity gate."""

    def __init__(self, gene_parameters, mode=p2.P2_MODE_FULL, rng_seed=0,
                 formal_enabled=False, profile=s62.PROFILE_EFFICIENT2):
        super(TransientNeuralTissue, self).__init__(
            gene_parameters, mode=mode, rng_seed=rng_seed,
            formal_enabled=formal_enabled, profile=profile,
        )
        self.organ_gate = 1.0
        self.organ_gate_steps = 0
        self.organ_dormant_steps = 0
        self.development_only = False
        self.probe_freeze_learning = False
        self.development_membrane_credit = 0.0
        self.development_membrane_body_before = 0.0
        self.development_membrane_used = 0.0

    def _develop_cell(self, port, index, dt, costs=True):
        i = int(index)
        status = self._status(port, i)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        protein_need = max(0.0, NG_TARGET_PROTEIN - float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]))
        membrane_need = max(0.0, NG_TARGET_MEMBRANE - float(tissue[p0.TISSUE_MEMBRANE]))
        signal_total = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        # Precursor signal transferred by the sentinel must be assembled before
        # active neural work can consume it; otherwise a just-born organ can
        # drain its own developmental stock and never become mature.
        signal_need = max(0.0, NG_TARGET_SIGNAL - float(tissue[p0.TISSUE_SIGNAL]))
        if protein_need + membrane_need + signal_need > 1e-12:
            membrane_request = min(membrane_need, 0.00017)
            credit = max(0.0, float(self.development_membrane_credit))
            if credit > 1e-14 and membrane_need > 0.0:
                membrane_request = min(membrane_need, credit)
            request = {
                'atp': 0.00078 if costs else 0.0,
                'protein': min(protein_need, 0.00042),
                'membrane': membrane_request,
                'signal': min(signal_need, 0.00017),
            }
            config = port._world.config
            old_rate = float(config.neural_membrane_rate)
            old_reserve = float(config.neural_membrane_reserve)
            if credit > 1e-14 and membrane_request > 0.0:
                config.neural_membrane_rate = max(old_rate, membrane_request / max(float(dt), 1e-12))
                config.neural_membrane_reserve = min(
                    max(0.0, float(self.development_membrane_body_before)),
                    max(0.0, old_reserve),
                )
            try:
                granted = port.allocate_budget(self.tissue_ids[i], request, dt)
            finally:
                config.neural_membrane_rate = old_rate
                config.neural_membrane_reserve = old_reserve
            membrane_granted = max(0.0, float(granted.get('membrane', 0.0)))
            if credit > 1e-14:
                self.development_membrane_credit = max(0.0, credit - membrane_granted)
                self.development_membrane_used += membrane_granted
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
            functional / NG_TARGET_PROTEIN,
            membrane / NG_TARGET_MEMBRANE,
            signal / NG_TARGET_SIGNAL,
        ], dtype=float)
        self.development[i] = float(clamp(np.min(ratios), 0.0, 1.0))
        self.mature[i] = bool(
            functional >= NG_MATURE_PROTEIN
            and membrane >= NG_MATURE_MEMBRANE
            and signal >= NG_MATURE_SIGNAL
        )
        return status

    def _material_capacity(self):
        functional = self.material_status[:, p0.TISSUE_FUNCTIONAL_PROTEIN]
        membrane = self.material_status[:, p0.TISSUE_MEMBRANE]
        signal = self.material_status[:, p0.TISSUE_SIGNAL] + self.store_status[:, p0.BUDGET_SIGNAL]
        capacity = np.minimum.reduce((
            functional / max(NG_TARGET_PROTEIN, 1e-12),
            membrane / max(NG_TARGET_MEMBRANE, 1e-12),
            signal / max(NG_TARGET_SIGNAL, 1e-12),
        ))
        return np.clip(capacity, 0.0, 1.0) * self.mature.astype(float)

    def _maintenance_cell(self, port, index, dt, config):
        i = int(index)
        status = self._status(port, i)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        if config.p2_tissue_turnover and (
            float(tissue[p0.TISSUE_DAMAGED_PROTEIN]) >= p2.P2_TURNOVER_DAMAGE
            or float(tissue[p0.TISSUE_AGGREGATE]) >= p2.P2_TURNOVER_AGGREGATE
        ):
            self._turnover_cell(port, i, '0.6.3-transient-neural-turnover')
            return False
        stores = np.asarray(status['stores'], dtype=float)
        base_wear = 0.0000080 * dt
        activity_wear = 0.0000100 * dt * (
            abs(float(self.hidden[i]))
            + 0.40 * float(self.signal_refractory[i])
            + 0.30 * abs(float(self.last_local_modulator[i]))
        )
        wear_request = base_wear + activity_wear + float(self.pending_wear[i])
        request = {
            'atp': max(0.0, NG_ATP_STORE - float(stores[p0.BUDGET_ATP])),
            'signal': max(0.0, NG_SIGNAL_STORE - float(stores[p0.BUDGET_SIGNAL])),
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
                self.cumulative_wear_material += float(
                    built['damaged_protein'] + built['aggregate']
                )
        self.pending_wear[i] = 0.0
        return True

    def pre_step(self, port, dt, config, gene_activity):
        if bool(self.development_only):
            dt = clamp(float(dt), 1.0 / 240.0, 0.10)
            for i in range(p2.P2_CELL_COUNT):
                self.cooldown[i] = max(0.0, self.cooldown[i] - dt)
            self.ensure_attachments(port, gene_activity)
            for i in range(p2.P2_CELL_COUNT):
                if self.cooldown[i] > 0.0:
                    continue
                try:
                    self._develop_cell(port, i, dt, costs=config.p2_neural_cost)
                except KeyError:
                    self.present[i] = False
                    self.mature[i] = False
            self.last_debt_valid = False
            return {
                'status': 'developing',
                'mature_cells': int(np.count_nonzero(self.mature)),
            }
        gate = clamp(float(self.organ_gate), 0.0, 1.0)
        if gate >= 1.0 - 1e-12:
            return super(TransientNeuralTissue, self).pre_step(port, dt, config, gene_activity)
        names = ('w_sensor', 'w_rec', 'bias', 'motor_gain', 'resource_lease',
                 'hidden', 'prev_hidden')
        backup = {name: getattr(self, name).copy() for name in names}
        flags = (config.p2_prediction, config.p2_recurrence, config.p2_plasticity,
                 config.diagnosis_enabled, config.diagnosis_mode)
        try:
            self.w_sensor *= gate
            self.w_rec *= gate
            self.bias *= gate
            self.motor_gain *= gate
            self.resource_lease *= gate
            self.hidden *= gate
            self.prev_hidden *= gate
            config.p2_prediction = False
            config.p2_recurrence = False
            config.p2_plasticity = False
            config.diagnosis_enabled = False
            config.diagnosis_mode = s61.DIAGNOSIS_OFF
            report = super(TransientNeuralTissue, self).pre_step(port, dt, config, gene_activity)
        finally:
            for name, value in backup.items():
                setattr(self, name, value)
            (config.p2_prediction, config.p2_recurrence, config.p2_plasticity,
             config.diagnosis_enabled, config.diagnosis_mode) = flags
        self.organ_gate_steps += 1
        self.organ_dormant_steps += 1
        return report

    def post_step(self, port, dt, config):
        # During an organ-value probe, inference and its physical cost remain
        # active, but predictor/plasticity updates are frozen in both A and B
        # phases.  This prevents the active phase from becoming a different
        # learned organ while the dormant phase is measured.
        if not bool(self.probe_freeze_learning):
            return super(TransientNeuralTissue, self).post_step(port, dt, config)
        flags = (config.p2_prediction, config.p2_plasticity)
        try:
            config.p2_prediction = False
            config.p2_plasticity = False
            return super(TransientNeuralTissue, self).post_step(port, dt, config)
        finally:
            config.p2_prediction, config.p2_plasticity = flags

    def state_dict(self):
        state = super(TransientNeuralTissue, self).state_dict()
        state.update({
            'organ_gate': self.organ_gate,
            'organ_gate_steps': self.organ_gate_steps,
            'organ_dormant_steps': self.organ_dormant_steps,
            'development_only': self.development_only,
            'probe_freeze_learning': self.probe_freeze_learning,
            'development_membrane_credit': self.development_membrane_credit,
            'development_membrane_body_before': self.development_membrane_body_before,
            'development_membrane_used': self.development_membrane_used,
        })
        return state

    @classmethod
    def from_state(cls, state):
        obj = s62.MetabolicAuditTissue.from_state(state)
        obj.__class__ = cls
        obj.organ_gate = float(state.get('organ_gate', 1.0))
        obj.organ_gate_steps = int(state.get('organ_gate_steps', 0))
        obj.organ_dormant_steps = int(state.get('organ_dormant_steps', 0))
        obj.development_only = bool(state.get('development_only', False))
        obj.probe_freeze_learning = bool(state.get('probe_freeze_learning', False))
        obj.development_membrane_credit = float(state.get('development_membrane_credit', 0.0))
        obj.development_membrane_body_before = float(state.get('development_membrane_body_before', 0.0))
        obj.development_membrane_used = float(state.get('development_membrane_used', 0.0))
        return obj


class Formal063ProtoCell(s62.Formal062ProtoCell):
    @classmethod
    def from_state(cls, rng, state):
        cell = s62.Formal062ProtoCell.from_state(rng, state)
        cell.__class__ = cls
        if state.get('p2_tissue') is not None:
            cell.p2_tissue = TransientNeuralTissue.from_state(state['p2_tissue'])
        return cell


class Formal063World(s62.Formal062World):
    def __init__(self, seed=101, initial_cells=1, config=None):
        config = config if config is not None else Formal063Config()
        if not isinstance(config, Formal063Config):
            config = Formal063Config.from_state(config.state_dict())
        self._ng_config_during_init = config
        self._ng_initialising = True
        self.neurogenesis_states = {}
        self.neurogenesis_world_steps = 0
        self.neurogenesis_developments = 0
        self.neurogenesis_reabsorptions = 0
        self.neurogenesis_extensions = 0
        self.neurogenesis_false_stable_triggers = 0
        self.neurogenesis_tissue_active_time = 0.0
        self.neurogenesis_tissue_development_time = 0.0
        self.neurogenesis_sentinel_cost_atp = 0.0
        self.neurogenesis_sentinel_wear = 0.0
        super(Formal063World, self).__init__(seed=seed, initial_cells=initial_cells, config=config)
        self.config = config
        self._ng_initialising = False
        for cell in self.cells:
            cell.__class__ = Formal063ProtoCell
            self._state_for(cell)
            if cell.p2_tissue is not None and not isinstance(cell.p2_tissue, TransientNeuralTissue):
                cell.p2_tissue = TransientNeuralTissue.from_state(cell.p2_tissue.state_dict())
        self._apply_static_policy_at_start()
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def _policy(self):
        return str(self.config.neurogenesis_policy)

    def _state_for(self, cell):
        key = str(int(cell.cell_id))
        state = self.neurogenesis_states.get(key)
        if state is None:
            state = NeurogenesisState(self._policy())
            self.neurogenesis_states[key] = state
        return state

    def _apply_static_policy_at_start(self):
        policy = self._policy()
        if policy in (POLICY_ALWAYS_E2, POLICY_ALWAYS_E4):
            profile = s62.PROFILE_EFFICIENT2 if policy == POLICY_ALWAYS_E2 else s62.PROFILE_EFFICIENT4
            for cell in self.living_cells():
                state = self._state_for(cell)
                state.phase = PHASE_DEVELOPING
                state.target_profile = profile
                state.active_profile = profile
            self._ensure_all_p2_tissues()
        elif policy in (POLICY_ALWAYS_NONE, POLICY_PREPARED_NONE):
            for cell in self.living_cells():
                self._state_for(cell).phase = PHASE_NONE

    def _transient_policy(self):
        return self._policy() in TRANSIENT_POLICIES

    def _new_p2_tissue(self, cell):
        config = getattr(self, '_ng_config_during_init', self.config)
        policy = str(config.neurogenesis_policy)
        if policy in (POLICY_ALWAYS_NONE, POLICY_PREPARED_NONE):
            return None
        if policy in TRANSIENT_POLICIES:
            if getattr(self, '_ng_initialising', False):
                return None
            state = self._state_for(cell)
            if state.phase not in (PHASE_DEVELOPING, PHASE_SETTLING, PHASE_PROBING, PHASE_LEASED):
                return None
            profile = state.target_profile
        elif policy == POLICY_ALWAYS_E4:
            profile = s62.PROFILE_EFFICIENT4
        else:
            profile = s62.PROFILE_EFFICIENT2

        params = p2.p2_gene_parameters(cell)
        activity = p2.p2_gene_activity(cell)
        if params is None:
            return None
        count = s62.profile_active_count(profile)
        mask = np.zeros(p2.P2_CELL_COUNT, dtype=bool)
        mask[s62.even_active_indices(count)] = True
        masked_activity = np.asarray(activity, dtype=float).copy()
        masked_activity[~mask] = 0.0
        if np.count_nonzero(masked_activity >= 0.016) < max(1, np.count_nonzero(mask)):
            return None
        tissue_seed = ((self.p2_seed * 1000003) ^ (int(cell.cell_id) * 9176)
                       ^ (int(cell.generation) * 7919) ^ 0x063D1A) & 0xFFFFFFFF
        tissue = TransientNeuralTissue(
            params, mode=config.p2_tissue_mode, rng_seed=tissue_seed,
            formal_enabled=False, profile=profile,
        )
        tissue.ensure_attachments(self.port_for(cell.cell_id), masked_activity)
        cell.p2_tissue = tissue
        cell.p2_tissue_births += 1
        self.p2_tissue_creations += 1
        self.formal_tissue_creations += 1
        self.neurogenesis_developments += 1
        state = self._state_for(cell)
        state.developments += 1
        state.development_start_age = float(self.age)
        state.active_profile = profile
        tissue.development_membrane_credit = float(state.development_membrane_credit)
        tissue.development_membrane_body_before = float(state.development_membrane_body_before)
        if policy in TRANSIENT_POLICIES and state.phase == PHASE_DEVELOPING:
            tissue.organ_gate = 0.0
            tissue.development_only = True
        return tissue

    def _ensure_p2_tissue(self, cell):
        if not cell.alive:
            return None
        policy = self._policy()
        if policy in (POLICY_ALWAYS_NONE, POLICY_PREPARED_NONE):
            return None
        if cell.p2_tissue is not None:
            return cell.p2_tissue
        if policy in TRANSIENT_POLICIES:
            state = self._state_for(cell)
            if state.phase not in (PHASE_DEVELOPING, PHASE_SETTLING, PHASE_PROBING, PHASE_LEASED):
                return None
        return self._new_p2_tissue(cell)

    def _ensure_all_p2_tissues(self):
        for cell in self.living_cells():
            if not isinstance(cell, Formal063ProtoCell):
                cell.__class__ = Formal063ProtoCell
            tissue = getattr(cell, 'p2_tissue', None)
            if tissue is not None and not isinstance(tissue, TransientNeuralTissue):
                cell.p2_tissue = TransientNeuralTissue.from_state(tissue.state_dict())
            self._state_for(cell)
            self._ensure_p2_tissue(cell)

    def _sentinel_status(self, port):
        try:
            return port.attachment_status(SENTINEL_ID)
        except KeyError:
            return None

    def _ensure_sentinel(self, cell, state, dt):
        if self._policy() in (POLICY_ALWAYS_NONE, POLICY_ALWAYS_E2, POLICY_ALWAYS_E4):
            return False
        if f06.formal_controller_activity(cell) < 0.016:
            return False
        port = self.port_for(cell.cell_id)
        status = self._sentinel_status(port)
        if status is None:
            port.attach(SENTINEL_ID, kind=SENTINEL_KIND)
            status = self._sentinel_status(port)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        protein_need = max(0.0, self.config.sentinel_target_protein - float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]))
        signal_total = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        signal_need = max(0.0, self.config.sentinel_target_signal - signal_total)
        atp_need = max(0.0, self.config.sentinel_target_atp - float(stores[p0.BUDGET_ATP]))
        protein_reserve_need = max(
            0.0,
            self.config.sentinel_protein_precursor_reserve
            - float(stores[p0.BUDGET_PROTEIN]),
        )
        membrane_reserve_need = max(
            0.0,
            self.config.sentinel_membrane_precursor_reserve
            - float(stores[p0.BUDGET_MEMBRANE]),
        )
        signal_reserve_need = max(
            0.0,
            self.config.sentinel_signal_precursor_reserve
            - max(0.0, float(stores[p0.BUDGET_SIGNAL]) - signal_need),
        )
        wear = max(0.0, self.config.sentinel_wear_rate * dt)
        assembly_atp = (protein_need * p0.ASSEMBLY_ATP_PER_PROTEIN
                        + signal_need * p0.ASSEMBLY_ATP_PER_SIGNAL
                        + wear * p0.ASSEMBLY_ATP_PER_PROTEIN)
        request = {
            'atp': min(atp_need + assembly_atp + self.config.sentinel_maintenance_atp_rate * dt, 0.0015),
            'protein': min(protein_need + wear + protein_reserve_need, 0.00240),
            'membrane': min(membrane_reserve_need, 0.00120),
            'signal': min(signal_need + signal_reserve_need, 0.00120),
        }
        if any(value > 1e-14 for value in request.values()):
            port.allocate_budget(SENTINEL_ID, request, dt)
        status = self._sentinel_status(port)
        stores = np.asarray(status['stores'], dtype=float)
        built = port.commit_material(
            SENTINEL_ID,
            protein=min(protein_need + wear, float(stores[p0.BUDGET_PROTEIN])),
            signal=min(signal_need, float(stores[p0.BUDGET_SIGNAL])),
            damaged_fraction=0.65 if wear > protein_need else 0.0,
            aggregate_fraction=0.10 if wear > protein_need else 0.0,
        )
        port_state = port._attachment(SENTINEL_ID)
        paid, _ = port._spend_energy_and_signal(
            port_state, self.config.sentinel_maintenance_atp_rate * dt, 0.0,
        )
        spent = float(paid) + float(built.get('atp_spent', 0.0))
        damaged = float(built.get('damaged_protein', 0.0) + built.get('aggregate', 0.0))
        state.sentinel_atp_spent += spent
        state.sentinel_material_wear += damaged
        self.neurogenesis_sentinel_cost_atp += spent
        self.neurogenesis_sentinel_wear += damaged
        status = self._sentinel_status(port)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        state.sentinel_ready = bool(
            float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]) >= self.config.sentinel_mature_protein
            and float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL]) >= self.config.sentinel_mature_signal
            and float(stores[p0.BUDGET_PROTEIN]) >= 0.95 * self.config.sentinel_protein_precursor_reserve
            and float(stores[p0.BUDGET_MEMBRANE]) >= 0.95 * self.config.sentinel_membrane_precursor_reserve
            and max(0.0, float(stores[p0.BUDGET_SIGNAL])) >= 0.95 * self.config.sentinel_signal_precursor_reserve
        )
        state.precursor_protein_reserved = float(stores[p0.BUDGET_PROTEIN])
        state.precursor_membrane_reserved = float(stores[p0.BUDGET_MEMBRANE])
        state.precursor_signal_reserved = float(stores[p0.BUDGET_SIGNAL])
        return state.sentinel_ready

    def _physical_profile(self, cell, port):
        snap = port.raw_sensor_fluxes(SENTINEL_ID)
        profiles = np.asarray(snap['external']['ligand_profiles'], dtype=float)
        concentrations = np.mean(np.maximum(profiles, 0.0), axis=1)
        return profiles, concentrations, snap

    def _update_sentinel_evidence(self, cell, state, dt):
        if not state.sentinel_ready:
            return
        if self.age - state.last_update_age + 1e-12 < self.config.sentinel_update_interval:
            return
        elapsed = max(dt, self.age - state.last_update_age if state.last_update_age > -1e8 else dt)
        state.last_update_age = float(self.age)
        port = self.port_for(cell.cell_id)
        profiles, concentrations, snap = self._physical_profile(cell, port)
        if state.profile_mean is None:
            state.profile_mean = profiles.copy()
            state.previous_profile = profiles.copy()
            state.previous_meaning = np.asarray(cell.meaning_mean, dtype=float).copy()
            state.baseline_meaning = np.asarray(cell.meaning_mean, dtype=float).copy()
            state.baseline_confidence = np.asarray(cell.meaning_confidence, dtype=float).copy()
            return

        delta = float(np.sqrt(np.mean((profiles - state.profile_mean) ** 2)))
        alpha = 1.0 - math.exp(-0.11 * elapsed)
        state.profile_mean += alpha * (profiles - state.profile_mean)
        state.profile_var += alpha * (delta * delta - state.profile_var)
        scale = math.sqrt(max(state.profile_var, 1e-8)) + 0.006
        raw_novelty = clamp(delta / (4.0 * scale), 0.0, 1.0)
        state.novelty += (1.0 - math.exp(-0.35 * elapsed)) * (raw_novelty - state.novelty)

        confidence = np.asarray(cell.meaning_confidence, dtype=float)
        meaning = np.asarray(cell.meaning_mean, dtype=float)
        present = concentrations > 0.004
        if np.any(present):
            deficit = float(np.mean((1.0 - np.clip(confidence[present], 0.0, 1.0))))
        else:
            deficit = 0.0
        state.information_deficit += (1.0 - math.exp(-0.18 * elapsed)) * (deficit - state.information_deficit)

        if self.age >= self.config.sentinel_baseline_age:
            ready = confidence >= self.config.sentinel_min_confidence
            if state.baseline_meaning is None:
                state.baseline_meaning = meaning.copy()
                state.baseline_confidence = confidence.copy()
            else:
                baseline = np.asarray(state.baseline_meaning, dtype=float)
                base_conf = np.asarray(state.baseline_confidence, dtype=float)
                sign_flip = (np.sign(meaning) != np.sign(baseline)) & ready & (base_conf >= self.config.sentinel_min_confidence)
                magnitude = np.abs(meaning - baseline)
                semantic = float(np.max(np.where(sign_flip, magnitude, 0.0))) if meaning.size else 0.0
                if semantic <= 0.0:
                    semantic = float(np.max(magnitude * np.minimum(confidence, base_conf))) if meaning.size else 0.0
                state.semantic_change += (1.0 - math.exp(-0.32 * elapsed)) * (clamp(semantic, 0.0, 1.0) - state.semantic_change)
                stable = (state.novelty < 0.18 and state.semantic_change < 0.16)
                if stable:
                    beta = 1.0 - math.exp(-0.012 * elapsed)
                    state.baseline_meaning += beta * (meaning - state.baseline_meaning)
                    state.baseline_confidence += beta * (confidence - state.baseline_confidence)

        command = np.asarray(getattr(cell, 'last_motor_command', np.zeros(2)), dtype=float)
        command_mag = float(np.linalg.norm(command))
        force = max(0.0, float(getattr(cell, 'last_motor_force', 0.0)))
        if command_mag >= 0.025:
            gain = force / max(command_mag, 1e-9)
            if state.baseline_motor_weight < 1.0 or state.semantic_change < 0.18:
                beta = 1.0 - math.exp(-0.035 * elapsed)
                if state.baseline_motor_weight <= 1e-9:
                    state.baseline_motor_gain = gain
                else:
                    state.baseline_motor_gain += beta * (gain - state.baseline_motor_gain)
                state.baseline_motor_weight += elapsed
            gamma = 1.0 - math.exp(-0.22 * elapsed)
            if state.recent_motor_gain <= 1e-12:
                state.recent_motor_gain = gain
            else:
                state.recent_motor_gain += gamma * (gain - state.recent_motor_gain)
        if state.baseline_motor_weight >= 2.0 and state.baseline_motor_gain > 1e-9:
            loss = 1.0 - state.recent_motor_gain / state.baseline_motor_gain
            state.mechanism_change += (1.0 - math.exp(-0.25 * elapsed)) * (clamp(loss, 0.0, 1.0) - state.mechanism_change)

        current_margin = float(cell.autopoietic_margin())
        current_atp = float(cell.pools[s5.POOL_ATP])
        if state.baseline_margin_weight <= 1e-9:
            state.baseline_margin = current_margin
        if state.baseline_atp_weight <= 1e-9:
            state.baseline_atp = current_atp
        stable_for_margin = state.novelty < 0.24 and state.semantic_change < 0.18 and current_margin > 0.45
        if stable_for_margin or state.baseline_margin_weight < 4.0:
            beta = 1.0 - math.exp(-0.020 * elapsed)
            state.baseline_margin += beta * (current_margin - state.baseline_margin)
            state.baseline_margin_weight += elapsed
        stable_for_atp = state.novelty < 0.19 and state.semantic_change < 0.14 and current_margin > 0.60
        if stable_for_atp or state.baseline_atp_weight < 4.0:
            beta = 1.0 - math.exp(-0.018 * elapsed)
            state.baseline_atp += beta * (current_atp - state.baseline_atp)
            state.baseline_atp_weight += elapsed
        decline = clamp((state.baseline_margin - current_margin) / max(state.baseline_margin, 0.15), 0.0, 1.0)
        atp_decline = clamp((state.baseline_atp - current_atp) / max(state.baseline_atp, 0.12), 0.0, 1.0)
        state.body_decline += (1.0 - math.exp(-0.30 * elapsed)) * (decline - state.body_decline)
        state.energy_decline += (1.0 - math.exp(-0.28 * elapsed)) * (atp_decline - state.energy_decline)

        body_risk = max(
            1.0 - current_margin,
            1.0 - float(cell.closure()),
            clamp(float(cell.damage_burden()), 0.0, 1.0),
        )
        semantic_gate = clamp(state.semantic_change / max(self.config.sentinel_semantic_change_threshold, 1e-9), 0.0, 1.0)
        novelty_gate = clamp((state.novelty - self.config.sentinel_profile_surprise_threshold) / max(1.0 - self.config.sentinel_profile_surprise_threshold, 1e-9), 0.0, 1.0)
        mechanism_gate = clamp((state.mechanism_change - self.config.sentinel_motor_loss_threshold) / max(1.0 - self.config.sentinel_motor_loss_threshold, 1e-9), 0.0, 1.0)
        context_gate = clamp(max(state.novelty, state.semantic_change) / 0.18, 0.0, 1.0)
        decline_gate = state.body_decline * context_gate
        energy_gate = state.energy_decline * clamp(max(state.novelty, state.semantic_change) / 0.13, 0.0, 1.0)
        state.demand_score = clamp(
            0.42 * semantic_gate
            + 0.26 * novelty_gate * state.information_deficit
            + 0.36 * mechanism_gate
            + 0.54 * decline_gate
            + 0.62 * energy_gate
            - 0.12 * body_risk,
            0.0, 1.0,
        )
        target_count = 4 if state.demand_score >= self.config.neurogenesis_four_cell_threshold else 2
        state.estimated_cost = (
            self.config.neurogenesis_four_cell_cost_score if target_count == 4
            else self.config.neurogenesis_two_cell_cost_score
        )
        state.estimated_value = self.config.neurogenesis_development_value_scale * state.demand_score
        state.net_value = state.estimated_value - state.estimated_cost
        state.previous_profile = profiles.copy()
        state.previous_meaning = meaning.copy()

    def _safe_for_development(self, cell):
        return bool(
            cell.autopoietic_margin() >= self.config.neurogenesis_safe_margin
            and float(cell.pools[s5.POOL_ATP]) >= self.config.neurogenesis_safe_atp
            and cell.closure() >= self.config.neurogenesis_safe_closure
        )

    def _random_trigger(self, cell, state):
        epoch = int(math.floor(self.age / self.config.neurogenesis_random_interval))
        if epoch <= state.random_epochs:
            return False
        state.random_epochs = epoch
        return _counter_uniform(self.p2_seed, cell.cell_id, epoch, 63) < self.config.neurogenesis_random_probability

    def _allocate_released_budget(self, cell, tissue_id, request, dt, body_before):
        port = self.port_for(cell.cell_id)
        config = self.config
        old = {
            'atp_reserve': float(config.neural_atp_reserve),
            'fuel_reserve': float(config.neural_fuel_reserve),
            'mineral_reserve': float(config.neural_mineral_reserve),
            'membrane_reserve': float(config.neural_membrane_reserve),
            'atp_rate': float(config.neural_atp_rate),
            'protein_rate': float(config.neural_protein_rate),
            'membrane_rate': float(config.neural_membrane_rate),
            'signal_rate': float(config.neural_signal_rate),
        }
        config.neural_atp_reserve = min(max(0.0, body_before['atp']), old['atp_reserve'])
        config.neural_fuel_reserve = min(max(0.0, body_before['fuel']), old['fuel_reserve'])
        config.neural_mineral_reserve = min(max(0.0, body_before['mineral']), old['mineral_reserve'])
        config.neural_membrane_reserve = min(max(0.0, body_before['membrane']), old['membrane_reserve'])
        config.neural_atp_rate = max(old['atp_rate'], float(request.get('atp', 0.0)) / max(dt, 1e-12))
        config.neural_protein_rate = max(old['protein_rate'], float(request.get('protein', 0.0)) / max(dt, 1e-12))
        config.neural_membrane_rate = max(old['membrane_rate'], float(request.get('membrane', 0.0)) / max(dt, 1e-12))
        config.neural_signal_rate = max(old['signal_rate'], float(request.get('signal', 0.0)) / max(dt, 1e-12))
        try:
            return port.allocate_budget(tissue_id, request, dt)
        finally:
            config.neural_atp_reserve = old['atp_reserve']
            config.neural_fuel_reserve = old['fuel_reserve']
            config.neural_mineral_reserve = old['mineral_reserve']
            config.neural_membrane_reserve = old['membrane_reserve']
            config.neural_atp_rate = old['atp_rate']
            config.neural_protein_rate = old['protein_rate']
            config.neural_membrane_rate = old['membrane_rate']
            config.neural_signal_rate = old['signal_rate']

    def _transfer_precursors_to_tissue(self, cell, state, tissue, dt):
        if tissue is None or getattr(tissue, '_precursor_transfer_complete', False):
            return False
        port = self.port_for(cell.cell_id)
        active = [int(i) for i in np.flatnonzero(tissue.audit_active_mask)]
        if not active:
            return False
        sent = self._sentinel_status(port)
        if sent is None:
            return False
        stores = np.asarray(sent['stores'], dtype=float)
        per_cell = {
            'protein': min(NG_TARGET_PROTEIN, float(stores[p0.BUDGET_PROTEIN]) / len(active)),
            'membrane': min(NG_TARGET_MEMBRANE, float(stores[p0.BUDGET_MEMBRANE]) / len(active)),
            'signal': min(NG_TARGET_SIGNAL, float(stores[p0.BUDGET_SIGNAL]) / len(active)),
        }
        required_fraction = min(
            per_cell['protein'] / max(NG_TARGET_PROTEIN, 1e-12),
            per_cell['membrane'] / max(NG_TARGET_MEMBRANE, 1e-12),
            per_cell['signal'] / max(NG_TARGET_SIGNAL, 1e-12),
        )
        if required_fraction < 0.94:
            return False
        total_used = {'protein': 0.0, 'membrane': 0.0, 'signal': 0.0}
        for index in active:
            request = dict(per_cell)
            body_before = {
                'atp': float(cell.pools[s5.POOL_ATP]),
                'fuel': float(cell.pools[s5.POOL_FUEL]),
                'mineral': float(cell.pools[s5.POOL_MINERAL]),
                'membrane': float(cell.pools[s5.POOL_MEM_PRECURSOR]),
            }
            returned = port.return_unused_budget(SENTINEL_ID, request)
            assembly = (
                p0.ASSEMBLY_ATP_PER_PROTEIN * float(returned['protein'])
                + p0.ASSEMBLY_ATP_PER_MEMBRANE * float(returned['membrane'])
                + p0.ASSEMBLY_ATP_PER_SIGNAL * float(returned['signal'])
                + NG_ATP_STORE
            )
            allocation = dict(returned)
            allocation['atp'] = assembly
            granted = self._allocate_released_budget(
                cell, tissue.tissue_ids[index], allocation, dt, body_before,
            )
            for name in total_used:
                total_used[name] += max(0.0, float(granted.get(name, 0.0)))
            # Any released but unallocated precursor is put back into the sentinel
            # before ordinary chemistry can consume it.
            residual = {
                name: max(0.0, float(returned.get(name, 0.0)) - float(granted.get(name, 0.0)))
                for name in total_used
            }
            if any(value > 1e-14 for value in residual.values()):
                current = {
                    'atp': float(cell.pools[s5.POOL_ATP]),
                    'fuel': max(0.0, float(cell.pools[s5.POOL_FUEL])
                                - p0.PROTEIN_FUEL_FRACTION * residual['protein']
                                - p0.SIGNAL_FUEL_FRACTION * residual['signal']),
                    'mineral': max(0.0, float(cell.pools[s5.POOL_MINERAL])
                                   - p0.PROTEIN_MINERAL_FRACTION * residual['protein']
                                   - p0.SIGNAL_MINERAL_FRACTION * residual['signal']),
                    'membrane': max(0.0, float(cell.pools[s5.POOL_MEM_PRECURSOR]) - residual['membrane']),
                }
                self._allocate_released_budget(cell, SENTINEL_ID, residual, dt, current)
        state.precursor_protein_released += total_used['protein']
        state.precursor_membrane_released += total_used['membrane']
        state.precursor_signal_released += total_used['signal']
        state.precursor_protein_used += total_used['protein']
        state.precursor_membrane_used += total_used['membrane']
        state.precursor_signal_used += total_used['signal']
        state.precursor_protein_reserved = max(0.0, state.precursor_protein_reserved - total_used['protein'])
        state.precursor_membrane_reserved = max(0.0, state.precursor_membrane_reserved - total_used['membrane'])
        state.precursor_signal_reserved = max(0.0, state.precursor_signal_reserved - total_used['signal'])
        state.development_membrane_credit = 0.0
        tissue.development_membrane_credit = 0.0
        tissue._precursor_transfer_complete = True
        return True

    def _release_precursor_membrane(self, cell, state, profile):
        port = self.port_for(cell.cell_id)
        status = self._sentinel_status(port)
        if status is None:
            return 0.0
        stores = np.asarray(status['stores'], dtype=float)
        count = max(1, s62.profile_active_count(profile))
        required = min(
            float(stores[p0.BUDGET_MEMBRANE]),
            count * NG_TARGET_MEMBRANE * 1.02,
        )
        if required <= 1e-14:
            return 0.0
        state.development_membrane_body_before = float(cell.pools[s5.POOL_MEM_PRECURSOR])
        returned = port.return_unused_budget(SENTINEL_ID, {'membrane': required})
        amount = max(0.0, float(returned.get('membrane', 0.0)))
        state.development_membrane_credit = amount
        state.precursor_membrane_released += amount
        state.precursor_membrane_reserved = max(
            0.0, float(stores[p0.BUDGET_MEMBRANE]) - amount,
        )
        return amount

    def _recover_unused_membrane_credit(self, cell, state, tissue, dt):
        credit = max(0.0, float(getattr(tissue, 'development_membrane_credit', 0.0)))
        if credit <= 1e-14:
            return 0.0
        port = self.port_for(cell.cell_id)
        body_amount = float(cell.pools[s5.POOL_MEM_PRECURSOR])
        amount = min(credit, body_amount)
        if amount <= 1e-14:
            return 0.0
        config = self.config
        old_rate = float(config.neural_membrane_rate)
        old_reserve = float(config.neural_membrane_reserve)
        config.neural_membrane_rate = max(old_rate, amount / max(float(dt), 1e-12))
        config.neural_membrane_reserve = max(0.0, body_amount - amount)
        try:
            granted = port.allocate_budget(SENTINEL_ID, {'membrane': amount}, dt)
        finally:
            config.neural_membrane_rate = old_rate
            config.neural_membrane_reserve = old_reserve
        recovered = max(0.0, float(granted.get('membrane', 0.0)))
        tissue.development_membrane_credit = max(0.0, credit - recovered)
        state.development_membrane_credit = tissue.development_membrane_credit
        state.precursor_membrane_recovered += recovered
        state.precursor_membrane_reserved += recovered
        return recovered

    def _request_development(self, cell, state, profile):
        state.trigger_attempts += 1
        if not self._safe_for_development(cell):
            state.trigger_denied_risk += 1
            return False
        if self.age < self.config.neurogenesis_min_age or self.age < state.cooldown_until:
            return False
        if state.phase not in (PHASE_NONE, PHASE_COOLDOWN):
            return False
        state.target_profile = str(profile)
        state.active_profile = str(profile)
        state.phase = PHASE_DEVELOPING
        state.development_start_age = float(self.age)
        state.last_trigger_age = float(self.age)
        state.development_membrane_credit = 0.0
        tissue = self._new_p2_tissue(cell)
        return tissue is not None

    def _maybe_trigger(self, cell, state):
        policy = self._policy()
        if policy not in TRANSIENT_POLICIES or not self.config.neurogenesis_enabled:
            return False
        if state.phase not in (PHASE_NONE, PHASE_COOLDOWN):
            return False
        if policy == POLICY_RANDOM:
            if not self._random_trigger(cell, state):
                return False
            profile = s62.PROFILE_EFFICIENT4 if _counter_uniform(self.p2_seed, cell.cell_id, state.random_epochs, 64) > 0.72 else s62.PROFILE_EFFICIENT2
            return self._request_development(cell, state, profile)

        if state.net_value <= 0.0 or state.demand_score < self.config.neurogenesis_score_threshold:
            state.trigger_denied_value += 1
            return False
        profile = s62.PROFILE_EFFICIENT4 if state.demand_score >= self.config.neurogenesis_four_cell_threshold else s62.PROFILE_EFFICIENT2
        if policy == POLICY_DEMAND_NO_PREDICTION:
            profile = s62.PROFILE_NO_PREDICTION if profile == s62.PROFILE_EFFICIENT4 else s62.PROFILE_MINIMAL4
            # The two-cell no-prediction variant is represented by efficient2
            # plus the config-level prediction flag below.
            if s62.profile_active_count(profile) > 2 and state.demand_score < self.config.neurogenesis_four_cell_threshold:
                profile = s62.PROFILE_EFFICIENT2
        return self._request_development(cell, state, profile)

    def _tissue_mature(self, tissue):
        active = np.flatnonzero(tissue.audit_active_mask)
        if active.size == 0:
            return False
        mature = int(np.count_nonzero(tissue.mature[active]))
        return bool(mature >= active.size)

    def _reabsorb_tissue(self, cell, state, reason='0.6.3-organ-reabsorption'):
        tissue = getattr(cell, 'p2_tissue', None)
        if tissue is None:
            state.phase = PHASE_COOLDOWN
            state.cooldown_until = float(self.age) + self.config.neurogenesis_cooldown
            return False
        port = self.port_for(cell.cell_id)
        ledger = getattr(tissue, 'module_ledger', None)
        if ledger is not None:
            state.archived_organ_atp += float(ledger.total_atp())
            state.archived_organ_material += float(ledger.total_material())
            state.archived_prediction_atp += float(ledger.atp.get(s62.MODULE_PREDICTION, 0.0))
            state.archived_recurrence_atp += float(ledger.atp.get(s62.MODULE_RECURRENCE, 0.0))
            state.archived_plasticity_atp += float(ledger.atp.get(s62.MODULE_PLASTICITY, 0.0))
        returned_material = 0.0
        returned_atp = 0.0
        attachment_ids = list(getattr(tissue, 'tissue_ids', []))
        attachment_ids.append(f06.FORMAL_CONTROLLER_ID)
        for tissue_id in attachment_ids:
            try:
                unused = port.return_unused_budget(tissue_id)
                returned_atp += float(unused.get('atp', 0.0))
                returned_material += sum(float(unused.get(name, 0.0)) for name in ('protein', 'membrane', 'signal'))
            except KeyError:
                pass
            try:
                returned = port.return_dead_tissue(tissue_id, reason=reason)
                returned_material += float(returned.get('material', 0.0))
                returned_atp += float(returned.get('atp_dissipated', 0.0))
            except KeyError:
                pass
        cell.p2_tissue = None
        state.phase = PHASE_COOLDOWN
        state.active_profile = s62.PROFILE_NO_TISSUE
        state.cooldown_until = float(self.age) + self.config.neurogenesis_cooldown
        state.reabsorptions += 1
        state.returned_material += returned_material
        state.returned_atp += returned_atp
        state.probe.active = False
        if tissue is not None:
            tissue.probe_freeze_learning = False
        self.neurogenesis_reabsorptions += 1
        return True

    def _update_lifecycle(self, cell, state, dt):
        tissue = getattr(cell, 'p2_tissue', None)
        if state.phase == PHASE_DEVELOPING:
            self.neurogenesis_tissue_development_time += dt
            if tissue is not None:
                tissue.organ_gate = 0.0
            if tissue is None:
                tissue = self._new_p2_tissue(cell)
            if tissue is not None and self._tissue_mature(tissue):
                state.phase = PHASE_SETTLING
                tissue.development_only = False
                state.mature_age = float(self.age)
                state.settle_until = float(self.age) + self.config.organ_settle_duration
                tissue.organ_gate = 1.0
            elif self.age - state.development_start_age > self.config.organ_development_timeout:
                state.development_timeouts += 1
                self._reabsorb_tissue(cell, state, reason='0.6.3-development-timeout')
        elif state.phase == PHASE_SETTLING:
            self.neurogenesis_tissue_active_time += dt
            if tissue is not None and self.age >= state.settle_until:
                state.phase = PHASE_PROBING
                state.probe.start(self.p2_seed, cell.cell_id, state.developments, tissue)
        elif state.phase == PHASE_PROBING:
            self.neurogenesis_tissue_active_time += dt
            if tissue is None:
                state.phase = PHASE_COOLDOWN
                state.cooldown_until = float(self.age) + self.config.neurogenesis_cooldown
            else:
                tissue.organ_gate = state.probe.current_gate()
                result = state.probe.record(tissue, cell.autopoietic_margin(), dt, self.config)
                if result is not None:
                    if result['positive']:
                        state.phase = PHASE_LEASED
                        state.lease_until = float(self.age) + self.config.organ_lease_duration
                        tissue.organ_gate = 1.0
                    elif self._policy() == POLICY_DEMAND_NO_REABSORB or not self.config.organ_reabsorb_nonpositive:
                        state.phase = PHASE_LEASED
                        state.lease_until = float(self.age) + self.config.organ_lease_duration
                        tissue.organ_gate = 1.0
                    else:
                        self._reabsorb_tissue(cell, state, reason='0.6.3-nonpositive-organ-value')
        elif state.phase == PHASE_LEASED:
            self.neurogenesis_tissue_active_time += dt
            if tissue is None:
                state.phase = PHASE_COOLDOWN
                state.cooldown_until = float(self.age) + self.config.neurogenesis_cooldown
            elif self.age >= state.lease_until:
                if state.extensions < self.config.organ_max_extensions:
                    state.extensions += 1
                    self.neurogenesis_extensions += 1
                    state.phase = PHASE_PROBING
                    state.probe.start(self.p2_seed, cell.cell_id, state.developments + state.extensions * 1000, tissue)
                elif self._policy() == POLICY_DEMAND_NO_REABSORB:
                    state.lease_until = float(self.age) + self.config.organ_lease_duration
                else:
                    self._reabsorb_tissue(cell, state, reason='0.6.3-organ-lease-expired')
        elif state.phase == PHASE_COOLDOWN:
            if self.age >= state.cooldown_until:
                state.phase = PHASE_NONE

    def _pre_p2_step(self, dt):
        # The parent calls this after body chemistry has produced current
        # physical sensor state.  Sentinel updates and tissue creation occur
        # before neural work for the same step.
        for cell in list(self.living_cells()):
            state = self._state_for(cell)
            self._ensure_sentinel(cell, state, dt)
            self._update_sentinel_evidence(cell, state, dt)
            self._maybe_trigger(cell, state)
            tissue = getattr(cell, 'p2_tissue', None)
            if state.phase == PHASE_DEVELOPING and isinstance(tissue, TransientNeuralTissue):
                self._transfer_precursors_to_tissue(cell, state, tissue, dt)
        super(Formal063World, self)._pre_p2_step(dt)
        for cell in list(self.living_cells()):
            state = self._state_for(cell)
            tissue = getattr(cell, 'p2_tissue', None)
            if tissue is None or not isinstance(tissue, TransientNeuralTissue):
                continue
            state.precursor_membrane_used = max(
                state.precursor_membrane_used, float(tissue.development_membrane_used),
            )
            state.development_membrane_credit = float(tissue.development_membrane_credit)
            self._recover_unused_membrane_credit(cell, state, tissue, dt)

    def _post_p2_step(self, dt):
        super(Formal063World, self)._post_p2_step(dt)
        if self._transient_policy():
            for cell in list(self.living_cells()):
                state = self._state_for(cell)
                self._update_lifecycle(cell, state, dt)

    def step(self, dt):
        super(Formal063World, self).step(dt)
        self.neurogenesis_world_steps += 1

    def _release_dead_cell(self, cell):
        self.neurogenesis_states.pop(str(int(cell.cell_id)), None)
        return super(Formal063World, self)._release_dead_cell(cell)

    def finite(self):
        return bool(
            super(Formal063World, self).finite()
            and all(state.finite() for state in self.neurogenesis_states.values())
        )

    def summary(self):
        output = super(Formal063World, self).summary()
        living = self.living_cells()
        states = [self._state_for(cell) for cell in living]
        phase_counts = {phase: 0 for phase in (
            PHASE_NONE, PHASE_DEVELOPING, PHASE_SETTLING, PHASE_PROBING,
            PHASE_LEASED, PHASE_COOLDOWN,
        )}
        for state in states:
            phase_counts[state.phase] = phase_counts.get(state.phase, 0) + 1
        active_ledgers = [
            cell.p2_tissue.module_ledger for cell in living
            if isinstance(getattr(cell, 'p2_tissue', None), TransientNeuralTissue)
        ]
        archived_atp = float(sum(state.archived_organ_atp for state in states))
        archived_material = float(sum(state.archived_organ_material for state in states))
        active_atp = float(sum(ledger.total_atp() for ledger in active_ledgers))
        active_material = float(sum(ledger.total_material() for ledger in active_ledgers))
        archived_prediction = float(sum(state.archived_prediction_atp for state in states))
        archived_recurrence = float(sum(state.archived_recurrence_atp for state in states))
        archived_plasticity = float(sum(state.archived_plasticity_atp for state in states))
        output.update({
            'build': BUILD,
            'neurogenesis_schema': SCHEMA_VERSION,
            'neurogenesis_policy': self._policy(),
            'neurogenesis_sentinels_ready': int(sum(state.sentinel_ready for state in states)),
            'neurogenesis_phase_none': phase_counts.get(PHASE_NONE, 0),
            'neurogenesis_phase_developing': phase_counts.get(PHASE_DEVELOPING, 0),
            'neurogenesis_phase_probing': phase_counts.get(PHASE_PROBING, 0),
            'neurogenesis_phase_leased': phase_counts.get(PHASE_LEASED, 0),
            'neurogenesis_developments': int(sum(state.developments for state in states)),
            'neurogenesis_reabsorptions': int(sum(state.reabsorptions for state in states)),
            'neurogenesis_extensions': int(sum(state.extensions for state in states)),
            'neurogenesis_probe_completed': int(sum(state.probe.completed for state in states)),
            'neurogenesis_probe_positive': int(sum(state.probe.positive for state in states)),
            'neurogenesis_probe_nonpositive': int(sum(state.probe.nonpositive for state in states)),
            'neurogenesis_mean_demand_score': _mean_or_zero(state.demand_score for state in states),
            'neurogenesis_mean_net_value': _mean_or_zero(state.net_value for state in states),
            'neurogenesis_mean_semantic_change': _mean_or_zero(state.semantic_change for state in states),
            'neurogenesis_mean_information_deficit': _mean_or_zero(state.information_deficit for state in states),
            'neurogenesis_mean_mechanism_change': _mean_or_zero(state.mechanism_change for state in states),
            'neurogenesis_mean_body_decline': _mean_or_zero(state.body_decline for state in states),
            'neurogenesis_mean_energy_decline': _mean_or_zero(state.energy_decline for state in states),
            'neurogenesis_sentinel_atp': float(sum(state.sentinel_atp_spent for state in states)),
            'neurogenesis_sentinel_wear': float(sum(state.sentinel_material_wear for state in states)),
            'neurogenesis_returned_material': float(sum(state.returned_material for state in states)),
            'neurogenesis_returned_atp': float(sum(state.returned_atp for state in states)),
            'neurogenesis_organ_atp_total': archived_atp + active_atp,
            'neurogenesis_organ_material_total': archived_material + active_material,
            'neurogenesis_prediction_atp_total': archived_prediction + float(sum(ledger.atp.get(s62.MODULE_PREDICTION, 0.0) for ledger in active_ledgers)),
            'neurogenesis_recurrence_atp_total': archived_recurrence + float(sum(ledger.atp.get(s62.MODULE_RECURRENCE, 0.0) for ledger in active_ledgers)),
            'neurogenesis_plasticity_atp_total': archived_plasticity + float(sum(ledger.atp.get(s62.MODULE_PLASTICITY, 0.0) for ledger in active_ledgers)),
            'neurogenesis_precursor_protein_reserved': float(sum(state.precursor_protein_reserved for state in states)),
            'neurogenesis_precursor_membrane_reserved': float(sum(state.precursor_membrane_reserved for state in states)),
            'neurogenesis_precursor_signal_reserved': float(sum(state.precursor_signal_reserved for state in states)),
            'neurogenesis_precursor_protein_released': float(sum(state.precursor_protein_released for state in states)),
            'neurogenesis_precursor_membrane_released': float(sum(state.precursor_membrane_released for state in states)),
            'neurogenesis_precursor_signal_released': float(sum(state.precursor_signal_released for state in states)),
            'neurogenesis_precursor_protein_used': float(sum(state.precursor_protein_used for state in states)),
            'neurogenesis_precursor_membrane_used': float(sum(state.precursor_membrane_used for state in states)),
            'neurogenesis_precursor_signal_used': float(sum(state.precursor_signal_used for state in states)),
            'neurogenesis_precursor_membrane_recovered': float(sum(state.precursor_membrane_recovered for state in states)),
            'neurogenesis_tissue_active_time': float(self.neurogenesis_tissue_active_time),
            'neurogenesis_tissue_development_time': float(self.neurogenesis_tissue_development_time),
        })
        return output

    def state_dict(self):
        state = super(Formal063World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'cells': [cell.state_dict() for cell in self.cells],
            'neurogenesis_states': {key: value.state_dict() for key, value in self.neurogenesis_states.items()},
            'neurogenesis_world_steps': self.neurogenesis_world_steps,
            'neurogenesis_developments': self.neurogenesis_developments,
            'neurogenesis_reabsorptions': self.neurogenesis_reabsorptions,
            'neurogenesis_extensions': self.neurogenesis_extensions,
            'neurogenesis_false_stable_triggers': self.neurogenesis_false_stable_triggers,
            'neurogenesis_tissue_active_time': self.neurogenesis_tissue_active_time,
            'neurogenesis_tissue_development_time': self.neurogenesis_tissue_development_time,
            'neurogenesis_sentinel_cost_atp': self.neurogenesis_sentinel_cost_atp,
            'neurogenesis_sentinel_wear': self.neurogenesis_sentinel_wear,
        })
        return state

    @classmethod
    def from_state(cls, state):
        base = dict(state)
        base['save_version'] = s62.SAVE_VERSION
        base['build'] = s62.BUILD
        allowed = set(s62.Formal062Config().__dict__.keys())
        base['config'] = {key: value for key, value in dict(state.get('config', {})).items() if key in allowed}
        world = s62.Formal062World.from_state(base)
        world.__class__ = cls
        world.config = Formal063Config.from_state(state.get('config', {}))
        world.cells = [Formal063ProtoCell.from_state(world.rng, item) for item in state['cells']]
        world.rng.bit_generator.state = state['rng_state']
        world._ng_config_during_init = world.config
        world._ng_initialising = False
        world.neurogenesis_states = {
            str(key): NeurogenesisState.from_state(value)
            for key, value in state.get('neurogenesis_states', {}).items()
        }
        for name in (
            'neurogenesis_world_steps', 'neurogenesis_developments',
            'neurogenesis_reabsorptions', 'neurogenesis_extensions',
            'neurogenesis_false_stable_triggers',
        ):
            setattr(world, name, int(state.get(name, 0)))
        for name in (
            'neurogenesis_tissue_active_time', 'neurogenesis_tissue_development_time',
            'neurogenesis_sentinel_cost_atp', 'neurogenesis_sentinel_wear',
        ):
            setattr(world, name, float(state.get(name, 0.0)))
        for cell in world.cells:
            cell.__class__ = Formal063ProtoCell
            if cell.p2_tissue is not None and not isinstance(cell.p2_tissue, TransientNeuralTissue):
                cell.p2_tissue = TransientNeuralTissue.from_state(cell.p2_tissue.state_dict())
            world._state_for(cell)
        return world

    def clone(self):
        return Formal063World.from_state(self.state_dict())


def apply_063_common_disturbance_tape(world, seed, step, stream=0):
    return s62.apply_062_common_disturbance_tape(world, seed, step, stream=stream)


def run_headless_trial(seed=101, seconds=120.0, initial_cells=1, config=None):
    world = Formal063World(seed=seed, initial_cells=initial_cells,
                           config=config or Formal063Config())
    dt = 1.0 / SIM_HZ
    margin_auc = 0.0
    uptake_start = float(world.p2_reward_uptake_total)
    living_steps = 0
    for step_index in range(int(round(float(seconds) * SIM_HZ))):
        if not world.living_cells():
            break
        world.step(dt)
        living = world.living_cells()
        margin = _mean_or_zero(cell.autopoietic_margin() for cell in living)
        margin_auc += margin * dt
        living_steps += 1
    result = world.summary()
    result.update({
        'seed': int(seed), 'seconds': float(seconds),
        'margin_auc': float(margin_auc),
        'uptake_delta': float(world.p2_reward_uptake_total - uptake_start),
        'living_steps': int(living_steps), 'finite': bool(world.finite()),
        'material_residual': float(world.matter_ledger_residual()),
    })
    return result


LOG_FIELDS = tuple(list(s62.LOG_FIELDS) + [
    'neurogenesis_policy', 'neurogenesis_sentinels_ready',
    'neurogenesis_phase_none', 'neurogenesis_phase_developing',
    'neurogenesis_phase_probing', 'neurogenesis_phase_leased',
    'neurogenesis_developments', 'neurogenesis_reabsorptions',
    'neurogenesis_extensions', 'neurogenesis_probe_completed',
    'neurogenesis_probe_positive', 'neurogenesis_probe_nonpositive',
    'neurogenesis_mean_demand_score', 'neurogenesis_mean_net_value',
    'neurogenesis_mean_semantic_change',
    'neurogenesis_mean_information_deficit',
    'neurogenesis_mean_mechanism_change', 'neurogenesis_sentinel_atp',
    'neurogenesis_sentinel_wear', 'neurogenesis_returned_material',
    'neurogenesis_returned_atp', 'neurogenesis_organ_atp_total',
    'neurogenesis_organ_material_total', 'neurogenesis_prediction_atp_total',
    'neurogenesis_recurrence_atp_total', 'neurogenesis_plasticity_atp_total',
    'neurogenesis_precursor_protein_reserved',
    'neurogenesis_precursor_membrane_reserved',
    'neurogenesis_precursor_signal_reserved',
    'neurogenesis_precursor_protein_used',
    'neurogenesis_precursor_membrane_used',
    'neurogenesis_precursor_signal_used',
    'neurogenesis_tissue_active_time', 'neurogenesis_tissue_development_time',
])


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), uuid.uuid4().hex[:8])
        self.last_age = -1e9
        self.status = 'WAIT'

    def log(self, world, reason='periodic', force=False):
        if not force and world.age - self.last_age < 10.0:
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
        self.status = 'OK'
        return True


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE,
                    session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    with open(log_path, 'r', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    sessions = {}
    for row in rows:
        sessions.setdefault(row['session_id'], []).append(row)
    fields = ('session_id', 'rows', 'final_age', 'final_cells', 'policy',
              'developments', 'reabsorptions', 'sentinel_atp')
    with open(session_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session_id, items in sessions.items():
            last = items[-1]
            writer.writerow({
                'session_id': session_id, 'rows': len(items),
                'final_age': last.get('age', ''), 'final_cells': last.get('cells', ''),
                'policy': last.get('neurogenesis_policy', ''),
                'developments': last.get('neurogenesis_developments', ''),
                'reabsorptions': last.get('neurogenesis_reabsorptions', ''),
                'sentinel_atp': last.get('neurogenesis_sentinel_atp', ''),
            })
    lines = [BUILD_LONG, 'sessions: {}'.format(len(sessions)), '']
    for session_id, items in sessions.items():
        last = items[-1]
        lines.append('{} age={} cells={} policy={} dev={} reabs={} probe={}/{} ledger={}'.format(
            session_id, last.get('age', ''), last.get('cells', ''),
            last.get('neurogenesis_policy', ''), last.get('neurogenesis_developments', ''),
            last.get('neurogenesis_reabsorptions', ''),
            last.get('neurogenesis_probe_positive', ''),
            last.get('neurogenesis_probe_nonpositive', ''),
            last.get('matter_residual', ''),
        ))
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return 'OK'


try:
    from scene import Scene, run, LANDSCAPE, background, fill, rect, text

    class SomaCell063Scene(s62.SomaCell062Scene):
        def setup(self):
            background(0.006, 0.012, 0.022)
            try:
                self.world = Formal063World.load(SAVE_FILE)
                self.save_status = 'LOAD'
            except Exception:
                self.world = Formal063World(
                    seed=101, initial_cells=2,
                    config=Formal063Config(
                        neurogenesis_policy=POLICY_DEMAND,
                        p2_environment=p2.P2_ENV_CUE_REVERSAL,
                    ),
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

        def _fresh_world(self):
            return Formal063World(
                seed=101, initial_cells=2,
                config=Formal063Config(
                    neurogenesis_policy=POLICY_DEMAND,
                    p2_environment=p2.P2_ENV_CUE_REVERSAL,
                ),
            )

        def draw(self):
            super(SomaCell063Scene, self).draw()
            summary = self.world.summary()
            fill(0.010, 0.018, 0.030, 0.97)
            rect(0.0, 152.0, self.size.w, 40.0)
            fill(0.80, 1.0, 0.93)
            text(
                '0.6.3 policy {} sentinel {} demand {:.2f} net {:+.2f} dev/reabs {}/{} probe +/− {}/{}'.format(
                    summary.get('neurogenesis_policy', 'n/a'),
                    summary.get('neurogenesis_sentinels_ready', 0),
                    summary.get('neurogenesis_mean_demand_score', 0.0),
                    summary.get('neurogenesis_mean_net_value', 0.0),
                    summary.get('neurogenesis_developments', 0),
                    summary.get('neurogenesis_reabsorptions', 0),
                    summary.get('neurogenesis_probe_positive', 0),
                    summary.get('neurogenesis_probe_nonpositive', 0),
                ),
                x=18, y=178, font_size=9, alignment=4,
            )
            text(
                'phases none/dev/probe/lease {}/{}/{}/{} sentinel ATP {:.6f} returned M {:.5f}'.format(
                    summary.get('neurogenesis_phase_none', 0),
                    summary.get('neurogenesis_phase_developing', 0),
                    summary.get('neurogenesis_phase_probing', 0),
                    summary.get('neurogenesis_phase_leased', 0),
                    summary.get('neurogenesis_sentinel_atp', 0.0),
                    summary.get('neurogenesis_returned_material', 0.0),
                ),
                x=18, y=162, font_size=9, alignment=4,
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
            seed=101, seconds=90.0, initial_cells=1,
            config=Formal063Config(
                neurogenesis_policy=POLICY_DEMAND,
                p2_environment=p2.P2_ENV_CUE_REVERSAL,
            ),
        ))
    else:
        run(SomaCell063Scene(), LANDSCAPE, show_fps=False)
