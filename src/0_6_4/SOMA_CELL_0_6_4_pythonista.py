# coding: utf-8
"""SOMA-CELL 0.6.4 — amortization boundary and conditional preparedness.

0.6.3 showed that a paid transient neural organ can occasionally help but did
not repay its readiness/development cost consistently in the tested short
challenge.  0.6.4 therefore separates a cheap material sentinel from the much
larger precursor reserve.  The cell may invest in preparedness only after
persistent or recurring physical evidence suggests that the expected future
information demand can amortize that investment.

The environment may use a single or periodic physical rule change for boundary
mapping.  The schedule is never exposed to the organism.  No teacher label,
correct direction, switch-time notification, free precursor stock, free neural
tissue, free memory inheritance, or direct body rewrite is introduced.
"""
from __future__ import division

import csv
import gc
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_3', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_3_pythonista as s63

s62 = s63.s62
s61 = s63.s61
f06 = s63.f06
p2 = s63.p2
p1 = s63.p1
p0 = s63.p0
s5 = s63.s5
s4 = s63.s4

BUILD = 'SOMA-CELL 0.6.4'
BUILD_LONG = BUILD + ' | amortization boundary / conditional preparedness'
SCHEMA_VERSION = '0.6.4-AB1.1'
SAVE_VERSION = 1
SAVE_FILE = 'soma_cell_0_6_4.pkl'
LOG_FILE = 'soma_cell_0_6_4_longrun.csv'
REPORT_FILE = 'soma_cell_0_6_4_report.txt'
SESSION_FILE = 'soma_cell_0_6_4_sessions.csv'
AUTO_SAVE_INTERVAL = 60.0
SIM_HZ = s63.SIM_HZ
clamp = s63.clamp
_atomic_pickle = s63._atomic_pickle

AMORT_CONDITIONAL = 'conditional'
AMORT_EAGER = 'eager'
AMORT_RANDOM = 'random_cost_matched'
AMORT_BARE = 'always_no_tissue'
AMORT_PREPARED_NONE = 'prepared_no_tissue'
AMORT_OPTION_NONE = 'option_no_tissue'
AMORT_ALWAYS_E2 = 'always_efficient2'
AMORT_POLICIES = frozenset((
    AMORT_CONDITIONAL, AMORT_EAGER, AMORT_RANDOM,
    AMORT_BARE, AMORT_PREPARED_NONE, AMORT_OPTION_NONE, AMORT_ALWAYS_E2,
))

RULE_SINGLE = 'single'
RULE_PERIODIC = 'periodic'
RULE_MODES = frozenset((RULE_SINGLE, RULE_PERIODIC))

PREP_MICRO = 'micro'
PREP_INVESTING = 'investing'
PREP_READY = 'ready'
PREP_RELEASING = 'releasing'
PREP_OFF = 'off'

_PARENT_POLICY = {
    AMORT_CONDITIONAL: s63.POLICY_DEMAND,
    AMORT_EAGER: s63.POLICY_DEMAND,
    AMORT_RANDOM: s63.POLICY_RANDOM,
    AMORT_BARE: s63.POLICY_ALWAYS_NONE,
    AMORT_PREPARED_NONE: s63.POLICY_PREPARED_NONE,
    AMORT_OPTION_NONE: s63.POLICY_PREPARED_NONE,
    AMORT_ALWAYS_E2: s63.POLICY_ALWAYS_E2,
}


def _mean_or_zero(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


class Formal064Config(s63.Formal063Config):
    """0.6.3 plus a conditional material-readiness investment gate."""

    def __init__(
        self,
        amortization_policy=AMORT_CONDITIONAL,
        readiness_cost_scale=1.0,
        preparedness_option_fraction=0.95,
        conditional_readiness_enabled=True,
        micro_sentinel_target_protein=0.00055,
        micro_sentinel_target_signal=0.00010,
        micro_sentinel_target_atp=0.00018,
        micro_sentinel_mature_protein=0.00038,
        micro_sentinel_mature_signal=0.000055,
        micro_sentinel_maintenance_atp_rate=1.4e-6,
        micro_sentinel_wear_rate=2.5e-8,
        preparedness_monitor_atp_per_update=1.2e-6,
        preparedness_min_age=5.5,
        preparedness_signal_threshold=0.112,
        preparedness_release_threshold=0.065,
        preparedness_min_persistence=1.25,
        preparedness_release_delay=5.5,
        preparedness_required_fraction=0.94,
        preparedness_value_rate=0.024,
        preparedness_semantic_bonus=0.012,
        preparedness_mechanism_bonus=0.014,
        preparedness_reserve_cost_weight=0.82,
        preparedness_organ_cost_weight=0.72,
        preparedness_recurrence_tau=24.0,
        preparedness_max_recurrence_credit=3.0,
        preparedness_net_threshold=0.0,
        preparedness_release_enabled=True,
        rule_mode=RULE_SINGLE,
        rule_first_change_age=10.0,
        rule_change_period=12.0,
        cue_visibility=1.0,
        task_label='default',
        external_test_harness=False,
        **kwargs
    ):
        policy = str(amortization_policy)
        if policy not in AMORT_POLICIES:
            raise ValueError('unknown amortization policy: ' + policy)
        rule_mode = str(rule_mode)
        if rule_mode not in RULE_MODES:
            raise ValueError('unknown rule mode: ' + rule_mode)
        if not bool(conditional_readiness_enabled) and policy == AMORT_CONDITIONAL:
            raise ValueError('conditional policy requires the paid readiness gate')
        if float(readiness_cost_scale) < 0.0:
            raise ValueError('readiness_cost_scale must be nonnegative')

        parent_policy = _PARENT_POLICY[policy]
        kwargs.setdefault('neurogenesis_policy', parent_policy)
        kwargs.setdefault('p2_switch_age', float(rule_first_change_age))
        kwargs.setdefault('p2_cue_alt_pulse', 0.12 * clamp(float(cue_visibility), 0.0, 1.5))
        super(Formal064Config, self).__init__(
            external_test_harness=external_test_harness, **kwargs
        )

        self.amortization_policy = policy
        self.readiness_cost_scale = float(readiness_cost_scale)
        self.preparedness_option_fraction = clamp(float(preparedness_option_fraction), 0.0, 1.0)
        self.conditional_readiness_enabled = bool(conditional_readiness_enabled)
        self.micro_sentinel_target_protein = float(micro_sentinel_target_protein)
        self.micro_sentinel_target_signal = float(micro_sentinel_target_signal)
        self.micro_sentinel_target_atp = float(micro_sentinel_target_atp)
        self.micro_sentinel_mature_protein = float(micro_sentinel_mature_protein)
        self.micro_sentinel_mature_signal = float(micro_sentinel_mature_signal)
        self.micro_sentinel_maintenance_atp_rate = float(micro_sentinel_maintenance_atp_rate)
        self.micro_sentinel_wear_rate = float(micro_sentinel_wear_rate)
        self.preparedness_monitor_atp_per_update = float(preparedness_monitor_atp_per_update)
        self.preparedness_min_age = float(preparedness_min_age)
        self.preparedness_signal_threshold = float(preparedness_signal_threshold)
        self.preparedness_release_threshold = float(preparedness_release_threshold)
        self.preparedness_min_persistence = float(preparedness_min_persistence)
        self.preparedness_release_delay = float(preparedness_release_delay)
        self.preparedness_required_fraction = clamp(float(preparedness_required_fraction), 0.1, 1.0)
        self.preparedness_value_rate = float(preparedness_value_rate)
        self.preparedness_semantic_bonus = float(preparedness_semantic_bonus)
        self.preparedness_mechanism_bonus = float(preparedness_mechanism_bonus)
        self.preparedness_reserve_cost_weight = float(preparedness_reserve_cost_weight)
        self.preparedness_organ_cost_weight = float(preparedness_organ_cost_weight)
        self.preparedness_recurrence_tau = max(0.5, float(preparedness_recurrence_tau))
        self.preparedness_max_recurrence_credit = max(0.0, float(preparedness_max_recurrence_credit))
        self.preparedness_net_threshold = float(preparedness_net_threshold)
        self.preparedness_release_enabled = bool(preparedness_release_enabled)
        self.rule_mode = rule_mode
        self.rule_first_change_age = float(rule_first_change_age)
        self.rule_change_period = max(2.0, float(rule_change_period))
        self.cue_visibility = float(cue_visibility)
        self.task_label = str(task_label)

    @classmethod
    def from_state(cls, state):
        state = dict(state or {})
        allowed = set(cls().__dict__.keys())
        allowed.discard('audit_active_cells')
        state.pop('audit_active_cells', None)
        return cls(**{key: value for key, value in state.items() if key in allowed})


class Formal064World(s63.Formal063World):
    def __init__(self, seed=101, initial_cells=1, config=None):
        config = config if config is not None else Formal064Config()
        if not isinstance(config, Formal064Config):
            config = Formal064Config.from_state(config.state_dict())
        self.amortization_rule_reversed = False
        self.amortization_rule_switches = 0
        self.amortization_world_steps = 0
        self.preparedness_investments = 0
        self.preparedness_release_events = 0
        self.preparedness_returned_material = 0.0
        self.preparedness_returned_atp = 0.0
        self.preparedness_monitor_atp = 0.0
        super(Formal064World, self).__init__(
            seed=seed, initial_cells=initial_cells, config=config,
        )
        self.config = config
        for cell in self.living_cells():
            self._ensure_064_state(self._state_for(cell))
        self.amortization_rule_reversed = self._scheduled_rule_reversed(self.age)
        self.initial_total_material = self.total_material()
        self.last_step_material_residual = 0.0

    def _amort_policy(self):
        return str(self.config.amortization_policy)

    def _conditional(self):
        return self._amort_policy() == AMORT_CONDITIONAL

    def _ensure_064_state(self, state):
        defaults = {
            'preparedness_phase': PREP_OFF if self._amort_policy() in (AMORT_BARE, AMORT_ALWAYS_E2) else PREP_MICRO,
            'preparedness_authorized': self._amort_policy() in (AMORT_EAGER, AMORT_RANDOM, AMORT_PREPARED_NONE),
            'preparedness_ready': False,
            'preparedness_level': 0.0,
            'preparedness_option_level': 0.0,
            'preparedness_expansion_level': 0.0,
            'preparedness_option_material': 0.0,
            'preparedness_expansion_material': 0.0,
            'preparedness_signal': 0.0,
            'preparedness_persistence': 0.0,
            'preparedness_episode_active': False,
            'preparedness_episode_duration': 0.0,
            'preparedness_mean_episode_duration': 0.0,
            'preparedness_episode_count': 0,
            'preparedness_last_event_age': -1e9,
            'preparedness_mean_interval': 0.0,
            'preparedness_horizon': 0.0,
            'preparedness_expected_benefit': 0.0,
            'preparedness_expected_cost': 0.0,
            'preparedness_net_value': 0.0,
            'preparedness_organ_net_value': 0.0,
            'preparedness_last_update_age': -1e9,
            'preparedness_low_since': -1.0,
            'preparedness_authorized_age': -1.0,
            'preparedness_ready_age': -1.0,
            'preparedness_investments': 0,
            'preparedness_release_events': 0,
            'preparedness_returned_material': 0.0,
            'preparedness_returned_atp': 0.0,
            'preparedness_monitor_atp': 0.0,
        }
        for key, value in defaults.items():
            if not hasattr(state, key):
                setattr(state, key, value)
        return state

    def _state_for(self, cell):
        state = super(Formal064World, self)._state_for(cell)
        return self._ensure_064_state(state)

    def _scheduled_rule_reversed(self, age):
        age = float(age)
        first = float(self.config.rule_first_change_age)
        if age < first:
            return False
        if self.config.rule_mode == RULE_SINGLE:
            return True
        epoch = int(math.floor((age - first) / max(self.config.rule_change_period, 1e-9)))
        return bool(epoch % 2 == 0)

    def _apply_rule_schedule(self):
        desired = bool(self._scheduled_rule_reversed(self.age))
        if desired != bool(self.amortization_rule_reversed):
            self.amortization_rule_reversed = desired
            self.amortization_rule_switches += 1
            # P2 reads only a physical law parameter.  The organism never sees
            # this switch variable or the transition time.
            self.p2_switch_recorded = False
        self.config.p2_switch_age = -1e9 if desired else 1e12

    def _prep_targets(self, fraction=1.0):
        scale = max(0.0, float(self.config.readiness_cost_scale))
        fraction = clamp(float(fraction), 0.0, 1.0)
        return (
            fraction * scale * float(self.config.sentinel_protein_precursor_reserve),
            fraction * scale * float(self.config.sentinel_membrane_precursor_reserve),
            fraction * scale * float(self.config.sentinel_signal_precursor_reserve),
        )

    def _option_targets(self):
        return self._prep_targets(self.config.preparedness_option_fraction)

    def _target_reserves(self, state):
        policy = self._amort_policy()
        if policy in (AMORT_EAGER, AMORT_RANDOM, AMORT_PREPARED_NONE):
            return self._prep_targets(1.0)
        if policy == AMORT_OPTION_NONE:
            return self._option_targets()
        if self._conditional():
            return self._prep_targets(1.0) if state.preparedness_authorized else self._option_targets()
        return (0.0, 0.0, 0.0)

    def _ensure_sentinel(self, cell, state, dt):
        policy = self._amort_policy()
        if policy in (AMORT_BARE, AMORT_ALWAYS_E2):
            state.preparedness_phase = PREP_OFF
            return False
        if f06.formal_controller_activity(cell) < 0.016:
            return False
        port = self.port_for(cell.cell_id)
        status = self._sentinel_status(port)
        if status is None:
            port.attach(s63.SENTINEL_ID, kind=s63.SENTINEL_KIND)
            status = self._sentinel_status(port)

        conditional = self._conditional()
        eager = policy in (AMORT_EAGER, AMORT_RANDOM, AMORT_PREPARED_NONE)
        if eager:
            state.preparedness_authorized = True
            if state.preparedness_authorized_age < 0.0:
                state.preparedness_authorized_age = float(self.age)

        if conditional:
            target_protein = self.config.micro_sentinel_target_protein
            target_signal = self.config.micro_sentinel_target_signal
            target_atp = self.config.micro_sentinel_target_atp
            mature_protein = self.config.micro_sentinel_mature_protein
            mature_signal = self.config.micro_sentinel_mature_signal
            maintenance = self.config.micro_sentinel_maintenance_atp_rate
            wear_rate = self.config.micro_sentinel_wear_rate
        else:
            target_protein = self.config.sentinel_target_protein
            target_signal = self.config.sentinel_target_signal
            target_atp = self.config.sentinel_target_atp
            mature_protein = self.config.sentinel_mature_protein
            mature_signal = self.config.sentinel_mature_signal
            maintenance = self.config.sentinel_maintenance_atp_rate
            wear_rate = self.config.sentinel_wear_rate

        target_reserves = self._target_reserves(state)
        status = self._sentinel_status(port)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        stores = np.asarray(status['stores'], dtype=float)
        protein_need = max(0.0, target_protein - float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]))
        signal_total = float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL])
        signal_need = max(0.0, target_signal - signal_total)
        atp_need = max(0.0, target_atp - float(stores[p0.BUDGET_ATP]))
        protein_reserve_need = max(0.0, target_reserves[0] - float(stores[p0.BUDGET_PROTEIN]))
        membrane_reserve_need = max(0.0, target_reserves[1] - float(stores[p0.BUDGET_MEMBRANE]))
        free_signal_after_target = max(0.0, float(stores[p0.BUDGET_SIGNAL]) - signal_need)
        signal_reserve_need = max(0.0, target_reserves[2] - free_signal_after_target)
        wear = max(0.0, wear_rate * dt)
        assembly_atp = (
            protein_need * p0.ASSEMBLY_ATP_PER_PROTEIN
            + signal_need * p0.ASSEMBLY_ATP_PER_SIGNAL
            + wear * p0.ASSEMBLY_ATP_PER_PROTEIN
        )
        request = {
            'atp': min(atp_need + assembly_atp + maintenance * dt, 0.0015),
            'protein': min(protein_need + wear + protein_reserve_need, 0.00240),
            'membrane': min(membrane_reserve_need, 0.00120),
            'signal': min(signal_need + signal_reserve_need, 0.00120),
        }
        if any(value > 1e-14 for value in request.values()):
            port.allocate_budget(s63.SENTINEL_ID, request, dt)
        status = self._sentinel_status(port)
        stores = np.asarray(status['stores'], dtype=float)
        built = port.commit_material(
            s63.SENTINEL_ID,
            protein=min(protein_need + wear, float(stores[p0.BUDGET_PROTEIN])),
            signal=min(signal_need, float(stores[p0.BUDGET_SIGNAL])),
            damaged_fraction=0.65 if wear > protein_need else 0.0,
            aggregate_fraction=0.10 if wear > protein_need else 0.0,
        )
        port_state = port._attachment(s63.SENTINEL_ID)
        paid, _ = port._spend_energy_and_signal(port_state, maintenance * dt, 0.0)
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
            float(tissue[p0.TISSUE_FUNCTIONAL_PROTEIN]) >= mature_protein
            and float(tissue[p0.TISSUE_SIGNAL] + stores[p0.BUDGET_SIGNAL]) >= mature_signal
            and float(stores[p0.BUDGET_ATP]) >= 0.55 * target_atp
        )
        reserve_amounts = (
            float(stores[p0.BUDGET_PROTEIN]),
            float(stores[p0.BUDGET_MEMBRANE]),
            max(0.0, float(stores[p0.BUDGET_SIGNAL])
                - max(0.0, target_signal - float(tissue[p0.TISSUE_SIGNAL]))),
        )
        full_targets = self._prep_targets(1.0)
        option_targets = self._option_targets()
        full_ratios = [amount / target for amount, target in zip(reserve_amounts, full_targets) if target > 1e-12]
        option_ratios = [amount / target for amount, target in zip(reserve_amounts, option_targets) if target > 1e-12]
        state.preparedness_level = float(clamp(min(full_ratios) if full_ratios else 0.0, 0.0, 1.0))
        state.preparedness_option_level = float(clamp(min(option_ratios) if option_ratios else 1.0, 0.0, 1.0))
        option_sum = float(sum(option_targets))
        full_sum = float(sum(full_targets))
        actual_sum = float(sum(min(amount, target) for amount, target in zip(reserve_amounts, full_targets)))
        state.preparedness_option_material = min(actual_sum, option_sum)
        state.preparedness_expansion_material = max(0.0, actual_sum - option_sum)
        expansion_target = max(0.0, full_sum - option_sum)
        state.preparedness_expansion_level = float(clamp(
            state.preparedness_expansion_material / max(expansion_target, 1e-12)
            if expansion_target > 1e-12 else 1.0, 0.0, 1.0,
        ))
        state.preparedness_ready = bool(
            state.preparedness_authorized
            and state.preparedness_level >= self.config.preparedness_required_fraction
        )
        if state.preparedness_ready and state.preparedness_ready_age < 0.0:
            state.preparedness_ready_age = float(self.age)
        if state.preparedness_authorized:
            state.preparedness_phase = PREP_READY if state.preparedness_ready else PREP_INVESTING
        else:
            state.preparedness_phase = PREP_MICRO
        state.precursor_protein_reserved = float(stores[p0.BUDGET_PROTEIN])
        state.precursor_membrane_reserved = float(stores[p0.BUDGET_MEMBRANE])
        state.precursor_signal_reserved = float(stores[p0.BUDGET_SIGNAL])
        return state.sentinel_ready

    def _release_preparedness(self, cell, state):
        if not state.preparedness_authorized:
            return False
        if getattr(cell, 'p2_tissue', None) is not None:
            return False
        if state.phase not in (s63.PHASE_NONE, s63.PHASE_COOLDOWN):
            return False
        port = self.port_for(cell.cell_id)
        status = self._sentinel_status(port)
        if status is None:
            state.preparedness_authorized = False
            state.preparedness_ready = False
            state.preparedness_phase = PREP_MICRO
            return False
        stores = np.asarray(status['stores'], dtype=float)
        tissue = np.asarray(status['tissue_material'], dtype=float)
        retain_signal = max(0.0, self.config.micro_sentinel_target_signal - float(tissue[p0.TISSUE_SIGNAL]))
        option_targets = self._option_targets() if self._conditional() else (0.0, 0.0, 0.0)
        amounts = {
            'atp': max(0.0, float(stores[p0.BUDGET_ATP]) - self.config.micro_sentinel_target_atp),
            'protein': max(0.0, float(stores[p0.BUDGET_PROTEIN]) - option_targets[0]),
            'membrane': max(0.0, float(stores[p0.BUDGET_MEMBRANE]) - option_targets[1]),
            'signal': max(0.0, float(stores[p0.BUDGET_SIGNAL]) - retain_signal - option_targets[2]),
        }
        returned = port.return_unused_budget(s63.SENTINEL_ID, amounts=amounts)
        material = sum(float(returned.get(name, 0.0)) for name in ('protein', 'membrane', 'signal'))
        atp = float(returned.get('atp', 0.0))
        state.preparedness_returned_material += material
        state.preparedness_returned_atp += atp
        state.preparedness_release_events += 1
        state.preparedness_authorized = False
        state.preparedness_ready = False
        state.preparedness_level = 0.0
        state.preparedness_phase = PREP_MICRO
        state.preparedness_low_since = -1.0
        self.preparedness_release_events += 1
        self.preparedness_returned_material += material
        self.preparedness_returned_atp += atp
        return bool(material + atp > 0.0)

    def _update_preparedness_evidence(self, cell, state):
        previous = float(state.preparedness_last_update_age)
        if state.last_update_age <= previous + 1e-12:
            return
        elapsed = max(1.0 / SIM_HZ, float(state.last_update_age - previous) if previous > -1e8 else self.config.sentinel_update_interval)
        state.preparedness_last_update_age = float(state.last_update_age)

        # The sentinel's inference consumes real ATP.  This is separate from
        # its baseline maintenance and is paid only on evidence updates.
        if state.sentinel_ready:
            try:
                port_state = self.port_for(cell.cell_id)._attachment(s63.SENTINEL_ID)
                paid, _ = self.port_for(cell.cell_id)._spend_energy_and_signal(
                    port_state, self.config.preparedness_monitor_atp_per_update, 0.0,
                )
                state.preparedness_monitor_atp += float(paid)
                self.preparedness_monitor_atp += float(paid)
            except KeyError:
                pass

        raw = max(
            float(state.demand_score),
            0.82 * float(state.semantic_change),
            0.82 * float(state.mechanism_change),
            0.50 * float(state.novelty) * float(state.information_deficit),
        )
        alpha = 1.0 - math.exp(-0.42 * elapsed)
        state.preparedness_signal += alpha * (raw - state.preparedness_signal)
        above = state.preparedness_signal >= self.config.preparedness_signal_threshold
        if above:
            if not state.preparedness_episode_active:
                if state.preparedness_last_event_age > -1e8:
                    interval = max(0.0, self.age - state.preparedness_last_event_age)
                    if state.preparedness_mean_interval <= 0.0:
                        state.preparedness_mean_interval = interval
                    else:
                        state.preparedness_mean_interval += 0.35 * (interval - state.preparedness_mean_interval)
                state.preparedness_last_event_age = float(self.age)
                state.preparedness_episode_count += 1
                state.preparedness_episode_active = True
                state.preparedness_episode_duration = 0.0
            state.preparedness_episode_duration += elapsed
            beta = 1.0 - math.exp(-0.33 * elapsed)
            state.preparedness_persistence += beta * (state.preparedness_signal - state.preparedness_persistence)
        else:
            if state.preparedness_episode_active:
                duration = float(state.preparedness_episode_duration)
                if state.preparedness_mean_episode_duration <= 0.0:
                    state.preparedness_mean_episode_duration = duration
                else:
                    state.preparedness_mean_episode_duration += 0.35 * (duration - state.preparedness_mean_episode_duration)
            state.preparedness_episode_active = False
            state.preparedness_episode_duration = 0.0
            state.preparedness_persistence *= math.exp(-0.30 * elapsed)

        recurrence = 0.0
        if state.preparedness_episode_count >= 2 and state.preparedness_mean_interval > 0.0:
            recurrence = min(
                self.config.preparedness_max_recurrence_credit,
                float(state.preparedness_episode_count - 1)
                * math.exp(-state.preparedness_mean_interval / self.config.preparedness_recurrence_tau),
            )
        historical_duration = max(
            state.preparedness_mean_episode_duration,
            0.5 * state.preparedness_episode_duration,
        )
        state.preparedness_horizon = float(
            state.preparedness_episode_duration + recurrence * historical_duration
        )
        option_targets = self._option_targets()
        full_targets = self._prep_targets(1.0)
        option_cost = self.config.preparedness_reserve_cost_weight * sum(option_targets)
        expansion_cost = self.config.preparedness_reserve_cost_weight * max(0.0, sum(full_targets) - sum(option_targets))
        reserve_cost = expansion_cost if self._conditional() and not state.preparedness_authorized else 0.0
        if self._amort_policy() in (AMORT_EAGER, AMORT_RANDOM, AMORT_PREPARED_NONE):
            reserve_cost = option_cost + expansion_cost
        elif self._amort_policy() == AMORT_OPTION_NONE:
            reserve_cost = option_cost
        organ_cost = self.config.preparedness_organ_cost_weight * self.config.neurogenesis_two_cell_cost_score
        state.preparedness_expected_cost = float(reserve_cost)
        state.preparedness_expected_benefit = float(
            self.config.preparedness_value_rate
            * state.preparedness_signal * state.preparedness_horizon
            + self.config.preparedness_semantic_bonus * state.semantic_change
            + self.config.preparedness_mechanism_bonus * state.mechanism_change
        )
        state.preparedness_net_value = float(
            state.preparedness_expected_benefit - state.preparedness_expected_cost
        )
        state.preparedness_organ_net_value = float(
            state.preparedness_expected_benefit - state.preparedness_expected_cost - organ_cost
        )

        if self._conditional() and not state.preparedness_authorized:
            if (
                self.age >= self.config.preparedness_min_age
                and state.preparedness_episode_duration >= self.config.preparedness_min_persistence
                and state.preparedness_net_value > self.config.preparedness_net_threshold
                and self._safe_for_development(cell)
            ):
                state.preparedness_authorized = True
                state.preparedness_authorized_age = float(self.age)
                state.preparedness_investments += 1
                state.preparedness_phase = PREP_INVESTING
                self.preparedness_investments += 1

        if self._conditional() and state.preparedness_authorized:
            if state.preparedness_signal <= self.config.preparedness_release_threshold:
                if state.preparedness_low_since < 0.0:
                    state.preparedness_low_since = float(self.age)
                if (
                    self.config.preparedness_release_enabled
                    and self.age - state.preparedness_low_since >= self.config.preparedness_release_delay
                ):
                    self._release_preparedness(cell, state)
            else:
                state.preparedness_low_since = -1.0

    def _update_sentinel_evidence(self, cell, state, dt):
        before = float(state.last_update_age)
        super(Formal064World, self)._update_sentinel_evidence(cell, state, dt)
        if state.last_update_age > before + 1e-12:
            self._update_preparedness_evidence(cell, state)

    def _maybe_trigger(self, cell, state):
        if self._conditional():
            if not state.preparedness_authorized or not state.preparedness_ready:
                state.trigger_denied_value += 1
                return False
            if state.preparedness_organ_net_value <= self.config.preparedness_net_threshold:
                state.trigger_denied_value += 1
                return False
        return super(Formal064World, self)._maybe_trigger(cell, state)

    def _pre_p2_step(self, dt):
        self._apply_rule_schedule()
        super(Formal064World, self)._pre_p2_step(dt)

    def step(self, dt):
        super(Formal064World, self).step(dt)
        self.amortization_world_steps += 1

    def finite(self):
        if not super(Formal064World, self).finite():
            return False
        values = [
            self.preparedness_returned_material, self.preparedness_returned_atp,
            self.preparedness_monitor_atp,
        ]
        for state in self.neurogenesis_states.values():
            self._ensure_064_state(state)
            values.extend([
                state.preparedness_level, state.preparedness_option_level,
                state.preparedness_expansion_level, state.preparedness_option_material,
                state.preparedness_expansion_material, state.preparedness_signal,
                state.preparedness_persistence, state.preparedness_horizon,
                state.preparedness_expected_benefit, state.preparedness_expected_cost,
                state.preparedness_net_value, state.preparedness_organ_net_value, state.preparedness_returned_material,
                state.preparedness_returned_atp, state.preparedness_monitor_atp,
            ])
        return bool(np.all(np.isfinite(values)))

    def summary(self):
        output = super(Formal064World, self).summary()
        states = [self._state_for(cell) for cell in self.living_cells()]
        output.update({
            'build': BUILD,
            'amortization_schema': SCHEMA_VERSION,
            'amortization_policy': self._amort_policy(),
            'amortization_task_label': self.config.task_label,
            'amortization_rule_mode': self.config.rule_mode,
            'amortization_rule_reversed': int(bool(self.amortization_rule_reversed)),
            'amortization_rule_switches': int(self.amortization_rule_switches),
            'amortization_cue_visibility': float(self.config.cue_visibility),
            'amortization_cue_delay': float(self.config.p2_cue_delay),
            'amortization_readiness_cost_scale': float(self.config.readiness_cost_scale),
            'amortization_option_fraction': float(self.config.preparedness_option_fraction),
            'preparedness_authorized': int(sum(bool(s.preparedness_authorized) for s in states)),
            'preparedness_ready': int(sum(bool(s.preparedness_ready) for s in states)),
            'preparedness_mean_level': _mean_or_zero(s.preparedness_level for s in states),
            'preparedness_mean_option_level': _mean_or_zero(s.preparedness_option_level for s in states),
            'preparedness_mean_expansion_level': _mean_or_zero(s.preparedness_expansion_level for s in states),
            'preparedness_option_material': float(sum(s.preparedness_option_material for s in states)),
            'preparedness_expansion_material': float(sum(s.preparedness_expansion_material for s in states)),
            'preparedness_mean_signal': _mean_or_zero(s.preparedness_signal for s in states),
            'preparedness_mean_persistence': _mean_or_zero(s.preparedness_persistence for s in states),
            'preparedness_mean_horizon': _mean_or_zero(s.preparedness_horizon for s in states),
            'preparedness_mean_expected_benefit': _mean_or_zero(s.preparedness_expected_benefit for s in states),
            'preparedness_mean_expected_cost': _mean_or_zero(s.preparedness_expected_cost for s in states),
            'preparedness_mean_net_value': _mean_or_zero(s.preparedness_net_value for s in states),
            'preparedness_mean_organ_net_value': _mean_or_zero(s.preparedness_organ_net_value for s in states),
            'preparedness_investments': int(sum(s.preparedness_investments for s in states)),
            'preparedness_release_events': int(sum(s.preparedness_release_events for s in states)),
            'preparedness_returned_material': float(sum(s.preparedness_returned_material for s in states)),
            'preparedness_returned_atp': float(sum(s.preparedness_returned_atp for s in states)),
            'preparedness_monitor_atp': float(sum(s.preparedness_monitor_atp for s in states)),
        })
        return output

    def state_dict(self):
        state = super(Formal064World, self).state_dict()
        state.update({
            'save_version': SAVE_VERSION,
            'build': BUILD,
            'config': self.config.state_dict(),
            'amortization_rule_reversed': self.amortization_rule_reversed,
            'amortization_rule_switches': self.amortization_rule_switches,
            'amortization_world_steps': self.amortization_world_steps,
            'preparedness_investments': self.preparedness_investments,
            'preparedness_release_events': self.preparedness_release_events,
            'preparedness_returned_material': self.preparedness_returned_material,
            'preparedness_returned_atp': self.preparedness_returned_atp,
            'preparedness_monitor_atp': self.preparedness_monitor_atp,
        })
        return state

    @classmethod
    def from_state(cls, state):
        base = dict(state)
        base['save_version'] = s63.SAVE_VERSION
        base['build'] = s63.BUILD
        allowed = set(s63.Formal063Config().__dict__.keys())
        base['config'] = {
            key: value for key, value in dict(state.get('config', {})).items()
            if key in allowed
        }
        world = s63.Formal063World.from_state(base)
        world.__class__ = cls
        world.config = Formal064Config.from_state(state.get('config', {}))
        for name in ('amortization_rule_switches', 'amortization_world_steps',
                     'preparedness_investments', 'preparedness_release_events'):
            setattr(world, name, int(state.get(name, 0)))
        world.amortization_rule_reversed = bool(state.get(
            'amortization_rule_reversed', world._scheduled_rule_reversed(world.age)
        ))
        for name in ('preparedness_returned_material', 'preparedness_returned_atp',
                     'preparedness_monitor_atp'):
            setattr(world, name, float(state.get(name, 0.0)))
        for cell in world.cells:
            world._ensure_064_state(world._state_for(cell))
        return world

    def clone(self):
        return Formal064World.from_state(self.state_dict())


def run_headless_trial(seed=101, seconds=120.0, initial_cells=1, config=None):
    world = Formal064World(
        seed=seed, initial_cells=initial_cells,
        config=config if config is not None else Formal064Config(),
    )
    dt = 1.0 / SIM_HZ
    margin_auc = 0.0
    uptake_start = float(world.p2_reward_uptake_total)
    living_steps = 0
    for _ in range(int(round(float(seconds) * SIM_HZ))):
        if not world.living_cells():
            break
        world.step(dt)
        living = world.living_cells()
        margin_auc += _mean_or_zero(cell.autopoietic_margin() for cell in living) * dt
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


LOG_FIELDS = tuple(list(s63.LOG_FIELDS) + [
    'amortization_policy', 'amortization_task_label', 'amortization_rule_mode',
    'amortization_rule_reversed', 'amortization_rule_switches',
    'amortization_cue_visibility', 'amortization_cue_delay',
    'amortization_readiness_cost_scale', 'amortization_option_fraction', 'preparedness_authorized',
    'preparedness_ready', 'preparedness_mean_level', 'preparedness_mean_option_level',
    'preparedness_mean_expansion_level', 'preparedness_option_material',
    'preparedness_expansion_material', 'preparedness_mean_signal',
    'preparedness_mean_persistence', 'preparedness_mean_horizon',
    'preparedness_mean_expected_benefit', 'preparedness_mean_expected_cost',
    'preparedness_mean_net_value', 'preparedness_mean_organ_net_value', 'preparedness_investments',
    'preparedness_release_events', 'preparedness_returned_material',
    'preparedness_returned_atp', 'preparedness_monitor_atp',
])


class LongRunLogger(object):
    def __init__(self, world, path=LOG_FILE):
        self.path = path
        self.session_id = '{}-{}'.format(int(time.time()), int(world.p2_seed))
        self.last_age = -1e9
        self.status = 'WAIT'

    def log(self, world, reason='periodic', force=False):
        if not force and world.age - self.last_age < 10.0:
            return False
        summary = world.summary()
        row = {key: summary.get(key, '') for key in LOG_FIELDS}
        row.update({'session_id': self.session_id, 'reason': reason,
                    'wall_time': time.time()})
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        with open(self.path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=('session_id', 'reason', 'wall_time') + LOG_FIELDS
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.last_age = float(world.age)
        self.status = 'OK'
        return True


def generate_report(log_path=LOG_FILE, report_path=REPORT_FILE,
                    session_path=SESSION_FILE):
    if not os.path.exists(log_path):
        return 'NO LOG'
    with open(log_path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    sessions = {}
    for row in rows:
        sessions.setdefault(row['session_id'], []).append(row)
    with open(session_path, 'w', newline='', encoding='utf-8') as handle:
        fields = ('session_id', 'rows', 'final_age', 'final_cells', 'policy',
                  'investments', 'developments', 'releases')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session_id, items in sessions.items():
            last = items[-1]
            writer.writerow({
                'session_id': session_id, 'rows': len(items),
                'final_age': last.get('age', ''), 'final_cells': last.get('cells', ''),
                'policy': last.get('amortization_policy', ''),
                'investments': last.get('preparedness_investments', ''),
                'developments': last.get('neurogenesis_developments', ''),
                'releases': last.get('preparedness_release_events', ''),
            })
    lines = [BUILD_LONG, 'sessions: {}'.format(len(sessions)), '']
    for session_id, items in sessions.items():
        last = items[-1]
        lines.append(
            '{} age={} cells={} policy={} prep={}/{} dev={} reabs={} switches={} ledger={}'.format(
                session_id, last.get('age', ''), last.get('cells', ''),
                last.get('amortization_policy', ''), last.get('preparedness_authorized', ''),
                last.get('preparedness_ready', ''), last.get('neurogenesis_developments', ''),
                last.get('neurogenesis_reabsorptions', ''), last.get('amortization_rule_switches', ''),
                last.get('matter_residual', ''),
            )
        )
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return 'OK'


try:
    from scene import Scene, run, LANDSCAPE, background, fill, rect, text

    class SomaCell064Scene(s63.SomaCell063Scene):
        def setup(self):
            background(0.006, 0.012, 0.022)
            try:
                self.world = Formal064World.load(SAVE_FILE)
                self.save_status = 'LOAD'
            except Exception:
                self.world = self._fresh_world()
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
            return Formal064World(
                seed=101, initial_cells=2,
                config=Formal064Config(
                    amortization_policy=AMORT_CONDITIONAL,
                    p2_environment=p2.P2_ENV_CUE_REVERSAL,
                    rule_mode=RULE_PERIODIC,
                    rule_first_change_age=10.0,
                    rule_change_period=12.0,
                    task_label='interactive-periodic',
                ),
            )

        def draw(self):
            super(SomaCell064Scene, self).draw()
            s = self.world.summary()
            # Replace the inherited 0.6.3 overlay in the same compact band;
            # no additional world area is hidden.
            fill(0.010, 0.018, 0.030, 0.97)
            rect(0.0, 152.0, self.size.w, 40.0)
            fill(0.82, 0.98, 1.0)
            text(
                '0.6.4 {} option {:.2f} auth/ready {}/{} full {:.2f} horizon {:.1f}'.format(
                    s.get('amortization_policy', 'n/a'),
                    s.get('amortization_option_fraction', 0.0),
                    s.get('preparedness_authorized', 0), s.get('preparedness_ready', 0),
                    s.get('preparedness_mean_level', 0.0),
                    s.get('preparedness_mean_horizon', 0.0),
                ),
                x=18, y=178, font_size=9, alignment=4,
            )
            text(
                'rule {} rev {} switches {} invest/release {}/{} dev/reabs {}/{}'.format(
                    s.get('amortization_rule_mode', 'n/a'),
                    s.get('amortization_rule_reversed', 0),
                    s.get('amortization_rule_switches', 0),
                    s.get('preparedness_investments', 0),
                    s.get('preparedness_release_events', 0),
                    s.get('neurogenesis_developments', 0),
                    s.get('neurogenesis_reabsorptions', 0),
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
            seed=101, seconds=52.0, initial_cells=1,
            config=Formal064Config(
                amortization_policy=AMORT_CONDITIONAL,
                p2_environment=p2.P2_ENV_CUE_REVERSAL,
                rule_mode=RULE_PERIODIC,
                rule_first_change_age=10.0,
                rule_change_period=12.0,
                task_label='headless-periodic',
            ),
        ))
    else:
        run(SomaCell064Scene(), LANDSCAPE, show_fps=False)
