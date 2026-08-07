# coding: utf-8
"""Deterministic engineering validation for SOMA-CELL 0.6.3.

The suite validates paid preparedness, physical evidence gating, transient
organ development, counterbalanced organ-value probing, conservative
reabsorption, save/restore, and the preregistered negative result.  It does
not claim that neural tissue is universally useless or that the system is
life or conscious.
"""
from __future__ import division
import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_3_pythonista as soma
import SOMA_CELL_0_6_2_pythonista as parent

CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_3_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_3_VALIDATION_RESULTS.txt')
EXPERIMENT_CSV = os.path.join(HERE, 'soma_cell_0_6_3_experiment_results.csv')
CONSOLIDATED = os.path.join(HERE, 'soma_cell_0_6_3_experiment_consolidated.json')
_LOCAL_PREREG = os.path.join(HERE, 'SOMA_CELL_0_6_3_PREREGISTRATION.json')
_PROJECT_PREREG = os.path.abspath(os.path.join(
    HERE, '..', '..', 'results', 'SOMA_CELL_0_6_3_PREREGISTRATION.json'))
PREREG = _LOCAL_PREREG if os.path.exists(_LOCAL_PREREG) else _PROJECT_PREREG


def exact(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict)
                and set(a) == set(b)
                and all(exact(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(exact(x, y) for x, y in zip(a, b)))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return float(a) == float(b)
    return a == b


def project(candidate, reference):
    if isinstance(reference, dict):
        return {key: project(candidate[key], value) for key, value in reference.items()}
    if isinstance(reference, list):
        return [project(candidate[i], value) for i, value in enumerate(reference)]
    if isinstance(reference, tuple):
        return tuple(project(candidate[i], value) for i, value in enumerate(reference))
    if isinstance(reference, np.ndarray):
        return np.asarray(candidate, dtype=reference.dtype).copy()
    return candidate


def config(policy=soma.POLICY_DEMAND, environment=None, **kwargs):
    values = dict(
        neurogenesis_policy=policy,
        p2_environment=environment or soma.p2.P2_ENV_CUE_REVERSAL,
        p2_switch_age=10.0,
        division=False,
        mutation=False,
        environmental_damage=False,
        external_inflow=False,
        formal_auto_start=False,
        audit_min_active_age=999.0,
        diagnosis_min_active_age=999.0,
    )
    values.update(kwargs)
    return soma.Formal063Config(**values)


def world(policy=soma.POLICY_DEMAND, seed=101, environment=None, **kwargs):
    return soma.Formal063World(
        seed=seed, initial_cells=1,
        config=config(policy=policy, environment=environment, **kwargs),
    )


def step(w, count):
    for _ in range(int(count)):
        w.step(1.0 / soma.SIM_HZ)
    return w


def run_seconds(w, seconds):
    return step(w, int(round(float(seconds) * soma.SIM_HZ)))


def cell_and_state(w):
    cell = w.living_cells()[0]
    return cell, w._state_for(cell)


def wait_phase(w, phases, seconds=30.0):
    wanted = set(phases if isinstance(phases, (list, tuple, set)) else (phases,))
    for _ in range(int(round(seconds * soma.SIM_HZ))):
        w.step(1.0 / soma.SIM_HZ)
        if not w.living_cells():
            break
        _, state = cell_and_state(w)
        if state.phase in wanted:
            return state.phase
    raise AssertionError('phase not reached: {}'.format(sorted(wanted)))


def experiment_rows():
    with open(EXPERIMENT_CSV, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def rows(kind, policy):
    return [row for row in experiment_rows()
            if row.get('kind') == kind and row.get('policy') == policy
            and not row.get('error')]


def fnum(row, key, default=0.0):
    value = row.get(key, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        if value.strip().lower() == 'true':
            return 1.0
        if value.strip().lower() == 'false':
            return 0.0
    try:
        return float(value or default)
    except Exception:
        return float(default)


def test_build_schema_and_frozen_defaults():
    c = soma.Formal063Config()
    assert soma.BUILD == 'SOMA-CELL 0.6.3'
    assert soma.SCHEMA_VERSION == '0.6.3-NG1.0'
    assert c.neurogenesis_min_age == 5.5
    assert c.neurogenesis_score_threshold == 0.035
    assert c.neurogenesis_two_cell_cost_score == 0.025
    assert c.organ_settle_duration == 8.0
    assert c.organ_plasticity_warmup == 3.0
    assert c.organ_probe_freeze_learning
    return 'build/schema and preregistered early transient-organ defaults fixed'


def test_preregistration_matches_source_hash():
    prereg = json.load(open(PREREG, encoding='utf-8'))
    digest = hashlib.sha256(open(os.path.join(HERE, 'SOMA_CELL_0_6_3_pythonista.py'), 'rb').read()).hexdigest()
    assert prereg['candidate_source_sha256'] == digest
    return 'preregistration source SHA-256 matches candidate'


def test_unmetered_sentinel_fails_closed():
    try:
        soma.Formal063Config(neurogenesis_controller_cost=False)
    except ValueError:
        return 'unmetered neurogenesis controller rejected'
    raise AssertionError('unmetered controller accepted')


def test_always_none_exact_parent_lockstep():
    pc = parent.Formal062Config(
        neural_profile=parent.PROFILE_NO_TISSUE,
        p2_environment=soma.p2.P2_ENV_NATIVE,
        division=False, mutation=False, environmental_damage=False,
        external_inflow=False, formal_auto_start=False,
        audit_min_active_age=999.0, diagnosis_min_active_age=999.0,
        p2_switch_age=10.0,
    )
    cc = config(
        policy=soma.POLICY_ALWAYS_NONE,
        environment=soma.p2.P2_ENV_NATIVE,
        neurogenesis_enabled=False,
    )
    a = parent.Formal062World(seed=313, initial_cells=1, config=pc)
    b = soma.Formal063World(seed=313, initial_cells=1, config=cc)
    for _ in range(72):
        a.step(1.0 / soma.SIM_HZ)
        b.step(1.0 / soma.SIM_HZ)
    child = b.state_dict()
    base = dict(child)
    base['save_version'] = parent.SAVE_VERSION
    base['build'] = parent.BUILD
    allowed = set(parent.Formal062Config().__dict__.keys())
    base['config'] = {key: value for key, value in child['config'].items() if key in allowed}
    base['cells'] = [parent.Formal062ProtoCell.from_state(b.rng, item).state_dict()
                     for item in child['cells']]
    reference = a.state_dict()
    assert exact(reference, project(base, reference))
    return 'always-no-tissue descendant exact 0.6.2 parent lockstep'


def test_prepared_control_is_equal_initial_matter_and_no_tissue():
    a = world(soma.POLICY_PREPARED_NONE, seed=411)
    b = world(soma.POLICY_DEMAND, seed=411)
    assert abs(a.total_material() - b.total_material()) < 1e-12
    run_seconds(a, 8.0)
    cell, state = cell_and_state(a)
    assert cell.p2_tissue is None and state.sentinel_ready
    assert state.sentinel_atp_spent > 0.0
    assert state.precursor_protein_reserved > 0.0
    assert state.precursor_membrane_reserved > 0.0
    assert state.precursor_signal_reserved > 0.0
    return 'prepared-no-tissue pays equal initial genetic readiness and real sentinel/reserves'


def test_sentinel_reads_only_physical_port_fields():
    w = world(soma.POLICY_PREPARED_NONE, seed=412)
    run_seconds(w, 8.0)
    cell, state = cell_and_state(w)
    assert state.sentinel_ready
    frame = w.port_for(cell.cell_id).raw_sensor_fluxes(soma.SENTINEL_ID)
    text = repr(sorted(frame.keys())) + repr(frame)
    for forbidden in ('reward', 'correct_direction', 'fitness', 'switch_age', 'target_position'):
        assert forbidden not in text
    assert 'ligand_profiles' in frame['external']
    return 'sentinel sees physical ligand/flux/body data, not privileged answers'


def test_demand_trigger_matures_two_cell_tissue_and_uses_precursors():
    w = world(seed=101)
    wait_phase(w, soma.PHASE_SETTLING, seconds=9.0)
    cell, state = cell_and_state(w)
    tissue = cell.p2_tissue
    assert isinstance(tissue, soma.TransientNeuralTissue)
    active = np.flatnonzero(tissue.audit_active_mask)
    assert active.size == 2 and np.all(tissue.mature[active])
    assert state.precursor_protein_used > 0.0
    assert state.precursor_membrane_used > 0.0
    assert state.precursor_signal_used > 0.0
    assert not tissue.development_only
    assert abs(w.matter_ledger_residual()) < 2e-5
    return 'demand creates mature two-cell organ from paid stored precursors'


def test_development_only_prevents_premature_computation():
    w = world(seed=101, neurogenesis_score_threshold=2.0,
              neurogenesis_min_age=0.0)
    run_seconds(w, 8.0)
    cell, state = cell_and_state(w)
    assert state.sentinel_ready and cell.p2_tissue is None
    state.net_value = 1.0
    state.demand_score = 1.0
    assert w._request_development(cell, state, soma.s62.PROFILE_EFFICIENT2)
    tissue = cell.p2_tissue
    assert tissue.development_only and np.max(np.abs(tissue.hidden)) == 0.0
    w._transfer_precursors_to_tissue(cell, state, tissue, 1.0 / soma.SIM_HZ)
    tissue.pre_step(w.port_for(cell.cell_id), 1.0 / soma.SIM_HZ,
                    w.config, soma.p2.p2_gene_activity(cell))
    assert np.max(np.abs(tissue.hidden)) == 0.0
    return 'development-only phase assembles material without neural computation'


def test_probe_freezes_learning_in_active_and_dormant_phases():
    w = world(seed=101)
    wait_phase(w, soma.PHASE_PROBING, seconds=17.0)
    cell, state = cell_and_state(w)
    tissue = cell.p2_tissue
    assert state.probe.active and tissue.probe_freeze_learning
    before = {
        'w_sensor': tissue.w_sensor.copy(), 'w_rec': tissue.w_rec.copy(),
        'predict_w': tissue.predict_w.copy(), 'motor_gain': tissue.motor_gain.copy(),
        'predictor_updates': tissue.predictor_updates,
        'plasticity_updates': tissue.plasticity_updates,
    }
    step(w, 24)
    assert np.array_equal(before['w_sensor'], tissue.w_sensor)
    assert np.array_equal(before['w_rec'], tissue.w_rec)
    assert np.array_equal(before['predict_w'], tissue.predict_w)
    assert np.array_equal(before['motor_gain'], tissue.motor_gain)
    assert before['predictor_updates'] == tissue.predictor_updates
    assert before['plasticity_updates'] == tissue.plasticity_updates
    return 'organ probe preserves learned weights/updates in both A and B windows'


def test_counterbalanced_probe_positive_leases_seed101():
    w = world(seed=101)
    run_seconds(w, 24.2)
    _, state = cell_and_state(w)
    assert state.probe.completed >= 1 and state.probe.positive >= 1
    assert state.phase == soma.PHASE_LEASED
    assert abs(state.probe.time_sum[soma.PROBE_ACTIVE] - state.probe.time_sum[soma.PROBE_DORMANT]) < 1e-9
    assert state.probe.last_quality >= w.config.organ_probe_min_quality
    return 'counterbalanced probe leases positive seed101 organ'


def test_nonpositive_probe_reabsorbs_and_archives_cost_seed202():
    w = world(seed=202)
    run_seconds(w, 26.0)
    cell, state = cell_and_state(w)
    summary = w.summary()
    assert state.probe.nonpositive >= 1 and state.reabsorptions >= 1
    assert cell.p2_tissue is None
    assert summary['neurogenesis_returned_material'] > 0.0
    assert summary['neurogenesis_organ_atp_total'] > 0.0
    assert state.archived_organ_atp > 0.0
    assert abs(w.matter_ledger_residual()) < 2e-5
    return 'nonpositive organ reabsorbed conservatively; spent cost remains archived'


def test_no_reabsorption_control_keeps_nonpositive_organ():
    w = world(soma.POLICY_DEMAND_NO_REABSORB, seed=202)
    run_seconds(w, 26.0)
    cell, state = cell_and_state(w)
    assert state.probe.nonpositive >= 1
    assert state.reabsorptions == 0 and cell.p2_tissue is not None
    assert state.phase == soma.PHASE_LEASED
    return 'no-reabsorption ablation retains a nonpositive organ'


def test_random_control_is_rng_nonconsuming_and_preparedness_matched():
    demand = world(soma.POLICY_DEMAND, seed=501)
    random = world(soma.POLICY_RANDOM, seed=501)
    assert abs(demand.total_material() - random.total_material()) < 1e-12
    run_seconds(demand, 5.0); run_seconds(random, 5.0)
    da = demand.port_for(demand.living_cells()[0].cell_id).attachment_status(soma.SENTINEL_ID)
    ra = random.port_for(random.living_cells()[0].cell_id).attachment_status(soma.SENTINEL_ID)
    assert np.array_equal(da['stores'], ra['stores'])
    state = random._state_for(random.living_cells()[0])
    random.age = 8.1
    before = copy.deepcopy(random.rng.bit_generator.state)
    random._random_trigger(random.living_cells()[0], state)
    assert exact(before, random.rng.bit_generator.state)
    return 'random control pays same readiness and uses stateless non-RNG trigger'


def test_no_prediction_ablation_zeroes_prediction_meter():
    w = world(soma.POLICY_DEMAND_NO_PREDICTION, seed=101)
    run_seconds(w, 26.0)
    s = w.summary()
    assert s['neurogenesis_developments'] >= 1
    assert s['neurogenesis_prediction_atp_total'] == 0.0
    assert s['neurogenesis_plasticity_atp_total'] >= 0.0
    return 'demand-no-prediction develops organ with zero prediction meter'


def test_no_plasticity_ablation_zeroes_plasticity_meter():
    w = world(soma.POLICY_DEMAND_NO_PLASTICITY, seed=101)
    run_seconds(w, 26.0)
    s = w.summary()
    assert s['neurogenesis_developments'] >= 1
    assert s['neurogenesis_plasticity_atp_total'] == 0.0
    return 'demand-no-plasticity develops organ with zero plasticity meter'


def test_save_restore_exact_during_probe():
    w = world(seed=101)
    wait_phase(w, soma.PHASE_PROBING, seconds=17.0)
    fd, path = tempfile.mkstemp(suffix='.pkl'); os.close(fd)
    try:
        w.save(path)
        restored = soma.Formal063World.load(path)
        assert exact(w.state_dict(), restored.state_dict())
        step(w, 48); step(restored, 48)
        assert exact(w.state_dict(), restored.state_dict())
    finally:
        try: os.remove(path)
        except OSError: pass
    return 'save/restore exact from active organ-probe state'


def test_clone_exact_after_reabsorption():
    w = world(seed=202)
    run_seconds(w, 26.0)
    clone = w.clone()
    assert exact(w.state_dict(), clone.state_dict())
    step(w, 30); step(clone, 30)
    assert exact(w.state_dict(), clone.state_dict())
    return 'archived organ ledger and cooldown clone exactly'


def test_no_free_neurogenesis_state_inheritance_and_parent_division_regression():
    parent_state = soma.NeurogenesisState(); parent_state.demand_score = 1.0
    parent_state.developments = 4; parent_state.archived_organ_atp = 9.0
    child_state = soma.NeurogenesisState()
    assert child_state.demand_score == 0.0 and child_state.developments == 0
    assert child_state.archived_organ_atp == 0.0
    import SOMA_CELL_0_6_1_validation as inherited
    inherited.test_no_free_061_learning_state_in_division()
    return 'daughter neurogenesis ledger is fresh; inherited material division PASS'


def test_ui_inherits_compact_visual_fix_and_descendant_reset():
    src = open(os.path.join(HERE, 'SOMA_CELL_0_6_3_pythonista.py'), encoding='utf-8').read()
    p62_path = os.path.join(HERE, 'SOMA_CELL_0_6_2_pythonista.py')
    if not os.path.exists(p62_path):
        p62_path = os.path.join(HERE, '..', '0_6_2', 'SOMA_CELL_0_6_2_pythonista.py')
    p61_path = os.path.join(HERE, 'SOMA_CELL_0_6_1_pythonista.py')
    if not os.path.exists(p61_path):
        p61_path = os.path.join(HERE, '..', '0_6_1', 'SOMA_CELL_0_6_1_pythonista.py')
    p62 = open(p62_path, encoding='utf-8').read()
    p61 = open(p61_path, encoding='utf-8').read()
    assert 'class SomaCell063Scene(s62.SomaCell062Scene)' in src
    assert 'def _fresh_world(self):' in src and 'Formal063World(' in src
    assert 'class SomaCell062Scene(s61.SomaCell061Scene)' in p62
    assert "s.get('formal_audits', 0)" in p61
    assert 'self.world = self._fresh_world()' in p61
    return 'compact visual-fix chain and descendant-safe double-tap reset inherited'


def test_stable_holdout_has_no_false_development():
    stable = rows('stable', soma.POLICY_DEMAND)
    assert len(stable) == 12
    assert sum(fnum(row, 'neurogenesis_developments') > 0.0 for row in stable) == 0
    assert all(fnum(row, 'finite') == 1.0 for row in stable)
    return 'stable moving-patch holdout false development 0/12'


def test_preregistered_primary_result_is_negative_by_pair_gate():
    consolidated = json.load(open(CONSOLIDATED, encoding='utf-8'))
    primary = consolidated['comparisons']['confirmation_margin_demand_vs_prepared']
    assert primary['n'] == 6
    assert primary['positive_pairs'] == 1
    assert primary['positive_pairs'] < consolidated['acceptance']['minimum_positive_margin_pairs']
    assert consolidated['scientific_status'] == 'NEGATIVE_RESULT_WITH_LIMITS'
    return 'primary demand-vs-prepared holdout positive only 1/6 pairs; negative result'


def test_demand_loses_to_bare_no_tissue_on_mean_margin():
    consolidated = json.load(open(CONSOLIDATED, encoding='utf-8'))
    comparison = consolidated['comparisons']['confirmation_margin_demand_vs_bare']
    assert comparison['mean_difference'] < 0.0 and comparison['positive_pairs'] == 1
    return 'demand mean margin below bare no-tissue; 1/6 positive pairs'


def test_demand_uptake_signal_is_inconsistent():
    consolidated = json.load(open(CONSOLIDATED, encoding='utf-8'))
    comparison = consolidated['comparisons']['confirmation_uptake_demand_vs_prepared']
    assert comparison['positive_pairs'] == 3 and comparison['n'] == 6
    return 'uptake gain is inconsistent: demand positive in 3/6 prepared-control pairs'


def test_experiment_rows_are_finite_and_materially_conservative():
    data = experiment_rows()
    assert len(data) == 74
    assert not [row for row in data if row.get('error')]
    assert all(fnum(row, 'finite') == 1.0 for row in data)
    maximum = max(abs(fnum(row, 'material_residual')) for row in data)
    assert maximum < 2e-5
    return '74 trials finite; max material residual {:.3e}'.format(maximum)


def test_precursor_and_archived_cost_telemetry_in_log_contract():
    required = {
        'neurogenesis_precursor_protein_reserved',
        'neurogenesis_precursor_membrane_reserved',
        'neurogenesis_precursor_signal_reserved',
        'neurogenesis_precursor_protein_used',
        'neurogenesis_precursor_membrane_used',
        'neurogenesis_precursor_signal_used',
        'neurogenesis_organ_atp_total',
        'neurogenesis_organ_material_total',
    }
    assert required.issubset(set(soma.LOG_FIELDS))
    return 'long-run contract records readiness, precursor use, and archived organ cost'


def test_finite_material_stress():
    w = world(seed=606)
    run_seconds(w, 32.0)
    assert w.finite()
    residual = abs(w.matter_ledger_residual())
    assert residual < 2e-5
    assert all(state.finite() for state in w.neurogenesis_states.values())
    return '32s stress finite; material residual {:.3e}'.format(residual)


TESTS = [
    ('build_schema_and_frozen_defaults', test_build_schema_and_frozen_defaults),
    ('preregistration_matches_source_hash', test_preregistration_matches_source_hash),
    ('unmetered_sentinel_fails_closed', test_unmetered_sentinel_fails_closed),
    ('always_none_exact_parent_lockstep', test_always_none_exact_parent_lockstep),
    ('prepared_control_is_equal_initial_matter_and_no_tissue', test_prepared_control_is_equal_initial_matter_and_no_tissue),
    ('sentinel_reads_only_physical_port_fields', test_sentinel_reads_only_physical_port_fields),
    ('demand_trigger_matures_two_cell_tissue_and_uses_precursors', test_demand_trigger_matures_two_cell_tissue_and_uses_precursors),
    ('development_only_prevents_premature_computation', test_development_only_prevents_premature_computation),
    ('probe_freezes_learning_in_active_and_dormant_phases', test_probe_freezes_learning_in_active_and_dormant_phases),
    ('counterbalanced_probe_positive_leases_seed101', test_counterbalanced_probe_positive_leases_seed101),
    ('nonpositive_probe_reabsorbs_and_archives_cost_seed202', test_nonpositive_probe_reabsorbs_and_archives_cost_seed202),
    ('no_reabsorption_control_keeps_nonpositive_organ', test_no_reabsorption_control_keeps_nonpositive_organ),
    ('random_control_is_rng_nonconsuming_and_preparedness_matched', test_random_control_is_rng_nonconsuming_and_preparedness_matched),
    ('no_prediction_ablation_zeroes_prediction_meter', test_no_prediction_ablation_zeroes_prediction_meter),
    ('no_plasticity_ablation_zeroes_plasticity_meter', test_no_plasticity_ablation_zeroes_plasticity_meter),
    ('save_restore_exact_during_probe', test_save_restore_exact_during_probe),
    ('clone_exact_after_reabsorption', test_clone_exact_after_reabsorption),
    ('no_free_neurogenesis_state_inheritance_and_parent_division_regression', test_no_free_neurogenesis_state_inheritance_and_parent_division_regression),
    ('ui_inherits_compact_visual_fix_and_descendant_reset', test_ui_inherits_compact_visual_fix_and_descendant_reset),
    ('stable_holdout_has_no_false_development', test_stable_holdout_has_no_false_development),
    ('preregistered_primary_result_is_negative_by_pair_gate', test_preregistered_primary_result_is_negative_by_pair_gate),
    ('demand_loses_to_bare_no_tissue_on_mean_margin', test_demand_loses_to_bare_no_tissue_on_mean_margin),
    ('demand_uptake_signal_is_inconsistent', test_demand_uptake_signal_is_inconsistent),
    ('experiment_rows_are_finite_and_materially_conservative', test_experiment_rows_are_finite_and_materially_conservative),
    ('precursor_and_archived_cost_telemetry_in_log_contract', test_precursor_and_archived_cost_telemetry_in_log_contract),
    ('finite_material_stress', test_finite_material_stress),
]


def run_one(name):
    for test_name, function in TESTS:
        if test_name == name:
            observed = function()
            return {'name': name, 'pass': 1, 'observed': observed}
    raise KeyError(name)


def write_results(results):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('name', 'pass', 'observed', 'error'))
        writer.writeheader(); writer.writerows(results)
    passed = sum(int(row.get('pass', 0)) for row in results)
    lines = [
        'SOMA-CELL 0.6.3 VALIDATION RESULTS',
        'Build: {}'.format(soma.BUILD),
        'Schema: {}'.format(soma.SCHEMA_VERSION),
        'Passed: {}/{}'.format(passed, len(results)),
        '',
    ]
    for row in results:
        lines.append('[{}] {}'.format('PASS' if row.get('pass') else 'FAIL', row['name']))
        if row.get('observed'):
            lines.append('  observed: ' + str(row['observed']))
        if row.get('error'):
            lines.append('  error: ' + str(row['error']))
    with open(TXT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--test')
    args = parser.parse_args()
    if args.list:
        for name, _ in TESTS:
            print(name)
        return 0
    if args.test:
        try:
            result = run_one(args.test)
        except Exception as exc:
            result = {'name': args.test, 'pass': 0, 'observed': '',
                      'error': '{}: {}\n{}'.format(type(exc).__name__, exc, traceback.format_exc())}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result['pass'] else 1
    results = []
    for name, function in TESTS:
        try:
            observed = function()
            results.append({'name': name, 'pass': 1, 'observed': observed, 'error': ''})
            print('[PASS]', name, observed, flush=True)
        except Exception as exc:
            error = '{}: {}\n{}'.format(type(exc).__name__, exc, traceback.format_exc())
            results.append({'name': name, 'pass': 0, 'observed': '', 'error': error})
            print('[FAIL]', name, error, flush=True)
    write_results(results)
    return 0 if all(row['pass'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
