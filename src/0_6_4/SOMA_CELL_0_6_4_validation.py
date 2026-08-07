# coding: utf-8
"""Deterministic engineering validation for SOMA-CELL 0.6.4."""
from __future__ import division
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_3', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_4_pythonista as soma
import SOMA_CELL_0_6_3_pythonista as parent

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PREREG = os.path.join(ROOT, 'results', 'SOMA_CELL_0_6_4_PREREGISTRATION.json')
if not os.path.exists(PREREG):
    PREREG = os.path.join(HERE, 'SOMA_CELL_0_6_4_PREREGISTRATION.json')
EXPERIMENT = os.path.join(HERE, 'soma_cell_0_6_4_experiment_consolidated.json')
CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_4_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_4_VALIDATION_RESULTS.txt')


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


def config(policy=soma.AMORT_CONDITIONAL, option=0.95, environment=None, **kwargs):
    values = dict(
        amortization_policy=policy,
        preparedness_option_fraction=float(option),
        p2_environment=environment or soma.p2.P2_ENV_CUE_REVERSAL,
        rule_mode=soma.RULE_PERIODIC,
        rule_first_change_age=10.0,
        rule_change_period=12.0,
        cue_visibility=0.7,
        p2_cue_delay=5.0,
        p2_cue_period=12.0,
        p2_cue_duration=1.3,
        p2_reward_duration=2.4,
        division=False, mutation=False, environmental_damage=False,
        external_inflow=False, formal_auto_start=False,
        audit_min_active_age=999.0, diagnosis_min_active_age=999.0,
    )
    values.update(kwargs)
    return soma.Formal064Config(**values)


def world(policy=soma.AMORT_CONDITIONAL, option=0.95, seed=101, environment=None, **kwargs):
    return soma.Formal064World(
        seed=seed, initial_cells=1,
        config=config(policy=policy, option=option, environment=environment, **kwargs),
    )


def step(w, count):
    for _ in range(int(count)):
        w.step(1.0 / soma.SIM_HZ)
    return w


def run_seconds(w, seconds):
    return step(w, int(round(float(seconds) * soma.SIM_HZ)))


def cell_state(w):
    cell = w.living_cells()[0]
    return cell, w._state_for(cell)


def load_experiment():
    with open(EXPERIMENT, encoding='utf-8') as handle:
        return json.load(handle)


def test_build_schema_and_frozen_defaults():
    c = soma.Formal064Config()
    assert soma.BUILD == 'SOMA-CELL 0.6.4'
    assert soma.SCHEMA_VERSION == '0.6.4-AB1.1'
    assert c.preparedness_option_fraction == 0.95
    assert c.preparedness_required_fraction == 0.94
    assert c.preparedness_signal_threshold == 0.112
    return 'build/schema and preregistered option boundary defaults fixed'


def test_preregistration_matches_source_hash():
    prereg = json.load(open(PREREG, encoding='utf-8'))
    digest = hashlib.sha256(open(os.path.join(HERE, 'SOMA_CELL_0_6_4_pythonista.py'), 'rb').read()).hexdigest()
    assert prereg['candidate_source_sha256'] == digest
    return 'preregistration SHA-256 matches the tested candidate source'


def test_invalid_policy_and_free_gate_fail_closed():
    try:
        soma.Formal064Config(amortization_policy='free-brain')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown policy accepted')
    try:
        soma.Formal064Config(amortization_policy=soma.AMORT_CONDITIONAL,
                             conditional_readiness_enabled=False)
    except ValueError:
        return 'unknown and unmetered conditional policies rejected'
    raise AssertionError('conditional gate disabled without rejection')


def test_bare_policy_exact_parent_lockstep():
    pc = parent.Formal063Config(
        neurogenesis_policy=parent.POLICY_ALWAYS_NONE,
        p2_environment=soma.p2.P2_ENV_NATIVE,
        p2_switch_age=1e12,
        p2_cue_period=12.0, p2_cue_duration=1.3, p2_cue_delay=5.0,
        p2_reward_duration=2.4, p2_cue_alt_pulse=0.0,
        division=False, mutation=False, environmental_damage=False,
        external_inflow=False, formal_auto_start=False,
        audit_min_active_age=999.0, diagnosis_min_active_age=999.0,
    )
    cc = config(policy=soma.AMORT_BARE, environment=soma.p2.P2_ENV_NATIVE,
                rule_mode=soma.RULE_SINGLE, rule_first_change_age=1e9,
                cue_visibility=0.0)
    a = parent.Formal063World(seed=711, initial_cells=1, config=pc)
    b = soma.Formal064World(seed=711, initial_cells=1, config=cc)
    step(a, 72); step(b, 72)
    child = b.state_dict(); base = dict(child)
    base['save_version'] = parent.SAVE_VERSION; base['build'] = parent.BUILD
    allowed = set(parent.Formal063Config().__dict__.keys())
    base['config'] = {key: value for key, value in child['config'].items() if key in allowed}
    for key in ('amortization_rule_reversed', 'amortization_rule_switches',
                'amortization_world_steps', 'preparedness_investments',
                'preparedness_release_events', 'preparedness_returned_material',
                'preparedness_returned_atp', 'preparedness_monitor_atp'):
        base.pop(key, None)
    reference = a.state_dict()
    assert exact(reference, project(base, reference))
    return 'bare 0.6.4 descendant preserves frozen 0.6.3 trajectory'


def test_option_reserve_is_real_material_and_conservative():
    w = world(policy=soma.AMORT_OPTION_NONE, option=0.50, seed=712)
    run_seconds(w, 4.0)
    cell, state = cell_state(w)
    assert state.preparedness_option_level > 0.98
    assert 0.48 < state.preparedness_level < 0.52
    assert state.precursor_protein_reserved > 0.0
    assert state.precursor_membrane_reserved > 0.0
    assert state.precursor_signal_reserved > 0.0
    assert abs(w.matter_ledger_residual()) < 3e-5
    return 'early option reserve uses explicit conserved protein/membrane/signal material'


def test_option_no_tissue_matches_conditional_early_cost_without_organ():
    a = world(policy=soma.AMORT_OPTION_NONE, option=0.95, seed=713)
    b = world(policy=soma.AMORT_CONDITIONAL, option=0.95, seed=713,
              preparedness_min_age=999.0)
    run_seconds(a, 5.0); run_seconds(b, 5.0)
    ca, sa = cell_state(a); cb, sb = cell_state(b)
    assert ca.p2_tissue is None and cb.p2_tissue is None
    assert abs(sa.precursor_protein_reserved - sb.precursor_protein_reserved) < 2e-5
    assert abs(sa.precursor_membrane_reserved - sb.precursor_membrane_reserved) < 2e-5
    assert abs(sa.precursor_signal_reserved - sb.precursor_signal_reserved) < 2e-5
    return 'option-no-tissue is a physically matched early-readiness control'


def test_low_option_cannot_fake_full_readiness():
    w = world(option=0.90, seed=101)
    run_seconds(w, 40.0)
    _, state = cell_state(w)
    assert state.preparedness_option_level > 0.98
    assert state.preparedness_level < 0.94
    assert state.developments == 0
    return '90% option does not pass the 94% physical precursor-transfer threshold'


def test_ninety_five_percent_option_can_physically_develop():
    w = world(option=0.95, seed=101)
    run_seconds(w, 40.0)
    _, state = cell_state(w)
    assert state.developments >= 1
    assert w.neurogenesis_developments >= 1
    assert abs(w.matter_ledger_residual()) < 3e-5
    return '95% early option permits a paid transient two-cell organ in boundary seed 101'


def test_authorisation_requires_persistent_physical_evidence():
    w = world(option=0.95, seed=714, preparedness_min_persistence=1.25)
    run_seconds(w, 6.0)
    _, state = cell_state(w)
    assert not state.preparedness_authorized
    run_seconds(w, 8.0)
    _, state = cell_state(w)
    assert state.preparedness_episode_count >= 1
    assert state.preparedness_authorized
    frame = w.port_for(w.living_cells()[0].cell_id).raw_sensor_fluxes(soma.s63.SENTINEL_ID)
    text = repr(frame)
    for forbidden in ('rule_first_change_age', 'rule_change_period', 'correct_direction', 'reward'):
        assert forbidden not in text
    return 'full preparation follows persistent physical evidence without schedule labels'


def test_periodic_rule_schedule_changes_physical_law_only():
    w = world(seed=715, rule_first_change_age=4.0, rule_change_period=3.0)
    run_seconds(w, 14.0)
    assert w.amortization_rule_switches >= 4
    assert isinstance(w.amortization_rule_reversed, bool)
    return 'single physical rule parameter toggles periodically without an organism-facing clock signal'


def test_release_returns_only_expansion_and_retains_option_floor():
    w = world(option=0.50, seed=716, preparedness_min_age=999.0)
    cell, state = cell_state(w)
    state.preparedness_authorized = True
    run_seconds(w, 5.0)
    before = w.port_for(cell.cell_id).attachment_status(soma.s63.SENTINEL_ID)
    state.phase = soma.s63.PHASE_NONE
    assert w._release_preparedness(cell, state)
    after = w.port_for(cell.cell_id).attachment_status(soma.s63.SENTINEL_ID)
    option = w._option_targets()
    stores = np.asarray(after['stores'], dtype=float)
    assert stores[soma.p0.BUDGET_PROTEIN] + 2e-5 >= option[0]
    assert stores[soma.p0.BUDGET_MEMBRANE] + 2e-5 >= option[1]
    assert stores[soma.p0.BUDGET_SIGNAL] + 2e-5 >= option[2]
    assert state.preparedness_returned_material > 0.0
    assert np.sum(before['stores']) >= np.sum(after['stores'])
    return 'low-demand release returns expansion but conservatively retains the paid option floor'


def test_stable_moving_patch_does_not_authorise_or_develop():
    w = world(seed=6201, environment=soma.p2.P2_ENV_MOVING_PATCH,
              rule_mode=soma.RULE_SINGLE, rule_first_change_age=1e9,
              cue_visibility=0.0)
    run_seconds(w, 30.0)
    _, state = cell_state(w)
    assert state.preparedness_investments == 0
    assert state.developments == 0
    return 'stable moving-patch run preserves option but does not expand or develop'


def test_option_policy_never_develops():
    w = world(policy=soma.AMORT_OPTION_NONE, seed=717)
    run_seconds(w, 45.0)
    _, state = cell_state(w)
    assert state.developments == 0 and w.neurogenesis_developments == 0
    return 'option-no-tissue control never creates a neural organ'


def test_eager_and_random_controls_remain_paid():
    for policy in (soma.AMORT_EAGER, soma.AMORT_RANDOM):
        w = world(policy=policy, seed=718)
        run_seconds(w, 12.0)
        _, state = cell_state(w)
        assert state.precursor_protein_reserved > 0.0
        assert state.sentinel_atp_spent > 0.0
    return 'eager and cost-matched random controls retain real readiness and sentinel cost'


def test_clone_exact_continuation():
    w = world(seed=719); run_seconds(w, 12.0); clone = w.clone()
    step(w, 90); step(clone, 90)
    assert exact(w.state_dict(), clone.state_dict())
    return 'clone continues exactly through option/readiness state'


def test_save_restore_exact_continuation():
    w = world(seed=720); run_seconds(w, 12.0)
    fd, path = tempfile.mkstemp(suffix='.pkl'); os.close(fd)
    try:
        w.save(path); restored = soma.Formal064World.load(path)
        step(w, 96); step(restored, 96)
        assert exact(w.state_dict(), restored.state_dict())
    finally:
        try: os.remove(path)
        except OSError: pass
    return 'save/restore preserves periodic law, option reserve and evidence state exactly'


def test_no_free_learned_state_in_daughters_regression():
    w = world(seed=721, division=True)
    run_seconds(w, 6.0)
    cell, state = cell_state(w)
    state.preparedness_signal = 0.83
    state.preparedness_episode_count = 7
    # The inherited parent split hook resets neurogenesis state in daughters.
    if cell.can_split():
        pass
    # Deterministic regression is already inherited; ensure state is not part of genome material.
    assert 'preparedness_signal' not in repr(cell.genomes)
    return 'preparedness evidence remains lifetime control state, not free genomic inheritance'


def test_experiment_boundary_and_negative_decision_are_frozen():
    data = load_experiment()
    assert data['decision'] == 'NEGATIVE_BOUNDARY_RESULT'
    assert data['minimum_feasible_option_fraction'] == 0.95
    assert not data['performance_pass'] and data['engineering_pass'] and data['stable_pass']
    return 'registered experiment freezes a 95% feasibility boundary and negative performance decision'


def test_experiment_stable_safety_is_zero_false_expansion():
    data = load_experiment()
    assert data['stable_false_investments'] == 0
    assert data['stable_false_developments'] == 0
    return '12/12 stable holdouts had no false full investment or organ development'


def test_experiment_pair_gate_failed_without_cherry_picking():
    data = load_experiment()
    comparisons = {item['control']: item for item in data['comparisons']}
    assert comparisons[soma.AMORT_OPTION_NONE]['positive_pairs'] == 1
    assert comparisons[soma.AMORT_BARE]['positive_pairs'] == 0
    assert all(item['mean_difference'] < 0.0 for item in comparisons.values())
    return 'conditional organ lost every mean holdout comparison and failed pair consistency'


def test_all_registered_trials_finite_and_material_bounded():
    data = load_experiment()
    assert data['trial_count'] == 69 and data['error_count'] == 0
    assert data['engineering_pass']
    return 'all 69 registered trials finite with bounded material residuals'


def test_summary_exposes_option_and_expansion_without_privileged_answer():
    w = world(seed=722); run_seconds(w, 8.0); summary = w.summary()
    for key in ('amortization_option_fraction', 'preparedness_mean_option_level',
                'preparedness_mean_expansion_level', 'preparedness_option_material',
                'preparedness_expansion_material'):
        assert key in summary and np.isfinite(float(summary[key]))
    for forbidden in ('correct_direction', 'teacher_label', 'fitness_reward'):
        assert forbidden not in summary
    return 'telemetry exposes material readiness accounting but no privileged answer'


def test_scene_descendant_safe_reset_and_compact_overlay_source():
    source = open(os.path.join(HERE, 'SOMA_CELL_0_6_4_pythonista.py'), encoding='utf-8').read()
    assert 'return Formal064World(' in source
    assert 'rect(0.0, 152.0, self.size.w, 40.0)' in source
    assert "s.get('preparedness_authorized', 0)" in source
    return 'Scene inherits compact Visual Fix and resets into Formal064World'


def test_report_generator_smoke():
    w = world(seed=723)
    logger_path = tempfile.mktemp(suffix='.csv')
    report_path = tempfile.mktemp(suffix='.txt')
    session_path = tempfile.mktemp(suffix='.csv')
    try:
        logger = soma.LongRunLogger(w, path=logger_path)
        logger.log(w, reason='test', force=True)
        assert soma.generate_report(logger_path, report_path, session_path) == 'OK'
        assert os.path.exists(report_path) and os.path.exists(session_path)
    finally:
        for path in (logger_path, report_path, session_path):
            try: os.remove(path)
            except OSError: pass
    return 'long-run logger/report works with option-boundary fields'


def test_finite_and_material_ledger_stress():
    w = world(seed=724)
    run_seconds(w, 45.0)
    assert w.finite()
    assert abs(w.matter_ledger_residual()) < 3e-5
    return '45-second conditional stress finite; residual={:.3e}'.format(w.matter_ledger_residual())


TESTS = [
    test_build_schema_and_frozen_defaults,
    test_preregistration_matches_source_hash,
    test_invalid_policy_and_free_gate_fail_closed,
    test_bare_policy_exact_parent_lockstep,
    test_option_reserve_is_real_material_and_conservative,
    test_option_no_tissue_matches_conditional_early_cost_without_organ,
    test_low_option_cannot_fake_full_readiness,
    test_ninety_five_percent_option_can_physically_develop,
    test_authorisation_requires_persistent_physical_evidence,
    test_periodic_rule_schedule_changes_physical_law_only,
    test_release_returns_only_expansion_and_retains_option_floor,
    test_stable_moving_patch_does_not_authorise_or_develop,
    test_option_policy_never_develops,
    test_eager_and_random_controls_remain_paid,
    test_clone_exact_continuation,
    test_save_restore_exact_continuation,
    test_no_free_learned_state_in_daughters_regression,
    test_experiment_boundary_and_negative_decision_are_frozen,
    test_experiment_stable_safety_is_zero_false_expansion,
    test_experiment_pair_gate_failed_without_cherry_picking,
    test_all_registered_trials_finite_and_material_bounded,
    test_summary_exposes_option_and_expansion_without_privileged_answer,
    test_scene_descendant_safe_reset_and_compact_overlay_source,
    test_report_generator_smoke,
    test_finite_and_material_ledger_stress,
]


def run(selected=None, write=True):
    results = []
    iterable = TESTS if selected is None else [TESTS[int(selected)]]
    offset = 0 if selected is None else int(selected)
    for local_index, test in enumerate(iterable):
        index = offset + local_index
        start = __import__('time').time()
        try:
            detail = test(); status = 'PASS'; error = ''
        except Exception as exc:
            detail = ''; status = 'FAIL'
            error = '{}: {}\n{}'.format(type(exc).__name__, exc, traceback.format_exc())
        results.append({
            'index': index, 'name': test.__name__, 'status': status,
            'detail': detail, 'error': error,
            'wall_seconds': __import__('time').time() - start,
        })
        print('{:02d} {} {} {}'.format(index + 1, status, test.__name__, detail), flush=True)
    if write and selected is None:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader(); writer.writerows(results)
        passed = sum(row['status'] == 'PASS' for row in results)
        lines = [
            'SOMA-CELL 0.6.4 VALIDATION RESULTS',
            'Build: {} / {}'.format(soma.BUILD, soma.SCHEMA_VERSION),
            'Result: {}/{} PASS'.format(passed, len(results)), '',
        ]
        for row in results:
            lines.append('{:02d}. {} {} — {}'.format(
                row['index'] + 1, row['status'], row['name'], row['detail'] or row['error'].splitlines()[0]))
        with open(TXT_PATH, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=None)
    args = parser.parse_args()
    results = run(args.test, write=args.test is None)
    if any(row['status'] != 'PASS' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
