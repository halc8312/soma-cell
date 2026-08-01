# coding: utf-8
"""Engineering comparison for SOMA-CELL 0.6-P2.

The release study contains two complementary designs:
1. independent whole-life runs for cost and broad behaviour;
2. exact-state post-switch twins with a step-indexed common disturbance tape.

Seeds 101/202/303 were used during task calibration.  Seeds 404/505/606/707/
808/909 are declared hold-outs and are reported separately.  This is a small
mechanism study, not a population-level statistical proof.
"""
from __future__ import division

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASELINE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, P1_DIR, P0_DIR, BASELINE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_P2_pythonista as p2

DEVELOPMENT_SEEDS = (101, 202, 303)
HOLDOUT_SEEDS = (404, 505, 606, 707, 808, 909)
RESULT_PATH = os.path.join(HERE, 'soma_cell_0_6_p2_experiment_results.csv')
SUMMARY_PATH = os.path.join(HERE, 'soma_cell_0_6_p2_experiment_summary.csv')
REPORT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P2_EXPERIMENT_REPORT.txt')
JSON_PATH = os.path.join(HERE, 'soma_cell_0_6_p2_experiment_results.json')
MAX_WORKERS = min(6, max(1, os.cpu_count() or 1))

INDEPENDENT_MODES = (
    p2.P2_MODE_FULL,
    p2.P2_MODE_FIXED,
    p2.P2_MODE_NO_RECURRENCE,
    p2.P2_MODE_NO_PREDICTION,
    p2.P2_MODE_NO_PLASTICITY,
    p2.P2_MODE_NONE,
)
TWIN_CONTROLS = (
    p2.P2_MODE_FIXED,
    p2.P2_MODE_NO_PLASTICITY,
    p2.P2_MODE_NO_PREDICTION,
    p2.P2_MODE_NO_RECURRENCE,
)


def _mean_margin(world):
    cells = world.living_cells()
    if not cells:
        return 0.0
    return float(np.mean([cell.autopoietic_margin() for cell in cells]))


def _run_independent(args):
    mode, seed = args
    config = p2.P2Config(
        p2_tissue_mode=mode,
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        p2_switch_age=54.0,
        p2_plasticity_warmup=34.0,
    )
    started = time.time()
    result = p2.run_headless_trial(
        seed=seed, seconds=100.0, initial_cells=1, config=config,
    )
    keys = (
        'cells', 'divisions', 'deaths', 'mean_margin_over_life',
        'pre_switch_mean_margin', 'post_switch_mean_margin',
        'uptake_during_trial', 'mean_atp', 'mean_damage',
        'p2_reward_uptake_total', 'p2_nonreward_uptake_total',
        'p2_cue_toxin_total', 'p2_prediction_error', 'p2_prediction_rms',
        'p2_predictor_updates', 'p2_plasticity_updates',
        'p2_cue_episode_updates', 'p2_recurrent_messages',
        'p2_rewire_count', 'p2_cell_prune_count',
        'p2_neural_atp_total', 'p2_neural_signal_total',
        'p2_wear_material_total', 'p2_ligand_sensor_values',
        'finite', 'final_mass_residual',
    )
    row = {
        'trial_type': 'independent',
        'seed_set': 'development',
        'condition': mode,
        'control': '',
        'seed': int(seed),
        'wall_seconds': time.time() - started,
    }
    for key in keys:
        value = result.get(key, '')
        if isinstance(value, np.ndarray):
            value = json.dumps(value.tolist(), separators=(',', ':'))
        row[key] = value
    return row


def _run_twin(args):
    control_mode, seed, seed_set = args
    config = p2.P2Config(
        p2_tissue_mode=p2.P2_MODE_FULL,
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        p2_switch_age=54.0,
        p2_plasticity_warmup=34.0,
    )
    started = time.time()
    world = p2.P2World(seed=seed, initial_cells=1, config=config)
    dt = 1.0 / p2.SIM_HZ
    pre_steps = int(round(config.p2_switch_age * p2.SIM_HZ))
    for step in range(pre_steps):
        p2.apply_p2_common_disturbance_tape(world, seed, step, stream=31)
        world.step(dt)
        if not world.living_cells():
            break
    treatment = world.clone()
    control = world.clone()
    p2.set_p2_runtime_mode(treatment, p2.P2_MODE_FULL)
    p2.set_p2_runtime_mode(control, control_mode)
    start_reward_t = treatment.p2_reward_uptake_total
    start_reward_c = control.p2_reward_uptake_total
    start_toxin_t = treatment.p2_cue_toxin_total
    start_toxin_c = control.p2_cue_toxin_total
    auc_t = 0.0
    auc_c = 0.0
    post_steps = int(round(46.0 * p2.SIM_HZ))
    for offset in range(post_steps):
        step = pre_steps + offset
        p2.apply_p2_common_disturbance_tape(treatment, seed, step, stream=31)
        p2.apply_p2_common_disturbance_tape(control, seed, step, stream=31)
        treatment.step(dt)
        control.step(dt)
        auc_t += _mean_margin(treatment) * dt
        auc_c += _mean_margin(control) * dt
        if not treatment.living_cells() or not control.living_cells():
            break
    st = treatment.summary()
    sc = control.summary()
    return {
        'trial_type': 'common_twin',
        'seed_set': seed_set,
        'condition': p2.P2_MODE_FULL,
        'control': control_mode,
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'margin_auc_treatment': auc_t,
        'margin_auc_control': auc_c,
        'margin_auc_difference': auc_t - auc_c,
        'final_margin_treatment': _mean_margin(treatment),
        'final_margin_control': _mean_margin(control),
        'final_margin_difference': _mean_margin(treatment) - _mean_margin(control),
        'reward_gain_treatment': treatment.p2_reward_uptake_total - start_reward_t,
        'reward_gain_control': control.p2_reward_uptake_total - start_reward_c,
        'reward_gain_difference': (
            treatment.p2_reward_uptake_total - start_reward_t
            - control.p2_reward_uptake_total + start_reward_c
        ),
        'toxin_gain_treatment': treatment.p2_cue_toxin_total - start_toxin_t,
        'toxin_gain_control': control.p2_cue_toxin_total - start_toxin_c,
        'toxin_gain_difference': (
            treatment.p2_cue_toxin_total - start_toxin_t
            - control.p2_cue_toxin_total + start_toxin_c
        ),
        'plasticity_treatment': st.get('p2_plasticity_updates', 0),
        'plasticity_control': sc.get('p2_plasticity_updates', 0),
        'predictor_treatment': st.get('p2_predictor_updates', 0),
        'predictor_control': sc.get('p2_predictor_updates', 0),
        'recurrent_messages_treatment': st.get('p2_recurrent_messages', 0),
        'recurrent_messages_control': sc.get('p2_recurrent_messages', 0),
        'cells_treatment': st.get('cells', 0),
        'cells_control': sc.get('cells', 0),
        'finite_treatment': int(treatment.finite()),
        'finite_control': int(control.finite()),
        'mass_residual_treatment': treatment.matter_ledger_residual(),
        'mass_residual_control': control.matter_ledger_residual(),
    }


def _write_results(rows):
    fields = sorted({key for row in rows for key in row})
    leading = ['trial_type', 'seed_set', 'condition', 'control', 'seed', 'wall_seconds']
    fields = leading + [field for field in fields if field not in leading]
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    with open(JSON_PATH, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def _summaries(rows):
    records = []
    independent = [row for row in rows if row['trial_type'] == 'independent']
    for mode in INDEPENDENT_MODES:
        group = [row for row in independent if row['condition'] == mode]
        record = {
            'trial_type': 'independent', 'seed_set': 'development',
            'condition': mode, 'control': '', 'n': len(group),
            'wins_positive': '',
        }
        for metric in (
            'mean_margin_over_life', 'pre_switch_mean_margin',
            'post_switch_mean_margin', 'p2_reward_uptake_total',
            'p2_cue_toxin_total', 'p2_prediction_error',
            'p2_neural_atp_total', 'final_mass_residual',
        ):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            record[metric + '_mean'] = float(np.mean(values))
            record[metric + '_sd'] = float(np.std(values))
        records.append(record)

    twins = [row for row in rows if row['trial_type'] == 'common_twin']
    for seed_set in ('development', 'holdout'):
        for control in TWIN_CONTROLS:
            group = [
                row for row in twins
                if row['seed_set'] == seed_set and row['control'] == control
            ]
            if not group:
                continue
            record = {
                'trial_type': 'common_twin', 'seed_set': seed_set,
                'condition': p2.P2_MODE_FULL, 'control': control,
                'n': len(group),
                'wins_positive': sum(float(row['margin_auc_difference']) > 0.0 for row in group),
            }
            for metric in (
                'margin_auc_difference', 'final_margin_difference',
                'reward_gain_difference', 'toxin_gain_difference',
                'mass_residual_treatment', 'mass_residual_control',
            ):
                values = np.asarray([float(row[metric]) for row in group], dtype=float)
                record[metric + '_mean'] = float(np.mean(values))
                record[metric + '_sd'] = float(np.std(values))
            records.append(record)
    return records


def _write_summary(records):
    fields = sorted({key for row in records for key in row})
    leading = ['trial_type', 'seed_set', 'condition', 'control', 'n', 'wins_positive']
    fields = leading + [field for field in fields if field not in leading]
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, '') for field in fields})


def _find(records, seed_set, control):
    for row in records:
        if (
            row['trial_type'] == 'common_twin'
            and row['seed_set'] == seed_set
            and row['control'] == control
        ):
            return row
    return None


def _write_report(records):
    dev = _find(records, 'development', p2.P2_MODE_FIXED)
    hold = _find(records, 'holdout', p2.P2_MODE_FIXED)
    no_plasticity = _find(records, 'development', p2.P2_MODE_NO_PLASTICITY)
    no_prediction = _find(records, 'development', p2.P2_MODE_NO_PREDICTION)
    no_recurrence = _find(records, 'development', p2.P2_MODE_NO_RECURRENCE)
    independent = {
        row['condition']: row for row in records
        if row['trial_type'] == 'independent'
    }
    lines = [
        'SOMA-CELL 0.6-P2 experiment report',
        '===================================',
        'build: {}'.format(p2.BUILD),
        'jobs: 36',
        '',
        'Primary exact-state comparison: full plastic tissue vs equal-material fixed tissue.',
    ]
    for label, row in (('development', dev), ('holdout', hold)):
        if row is None:
            continue
        lines.extend([
            '{} seeds: n={}, AUC difference mean={:+.6f}, wins={}/{}, final margin difference={:+.6f}, reward difference={:+.6f}, toxin difference={:+.6f}'.format(
                label, row['n'], row['margin_auc_difference_mean'],
                row['wins_positive'], row['n'], row['final_margin_difference_mean'],
                row['reward_gain_difference_mean'], row['toxin_gain_difference_mean'],
            ),
        ])
    lines.extend(['', 'Development-seed module ablations:'])
    for label, row in (
        ('no plasticity', no_plasticity),
        ('no prediction', no_prediction),
        ('no recurrence', no_recurrence),
    ):
        if row is None:
            continue
        lines.append('  full vs {}: AUC {:+.6f}, wins={}/{}'.format(
            label, row['margin_auc_difference_mean'], row['wins_positive'], row['n']))
    lines.extend(['', 'Independent whole-life means:'])
    for mode in INDEPENDENT_MODES:
        row = independent.get(mode)
        if row is None:
            continue
        lines.append(
            '  {:14s} lifetime={:.6f} post={:.6f} reward={:.6f} toxin={:.6f} neural_ATP={:.6f}'.format(
                mode, row['mean_margin_over_life_mean'],
                row['post_switch_mean_margin_mean'],
                row['p2_reward_uptake_total_mean'],
                row['p2_cue_toxin_total_mean'],
                row['p2_neural_atp_total_mean'],
            )
        )
    lines.extend([
        '',
        'Interpretation:',
        '- The common disturbance tape is step-indexed; branch-dependent RNG consumption cannot create unrelated future noise.',
        '- Development seeds were used while calibrating the physical microfluidic trap, cue duration and effector scale.',
        '- Holdout seeds were not used for that calibration and are reported separately.',
        '- Prediction learned its local target but its net body value was not established.',
        '- Recurrence had only a very small mean paired effect in the present task.',
        '- Plasticity showed a limited positive paired signal on development seeds.',
        '- Independent whole-life results include development cost and trajectory divergence; they are not causal substitutes for twins.',
        '- No result establishes consciousness, open-ended evolution, life status or a general superiority of larger neural tissue.',
    ])
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def run_experiment():
    jobs = []
    for mode in INDEPENDENT_MODES:
        for seed in DEVELOPMENT_SEEDS:
            jobs.append(('independent', (mode, seed)))
    for control in TWIN_CONTROLS:
        for seed in DEVELOPMENT_SEEDS:
            jobs.append(('twin', (control, seed, 'development')))
    for seed in HOLDOUT_SEEDS:
        jobs.append(('twin', (p2.P2_MODE_FIXED, seed, 'holdout')))

    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for kind, args in jobs:
            function = _run_independent if kind == 'independent' else _run_twin
            futures.append(executor.submit(function, args))
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if row['trial_type'] == 'common_twin':
                print('{}/{} twin {} {} seed {} AUC {:+.5f}'.format(
                    completed, len(futures), row['seed_set'], row['control'],
                    row['seed'], row['margin_auc_difference'],
                ), flush=True)
            else:
                print('{}/{} independent {} seed {} post {:.5f}'.format(
                    completed, len(futures), row['condition'], row['seed'],
                    float(row['post_switch_mean_margin']),
                ), flush=True)
    rows.sort(key=lambda row: (
        row['trial_type'], row['seed_set'], row['condition'], row['control'], int(row['seed'])
    ))
    _write_results(rows)
    records = _summaries(rows)
    _write_summary(records)
    _write_report(records)
    print('completed {} jobs in {:.1f}s'.format(len(rows), time.time() - started), flush=True)
    return rows


if __name__ == '__main__':
    run_experiment()
