# coding: utf-8
"""Engineering comparison for SOMA-CELL 0.6-P1.

The decisive comparison is one material neuron versus an equal-target,
equal-maintenance non-informational dummy in the deterministic moving-patch
environment.  Additional ablations separate predictor, effector and gene/tissue
costs.  This is a small mechanism study, not a statistical performance claim.
"""
from __future__ import division

import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASELINE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, P0_DIR, BASELINE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_P1_pythonista as p1

SEEDS = (101, 202, 303)
SECONDS = 40.0
RESULT_PATH = os.path.join(HERE, 'soma_cell_0_6_p1_experiment_results.csv')
SUMMARY_PATH = os.path.join(HERE, 'soma_cell_0_6_p1_experiment_summary.csv')
REPORT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P1_EXPERIMENT_REPORT.txt')
RESUME_COMPLETED = True

CONDITIONS = (
    ('moving_neuron', dict(
        p1_tissue_mode=p1.TISSUE_NEURON,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('moving_dummy_equal_cost', dict(
        p1_tissue_mode=p1.TISSUE_DUMMY,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('moving_no_prediction', dict(
        p1_tissue_mode=p1.TISSUE_NO_PREDICTION,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('moving_no_effector', dict(
        p1_tissue_mode=p1.TISSUE_NO_EFFECTOR,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('moving_gene_only', dict(
        p1_tissue_mode=p1.TISSUE_NONE,
        p1_install_gene=True,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('moving_no_gene_no_tissue', dict(
        p1_tissue_mode=p1.TISSUE_NONE,
        p1_install_gene=False,
        p1_environment=p1.ENV_MOVING_PATCH,
    )),
    ('stable_neuron', dict(
        p1_tissue_mode=p1.TISSUE_NEURON,
        p1_environment=p1.ENV_STABLE_PATCH,
    )),
    ('stable_dummy_equal_cost', dict(
        p1_tissue_mode=p1.TISSUE_DUMMY,
        p1_environment=p1.ENV_STABLE_PATCH,
    )),
)

RESULT_FIELDS = (
    'condition', 'seed', 'seconds', 'simulated_seconds', 'wall_seconds',
    'cells', 'divisions', 'deaths', 'mean_margin_over_life',
    'mean_patch_distance_over_life', 'uptake_during_trial',
    'mean_atp', 'mean_damage', 'p1_tissues', 'p1_mature_tissues',
    'p1_tissue_creations', 'p1_tissue_turnovers',
    'p1_predictor_updates', 'p1_prediction_error',
    'p1_motor_force_total', 'p1_neural_atp_total',
    'p1_neural_signal_total', 'p1_wear_material_total',
    'neural_budget_requested', 'neural_budget_granted',
    'neural_budget_spent', 'neural_spent_atp', 'neural_spent_signal',
    'neural_effector_calls', 'p1_patch_switches', 'finite',
    'final_mass_residual',
)

SUMMARY_METRICS = (
    'mean_margin_over_life', 'mean_patch_distance_over_life',
    'uptake_during_trial', 'mean_atp', 'mean_damage',
    'p1_prediction_error', 'p1_motor_force_total',
    'p1_neural_atp_total', 'p1_neural_signal_total',
    'p1_wear_material_total', 'neural_budget_spent',
    'p1_patch_switches',
)


def _load_completed():
    completed = set()
    if not RESUME_COMPLETED or not os.path.exists(RESULT_PATH):
        return completed
    with open(RESULT_PATH, 'r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            completed.add((row['condition'], int(row['seed'])))
    return completed


def _append_result(row):
    exists = os.path.exists(RESULT_PATH)
    with open(RESULT_PATH, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in RESULT_FIELDS})


def _read_results():
    with open(RESULT_PATH, 'r', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['seed'] = int(row['seed'])
        for field in RESULT_FIELDS:
            if field in ('condition', 'seed'):
                continue
            try:
                row[field] = float(row[field])
            except Exception:
                pass
    return rows


def run_experiment():
    completed = _load_completed()
    for condition, kwargs in CONDITIONS:
        for seed in SEEDS:
            key = (condition, int(seed))
            if key in completed:
                continue
            config = p1.P1Config(**kwargs)
            wall_start = time.time()
            result = p1.run_headless_trial(
                seed=seed, seconds=SECONDS, initial_cells=1, config=config,
            )
            result['condition'] = condition
            result['seed'] = int(seed)
            result['seconds'] = float(SECONDS)
            result['wall_seconds'] = time.time() - wall_start
            _append_result(result)
            print('{} seed {} margin {:.6f} uptake {:.5f}'.format(
                condition, seed, result['mean_margin_over_life'],
                result['uptake_during_trial'],
            ), flush=True)
    rows = _read_results()
    write_summary(rows)
    write_report(rows)
    return rows


def write_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['condition'], []).append(row)
    fields = ['condition', 'n', 'survived', 'finite']
    for metric in SUMMARY_METRICS:
        fields.extend([metric + '_mean', metric + '_sd'])
    fields.extend([
        'margin_gain_vs_moving_dummy', 'uptake_gain_vs_moving_dummy',
        'wins_margin_vs_moving_dummy', 'wins_uptake_vs_moving_dummy',
    ])
    dummy = {
        int(row['seed']): row for row in grouped.get('moving_dummy_equal_cost', [])
    }
    output = []
    for condition, items in grouped.items():
        record = {
            'condition': condition,
            'n': len(items),
            'survived': sum(int(float(row['cells']) > 0) for row in items),
            'finite': sum(int(float(row['finite']) > 0) for row in items),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in items], dtype=float)
            record[metric + '_mean'] = float(np.mean(values))
            record[metric + '_sd'] = float(np.std(values, ddof=0))
        paired = [(row, dummy.get(int(row['seed']))) for row in items]
        paired = [(a, b) for a, b in paired if b is not None]
        if paired:
            margin_diffs = [float(a['mean_margin_over_life']) - float(b['mean_margin_over_life']) for a, b in paired]
            uptake_diffs = [float(a['uptake_during_trial']) - float(b['uptake_during_trial']) for a, b in paired]
            record['margin_gain_vs_moving_dummy'] = float(np.mean(margin_diffs))
            record['uptake_gain_vs_moving_dummy'] = float(np.mean(uptake_diffs))
            record['wins_margin_vs_moving_dummy'] = sum(value > 0 for value in margin_diffs)
            record['wins_uptake_vs_moving_dummy'] = sum(value > 0 for value in uptake_diffs)
        else:
            record['margin_gain_vs_moving_dummy'] = ''
            record['uptake_gain_vs_moving_dummy'] = ''
            record['wins_margin_vs_moving_dummy'] = ''
            record['wins_uptake_vs_moving_dummy'] = ''
        output.append(record)
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in output:
            writer.writerow(record)


def _condition_map(rows):
    result = {}
    for row in rows:
        result.setdefault(row['condition'], {})[int(row['seed'])] = row
    return result


def write_report(rows):
    grouped = _condition_map(rows)
    neuron = grouped.get('moving_neuron', {})
    dummy = grouped.get('moving_dummy_equal_cost', {})
    paired_seeds = sorted(set(neuron) & set(dummy))
    margin_diffs = [
        float(neuron[seed]['mean_margin_over_life'])
        - float(dummy[seed]['mean_margin_over_life'])
        for seed in paired_seeds
    ]
    uptake_diffs = [
        float(neuron[seed]['uptake_during_trial'])
        - float(dummy[seed]['uptake_during_trial'])
        for seed in paired_seeds
    ]
    acceptance = bool(
        paired_seeds
        and all(value > 0.04 for value in margin_diffs)
        and all(value > 0.50 for value in uptake_diffs)
    )
    lines = [
        'SOMA-CELL 0.6-P1 engineering experiment',
        'build: {}'.format(p1.BUILD),
        'seeds: {}'.format(', '.join(map(str, SEEDS))),
        'seconds per trial: {}'.format(SECONDS),
        'trials: {}'.format(len(rows)),
        '',
        'P1 acceptance (moving neuron > equal-cost dummy): {}'.format('PASS' if acceptance else 'FAIL'),
        'paired margin gains: {}'.format(', '.join('{:+.6f}'.format(v) for v in margin_diffs)),
        'paired uptake gains: {}'.format(', '.join('{:+.6f}'.format(v) for v in uptake_diffs)),
        '',
        'Interpretation:',
        '- The dummy has the same tissue targets, maintenance/wear schedule, command magnitude and P0 port constraints.',
        '- Its direction is generated by an internal oscillator rather than physical chemistry.',
        '- Divergence in realised cost after action is allowed: resource acquisition itself determines whether future requests can be paid.',
        '- Predictor benefit is evaluated separately and is not required to claim the one-cell information-value result.',
        '',
        'Condition means:',
    ]
    for condition, items in grouped.items():
        values = list(items.values())
        lines.append(
            '{}: margin {:.6f}, uptake {:.6f}, distance {:.6f}, neural ATP {:.6f}, pred err {:.6f}'.format(
                condition,
                np.mean([float(r['mean_margin_over_life']) for r in values]),
                np.mean([float(r['uptake_during_trial']) for r in values]),
                np.mean([float(r['mean_patch_distance_over_life']) for r in values]),
                np.mean([float(r['p1_neural_atp_total']) for r in values]),
                np.mean([float(r['p1_prediction_error']) for r in values]),
            )
        )
    lines.extend([
        '',
        'Limits:',
        '- n=3 seeds is mechanism confirmation, not statistical generalisation.',
        '- One neuron has no recurrence and no three-factor sensorimotor plasticity; those remain P2 work.',
        '- The local predictor learns, but this experiment does not establish a consistent net behavioural benefit from prediction itself.',
        '- The controlled patch world is intentionally small and deterministic.',
    ])
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    run_experiment()
