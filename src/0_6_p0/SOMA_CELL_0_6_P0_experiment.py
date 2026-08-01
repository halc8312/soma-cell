# coding: utf-8
"""Engineering comparison experiment for SOMA-CELL 0.6-P0.

P0 deliberately contains no information-processing neuron.  The experiment
therefore asks engineering questions only:

* does an unattached or Null-attached P0 body remain exactly identical to 0.5?
* does repeated allocate/return remain physically neutral?
* can the paid effector API be stressed without numerical or material failure?

The scripted stress controller is an external test harness, not a SOMA neural
system and not evidence of adaptive benefit.
"""
from __future__ import division

import csv
import hashlib
import math
import os
import pickle
import statistics
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'baseline'))
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import SOMA_CELL_0_5_pythonista as baseline
import SOMA_CELL_0_6_P0_pythonista as p0

RESULTS_CSV = os.path.join(BASE_DIR, 'soma_cell_0_6_p0_experiment_results.csv')
SUMMARY_CSV = os.path.join(BASE_DIR, 'soma_cell_0_6_p0_experiment_summary.csv')
REPORT_TXT = os.path.join(BASE_DIR, 'SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt')

SEEDS = (101, 202, 303)
SECONDS = 5.0
INITIAL_CELLS = 2
CONDITIONS = (
    'frozen_05',
    'p0_unattached',
    'p0_null_attachment',
    'p0_budget_roundtrip',
    'p0_paid_effector_stress',
)


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            aa = np.asarray(a)
            bb = np.asarray(b)
            return aa.dtype == bb.dtype and aa.shape == bb.shape and np.array_equal(aa, bb)
        except Exception:
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            _equal(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return float(a) == float(b)
    return a == b


def _digest(state):
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _run_baseline(seed, seconds):
    world = baseline.EcologicalWorld(seed=seed, initial_cells=INITIAL_CELLS)
    dt = 1.0 / p0.SIM_HZ
    margin_integral = 0.0
    living_time = 0.0
    max_residual = 0.0
    for _ in range(int(round(seconds * p0.SIM_HZ))):
        world.step(dt)
        summary = world.summary()
        max_residual = max(max_residual, abs(world.matter_ledger_residual()))
        if summary['cells'] > 0:
            margin_integral += summary['mean_autopoietic_margin'] * dt
            living_time += dt
    return world, {
        'time_mean_margin': margin_integral / max(living_time, 1e-12),
        'max_abs_matter_residual': max_residual,
    }


def _attach_null_all(world):
    for cell in world.living_cells():
        if 'null' not in cell.neural_attachments:
            world.attach_null_tissue(cell.cell_id, 'null')


def _ensure_stress_attachment(world, cell):
    port = world.port_for(cell.cell_id)
    if 'stress' not in cell.neural_attachments:
        port.attach('stress', kind='scripted-interface-stress')
        port.allocate_budget(
            'stress',
            {'atp': 0.006, 'protein': 0.003, 'membrane': 0.0018, 'signal': 0.0016},
            0.10,
        )
        port.commit_material(
            'stress', protein=0.0020, membrane=0.0010, signal=0.0008,
            damaged_fraction=0.08, aggregate_fraction=0.04,
        )
        port.return_unused_budget('stress')
    return port


def _run_p0(seed, seconds, condition):
    world = p0.P0World(seed=seed, initial_cells=INITIAL_CELLS)
    dt = 1.0 / p0.SIM_HZ
    if condition == 'p0_null_attachment':
        _attach_null_all(world)
    if condition == 'p0_budget_roundtrip':
        for cell in world.living_cells():
            world.port_for(cell.cell_id).attach('roundtrip', kind='roundtrip-probe')

    margin_integral = 0.0
    living_time = 0.0
    max_residual = 0.0
    steps = int(round(seconds * p0.SIM_HZ))
    for step in range(steps):
        if condition == 'p0_budget_roundtrip':
            for cell in list(world.living_cells()):
                port = world.port_for(cell.cell_id)
                if 'roundtrip' not in cell.neural_attachments:
                    port.attach('roundtrip', kind='roundtrip-probe')
                port.allocate_budget(
                    'roundtrip',
                    {'atp': 0.0030, 'protein': 0.00030,
                     'membrane': 0.00020, 'signal': 0.00010},
                    dt,
                )
                port.return_unused_budget('roundtrip')
        elif condition == 'p0_paid_effector_stress':
            for cell in list(world.living_cells()):
                port = _ensure_stress_attachment(world, cell)
                port.allocate_budget(
                    'stress', {'atp': 0.0040, 'signal': 0.00080}, dt,
                )
                angle = 0.19 * step + 0.71 * cell.cell_id
                port.apply_effector_fluxes(
                    'stress',
                    {
                        'motor': (math.cos(angle), math.sin(angle)),
                        'transporter_polarity': (
                            math.cos(angle + 0.55), math.sin(angle + 0.55)
                        ),
                        'repair_polarity': (
                            math.cos(angle - 0.35), math.sin(angle - 0.35)
                        ),
                        'quiescence': 0.12 if (step // 24) % 2 else 0.0,
                    },
                    dt,
                )
                port.return_unused_budget('stress')
        world.step(dt)
        summary = world.summary()
        max_residual = max(max_residual, abs(world.matter_ledger_residual()))
        if summary['cells'] > 0:
            margin_integral += summary['mean_autopoietic_margin'] * dt
            living_time += dt
    return world, {
        'time_mean_margin': margin_integral / max(living_time, 1e-12),
        'max_abs_matter_residual': max_residual,
    }


def _row(condition, seed, seconds, world, metrics, baseline_state, baseline_rng):
    state = world.state_dict()
    if condition == 'frozen_05':
        physical_state = state
        summary = world.summary()
        exact_state = True
        exact_rng = True
        p0_summary = {}
    else:
        physical_state = p0.baseline_projection(world)
        summary = world.summary()
        exact_state = _equal(physical_state, baseline_state)
        exact_rng = world.rng.bit_generator.state == baseline_rng
        p0_summary = summary
    return {
        'condition': condition,
        'seed': int(seed),
        'seconds': float(seconds),
        'cells': int(summary['cells']),
        'divisions': int(summary['divisions']),
        'deaths': int(summary['deaths']),
        'time_mean_margin': float(metrics['time_mean_margin']),
        'final_margin': float(summary['mean_autopoietic_margin']),
        'final_atp': float(summary['mean_atp']),
        'matter_residual': float(summary['matter_residual']),
        'max_abs_matter_residual': float(metrics['max_abs_matter_residual']),
        'exact_baseline_state': int(bool(exact_state)),
        'exact_baseline_rng': int(bool(exact_rng)),
        'physical_digest': _digest(physical_state),
        'neural_attachments': int(p0_summary.get('neural_attachments', 0)),
        'neural_budget_calls': int(p0_summary.get('neural_budget_calls', 0)),
        'neural_effector_calls': int(p0_summary.get('neural_effector_calls', 0)),
        'neural_effector_rejections': int(p0_summary.get('neural_effector_rejections', 0)),
        'neural_allocated_atp': float(p0_summary.get('neural_allocated_atp', 0.0)),
        'neural_allocated_material': float(p0_summary.get('neural_allocated_material', 0.0)),
        'neural_returned_atp': float(p0_summary.get('neural_returned_atp', 0.0)),
        'neural_returned_material': float(p0_summary.get('neural_returned_material', 0.0)),
        'neural_spent_atp': float(p0_summary.get('neural_spent_atp', 0.0)),
        'neural_spent_signal': float(p0_summary.get('neural_spent_signal', 0.0)),
        'finite': int(bool(world.finite())),
    }


def run_experiment(seeds=SEEDS, seconds=SECONDS):
    rows = []
    for seed in seeds:
        baseline_world, baseline_metrics = _run_baseline(seed, seconds)
        baseline_state = baseline_world.state_dict()
        baseline_rng = baseline_world.rng.bit_generator.state
        rows.append(_row(
            'frozen_05', seed, seconds, baseline_world, baseline_metrics,
            baseline_state, baseline_rng,
        ))
        for condition in CONDITIONS[1:]:
            world, metrics = _run_p0(seed, seconds, condition)
            rows.append(_row(
                condition, seed, seconds, world, metrics,
                baseline_state, baseline_rng,
            ))

    fields = list(rows[0].keys())
    with open(RESULTS_CSV, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for condition in CONDITIONS:
        group = [row for row in rows if row['condition'] == condition]
        summaries.append({
            'condition': condition,
            'trials': len(group),
            'mean_time_margin': statistics.mean(row['time_mean_margin'] for row in group),
            'mean_final_margin': statistics.mean(row['final_margin'] for row in group),
            'mean_final_atp': statistics.mean(row['final_atp'] for row in group),
            'mean_cells': statistics.mean(row['cells'] for row in group),
            'mean_max_abs_residual': statistics.mean(row['max_abs_matter_residual'] for row in group),
            'exact_state_trials': sum(row['exact_baseline_state'] for row in group),
            'exact_rng_trials': sum(row['exact_baseline_rng'] for row in group),
            'finite_trials': sum(row['finite'] for row in group),
            'mean_budget_calls': statistics.mean(row['neural_budget_calls'] for row in group),
            'mean_effector_calls': statistics.mean(row['neural_effector_calls'] for row in group),
            'mean_spent_atp': statistics.mean(row['neural_spent_atp'] for row in group),
            'mean_spent_signal': statistics.mean(row['neural_spent_signal'] for row in group),
        })
    summary_fields = list(summaries[0].keys())
    with open(SUMMARY_CSV, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        'SOMA-CELL 0.6-P0 engineering comparison',
        'build: {}'.format(p0.BUILD),
        'seeds: {}'.format(', '.join(str(x) for x in seeds)),
        'seconds per trial: {}'.format(seconds),
        'trials: {}'.format(len(rows)),
        '',
        'P0 contains no neuron. Scripted actuation is an external interface stress harness.',
        '',
    ]
    for summary in summaries:
        lines.append(
            '{condition}: n={trials}, margin={mean_time_margin:.8f}, '
            'exact={exact_state_trials}/{trials}, rng={exact_rng_trials}/{trials}, '
            'finite={finite_trials}/{trials}, maxResidual={mean_max_abs_residual:.3e}, '
            'effectorCalls={mean_effector_calls:.1f}, spentATP={mean_spent_atp:.6f}'.format(
                **summary
            )
        )
    lines.extend([
        '',
        'Interpretation:',
        '- unattached, Null-attached and round-trip conditions should remain exact 0.5 physical twins;',
        '- paid stress is expected to diverge because it deliberately spends ATP/signal and actuates the body;',
        '- no adaptive-intelligence claim is made from this experiment.',
    ])
    with open(REPORT_TXT, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return rows, summaries


if __name__ == '__main__':
    run_experiment()
    print(open(REPORT_TXT, encoding='utf-8').read(), end='')
