# coding: utf-8
"""Engineering comparison for formal SOMA-CELL 0.6.

The study separates four questions:
1. broad cost/behaviour across formal-controller ablations;
2. stable-environment safety of the source-resolved change gate;
3. a known-mechanism positive-control twin in which a contradicted two-cell
   motor coalition is materially de-leased;
4. the harder automatic cue-reversal chain, reported even when it produces no
   feedback benefit.

The controlled positive assay is deliberately labelled as such.  It verifies
that *correctly targeted* material feedback can improve physical uptake; it is
not evidence that the natural auditor always discovers the correct claim.
"""
from __future__ import division

import csv
import json
import os
import sys
import time
import multiprocessing as mp
import subprocess
import concurrent.futures

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    HERE,
    os.path.abspath(os.path.join(HERE, '..', '0_6_p2')),
    os.path.abspath(os.path.join(HERE, '..', '0_6_p1')),
    os.path.abspath(os.path.join(HERE, '..', '0_6_p0')),
    os.path.abspath(os.path.join(HERE, '..', 'baseline')),
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_pythonista as soma

p2 = soma.p2

DEVELOPMENT_SEEDS = (101, 202, 303)
HOLDOUT_SEEDS = (404, 505, 606, 707, 808, 909)
CONTROLLED_SEEDS = (101, 202, 303, 404, 505, 606)
RESULT_PATH = os.path.join(HERE, 'soma_cell_0_6_experiment_results.csv')
SUMMARY_PATH = os.path.join(HERE, 'soma_cell_0_6_experiment_summary.csv')
JSON_PATH = os.path.join(HERE, 'soma_cell_0_6_experiment_results.json')
REPORT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_EXPERIMENT_REPORT.txt')

INDEPENDENT_MODES = (
    soma.FORMAL_MODE_FULL,
    soma.FORMAL_MODE_NO_AUDIT,
    soma.FORMAL_MODE_NO_CALIBRATION,
    soma.FORMAL_MODE_NO_CHANGE_GATE,
    soma.FORMAL_MODE_NO_FEEDBACK,
    soma.FORMAL_MODE_NO_FALSIFICATION,
)


def _mean_margin(world):
    cells = world.living_cells()
    if not cells:
        return 0.0
    return float(np.mean([cell.autopoietic_margin() for cell in cells]))


def _run_steps(world, seed, start_step, steps, stream):
    dt = 1.0 / soma.SIM_HZ
    auc = 0.0
    max_change = 0.0
    max_state = 0.0
    max_mechanism = 0.0
    max_semantic = 0.0
    action_x_auc = 0.0
    for offset in range(int(steps)):
        step = int(start_step) + offset
        soma.apply_formal_common_disturbance_tape(world, seed, step, stream=stream)
        world.step(dt)
        auc += _mean_margin(world) * dt
        cells = world.living_cells()
        if cells and isinstance(getattr(cells[0], 'p2_tissue', None), soma.FormalMaterialTissue):
            tissue = cells[0].p2_tissue
            sentinel = tissue.change_sentinel
            max_change = max(max_change, float(sentinel.global_probability))
            max_state = max(max_state, float(sentinel.state_probability))
            max_mechanism = max(max_mechanism, float(sentinel.mechanism_probability))
            max_semantic = max(max_semantic, float(sentinel.semantic_probability))
            action_x_auc += float(tissue.last_action[0]) * dt
    return {
        'margin_auc': auc,
        'max_change_probability': max_change,
        'max_state_probability': max_state,
        'max_mechanism_probability': max_mechanism,
        'max_semantic_probability': max_semantic,
        'action_x_auc': action_x_auc,
    }


def run_independent(mode, seed, seconds=45.0):
    config = soma.Formal06Config(
        formal_mode=mode,
        p2_environment=p2.P2_ENV_MOVING_PATCH,
        p2_patch_period=12.0,
        audit_min_active_age=12.0,
        audit_cooldown=6.0,
        audit_measure_duration=0.20,
        audit_washout_duration=0.04,
        change_warmup=10.0,
    )
    started = time.time()
    world = soma.Formal06World(seed=seed, initial_cells=1, config=config)
    metrics = _run_steps(
        world, seed, 0, int(round(seconds * soma.SIM_HZ)), stream=91,
    )
    summary = world.summary()
    row = {
        'trial_type': 'independent_stable',
        'seed_set': 'development',
        'condition': mode,
        'control': '',
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'mean_margin': metrics['margin_auc'] / max(seconds, 1e-9),
        'margin_auc': metrics['margin_auc'],
        'max_change_probability': metrics['max_change_probability'],
        'max_state_probability': metrics['max_state_probability'],
        'max_mechanism_probability': metrics['max_mechanism_probability'],
        'max_semantic_probability': metrics['max_semantic_probability'],
        'feedback_events': summary.get('formal_feedback_events', 0),
        'lease_reductions': summary.get('formal_lease_reductions', 0),
        'audits': summary.get('formal_audits', 0),
        'calibration_updates': summary.get('formal_calibration_updates', 0),
        'reward_uptake': summary.get('p2_reward_uptake_total', 0.0),
        'finite': int(world.finite()),
        'material_residual': world.matter_ledger_residual(),
    }
    return row


def run_stable_safety(seed, seconds=45.0):
    row = run_independent(soma.FORMAL_MODE_FULL, seed, seconds=seconds)
    row['trial_type'] = 'stable_safety'
    row['seed_set'] = 'holdout'
    return row


def _develop_controlled_world(seed):
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_MOVING_PATCH,
        p2_patch_period=1000.0,
        formal_auto_start=False,
        audit_min_active_age=999.0,
        change_warmup=0.0,
        p2_plasticity=False,
        p2_plasticity_warmup=999.0,
        sensorimotor=False,
        neural_atp_reserve=0.0,
        neural_fuel_reserve=0.0,
        neural_mineral_reserve=0.0,
        neural_membrane_reserve=0.0,
        neural_atp_rate=5.0,
        neural_protein_rate=5.0,
        neural_membrane_rate=5.0,
        neural_signal_rate=5.0,
    )
    world = soma.Formal06World(seed=seed, initial_cells=1, config=config)
    dt = 1.0 / soma.SIM_HZ
    used_steps = 0
    for used_steps in range(100):
        soma.apply_formal_common_disturbance_tape(world, seed, used_steps, stream=81)
        world.step(dt)
        tissue = world.living_cells()[0].p2_tissue
        if tissue.controller_mature and np.all(tissue.mature):
            break
    cell = world.living_cells()[0]
    tissue = cell.p2_tissue
    cell.pos[:] = [0.5, 0.5]
    cell.vel[:] = 0.0
    cell.surface_flux[:] = 0.0
    # Known-mechanism assay: two dominant cells have an inverted motor mapping;
    # six weaker cells retain the physically useful direction.  The two-cell
    # coalition is the audited claim group.  This setup is test scaffolding,
    # never a hidden runtime rule.
    tissue.w_sensor[:] = 0.0
    tissue.w_rec[:] = 0.0
    tissue.hidden[:] = 0.0
    tissue.prev_hidden[:] = 0.0
    tissue.bias[:] = 0.20
    tissue.tau[:] = 0.06
    tissue.homeo_gain[:] = 1.0
    tissue.probe_state[:] = 0.0
    tissue.motor_probe[:] = 0.0
    tissue.prediction_error[:] = 0.0
    tissue.reopen_reserve[:] = 0.0
    tissue.maturity[:] = 0.90
    tissue.resource_lease[:] = 1.0
    tissue.preferred[:] = [1.0, 0.0]
    tissue.motor_gain[:] = 0.30
    tissue.preferred[:2] = [-1.0, 0.0]
    tissue.motor_gain[:2] = 1.20
    tissue.change_sentinel.armed = True
    tissue.change_sentinel.mechanism_probability = 0.98
    tissue.change_sentinel.mechanism_channel_probability[0] = 0.98
    tissue.change_sentinel.channel_probability[0] = 0.98
    return world, used_steps + 1


def run_controlled_feedback_twin(seed, horizon=6.0):
    started = time.time()
    base, development_steps = _develop_controlled_world(seed)
    treatment = base.clone()
    control = base.clone()
    claim = np.zeros(p2.P2_CELL_COUNT, dtype=bool)
    claim[:2] = True
    tissue = treatment.living_cells()[0].p2_tissue
    magnitude = tissue._apply_feedback({
        'status': 'contradicted',
        'quality': 1.0,
        'reliability': 1.0,
        'claim_mask': claim,
        'channel': 0,
        'target_effect': -0.005,
    }, treatment.config)
    start_reward_t = treatment.p2_reward_uptake_total
    start_reward_c = control.p2_reward_uptake_total
    mt = _run_steps(
        treatment, seed, development_steps,
        int(round(horizon * soma.SIM_HZ)), stream=82,
    )
    mc = _run_steps(
        control, seed, development_steps,
        int(round(horizon * soma.SIM_HZ)), stream=82,
    )
    # The loops above each receive the same step-indexed tape.  They are run
    # sequentially but remain counterfactually paired by deterministic state.
    cell_t = treatment.living_cells()[0]
    cell_c = control.living_cells()[0]
    return {
        'trial_type': 'controlled_feedback_twin',
        'seed_set': 'controlled',
        'condition': 'verified_counterexample_feedback',
        'control': 'same_body_no_feedback',
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'feedback_magnitude': magnitude,
        'margin_auc_treatment': mt['margin_auc'],
        'margin_auc_control': mc['margin_auc'],
        'margin_auc_difference': mt['margin_auc'] - mc['margin_auc'],
        'reward_gain_treatment': treatment.p2_reward_uptake_total - start_reward_t,
        'reward_gain_control': control.p2_reward_uptake_total - start_reward_c,
        'reward_gain_difference': (
            treatment.p2_reward_uptake_total - start_reward_t
            - control.p2_reward_uptake_total + start_reward_c
        ),
        'action_x_auc_treatment': mt['action_x_auc'],
        'action_x_auc_control': mc['action_x_auc'],
        'action_x_auc_difference': mt['action_x_auc'] - mc['action_x_auc'],
        'x_displacement_treatment': float(p2.wrapped_delta(np.asarray([0.5, 0.5]), cell_t.pos)[0]),
        'x_displacement_control': float(p2.wrapped_delta(np.asarray([0.5, 0.5]), cell_c.pos)[0]),
        'final_margin_difference': _mean_margin(treatment) - _mean_margin(control),
        'finite_treatment': int(treatment.finite()),
        'finite_control': int(control.finite()),
        'material_residual_treatment': treatment.matter_ledger_residual(),
        'material_residual_control': control.matter_ledger_residual(),
    }


def run_automatic_reversal_twin(seed, horizon=24.0):
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        p2_switch_age=40.0,
        p2_plasticity_warmup=24.0,
        audit_min_active_age=18.0,
        audit_cooldown=5.0,
        audit_measure_duration=0.18,
        audit_washout_duration=0.04,
        change_warmup=14.0,
        change_semantic_min_episodes=4.0,
        change_semantic_window=2.5,
    )
    started = time.time()
    base = soma.Formal06World(seed=seed, initial_cells=1, config=config)
    pre_steps = int(round(config.p2_switch_age * soma.SIM_HZ))
    _run_steps(base, seed, 0, pre_steps, stream=101)
    treatment = base.clone()
    control = base.clone()
    soma.set_formal_runtime_mode(treatment, soma.FORMAL_MODE_FULL)
    soma.set_formal_runtime_mode(control, soma.FORMAL_MODE_NO_FEEDBACK)
    start_reward_t = treatment.p2_reward_uptake_total
    start_reward_c = control.p2_reward_uptake_total
    start_toxin_t = treatment.p2_cue_toxin_total
    start_toxin_c = control.p2_cue_toxin_total
    mt = _run_steps(
        treatment, seed, pre_steps,
        int(round(horizon * soma.SIM_HZ)), stream=102,
    )
    mc = _run_steps(
        control, seed, pre_steps,
        int(round(horizon * soma.SIM_HZ)), stream=102,
    )
    st = treatment.summary()
    sc = control.summary()
    return {
        'trial_type': 'automatic_reversal_twin',
        'seed_set': 'development',
        'condition': soma.FORMAL_MODE_FULL,
        'control': soma.FORMAL_MODE_NO_FEEDBACK,
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'margin_auc_treatment': mt['margin_auc'],
        'margin_auc_control': mc['margin_auc'],
        'margin_auc_difference': mt['margin_auc'] - mc['margin_auc'],
        'reward_gain_difference': (
            treatment.p2_reward_uptake_total - start_reward_t
            - control.p2_reward_uptake_total + start_reward_c
        ),
        'toxin_gain_difference': (
            treatment.p2_cue_toxin_total - start_toxin_t
            - control.p2_cue_toxin_total + start_toxin_c
        ),
        'feedback_events_treatment': st.get('formal_feedback_events', 0),
        'feedback_events_control': sc.get('formal_feedback_events', 0),
        'change_probability_treatment': st.get('formal_change_probability', 0.0),
        'audits_treatment': st.get('formal_audits', 0),
        'calibration_updates_treatment': st.get('formal_calibration_updates', 0),
        'finite_treatment': int(treatment.finite()),
        'finite_control': int(control.finite()),
        'material_residual_treatment': treatment.matter_ledger_residual(),
        'material_residual_control': control.matter_ledger_residual(),
    }


def _write_rows(rows):
    fields = sorted({key for row in rows for key in row})
    leading = ['trial_type', 'seed_set', 'condition', 'control', 'seed', 'wall_seconds']
    fields = leading + [key for key in fields if key not in leading]
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    with open(JSON_PATH, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def _aggregate(rows):
    groups = {}
    for row in rows:
        key = (row['trial_type'], row.get('seed_set', ''), row.get('condition', ''), row.get('control', ''))
        groups.setdefault(key, []).append(row)
    records = []
    for key, group in sorted(groups.items()):
        record = {
            'trial_type': key[0], 'seed_set': key[1],
            'condition': key[2], 'control': key[3], 'n': len(group),
        }
        numeric_keys = sorted({
            field for row in group for field, value in row.items()
            if field not in ('seed', 'wall_seconds')
            and isinstance(value, (int, float, np.integer, np.floating))
        })
        for field in numeric_keys:
            values = np.asarray([float(row[field]) for row in group if field in row], dtype=float)
            if values.size:
                record[field + '_mean'] = float(np.mean(values))
                record[field + '_sd'] = float(np.std(values))
        if any('margin_auc_difference' in row for row in group):
            values = [float(row.get('margin_auc_difference', 0.0)) for row in group]
            record['wins_positive'] = int(sum(value > 0.0 for value in values))
        if any('reward_gain_difference' in row for row in group):
            values = [float(row.get('reward_gain_difference', 0.0)) for row in group]
            record['reward_wins_positive'] = int(sum(value > 0.0 for value in values))
        records.append(record)
    fields = sorted({key for record in records for key in record})
    leading = ['trial_type', 'seed_set', 'condition', 'control', 'n', 'wins_positive', 'reward_wins_positive']
    fields = leading + [key for key in fields if key not in leading]
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, '') for key in fields})
    return records


def _find(records, trial_type, condition=None):
    for record in records:
        if record['trial_type'] != trial_type:
            continue
        if condition is not None and record['condition'] != condition:
            continue
        return record
    return {}


def _write_report(rows, records):
    stable = _find(records, 'stable_safety', soma.FORMAL_MODE_FULL)
    controlled = _find(records, 'controlled_feedback_twin', 'verified_counterexample_feedback')
    automatic = _find(records, 'automatic_reversal_twin', soma.FORMAL_MODE_FULL)
    lines = [
        'SOMA-CELL 0.6 EXPERIMENT REPORT',
        'Build: {}'.format(soma.BUILD),
        '',
        'Design:',
        '- 3 seeds x 6 independent formal-controller ablations in a stable moving-patch world.',
        '- 6 unseen/holdout stable-safety runs for false mechanism-change feedback.',
        '- 6 known-mechanism positive-control twins.',
        '- 3 automatic cue-reversal twins (full feedback vs feedback disabled).',
        '',
        'Stable safety:',
        '  max actionable change probability mean: {:.6g}'.format(stable.get('max_change_probability_mean', float('nan'))),
        '  feedback events mean: {:.6g}'.format(stable.get('feedback_events_mean', float('nan'))),
        '  lease reductions mean: {:.6g}'.format(stable.get('lease_reductions_mean', float('nan'))),
        '  state-only diagnostic probability may vary without opening the strong gate.',
        '',
        'Known-mechanism positive control:',
        '  margin AUC difference mean: {:+.9g}'.format(controlled.get('margin_auc_difference_mean', float('nan'))),
        '  positive margin pairs: {}/{}'.format(controlled.get('wins_positive', ''), controlled.get('n', '')),
        '  material uptake difference mean: {:+.9g}'.format(controlled.get('reward_gain_difference_mean', float('nan'))),
        '  positive uptake pairs: {}/{}'.format(controlled.get('reward_wins_positive', ''), controlled.get('n', '')),
        '  This is a controlled verified-counterexample assay, not the natural discovery chain.',
        '',
        'Automatic cue-reversal chain:',
        '  margin AUC difference mean: {:+.9g}'.format(automatic.get('margin_auc_difference_mean', float('nan'))),
        '  feedback events mean: {:.6g}'.format(automatic.get('feedback_events_treatment_mean', float('nan'))),
        '  automatic audits/calibration can run even when no high-quality change-gated counterexample is produced.',
        '',
        'Interpretation:',
        '- Source separation prevented routine metabolic state variation from opening strong re-plasticity in the stable holdouts.',
        '- Correctly targeted material lease withdrawal can improve physical uptake after a known motor-mechanism failure.',
        '- The fully automatic audit -> counterexample -> feedback -> benefit chain was not established by this short three-seed reversal assay.',
        '- These are mechanism studies, not proof of life, consciousness, or universal adaptive benefit.',
    ]
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _dispatch_task(task):
    kind, args = task
    if kind == 'independent':
        return run_independent(*args)
    if kind == 'stable':
        return run_stable_safety(*args)
    if kind == 'controlled':
        return run_controlled_feedback_twin(*args)
    if kind == 'automatic':
        return run_automatic_reversal_twin(*args)
    raise ValueError(kind)


def _task_specs():
    tasks = []
    for mode in INDEPENDENT_MODES:
        for seed in DEVELOPMENT_SEEDS:
            tasks.append(('independent', (mode, seed)))
    for seed in HOLDOUT_SEEDS:
        tasks.append(('stable', (seed,)))
    for seed in CONTROLLED_SEEDS:
        tasks.append(('controlled', (seed,)))
    for seed in DEVELOPMENT_SEEDS:
        tasks.append(('automatic', (seed,)))
    return tasks


def _run_task_child(task):
    kind, args = task
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--one', kind, json.dumps(list(args))],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=600,
    )
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith('SOMA_EXPERIMENT_JSON='):
            payload = json.loads(line.split('=', 1)[1])
            break
    if payload is None or completed.returncode != 0:
        raise RuntimeError(
            'task {} {} failed exit={} stdout={} stderr={}'.format(
                kind, args, completed.returncode,
                completed.stdout[-1200:], completed.stderr[-1200:],
            )
        )
    return payload


def run_fresh(workers=2):
    tasks = _task_specs()
    rows = []
    workers = max(1, min(int(workers), 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_task_child, task): task for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            row = future.result()
            rows.append(row)
            print('{}/{} {} {} seed {}'.format(
                index, len(tasks), row['trial_type'], row['condition'], row['seed'],
            ), flush=True)
    # Restore deterministic task order in artifacts.
    order = {(kind, tuple(args)): i for i, (kind, args) in enumerate(tasks)}
    def row_key(row):
        if row['trial_type'] == 'independent_stable':
            task = ('independent', (row['condition'], int(row['seed'])))
        elif row['trial_type'] == 'stable_safety':
            task = ('stable', (int(row['seed']),))
        elif row['trial_type'] == 'controlled_feedback_twin':
            task = ('controlled', (int(row['seed']),))
        else:
            task = ('automatic', (int(row['seed']),))
        return order[task]
    rows.sort(key=row_key)
    _write_rows(rows)
    records = _aggregate(rows)
    _write_report(rows, records)
    print('Wrote {} rows'.format(len(rows)), flush=True)
    return rows, records


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == '--one':
        kind = argv[1]
        args = tuple(json.loads(argv[2]))
        row = _dispatch_task((kind, args))
        print('SOMA_EXPERIMENT_JSON=' + json.dumps(row, ensure_ascii=False), flush=True)
        return 0
    workers = 2
    if len(argv) >= 2 and argv[0] == '--fresh':
        workers = int(argv[1])
    elif argv and argv[0] == '--fresh':
        workers = 2
    run_fresh(workers=workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
