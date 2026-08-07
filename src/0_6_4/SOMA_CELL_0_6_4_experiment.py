# coding: utf-8
"""Preregistered amortization-boundary experiments for SOMA-CELL 0.6.4.

Each physical-world trial runs in a fresh spawned interpreter so that mutable
configuration used by one neural profile cannot contaminate another profile.
A failed or timed-out trial is recorded rather than silently omitted.
"""
from __future__ import division
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_3', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_4_pythonista as soma

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PREREG = os.path.join(ROOT, 'results', 'SOMA_CELL_0_6_4_PREREGISTRATION.json')
RESULTS = os.path.join(HERE, 'soma_cell_0_6_4_experiment_results.csv')
SUMMARY = os.path.join(HERE, 'soma_cell_0_6_4_experiment_summary.csv')
CONSOLIDATED = os.path.join(HERE, 'soma_cell_0_6_4_experiment_consolidated.json')
REPORT = os.path.join(HERE, 'SOMA_CELL_0_6_4_EXPERIMENT_REPORT.txt')

POLICY_BY_NAME = {
    soma.AMORT_CONDITIONAL: soma.AMORT_CONDITIONAL,
    soma.AMORT_OPTION_NONE: soma.AMORT_OPTION_NONE,
    soma.AMORT_BARE: soma.AMORT_BARE,
    soma.AMORT_EAGER: soma.AMORT_EAGER,
    soma.AMORT_ALWAYS_E2: soma.AMORT_ALWAYS_E2,
    soma.AMORT_RANDOM: soma.AMORT_RANDOM,
}


def load_prereg():
    with open(PREREG, encoding='utf-8') as handle:
        return json.load(handle)


def source_digest():
    with open(os.path.join(HERE, 'SOMA_CELL_0_6_4_pythonista.py'), 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def base_config(policy, option_fraction, task, stable=False):
    common = dict(
        amortization_policy=policy,
        preparedness_option_fraction=float(option_fraction),
        division=False,
        mutation=False,
        environmental_damage=False,
        external_inflow=False,
        formal_auto_start=False,
        audit_min_active_age=999.0,
        diagnosis_min_active_age=999.0,
        task_label=str(task.get('task_label', 'registered')),
    )
    if stable:
        common.update(dict(
            p2_environment=soma.p2.P2_ENV_MOVING_PATCH,
            rule_mode=soma.RULE_SINGLE,
            rule_first_change_age=1e9,
            rule_change_period=1e9,
            cue_visibility=0.0,
            p2_switch_age=1e9,
        ))
    else:
        common.update(dict(
            p2_environment=soma.p2.P2_ENV_CUE_REVERSAL,
            rule_mode=str(task['rule_mode']),
            rule_first_change_age=float(task['rule_first_change_age']),
            rule_change_period=float(task['rule_change_period']),
            cue_visibility=float(task['cue_visibility']),
            p2_cue_delay=float(task['cue_delay']),
            p2_cue_period=float(task['cue_period']),
            p2_cue_duration=float(task['cue_duration']),
            p2_reward_duration=float(task['reward_duration']),
        ))
    return soma.Formal064Config(**common)


def trial(spec):
    start = time.time()
    kind = str(spec['kind'])
    policy = str(spec['policy'])
    option = float(spec.get('option_fraction', 0.95))
    task = dict(spec.get('task', {}))
    stable = kind == 'stable'
    cfg = base_config(policy, option, task, stable=stable)
    result = soma.run_headless_trial(
        seed=int(spec['seed']), seconds=float(spec['seconds']),
        initial_cells=1, config=cfg,
    )
    result.update({
        'kind': kind,
        'condition': str(spec['condition']),
        'policy': policy,
        'option_fraction': option,
        'seed': int(spec['seed']),
        'seconds': float(spec['seconds']),
        'task_label': str(task.get('task_label', kind)),
        'wall_seconds': float(time.time() - start),
        'error': '',
    })
    return result


def _child(spec, output):
    try:
        output.put(('ok', trial(spec)))
    except Exception as exc:
        output.put(('error', {
            'kind': spec.get('kind', ''),
            'condition': spec.get('condition', ''),
            'policy': spec.get('policy', ''),
            'seed': spec.get('seed', ''),
            'seconds': spec.get('seconds', ''),
            'option_fraction': spec.get('option_fraction', ''),
            'task_label': spec.get('task', {}).get('task_label', ''),
            'error': '{}: {}'.format(type(exc).__name__, exc),
            'traceback': traceback.format_exc()[-2200:],
        }))


def isolated_trial(spec, timeout_seconds=95.0):
    ctx = mp.get_context('spawn')
    output = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child, args=(spec, output))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            'kind': spec.get('kind', ''), 'condition': spec.get('condition', ''),
            'policy': spec.get('policy', ''), 'seed': spec.get('seed', ''),
            'seconds': spec.get('seconds', ''),
            'option_fraction': spec.get('option_fraction', ''),
            'task_label': spec.get('task', {}).get('task_label', ''),
            'error': 'TIMEOUT after {:.1f}s'.format(timeout_seconds),
        }
    try:
        status, payload = output.get(timeout=2.0)
    except queue.Empty:
        return {
            'kind': spec.get('kind', ''), 'condition': spec.get('condition', ''),
            'policy': spec.get('policy', ''), 'seed': spec.get('seed', ''),
            'seconds': spec.get('seconds', ''),
            'option_fraction': spec.get('option_fraction', ''),
            'task_label': spec.get('task', {}).get('task_label', ''),
            'error': 'worker exited without result (exitcode={})'.format(process.exitcode),
        }
    return payload


def registered_task(prereg):
    frozen = prereg['frozen_candidate']
    return {
        'task_label': 'registered-holdout',
        'rule_mode': frozen['rule_mode'],
        'rule_first_change_age': frozen['rule_first_change_age'],
        'rule_change_period': frozen['rule_change_period'],
        'cue_visibility': frozen['cue_visibility'],
        'cue_delay': frozen['cue_delay'],
        'cue_period': frozen['cue_period'],
        'cue_duration': frozen['cue_duration'],
        'reward_duration': frozen['reward_duration'],
    }


def jobs(prereg):
    task = registered_task(prereg)
    candidate_option = float(prereg['frozen_candidate']['preparedness_option_fraction'])
    out = []
    for seed in prereg['development_boundary_assay']['seeds']:
        for fraction in prereg['development_boundary_assay']['option_fractions']:
            out.append({
                'kind': 'boundary', 'condition': 'conditional_{:.2f}'.format(fraction),
                'policy': soma.AMORT_CONDITIONAL, 'option_fraction': float(fraction),
                'seed': int(seed), 'seconds': float(prereg['development_boundary_assay']['seconds']),
                'task': task,
            })
    for seed in prereg['stable_safety']['seeds']:
        out.append({
            'kind': 'stable', 'condition': 'conditional',
            'policy': soma.AMORT_CONDITIONAL, 'option_fraction': candidate_option,
            'seed': int(seed), 'seconds': float(prereg['stable_safety']['seconds']),
            'task': {'task_label': 'stable-moving-patch'},
        })
    for seed in prereg['holdout']['seeds']:
        for policy in prereg['holdout']['policies']:
            option = candidate_option if policy in (soma.AMORT_CONDITIONAL, soma.AMORT_OPTION_NONE) else 1.0
            out.append({
                'kind': 'holdout', 'condition': policy, 'policy': policy,
                'option_fraction': option, 'seed': int(seed),
                'seconds': float(prereg['frozen_candidate']['trial_seconds']),
                'task': task,
            })
    return out


def write_rows(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(RESULTS, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            if value.strip().lower() == 'true':
                return 1.0
            if value.strip().lower() == 'false':
                return 0.0
        return float(value or default)
    except Exception:
        return float(default)


def grouped(rows, kind, condition):
    return [r for r in rows if r.get('kind') == kind and r.get('condition') == condition and not r.get('error')]


def paired(rows, treatment, control, metric):
    a = {int(float(r['seed'])): fnum(r, metric) for r in grouped(rows, 'holdout', treatment)}
    b = {int(float(r['seed'])): fnum(r, metric) for r in grouped(rows, 'holdout', control)}
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    return {
        'treatment': treatment, 'control': control, 'metric': metric,
        'seeds': seeds, 'differences': diffs, 'n': len(diffs),
        'mean_difference': float(np.mean(diffs)) if diffs else 0.0,
        'positive_pairs': int(sum(value > 0.0 for value in diffs)),
    }


def aggregate(rows, prereg):
    valid = [r for r in rows if not r.get('error')]
    summary_rows = []
    for kind in sorted(set(r.get('kind') for r in valid)):
        for condition in sorted(set(r.get('condition') for r in valid if r.get('kind') == kind)):
            items = grouped(valid, kind, condition)
            if not items:
                continue
            summary_rows.append({
                'kind': kind, 'condition': condition, 'n': len(items),
                'mean_margin_auc': float(np.mean([fnum(r, 'margin_auc') for r in items])),
                'mean_uptake_delta': float(np.mean([fnum(r, 'uptake_delta') for r in items])),
                'mean_developments': float(np.mean([fnum(r, 'neurogenesis_developments') for r in items])),
                'mean_investments': float(np.mean([fnum(r, 'preparedness_investments') for r in items])),
                'mean_option_level': float(np.mean([fnum(r, 'preparedness_mean_option_level') for r in items])),
                'mean_full_level': float(np.mean([fnum(r, 'preparedness_mean_level') for r in items])),
                'max_abs_material_residual': float(max(abs(fnum(r, 'material_residual')) for r in items)),
                'finite_runs': int(sum(bool(r.get('finite')) for r in items)),
            })
    with open(SUMMARY, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    fractions = prereg['development_boundary_assay']['option_fractions']
    boundary = []
    for fraction in fractions:
        condition = 'conditional_{:.2f}'.format(fraction)
        items = grouped(valid, 'boundary', condition)
        boundary.append({
            'fraction': float(fraction), 'n': len(items),
            'developed': int(sum(fnum(r, 'neurogenesis_developments') > 0 for r in items)),
            'mean_margin_auc': float(np.mean([fnum(r, 'margin_auc') for r in items])) if items else 0.0,
        })
    feasible = [item['fraction'] for item in boundary if item['n'] and item['developed'] >= 2]
    minimum_feasible = min(feasible) if feasible else None

    stable = grouped(valid, 'stable', 'conditional')
    stable_false_investments = int(sum(fnum(r, 'preparedness_investments') > 0 for r in stable))
    stable_false_developments = int(sum(fnum(r, 'neurogenesis_developments') > 0 for r in stable))

    comparisons = []
    for control in prereg['holdout']['pairwise_controls']:
        comparisons.append(paired(valid, soma.AMORT_CONDITIONAL, control, 'margin_auc'))
    required = 4
    performance_pass = bool(comparisons) and all(
        item['n'] == 6 and item['mean_difference'] > 0.0 and item['positive_pairs'] >= required
        for item in comparisons
    )
    engineering_pass = all(bool(r.get('finite')) for r in valid) and all(
        abs(fnum(r, 'material_residual')) < 3.0e-5 for r in valid
    ) and not [r for r in rows if r.get('error')]
    stable_pass = bool(len(stable) == 12 and stable_false_investments == 0 and stable_false_developments == 0)
    decision = 'PASS_REGION_FOUND' if performance_pass and stable_pass and engineering_pass else 'NEGATIVE_BOUNDARY_RESULT'
    consolidated = {
        'build': soma.BUILD, 'schema': soma.SCHEMA_VERSION,
        'source_sha256': source_digest(), 'preregistration': prereg,
        'trial_count': len(rows), 'error_count': len([r for r in rows if r.get('error')]),
        'engineering_pass': engineering_pass, 'stable_pass': stable_pass,
        'performance_pass': performance_pass, 'decision': decision,
        'boundary': boundary, 'minimum_feasible_option_fraction': minimum_feasible,
        'stable_false_investments': stable_false_investments,
        'stable_false_developments': stable_false_developments,
        'comparisons': comparisons,
    }
    with open(CONSOLIDATED, 'w', encoding='utf-8') as handle:
        json.dump(consolidated, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return consolidated, summary_rows


def write_report(consolidated, summary_rows):
    lines = [
        'SOMA-CELL 0.6.4 EXPERIMENT REPORT',
        'Build: {} / {}'.format(consolidated['build'], consolidated['schema']),
        'Decision: {}'.format(consolidated['decision']),
        '',
        'Engineering pass: {}'.format(consolidated['engineering_pass']),
        'Stable-safety pass: {}'.format(consolidated['stable_pass']),
        'Performance gate pass: {}'.format(consolidated['performance_pass']),
        'Trials: {} errors: {}'.format(consolidated['trial_count'], consolidated['error_count']),
        '',
        'Physical readiness boundary:',
    ]
    for item in consolidated['boundary']:
        lines.append('  option {:.2f}: developed {}/{} mean margin AUC {:.6f}'.format(
            item['fraction'], item['developed'], item['n'], item['mean_margin_auc']))
    lines.extend([
        '  minimum feasible option fraction: {}'.format(consolidated['minimum_feasible_option_fraction']),
        '',
        'Stable moving-patch safety:',
        '  false investments: {}'.format(consolidated['stable_false_investments']),
        '  false developments: {}'.format(consolidated['stable_false_developments']),
        '',
        'Preregistered holdout comparisons (conditional minus control):',
    ])
    for item in consolidated['comparisons']:
        lines.append('  vs {}: mean {:+.8f}, positive {}/{}'.format(
            item['control'], item['mean_difference'], item['positive_pairs'], item['n']))
    lines.extend([
        '',
        'Interpretation:',
        '- Option reserve is real material paid before evidence is available.',
        '- A high minimum feasible option fraction means late conditional preparation cannot escape early substrate depletion.',
        '- A negative gate retires or shrinks the grammar by default; it is not a universal proof that neural tissue is useless.',
        '- These are digital mechanism assays, not proof of life or consciousness.',
    ])
    with open(REPORT, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=95.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int, default=-1)
    args = parser.parse_args()
    prereg = load_prereg()
    if source_digest() != prereg['candidate_source_sha256']:
        raise SystemExit('candidate source SHA-256 differs from preregistration')
    specs = jobs(prereg)
    end = len(specs) if args.end_index < 0 else min(len(specs), args.end_index)
    selected = specs[max(0, args.start_index):end]
    rows = []
    for index, spec in enumerate(selected, max(0, args.start_index) + 1):
        row = isolated_trial(spec, timeout_seconds=args.timeout)
        rows.append(row)
        print('{}/{} {} {} {} margin={} dev={} error={}'.format(
            index, len(specs), row.get('kind'), row.get('seed'), row.get('condition'),
            row.get('margin_auc', ''), row.get('neurogenesis_developments', ''),
            row.get('error', '')), flush=True)
    # Full aggregation requires all registered rows. Partial runs are kept in a
    # shard file so a failed long job cannot destroy completed evidence.
    shard = RESULTS + '.{:03d}_{:03d}.jsonl'.format(max(0, args.start_index), end)
    with open(shard, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    if args.start_index == 0 and end == len(specs):
        write_rows(rows)
        consolidated, summary_rows = aggregate(rows, prereg)
        write_report(consolidated, summary_rows)
        print(json.dumps(consolidated, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
