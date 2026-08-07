# coding: utf-8
"""Preregistered mechanism experiments for SOMA-CELL 0.6.3.

The primary question is whether paid demand-driven transient neurogenesis
outperforms an equal-preparedness non-neural control in a short cue-reversal
assay.  A negative result is a valid release outcome.
"""
from __future__ import division
import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_2', '../0_6_1', '../0_6', '../0_6_p2',
            '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_3_pythonista as soma

PREREG = os.path.abspath(os.path.join(HERE, '..', '..', 'results',
                                      'SOMA_CELL_0_6_3_PREREGISTRATION.json'))
RESULTS = os.path.join(HERE, 'soma_cell_0_6_3_experiment_results.csv')
SUMMARY = os.path.join(HERE, 'soma_cell_0_6_3_experiment_summary.csv')
REPORT = os.path.join(HERE, 'SOMA_CELL_0_6_3_EXPERIMENT_REPORT.txt')
CONSOLIDATED = os.path.join(HERE, 'soma_cell_0_6_3_experiment_consolidated.json')


def load_prereg():
    with open(PREREG, encoding='utf-8') as handle:
        return json.load(handle)


def config_for(policy, environment, switch_age=10.0):
    return soma.Formal063Config(
        neurogenesis_policy=policy,
        p2_environment=environment,
        p2_switch_age=float(switch_age),
        division=False,
        mutation=False,
        environmental_damage=False,
        external_inflow=False,
        formal_auto_start=False,
        audit_min_active_age=999.0,
        diagnosis_min_active_age=999.0,
    )


def trial(kind, policy, seed, seconds, environment, switch_age=10.0):
    start = time.time()
    cfg = config_for(policy, environment, switch_age=switch_age)
    result = soma.run_headless_trial(
        seed=int(seed), seconds=float(seconds), initial_cells=1, config=cfg,
    )
    result.update({
        'kind': str(kind), 'policy': str(policy), 'seed': int(seed),
        'environment': str(environment), 'switch_age': float(switch_age),
        'wall_seconds': float(time.time() - start),
    })
    return result


def job(spec):
    kind, policy, seed, seconds, environment, switch_age = spec
    try:
        return trial(kind, policy, seed, seconds, environment, switch_age)
    except Exception as exc:
        return {
            'kind': kind, 'policy': policy, 'seed': seed,
            'environment': environment,
            'error': '{}: {}'.format(type(exc).__name__, exc),
            'traceback': traceback.format_exc()[-1800:],
        }


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
            lowered = value.strip().lower()
            if lowered == 'true':
                return 1.0
            if lowered == 'false':
                return 0.0
        return float(value or default)
    except Exception:
        return float(default)


def grouped(rows, kind, policy):
    return [r for r in rows if r.get('kind') == kind and r.get('policy') == policy and not r.get('error')]


def paired(rows, kind, treatment, control, metric):
    a = {int(float(r['seed'])): fnum(r, metric) for r in grouped(rows, kind, treatment)}
    b = {int(float(r['seed'])): fnum(r, metric) for r in grouped(rows, kind, control)}
    seeds = sorted(set(a) & set(b))
    diffs = [a[seed] - b[seed] for seed in seeds]
    return {
        'treatment': treatment, 'control': control, 'metric': metric,
        'seeds': seeds, 'differences': diffs,
        'mean_difference': float(np.mean(diffs)) if diffs else 0.0,
        'positive_pairs': int(sum(value > 0.0 for value in diffs)),
        'nonnegative_pairs': int(sum(value >= 0.0 for value in diffs)),
        'n': len(diffs),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        if row.get('error'):
            continue
        groups.setdefault((row.get('kind'), row.get('policy')), []).append(row)
    summaries = []
    metrics = (
        'margin_auc', 'uptake_delta', 'neurogenesis_developments',
        'neurogenesis_reabsorptions', 'neurogenesis_probe_positive',
        'neurogenesis_probe_nonpositive', 'neurogenesis_sentinel_atp',
        'neurogenesis_sentinel_wear', 'neurogenesis_organ_atp_total',
        'neurogenesis_organ_material_total', 'neurogenesis_prediction_atp_total',
        'neurogenesis_recurrence_atp_total', 'neurogenesis_plasticity_atp_total',
        'neurogenesis_returned_material', 'neurogenesis_returned_atp',
        'material_residual', 'finite',
    )
    for (kind, policy), items in sorted(groups.items()):
        summary = {'kind': kind, 'policy': policy, 'n': len(items)}
        for metric in metrics:
            values = [fnum(item, metric) for item in items if metric in item]
            if values:
                summary['mean_' + metric] = float(np.mean(values))
                summary['positive_' + metric] = int(sum(value > 0.0 for value in values))
        summary['max_abs_material_residual'] = max(
            [abs(fnum(item, 'material_residual')) for item in items] or [0.0]
        )
        summaries.append(summary)

    comparisons = {
        'confirmation_margin_demand_vs_prepared': paired(
            rows, 'confirmation', soma.POLICY_DEMAND,
            soma.POLICY_PREPARED_NONE, 'margin_auc'),
        'confirmation_margin_demand_vs_bare': paired(
            rows, 'confirmation', soma.POLICY_DEMAND,
            soma.POLICY_ALWAYS_NONE, 'margin_auc'),
        'confirmation_uptake_demand_vs_prepared': paired(
            rows, 'confirmation', soma.POLICY_DEMAND,
            soma.POLICY_PREPARED_NONE, 'uptake_delta'),
        'confirmation_margin_demand_vs_always_e2': paired(
            rows, 'confirmation', soma.POLICY_DEMAND,
            soma.POLICY_ALWAYS_E2, 'margin_auc'),
        'confirmation_margin_demand_vs_random': paired(
            rows, 'confirmation', soma.POLICY_DEMAND,
            soma.POLICY_RANDOM, 'margin_auc'),
    }
    stable = grouped(rows, 'stable', soma.POLICY_DEMAND)
    stable_triggers = int(sum(fnum(row, 'neurogenesis_developments') > 0.0 for row in stable))
    gate = load_prereg()['acceptance']
    primary = comparisons['confirmation_margin_demand_vs_prepared']
    positive = bool(
        primary['n'] == len(load_prereg()['confirmation']['seeds'])
        and primary['mean_difference'] > gate['minimum_mean_margin_auc_gain']
        and primary['positive_pairs'] >= gate['minimum_positive_margin_pairs']
        and stable_triggers <= gate['maximum_stable_false_developments']
    )
    interpretation = (
        'POSITIVE_WITH_LIMITS' if positive
        else 'NEGATIVE_RESULT_WITH_LIMITS'
    )
    consolidated = {
        'build': soma.BUILD, 'schema': soma.SCHEMA_VERSION,
        'summaries': summaries, 'comparisons': comparisons,
        'stable_false_developments': stable_triggers,
        'stable_n': len(stable), 'acceptance': gate,
        'scientific_status': interpretation,
        'errors': [row for row in rows if row.get('error')],
    }
    with open(CONSOLIDATED, 'w', encoding='utf-8') as handle:
        json.dump(consolidated, handle, ensure_ascii=False, indent=2, sort_keys=True)

    fields = []
    for row in summaries:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(SUMMARY, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        'SOMA-CELL 0.6.3 EXPERIMENT REPORT',
        'Build: {} | Schema: {}'.format(soma.BUILD, soma.SCHEMA_VERSION),
        'Scientific status: ' + interpretation,
        '',
        'Preregistered primary comparison:',
        json.dumps(primary, ensure_ascii=False, sort_keys=True),
        '',
        'Demand vs bare no-tissue:',
        json.dumps(comparisons['confirmation_margin_demand_vs_bare'], ensure_ascii=False, sort_keys=True),
        '',
        'Demand uptake vs prepared no-tissue:',
        json.dumps(comparisons['confirmation_uptake_demand_vs_prepared'], ensure_ascii=False, sort_keys=True),
        '',
        'Stable safety: false developments {}/{}'.format(stable_triggers, len(stable)),
        '',
        'Group summaries:',
    ]
    lines.extend(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in summaries)
    lines.extend([
        '',
        'Interpretation:',
        '- Primary inference uses demand vs prepared_no_tissue because both pay the latent gene/sentinel/precursor-readiness cost.',
        '- always_no_tissue is a secondary bare-body ceiling, not the cost-matched preparedness control.',
        '- A negative result means the transient organ did not repay its measured short-task cost; it does not establish that neural tissue is universally useless.',
        '- These are digital mechanism assays, not proof of life or consciousness.',
    ])
    with open(REPORT, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return consolidated


def specs_for(stage):
    prereg = load_prereg()
    if stage == 'screening':
        item = prereg['screening']
    elif stage == 'confirmation':
        item = prereg['confirmation']
    elif stage == 'stable':
        item = prereg['stable_safety']
    else:
        raise ValueError(stage)
    env = getattr(soma.p2, item['environment_constant'])
    return [
        (stage, policy, seed, item['seconds'], env, item.get('switch_age', 10.0))
        for policy in item['policies'] for seed in item['seeds']
    ]


def run_stage(stage, workers=4):
    specs = specs_for(stage)
    rows = []
    if os.path.exists(RESULTS):
        with open(RESULTS, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
    done = {
        (row.get('kind'), row.get('policy'), int(float(row.get('seed', -1))))
        for row in rows if not row.get('error')
    }
    todo = [spec for spec in specs if (spec[0], spec[1], spec[2]) not in done]
    print('{} jobs {}'.format(stage, len(todo)), flush=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(workers)), max_tasks_per_child=1,
    ) as pool:
        for row in pool.map(job, todo):
            rows.append(row)
            write_rows(rows)
            print(row.get('kind'), row.get('policy'), row.get('seed'),
                  'ERR' if row.get('error') else 'OK', flush=True)
    consolidated = summarize(rows)
    print(json.dumps(consolidated, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('screening', 'confirmation', 'stable', 'summarize'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'summarize':
        with open(RESULTS, newline='', encoding='utf-8') as handle:
            print(json.dumps(summarize(list(csv.DictReader(handle))), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    raise SystemExit(run_stage(args.stage, args.workers))
