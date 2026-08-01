# coding: utf-8
"""Comparative assays for SOMA-CELL 0.5.

Three deliberately different contexts are separated:

1. corpse_resource: a resource-poor cell encounters a clean or toxic body;
2. corpse_hgt: a controlled donor body carries six physical copies of one
   reaction gene missing from the recipient;
3. mobile_element: one cell begins with a materially integrated selfish gene
   and two nearby cells can acquire its exported polymer.

These are mechanism/fitness probes, not a claim of open-ended evolution.
"""
from __future__ import division

import csv
import gc
import os
import time
from collections import defaultdict

import numpy as np

import SOMA_CELL_0_5_pythonista as soma

BASE_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(BASE_DIR, 'soma_cell_0_5_experiment_results.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'soma_cell_0_5_experiment_summary.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'SOMA_CELL_0_5_EXPERIMENT_REPORT.txt')

SEEDS = (101, 202, 303)


def _reaction_gene(sequence, reaction):
    sequence = np.asarray(sequence, dtype=np.uint8)
    for spec in soma.g2.parse_genes(sequence):
        if spec['role'] == soma.ROLE_GENERIC and int(spec.get('reaction', -1)) == int(reaction):
            start = int(spec['start'])
            return sequence[start:start + soma.g2.GENE_SPAN].copy()
    raise AssertionError('reaction gene not found')


def _filter_proteins_to_genome(cell):
    cell._refresh_gene_cache()
    cell.proteins = {
        fingerprint: amount for fingerprint, amount in cell.proteins.items()
        if fingerprint in cell.gene_specs
    }
    cell._sync_protein_pool()
    cell._clean_control_state()


def _remove_ecology_protein(cell, kind):
    removed = 0.0
    for fingerprint, spec in list(cell.ecology_specs()):
        if int(spec['parameter']) % soma.ECOLOGY_COUNT == int(kind):
            removed += float(cell.proteins.pop(fingerprint, 0.0))
    cell.pools[soma.POOL_WASTE] += removed
    cell._sync_protein_pool()
    return removed


def _redistribute_protein(cell, kind, amount=0.030):
    target = None
    for fingerprint, spec in cell.ecology_specs():
        if int(spec['parameter']) % soma.ECOLOGY_COUNT == int(kind):
            target = fingerprint
            break
    if target is None:
        raise AssertionError('requested ecology protein gene absent')
    remaining = float(amount)
    for fingerprint in list(cell.proteins):
        if fingerprint == target:
            continue
        take = min(remaining, max(0.0, cell.proteins[fingerprint] - 1e-8))
        cell.proteins[fingerprint] -= take
        remaining -= take
        if remaining <= 1e-12:
            break
    moved = amount - remaining
    cell.proteins[target] = cell.proteins.get(target, 0.0) + moved
    cell._sync_protein_pool()
    return moved


def _reset_ledger(world):
    world.field.injected_material = 0.0
    world.field.dissipated_material = 0.0
    world.initial_total_material = world.total_material()


def _run_world(world, seconds, path_recipient=None):
    dt = 1.0 / soma.SIM_HZ
    margin_integral = 0.0
    living_time = 0.0
    first_path_age = None
    max_abs_residual = 0.0
    steps = int(round(float(seconds) * soma.SIM_HZ))
    for step in range(steps):
        world.step(dt)
        alive = world.living_cells()
        if alive:
            margin_integral += float(np.mean([
                cell.autopoietic_margin() for cell in alive
            ])) * dt
            living_time += dt
        if step % int(max(1, soma.SIM_HZ)) == 0 or step + 1 == steps:
            max_abs_residual = max(
                max_abs_residual, abs(world.matter_ledger_residual())
            )
        if path_recipient is not None and first_path_age is None:
            if any(soma.g2.sequence_has_novel_path(g) for g in path_recipient.genomes):
                first_path_age = float(world.age)
    return (
        margin_integral / max(dt, living_time),
        first_path_age,
        max_abs_residual,
    )


def _base_row(world, scenario, condition, seed, seconds, mean_margin,
              first_path_age, max_abs_residual, wall_seconds,
              recipient=None, initial_mobile_carriers=0):
    summary = world.summary()
    if recipient is None:
        recipient = world.living_cells()[0] if world.living_cells() else None
    recipient_novel = bool(
        recipient is not None
        and any(soma.g2.sequence_has_novel_path(g) for g in recipient.genomes)
    )
    recipient_length = int(len(recipient.genomes[0])) if recipient is not None and recipient.genomes else 0
    mobile_carriers = int(summary.get('cells_with_mobile_gene', 0))
    return {
        'scenario': scenario,
        'condition': condition,
        'seed': int(seed),
        'seconds': float(seconds),
        'wall_seconds': float(wall_seconds),
        'final_cells': int(summary['cells']),
        'births': int(summary['births']),
        'divisions': int(summary['divisions']),
        'deaths': int(summary['deaths']),
        'max_generation': int(summary['max_generation']),
        'mean_margin_over_time': float(mean_margin),
        'final_margin': float(summary['mean_autopoietic_margin']),
        'mean_atp': float(summary['mean_atp']),
        'mean_damage': float(summary['mean_damage']),
        'mean_loop': float(summary['mean_loop']),
        'corpse_material': float(summary['corpse_material']),
        'corpse_toxic_material': float(summary['corpse_toxic_material']),
        'necrophagy_mass': float(summary['necrophagy_mass']),
        'necrotoxin_uptake': float(summary['necrotoxin_uptake']),
        'edna_fragments': int(summary['edna_fragments']),
        'edna_material': float(summary['edna_material']),
        'hgt_attempts': int(summary['hgt_attempts']),
        'hgt_integrations': int(summary['hgt_integrations']),
        'hgt_digestions': int(summary['hgt_digestions']),
        'hgt_novel_path_events': int(summary['hgt_novel_path_events']),
        'recipient_novel_path': int(recipient_novel),
        'first_path_age': '' if first_path_age is None else float(first_path_age),
        'recipient_genome_length': recipient_length,
        'mean_genome_length': float(summary['mean_genome_length']),
        'max_genome_length': int(summary['max_genome_length']),
        'mobile_exports': int(summary['mobile_exports']),
        'mobile_carriers': mobile_carriers,
        'new_mobile_carriers': max(0, mobile_carriers - int(initial_mobile_carriers)),
        'ecology_atp_total': float(summary['ecology_atp_total']),
        'cells_with_foreign_information': int(summary['cells_with_foreign_information']),
        'total_material': float(summary['total_material']),
        'matter_residual': float(summary['matter_residual']),
        'max_abs_matter_residual': float(max_abs_residual),
    }


def run_corpse_resource(seed, condition):
    toxic = condition.startswith('toxic_')
    config_kwargs = dict(necrophagy_rate_scale=8.0, division=False, mutation=False)
    if condition.endswith('no_nutrition'):
        config_kwargs['corpse_nutrition'] = False
    if condition == 'clean_immediate_recycle':
        config_kwargs['immediate_dead_recycling'] = True
    world = soma.EcologicalWorld(
        seed=seed, initial_cells=2,
        config=soma.EcologyConfig(**config_kwargs),
    )
    recipient, donor = world.cells
    recipient.pos[:] = (0.50, 0.50)
    donor.pos[:] = (0.50, 0.50)
    # A deliberately resource-poor initial context makes corpse material
    # ecologically relevant without changing chemistry during the assay.
    world.field.amount *= 0.12
    recipient.pools[soma.POOL_FUEL] = 0.10
    recipient.pools[soma.POOL_MINERAL] = 0.12
    recipient.pools[soma.POOL_ATP] = 0.20
    if toxic:
        donor.pools[soma.POOL_WASTE] += 0.45
        donor.pools[soma.POOL_REACTIVE] += 0.10
        donor._sync_damage_pool()
    if condition == 'toxic_no_detox':
        _remove_ecology_protein(recipient, soma.ECO_DETOX)
    _reset_ledger(world)
    donor.alive = False
    donor.death_reason = 'controlled-resource-corpse'
    world._handle_divisions_and_deaths()
    start = time.time()
    mean_margin, first_path, max_residual = _run_world(world, 45.0)
    return _base_row(
        world, 'corpse_resource', condition, seed, 45.0,
        mean_margin, first_path, max_residual, time.time() - start,
        recipient=recipient,
    )


def run_corpse_hgt(seed, condition):
    kwargs = dict(
        corpse_decay_scale=25.0,
        dna_decay_scale=0.10,
        hgt_rate_scale=100.0,
        division=False,
        mutation=False,
    )
    if condition == 'hgt_no_edna':
        kwargs['extracellular_dna'] = False
    elif condition == 'hgt_no_recombination':
        kwargs['recombination'] = False
    elif condition == 'hgt_no_competence':
        kwargs['competence'] = False
    elif condition == 'hgt_no_restriction':
        kwargs['restriction'] = False
    world = soma.EcologicalWorld(
        seed=seed, initial_cells=2,
        config=soma.EcologyConfig(**kwargs),
    )
    recipient, donor = world.cells
    recipient.pos[:] = (0.50, 0.50)
    donor.pos[:] = (0.50, 0.50)
    recipient.genomes = [soma.founding_genome_05(complete_alt=False)]
    recipient.genome_lesions = [0.0]
    _filter_proteins_to_genome(recipient)
    target = _reaction_gene(
        soma.founding_genome_05(True), soma.g2.REACTION_INTERMEDIATE_TO_WASTE
    )
    # Six physical copies provide repeated opportunities; each failed uptake is
    # still digested/rejected and therefore not retried for free.
    donor.genomes = [target.copy() for _ in range(6)]
    donor.genome_lesions = [0.0] * 6
    _filter_proteins_to_genome(donor)
    _reset_ledger(world)
    donor.alive = False
    donor.death_reason = 'controlled-gene-donor-lysis'
    world._handle_divisions_and_deaths()
    if world.corpses:
        world.corpses[0].dna_release_progress = 0.95
    start = time.time()
    mean_margin, first_path, max_residual = _run_world(
        world, 25.0, path_recipient=recipient
    )
    return _base_row(
        world, 'corpse_hgt', condition, seed, 25.0,
        mean_margin, first_path, max_residual, time.time() - start,
        recipient=recipient,
    )


def run_mobile_element(seed, condition):
    kwargs = dict(hgt_rate_scale=80.0, mobile_rate_scale=200.0, division=False, mutation=False)
    if condition == 'mobile_export_off':
        kwargs['mobile_elements'] = False
    elif condition == 'mobile_no_restriction':
        kwargs['restriction'] = False
    world = soma.EcologicalWorld(
        seed=seed, initial_cells=3,
        config=soma.EcologyConfig(**kwargs),
    )
    positions = ((0.50, 0.50), (0.54, 0.50), (0.46, 0.50))
    for cell, position in zip(world.cells, positions):
        cell.pos[:] = position
        cell.pools[soma.POOL_NUCLEOTIDE] = max(cell.pools[soma.POOL_NUCLEOTIDE], 0.80)
        cell.pools[soma.POOL_ATP] = max(cell.pools[soma.POOL_ATP], 0.80)
    donor = world.cells[0]
    fragment = soma.DNAFragment(
        soma.mobile_element_sequence(), donor.pos,
        origin_lineage=-7, origin_cell=-7,
        origin_hash='controlled-mobile-inoculum', mobile=True,
    )
    world.edna.fragments.append(fragment)
    _reset_ledger(world)
    fragment = world.edna.fragments.pop()
    donor.integrate_fragment(fragment, world, force=True)
    _redistribute_protein(donor, soma.ECO_MOBILE, 0.030)
    # Prevent the donor from repeatedly re-importing its own exports so spread
    # to neighbouring recipients is measurable.
    _remove_ecology_protein(donor, soma.ECO_COMPETENCE)
    _reset_ledger(world)
    start = time.time()
    mean_margin, first_path, max_residual = _run_world(world, 45.0)
    return _base_row(
        world, 'mobile_element', condition, seed, 45.0,
        mean_margin, first_path, max_residual, time.time() - start,
        recipient=donor, initial_mobile_carriers=1,
    )


CONDITIONS = (
    ('corpse_resource', 'clean_full'),
    ('corpse_resource', 'clean_no_nutrition'),
    ('corpse_resource', 'clean_immediate_recycle'),
    ('corpse_resource', 'toxic_full'),
    ('corpse_resource', 'toxic_no_detox'),
    ('corpse_resource', 'toxic_no_nutrition'),
    ('corpse_hgt', 'hgt_full'),
    ('corpse_hgt', 'hgt_no_edna'),
    ('corpse_hgt', 'hgt_no_recombination'),
    ('corpse_hgt', 'hgt_no_competence'),
    ('corpse_hgt', 'hgt_no_restriction'),
    ('mobile_element', 'mobile_export_on'),
    ('mobile_element', 'mobile_export_off'),
    ('mobile_element', 'mobile_no_restriction'),
)


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scenario'], row['condition'])].append(row)
    metrics = (
        'wall_seconds', 'final_cells', 'deaths', 'mean_margin_over_time',
        'final_margin', 'mean_atp', 'mean_damage', 'mean_loop',
        'corpse_material', 'necrophagy_mass', 'necrotoxin_uptake',
        'edna_fragments', 'edna_material', 'hgt_attempts',
        'hgt_integrations', 'hgt_digestions', 'hgt_novel_path_events',
        'recipient_novel_path', 'recipient_genome_length',
        'mean_genome_length', 'max_genome_length', 'mobile_exports',
        'mobile_carriers', 'new_mobile_carriers', 'ecology_atp_total',
        'cells_with_foreign_information', 'matter_residual',
        'max_abs_matter_residual',
    )
    summary_rows = []
    for key in sorted(groups):
        items = groups[key]
        row = {
            'scenario': key[0],
            'condition': key[1],
            'trials': len(items),
            'survival_rate': float(np.mean([item['final_cells'] > 0 for item in items])),
            'novel_path_success_rate': float(np.mean([item['recipient_novel_path'] for item in items])),
            'mobile_spread_rate': float(np.mean([item['new_mobile_carriers'] > 0 for item in items])),
        }
        path_times = [float(item['first_path_age']) for item in items if item['first_path_age'] != '']
        row['mean_first_path_age'] = float(np.mean(path_times)) if path_times else ''
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            row['mean_' + metric] = float(np.mean(values))
            row['min_' + metric] = float(np.min(values))
            row['max_' + metric] = float(np.max(values))
        summary_rows.append(row)
    return summary_rows


def _write_report(rows, summaries):
    lookup = {(row['scenario'], row['condition']): row for row in summaries}
    def get(scenario, condition, metric):
        return lookup[(scenario, condition)][metric]

    lines = [
        'SOMA-CELL 0.5 comparative experiment report',
        'build: {}'.format(soma.BUILD),
        'trials: {} ({} seeds x {} conditions)'.format(
            len(rows), len(SEEDS), len(CONDITIONS)
        ),
        '',
        'CORPSE RESOURCE ASSAY (45 s)',
    ]
    for condition in (
        'clean_full', 'clean_no_nutrition', 'clean_immediate_recycle',
        'toxic_full', 'toxic_no_detox', 'toxic_no_nutrition',
    ):
        lines.append(
            '{:<26s} margin={:.6f} necrophagy={:.6f} toxin={:.6f} damage={:.6f}'.format(
                condition,
                get('corpse_resource', condition, 'mean_mean_margin_over_time'),
                get('corpse_resource', condition, 'mean_necrophagy_mass'),
                get('corpse_resource', condition, 'mean_necrotoxin_uptake'),
                get('corpse_resource', condition, 'mean_mean_damage'),
            )
        )
    lines.extend(['', 'CONTROLLED CORPSE-TO-RECIPIENT HGT ASSAY (25 s)'])
    for condition in (
        'hgt_full', 'hgt_no_edna', 'hgt_no_recombination',
        'hgt_no_competence', 'hgt_no_restriction',
    ):
        lines.append(
            '{:<26s} path_success={:.3f} integrations={:.3f} digestions={:.3f} genome={:.2f}'.format(
                condition,
                get('corpse_hgt', condition, 'novel_path_success_rate'),
                get('corpse_hgt', condition, 'mean_hgt_integrations'),
                get('corpse_hgt', condition, 'mean_hgt_digestions'),
                get('corpse_hgt', condition, 'mean_recipient_genome_length'),
            )
        )
    lines.extend(['', 'CONTROLLED MOBILE-ELEMENT ASSAY (45 s)'])
    for condition in ('mobile_export_on', 'mobile_export_off', 'mobile_no_restriction'):
        lines.append(
            '{:<26s} exports={:.3f} carriers={:.3f} spread_rate={:.3f} margin={:.6f} genome={:.2f}'.format(
                condition,
                get('mobile_element', condition, 'mean_mobile_exports'),
                get('mobile_element', condition, 'mean_mobile_carriers'),
                get('mobile_element', condition, 'mobile_spread_rate'),
                get('mobile_element', condition, 'mean_mean_margin_over_time'),
                get('mobile_element', condition, 'mean_mean_genome_length'),
            )
        )
    max_residual = max(abs(float(row['matter_residual'])) for row in rows)
    max_step_residual = max(abs(float(row['max_abs_matter_residual'])) for row in rows)
    lines.extend([
        '',
        'ACCOUNTING',
        'max final |matter residual|: {:.6e}'.format(max_residual),
        'max observed |matter residual|: {:.6e}'.format(max_step_residual),
        '',
        'Interpretation limits:',
        '- n=3 per condition is a mechanism comparison, not a statistical proof.',
        '- The HGT donor deliberately carries six copies of one missing gene.',
        '- The mobile-element donor begins with a controlled material inoculum and pre-existing mobile protein composition.',
        '- Reaction grammar and integration rules remain human-defined; this is not open-ended evolution.',
    ])
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _read_existing_results():
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, 'r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _normalise_row_types(row):
    integer_fields = {
        'seed', 'final_cells', 'births', 'divisions', 'deaths',
        'max_generation', 'edna_fragments', 'hgt_attempts',
        'hgt_integrations', 'hgt_digestions', 'hgt_novel_path_events',
        'recipient_novel_path', 'recipient_genome_length', 'max_genome_length',
        'mobile_exports', 'mobile_carriers', 'new_mobile_carriers',
        'cells_with_foreign_information',
    }
    numeric_fields = set(row) - {'scenario', 'condition', 'first_path_age'}
    converted = dict(row)
    for key in numeric_fields:
        if key in ('scenario', 'condition'):
            continue
        value = row[key]
        if value == '':
            continue
        converted[key] = int(float(value)) if key in integer_fields else float(value)
    if row.get('first_path_age', '') != '':
        converted['first_path_age'] = float(row['first_path_age'])
    return converted


def _write_summaries_from_disk():
    raw = _read_existing_results()
    rows = [_normalise_row_types(row) for row in raw]
    if not rows:
        return [], []
    summaries = _aggregate(rows)
    summary_fields = list(summaries[0].keys())
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    present = {(row['scenario'], row['condition']) for row in summaries}
    required = set(CONDITIONS)
    if required.issubset(present):
        _write_report(rows, summaries)
    return rows, summaries


def run_experiment():
    existing_raw = _read_existing_results()
    existing = {
        (row['scenario'], row['condition'], int(float(row['seed'])))
        for row in existing_raw
    }
    batch_start = max(0, int(os.environ.get('SOMA_BATCH_START', '0')))
    batch_end = min(
        len(CONDITIONS),
        int(os.environ.get('SOMA_BATCH_END', str(len(CONDITIONS))))
    )
    fieldnames = None
    if existing_raw:
        fieldnames = list(existing_raw[0].keys())

    for condition_index, (scenario, condition) in enumerate(CONDITIONS):
        if condition_index < batch_start or condition_index >= batch_end:
            continue
        for seed in SEEDS:
            key = (scenario, condition, int(seed))
            if key in existing:
                print('skip completed {} / {} / seed {}'.format(
                    scenario, condition, seed
                ), flush=True)
                continue
            print('{} / {} / seed {}'.format(scenario, condition, seed), flush=True)
            if scenario == 'corpse_resource':
                row = run_corpse_resource(seed, condition)
            elif scenario == 'corpse_hgt':
                row = run_corpse_hgt(seed, condition)
            else:
                row = run_mobile_element(seed, condition)
            if fieldnames is None:
                fieldnames = list(row.keys())
            write_header = not os.path.exists(RESULTS_PATH) or os.path.getsize(RESULTS_PATH) == 0
            with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            existing.add(key)
            gc.collect()

    rows, summaries = _write_summaries_from_disk()
    print('results on disk: {} trials'.format(len(rows)), flush=True)
    return rows, summaries


if __name__ == '__main__':
    run_experiment()
