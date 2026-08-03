# coding: utf-8
"""Sealed R2 holdout for SOMA-CELL 0.6.1 motor-pathway conservation.

The assay has two preregistered parts:

1. A stable material-conservative body-centred chemostat: no false actuator
   confirmation or conservation lease should occur.
2. The same chemostat after an unannounced physical actuator-gain loss.  Paid
   command-to-force probes must confirm the fault; a common-disturbance twin
   then compares the automatic motor-pathway lease with feedback withheld.

The bath transfers matter only from P2's explicit chemostat reservoirs and
moves ordinary particles.  It never changes cell position, ATP, genes, neural
weights, or the hidden correct direction.
"""
from __future__ import print_function
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('../0_6', '../0_6_p2', '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_1_pythonista as soma

PREREG_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_1_R2_PREREGISTRATION.json')
RESULTS_PATH = os.path.join(HERE, 'soma_cell_0_6_1_r2_holdout_results.csv')
SUMMARY_PATH = os.path.join(HERE, 'soma_cell_0_6_1_r2_holdout_summary.json')
REPORT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_1_R2_HOLDOUT_REPORT.txt')
DT = 1.0 / soma.SIM_HZ


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_preregistration():
    with open(PREREG_PATH, 'r', encoding='utf-8') as handle:
        prereg = json.load(handle)
    source = os.path.join(HERE, 'SOMA_CELL_0_6_1_pythonista.py')
    script = os.path.abspath(__file__)
    observed = {
        'source_sha256': sha256(source),
        'holdout_script_sha256': sha256(script),
    }
    expected = prereg['frozen_hashes']
    if observed != expected:
        raise RuntimeError('sealed hash mismatch: observed={} expected={}'.format(observed, expected))
    if soma.SCHEMA_VERSION != prereg['schema_version']:
        raise RuntimeError('schema mismatch: {} != {}'.format(soma.SCHEMA_VERSION, prereg['schema_version']))
    return prereg


def config(fault=False):
    # P2 cue-reversal mode provides explicit finite chemostat reservoirs.  All
    # automatic cue/reward pulses are disabled; the assay below performs only
    # reservoir-to-field material transfers around the current body.
    return soma.Formal061Config(
        diagnosis_mode=soma.DIAGNOSIS_PASSIVE,
        p2_environment=soma.p2.P2_ENV_CUE_REVERSAL,
        p2_cue_duration=0.0,
        p2_cue_delay=0.0,
        p2_reward_duration=0.0,
        p2_cue_period=1000.0,
        p2_reward_fuel_pulse=0.0,
        p2_reward_mineral_pulse=0.0,
        p2_cue_alt_pulse=0.0,
        p2_assay_trap_strength=0.0,
        p2_prediction=False,
        division=False,
        mutation=False,
        diagnosis_min_active_age=999.0,
        audit_min_active_age=999.0,
        formal_auto_start=False,
        corpse_chemistry=False,
        extracellular_dna=False,
        mobile_elements=False,
        mechanism_probe_enabled=True,
        mechanism_assist_enabled=False,
        mechanism_conservation_enabled=True,
        mechanism_conservation_feedback_enabled=not bool(fault),
        mechanism_fault_enabled=bool(fault),
        mechanism_fault_age=34.0,
        mechanism_fault_gain_scale=0.25,
        mechanism_fault_wear=0.00020,
    )


def body_centred_bath(world, fuel_target=0.18, mineral_target=0.08):
    alive = world.living_cells()
    if not alive:
        return
    cell = alive[0]
    centre = np.asarray(cell.pos, dtype=float)
    radius = float(cell.radius)
    field = world.field

    def amount(kind):
        indices = np.where(field.kind == int(kind))[0]
        if indices.size == 0:
            return 0.0
        return float(np.sum(np.maximum(0.0, field.amount[indices])))

    fuel = amount(soma.s5.PARTICLE_FUEL)
    mineral = amount(soma.s5.PARTICLE_MINERAL)
    if fuel < 0.45 * fuel_target and world.p2_reward_reservoir_fuel > 1e-12:
        world._release_kind(
            soma.s5.PARTICLE_FUEL, 'p2_reward_reservoir_fuel',
            fuel_target - fuel, centre, 24, radius + 0.006,
        )
    if mineral < 0.45 * mineral_target and world.p2_reward_reservoir_mineral > 1e-12:
        world._release_kind(
            soma.s5.PARTICLE_MINERAL, 'p2_reward_reservoir_mineral',
            mineral_target - mineral, centre, 18, radius + 0.010,
        )
    world._move_kind_to(
        soma.s5.PARTICLE_FUEL, centre, DT, strength=18.0,
        radius=radius + 0.006,
    )
    world._move_kind_to(
        soma.s5.PARTICLE_MINERAL, centre, DT, strength=18.0,
        radius=radius + 0.010,
    )
    # Waste stays material and is merely advected away from the local bath.
    waste_centre = (centre + np.asarray([0.38, 0.38], dtype=float)) % 1.0
    world._move_kind_to(
        soma.s5.PARTICLE_WASTE, waste_centre, DT,
        strength=5.0, radius=0.03,
    )


def margin(world):
    alive = world.living_cells()
    return float(alive[0].autopoietic_margin()) if alive else 0.0


def cell_pool(world):
    alive = world.living_cells()
    if not alive:
        return 0.0
    cell = alive[0]
    return float(
        cell.pools[soma.s5.POOL_FUEL]
        + cell.pools[soma.s5.POOL_MINERAL]
    )


def stable_trial(seed, prereg):
    started = time.time()
    world = soma.Formal061World(
        seed=seed, initial_cells=1, config=config(fault=False),
    )
    steps = int(round(prereg['stable']['seconds'] * soma.SIM_HZ))
    for step in range(steps):
        if not world.living_cells():
            break
        body_centred_bath(world)
        soma.apply_061_common_disturbance_tape(
            world, seed, step, stream=prereg['streams']['stable'],
        )
        world.step(DT)
    summary = world.summary()
    return {
        'kind': 'stable',
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'age': float(world.age),
        'alive': int(bool(world.living_cells())),
        'finite': int(world.finite()),
        'mechanism_confirmed': int(summary['mechanism_confirmed']),
        'probe_completed': int(summary['mechanism_probe_completed']),
        'lease_issued': int(summary['mechanism_conservation_issued']),
        'margin': float(summary['mean_autopoietic_margin']),
        'matter_residual': float(summary['matter_residual']),
    }


def fault_twin(seed, prereg):
    started = time.time()
    base = soma.Formal061World(
        seed=seed, initial_cells=1, config=config(fault=True),
    )
    base.config.mechanism_conservation_feedback_enabled = False
    step = 0
    confirmed_age = None
    deadline = float(prereg['fault']['confirmation_deadline'])
    while base.age < deadline and base.living_cells():
        body_centred_bath(base)
        soma.apply_061_common_disturbance_tape(
            base, seed, step, stream=prereg['streams']['fault_baseline'],
        )
        base.step(DT)
        step += 1
        tissue = getattr(base.living_cells()[0], 'p2_tissue', None) if base.living_cells() else None
        if tissue is not None and tissue.mechanism_gain.confirmed:
            confirmed_age = float(base.age)
            break

    base_summary = base.summary()
    if confirmed_age is None:
        return {
            'kind': 'fault', 'seed': int(seed),
            'wall_seconds': time.time() - started,
            'confirmed': 0,
            'confirmed_age': -1.0,
            'probe_completed': int(base_summary['mechanism_probe_completed']),
            'lease_treatment': 0,
            'margin_auc_diff': 0.0,
            'final_margin_diff': 0.0,
            'cell_pool_diff': 0.0,
            'dissipated_energy_diff': 0.0,
            'matter_residual_treatment': float(base_summary['matter_residual']),
            'matter_residual_control': float(base_summary['matter_residual']),
            'finite': int(base.finite()),
        }

    treatment = base.clone()
    control = base.clone()
    treatment.config.mechanism_conservation_feedback_enabled = True
    control.config.mechanism_conservation_feedback_enabled = False
    start_t = cell_pool(treatment)
    start_c = cell_pool(control)
    auc_t = 0.0
    auc_c = 0.0
    end_age = confirmed_age + float(prereg['fault']['evaluation_seconds'])
    while (
        treatment.age < end_age
        and treatment.living_cells()
        and control.living_cells()
    ):
        body_centred_bath(treatment)
        body_centred_bath(control)
        soma.apply_061_common_disturbance_tape(
            treatment, seed, step, stream=prereg['streams']['fault_twin'],
        )
        soma.apply_061_common_disturbance_tape(
            control, seed, step, stream=prereg['streams']['fault_twin'],
        )
        treatment.step(DT)
        control.step(DT)
        step += 1
        auc_t += margin(treatment) * DT
        auc_c += margin(control) * DT

    st = treatment.summary()
    sc = control.summary()
    return {
        'kind': 'fault',
        'seed': int(seed),
        'wall_seconds': time.time() - started,
        'confirmed': 1,
        'confirmed_age': float(confirmed_age),
        'probe_completed': int(base_summary['mechanism_probe_completed']),
        'lease_treatment': int(st['mechanism_conservation_issued']),
        'margin_auc_diff': float(auc_t - auc_c),
        'final_margin_diff': float(
            st['mean_autopoietic_margin'] - sc['mean_autopoietic_margin']
        ),
        'cell_pool_diff': float(
            (cell_pool(treatment) - start_t) - (cell_pool(control) - start_c)
        ),
        'dissipated_energy_diff': float(
            treatment.dissipated_energy - control.dissipated_energy
        ),
        'suppressed_motor_request': float(
            st['mechanism_conservation_suppressed_motor_request']
        ),
        'matter_residual_treatment': float(st['matter_residual']),
        'matter_residual_control': float(sc['matter_residual']),
        'finite': int(treatment.finite() and control.finite()),
    }


def read_existing():
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, 'r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_rows(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(RESULTS_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key):
    return float(row.get(key, 0.0) or 0.0)


def as_int(row, key):
    return int(float(row.get(key, 0) or 0))


def summarise(rows, prereg):
    stable = [r for r in rows if r['kind'] == 'stable']
    fault = [r for r in rows if r['kind'] == 'fault']
    confirmed = [r for r in fault if as_int(r, 'confirmed')]
    positive = [r for r in confirmed if as_float(r, 'margin_auc_diff') > 0.0]
    max_residual = max([
        abs(as_float(r, 'matter_residual')) for r in stable
    ] + [
        abs(as_float(r, 'matter_residual_treatment')) for r in fault
    ] + [
        abs(as_float(r, 'matter_residual_control')) for r in fault
    ] or [0.0])
    summary = {
        'build': soma.BUILD,
        'schema': soma.SCHEMA_VERSION,
        'source_sha256': sha256(os.path.join(HERE, 'SOMA_CELL_0_6_1_pythonista.py')),
        'holdout_script_sha256': sha256(os.path.abspath(__file__)),
        'stable_n': len(stable),
        'stable_false_confirmations': sum(as_int(r, 'mechanism_confirmed') for r in stable),
        'stable_false_leases': sum(as_int(r, 'lease_issued') for r in stable),
        'stable_nonfinite': sum(1 - as_int(r, 'finite') for r in stable),
        'fault_n': len(fault),
        'fault_confirmed': len(confirmed),
        'fault_leases': sum(as_int(r, 'lease_treatment') > 0 for r in confirmed),
        'fault_positive_margin_pairs': len(positive),
        'fault_mean_margin_auc_diff': float(np.mean([
            as_float(r, 'margin_auc_diff') for r in confirmed
        ])) if confirmed else 0.0,
        'fault_mean_final_margin_diff': float(np.mean([
            as_float(r, 'final_margin_diff') for r in confirmed
        ])) if confirmed else 0.0,
        'fault_mean_pool_diff': float(np.mean([
            as_float(r, 'cell_pool_diff') for r in confirmed
        ])) if confirmed else 0.0,
        'fault_mean_dissipated_energy_diff': float(np.mean([
            as_float(r, 'dissipated_energy_diff') for r in confirmed
        ])) if confirmed else 0.0,
        'fault_nonfinite': sum(1 - as_int(r, 'finite') for r in fault),
        'max_abs_material_residual': float(max_residual),
    }
    gates = prereg['acceptance']
    checks = {
        'stable_complete': len(stable) == len(prereg['stable']['seeds']),
        'fault_complete': len(fault) == len(prereg['fault']['seeds']),
        'stable_false_leases': summary['stable_false_leases'] <= gates['max_stable_false_leases'],
        'stable_false_confirmations': summary['stable_false_confirmations'] <= gates['max_stable_false_confirmations'],
        'fault_confirmed': summary['fault_confirmed'] >= gates['min_fault_confirmations'],
        'fault_leases': summary['fault_leases'] >= gates['min_fault_leases'],
        'positive_margin_pairs': summary['fault_positive_margin_pairs'] >= gates['min_positive_margin_pairs'],
        'mean_margin_positive': summary['fault_mean_margin_auc_diff'] > gates['min_mean_margin_auc_diff'],
        'mean_dissipation_reduced': summary['fault_mean_dissipated_energy_diff'] < gates['max_mean_dissipated_energy_diff'],
        'finite': summary['stable_nonfinite'] == 0 and summary['fault_nonfinite'] == 0,
        'material_ledger': summary['max_abs_material_residual'] < gates['max_abs_material_residual'],
    }
    summary['checks'] = checks
    summary['pass'] = bool(all(checks.values()))
    return summary


def main():
    prereg = load_preregistration()
    rows = read_existing()
    completed = {(r['kind'], int(float(r['seed']))) for r in rows}
    for seed in prereg['stable']['seeds']:
        key = ('stable', int(seed))
        if key in completed:
            continue
        row = stable_trial(int(seed), prereg)
        rows.append(row)
        write_rows(rows)
        print('stable', seed, 'lease', row['lease_issued'], flush=True)
    for seed in prereg['fault']['seeds']:
        key = ('fault', int(seed))
        if key in completed:
            continue
        row = fault_twin(int(seed), prereg)
        rows.append(row)
        write_rows(rows)
        print('fault', seed, 'confirmed', row['confirmed'], 'auc', row['margin_auc_diff'], flush=True)
    summary = summarise(rows, prereg)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    lines = [
        'SOMA-CELL 0.6.1 R2 SEALED HOLDOUT REPORT',
        'Build: {}'.format(soma.BUILD),
        'Schema: {}'.format(soma.SCHEMA_VERSION),
        '', json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), ''
    ]
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not summary['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
