# coding: utf-8
"""Deterministic mechanism validation for SOMA-CELL 0.6.

The suite validates the narrow engineering/scientific claims of the formal
0.6 layer: a gene-derived materially paid controller can run reversible
ABBA/BAAB neural-output interventions, estimate an effect against an
activity/motor-matched sham, calibrate its predictions, gate re-plasticity by
physical change evidence, and route its own matter through the existing
SOMA-CELL corpse/eDNA chemistry.  It does not establish life, consciousness,
or universal adaptive benefit.
"""
from __future__ import division

import csv
import hashlib
import os
import pickle
import tempfile
import traceback
import json
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p2'))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
import sys
for candidate in (HERE, P2_DIR, P1_DIR, P0_DIR, BASE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_pythonista as soma

p2 = soma.p2
p1 = soma.p1
p0 = soma.p0
s5 = soma.s5

CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_VALIDATION_RESULTS.txt')
P2_SOURCE_PATH = os.path.join(P2_DIR, 'SOMA_CELL_0_6_P2_pythonista.py')
if not os.path.isfile(P2_SOURCE_PATH):
    P2_SOURCE_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P2_pythonista.py')
P2_SHA256 = '03cd333838f2bee0dde634e8d55ca2b2c7ed285c9104a52a26bcee4a07ce1693'


def exact_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a, b)
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict) and isinstance(b, dict)
            and set(a) == set(b)
            and all(exact_equal(a[key], b[key]) for key in a)
        )
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            type(a) is type(b) and len(a) == len(b)
            and all(exact_equal(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return float(a) == float(b)
    return a == b


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return '{:.12g}'.format(float(value))
    return str(value)


def _record(rows, name, passed, observed, criterion):
    status = 'PASS' if bool(passed) else 'FAIL'
    print('[{}] {}'.format(status, name), flush=True)
    rows.append({
        'test': name,
        'pass': status,
        'observed': _fmt(observed),
        'criterion': str(criterion),
    })


def first_cell(world):
    living = world.living_cells()
    if not living:
        raise AssertionError('expected a living cell')
    return living[0]


def high_budget_config(**kwargs):
    values = dict(
        p2_tissue_mode=p2.P2_MODE_FULL,
        p2_environment=p2.P2_ENV_NATIVE,
        p2_effectors=False,
        p2_plasticity_warmup=999.0,
        sensorimotor=False,
        environmental_damage=False,
        external_inflow=False,
        neural_atp_reserve=0.0,
        neural_fuel_reserve=0.0,
        neural_mineral_reserve=0.0,
        neural_membrane_reserve=0.0,
        neural_atp_rate=4.0,
        neural_protein_rate=4.0,
        neural_membrane_rate=4.0,
        neural_signal_rate=4.0,
        audit_min_active_age=999.0,
        formal_auto_start=False,
    )
    values.update(kwargs)
    return soma.Formal06Config(**values)


def develop_formal(world, max_steps=480):
    dt = 1.0 / soma.SIM_HZ
    for step in range(max_steps):
        world.step(dt)
        cell = first_cell(world)
        tissue = cell.p2_tissue
        if (
            isinstance(tissue, soma.FormalMaterialTissue)
            and np.count_nonzero(tissue.mature) == p2.P2_CELL_COUNT
            and tissue.controller_mature
        ):
            return step + 1, tissue
    raise AssertionError('formal tissue/controller did not mature')


def _p2_projection(formal_world, reference_world):
    """Project formal state onto the exact P2 schema for lockstep tests."""
    source = formal_world.state_dict()
    result = dict(source)
    for key in list(result):
        if key.startswith('formal_'):
            result.pop(key)
    result['save_version'] = p2.SAVE_VERSION
    result['build'] = p2.BUILD
    p2_config_keys = set(reference_world.config.__dict__.keys())
    result['config'] = {
        key: value for key, value in source['config'].items()
        if key in p2_config_keys
    }
    reference_cell_state = reference_world.cells[0].state_dict()
    cell_keys = set(reference_cell_state.keys())
    tissue_keys = set(reference_cell_state['p2_tissue'].keys())
    projected_cells = []
    for cell_state in source['cells']:
        projected = {key: value for key, value in cell_state.items() if key in cell_keys}
        projected['cell_class'] = reference_cell_state['cell_class']
        if projected.get('p2_tissue') is not None:
            projected['p2_tissue'] = {
                key: value for key, value in projected['p2_tissue'].items()
                if key in tissue_keys
            }
        projected_cells.append(projected)
    result['cells'] = projected_cells
    return result


def _controller_material(world, cell):
    state = world.port_for(cell.cell_id).attachment_status(soma.FORMAL_CONTROLLER_ID)
    return float(np.sum(state['tissue_material']) + np.sum(state['stores'][1:]))


def _run_synthetic_audit(inject_hard_event=True):
    config = soma.Formal06Config(
        audit_measure_duration=0.10,
        audit_washout_duration=0.05,
        audit_max_retries=1,
        audit_cooldown=1.0,
        change_warmup=0.0,
    )
    auditor = soma.MaterialCausalAuditor(True)
    claim = np.zeros(p2.P2_CELL_COUNT, dtype=bool); claim[:2] = True
    sham = np.zeros(p2.P2_CELL_COUNT, dtype=bool); sham[2:4] = True
    program = {
        'family': 'controlled',
        'claim_mask': claim,
        'sham_mask': sham,
        'channel': 0,
        'raw_prediction': 0.003,
        'calibrated_prediction': 0.003,
        'reliability': 0.8,
        'commands': [{'op': 'gate_cells', 'count': 2, 'strength': 0.88}],
        'complexity': 1,
    }
    debt = np.ones(soma.FORMAL_CHANNELS, dtype=float) * 0.20
    context = np.zeros(soma.FORMAL_CONTEXT_DIM, dtype=float)
    rng = np.random.default_rng(3)
    assert auditor.start(program, debt, context, 10.0, rng, config)
    result = None
    injected = False
    dt = 0.025
    for step in range(600):
        rate = 0.0
        if auditor.active and auditor.stage == auditor.STAGE_MEASURE and auditor.current_condition() == 1:
            rate = 0.004 if auditor.current_arm() == 0 else 0.001
        debt = debt + np.asarray([rate, 0.0, 0.0, 0.0]) * dt
        flags = 0
        if inject_hard_event and not injected and auditor.active and auditor.stage == auditor.STAGE_MEASURE:
            flags = soma.AUDIT_EVENT_RESOURCE
            injected = True
        result = auditor.observe(
            debt, context, flags, 1.0, dt, 10.0 + step * dt,
            config, cost_atp=0.01, cost_material=0.001,
        )
        if result is not None:
            break
    if result is None:
        raise AssertionError('controlled audit did not complete')
    return auditor, result


def test_build_and_frozen_p2():
    with open(P2_SOURCE_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == P2_SHA256
    assert soma.BUILD == 'SOMA-CELL 0.6.0'
    assert soma.FORMAL_SCHEMA_VERSION.startswith('0.6-F')
    assert p2.BUILD == 'SOMA-CELL 0.6-P2.0'
    return 'build={}, schema={}, frozen P2 sha={}'.format(
        soma.BUILD, soma.FORMAL_SCHEMA_VERSION, digest,
    )


def test_no_formal_cassette_exact_p2_lockstep():
    observations = []
    for seed in (101, 202, 303):
        control = p2.P2World(
            seed=seed, initial_cells=1,
            config=p2.P2Config(p2_environment=p2.P2_ENV_CUE_REVERSAL),
        )
        candidate = soma.Formal06World(
            seed=seed, initial_cells=1,
            config=soma.Formal06Config(
                p2_environment=p2.P2_ENV_CUE_REVERSAL,
                formal_install_gene=False,
                formal_bootstrap_protein=False,
                audit_enabled=False,
                calibration_enabled=False,
                change_detection_enabled=False,
                feedback_enabled=False,
                falsification_enabled=False,
            ),
        )
        for _ in range(240):
            control.step(1.0 / p2.SIM_HZ)
            candidate.step(1.0 / soma.SIM_HZ)
        projection = _p2_projection(candidate, control)
        assert exact_equal(control.state_dict(), projection)
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        observations.append('{}:exact'.format(seed))
    return ', '.join(observations)


def test_unmetered_formal_controller_rejected():
    try:
        soma.Formal06Config(audit_controller_cost=False)
        raise AssertionError('unmetered controller was accepted')
    except ValueError:
        pass
    return 'costless formal control fails closed'


def test_material_gene_and_controller_development():
    no_gene = soma.Formal06World(
        seed=310, initial_cells=1,
        config=high_budget_config(
            formal_install_gene=False,
            formal_bootstrap_protein=False,
        ),
    )
    with_gene = soma.Formal06World(seed=310, initial_cells=1, config=high_budget_config())
    extra = with_gene.total_material() - no_gene.total_material()
    steps, tissue = develop_formal(with_gene)
    cell = first_cell(with_gene)
    assert extra > 0.0
    formal_controller_specs = soma.formal_controller_specs(cell)
    assert formal_controller_specs
    assert tissue.controller_mature
    assert soma.FORMAL_CONTROLLER_ID in cell.neural_attachments
    return 'extra founder matter={:.9g}, mature in {} steps, gene fp={}'.format(
        extra, steps, formal_controller_specs[0][0],
    )


def test_controller_activity_cost_is_paid_and_conservative():
    world = soma.Formal06World(seed=320, initial_cells=1, config=high_budget_config())
    _, tissue = develop_formal(world)
    cell = first_cell(world); port = world.port_for(cell.cell_id)
    before_total = world.total_material()
    before_atp = tissue.controller_atp_spent
    before_wear = tissue.controller_material_wear
    for _ in range(24):
        tissue._pay_controller_cost(port, 1.0 / soma.SIM_HZ, active=True, group_size=2)
    assert tissue.controller_atp_spent > before_atp
    assert tissue.controller_material_wear > before_wear
    assert abs(world.total_material() - before_total) < 2e-10
    return 'ATP={:.8g}, wear={:.8g}, material delta={:.3e}'.format(
        tissue.controller_atp_spent - before_atp,
        tissue.controller_material_wear - before_wear,
        world.total_material() - before_total,
    )


def test_counterbalanced_audit_and_hard_event_retry():
    auditor, result = _run_synthetic_audit(inject_hard_event=True)
    assert result['status'] == 'supported'
    assert abs(result['target_effect'] - 0.003) < 1e-10
    assert auditor.retried == 1
    assert len(auditor.records) == 8
    assert all(len(sequence) == 4 for sequence in auditor.period_sequence)
    assert all(np.array_equal(np.sort(sequence), np.asarray([0, 0, 1, 1])) for sequence in auditor.period_sequence)
    return 'effect={:.9g}, quality={:.6g}, retries={}'.format(
        result['target_effect'], result['quality'], auditor.retried,
    )


def test_natural_audit_completes_and_costs():
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        audit_min_active_age=5.0,
        audit_cooldown=2.0,
        audit_measure_duration=0.20,
        audit_washout_duration=0.04,
    )
    world = soma.Formal06World(seed=101, initial_cells=1, config=config)
    for _ in range(int(22 * soma.SIM_HZ)):
        world.step(1.0 / soma.SIM_HZ)
    summary = world.summary()
    assert summary['formal_audits'] >= 1
    assert summary['formal_audit_cost_atp'] > 0.0
    assert summary['formal_controller_atp'] > 0.0
    assert summary['formal_calibration_updates'] >= 1
    return 'audits={}, audit ATP={:.8g}, calibration={}'.format(
        summary['formal_audits'], summary['formal_audit_cost_atp'],
        summary['formal_calibration_updates'],
    )


def test_calibration_improves_heldout_mae():
    ledger = soma.CausalCalibrationLedger()
    mask = np.zeros(p2.P2_CELL_COUNT, dtype=bool); mask[:2] = True
    rng = np.random.default_rng(7)
    for raw in np.linspace(-0.009, 0.009, 80):
        observed = 0.001 + 0.45 * raw + rng.normal(0.0, 0.00015)
        ledger.update(1, raw, observed, 0.9, mask)
    raw_error = []; calibrated_error = []
    for raw in np.linspace(-0.0085, 0.0085, 31):
        observed = 0.001 + 0.45 * raw
        calibrated, _ = ledger.predict(1, raw)
        raw_error.append(abs(raw - observed))
        calibrated_error.append(abs(calibrated - observed))
    raw_mae = float(np.mean(raw_error)); cal_mae = float(np.mean(calibrated_error))
    assert cal_mae < 0.15 * raw_mae
    return 'heldout raw MAE={:.8g}, calibrated MAE={:.8g}'.format(raw_mae, cal_mae)


def _arm_change_sentinel(sentinel, config, steps=40, dt=0.1):
    state = np.asarray([0.35, 0.10, 0.08, 0.28, 0.30, 0.24, 0.07, 0.04], dtype=float)
    mechanism = np.asarray([0.82, 0.08, 0.90, 0.03, 0.94, 0.92, 0.02, 0.88], dtype=float)
    for step in range(int(steps)):
        sentinel.update(state, mechanism, dt, step * dt, config)
    if not sentinel.armed:
        raise AssertionError('change sentinel did not arm')
    return state, mechanism


def test_source_separated_change_sentinel_blocks_routine_state_noise():
    config = soma.Formal06Config(change_warmup=0.0)
    sentinel = soma.MaterialChangeSentinel(True)
    state, mechanism = _arm_change_sentinel(sentinel, config, steps=45)
    dt = 0.1
    max_state = 0.0
    max_actionable = 0.0
    # A large but purely state-level excursion (ATP/debt/proteostasis) may be
    # diagnostically interesting, yet it must not by itself open the strong
    # mechanism-change feedback gate.
    noisy_state = state + np.asarray([0.80, 0.40, 0.40, 0.70, 0.70, 0.70, 0.50, 0.50])
    for step in range(100):
        sentinel.update(noisy_state, mechanism, dt, 4.5 + step * dt, config)
        max_state = max(max_state, sentinel.state_probability)
        max_actionable = max(max_actionable, sentinel.global_probability)
    assert max_state > 0.50
    assert max_actionable < 0.12
    assert sentinel.events == 0

    changed_max = 0.0
    shifted = mechanism.copy()
    shifted[0] = -0.85       # longitudinal action-to-motion transfer reverses
    shifted[2] = 0.38        # tissue capacity falls
    shifted[3] = 0.56        # neural damage rises
    shifted[7] = 0.42        # controller capacity falls
    for step in range(120):
        sentinel.update(state, shifted, dt, 20.5 + step * dt, config)
        changed_max = max(changed_max, sentinel.global_probability)
    assert changed_max > soma.FORMAL_CHANGE_EVENT_THRESHOLD
    assert sentinel.mechanism_probability > 0.35
    assert sentinel.events >= 1
    return 'routine state max={:.4f}, actionable stable={:.4f}, mechanism change={:.4f}'.format(
        max_state, max_actionable, changed_max,
    )


def test_sparse_ligand_outcome_reversal_is_environment_evidence():
    config = soma.Formal06Config(
        change_warmup=0.0,
        change_semantic_min_episodes=5.0,
        change_semantic_min_effect=0.0035,
    )
    sentinel = soma.MaterialChangeSentinel(True)
    _arm_change_sentinel(sentinel, config, steps=45)
    age = 5.0
    # Establish a repeated beneficial material consequence for ligand 0.
    for value in (0.0110, 0.0120, 0.0105, 0.0115, 0.0122, 0.0111, 0.0118):
        sentinel.observe_semantic(0, np.asarray([value, 0.0, 0.0, 0.0]), 0.95, age, config)
        age += 1.0
    before = sentinel.semantic_channel_probability[0]
    # No-contact time is intentionally omitted: it is not zero evidence.
    for value in (-0.0120, -0.0130, -0.0115, -0.0135, -0.0128, -0.0132):
        sentinel.observe_semantic(0, np.asarray([value, 0.0, 0.0, 0.0]), 0.95, age, config)
        age += 1.0
    after = sentinel.semantic_channel_probability[0]
    assert before < 0.15
    assert after > 0.55
    assert sentinel.semantic_probability > 0.55
    assert sentinel.feedback_source(0) == 'semantics'
    return 'semantic channel {:.4f}->{:.4f}; events={}'.format(
        before, after, sentinel.semantic_events,
    )


def test_reliable_audit_surprise_becomes_targeted_change_evidence():
    config = soma.Formal06Config(change_warmup=0.0)
    sentinel = soma.MaterialChangeSentinel(True)
    _arm_change_sentinel(sentinel, config, steps=45)
    low = sentinel.observe_audit_surprise(
        2, 0.0040, -0.0045, quality=0.92, reliability=0.20,
        age=5.0, config=config,
    )
    low_probability = sentinel.feedback_probability(2)
    high = sentinel.observe_audit_surprise(
        2, 0.0040, -0.0045, quality=0.92, reliability=0.88,
        age=6.0, config=config,
    )
    high_probability = sentinel.feedback_probability(2)
    assert low < 1e-8 and low_probability < 0.05
    assert high > 0.55 and high_probability > 0.65
    assert sentinel.feedback_source(2) == 'mechanism'
    return 'low={:.4g}, high={:.4f}, targeted probability={:.4f}'.format(
        low, high, high_probability,
    )


def test_change_gated_feedback_protects_stable_tissue():
    world = soma.Formal06World(seed=350, initial_cells=1, config=high_budget_config())
    _, tissue = develop_formal(world)
    claim = np.zeros(p2.P2_CELL_COUNT, dtype=bool); claim[:2] = True
    result = {
        'status': 'contradicted', 'quality': 0.9, 'reliability': 0.8,
        'claim_mask': claim, 'channel': 0,
    }
    stable = soma.FormalMaterialTissue.from_state(tissue.state_dict())
    changed = soma.FormalMaterialTissue.from_state(tissue.state_dict())
    stable.maturity[:] = 0.9; changed.maturity[:] = 0.9
    stable.resource_lease[:] = 1.0; changed.resource_lease[:] = 1.0
    stable.change_sentinel.mechanism_channel_probability[:] = 0.0
    stable.change_sentinel.semantic_channel_probability[:] = 0.0
    stable.change_sentinel.channel_probability[:] = 0.0
    changed.change_sentinel.mechanism_probability = 0.94
    changed.change_sentinel.mechanism_channel_probability[0] = 0.94
    changed.change_sentinel.channel_probability[0] = 0.94
    gs = stable._apply_feedback(result, world.config)
    gc = changed._apply_feedback(result, world.config)
    stable_loss = float(np.mean(0.9 - stable.maturity[claim]))
    changed_loss = float(np.mean(0.9 - changed.maturity[claim]))
    stable_lease_loss = float(np.mean(1.0 - stable.resource_lease[claim]))
    changed_lease_loss = float(np.mean(1.0 - changed.resource_lease[claim]))
    assert gs == 0.0 and stable_loss == 0.0 and stable_lease_loss == 0.0
    assert gc > 0.10
    assert changed_loss > 0.02
    assert changed_lease_loss > 0.02
    return 'stable gate={:.6g}, changed gate={:.6g}, maturity loss ratio={:.4g}, lease loss={:.4f}'.format(
        gs, gc, changed_loss / max(stable_loss, 1e-12), changed_lease_loss,
    )


def test_small_compiler_and_random_cost_match():
    world = soma.Formal06World(seed=360, initial_cells=1, config=high_budget_config())
    _, tissue = develop_formal(world)
    tissue.causal_estimate[:] = 0.0
    tissue.causal_estimate[0, 2] = 0.010
    tissue.causal_estimate[1, 2] = 0.008
    tissue.activity_mean[:] = np.linspace(0.1, 0.8, p2.P2_CELL_COUNT)
    tissue.hidden[:] = np.linspace(-0.7, 0.7, p2.P2_CELL_COUNT)
    tissue.causal_right_gate[:] = 0.8
    tissue.calibrator.cell_trust[:] = 0.7
    debt = np.asarray([0.2, 0.2, 0.8, 0.2])
    compiled = tissue.compiler.compile(tissue, debt, world.config)
    random_program = tissue.compiler.random_matched_program(tissue, debt, world.config)
    assert compiled['complexity'] <= soma.FORMAL_MAX_COMMANDS
    assert random_program['complexity'] == compiled['complexity']
    assert random_program['commands'][0]['count'] == compiled['commands'][0]['count']
    assert random_program['commands'][0]['strength'] == compiled['commands'][0]['strength']
    return 'compiled channel={}, complexity={}, matched random complexity={}'.format(
        compiled['channel'], compiled['complexity'], random_program['complexity'],
    )


def test_audit_freezes_fast_learning_state():
    config = high_budget_config(
        audit_min_active_age=999.0,
        audit_freeze_learning=True,
    )
    world = soma.Formal06World(seed=370, initial_cells=1, config=config)
    _, tissue = develop_formal(world)
    debt = np.ones(soma.FORMAL_CHANNELS) * 0.2
    context = np.zeros(soma.FORMAL_CONTEXT_DIM)
    program = tissue.compiler.compile(tissue, debt, world.config)
    assert tissue.auditor.start(program, debt, context, world.age, tissue.formal_rng, world.config)
    # Advance through wash so the next world step contains an actual gated measure.
    tissue.auditor.stage = tissue.auditor.STAGE_MEASURE
    tissue.auditor.timer = 0.0
    names = ('w_sensor','w_rec','bias','motor_gain','predict_w','causal_estimate','maturity','reopen_reserve','elig_bias','elig_sensor','elig_rec','elig_motor','cue_eligibility','ligand_trace')
    before = {name: getattr(tissue, name).copy() for name in names}
    world.step(1.0 / soma.SIM_HZ)
    assert all(np.array_equal(before[name], getattr(tissue, name)) for name in names)
    return '14 learned arrays exact during active intervention'


def test_evidence_expires_with_time():
    config = soma.Formal06Config(audit_evidence_half_life=10.0)
    auditor = soma.MaterialCausalAuditor(True)
    auditor.evidence_weight[0, 1] = 3.0
    auditor.evidence_age[0, 1] = 0.0
    early = float(auditor.evidence_gate(0.0, config)[0, 1])
    late = float(auditor.evidence_gate(100.0, config)[0, 1])
    assert early > 0.90 and abs(late - 0.5) < 0.01
    return 'gate {:.6g}->{:.6g} after 10 half-lives'.format(early, late)


def test_controller_matter_routes_to_corpse():
    world = soma.Formal06World(seed=380, initial_cells=1, config=high_budget_config())
    _, _ = develop_formal(world)
    cell = first_cell(world)
    controller_mass = _controller_material(world, cell)
    before = world.total_material()
    cell.alive = False; cell.death_reason = 'formal-validation'
    world._handle_divisions_and_deaths()
    assert not world.cells and len(world.corpses) == 1
    assert world.p0_returned_material >= controller_mass - 2e-10
    assert abs(world.total_material() - before) < 8e-5
    return 'controller matter={:.8g}, returned={:.8g}, corpse={:.8g}'.format(
        controller_mass, world.p0_returned_material, world.corpses[0].material_mass(),
    )


def _integrate_formal_fragment(world, copies=4):
    cell = first_cell(world)
    sequence = np.concatenate([soma.make_formal_controller_gene() for _ in range(copies)]).astype(np.uint8)
    fragment = s5.DNAFragment(
        sequence, cell.pos, origin_lineage=-77, origin_cell=-77,
        origin_hash='controlled-formal-hgt', mobile=False,
    )
    world.edna.fragments.append(fragment)
    world.initial_total_material = world.total_material()
    fragment = world.edna.fragments.pop()
    if not cell.integrate_fragment(fragment, world, force=True):
        raise AssertionError('controlled formal HGT integration failed')
    return cell, sequence


def test_material_hgt_activates_controller_without_learning_copy():
    config = high_budget_config(
        formal_install_gene=False,
        formal_bootstrap_protein=False,
        neural_hgt_enabled=True,
        audit_min_active_age=999.0,
    )
    world = soma.Formal06World(seed=390, initial_cells=1, config=config)
    cell, sequence = _integrate_formal_fragment(world)
    tissue = cell.p2_tissue
    tissue.calibrator.weight[:] = 0.0
    tissue.auditor.evidence_weight[:] = 0.0
    for _ in range(500):
        world.step(1.0 / soma.SIM_HZ)
        if tissue.controller_mature:
            break
    assert tissue.controller_mature
    assert world.formal_gene_hgt_activations >= 1
    assert np.all(tissue.calibrator.weight == 0.0)
    assert np.all(tissue.auditor.evidence_weight == 0.0)
    assert len(sequence) * s5.MONOMER_MASS > 0.0
    return 'HGT controller mature in {} steps; no audit/calibration state copied'.format(_ + 1)


def test_no_hgt_mode_blocks_foreign_phenotype():
    config = high_budget_config(
        formal_mode=soma.FORMAL_MODE_NO_HGT,
        formal_install_gene=False,
        formal_bootstrap_protein=False,
        neural_hgt_enabled=False,
        audit_min_active_age=2.0,
    )
    world = soma.Formal06World(seed=391, initial_cells=1, config=config)
    cell, _ = _integrate_formal_fragment(world)
    for _ in range(240):
        world.step(1.0 / soma.SIM_HZ)
    tissue = cell.p2_tissue
    assert soma.formal_controller_activity(cell) > 0.016
    assert not tissue.formal_enabled
    assert not tissue.controller_mature
    assert world.formal_gene_hgt_activations == 0
    return 'foreign gene translated (activity {:.5g}) but phenotype stayed blocked'.format(
        soma.formal_controller_activity(cell),
    )


def test_division_recycles_learning_and_retains_formal_gene():
    world = soma.Formal06World(
        seed=400, initial_cells=1,
        config=high_budget_config(mutation=False, external_replicase=True),
    )
    _, tissue = develop_formal(world)
    tissue.w_sensor[0, 0] = 0.777
    tissue.calibrator.weight[0] = 4.0
    tissue.auditor.evidence_weight[0, 0] = 2.0
    cell = first_cell(world)
    cell.pools[s5.POOL_NUCLEOTIDE] = max(cell.pools[s5.POOL_NUCLEOTIDE], 9.0)
    cell.pools[s5.POOL_ATP] = max(cell.pools[s5.POOL_ATP], 9.0)
    steps = 0
    while len(cell.genomes) < 2 and steps < 16000:
        cell._replicate_genome(world, 0.1, world.config)
        steps += 1
    assert len(cell.genomes) >= 2
    cell.division_progress = 1.0
    cell.septum_mass = max(cell.septum_mass, 0.12)
    daughters = cell.split(world)
    assert daughters is not None and len(daughters) == 2
    assert all(d.p2_tissue is None for d in daughters)
    assert all(not d.neural_attachments for d in daughters)
    assert all(len(soma.formal_controller_specs(d)) >= 1 for d in daughters)
    assert all(d.formal_gene_installed for d in daughters)
    assert abs(world.division_residual) < 3e-8
    return 'copy steps={}, learned arrays recycled; material gene retained'.format(steps)


def test_save_restore_mid_audit_exact():
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        audit_min_active_age=5.0,
        audit_cooldown=2.0,
    )
    world = soma.Formal06World(seed=410, initial_cells=1, config=config)
    for _ in range(800):
        world.step(1.0 / soma.SIM_HZ)
        tissue = first_cell(world).p2_tissue
        if tissue.auditor.active and tissue.auditor.stage == tissue.auditor.STAGE_MEASURE:
            break
    else:
        raise AssertionError('no mid-audit state reached')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'formal.pkl')
        world.save(path)
        restored = soma.Formal06World.load(path)
    for _ in range(120):
        world.step(1.0 / soma.SIM_HZ)
        restored.step(1.0 / soma.SIM_HZ)
    assert exact_equal(world.state_dict(), restored.state_dict())
    return 'mid-audit save/restore exact for 120 steps'


def test_common_disturbance_identical_twins_exact():
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        audit_min_active_age=5.0,
        audit_cooldown=2.0,
    )
    a = soma.Formal06World(seed=420, initial_cells=1, config=config)
    for step in range(180):
        soma.apply_formal_common_disturbance_tape(a, 8821, step)
        a.step(1.0 / soma.SIM_HZ)
    b = a.clone()
    for step in range(180, 300):
        soma.apply_formal_common_disturbance_tape(a, 8821, step)
        soma.apply_formal_common_disturbance_tape(b, 8821, step)
        a.step(1.0 / soma.SIM_HZ); b.step(1.0 / soma.SIM_HZ)
    assert exact_equal(a.state_dict(), b.state_dict())
    return 'identical treatment twins remain bit-exact under common tape'


def test_finite_and_material_ledger():
    config = soma.Formal06Config(
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        audit_min_active_age=5.0,
        audit_cooldown=2.0,
    )
    residuals = []
    for seed in (431, 432, 433):
        world = soma.Formal06World(seed=seed, initial_cells=1, config=config)
        for _ in range(int(35 * soma.SIM_HZ)):
            if not world.living_cells():
                break
            world.step(1.0 / soma.SIM_HZ)
        assert world.finite()
        residuals.append(abs(world.matter_ledger_residual()))
    assert max(residuals) < 8e-5
    return '3 seeds finite, max ledger residual={:.3e}'.format(max(residuals))


TESTS = [
    ('build_and_frozen_p2', test_build_and_frozen_p2),
    ('no_formal_cassette_exact_p2_lockstep', test_no_formal_cassette_exact_p2_lockstep),
    ('unmetered_formal_controller_rejected', test_unmetered_formal_controller_rejected),
    ('material_gene_and_controller_development', test_material_gene_and_controller_development),
    ('controller_activity_cost_is_paid_and_conservative', test_controller_activity_cost_is_paid_and_conservative),
    ('counterbalanced_audit_and_hard_event_retry', test_counterbalanced_audit_and_hard_event_retry),
    ('natural_audit_completes_and_costs', test_natural_audit_completes_and_costs),
    ('calibration_improves_heldout_mae', test_calibration_improves_heldout_mae),
    ('source_separated_change_sentinel_blocks_routine_state_noise', test_source_separated_change_sentinel_blocks_routine_state_noise),
    ('sparse_ligand_outcome_reversal_is_environment_evidence', test_sparse_ligand_outcome_reversal_is_environment_evidence),
    ('reliable_audit_surprise_becomes_targeted_change_evidence', test_reliable_audit_surprise_becomes_targeted_change_evidence),
    ('change_gated_feedback_protects_stable_tissue', test_change_gated_feedback_protects_stable_tissue),
    ('small_compiler_and_random_cost_match', test_small_compiler_and_random_cost_match),
    ('audit_freezes_fast_learning_state', test_audit_freezes_fast_learning_state),
    ('evidence_expires_with_time', test_evidence_expires_with_time),
    ('controller_matter_routes_to_corpse', test_controller_matter_routes_to_corpse),
    ('material_hgt_activates_controller_without_learning_copy', test_material_hgt_activates_controller_without_learning_copy),
    ('no_hgt_mode_blocks_foreign_phenotype', test_no_hgt_mode_blocks_foreign_phenotype),
    ('division_recycles_learning_and_retains_formal_gene', test_division_recycles_learning_and_retains_formal_gene),
    ('save_restore_mid_audit_exact', test_save_restore_mid_audit_exact),
    ('common_disturbance_identical_twins_exact', test_common_disturbance_identical_twins_exact),
    ('finite_and_material_ledger', test_finite_and_material_ledger),
]


def _run_named_test(name):
    mapping = dict(TESTS)
    if name not in mapping:
        raise KeyError('unknown validation test: {}'.format(name))
    try:
        observed = mapping[name]()
        return {
            'test': name,
            'pass': 'PASS',
            'observed': _fmt(observed),
            'criterion': 'deterministic criterion satisfied',
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            'test': name,
            'pass': 'FAIL',
            'observed': '{}: {}'.format(type(exc).__name__, exc),
            'criterion': 'must pass',
        }


def _write_validation_outputs(rows):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['test','pass','observed','criterion'])
        writer.writeheader(); writer.writerows(rows)
    passed = sum(row['pass'] == 'PASS' for row in rows)
    lines = [
        'SOMA-CELL 0.6 VALIDATION RESULTS',
        'Build: {}'.format(soma.BUILD),
        'Schema: {}'.format(soma.FORMAL_SCHEMA_VERSION),
        'Execution: fresh interpreter per test' if rows else 'Execution: no tests',
        'Passed: {}/{}'.format(passed, len(rows)),
        '',
    ]
    for row in rows:
        lines.append('[{}] {}'.format(row['pass'], row['test']))
        lines.append('  observed: {}'.format(row['observed']))
        lines.append('  criterion: {}'.format(row['criterion']))
    with open(TXT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('Validation: {}/{} PASS'.format(passed, len(rows)), flush=True)
    if passed != len(rows):
        raise SystemExit(1)
    return rows


def run_validation(fresh=False):
    if not fresh:
        rows = []
        for name, _function in TESTS:
            row = _run_named_test(name)
            print('[{}] {}'.format(row['pass'], name), flush=True)
            rows.append(row)
        return _write_validation_outputs(rows)

    # A fresh interpreter per case prevents large cyclic chemical worlds from
    # accumulating across a long validation run.  This is the official CPython
    # release path.  Pythonista can still call individual tests or the direct
    # in-process runner above.
    rows = []
    script = os.path.abspath(__file__)
    for index, (name, _function) in enumerate(TESTS, 1):
        completed = subprocess.run(
            [sys.executable, script, '--one', name],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=420,
        )
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith('SOMA_VALIDATION_JSON='):
                payload = json.loads(line.split('=', 1)[1])
                break
        if payload is None:
            payload = {
                'test': name,
                'pass': 'FAIL',
                'observed': 'child exit {} without JSON; stderr={}'.format(
                    completed.returncode, completed.stderr[-1200:]),
                'criterion': 'must pass',
            }
        rows.append(payload)
        print('{}/{} [{}] {}'.format(index, len(TESTS), payload['pass'], name), flush=True)
    return _write_validation_outputs(rows)


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == '--one':
        row = _run_named_test(argv[1])
        print('SOMA_VALIDATION_JSON=' + json.dumps(row, ensure_ascii=False), flush=True)
        return 0 if row['pass'] == 'PASS' else 1
    if argv and argv[0] == '--fresh':
        run_validation(fresh=True)
        return 0
    run_validation(fresh=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
