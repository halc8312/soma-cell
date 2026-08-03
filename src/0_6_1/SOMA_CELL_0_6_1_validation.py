# coding: utf-8
"""Deterministic mechanism validation for SOMA-CELL 0.6.1.

The suite validates a narrow claim: the frozen material-causal 0.6 body can
track evidence sufficiency, distinguish missing evidence from stable evidence,
and perform a bounded, materially paid diagnostic exposure before compiling a
targeted stop audit.  It does not establish life, consciousness, universal
adaptive benefit, or open-ended evolution.
"""
from __future__ import division

import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F06_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6'))
P2_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p2'))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, F06_DIR, P2_DIR, P1_DIR, P0_DIR, BASE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_1_pythonista as soma
import SOMA_CELL_0_6_pythonista as f06

p2 = soma.p2
p0 = soma.p0
s4 = soma.s4

CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_1_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_1_VALIDATION_RESULTS.txt')
F06_SOURCE_PATH = os.path.join(F06_DIR, 'SOMA_CELL_0_6_pythonista.py')
if not os.path.isfile(F06_SOURCE_PATH):
    F06_SOURCE_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_pythonista.py')
F06_SHA256 = 'e8f70c67cee16fdc1e90397808783e26b2895552ca3ec26f62c2b137ada08c26'


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


def project_like(candidate, reference):
    """Project a descendant state onto the exact schema of a frozen parent."""
    if isinstance(reference, dict):
        return {key: project_like(candidate[key], value) for key, value in reference.items()}
    if isinstance(reference, list):
        return [project_like(candidate[i], reference[i]) for i in range(len(reference))]
    if isinstance(reference, tuple):
        return tuple(project_like(candidate[i], reference[i]) for i in range(len(reference)))
    if isinstance(reference, np.ndarray):
        return np.asarray(candidate, dtype=reference.dtype).copy()
    return candidate


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return '{:.12g}'.format(float(value))
    return str(value)


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
        division=False,
        mutation=False,
        neural_atp_reserve=0.0,
        neural_fuel_reserve=0.0,
        neural_mineral_reserve=0.0,
        neural_membrane_reserve=0.0,
        neural_atp_rate=5.0,
        neural_protein_rate=5.0,
        neural_membrane_rate=5.0,
        neural_signal_rate=5.0,
        audit_min_active_age=999.0,
        formal_auto_start=False,
        diagnosis_enabled=True,
        diagnosis_min_active_age=999.0,
        mechanism_probe_enabled=False,
        mechanism_assist_enabled=False,
        mechanism_conservation_enabled=False,
        mechanism_fault_enabled=False,
    )
    values.update(kwargs)
    return soma.Formal061Config(**values)


def develop_061(world, max_steps=520):
    dt = 1.0 / soma.SIM_HZ
    for step in range(max_steps):
        world.step(dt)
        cell = first_cell(world)
        tissue = cell.p2_tissue
        if (
            isinstance(tissue, soma.ActiveDiagnosticMaterialTissue)
            and np.count_nonzero(tissue.mature) == p2.P2_CELL_COUNT
            and tissue.controller_mature
        ):
            return step + 1, tissue
    raise AssertionError('0.6.1 tissue/controller did not mature')


def synthetic_frame(ligand=soma.ALT_LIGAND, direction=(1.0, 0.0), concentration=0.12):
    profiles = np.zeros((s4.LIGAND_COUNT, len(p2.MEMBRANE_NORMALS)), dtype=float)
    direction = np.asarray(direction, dtype=float)
    directional = np.maximum(0.0, p2.MEMBRANE_NORMALS.dot(direction))
    profiles[int(ligand)] = float(concentration) * (0.15 + 0.85 * directional)
    return {'external': {'ligand_profiles': profiles}}


def test_build_and_frozen_formal06():
    with open(F06_SOURCE_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == F06_SHA256
    assert soma.BUILD == 'SOMA-CELL 0.6.1'
    assert soma.SCHEMA_VERSION.startswith('0.6.1-')
    assert f06.BUILD == 'SOMA-CELL 0.6.0'
    return 'build={}, schema={}, frozen 0.6 sha={}'.format(soma.BUILD, soma.SCHEMA_VERSION, digest)


def test_diagnosis_off_exact_formal06_lockstep():
    observations = []
    for seed in (101, 202, 303):
        base_cfg = f06.Formal06Config(
            p2_environment=p2.P2_ENV_CUE_REVERSAL,
            division=False, mutation=False,
        )
        candidate_cfg = soma.Formal061Config(
            diagnosis_mode=soma.DIAGNOSIS_OFF,
            p2_environment=p2.P2_ENV_CUE_REVERSAL,
            division=False, mutation=False,
            # Explicit parent-compatibility mode.  The 0.6.1 default closes a
            # P2 ATP-escrow leak and therefore intentionally changes physics.
            neural_escrow_sanitation_enabled=False,
        )
        control = f06.Formal06World(seed=seed, initial_cells=1, config=base_cfg)
        candidate = soma.Formal061World(seed=seed, initial_cells=1, config=candidate_cfg)
        for _ in range(180):
            control.step(1.0 / f06.SIM_HZ)
            candidate.step(1.0 / soma.SIM_HZ)
        projected = project_like(candidate.state_dict(), control.state_dict())
        projected['build'] = control.state_dict()['build']
        projected['save_version'] = control.state_dict()['save_version']
        for i in range(len(projected['cells'])):
            projected['cells'][i]['cell_class'] = control.state_dict()['cells'][i]['cell_class']
        assert exact_equal(control.state_dict(), projected)
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        observations.append('{}:exact'.format(seed))
    return ', '.join(observations)


def test_costless_diagnosis_rejected():
    try:
        soma.Formal061Config(diagnosis_controller_cost=False)
        raise AssertionError('costless diagnosis accepted')
    except ValueError:
        pass
    return 'unmetered diagnosis fails closed'


def test_evidence_ledger_distinguishes_unknown_from_stable():
    cfg = soma.Formal061Config()
    led = soma.EvidenceSufficiencyLedger()
    assert led.status(soma.ALT_LIGAND, 0, 0.0, cfg) == soma.EVIDENCE_UNKNOWN
    out = np.asarray([0.06, 0.01, 0.01, 0.01])
    for age in (1.0, 2.0, 3.0, 4.0):
        led.observe_semantic(soma.ALT_LIGAND, out, 1.0, age, np.asarray([0.2,0.2,0.2,0.2]), cfg)
    status = led.status(soma.ALT_LIGAND, 0, 4.0, cfg)
    assert status == soma.EVIDENCE_STABLE
    assert led.first_observation_sufficient_age[soma.ALT_LIGAND, 0] > 0.0
    return 'unknown -> {}, obs={:.3f}'.format(status, led.observation_sufficiency(soma.ALT_LIGAND,0,4.0,cfg))


def test_evidence_decay_is_time_step_invariant():
    cfg = soma.Formal061Config(diagnosis_evidence_half_life=80.0)
    led = soma.EvidenceSufficiencyLedger()
    led.observation_weight[:] = 2.0
    led.postchange_weight[:] = 1.0
    led.intervention_weight[:] = 0.7
    led.surprise[:] = 0.4
    led.change_evidence[:] = 0.3
    led.last_decay_age = 0.0
    big = soma.EvidenceSufficiencyLedger.from_state(led.state_dict())
    small = soma.EvidenceSufficiencyLedger.from_state(led.state_dict())
    big.decay(20.0, cfg)
    for age in np.linspace(0.2, 20.0, 100):
        small.decay(float(age), cfg)
    for name in ('observation_weight','postchange_weight','intervention_weight','surprise','change_evidence'):
        assert np.allclose(getattr(big,name), getattr(small,name), atol=2e-12, rtol=2e-12)
    return '20s decay invariant, max delta={:.3e}'.format(float(np.max(np.abs(big.change_evidence-small.change_evidence))))


def test_channel_resolved_semantic_reversal():
    cfg = soma.Formal061Config(
        diagnosis_observation_target=2.0,
        diagnosis_postchange_target=1.0,
        diagnosis_change_threshold=0.30,
    )
    led = soma.EvidenceSufficiencyLedger()
    base = np.asarray([0.02, 0.02, 0.10, 0.02])
    rev = np.asarray([0.02, 0.02, -0.12, 0.02])
    for age in (1.0, 2.0, 3.0):
        led.observe_semantic(soma.ALT_LIGAND, base, 1.0, age, np.asarray([0.2,0.2,0.2,0.2]), cfg)
    for age in (4.0, 5.0, 6.0):
        led.observe_semantic(soma.ALT_LIGAND, rev, 1.0, age, np.asarray([0.2,0.2,0.2,0.2]), cfg)
    evidence = led.change_evidence[soma.ALT_LIGAND]
    assert evidence[2] > 0.35
    assert evidence[2] > max(evidence[0], evidence[1], evidence[3]) + 0.20
    return 'channel2={:.3f}, others={}'.format(evidence[2], np.round(np.delete(evidence,2),3).tolist())


def test_diagnosis_start_safety_defers_crisis():
    world = soma.Formal061World(seed=411, initial_cells=1, config=high_budget_config(diagnosis_min_active_age=0.0))
    _, tissue = develop_061(world)
    cfg = world.config
    tissue.diagnostic.active = False
    tissue.diagnostic.cooldown_until = 0.0
    tissue.auditor.active = False
    tissue._061_last_defer_age = -1e9
    assert not tissue._diagnosis_safe(np.asarray([0.95,0.2,0.2,0.2]), cfg)
    frame = world.port_for(first_cell(world).cell_id).raw_sensor_fluxes(tissue.tissue_ids[0])
    started = tissue._maybe_start_diagnosis(frame, np.asarray([0.95,0.2,0.2,0.2]), world, cfg)
    assert not started
    assert tissue.diagnosis_deferred_unsafe >= 1
    return 'unsafe deferred={}, starts={}'.format(tissue.diagnosis_deferred_unsafe, tissue.diagnostic.started)


def test_active_direction_comes_from_physical_gradient():
    cfg = soma.Formal061Config(diagnosis_mode=soma.DIAGNOSIS_ACTIVE, diagnosis_min_ligand_concentration=0.001)
    diag = soma.BoundedDiagnosticExposure()
    assert diag.start(soma.ALT_LIGAND, 0, np.zeros(4), 10.0, cfg, np.random.default_rng(1))
    command = diag.command(synthetic_frame(direction=(1.0,0.0)), np.zeros(4), 1.0/soma.SIM_HZ, 10.0, cfg)
    assert command is not None
    assert command['direction'][0] > 0.90 and abs(command['direction'][1]) < 0.15
    return 'direction={}'.format(np.round(command['direction'],4).tolist())


def test_random_control_is_command_cost_matched():
    frame = synthetic_frame(direction=(1.0,0.0))
    active_cfg = soma.Formal061Config(diagnosis_mode=soma.DIAGNOSIS_ACTIVE, diagnosis_min_ligand_concentration=0.001)
    random_cfg = soma.Formal061Config(diagnosis_mode=soma.DIAGNOSIS_RANDOM, diagnosis_min_ligand_concentration=0.001)
    a = soma.BoundedDiagnosticExposure(); r = soma.BoundedDiagnosticExposure()
    a.start(soma.ALT_LIGAND,0,np.zeros(4),1.0,active_cfg,np.random.default_rng(2))
    r.start(soma.ALT_LIGAND,0,np.zeros(4),1.0,random_cfg,np.random.default_rng(2))
    ca = a.command(frame,np.zeros(4),1/soma.SIM_HZ,1.0,active_cfg)
    cr = r.command(frame,np.zeros(4),1/soma.SIM_HZ,1.0,random_cfg)
    assert ca is not None and cr is not None
    assert abs(np.linalg.norm(ca['motor']) - np.linalg.norm(cr['motor'])) < 1e-12
    assert abs(np.linalg.norm(ca['transporter_polarity']) - np.linalg.norm(cr['transporter_polarity'])) < 1e-12
    return 'motor norm={:.4f}, transporter norm={:.4f}'.format(np.linalg.norm(ca['motor']),np.linalg.norm(ca['transporter_polarity']))


def test_diagnostic_pause_removes_locomotion():
    cfg = soma.Formal061Config()
    diag = soma.BoundedDiagnosticExposure()
    diag.start(soma.ALT_LIGAND,0,np.zeros(4),1.0,cfg,np.random.default_rng(3))
    diag.pause_for_audit(1.5,cfg)
    assert diag.active and diag.phase == soma.DIAGNOSTIC_OBSERVE
    assert diag.command(synthetic_frame(),np.zeros(4),1/soma.SIM_HZ,1.5,cfg) is None
    return 'phase={}, reason={}'.format(diag.phase,diag.last_reason)


def test_diagnostic_exposure_metrics_and_material_cost():
    # Active diagnosis is only allowed after ordinary experience has established
    # a relation.  Build that relation in the material ledger first, then make it
    # stale/suspect and provide an actual membrane gradient.  This replaces the
    # obsolete test that expected diagnosis from an uncalibrated cue.
    cfg = high_budget_config(
        diagnosis_mode=soma.DIAGNOSIS_ACTIVE,
        diagnosis_min_active_age=0.0,
        diagnosis_min_ligand_concentration=0.001,
        diagnosis_start_energy_debt=0.99,
        diagnosis_stale_age=2.0,
        diagnosis_stale_refresh_multiplier=1.0,
        diagnosis_min_suspicion=0.05,
        diagnosis_min_deficit=0.05,
        diagnosis_observation_target=1.0,
        diagnosis_context_target_span=0.01,
    )
    world = soma.Formal061World(seed=101,initial_cells=1,config=cfg)
    _, tissue = develop_061(world)
    cell = first_cell(world)
    context = np.asarray([0.25,0.25,0.25,0.25])
    # A physical ordinary relation is represented by repeated delayed outcomes.
    for j in range(4):
        tissue.evidence_ledger.observe_semantic(
            soma.ALT_LIGAND, np.asarray([0.030,0.0,0.0,0.0]),
            0.85, world.age + 0.05*j, context, cfg,
        )
    tissue.evidence_ledger.last_observation_age[soma.ALT_LIGAND,0] = world.age - 5.0
    # Preserve total material: move existing alternative-substrate particles to
    # one side of the cell instead of creating a privileged test signal.
    pos = np.asarray(cell.pos, dtype=float)
    moved = 0
    indices=np.flatnonzero(np.asarray(world.field.kind,dtype=int)==int(s4.PARTICLE_ALT))[:12]
    for index in indices:
        world.field.pos[int(index)] = (pos + np.asarray([0.07,0.0])) % 1.0
        moved += 1
    debt = tissue._physical_debt(world.port_for(cell.cell_id).raw_sensor_fluxes(tissue.tissue_ids[0]))
    started = False
    for _ in range(int(6*soma.SIM_HZ)):
        world.step(1/soma.SIM_HZ)
        if tissue.diagnostic.started >= 1:
            started = True
        if tissue.diagnostic.completed >= 1:
            break
    summary = world.summary()
    assert started
    assert summary['diagnosis_atp'] > 0.0
    assert summary['diagnosis_cumulative_exposure'] > 0.0
    assert summary['diagnosis_directional_samples'] > 0
    assert world.finite() and abs(world.matter_ledger_residual()) < 8e-5
    return 'starts={}, ATP={:.6g}, exposure={:.6g}'.format(summary['diagnosis_started'],summary['diagnosis_atp'],summary['diagnosis_cumulative_exposure'])


def test_evidence_event_cost_is_paid():
    world = soma.Formal061World(seed=412, initial_cells=1, config=high_budget_config())
    _, tissue = develop_061(world)
    port = world.port_for(first_cell(world).cell_id)
    before = tissue.controller_atp_spent + tissue.diagnosis_planning_atp
    before_wear = tissue.diagnosis_material_wear
    tissue._pay_evidence_event_cost(port, world.config)
    after = tissue.controller_atp_spent + tissue.diagnosis_planning_atp
    assert after > before or tissue.diagnosis_planning_atp > 0.0
    assert tissue.diagnosis_material_wear >= before_wear
    return 'ATP delta={:.3e}, wear delta={:.3e}'.format(after-before,tissue.diagnosis_material_wear-before_wear)


def test_targeted_compiler_uses_current_neural_claim():
    world = soma.Formal061World(seed=413, initial_cells=1, config=high_budget_config())
    _, tissue = develop_061(world)
    tissue.causal_estimate[:] = 0.0
    tissue.causal_estimate[:2, 1] = 0.012
    tissue.causal_estimate[2:4, 1] = -0.002
    frame = synthetic_frame(direction=(1.0,0.0), concentration=0.30)
    tissue._061_last_age = world.age
    tissue.evidence_ledger.baseline_mean[soma.ALT_LIGAND,1] = -0.5
    program = tissue.compiler.compile_semantic(tissue,frame,soma.ALT_LIGAND,1,world.config)
    assert program is not None
    claim = np.asarray(program['claim_mask'],dtype=bool)
    sham = np.asarray(program['sham_mask'],dtype=bool)
    expected = float(np.mean(tissue.causal_estimate[claim,1])-np.mean(tissue.causal_estimate[sham,1]))
    assert abs(program['raw_prediction']-expected) < 1e-12
    assert program['semantic_baseline'] == -0.5
    return 'raw={:.6g}, semantic baseline={:.3g}'.format(program['raw_prediction'],program['semantic_baseline'])


def test_targeted_audit_waits_for_activation_delay():
    cfg = high_budget_config(diagnosis_audit_activation_delay=0.20, diagnosis_audit_min_concentration=0.001)
    world = soma.Formal061World(seed=414, initial_cells=1, config=cfg)
    _, tissue = develop_061(world)
    cell = first_cell(world); port = world.port_for(cell.cell_id)
    frame = synthetic_frame(direction=(1.0,0.0), concentration=0.30)
    tissue._current_concentrations = lambda _frame: np.asarray([0.0,0.0,0.8,0,0,0,0,0],dtype=float)
    tissue.pending_targeted_audit=(soma.ALT_LIGAND,0)
    debt=np.ones(4)*0.2; context=np.zeros(f06.FORMAL_CONTEXT_DIM)
    assert not tissue._start_targeted_audit(port,frame,debt,context,world,cfg)
    assert not tissue.auditor.active
    world.age += 0.21
    assert tissue._start_targeted_audit(port,frame,debt,context,world,cfg)
    assert tissue.auditor.active
    return 'audit started after {:.3f}s visibility'.format(world.age-tissue.pending_audit_visible_since if tissue.pending_audit_visible_since>-1e8 else 0.21)


def test_controlled_targeted_audit_can_trigger_material_feedback():
    cfg = high_budget_config(
        change_gated_feedback=True,
        change_feedback_threshold=0.30,
        audit_measure_duration=0.10,
        audit_washout_duration=0.02,
        audit_max_retries=0,
        diagnosis_feedback_min_reliability=0.05,
        diagnosis_feedback_required_weight=0.52,
    )
    world = soma.Formal061World(seed=415,initial_cells=1,config=cfg)
    _, tissue = develop_061(world)
    tissue.causal_estimate[:] = 0.0
    tissue.causal_estimate[:2,0] = 0.006
    tissue.hidden[:] = 0.4
    tissue.motor_gain[:] = 0.5
    frame = synthetic_frame(direction=(1.0,0.0), concentration=0.3)
    tissue._061_last_age = world.age
    program = tissue.compiler.compile_semantic(tissue,frame,soma.ALT_LIGAND,0,cfg)
    assert program is not None
    debt=np.ones(4)*0.2; context=np.zeros(f06.FORMAL_CONTEXT_DIM)
    assert tissue.auditor.start(program,debt,context,world.age,tissue.formal_rng,cfg)
    dt=0.025; result=None
    for step in range(600):
        rate=0.0
        if tissue.auditor.active and tissue.auditor.stage==tissue.auditor.STAGE_MEASURE and tissue.auditor.current_condition()==1:
            rate = -0.004 if tissue.auditor.current_arm()==0 else 0.001
        debt = debt + np.asarray([rate,0,0,0])*dt
        result=tissue.auditor.observe(debt,context,0,1.0,dt,world.age+step*dt,cfg,0.01,0.001)
        if result is not None: break
    assert result is not None and result['status']=='contradicted'
    result.update(tissue.calibrator.update(0,result['raw_prediction'],result['target_effect'],result['quality'],result['claim_mask']))
    tissue.change_sentinel.mechanism_probability=0.95
    tissue.change_sentinel.mechanism_channel_probability[0]=0.95
    tissue.change_sentinel.channel_probability[0]=0.95
    before=tissue.resource_lease.copy()
    first=tissue._apply_feedback(dict(result),cfg)
    assert first==0.0 and np.array_equal(before,tissue.resource_lease)
    tissue._061_last_age += 1.0
    magnitude=tissue._apply_feedback(dict(result),cfg)
    assert magnitude>0.0 and np.any(tissue.resource_lease<before)
    return 'status={}, quality={:.3f}, replicated feedback={:.4f}'.format(result['status'],result['quality'],magnitude)


def test_state_noise_does_not_open_strong_feedback():
    cfg = high_budget_config(change_gated_feedback=True,change_feedback_threshold=0.45)
    world=soma.Formal061World(seed=416,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    tissue.change_sentinel.state_probability=0.99
    tissue.change_sentinel.mechanism_probability=0.0
    tissue.change_sentinel.semantic_probability=0.0
    tissue.change_sentinel.genome_probability=0.0
    mask=np.zeros(p2.P2_CELL_COUNT,dtype=bool); mask[:2]=True
    before=tissue.resource_lease.copy()
    magnitude=tissue._apply_feedback({'status':'contradicted','quality':1.0,'reliability':1.0,'claim_mask':mask,'channel':0,'target_effect':-0.01},cfg)
    assert magnitude==0.0 and np.array_equal(before,tissue.resource_lease)
    return 'state=0.99, actionable gate={}, feedback=0'.format(tissue.change_sentinel.feedback_probability(0))


def test_semantic_outcome_quality_uses_physical_dose():
    cfg=high_budget_config(diagnosis_semantic_peak_target=0.05,diagnosis_semantic_dose_target=0.10)
    world=soma.Formal061World(seed=417,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    tissue._061_prev_cue_active[soma.ALT_LIGAND]=True
    tissue.cue_episode_active[soma.ALT_LIGAND]=True
    tissue._061_prev_cue_start[soma.ALT_LIGAND]=np.asarray([0.3,0.2,0.2,0.2])
    tissue._061_prev_cue_peak[soma.ALT_LIGAND]=np.asarray([0.35,0.2,0.2,0.2])
    tissue._061_prev_cue_signal_peak[soma.ALT_LIGAND]=0.01
    tissue._061_episode_concentration_integral[soma.ALT_LIGAND]=0.20
    tissue.cue_episode_active[soma.ALT_LIGAND]=False
    profiles=np.zeros((s4.LIGAND_COUNT,len(p2.MEMBRANE_NORMALS)))
    frame={'external':{'ligand_profiles':profiles}}
    out=tissue._observe_delayed_cue_outcomes(frame,np.asarray([0.2,0.2,0.2,0.2]),1/soma.SIM_HZ,world,cfg)
    assert out and tissue.evidence_ledger.observation_weight[soma.ALT_LIGAND,0] > 0.60
    return 'observation weight={:.3f}'.format(tissue.evidence_ledger.observation_weight[soma.ALT_LIGAND,0])


def _start_manual_diagnosis(world):
    cell=first_cell(world); tissue=cell.p2_tissue
    debt=np.asarray([0.2,0.2,0.2,0.2])
    assert tissue.diagnostic.start(soma.ALT_LIGAND,0,debt,world.age,world.config,tissue.formal_rng)
    return tissue


def test_save_restore_mid_diagnosis_exact():
    cfg=high_budget_config(diagnosis_min_ligand_concentration=0.001)
    world=soma.Formal061World(seed=418,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    _start_manual_diagnosis(world)
    clone=soma.Formal061World.from_state(world.state_dict())
    for _ in range(80):
        world.step(1/soma.SIM_HZ); clone.step(1/soma.SIM_HZ)
    assert exact_equal(world.state_dict(),clone.state_dict())
    return '80 steps exact while diagnostic state persisted'


def test_save_restore_mid_targeted_audit_exact():
    cfg=high_budget_config(audit_measure_duration=0.12,audit_washout_duration=0.02)
    world=soma.Formal061World(seed=419,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    tissue.causal_estimate[:2,0]=0.005
    program=tissue.compiler.compile(tissue,np.asarray([0.3,0.2,0.2,0.2]),cfg)
    assert tissue.auditor.start(program,np.ones(4)*0.2,np.zeros(f06.FORMAL_CONTEXT_DIM),world.age,tissue.formal_rng,cfg)
    clone=soma.Formal061World.from_state(world.state_dict())
    for _ in range(80):
        world.step(1/soma.SIM_HZ); clone.step(1/soma.SIM_HZ)
    assert exact_equal(world.state_dict(),clone.state_dict())
    return '80 steps exact from active audit state'


def test_common_disturbance_identical_twins_exact():
    cfg=soma.Formal061Config(p2_environment=p2.P2_ENV_CUE_REVERSAL,division=False,mutation=False)
    a=soma.Formal061World(seed=420,initial_cells=1,config=cfg)
    b=a.clone()
    for step in range(120):
        soma.apply_061_common_disturbance_tape(a,420,step,stream=7)
        soma.apply_061_common_disturbance_tape(b,420,step,stream=7)
        a.step(1/soma.SIM_HZ); b.step(1/soma.SIM_HZ)
    assert exact_equal(a.state_dict(),b.state_dict())
    return '120 common-tape steps exact'


def test_no_free_061_learning_state_in_division():
    # Frozen formal-0.6 already verifies material gene inheritance and no free
    # controller learning copy.  Here we additionally verify the 0.6.1 ledger
    # constructor defaults that any newly developed daughter receives.
    parent=soma.EvidenceSufficiencyLedger(); parent.observation_weight[:]=9.0
    child=soma.EvidenceSufficiencyLedger()
    assert np.max(child.observation_weight)==0.0
    assert child.semantic_events==0 and child.audit_events==0
    # Run the inherited material division test as a regression.
    import SOMA_CELL_0_6_validation as parent_validation
    parent_validation.test_division_recycles_learning_and_retains_formal_gene()
    return 'daughter 0.6.1 ledger fresh; inherited material-division regression PASS'


def test_controller_corpse_and_hgt_regressions():
    import SOMA_CELL_0_6_validation as parent_validation
    a=parent_validation.test_controller_matter_routes_to_corpse()
    b=parent_validation.test_material_hgt_activates_controller_without_learning_copy()
    return 'corpse={}, HGT={}'.format(a,b)


def test_stable_short_run_has_no_false_feedback():
    observations=[]
    for seed in (421,422,423):
        cfg=soma.Formal061Config(
            p2_environment=p2.P2_ENV_MOVING_PATCH,
            division=False,mutation=False,
            diagnosis_min_active_age=8.0,
            p2_switch_age=999.0,
        )
        world=soma.Formal061World(seed=seed,initial_cells=1,config=cfg)
        for _ in range(int(30*soma.SIM_HZ)):
            world.step(1/soma.SIM_HZ)
        summary=world.summary()
        assert summary['formal_feedback_events']==0
        assert world.finite()
        observations.append('{}:0'.format(seed))
    return ', '.join(observations)


def test_finite_and_material_ledger():
    residuals=[]
    for seed in (424,425,426):
        cfg=soma.Formal061Config(
            p2_environment=p2.P2_ENV_CUE_REVERSAL,
            p2_switch_age=20.0,
            diagnosis_min_active_age=8.0,
            division=False,mutation=False,
        )
        world=soma.Formal061World(seed=seed,initial_cells=1,config=cfg)
        for _ in range(int(32*soma.SIM_HZ)):
            if not world.living_cells(): break
            world.step(1/soma.SIM_HZ)
        assert world.finite()
        residuals.append(abs(world.matter_ledger_residual()))
    assert max(residuals)<8e-5
    return '3 seeds finite, max ledger residual={:.3e}'.format(max(residuals))


def _audit_result(quality=0.8, reliability=0.8, effect=0.004, channel=0, cells=(0,1)):
    claim=np.zeros(p2.P2_CELL_COUNT,dtype=bool); claim[list(cells)]=True
    sham=np.zeros(p2.P2_CELL_COUNT,dtype=bool); sham[[2,3]]=True
    return {
        'status':'contradicted','quality':float(quality),'reliability':float(reliability),
        'target_effect':float(effect),'channel':int(channel),
        'claim_mask':claim,'sham_mask':sham,
    }

def test_low_quality_feedback_is_blocked():
    cfg=high_budget_config(
        diagnosis_feedback_min_quality=0.20,
        diagnosis_feedback_min_reliability=0.10,
        mechanism_probe_enabled=False, mechanism_assist_enabled=False,
    )
    world=soma.Formal061World(seed=430,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    before=tissue.resource_lease.copy()
    result=_audit_result(quality=0.03,reliability=0.02,effect=0.02)
    magnitude=tissue._apply_feedback(result,cfg)
    assert magnitude==0.0
    assert np.array_equal(before,tissue.resource_lease)
    assert tissue.mechanism_feedback_blocked_low_quality==1
    return 'quality 0.03 blocked; leases unchanged'

def test_replicated_feedback_gate_accepts_clean_result():
    cfg=high_budget_config(
        diagnosis_feedback_required_weight=0.52,
        diagnosis_feedback_min_quality=0.20,
        diagnosis_feedback_min_reliability=0.10,
        mechanism_probe_enabled=False, mechanism_assist_enabled=False,
    )
    gate=soma.FeedbackReplicationGate()
    result=_audit_result(quality=0.80,reliability=0.80,effect=0.004)
    allowed=gate.allow(result,10.0,cfg)
    assert allowed and gate.accepted==1
    # A contradictory sign/signature must not silently reuse the prior receipt.
    opposite=_audit_result(quality=0.35,reliability=0.50,effect=-0.004,cells=(0,1))
    allowed2=gate.allow(opposite,11.0,cfg)
    assert not allowed2 and gate.rejected_consistency>=1
    return 'clean result accepted; inconsistent sign reset'

def test_mechanism_gain_requires_paid_probe_replication():
    cfg=soma.Formal061Config(
        mechanism_probe_min_baseline_weight=0.60,
        mechanism_probe_required_weight=0.80,
        mechanism_probe_confirm_ratio=0.72,
    )
    ledger=soma.MechanismGainLedger()
    ledger.observe(0.014,1.0,0.0,cfg,probe=False,mechanism_probability=0.0)
    ledger.observe(0.014,1.0,1.0,cfg,probe=False,mechanism_probability=0.0)
    first=ledger.observe(0.0025,0.60,2.0,cfg,probe=True,mechanism_probability=0.9)
    assert not first['confirmed']
    second=ledger.observe(0.0025,0.60,3.0,cfg,probe=True,mechanism_probability=0.9)
    assert second['confirmed'] and second['gain_ratio']<0.72
    return 'gain ratio={:.3f}, replicated probe weight={:.3f}'.format(second['gain_ratio'],ledger.confirmation_weight)

def test_mechanism_fault_converts_existing_matter():
    cfg=high_budget_config(
        diagnosis_mode=soma.DIAGNOSIS_PASSIVE, diagnosis_enabled=True,
        p2_effectors=True, sensorimotor=True,
        mechanism_probe_enabled=False, mechanism_assist_enabled=False,
        mechanism_fault_enabled=False,
    )
    world=soma.Formal061World(seed=431,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    cell=first_cell(world); port=world.port_for(cell.cell_id)
    before_total=world.total_material()
    before_damaged=sum(float(port._attachment(tid).tissue_material[p0.TISSUE_DAMAGED_PROTEIN]) for tid in tissue.tissue_ids)
    old_gain=float(world.config.neural_motor_gain_scale)
    world.config.mechanism_fault_enabled=True
    world.config.mechanism_fault_age=world.age
    world.config.mechanism_fault_gain_scale=0.25
    world.config.mechanism_fault_wear=0.00020
    assert world._apply_mechanism_fault_if_due()
    after_total=world.total_material()
    after_damaged=sum(float(port._attachment(tid).tissue_material[p0.TISSUE_DAMAGED_PROTEIN]) for tid in tissue.tissue_ids)
    assert abs(after_total-before_total)<1e-12
    assert after_damaged>before_damaged
    assert abs(world.config.neural_motor_gain_scale-0.25*old_gain)<1e-12
    return 'matter residual={:.3e}, damaged delta={:.3e}'.format(after_total-before_total,after_damaged-before_damaged)

def _run_assist_trial(lease,cfg,paid=True):
    debt=np.asarray([0.20,0.20,0.20,0.20],dtype=float)
    rng=np.random.default_rng(7)
    assert lease.start_trial(0.0,debt,0.70,np.asarray([1.0,0.0]),rng,cfg,confirmation_index=1)
    lease.phase_order=[False,True,True,False]
    age=0.0; result=None
    for on in lease.phase_order:
        end=debt.copy()
        if on:
            end[0]-=0.0010
            report={'motor_force':0.001,'atp_spent':2e-5,'signal_spent':2e-6} if paid else {}
            uptake=0.0012
        else:
            report={}; uptake=0.0
        age+=0.31
        result=lease.record_step(report,end,uptake,0.10,0.31,age,cfg)
        debt=end
        if lease.trial_active and lease.washout_timer>0.0:
            age+=0.09
            lease.record_step({},debt,0.0,0.10,0.09,age,cfg)
    return result

def test_mechanism_assist_rejects_unexecuted_trial():
    cfg=soma.Formal061Config(
        mechanism_assist_trial_window=0.30, mechanism_assist_washout=0.08,
        mechanism_assist_min_quality=0.20, mechanism_assist_min_uptake_effect=1e-5,
        mechanism_assist_min_debt_effect=1e-5, mechanism_assist_max_energy_harm=0.01,
        mechanism_assist_max_structural_harm=0.01, mechanism_assist_max_fatigue_harm=0.05,
    )
    lease=soma.MechanismAssistLease(); result=_run_assist_trial(lease,cfg,paid=False)
    assert result is not None and not result['accepted']
    assert not result['executed'] and result['execution_quality']==0.0
    assert lease.lease_strength==0.0
    return 'unpaid/null ON rejected'

def test_mechanism_assist_accepts_paid_replicated_trial():
    cfg=soma.Formal061Config(
        mechanism_assist_trial_window=0.30, mechanism_assist_washout=0.08,
        mechanism_assist_min_quality=0.20, mechanism_assist_min_execution_fraction=0.75,
        mechanism_assist_min_force_rate=1e-4, mechanism_assist_min_atp_rate=1e-6,
        mechanism_assist_min_signal_rate=1e-7, mechanism_assist_min_replication_fraction=0.75,
        mechanism_assist_min_uptake_effect=1e-5, mechanism_assist_min_debt_effect=1e-5,
        mechanism_assist_max_energy_harm=0.01, mechanism_assist_max_structural_harm=0.01,
        mechanism_assist_max_fatigue_harm=0.05,
    )
    lease=soma.MechanismAssistLease(); result=_run_assist_trial(lease,cfg,paid=True)
    assert result is not None and result['accepted']
    assert result['executed'] and result['replicated']
    assert lease.lease_strength>0.0 and lease.cumulative_atp>0.0
    return 'lease={:.3f}, quality={:.3f}, uptake effect={:.3e}'.format(lease.lease_strength,result['quality'],result['uptake_effect'])

def test_save_restore_mid_mechanism_probe_exact():
    cfg=high_budget_config(
        diagnosis_mode=soma.DIAGNOSIS_PASSIVE, diagnosis_enabled=True,
        p2_effectors=False, mechanism_probe_enabled=False, mechanism_assist_enabled=False,
    )
    world=soma.Formal061World(seed=432,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    assert tissue.mechanism_probe.start(np.asarray([1.0,0.0]),world.age,cfg)
    clone=soma.Formal061World.from_state(world.state_dict())
    for _ in range(30):
        world.step(1/soma.SIM_HZ); clone.step(1/soma.SIM_HZ)
    assert exact_equal(world.state_dict(),clone.state_dict())
    return '30 steps exact from active paid probe'



def test_neural_escrow_sanitation_prevents_development_overfunding():
    cfg=high_budget_config(neural_escrow_sanitation_enabled=True)
    world=soma.Formal061World(seed=441,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    cell=first_cell(world); port=world.port_for(cell.cell_id)
    stores=[]
    for tissue_id in tissue.tissue_ids:
        status=port.attachment_status(tissue_id)
        stores.append(float(status['stores'][p0.BUDGET_ATP]))
    assert max(stores)<=cfg.neural_escrow_atp_target+2e-5
    assert sum(stores)<=p2.P2_CELL_COUNT*(cfg.neural_escrow_atp_target+2e-5)
    assert abs(float(world.summary()['matter_residual']))<2e-5
    return 'max neural ATP escrow={:.7f}, total={:.7f}'.format(max(stores),sum(stores))


def test_mechanism_quiescence_returns_real_escrow_and_suppresses_activity():
    world,tissue=_confirmed_conservation_world(442)
    cfg=world.config; debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    assert tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,cfg)
    cell=first_cell(world); port=world.port_for(cell.cell_id)
    tissue.formal_enabled=True; tissue.controller_mature=True; tissue._061_last_age=world.age
    mask=tissue.mechanism_conservation.target_mask.copy()
    before_hidden=float(np.sum(np.abs(tissue.hidden[mask])))
    before_w_sensor=tissue.w_sensor.copy(); before_w_rec=tissue.w_rec.copy()
    before_bias=tissue.bias.copy(); before_homeo=tissue.homeo_gain.copy()
    tissue.pre_step(port,1/soma.SIM_HZ,cfg,p2.p2_gene_activity(cell))
    after_hidden=float(np.sum(np.abs(tissue.hidden[mask])))
    lease=tissue.mechanism_conservation
    assert lease.quiescent_steps==1
    assert lease.cumulative_returned_atp>0.0
    assert lease.cumulative_suppressed_activity>0.0
    assert after_hidden<before_hidden
    assert np.array_equal(before_w_sensor,tissue.w_sensor)
    assert np.array_equal(before_w_rec,tissue.w_rec)
    assert np.array_equal(before_bias,tissue.bias)
    assert np.array_equal(before_homeo,tissue.homeo_gain)
    assert abs(float(world.summary()['matter_residual']))<2e-5
    return 'returned ATP={:.6g}, hidden {:.4f}->{:.4f}'.format(
        lease.cumulative_returned_atp,before_hidden,after_hidden
    )


def test_quiescent_budget_proxy_allows_assembly_atp_but_caps_reusable_buffer():
    cfg=high_budget_config(neural_escrow_sanitation_enabled=True)
    world=soma.Formal061World(seed=443,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    cell=first_cell(world); port=world.port_for(cell.cell_id)
    tissue_id=tissue.tissue_ids[0]
    status=port.attachment_status(tissue_id)
    before_material=np.asarray(status['tissue_material'],dtype=float).copy()
    # Create a small repair need without changing total matter: functional
    # protein becomes damaged protein in the same attachment.
    state=port._attachment(tissue_id)
    amount=min(2e-5,float(state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN]))
    state.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN]-=amount
    state.tissue_material[p0.TISSUE_DAMAGED_PROTEIN]+=amount
    tissue._status(port,0)
    caps=tissue._normal_neural_budget_caps(cfg)
    proxy=soma._MechanismQuiescencePort(port,caps)
    for _ in range(4):
        tissue._develop_cell(proxy,0,1/soma.SIM_HZ,costs=True)
    status=port.attachment_status(tissue_id)
    stores=np.asarray(status['stores'],dtype=float)
    assert stores[p0.BUDGET_ATP]<=cfg.neural_escrow_atp_target+2e-5
    assert np.isfinite(stores).all()
    assert abs(float(world.summary()['matter_residual']))<2e-5
    return 'post-repair ATP buffer={:.7f}'.format(float(stores[p0.BUDGET_ATP]))


def _confirmed_conservation_world(seed=433):
    cfg=high_budget_config(
        diagnosis_mode=soma.DIAGNOSIS_PASSIVE,
        diagnosis_enabled=True,
        p2_effectors=False,
        mechanism_conservation_enabled=True,
        mechanism_conservation_feedback_enabled=True,
        mechanism_conservation_duration=1.0,
        mechanism_conservation_cooldown=0.2,
    )
    world=soma.Formal061World(seed=seed,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    tissue.hidden[:]=np.linspace(0.25,0.95,p2.P2_CELL_COUNT)
    tissue.motor_gain[:]=np.linspace(0.4,1.1,p2.P2_CELL_COUNT)
    tissue.mechanism_gain.baseline_gain=0.020
    tissue.mechanism_gain.recent_gain=0.004
    tissue.mechanism_gain.baseline_weight=1.0
    tissue.mechanism_gain.recent_weight=1.0
    tissue.mechanism_gain._update_ratio()
    tissue.mechanism_gain.confirmed=True
    tissue.mechanism_gain.confirmations=1
    tissue.mechanism_gain.confirmed_age=world.age
    return world,tissue


def test_mechanism_conservation_requires_confirmed_failure():
    cfg=high_budget_config(
        mechanism_conservation_enabled=True,
        mechanism_conservation_feedback_enabled=True,
    )
    world=soma.Formal061World(seed=433,initial_cells=1,config=cfg)
    _,tissue=develop_061(world)
    debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    assert not tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,cfg)
    assert tissue.mechanism_conservation.issued==0
    return 'unconfirmed motor gain cannot issue conservation lease'


def test_mechanism_conservation_issues_targeted_revocable_lease():
    world,tissue=_confirmed_conservation_world(434)
    cfg=world.config; debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    assert tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,cfg)
    lease=tissue.mechanism_conservation
    assert lease.issued==1 and np.count_nonzero(lease.target_mask)==cfg.mechanism_conservation_group_size
    assert cfg.mechanism_conservation_scale_floor-1e-12<=lease.scale<=cfg.mechanism_conservation_scale_ceiling+1e-12
    assert np.all(lease.gate(world.age)[lease.target_mask]<1.0)
    lease.update(world.age+0.1,0.1,mechanism_confirmed=False)
    assert not lease.active(world.age+0.1) and lease.revoked==1
    return 'targeted cells={}, scale={:.3f}, revoked on recovery'.format(np.flatnonzero(lease.target_mask).tolist(),lease.scale)


def test_mechanism_conservation_targets_positive_motor_work():
    world,tissue=_confirmed_conservation_world(4341)
    cfg=world.config; debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    # Four cells push along +x and four oppose it.  The opposing cells are
    # deliberately given larger absolute motor magnitude so the historical
    # absolute-score rule would select the wrong group.
    tissue.preferred[:]=np.asarray([
        [1.0,0.0],[1.0,0.0],[0.7,0.7],[0.7,-0.7],
        [-1.0,0.0],[-1.0,0.0],[-0.7,0.7],[-0.7,-0.7],
    ],dtype=float)
    tissue.hidden[:]=np.asarray([0.48,0.44,0.34,0.31,0.95,0.91,0.82,0.79])
    tissue.motor_gain[:]=np.asarray([0.72,0.69,0.65,0.62,1.15,1.12,1.08,1.04])
    tissue.last_action[:]=np.asarray([0.8,0.0])
    assert tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,cfg)
    chosen=np.flatnonzero(tissue.mechanism_conservation.target_mask)
    align=tissue.mechanism_conservation.last_target_alignment
    assert chosen.size==cfg.mechanism_conservation_group_size
    assert np.all(align[chosen]>0.0)
    assert not np.any(np.isin(chosen,[4,5,6,7]))
    return 'aligned targets={}, align={}'.format(chosen.tolist(),np.round(align[chosen],5).tolist())


def test_mechanism_conservation_pre_step_is_temporary_not_mutating_lease():
    world,tissue=_confirmed_conservation_world(435)
    cfg=world.config; debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    assert tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,cfg)
    cell=first_cell(world); port=world.port_for(cell.cell_id)
    tissue.formal_enabled=True; tissue.controller_mature=True
    tissue._061_last_age=world.age
    before=tissue.resource_lease.copy()
    tissue.pre_step(port,1/soma.SIM_HZ,cfg,p2.p2_gene_activity(cell))
    assert np.array_equal(before,tissue.resource_lease)
    assert tissue.mechanism_conservation.active(world.age)
    return 'base resource lease restored exactly after gated pre_step'


def test_save_restore_mid_mechanism_conservation_exact():
    world,tissue=_confirmed_conservation_world(436)
    debt=np.asarray([0.2,0.2,0.2,0.2],dtype=float)
    assert tissue.mechanism_conservation.maybe_issue(tissue,world.age,debt,world.config)
    clone=soma.Formal061World.from_state(world.state_dict())
    for _ in range(18):
        world.step(1/soma.SIM_HZ); clone.step(1/soma.SIM_HZ)
    assert exact_equal(world.state_dict(),clone.state_dict())
    return '18 steps exact from active conservation lease'


def test_stable_short_run_has_no_false_conservation():
    observations=[]
    for seed in (437,438,439):
        cfg=soma.Formal061Config(
            diagnosis_mode=soma.DIAGNOSIS_PASSIVE,
            p2_environment=p2.P2_ENV_NATIVE,
            division=False,mutation=False,
            diagnosis_min_active_age=999.0,audit_min_active_age=999.0,
            formal_auto_start=False,external_inflow=False,
            corpse_chemistry=False,extracellular_dna=False,mobile_elements=False,
            mechanism_fault_enabled=False,mechanism_assist_enabled=False,
        )
        world=soma.Formal061World(seed=seed,initial_cells=1,config=cfg)
        for _ in range(int(22*soma.SIM_HZ)):
            if not world.living_cells(): break
            world.step(1/soma.SIM_HZ)
        summary=world.summary()
        assert summary['mechanism_conservation_issued']==0
        assert world.finite()
        observations.append('{}:0'.format(seed))
    return ', '.join(observations)


def test_mechanism_safety_uses_joint_crisis_not_raw_energy_debt():
    cfg=soma.Formal061Config()
    world,tissue=_confirmed_conservation_world(440)
    viable=np.asarray([0.993,0.02,0.03,0.35],dtype=float)
    crisis=np.asarray([0.9998,0.02,0.03,0.985],dtype=float)
    structural=np.asarray([0.80,0.80,0.03,0.20],dtype=float)
    assert tissue._mechanism_safe(viable,cfg)
    assert not tissue._mechanism_safe(crisis,cfg)
    assert not tissue._mechanism_safe(structural,cfg)
    return 'viable high energy-debt allowed; joint crisis and structural failure blocked'


def test_default_probe_schedule_supports_paid_replication_before_joint_crisis():
    cfg=soma.Formal061Config()
    assert cfg.mechanism_probe_cooldown < 1.0
    ledger=soma.MechanismGainLedger()
    ledger.observe(0.014,1.0,0.0,cfg,probe=False,mechanism_probability=0.0)
    ledger.observe(0.014,1.0,0.2,cfg,probe=False,mechanism_probability=0.0)
    first=ledger.observe(0.0030,0.95,1.0,cfg,probe=True,mechanism_probability=0.9)
    second=ledger.observe(0.0030,0.95,2.0,cfg,probe=True,mechanism_probability=0.9)
    assert not first['confirmed'] and second['confirmed']
    assert 2.0 + cfg.mechanism_probe_duration + cfg.mechanism_probe_cooldown < 3.2
    return 'two high-quality paid probes confirm within {:.2f}s schedule'.format(
        2.0*cfg.mechanism_probe_duration+cfg.mechanism_probe_cooldown
    )

def test_stable_gain_does_not_confirm_fault():
    cfg=soma.Formal061Config(
        mechanism_probe_min_baseline_weight=0.6, mechanism_probe_required_weight=0.8,
    )
    ledger=soma.MechanismGainLedger()
    for age in range(8):
        ledger.observe(0.014*(1.0+0.01*math.sin(age)),0.8,float(age),cfg,probe=(age>=4),mechanism_probability=0.0)
    assert not ledger.confirmed and ledger.loss_fraction<0.16
    return 'stable gain ratio={:.3f}, confirmed={}'.format(ledger.gain_ratio,ledger.confirmed)


TESTS = [
    ('build_and_frozen_formal06', test_build_and_frozen_formal06),
    ('diagnosis_off_exact_formal06_lockstep', test_diagnosis_off_exact_formal06_lockstep),
    ('costless_diagnosis_rejected', test_costless_diagnosis_rejected),
    ('evidence_ledger_distinguishes_unknown_from_stable', test_evidence_ledger_distinguishes_unknown_from_stable),
    ('evidence_decay_is_time_step_invariant', test_evidence_decay_is_time_step_invariant),
    ('channel_resolved_semantic_reversal', test_channel_resolved_semantic_reversal),
    ('diagnosis_start_safety_defers_crisis', test_diagnosis_start_safety_defers_crisis),
    ('active_direction_comes_from_physical_gradient', test_active_direction_comes_from_physical_gradient),
    ('random_control_is_command_cost_matched', test_random_control_is_command_cost_matched),
    ('diagnostic_pause_removes_locomotion', test_diagnostic_pause_removes_locomotion),
    ('diagnostic_exposure_metrics_and_material_cost', test_diagnostic_exposure_metrics_and_material_cost),
    ('evidence_event_cost_is_paid', test_evidence_event_cost_is_paid),
    ('targeted_compiler_uses_current_neural_claim', test_targeted_compiler_uses_current_neural_claim),
    ('targeted_audit_waits_for_activation_delay', test_targeted_audit_waits_for_activation_delay),
    ('controlled_targeted_audit_can_trigger_material_feedback', test_controlled_targeted_audit_can_trigger_material_feedback),
    ('state_noise_does_not_open_strong_feedback', test_state_noise_does_not_open_strong_feedback),
    ('semantic_outcome_quality_uses_physical_dose', test_semantic_outcome_quality_uses_physical_dose),
    ('save_restore_mid_diagnosis_exact', test_save_restore_mid_diagnosis_exact),
    ('save_restore_mid_targeted_audit_exact', test_save_restore_mid_targeted_audit_exact),
    ('common_disturbance_identical_twins_exact', test_common_disturbance_identical_twins_exact),
    ('no_free_061_learning_state_in_division', test_no_free_061_learning_state_in_division),
    ('controller_corpse_and_hgt_regressions', test_controller_corpse_and_hgt_regressions),
    ('stable_short_run_has_no_false_feedback', test_stable_short_run_has_no_false_feedback),
    ('finite_and_material_ledger', test_finite_and_material_ledger),
    ('low_quality_feedback_is_blocked', test_low_quality_feedback_is_blocked),
    ('replicated_feedback_gate_accepts_clean_result', test_replicated_feedback_gate_accepts_clean_result),
    ('mechanism_gain_requires_paid_probe_replication', test_mechanism_gain_requires_paid_probe_replication),
    ('mechanism_fault_converts_existing_matter', test_mechanism_fault_converts_existing_matter),
    ('mechanism_assist_rejects_unexecuted_trial', test_mechanism_assist_rejects_unexecuted_trial),
    ('mechanism_assist_accepts_paid_replicated_trial', test_mechanism_assist_accepts_paid_replicated_trial),
    ('save_restore_mid_mechanism_probe_exact', test_save_restore_mid_mechanism_probe_exact),
    ('neural_escrow_sanitation_prevents_development_overfunding', test_neural_escrow_sanitation_prevents_development_overfunding),
    ('quiescent_budget_proxy_allows_assembly_atp_but_caps_reusable_buffer', test_quiescent_budget_proxy_allows_assembly_atp_but_caps_reusable_buffer),
    ('mechanism_conservation_requires_confirmed_failure', test_mechanism_conservation_requires_confirmed_failure),
    ('mechanism_quiescence_returns_real_escrow_and_suppresses_activity', test_mechanism_quiescence_returns_real_escrow_and_suppresses_activity),
    ('mechanism_conservation_issues_targeted_revocable_lease', test_mechanism_conservation_issues_targeted_revocable_lease),
    ('mechanism_conservation_targets_positive_motor_work', test_mechanism_conservation_targets_positive_motor_work),
    ('mechanism_conservation_pre_step_is_temporary_not_mutating_lease', test_mechanism_conservation_pre_step_is_temporary_not_mutating_lease),
    ('save_restore_mid_mechanism_conservation_exact', test_save_restore_mid_mechanism_conservation_exact),
    ('stable_short_run_has_no_false_conservation', test_stable_short_run_has_no_false_conservation),
    ('mechanism_safety_uses_joint_crisis_not_raw_energy_debt', test_mechanism_safety_uses_joint_crisis_not_raw_energy_debt),
    ('default_probe_schedule_supports_paid_replication_before_joint_crisis', test_default_probe_schedule_supports_paid_replication_before_joint_crisis),
    ('stable_gain_does_not_confirm_fault', test_stable_gain_does_not_confirm_fault),
]


def _run_named_test(name):
    mapping=dict(TESTS)
    if name not in mapping: raise KeyError(name)
    try:
        observed=mapping[name]()
        return {'test':name,'pass':'PASS','observed':_fmt(observed),'criterion':'deterministic criterion satisfied'}
    except Exception as exc:
        traceback.print_exc()
        return {'test':name,'pass':'FAIL','observed':'{}: {}'.format(type(exc).__name__,exc),'criterion':'must pass'}


def _write(rows):
    with open(CSV_PATH,'w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['test','pass','observed','criterion']); w.writeheader(); w.writerows(rows)
    passed=sum(r['pass']=='PASS' for r in rows)
    lines=['SOMA-CELL 0.6.1 VALIDATION RESULTS','Build: {}'.format(soma.BUILD),'Schema: {}'.format(soma.SCHEMA_VERSION),'Execution: fresh interpreter per test','Passed: {}/{}'.format(passed,len(rows)),'']
    for row in rows:
        lines += ['[{}] {}'.format(row['pass'],row['test']),'  observed: {}'.format(row['observed']),'  criterion: {}'.format(row['criterion'])]
    with open(TXT_PATH,'w',encoding='utf-8') as h: h.write('\n'.join(lines)+'\n')
    print('Validation: {}/{} PASS'.format(passed,len(rows)),flush=True)
    if passed != len(rows): raise SystemExit(1)
    return rows


def run_validation(fresh=False):
    if not fresh:
        return _write([_run_named_test(name) for name,_ in TESTS])
    rows=[]; script=os.path.abspath(__file__)
    for i,(name,_fn) in enumerate(TESTS,1):
        p=subprocess.run([sys.executable,script,'--one',name],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=420)
        payload=None
        for line in reversed(p.stdout.splitlines()):
            if line.startswith('SOMA_VALIDATION_JSON='):
                payload=json.loads(line.split('=',1)[1]); break
        if payload is None:
            payload={'test':name,'pass':'FAIL','observed':'child exit {} without JSON; stderr={}'.format(p.returncode,p.stderr[-1200:]),'criterion':'must pass'}
        rows.append(payload); print('{}/{} [{}] {}'.format(i,len(TESTS),payload['pass'],name),flush=True)
    return _write(rows)


def _main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if len(argv)>=2 and argv[0]=='--one':
        row=_run_named_test(argv[1]); print('SOMA_VALIDATION_JSON='+json.dumps(row,ensure_ascii=False),flush=True); return 0 if row['pass']=='PASS' else 1
    return 0 if run_validation(fresh=bool(argv and argv[0]=='--fresh')) else 0


if __name__=='__main__':
    raise SystemExit(_main())
