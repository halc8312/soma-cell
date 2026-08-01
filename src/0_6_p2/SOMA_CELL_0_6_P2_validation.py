# coding: utf-8
"""Deterministic validation for SOMA-CELL 0.6-P2.

P2 adds an eight-compartment material recurrent tissue to the frozen P1/P0
chemical body.  Validation separates material development, finite recurrent
signalling, local prediction, three-factor plasticity, developmental warm-up,
conservative rewiring/turnover, non-inherited learned state, and an exact-state
common-disturbance cue-reversal comparison.
"""
from __future__ import division

import csv
import hashlib
import os
import pickle
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p1'))
P0_DIR = os.path.abspath(os.path.join(HERE, '..', '0_6_p0'))
BASELINE_DIR = os.path.abspath(os.path.join(HERE, '..', 'baseline'))
for candidate in (HERE, P1_DIR, P0_DIR, BASELINE_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import SOMA_CELL_0_6_P1_pythonista as p1
import SOMA_CELL_0_6_P2_pythonista as p2

p0 = p2.p0
s5 = p2.s5
CSV_PATH = os.path.join(HERE, 'soma_cell_0_6_p2_validation.csv')
TXT_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P2_VALIDATION_RESULTS.txt')
P1_SOURCE_PATH = os.path.join(P1_DIR, 'SOMA_CELL_0_6_P1_pythonista.py')
if not os.path.isfile(P1_SOURCE_PATH):
    P1_SOURCE_PATH = os.path.join(HERE, 'SOMA_CELL_0_6_P1_pythonista.py')
P1_SHA256 = '9e3ce9c59cb81157baea2d57049339c322e450930bc2ed0292fc1f2ffdfba86c'


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


def state_hash(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def assert_close(a, b, tolerance, label):
    if abs(float(a) - float(b)) > float(tolerance):
        raise AssertionError('{}: {} != {} (tol {})'.format(label, a, b, tolerance))


def first_cell(world):
    living = world.living_cells()
    if not living:
        raise AssertionError('expected a living cell')
    return living[0]


def high_budget_config(mode=p2.P2_MODE_FULL, **kwargs):
    values = dict(
        p2_tissue_mode=mode,
        p2_environment=p2.P2_ENV_NATIVE,
        p2_effectors=False,
        p2_plasticity_warmup=999.0,
        sensorimotor=False,
        environmental_damage=False,
        neural_atp_reserve=0.0,
        neural_fuel_reserve=0.0,
        neural_mineral_reserve=0.0,
        neural_membrane_reserve=0.0,
        neural_atp_rate=4.0,
        neural_protein_rate=4.0,
        neural_membrane_rate=4.0,
        neural_signal_rate=4.0,
    )
    values.update(kwargs)
    return p2.P2Config(**values)


def develop(world, max_steps=360):
    dt = 1.0 / p2.SIM_HZ
    for step in range(max_steps):
        world.step(dt)
        tissue = first_cell(world).p2_tissue
        if tissue is not None and np.count_nonzero(tissue.mature) == p2.P2_CELL_COUNT:
            return step + 1, tissue
    raise AssertionError('P2 tissue did not mature')


def tissue_material(world, cell):
    port = world.port_for(cell.cell_id)
    total = 0.0
    for index in range(p2.P2_CELL_COUNT):
        status = port.attachment_status(p2.p2_tissue_id(index))
        total += float(np.sum(status['tissue_material']) + np.sum(status['stores'][1:]))
    return total


def test_build_and_frozen_p1():
    with open(P1_SOURCE_PATH, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == P1_SHA256
    assert p2.BUILD == 'SOMA-CELL 0.6-P2.0'
    assert p2.P2_SCHEMA_VERSION.startswith('0.6-P2')
    assert p1.BUILD == 'SOMA-CELL 0.6-P1.0'
    return 'build={}, schema={}, P1 sha={}'.format(p2.BUILD, p2.P2_SCHEMA_VERSION, digest)


def test_no_cassette_exact_p1_lockstep():
    observations = []
    for seed in (101, 202, 303):
        control = p1.P1World(
            seed=seed, initial_cells=1,
            config=p1.P1Config(
                p1_tissue_mode=p1.TISSUE_NONE,
                p1_install_gene=False,
                p1_environment=p1.ENV_NATIVE,
            ),
        )
        candidate = p2.P2World(
            seed=seed, initial_cells=1,
            config=p2.P2Config(
                p2_tissue_mode=p2.P2_MODE_NONE,
                p2_install_genes=False,
                p2_environment=p2.P2_ENV_NATIVE,
            ),
        )
        for _ in range(240):
            control.step(1.0 / p1.SIM_HZ)
            candidate.step(1.0 / p2.SIM_HZ)
        projection = p2.p1_projection(candidate)
        assert exact_equal(control.state_dict(), projection)
        assert control.rng.bit_generator.state == candidate.rng.bit_generator.state
        observations.append('{}:{}'.format(seed, state_hash(projection)[:12]))
    return '3 seeds x 240 steps exact: ' + ', '.join(observations)


def test_unmetered_and_assisted_tissue_rejected():
    for kwargs in ({'p2_neural_cost': False}, {'p2_external_assistance': True}):
        try:
            p2.P2Config(**kwargs)
            raise AssertionError('invalid P2 configuration accepted: {}'.format(kwargs))
        except ValueError:
            pass
    return 'unmetered tissue and runtime assistance fail closed'


def test_cassette_is_material_and_required():
    no_gene = p2.P2World(
        seed=310, initial_cells=1,
        config=p2.P2Config(
            p2_tissue_mode=p2.P2_MODE_FULL,
            p2_install_genes=False,
            p2_environment=p2.P2_ENV_NATIVE,
        ),
    )
    for _ in range(120):
        no_gene.step(1.0 / p2.SIM_HZ)
    assert first_cell(no_gene).p2_tissue is None

    base = p1.P1World(
        seed=311, initial_cells=1,
        config=p1.P1Config(p1_tissue_mode=p1.TISSUE_NONE, p1_install_gene=False),
    )
    candidate = p2.P2World(
        seed=311, initial_cells=1,
        config=p2.P2Config(p2_tissue_mode=p2.P2_MODE_NONE, p2_install_genes=True),
    )
    cell = first_cell(candidate)
    specs = p2.p2_neuron_specs(cell)
    extra = candidate.total_material() - base.total_material()
    assert len({item[0] for item in specs}) == p2.P2_CELL_COUNT
    assert extra > 0.0
    return 'eight indexed genes, explicit founder matter={:.9g}'.format(extra)


def test_gene_driven_eight_cell_development():
    world = p2.P2World(seed=320, initial_cells=1, config=high_budget_config())
    steps, tissue = develop(world)
    cell = first_cell(world)
    assert len(cell.neural_attachments) == p2.P2_CELL_COUNT
    assert np.all(tissue.mature)
    assert np.all(tissue.development >= 0.80)
    assert np.count_nonzero(p2.p2_gene_activity(cell) >= 0.016) == p2.P2_CELL_COUNT
    return 'all 8 material compartments mature in {} steps'.format(steps)


def test_full_and_fixed_share_initial_material_targets():
    full = p2.P2World(seed=330, initial_cells=1, config=high_budget_config(p2.P2_MODE_FULL))
    fixed = p2.P2World(seed=330, initial_cells=1, config=high_budget_config(p2.P2_MODE_FIXED))
    sf, tf = develop(full)
    sx, tx = develop(fixed)
    assert sf == sx
    cf, cx = first_cell(full), first_cell(fixed)
    pf, px = full.port_for(cf.cell_id), fixed.port_for(cx.cell_id)
    for index in range(p2.P2_CELL_COUNT):
        a = pf.attachment_status(p2.p2_tissue_id(index))
        b = px.attachment_status(p2.p2_tissue_id(index))
        assert np.array_equal(a['tissue_material'], b['tissue_material'])
        assert np.array_equal(a['stores'], b['stores'])
    assert np.array_equal(tf.rec_mask, tx.rec_mask)
    return 'equal material/stores through development; {} steps'.format(sf)


def test_physical_gradient_controls_tissue_direction():
    world = p2.P2World(seed=340, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    cell = first_cell(world)
    port = world.port_for(cell.cell_id)
    indices = np.where(world.field.kind == s5.PARTICLE_FUEL)[0]
    original = world.field.pos.copy()
    tissue.probe_state[:] = 0.0
    tissue.motor_probe[:] = 0.0
    tissue.bias[:] = 0.0
    tissue.hidden[:] = 0.0
    tissue.prev_hidden[:] = 0.0
    tissue.signal_refractory[:] = 0.0
    world.field.pos[indices] = (cell.pos + np.asarray([0.025, 0.0])) % 1.0
    tissue.pre_step(port, 1.0 / p2.SIM_HZ, world.config, p2.p2_gene_activity(cell))
    right = tissue.last_action.copy()
    tissue.hidden[:] = 0.0
    tissue.prev_hidden[:] = 0.0
    tissue.signal_refractory[:] = 0.0
    world.field.pos[indices] = (cell.pos + np.asarray([-0.025, 0.0])) % 1.0
    tissue.pre_step(port, 1.0 / p2.SIM_HZ, world.config, p2.p2_gene_activity(cell))
    left = tissue.last_action.copy()
    world.field.pos = original
    assert right[0] > 0.005 and left[0] < -0.02
    return 'right dx={:.4f}, left dx={:.4f}'.format(right[0], left[0])


def test_recurrence_gate_and_finite_messages():
    world = p2.P2World(seed=350, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    control = world.clone()
    p2.set_p2_runtime_mode(control, p2.P2_MODE_NO_RECURRENCE)
    t_full = first_cell(world).p2_tissue
    t_off = first_cell(control).p2_tissue
    pattern = np.asarray([0.75, -0.45, 0.62, 0.20, -0.30, 0.55, -0.66, 0.42])
    for item in (t_full, t_off):
        item.hidden[:] = pattern
        item.probe_state[:] = 0.0
        item.motor_probe[:] = 0.0
        item.w_sensor[:] = 0.0
    p2.apply_p2_common_disturbance_tape(world, 350, 999)
    p2.apply_p2_common_disturbance_tape(control, 350, 999)
    t_full.pre_step(world.port_for(first_cell(world).cell_id), 1.0 / p2.SIM_HZ, world.config, p2.p2_gene_activity(first_cell(world)))
    t_off.pre_step(control.port_for(first_cell(control).cell_id), 1.0 / p2.SIM_HZ, control.config, p2.p2_gene_activity(first_cell(control)))
    assert np.max(np.abs(t_full.last_recurrent_drive)) > 1e-4
    assert np.max(np.abs(t_off.last_recurrent_drive)) == 0.0
    assert np.all(t_full.signal_refractory >= 0.0) and np.all(t_full.signal_refractory <= 0.96)
    return 'full recurrent max={:.5f}; ablation=0; messages={}'.format(np.max(np.abs(t_full.last_recurrent_drive)), t_full.recurrent_messages)


def test_local_predictor_reduces_error():
    world = p2.P2World(
        seed=360, initial_cells=1,
        config=p2.P2Config(
            p2_environment=p2.P2_ENV_MOVING_PATCH,
            p2_effectors=False,
            p2_plasticity_warmup=999.0,
        ),
    )
    values = []
    for _ in range(700):
        world.step(1.0 / p2.SIM_HZ)
        tissue = first_cell(world).p2_tissue
        if tissue is not None and tissue.predictor_updates:
            values.append(tissue.last_prediction_rms)
    assert len(values) > 500
    early = float(np.mean(values[:100]))
    late = float(np.mean(values[-100:]))
    assert late < early * 0.75
    return 'prediction RMS {:.6f} -> {:.6f}'.format(early, late)


def test_developmental_warmup_blocks_plasticity():
    world = p2.P2World(
        seed=370, initial_cells=1,
        config=high_budget_config(
            p2_plasticity_warmup=100.0,
            p2_environment=p2.P2_ENV_MOVING_PATCH,
        ),
    )
    _, tissue = develop(world)
    sensor = tissue.w_sensor.copy()
    recurrent = tissue.w_rec.copy()
    motor = tissue.motor_gain.copy()
    for _ in range(180):
        world.step(1.0 / p2.SIM_HZ)
    assert tissue.active_age < 100.0
    assert tissue.plasticity_updates == 0
    assert tissue.ligand_updates == 0
    assert np.array_equal(sensor, tissue.w_sensor)
    assert np.array_equal(recurrent, tissue.w_rec)
    assert np.array_equal(motor, tissue.motor_gain)
    return 'active age {:.2f}; no long-term update'.format(tissue.active_age)


def test_three_factor_update_has_opposite_signs():
    world = p2.P2World(seed=380, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    positive = p2.EightCellMaterialTissue.from_state(tissue.state_dict())
    negative = p2.EightCellMaterialTissue.from_state(tissue.state_dict())
    for item in (positive, negative):
        item.mature[:] = True
        item.maturity[:] = 0.40
        item.prediction_error[:] = 0.10
        item.elig_sensor[:] = 0.0
        item.elig_sensor[0, 0] = 1.0
    base = float(tissue.w_sensor[0, 0])
    positive._apply_three_factor_update(1.0, world.config, np.asarray([0.20] + [0.0] * 7), 1.0)
    negative._apply_three_factor_update(1.0, world.config, np.asarray([-0.20] + [0.0] * 7), 1.0)
    assert positive.w_sensor[0, 0] > base
    assert negative.w_sensor[0, 0] < base
    return 'base={:.6f}, positive={:.6f}, negative={:.6f}'.format(base, positive.w_sensor[0, 0], negative.w_sensor[0, 0])


def test_episode_outcome_changes_ligand_sign():
    world = p2.P2World(seed=390, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    positive = tissue._ligand_episode_delta(world.config, +0.12, 0.03)
    negative = tissue._ligand_episode_delta(world.config, -0.12, 0.03)
    assert positive > 0.0 and negative < 0.0
    assert abs(negative) > abs(positive)
    return 'positive={:+.6f}, deterioration={:+.6f}'.format(positive, negative)


def test_rewiring_preserves_edge_count_and_charges_wear():
    world = p2.P2World(seed=400, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    tissue.mature[:] = True
    tissue.edge_correlation[:] = 0.0
    absent = np.argwhere((~tissue.rec_mask) & (~np.eye(p2.P2_CELL_COUNT, dtype=bool)))
    tissue.edge_correlation[tuple(absent[0])] = 0.9
    before_edges = int(np.count_nonzero(tissue.rec_mask))
    before_wear = float(np.sum(tissue.pending_wear))
    assert tissue._rewire_weak_edge()
    after_edges = int(np.count_nonzero(tissue.rec_mask))
    assert before_edges == after_edges
    assert float(np.sum(tissue.pending_wear)) > before_wear
    return 'edges={} preserved; rewire count={}'.format(after_edges, tissue.rewire_count)


def test_force_prune_returns_material_conservatively():
    world = p2.P2World(seed=410, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    cell = first_cell(world)
    before = world.total_material()
    status = world.port_for(cell.cell_id).attachment_status(p2.p2_tissue_id(0))
    mass = float(np.sum(status['tissue_material']) + np.sum(status['stores'][1:]))
    tissue.force_prune_cell(world.port_for(cell.cell_id), 0)
    assert p2.p2_tissue_id(0) not in cell.neural_attachments
    assert_close(before, world.total_material(), 4e-12, 'prune material')
    assert tissue.cooldown[0] > 0.0 and not tissue.mature[0]
    return 'returned cell material={:.7g}'.format(mass)


def test_damage_turnover_is_conservative():
    world = p2.P2World(seed=420, initial_cells=1, config=high_budget_config())
    _, tissue = develop(world)
    cell = first_cell(world)
    attachment = cell.neural_attachments[p2.p2_tissue_id(1)]
    shift = min(0.0052, float(attachment.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN]))
    attachment.tissue_material[p0.TISSUE_FUNCTIONAL_PROTEIN] -= shift
    attachment.tissue_material[p0.TISSUE_DAMAGED_PROTEIN] += shift
    before = world.total_material()
    tissue.pre_step(world.port_for(cell.cell_id), 1.0 / p2.SIM_HZ, world.config, p2.p2_gene_activity(cell))
    assert p2.p2_tissue_id(1) not in cell.neural_attachments
    assert_close(before, world.total_material(), 4e-12, 'damage turnover material')
    return 'damaged mass {:.6g} returned'.format(shift)


def test_host_death_routes_eight_cell_matter_to_corpse():
    world = p2.P2World(seed=430, initial_cells=1, config=high_budget_config())
    _, _ = develop(world)
    cell = first_cell(world)
    mass = tissue_material(world, cell)
    cell.alive = False
    cell.death_reason = 'p2-validation'
    world._handle_divisions_and_deaths()
    assert not world.cells and len(world.corpses) == 1
    assert world.p0_returned_material >= mass - 2e-10
    assert abs(world.matter_ledger_residual()) < 8e-5
    return 'eight-cell matter={:.7g}, corpse matter={:.7g}'.format(mass, world.corpses[0].material_mass())


def test_division_recycles_learned_state_and_retains_genes():
    world = p2.P2World(
        seed=440, initial_cells=1,
        config=high_budget_config(mutation=False, external_replicase=True),
    )
    _, tissue = develop(world)
    tissue.w_sensor[0, 0] = 0.777
    tissue.maturity[:] = 0.88
    cell = first_cell(world)
    cell.pools[s5.POOL_NUCLEOTIDE] = max(cell.pools[s5.POOL_NUCLEOTIDE], 8.0)
    cell.pools[s5.POOL_ATP] = max(cell.pools[s5.POOL_ATP], 8.0)
    steps = 0
    while len(cell.genomes) < 2 and steps < 12000:
        cell._replicate_genome(world, 0.1, world.config)
        steps += 1
    assert len(cell.genomes) >= 2
    mass = tissue_material(world, cell)
    cell.division_progress = 1.0
    cell.septum_mass = max(cell.septum_mass, 0.12)
    daughters = cell.split(world)
    assert daughters is not None and len(daughters) == 2
    assert all(d.p2_tissue is None for d in daughters)
    assert all(not d.neural_attachments for d in daughters)
    assert all(len({x[0] for x in p2.p2_neuron_specs(d)}) == p2.P2_CELL_COUNT for d in daughters)
    assert world.p0_returned_material >= mass - 2e-10
    assert abs(world.division_residual) < 3e-8
    return 'copy steps={}, learned arrays recycled; genes retained'.format(steps)


def test_save_restore_exact():
    world = p2.P2World(
        seed=450, initial_cells=1,
        config=p2.P2Config(p2_environment=p2.P2_ENV_CUE_REVERSAL),
    )
    for _ in range(220):
        world.step(1.0 / p2.SIM_HZ)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'p2.pkl')
        world.save(path)
        restored = p2.P2World.load(path)
    assert exact_equal(world.state_dict(), restored.state_dict())
    for _ in range(120):
        world.step(1.0 / p2.SIM_HZ)
        restored.step(1.0 / p2.SIM_HZ)
    assert exact_equal(world.state_dict(), restored.state_dict())
    return 'checkpoint and 120-step continuation exact'


def test_logger_is_noninterfering():
    observed = p2.P2World(seed=460, initial_cells=1, config=p2.P2Config(p2_environment=p2.P2_ENV_MOVING_PATCH))
    control = observed.clone()
    with tempfile.TemporaryDirectory() as directory:
        logger = p2.LongRunLogger(observed, path=os.path.join(directory, 'p2.csv'))
        for step in range(120):
            observed.step(1.0 / p2.SIM_HZ)
            control.step(1.0 / p2.SIM_HZ)
            if step % 15 == 0:
                logger.log(observed, force=True)
        assert os.path.getsize(logger.path) > 0
    assert exact_equal(observed.state_dict(), control.state_dict())
    return 'logger-neutral 120-step continuation'


def _post_switch_twin(seed=101, control_mode=p2.P2_MODE_FIXED, horizon=46.0):
    config = p2.P2Config(
        p2_tissue_mode=p2.P2_MODE_FULL,
        p2_environment=p2.P2_ENV_CUE_REVERSAL,
        p2_switch_age=54.0,
        p2_plasticity_warmup=34.0,
    )
    base = p2.P2World(seed=seed, initial_cells=1, config=config)
    dt = 1.0 / p2.SIM_HZ
    pre_steps = int(round(config.p2_switch_age * p2.SIM_HZ))
    for step in range(pre_steps):
        p2.apply_p2_common_disturbance_tape(base, seed, step, stream=31)
        base.step(dt)
    treatment = base.clone()
    control = base.clone()
    p2.set_p2_runtime_mode(treatment, p2.P2_MODE_FULL)
    p2.set_p2_runtime_mode(control, control_mode)
    treatment_auc = 0.0
    control_auc = 0.0
    for offset in range(int(round(horizon * p2.SIM_HZ))):
        step = pre_steps + offset
        p2.apply_p2_common_disturbance_tape(treatment, seed, step, stream=31)
        p2.apply_p2_common_disturbance_tape(control, seed, step, stream=31)
        treatment.step(dt)
        control.step(dt)
        treatment_auc += first_cell(treatment).autopoietic_margin() * dt
        control_auc += first_cell(control).autopoietic_margin() * dt
    return treatment, control, treatment_auc - control_auc


def test_common_disturbance_same_mode_is_exact():
    world = p2.P2World(seed=470, initial_cells=1, config=p2.P2Config(p2_environment=p2.P2_ENV_CUE_REVERSAL))
    for step in range(180):
        p2.apply_p2_common_disturbance_tape(world, 470, step, stream=8)
        world.step(1.0 / p2.SIM_HZ)
    twin = world.clone()
    for offset in range(120):
        step = 180 + offset
        p2.apply_p2_common_disturbance_tape(world, 470, step, stream=8)
        p2.apply_p2_common_disturbance_tape(twin, 470, step, stream=8)
        world.step(1.0 / p2.SIM_HZ)
        twin.step(1.0 / p2.SIM_HZ)
    assert exact_equal(world.state_dict(), twin.state_dict())
    return 'same-mode twin exact for 120 taped steps'


def test_cue_reversal_full_vs_equal_fixed_twin():
    treatment, control, gain = _post_switch_twin(seed=101, control_mode=p2.P2_MODE_FIXED)
    assert gain > 0.20
    reward_gain = treatment.p2_reward_uptake_total - control.p2_reward_uptake_total
    return 'post-switch margin AUC gain={:+.6f}, reward difference={:+.6f}'.format(gain, reward_gain)


def test_finite_and_material_ledger_stress():
    world = p2.P2World(seed=480, initial_cells=2, config=p2.P2Config(p2_environment=p2.P2_ENV_CUE_REVERSAL))
    maximum = 0.0
    for _ in range(420):
        world.step(1.0 / p2.SIM_HZ)
        maximum = max(maximum, abs(world.matter_ledger_residual()))
        assert world.finite()
    port_max = max([
        cell.neural_budget_ledger.maximum_material_residual
        for cell in world.living_cells()
    ] or [0.0])
    assert maximum < 8e-5
    assert port_max < 3e-10
    return '420 steps finite; world max={:.3e}, port max={:.3e}'.format(maximum, port_max)


TESTS = (
    ('build_and_frozen_p1', test_build_and_frozen_p1),
    ('no_cassette_exact_p1_lockstep', test_no_cassette_exact_p1_lockstep),
    ('unmetered_and_assisted_tissue_rejected', test_unmetered_and_assisted_tissue_rejected),
    ('cassette_is_material_and_required', test_cassette_is_material_and_required),
    ('gene_driven_eight_cell_development', test_gene_driven_eight_cell_development),
    ('full_and_fixed_share_initial_material_targets', test_full_and_fixed_share_initial_material_targets),
    ('physical_gradient_controls_tissue_direction', test_physical_gradient_controls_tissue_direction),
    ('recurrence_gate_and_finite_messages', test_recurrence_gate_and_finite_messages),
    ('local_predictor_reduces_error', test_local_predictor_reduces_error),
    ('developmental_warmup_blocks_plasticity', test_developmental_warmup_blocks_plasticity),
    ('three_factor_update_has_opposite_signs', test_three_factor_update_has_opposite_signs),
    ('episode_outcome_changes_ligand_sign', test_episode_outcome_changes_ligand_sign),
    ('rewiring_preserves_edge_count_and_charges_wear', test_rewiring_preserves_edge_count_and_charges_wear),
    ('force_prune_returns_material_conservatively', test_force_prune_returns_material_conservatively),
    ('damage_turnover_is_conservative', test_damage_turnover_is_conservative),
    ('host_death_routes_eight_cell_matter_to_corpse', test_host_death_routes_eight_cell_matter_to_corpse),
    ('division_recycles_learned_state_and_retains_genes', test_division_recycles_learned_state_and_retains_genes),
    ('save_restore_exact', test_save_restore_exact),
    ('logger_is_noninterfering', test_logger_is_noninterfering),
    ('common_disturbance_same_mode_is_exact', test_common_disturbance_same_mode_is_exact),
    ('cue_reversal_full_vs_equal_fixed_twin', test_cue_reversal_full_vs_equal_fixed_twin),
    ('finite_and_material_ledger_stress', test_finite_and_material_ledger_stress),
)


def run_validation():
    rows = []
    for name, function in TESTS:
        try:
            observed = function()
            passed = True
            error = ''
            print('[PASS] {}'.format(name), flush=True)
        except Exception as exc:
            observed = ''
            passed = False
            error = '{}: {}'.format(type(exc).__name__, exc)
            print('[FAIL] {} — {}'.format(name, error), flush=True)
            traceback.print_exc()
        rows.append({'test': name, 'passed': int(passed), 'observed': observed, 'error': error})

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('test', 'passed', 'observed', 'error'))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row['passed'] for row in rows)
    lines = [
        'SOMA-CELL 0.6-P2 deterministic validation',
        'build: {}'.format(p2.BUILD),
        'P2 schema: {}'.format(p2.P2_SCHEMA_VERSION),
        'P1 baseline: {}'.format(p1.BUILD),
        'result: {}/{} PASS'.format(passed, len(rows)),
        '',
    ]
    for row in rows:
        lines.append('[{}] {}'.format('PASS' if row['passed'] else 'FAIL', row['test']))
        if row['observed']:
            lines.append('  observed: {}'.format(row['observed']))
        if row['error']:
            lines.append('  error: {}'.format(row['error']))
    with open(TXT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    if passed != len(rows):
        raise SystemExit('validation failed: {}/{} PASS'.format(passed, len(rows)))
    return rows


if __name__ == '__main__':
    run_validation()
