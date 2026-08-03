# coding: utf-8
"""R2 development assays for SOMA-CELL 0.6.1-D1.4.

This module is deliberately separate from the final holdout.  It uses a
body-centred microfluidic bath to isolate the physiological value of a
confirmed actuator-fault conservation lease from spatial foraging geometry.
Existing particles are repositioned; no matter is created or destroyed.
"""
from __future__ import print_function
import math, os, sys, time
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
    path=os.path.abspath(os.path.join(HERE,rel))
    if path not in sys.path: sys.path.insert(0,path)
import SOMA_CELL_0_6_1_pythonista as soma

BRANCH_AGE=27.0
FAULT_AGE=28.0
FINAL_AGE=52.0


def config(fault=False):
    return soma.Formal061Config(
        diagnosis_mode=soma.DIAGNOSIS_PASSIVE,
        p2_environment=soma.p2.P2_ENV_MOVING_PATCH,
        p2_patch_period=12.0,
        division=False, mutation=False,
        diagnosis_min_active_age=999.0, audit_min_active_age=999.0,
        formal_auto_start=False, p2_prediction=False,
        mechanism_probe_enabled=True, mechanism_assist_enabled=False,
        mechanism_fault_enabled=bool(fault), mechanism_fault_age=FAULT_AGE,
        mechanism_fault_gain_scale=0.25, mechanism_fault_wear=0.00020,
    )


def _ring_positions(centre, count, radius, phase=0.0, radial_spread=0.18):
    centre=np.asarray(centre,dtype=float).reshape(2)
    out=np.zeros((count,2),dtype=float)
    for j in range(count):
        a=phase+2.0*math.pi*(j+0.5)/max(count,1)
        r=radius*(1.0+radial_spread*math.sin(3.0*a+0.37))
        out[j]=(centre+r*np.array([math.cos(a),math.sin(a)]))%1.0
    return out


def _cloud_positions(centre, count, radius, phase=0.0):
    centre=np.asarray(centre,dtype=float).reshape(2)
    out=np.zeros((count,2),dtype=float)
    golden=2.399963229728653
    for j in range(count):
        a=phase+golden*j
        r=radius*math.sqrt((j+0.5)/max(count,1))
        out[j]=(centre+r*np.array([math.cos(a),math.sin(a)]))%1.0
    return out


def _set_kind_positions(world, kind, positions):
    idx=np.flatnonzero(world.field.kind==int(kind))
    if len(idx)==0: return
    arr=np.asarray(positions,dtype=float)
    if arr.shape[0]!=len(idx): raise ValueError('position count mismatch')
    world.field.pos[idx]=arr%1.0


def cue_direction(seed):
    angle=2.0*math.pi*((int(seed)*0.6180339887498949+0.1732050807568877)%1.0)
    return np.asarray([math.cos(angle),math.sin(angle)],dtype=float)


def apply_body_centred_bath(world, direction):
    """Keep the same material bath around each twin's own current body.

    Fuel/mineral are isotropic, so locomotion is not required for maintenance.
    A neutral ALT lobe provides a directional sensorimotor demand that makes a
    broken actuator metabolically costly.  Waste is held on the opposite side.
    Only particle positions change; all amounts and kinds are untouched.
    """
    living=world.living_cells()
    if not living: return
    cell=living[0]
    centre=np.asarray(cell.pos,dtype=float)
    direction=np.asarray(direction,dtype=float); direction=direction/max(np.linalg.norm(direction),1e-12)
    perp=np.asarray([-direction[1],direction[0]])
    fuel_idx=np.flatnonzero(world.field.kind==soma.s5.PARTICLE_FUEL)
    mineral_idx=np.flatnonzero(world.field.kind==soma.s5.PARTICLE_MINERAL)
    alt_idx=np.flatnonzero(world.field.kind==soma.s5.PARTICLE_ALT)
    waste_idx=np.flatnonzero(world.field.kind==soma.s5.PARTICLE_WASTE)
    _set_kind_positions(world,soma.s5.PARTICLE_FUEL,_ring_positions(centre,len(fuel_idx),0.052,phase=0.1))
    _set_kind_positions(world,soma.s5.PARTICLE_MINERAL,_ring_positions(centre,len(mineral_idx),0.064,phase=0.7))
    alt_centre=(centre+0.082*direction+0.006*perp)%1.0
    waste_centre=(centre-0.115*direction)%1.0
    _set_kind_positions(world,soma.s5.PARTICLE_ALT,_cloud_positions(alt_centre,len(alt_idx),0.020,phase=0.3))
    _set_kind_positions(world,soma.s5.PARTICLE_WASTE,_cloud_positions(waste_centre,len(waste_idx),0.035,phase=1.1))


def whole_atp(world):
    living=world.living_cells()
    if not living:return 0.0
    cell=living[0]; total=float(cell.pools[soma.s5.POOL_ATP]);port=world.port_for(cell.cell_id)
    for tissue_id in list(cell.neural_attachments.keys()):
        try: total+=float(port.attachment_status(tissue_id)['stores'][soma.p0.BUDGET_ATP])
        except Exception: pass
    return total


def run_tethered_stable_single(seed, final_age=FINAL_AGE):
    world=soma.Formal061World(seed=seed,initial_cells=1,config=config(False));dt=1.0/soma.SIM_HZ;step=0
    direction=cue_direction(seed)
    while world.age<final_age and world.living_cells():
        if world.age>=BRANCH_AGE:
            world.config.p2_environment=soma.p2.P2_ENV_NATIVE
            apply_body_centred_bath(world,direction)
        soma.apply_061_common_disturbance_tape(world,seed,step,stream=2709)
        world.step(dt);step+=1
    summary=world.summary()
    return {
        'seed':int(seed),'kind':'stable_tethered','alive':int(bool(world.living_cells())),
        'finite':int(world.finite()),'confirmed':int(summary['mechanism_confirmed']),
        'lease':int(summary['mechanism_conservation_issued']),
        'feedback':int(summary['mechanism_conservation_feedback_events']),
        'margin':float(summary['mean_autopoietic_margin']),
        'returned_atp':float(summary['mechanism_conservation_returned_atp']),
        'residual':float(summary['matter_residual']),
    }


def run_tethered_fault_twin(seed, fault=True, feedback=True, final_age=FINAL_AGE):
    base=soma.Formal061World(seed=seed,initial_cells=1,config=config(fault));dt=1.0/soma.SIM_HZ;step=0
    while base.age<BRANCH_AGE and base.living_cells():
        soma.apply_061_common_disturbance_tape(base,seed,step,stream=2711);base.step(dt);step+=1
    if not base.living_cells(): raise RuntimeError('base died')
    direction=cue_direction(seed)
    treatment=base.clone();control=base.clone()
    for world in (treatment,control):
        world.config.p2_environment=soma.p2.P2_ENV_NATIVE
        world.config.mechanism_conservation_quiescence_enabled=True
        world.config.mechanism_assist_enabled=False
    treatment.config.mechanism_conservation_feedback_enabled=bool(feedback)
    control.config.mechanism_conservation_feedback_enabled=False
    auc_t=auc_c=atp_t=atp_c=0.0; uptake0_t=treatment.p2_reward_uptake_total;uptake0_c=control.p2_reward_uptake_total
    motor_atp_t=motor_atp_c=0.0
    while treatment.age<final_age and treatment.living_cells() and control.living_cells():
        apply_body_centred_bath(treatment,direction);apply_body_centred_bath(control,direction)
        soma.apply_061_common_disturbance_tape(treatment,seed,step,stream=2712)
        soma.apply_061_common_disturbance_tape(control,seed,step,stream=2712)
        treatment.step(dt);control.step(dt);step+=1
        ct=treatment.living_cells()[0];cc=control.living_cells()[0]
        auc_t+=ct.autopoietic_margin()*dt;auc_c+=cc.autopoietic_margin()*dt
        atp_t+=whole_atp(treatment)*dt;atp_c+=whole_atp(control)*dt
        rt=getattr(ct.p2_tissue,'_061_last_base_effector_report',{}) or {}
        rc=getattr(cc.p2_tissue,'_061_last_base_effector_report',{}) or {}
        motor_atp_t+=float(rt.get('atp_spent',0.0));motor_atp_c+=float(rc.get('atp_spent',0.0))
    st=treatment.summary();sc=control.summary()
    return {
        'seed':int(seed),'fault':int(bool(fault)),'feedback':int(bool(feedback)),
        'alive_t':int(bool(treatment.living_cells())),'alive_c':int(bool(control.living_cells())),
        'confirmed_t':int(st['mechanism_confirmed']),'confirmed_c':int(sc['mechanism_confirmed']),
        'lease_t':int(st['mechanism_conservation_issued']),'lease_c':int(sc['mechanism_conservation_issued']),
        'auc_t':float(auc_t),'auc_c':float(auc_c),'auc_diff':float(auc_t-auc_c),
        'whole_atp_auc_diff':float(atp_t-atp_c),
        'motor_atp_diff':float(motor_atp_t-motor_atp_c),
        'uptake_diff':float((treatment.p2_reward_uptake_total-uptake0_t)-(control.p2_reward_uptake_total-uptake0_c)),
        'dissipated_energy_diff':float(treatment.dissipated_energy-control.dissipated_energy),
        'returned_atp':float(st['mechanism_conservation_returned_atp']),
        'suppressed_activity':float(st['mechanism_conservation_suppressed_activity']),
        'target_alignment':(treatment.living_cells()[0].p2_tissue.mechanism_conservation.last_target_alignment.tolist() if treatment.living_cells() else []),
        'target_mask':(treatment.living_cells()[0].p2_tissue.mechanism_conservation.target_mask.astype(int).tolist() if treatment.living_cells() else []),
        'residual_t':float(st['matter_residual']),'residual_c':float(sc['matter_residual']),
    }

if __name__=='__main__':
    import json
    for seed in (6101,6102,6103):
        t=time.time();r=run_tethered_fault_twin(seed,True,True);r['wall']=time.time()-t;print(json.dumps(r,sort_keys=True),flush=True)
