# coding: utf-8
"""Development-only assays for SOMA-CELL 0.6.1-D1.3.

These seeds are explicitly *not* final holdouts.  The script measures the
repaired neural-escrow accounting and the physically quiescent conservation
lease before a new preregistration is sealed.
"""
from __future__ import print_function
import csv, json, math, os, sys, time
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
    path=os.path.abspath(os.path.join(HERE,rel))
    if path not in sys.path: sys.path.insert(0,path)
import SOMA_CELL_0_6_1_pythonista as soma

STABLE_SEEDS=tuple(range(3101,3111))
FAULT_SEEDS=tuple(range(3001,3007))
STABLE_SECONDS=50.0
BRANCH_AGE=27.0
FAULT_AGE=28.0
FINAL_AGE=52.0
PATCH_RADIUS=0.17
RESULTS=os.path.join(HERE,'soma_cell_0_6_1_development_results.csv')
SUMMARY=os.path.join(HERE,'soma_cell_0_6_1_development_summary.json')
REPORT=os.path.join(HERE,'SOMA_CELL_0_6_1_DEVELOPMENT_REPORT.txt')


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


def uptake(world):
    return float(world.p2_reward_uptake_total)


def whole_atp(world):
    alive=world.living_cells()
    if not alive: return 0.0
    cell=alive[0]; total=float(cell.pools[soma.s5.POOL_ATP]); port=world.port_for(cell.cell_id)
    for tissue_id in list(cell.neural_attachments.keys()):
        try: total+=float(port.attachment_status(tissue_id)['stores'][soma.p0.BUDGET_ATP])
        except Exception: pass
    return total


def place_kind(world,kind,centre,ring_radius):
    idx=np.where(world.field.kind==int(kind))[0]; centre=np.asarray(centre,float)
    for local,particle_index in enumerate(idx):
        angle=2*math.pi*(local+0.5)/max(len(idx),1)
        ring=ring_radius*(0.35+0.65*((local%5)/4.0))
        world.field.pos[particle_index]=(centre+ring*np.array([math.cos(angle),math.sin(angle)]))%1.0


def sensor_direction(cell,seed):
    direction=np.asarray(cell.p2_tissue._action_direction(),float).reshape(2)
    norm=float(np.linalg.norm(direction))
    if norm<=1e-10:
        angle=2*math.pi*((seed*0.61803398875)%1.0)
        return np.array([math.cos(angle),math.sin(angle)])
    return direction/norm


def install_patch(world,target):
    neutral=np.array([0.5,0.5])
    world.config.p2_environment=soma.p2.P2_ENV_NATIVE
    place_kind(world,soma.s5.PARTICLE_FUEL,target,0.021)
    place_kind(world,soma.s5.PARTICLE_MINERAL,target,0.026)
    place_kind(world,soma.s5.PARTICLE_ALT,neutral,0.05)
    place_kind(world,soma.s5.PARTICLE_WASTE,neutral,0.06)


def move_patch(world,target,dt):
    neutral=np.array([0.5,0.5])
    world._move_kind_to(soma.s5.PARTICLE_FUEL,target,dt,strength=12.0,radius=0.021)
    world._move_kind_to(soma.s5.PARTICLE_MINERAL,target,dt,strength=12.0,radius=0.026)
    world._move_kind_to(soma.s5.PARTICLE_ALT,neutral,dt,strength=5.0,radius=0.05)
    world._move_kind_to(soma.s5.PARTICLE_WASTE,neutral,dt,strength=5.0,radius=0.06)


def stable_trial(seed):
    started=time.time(); world=soma.Formal061World(seed=seed,initial_cells=1,config=config(False));dt=1/soma.SIM_HZ
    for step in range(int(round(STABLE_SECONDS*soma.SIM_HZ))):
        if not world.living_cells(): break
        soma.apply_061_common_disturbance_tape(world,seed,step,stream=1901);world.step(dt)
    s=world.summary()
    return {'kind':'stable','seed':seed,'wall':time.time()-started,'age':world.age,'alive':int(bool(world.living_cells())),
            'finite':int(world.finite()),'confirmed':int(s['mechanism_confirmed']),
            'lease_issued':int(s['mechanism_conservation_issued']),
            'margin':float(s['mean_autopoietic_margin']),'residual':float(s['matter_residual']),
            'sanitation_atp':float(s['neural_escrow_returned_atp'])}


def fault_twin(seed,quiescence_control=False):
    started=time.time();base=soma.Formal061World(seed=seed,initial_cells=1,config=config(True));dt=1/soma.SIM_HZ;step=0
    while base.age<BRANCH_AGE and base.living_cells():
        soma.apply_061_common_disturbance_tape(base,seed,step,stream=1911);base.step(dt);step+=1
    if not base.living_cells(): raise RuntimeError('base died')
    cell=base.living_cells()[0];direction=sensor_direction(cell,seed);target=(cell.pos+PATCH_RADIUS*direction)%1.0
    treatment=base.clone();control=base.clone();install_patch(treatment,target);install_patch(control,target)
    treatment.config.mechanism_conservation_feedback_enabled=True
    control.config.mechanism_conservation_feedback_enabled=bool(quiescence_control)
    treatment.config.mechanism_conservation_quiescence_enabled=True
    control.config.mechanism_conservation_quiescence_enabled=False if quiescence_control else True
    treatment.config.mechanism_assist_enabled=False;control.config.mechanism_assist_enabled=False
    start_t=uptake(treatment);start_c=uptake(control)
    auc_t=auc_c=atp_t=atp_c=dist_t=dist_c=0.0;count=0
    while treatment.age<FINAL_AGE and treatment.living_cells() and control.living_cells():
        move_patch(treatment,target,dt);move_patch(control,target,dt)
        soma.apply_061_common_disturbance_tape(treatment,seed,step,stream=1912)
        soma.apply_061_common_disturbance_tape(control,seed,step,stream=1912)
        treatment.step(dt);control.step(dt);step+=1
        ct=treatment.living_cells()[0];cc=control.living_cells()[0]
        auc_t+=ct.autopoietic_margin()*dt;auc_c+=cc.autopoietic_margin()*dt
        atp_t+=whole_atp(treatment)*dt;atp_c+=whole_atp(control)*dt
        dist_t+=float(np.linalg.norm(soma.p2.wrapped_delta(ct.pos,target)))
        dist_c+=float(np.linalg.norm(soma.p2.wrapped_delta(cc.pos,target)));count+=1
    st=treatment.summary();sc=control.summary()
    return {'kind':'quiescence_vs_output_only' if quiescence_control else 'fault_vs_no_lease','seed':seed,
            'wall':time.time()-started,'confirmed_t':int(st['mechanism_confirmed']),
            'lease_t':int(st['mechanism_conservation_issued']),'lease_c':int(sc['mechanism_conservation_issued']),
            'auc_t':auc_t,'auc_c':auc_c,'auc_diff':auc_t-auc_c,
            'whole_atp_auc_diff':atp_t-atp_c,'uptake_diff':(uptake(treatment)-start_t)-(uptake(control)-start_c),
            'dist_diff':(dist_t-dist_c)/max(count,1),'final_margin_diff':float(st['mean_autopoietic_margin']-sc['mean_autopoietic_margin']),
            'dissipated_energy_diff':float(treatment.dissipated_energy-control.dissipated_energy),
            'conservation_returned_atp':float(st['mechanism_conservation_returned_atp']),
            'sanitation_returned_atp':float(st['neural_escrow_returned_atp']),
            'suppressed_activity':float(st['mechanism_conservation_suppressed_activity']),
            'quiescent_steps':int(st['mechanism_conservation_quiescent_steps']),
            'residual_t':float(st['matter_residual']),'residual_c':float(sc['matter_residual'])}


def write(rows):
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    with open(RESULTS,'w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    stable=[r for r in rows if r['kind']=='stable']; fault=[r for r in rows if r['kind']=='fault_vs_no_lease']; iso=[r for r in rows if r['kind']=='quiescence_vs_output_only']
    summary={'build':soma.BUILD,'schema':soma.SCHEMA_VERSION,'stable_n':len(stable),'stable_false_lease':sum(r['lease_issued'] for r in stable),
             'stable_nonfinite':sum(1-r['finite'] for r in stable),'fault_n':len(fault),'fault_issued':sum(r['lease_t']>0 for r in fault),
             'fault_positive_auc':sum(r['auc_diff']>0 for r in fault),'fault_mean_auc':float(np.mean([r['auc_diff'] for r in fault])) if fault else 0.0,
             'fault_mean_whole_atp_auc':float(np.mean([r['whole_atp_auc_diff'] for r in fault])) if fault else 0.0,
             'fault_mean_uptake':float(np.mean([r['uptake_diff'] for r in fault])) if fault else 0.0,
             'isolation_n':len(iso),'isolation_positive_auc':sum(r['auc_diff']>0 for r in iso),
             'isolation_mean_auc':float(np.mean([r['auc_diff'] for r in iso])) if iso else 0.0,
             'max_abs_residual':max([abs(r.get('residual',0)) for r in stable]+[abs(r.get('residual_t',0)) for r in fault+iso]+[abs(r.get('residual_c',0)) for r in fault+iso] or [0])}
    with open(SUMMARY,'w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2,sort_keys=True)
    with open(REPORT,'w',encoding='utf-8') as h:h.write('SOMA-CELL 0.6.1 DEVELOPMENT REPORT\n'+json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True))


def main():
    rows=[]
    for seed in STABLE_SEEDS:
        row=stable_trial(seed);rows.append(row);print('stable',seed,row['lease_issued'],flush=True)
    for seed in FAULT_SEEDS:
        row=fault_twin(seed,False);rows.append(row);print('fault',seed,row['auc_diff'],flush=True)
    for seed in FAULT_SEEDS[:3]:
        row=fault_twin(seed,True);rows.append(row);print('isolation',seed,row['auc_diff'],flush=True)
    write(rows)

if __name__=='__main__':main()
