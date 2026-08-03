# coding: utf-8
from __future__ import print_function
import argparse, concurrent.futures, csv, hashlib, json, math, os, sys, time, traceback
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,'..','0_6_1'))
import SOMA_CELL_0_6_2_pythonista as soma
import SOMA_CELL_0_6_1_R2_development as r2base
PREREG=os.path.join(HERE,'..','..','results','SOMA_CELL_0_6_2_R4_RELEASE_RECORD.json')
if not os.path.isfile(PREREG): PREREG=os.path.join(HERE,'SOMA_CELL_0_6_2_R4_RELEASE_RECORD.json')
RESULTS=os.path.join(HERE,'soma_cell_0_6_2_experiment_results.csv'); SUMMARY=os.path.join(HERE,'soma_cell_0_6_2_experiment_summary.csv'); REPORT=os.path.join(HERE,'SOMA_CELL_0_6_2_EXPERIMENT_REPORT.txt')

def sha256(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def load_prereg():
 data=json.load(open(PREREG,encoding='utf-8')); actual=sha256(os.path.join(HERE,'SOMA_CELL_0_6_2_pythonista.py'))
 if actual!=data['source_sha256']: raise RuntimeError('source hash mismatch {} != {}'.format(actual,data['source_sha256']))
 return data

def profile_config(profile,seconds,switch_age,warmup,stable=False,fault=False):
 env=soma.p2.P2_ENV_MOVING_PATCH if stable or fault else soma.p2.P2_ENV_CUE_REVERSAL
 return soma.Formal062Config(neural_profile=profile,p2_environment=env,p2_switch_age=switch_age,p2_plasticity_warmup=warmup,division=False,mutation=False,
  diagnosis_min_active_age=999.0 if stable or fault else 24.0,audit_min_active_age=999.0 if stable or fault else 18.0,formal_auto_start=not(stable or fault),
  mechanism_probe_enabled=True,mechanism_assist_enabled=False,mechanism_fault_enabled=bool(fault),mechanism_fault_age=max(8.0,switch_age+1.0),mechanism_fault_gain_scale=0.25,mechanism_fault_wear=0.00020)

def profile_trial(profile,seed,seconds,switch_age,warmup,stage):
 c=profile_config(profile,seconds,switch_age,warmup); r=soma.run_headless_trial(seed=seed,seconds=seconds,initial_cells=1,config=c); r.update({'kind':stage,'profile':profile}); return r

def stable_single(profile,seed,seconds=30.0):
 c=profile_config(profile,seconds,12.0,6.0,stable=True,fault=False); w=soma.Formal062World(seed=seed,initial_cells=1,config=c); dt=1.0/soma.SIM_HZ; direction=r2base.cue_direction(seed); step=0
 while w.age<seconds and w.living_cells():
  if w.age>=14.0: w.config.p2_environment=soma.p2.P2_ENV_NATIVE; r2base.apply_body_centred_bath(w,direction)
  soma.apply_062_common_disturbance_tape(w,seed,step,stream=6202); w.step(dt); step+=1
 s=w.summary(); return {'kind':'stable','profile':profile,'seed':seed,'finite':int(w.finite()),'false_lease':int(s.get('mechanism_conservation_issued',0)),'false_feedback':int(s.get('mechanism_conservation_feedback_events',0)),'margin':float(s.get('mean_autopoietic_margin',0.0)),'residual':float(s.get('matter_residual',0.0))}

def whole_atp(world):
 living=world.living_cells()
 if not living:return 0.0
 cell=living[0]; total=float(cell.pools[soma.s5.POOL_ATP]); port=world.port_for(cell.cell_id)
 for tid in list(cell.neural_attachments):
  try:total+=float(port.attachment_status(tid)['stores'][soma.p0.BUDGET_ATP])
  except Exception:pass
 return total

def fault_twin(profile,seed,seconds=34.0):
 branch=16.0; fault_age=17.0
 c=profile_config(profile,seconds,fault_age-1.0,6.0,stable=False,fault=True); c.p2_environment=soma.p2.P2_ENV_MOVING_PATCH; c.diagnosis_min_active_age=999.0; c.audit_min_active_age=999.0; c.formal_auto_start=False; c.mechanism_fault_age=fault_age; c.mechanism_probe_min_active_age=10.0
 base=soma.Formal062World(seed=seed,initial_cells=1,config=c); dt=1.0/soma.SIM_HZ; step=0
 while base.age<branch and base.living_cells(): soma.apply_062_common_disturbance_tape(base,seed,step,stream=6203); base.step(dt); step+=1
 if not base.living_cells():raise RuntimeError('base died')
 direction=r2base.cue_direction(seed); t=base.clone(); ctrl=base.clone()
 for w in (t,ctrl):w.config.p2_environment=soma.p2.P2_ENV_NATIVE; w.config.mechanism_conservation_quiescence_enabled=True; w.config.mechanism_assist_enabled=False
 t.config.mechanism_conservation_feedback_enabled=True; ctrl.config.mechanism_conservation_feedback_enabled=False
 auc_t=auc_c=0.0; motor_t=motor_c=0.0; up0t=t.p2_reward_uptake_total; up0c=ctrl.p2_reward_uptake_total
 while t.age<seconds and t.living_cells() and ctrl.living_cells():
  r2base.apply_body_centred_bath(t,direction); r2base.apply_body_centred_bath(ctrl,direction); soma.apply_062_common_disturbance_tape(t,seed,step,stream=6204); soma.apply_062_common_disturbance_tape(ctrl,seed,step,stream=6204); t.step(dt); ctrl.step(dt); step+=1
  ct=t.living_cells()[0]; cc=ctrl.living_cells()[0]; auc_t+=ct.autopoietic_margin()*dt; auc_c+=cc.autopoietic_margin()*dt
  rt=getattr(ct.p2_tissue,'_061_last_base_effector_report',{}) or {}; rc=getattr(cc.p2_tissue,'_061_last_base_effector_report',{}) or {}; motor_t+=float(rt.get('atp_spent',0.0)); motor_c+=float(rc.get('atp_spent',0.0))
 st=t.summary(); sc=ctrl.summary(); return {'kind':'fault','profile':profile,'seed':seed,'finite':int(t.finite() and ctrl.finite()),'confirmed':int(st.get('mechanism_confirmed',0)),'lease':int(st.get('mechanism_conservation_issued',0)),'auc_diff':float(auc_t-auc_c),'motor_atp_diff':float(motor_t-motor_c),'dissipated_diff':float(t.dissipated_energy-ctrl.dissipated_energy),'uptake_diff':float((t.p2_reward_uptake_total-up0t)-(ctrl.p2_reward_uptake_total-up0c)),'residual_t':float(st.get('matter_residual',0.0)),'residual_c':float(sc.get('matter_residual',0.0))}

def job(spec):
 kind,profile,seed,seconds,switch,warm=spec; start=time.time()
 try:
  if kind in ('screening','confirmation'):row=profile_trial(profile,seed,seconds,switch,warm,kind)
  elif kind=='stable':row=stable_single(profile,seed,seconds)
  elif kind=='fault':row=fault_twin(profile,seed,seconds)
  else:raise ValueError(kind)
  row['wall_seconds']=time.time()-start; return row
 except Exception as e:return {'kind':kind,'profile':profile,'seed':seed,'error':'{}: {}'.format(type(e).__name__,e),'traceback':traceback.format_exc()[-1500:]}

def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with open(RESULTS,'w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def fnum(r,k,d=0.0):
 try:return float(r.get(k,d) or d)
 except:return float(d)

def summarize(rows):
 groups={}
 for r in rows:
  if r.get('error'):continue
  groups.setdefault((r.get('kind'),r.get('profile')),[]).append(r)
 out=[]
 for (kind,profile),rs in sorted(groups.items()):
  rec={'kind':kind,'profile':profile,'n':len(rs),'errors':0}
  for k in ('margin_auc','uptake_delta','metabolic_module_atp_total','metabolic_module_material_total','auc_diff','motor_atp_diff','dissipated_diff','uptake_diff','false_lease','false_feedback','confirmed','lease','material_residual'):
   vals=[fnum(r,k) for r in rs if k in r]
   if vals:rec['mean_'+k]=float(np.mean(vals));rec['positive_'+k]=int(sum(v>0 for v in vals))
  residuals=[]
  for r in rs:
   for k in ('material_residual','residual','residual_t','residual_c'):
    if k in r:residuals.append(abs(fnum(r,k)))
  rec['max_abs_residual']=max(residuals or [0.0]);out.append(rec)
 with open(SUMMARY,'w',newline='',encoding='utf-8') as h:
  fields=[]
  for r in out:
   for k in r:
    if k not in fields:fields.append(k)
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(out)
 lines=['SOMA-CELL 0.6.2 EXPERIMENT REPORT','Build: {} | Schema: {}'.format(soma.BUILD,soma.SCHEMA_VERSION),'']
 for r in out:lines.append(json.dumps(r,sort_keys=True))
 with open(REPORT,'w',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
 return out

def run(stage,workers=6):
 p=load_prereg(); specs=[]
 if stage=='screening':
  d=p['screening']; specs=[(stage,x,s,d['seconds'],d['switch_age'],d['plasticity_warmup']) for x in d['profiles'] for s in d['seeds']]
 elif stage=='confirmation':
  d=p['confirmation']; specs=[(stage,x,s,d['seconds'],d['switch_age'],d['plasticity_warmup']) for x in d['profiles'] for s in d['seeds']]
 elif stage=='stable':
  d=p['stable_safety']; specs=[(stage,x,s,d['seconds'],0,0) for x in d['profiles'] for s in d['seeds']]
 elif stage=='fault':
  d=p['actuator_fault']; specs=[(stage,x,s,d['seconds'],0,0) for x in d['profiles'] for s in d['seeds']]
 else:raise ValueError(stage)
 existing=[]
 if os.path.exists(RESULTS):
  with open(RESULTS,newline='',encoding='utf-8') as h:existing=list(csv.DictReader(h))
 done={(r.get('kind'),r.get('profile'),int(float(r.get('seed',-1)))) for r in existing if not r.get('error')}; todo=[s for s in specs if (s[0],s[1],s[2]) not in done]; rows=list(existing)
 print(stage,'jobs',len(todo),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
  for row in ex.map(job,todo):rows.append(row);write(rows);print(row.get('kind'),row.get('profile'),row.get('seed'),'ERR' if row.get('error') else 'OK',flush=True)
 print(json.dumps(summarize(rows),indent=2));return 0
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screening','confirmation','stable','fault','summarize']);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
 if a.stage=='summarize':print(json.dumps(summarize(list(csv.DictReader(open(RESULTS,encoding='utf-8')))),indent=2));raise SystemExit(0)
 raise SystemExit(run(a.stage,a.workers))
