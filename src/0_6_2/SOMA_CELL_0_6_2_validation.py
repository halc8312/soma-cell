# coding: utf-8
"""Deterministic engineering validation for SOMA-CELL 0.6.2.

This suite validates physical module metering, conservative tissue reduction,
parent lockstep, save/restore, and the preregistered negative selection result.
It does not claim that neural tissue is generally adaptive or that the system is life.
"""
from __future__ import division
import argparse,csv,hashlib,json,os,pickle,sys,tempfile,traceback
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('.','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
 p=os.path.abspath(os.path.join(HERE,rel));
 if p not in sys.path:sys.path.insert(0,p)
import SOMA_CELL_0_6_2_pythonista as soma
import SOMA_CELL_0_6_1_pythonista as parent
CSV_PATH=os.path.join(HERE,'soma_cell_0_6_2_validation.csv')
TXT_PATH=os.path.join(HERE,'SOMA_CELL_0_6_2_VALIDATION_RESULTS.txt')
EXPERIMENT_CSV=os.path.join(HERE,'soma_cell_0_6_2_experiment_results.csv')

def exact(a,b):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):return float(a)==float(b)
 return a==b

def project(candidate,reference):
 if isinstance(reference,dict):return {k:project(candidate[k],v) for k,v in reference.items()}
 if isinstance(reference,list):return [project(candidate[i],v) for i,v in enumerate(reference)]
 if isinstance(reference,tuple):return tuple(project(candidate[i],v) for i,v in enumerate(reference))
 if isinstance(reference,np.ndarray):return np.asarray(candidate,dtype=reference.dtype).copy()
 return candidate

def strip_to_parent(child):
 ref=parent.Formal061World.from_state(parent.Formal061World(seed=1,initial_cells=1).state_dict()).state_dict()
 # Project recursively only keys that exist in a real parent state.
 return project(child,ref)

def config(profile=soma.PROFILE_FULL8,**kw):
 d=dict(neural_profile=profile,p2_environment=soma.p2.P2_ENV_NATIVE,division=False,mutation=False,environmental_damage=False,external_inflow=False,diagnosis_min_active_age=999.0,audit_min_active_age=999.0,formal_auto_start=False,p2_plasticity_warmup=0.0)
 d.update(kw);return soma.Formal062Config(**d)

def world(profile=soma.PROFILE_FULL8,**kw):return soma.Formal062World(seed=101,initial_cells=1,config=config(profile,**kw))

def step(w,n=24):
 for _ in range(n):w.step(1.0/soma.SIM_HZ)
 return w

def tissue(w):
 c=w.living_cells()[0]; assert isinstance(c.p2_tissue,soma.MetabolicAuditTissue);return c.p2_tissue

def test_build_schema_and_selection_metadata():
 assert soma.BUILD=='SOMA-CELL 0.6.2';assert soma.SCHEMA_VERSION=='0.6.2-M1.1';assert soma.AUDIT_SELECTED_PROFILE==soma.PROFILE_NO_TISSUE;assert soma.MINIMAL_NEURAL_PROFILE==soma.PROFILE_EFFICIENT2
 return 'build/schema and negative audit winner fixed'

def test_ui1_inheritance_and_safe_reset_source():
 src=open(os.path.join(HERE,'SOMA_CELL_0_6_2_pythonista.py'),encoding='utf-8').read()
 parent_path=os.path.join(HERE,'..','0_6_1','SOMA_CELL_0_6_1_pythonista.py')
 if not os.path.isfile(parent_path):parent_path=os.path.join(HERE,'SOMA_CELL_0_6_1_pythonista.py')
 p=open(parent_path,encoding='utf-8').read()
 assert 'class SomaCell062Scene(s61.SomaCell061Scene)' in src and 'def _fresh_world(self):return Formal062World' in src
 assert "s.get('formal_audits', 0)" in p and 'self.world = self._fresh_world()' in p
 return 'compact UI1 renderer inherited; reset constructs descendant world'

def test_unmetered_fails_closed():
 try:soma.Formal062Config(metabolic_meter_enabled=False)
 except ValueError:return 'unmetered runtime rejected'
 raise AssertionError('unmetered accepted')

def test_full8_unmetered_harness_parent_lockstep():
 pc=parent.Formal061Config(p2_environment=soma.p2.P2_ENV_NATIVE,division=False,mutation=False,environmental_damage=False,external_inflow=False,diagnosis_min_active_age=999,audit_min_active_age=999,formal_auto_start=False,p2_plasticity_warmup=0)
 cc=config(soma.PROFILE_FULL8,metabolic_meter_enabled=False,external_test_harness=True)
 a=parent.Formal061World(seed=313,initial_cells=1,config=pc);b=soma.Formal062World(seed=313,initial_cells=1,config=cc)
 for _ in range(45):a.step(1/soma.SIM_HZ);b.step(1/soma.SIM_HZ)
 # Build a parent state directly from the descendant's parent-compatible payload.
 bs=b.state_dict();base=dict(bs);base['save_version']=parent.SAVE_VERSION;base['build']=parent.BUILD;allowed=set(parent.Formal061Config().__dict__.keys());base['config']={k:v for k,v in bs['config'].items() if k in allowed};base['cells']=[parent.Formal061ProtoCell.from_state(b.rng,c).state_dict() for c in bs['cells']]
 assert exact(a.state_dict(),project(base,a.state_dict()))
 return 'full8 zero-meter explicit harness exact parent lockstep'

def test_profile_active_counts_and_no_tissue():
 expected={soma.PROFILE_FULL8:8,soma.PROFILE_SIZE4:4,soma.PROFILE_EFFICIENT4:4,soma.PROFILE_SIZE2:2,soma.PROFILE_EFFICIENT2:2,soma.PROFILE_SIZE1:1}
 for p,n in expected.items():
  w=step(world(p),28);assert int(np.count_nonzero(tissue(w).audit_active_mask))==n
 w=step(world(soma.PROFILE_NO_TISSUE),10);assert not any(isinstance(getattr(c,'p2_tissue',None),soma.MetabolicAuditTissue) for c in w.living_cells())
 return 'active profile counts 8/4/2/1/0'

def test_inactive_cells_retired_and_material_returned():
 w=step(world(soma.PROFILE_FULL8),30);t=tissue(w);before=w.matter_ledger_residual();soma.set_neural_profile(w,soma.PROFILE_SIZE4);w.step(1/soma.SIM_HZ);t=tissue(w)
 assert t.module_ledger.retired_cells==4 and np.count_nonzero(t.audit_active_mask)==4 and t.module_ledger.returned_material>0
 assert abs(w.matter_ledger_residual())<2e-5 and abs(before)<2e-5
 return 'four inactive compartments conservatively retired'

def test_prediction_meter_and_ablation():
 a=step(world(soma.PROFILE_FULL8),36).summary();b=step(world(soma.PROFILE_NO_PREDICTION),36).summary()
 assert a['metabolic_prediction_atp']>0 and a['metabolic_prediction_wear']>0;assert b['metabolic_prediction_atp']==0 and b['metabolic_prediction_wear']==0
 return 'prediction has explicit ATP/material cost and no-prediction zeros it'

def test_recurrence_meter_and_ablation():
 a=step(world(soma.PROFILE_FULL8),36).summary();b=step(world(soma.PROFILE_NO_RECURRENCE),36).summary()
 assert a['metabolic_recurrence_atp']>0 and a['metabolic_recurrence_wear']>0;assert b['metabolic_recurrence_atp']==0 and b['metabolic_recurrence_wear']==0
 return 'recurrence has explicit ATP/material cost and ablation zeros it'

def test_plasticity_meter_and_ablation():
 a=step(world(soma.PROFILE_FULL8),72).summary();b=step(world(soma.PROFILE_NO_PLASTICITY),72).summary()
 assert a['metabolic_plasticity_atp']>=0 and b['metabolic_plasticity_atp']==0 and b['p2_plasticity_updates']==0
 return 'plasticity ablation prevents updates and explicit charge'

def test_meter_is_materially_paid():
 w=world(soma.PROFILE_FULL8);step(w,36);s=w.summary();assert s['metabolic_module_atp_total']>0 and s['metabolic_module_material_total']>0 and abs(s['matter_residual'])<2e-5
 return 'module ledgers correspond to paid ATP and damaged material'

def test_from_state_clone_and_derived_config_field():
 w=step(world(soma.PROFILE_EFFICIENT2),18);c=w.clone();assert c.config.neural_profile==soma.PROFILE_EFFICIENT2;assert c.config.audit_active_cells==2;assert exact(w.state_dict(),c.state_dict())
 return 'clone exact; derived audit_active_cells does not leak into constructor'

def test_save_restore_exact():
 w=step(world(soma.PROFILE_EFFICIENT4),27)
 fd,path=tempfile.mkstemp(suffix='.pkl');os.close(fd)
 try:w.save(path);r=soma.Formal062World.load(path);assert exact(w.state_dict(),r.state_dict());step(w,40);step(r,40);assert exact(w.state_dict(),r.state_dict())
 finally:
  try:os.remove(path)
  except OSError:pass
 return 'save/restore and subsequent 40-step continuation exact'

def test_common_disturbance_tape_exact():
 a=world(soma.PROFILE_EFFICIENT2);b=a.clone()
 for i in range(30):soma.apply_062_common_disturbance_tape(a,91,i,7);soma.apply_062_common_disturbance_tape(b,91,i,7);a.step(1/soma.SIM_HZ);b.step(1/soma.SIM_HZ)
 assert exact(a.state_dict(),b.state_dict());return 'common disturbance twin exact'

def test_module_ledger_roundtrip():
 w=step(world(soma.PROFILE_FULL8),45);st=tissue(w).module_ledger.state_dict();o=soma.NeuralModuleLedger.from_state(st);assert exact(st,o.state_dict()) and o.total_atp()>0
 return 'module ledger roundtrip exact'

def _experiment_rows():return list(csv.DictReader(open(EXPERIMENT_CSV,encoding='utf-8')))

def _group(kind,profile):return [r for r in _experiment_rows() if r.get('kind')==kind and r.get('profile')==profile and not r.get('error')]

def test_preregistered_screening_negative_neural_net_value():
 full=np.mean([float(r['margin_auc']) for r in _group('screening','full8')]);none=np.mean([float(r['margin_auc']) for r in _group('screening','no_tissue')]);assert len(_group('screening','full8'))==4 and none>full
 return 'screening no-tissue margin {:.4f} > full8 {:.4f}'.format(none,full)

def test_preregistered_confirmation_negative_neural_net_value():
 none=np.mean([float(r['margin_auc']) for r in _group('confirmation','no_tissue')]);e2=np.mean([float(r['margin_auc']) for r in _group('confirmation','efficient2')]);assert len(_group('confirmation','no_tissue'))==6 and none>e2
 return 'confirmation no-tissue margin {:.4f} > best neural efficient2 {:.4f}'.format(none,e2)

def test_prediction_retained_by_preregistered_rule():
 full={int(float(r['seed'])):float(r['margin_auc']) for r in _group('screening','full8')};alt={int(float(r['seed'])):float(r['margin_auc']) for r in _group('screening','no_prediction')};wins=sum(alt[s]>full[s] for s in full);assert wins<3
 return 'no-prediction wins {}/4; prediction not rejected'.format(wins)

def test_recurrence_retired_by_preregistered_rule():
 full={int(float(r['seed'])):float(r['margin_auc']) for r in _group('screening','full8')};alt={int(float(r['seed'])):float(r['margin_auc']) for r in _group('screening','no_recurrence')};wins=sum(full[s]>alt[s] for s in full);assert wins<3
 return 'full recurrence wins {}/4; recurrence fails retain gate'.format(wins)

def test_efficient2_is_smallest_neural_profile_not_selected_winner():
 e2=np.mean([float(r['margin_auc']) for r in _group('confirmation','efficient2')]);e4=np.mean([float(r['margin_auc']) for r in _group('confirmation','efficient4')]);full=np.mean([float(r['margin_auc']) for r in _group('confirmation','full8')]);none=np.mean([float(r['margin_auc']) for r in _group('confirmation','no_tissue')]);assert e2>e4 and e2>full and e2<none and soma.MINIMAL_NEURAL_PROFILE=='efficient2'
 return 'efficient2 best tested neural profile but loses to no_tissue'

def test_stable_safety_no_false_lease():
 rows=_group('stable','efficient2')+_group('stable','efficient4');assert len(rows)==20;assert sum(int(float(r['false_lease'])) for r in rows)==0;assert sum(int(float(r['false_feedback'])) for r in rows)==0
 return 'stable holdout false lease/feedback 0/20'

def test_full8_fault_conservation_positive():
 rows=_group('fault','full8');assert len(rows)==6;assert sum(int(float(r['confirmed'])) for r in rows)==6;assert sum(int(float(r['lease'])) for r in rows)==6;assert sum(float(r['auc_diff'])>0 for r in rows)==6
 return 'full8 fault conservation confirmed and positive 6/6'

def test_small_profiles_make_conservation_redundant_but_unproven():
 rows=_group('fault','efficient2')+_group('fault','efficient4');assert len(rows)==12;assert sum(int(float(r['lease'])) for r in rows)==0;assert sum(abs(float(r['auc_diff']))>1e-15 for r in rows)==0
 return 'reduced profiles issued no conservation lease; robustness benefit not established'

def test_finite_and_material_residual():
 for p in (soma.PROFILE_FULL8,soma.PROFILE_EFFICIENT4,soma.PROFILE_EFFICIENT2,soma.PROFILE_NO_TISSUE):
  w=step(world(p),90);assert w.finite() and abs(w.matter_ledger_residual())<2e-5
 return 'four profiles finite; material ledger within 2e-5'

def test_summary_contract():
 s=step(world(soma.PROFILE_EFFICIENT2),20).summary();required=('metabolic_profile','metabolic_active_cells','metabolic_prediction_atp','metabolic_recurrence_atp','metabolic_plasticity_atp','metabolic_audit_selected_profile','metabolic_minimal_neural_profile')
 assert all(k in s for k in required);return 'summary exposes module costs and audit selection'

TESTS=[
 ('build_schema_and_selection_metadata',test_build_schema_and_selection_metadata),('ui1_inheritance_and_safe_reset_source',test_ui1_inheritance_and_safe_reset_source),('unmetered_fails_closed',test_unmetered_fails_closed),('full8_unmetered_harness_parent_lockstep',test_full8_unmetered_harness_parent_lockstep),('profile_active_counts_and_no_tissue',test_profile_active_counts_and_no_tissue),('inactive_cells_retired_and_material_returned',test_inactive_cells_retired_and_material_returned),('prediction_meter_and_ablation',test_prediction_meter_and_ablation),('recurrence_meter_and_ablation',test_recurrence_meter_and_ablation),('plasticity_meter_and_ablation',test_plasticity_meter_and_ablation),('meter_is_materially_paid',test_meter_is_materially_paid),('from_state_clone_and_derived_config_field',test_from_state_clone_and_derived_config_field),('save_restore_exact',test_save_restore_exact),('common_disturbance_tape_exact',test_common_disturbance_tape_exact),('module_ledger_roundtrip',test_module_ledger_roundtrip),('preregistered_screening_negative_neural_net_value',test_preregistered_screening_negative_neural_net_value),('preregistered_confirmation_negative_neural_net_value',test_preregistered_confirmation_negative_neural_net_value),('prediction_retained_by_preregistered_rule',test_prediction_retained_by_preregistered_rule),('recurrence_retired_by_preregistered_rule',test_recurrence_retired_by_preregistered_rule),('efficient2_is_smallest_neural_profile_not_selected_winner',test_efficient2_is_smallest_neural_profile_not_selected_winner),('stable_safety_no_false_lease',test_stable_safety_no_false_lease),('full8_fault_conservation_positive',test_full8_fault_conservation_positive),('small_profiles_make_conservation_redundant_but_unproven',test_small_profiles_make_conservation_redundant_but_unproven),('finite_and_material_residual',test_finite_and_material_residual),('summary_contract',test_summary_contract)]

def run_one(name):
 fn=dict(TESTS)[name]
 try:return {'name':name,'pass':1,'detail':str(fn())}
 except Exception as e:return {'name':name,'pass':0,'detail':'{}: {}'.format(type(e).__name__,e),'traceback':traceback.format_exc()}

def write(rows):
 fields=('name','pass','detail','traceback')
 with open(CSV_PATH,'w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines=['SOMA-CELL 0.6.2 VALIDATION RESULTS','Build: {} | Schema: {}'.format(soma.BUILD,soma.SCHEMA_VERSION),'Passed: {}/{}'.format(sum(r['pass'] for r in rows),len(rows)),'']
 for r in rows:lines.append('[{}] {}\n  {}'.format('PASS' if r['pass'] else 'FAIL',r['name'],r['detail']))
 with open(TXT_PATH,'w',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--test');a=ap.parse_args()
 if a.test:
  r=run_one(a.test);print(json.dumps(r,ensure_ascii=False));return 0 if r['pass'] else 1
 rows=[]
 for i,(name,_) in enumerate(TESTS,1):
  r=run_one(name);rows.append(r);print('{}/{} [{}] {}'.format(i,len(TESTS),'PASS' if r['pass'] else 'FAIL',name),flush=True)
 write(rows);return 0 if all(r['pass'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
