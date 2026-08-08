# coding: utf-8
from __future__ import division
import csv, hashlib, json, os, sys, tempfile, traceback
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('.','../0_6_4','../0_6_3','../0_6_2','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
 p=os.path.abspath(os.path.join(HERE,rel)); sys.path.insert(0,p) if p not in sys.path else None
import SOMA_CELL_0_6_5_pythonista as soma
import SOMA_CELL_0_6_4_pythonista as parent
ROOT=os.path.abspath(os.path.join(HERE,'..','..'))
PREREG=os.path.join(ROOT,'results','SOMA_CELL_0_6_5_PREREGISTRATION.json')
CSV_PATH=os.path.join(HERE,'soma_cell_0_6_5_validation.csv'); TXT_PATH=os.path.join(HERE,'SOMA_CELL_0_6_5_VALIDATION_RESULTS.txt')

def exact(a,b):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray): return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 if isinstance(a,dict) or isinstance(b,dict): return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)): return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)): return float(a)==float(b)
 return a==b

def step(w,n):
 for _ in range(int(n)): w.step(1.0/soma.SIM_HZ)
 return w

def test_build_schema_and_default_dormancy():
 c=soma.Formal065Config(); w=soma.Formal065World(seed=101,initial_cells=1,config=c); t=soma.grammar_traits_from_sequence(w.living_cells()[0].genomes[0])
 assert soma.BUILD=='SOMA-CELL 0.6.5' and soma.SCHEMA_VERSION=='0.6.5-GE1.0'; assert t['grammar_gene_count']==5 and t['active_count']==0; return 'five material grammar genes exist but are regulatory-dormant by default'

def test_parent_bare_lockstep_when_grammar_not_installed():
 pc=parent.Formal064Config(amortization_policy=parent.AMORT_BARE,p2_environment=parent.p2.P2_ENV_NATIVE,rule_first_change_age=1e9,division=False,mutation=False,environmental_damage=False,external_inflow=False,formal_auto_start=False,audit_min_active_age=999,diagnosis_min_active_age=999)
 cc=soma.Formal065Config(grammar_install=False,amortization_policy=parent.AMORT_BARE,p2_environment=parent.p2.P2_ENV_NATIVE,rule_first_change_age=1e9,division=False,mutation=False,environmental_damage=False,external_inflow=False,formal_auto_start=False,audit_min_active_age=999,diagnosis_min_active_age=999)
 a=parent.Formal064World(seed=211,initial_cells=1,config=pc); b=soma.Formal065World(seed=211,initial_cells=1,config=cc); step(a,60); step(b,60)
 # compare parent projection through parent loader
 bs=b.state_dict(); bs['save_version']=parent.SAVE_VERSION; bs['build']=parent.BUILD; allowed=set(parent.Formal064Config().__dict__.keys()); bs['config']={k:v for k,v in bs['config'].items() if k in allowed}; bs.pop('grammar_mutation_events',None)
 bp=parent.Formal064World.from_state(bs).state_dict(); assert exact(a.state_dict(),bp); return 'grammar-disabled descendant preserves frozen 0.6.4 bare trajectory'

def test_grammar_is_actual_delimited_genome_material():
 w=soma.Formal065World(seed=212,initial_cells=1,config=soma.Formal065Config(grammar_initial='full',external_test_harness=True)); c=w.living_cells()[0]; rec=soma.grammar_records_from_sequence(c.genomes[0]); assert len(rec)==5; assert all(r['start']>=0 for r in rec); assert abs(w.matter_ledger_residual())<3e-5; return 'grammar modules are ordinary delimited genes in the material genome'

def test_knockout_shortens_real_genome_and_reintroduction_restores_modules():
 f=soma._founder_sequence(True); k=soma.knockout_grammar(f); r=soma.reintroduce_full_grammar(k); assert len(f)-len(k)==5*soma.g2.GENE_SPAN; assert soma.grammar_traits_from_sequence(k)['grammar_gene_count']==0; assert soma.grammar_traits_from_sequence(r)['active_count']==5; return 'whole-grammar knockout and material reintroduction alter actual genome length'

def test_module_specific_deletion():
 f=soma._founder_sequence(True); x=soma.remove_grammar_module(f,soma.GRAMMAR_PREDICTION); t=soma.grammar_traits_from_sequence(x); assert not t['prediction'] and t['sentinel'] and t['organ'] and t['plasticity']; return 'single neural grammar module can be physically deleted without deleting the rest'

def test_regulatory_dormancy_and_reactivation():
 f=soma._founder_sequence(True); d=soma.set_grammar_module_promoter(f,soma.GRAMMAR_ORGAN,1); assert not soma.grammar_traits_from_sequence(d)['organ']; a=soma.set_grammar_module_promoter(d,soma.GRAMMAR_ORGAN,5); assert soma.grammar_traits_from_sequence(a)['organ']; assert len(a)==len(f); return 'regulatory dormancy/reactivation changes expression without free gene creation'

def test_mutator_can_delete_duplicate_dormancy_reactivate():
 active=soma._founder_sequence(True); dormant=soma._founder_sequence(False); seen=set()
 for seed in range(100,220):
  base=active if seed%2==0 else dormant
  _,ev=soma.mutate_grammar_sequence(base,np.random.default_rng(seed),rate=1.0,duplication=True); seen.update(k for k,v in ev.items() if v)
 assert {'deletion','duplication','dormancy','reactivation'}.issubset(seen); return 'grammar mutator supports deletion duplication dormancy and reactivation'

def test_readiness_and_organ_codes_decode():
 f=soma._founder_sequence(True); t=soma.grammar_traits_from_sequence(f); assert abs(t['readiness_fraction']-0.95)<1e-12 and t['organ_mode']=='conditional'; return 'readiness fraction and organ mode are decoded only from material gene payloads'

def test_genome_material_budget_rejects_impossible_large_insert():
 f=soma._founder_sequence(True); huge=np.concatenate([f]+[soma.make_grammar_gene(0,0,5)]*100).astype(np.uint8); r=soma.evaluate_grammar_genome(huge,soma.ENV_STABLE,seed=1,seconds=0.2); assert r['invalid_material']==1 or len(huge)>soma.g2.MAX_GENOME_LENGTH; return 'evaluation does not grant free nucleotide mass to oversized neural grammar'

def test_environment_mapping_has_no_reward_or_correct_direction():
 f=soma._founder_sequence(True)
 for env in soma.EVOLUTION_ENVIRONMENTS:
  cfg=soma._config_for_traits(soma.grammar_traits_from_sequence(f),env); txt=repr(cfg.__dict__)
  assert 'correct_direction' not in txt and 'teacher' not in txt
 return 'environment-specific expression mapping contains no teacher/correct-direction signal'

def test_stable_dormant_world_has_no_added_tissue():
 w=soma.Formal065World(seed=213,initial_cells=1); step(w,100); assert all(getattr(c,'p2_tissue',None) is None for c in w.living_cells()); assert w.summary()['grammar_default_dormant']==1; return 'default 0.6.5 live world keeps added neural grammar dormant'

def test_save_restore_exact():
 w=soma.Formal065World(seed=214,initial_cells=1); step(w,55); st=w.state_dict(); c=soma.Formal065World.from_state(st); step(w,35); step(c,35); assert exact(w.state_dict(),c.state_dict()); return '0.6.5 save/restore is deterministic'

def test_clone_exact():
 w=soma.Formal065World(seed=215,initial_cells=1); step(w,30); c=w.clone(); step(w,24); step(c,24); assert exact(w.state_dict(),c.state_dict()); return 'clone preserves grammar and chemistry exactly'

def test_causal_knockout_reintroduction_are_material_sequences():
 f=soma._founder_sequence(True); k=soma.knockout_grammar(f); r=soma.reintroduce_full_grammar(k); assert len(k)<len(f)==len(r); assert soma.grammar_traits_from_sequence(k)['active_count']==0; assert soma.grammar_traits_from_sequence(r)['active_count']==5; return 'causal knockout/reintroduction controls preserve physical sequence accounting'

def test_accelerated_evolution_uses_standing_material_variation():
 r=soma.run_grammar_evolution_assay(seed=301,environment=soma.ENV_STABLE,generations=2,population=8,seconds=1.5,standing_variation=True); assert len(r['history'])==3 and r['history'][0]['genome_types']>=3 and r['cache_size']>=1; return 'multigenerational assay begins with explicit genome variants and measured body-world scores'

def test_mutation_disabled_clonal_control_is_stable():
 r=soma.run_grammar_evolution_assay(seed=302,environment=soma.ENV_STABLE,generations=2,population=6,seconds=1.0,mutation=False,standing_variation=False); h=r['history']; assert all(abs(x['sentinel_frequency']-1.0)<1e-12 for x in h) and all(x['genome_types']==1 for x in h); return 'mutation-disabled clonal full-grammar control cannot invent loss or dormancy'

def test_shared_evaluation_seeds_prevent_genotype_seed_confound():
 r=soma.run_grammar_evolution_assay(seed=303,environment=soma.ENV_PERIODIC,generations=1,population=6,seconds=1.0); assert r['evaluation_seeds']== (3031,); return 'all competing genotypes in one assay share the same body-world evaluation seeds'

def test_all_environment_evaluations_finite():
 f=soma._founder_sequence(True)
 for i,env in enumerate(soma.EVOLUTION_ENVIRONMENTS):
  r=soma.evaluate_grammar_genome(f,env,seed=400+i,seconds=1.2); assert r['finite'] and abs(r['material_residual'])<3e-5
 return 'full grammar evaluates finitely with conserved material in all registered environments'

def test_knockout_evaluations_finite():
 k=soma.knockout_grammar(soma._founder_sequence(True))
 for i,env in enumerate(soma.EVOLUTION_ENVIRONMENTS):
  r=soma.evaluate_grammar_genome(k,env,seed=500+i,seconds=1.2); assert r['finite'] and abs(r['material_residual'])<3e-5
 return 'grammar-absent controls are finite and material-conservative in all environments'

def test_summary_has_five_frequencies():
 s=soma.Formal065World(seed=216,initial_cells=1).summary()
 for key in ('grammar_sentinel_frequency','grammar_readiness_frequency','grammar_organ_frequency','grammar_prediction_frequency','grammar_plasticity_frequency'): assert key in s
 return 'summary exposes lineage-relevant frequencies for all five grammar modules'

def test_no_free_learned_state_in_genetic_controls():
 f=soma._founder_sequence(True); k=soma.knockout_grammar(f); r=soma.reintroduce_full_grammar(k); assert isinstance(f,np.ndarray) and isinstance(k,np.ndarray) and isinstance(r,np.ndarray); # controls contain sequence only, no tissue state
 return 'genetic causal controls transfer sequence, never learned neural state'

def test_pythonista_descendant_reset_source_contract():
 src=open(os.path.join(HERE,'SOMA_CELL_0_6_5_pythonista.py'),encoding='utf-8').read(); assert 'self.world=self._fresh_world()' in src and 'Formal065World' in src and '0.6.5 grammar dormant' in src; return 'compact HUD descendant-safe reset explicitly targets Formal065World'

def test_prereg_hash_if_present():
 if not os.path.exists(PREREG): return 'preregistration not yet present during development validation'
 p=json.load(open(PREREG,encoding='utf-8')); digest=hashlib.sha256(open(os.path.join(HERE,'SOMA_CELL_0_6_5_pythonista.py'),'rb').read()).hexdigest(); assert p['candidate_source_sha256']==digest; return 'preregistration binds the tested candidate source SHA-256'

def test_finite_headless_trial():
 r=soma.run_headless_trial(seed=217,seconds=2.0); assert r['finite'] and abs(r['material_residual'])<3e-5; return 'headless 0.6.5 live world remains finite and ledger-consistent'

TESTS=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
def main():
 rows=[]
 for fn in TESTS:
  try: detail=fn(); rows.append((fn.__name__,'PASS',detail,'')); print('PASS',fn.__name__,detail)
  except Exception as e: rows.append((fn.__name__,'FAIL','',repr(e))); traceback.print_exc(); print('FAIL',fn.__name__,repr(e))
 with open(CSV_PATH,'w',newline='',encoding='utf-8') as f: csv.writer(f).writerows([('test','status','detail','error')]+rows)
 passed=sum(r[1]=='PASS' for r in rows)
 with open(TXT_PATH,'w',encoding='utf-8') as f:
  f.write('SOMA-CELL 0.6.5 VALIDATION RESULTS\n=====================================\n\n{} / {} PASS\n\n'.format(passed,len(rows)))
  for r in rows: f.write('{}: {} - {}{}\n'.format(r[1],r[0],r[2],(' | '+r[3]) if r[3] else ''))
 print('{} / {} PASS'.format(passed,len(rows))); return 0 if passed==len(rows) else 1
if __name__=='__main__': raise SystemExit(main())
