# coding: utf-8
import csv,json,os,sys,traceback
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..'))
for rel in ('.','../0_6_4','../0_6_3','../0_6_2','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
 p=os.path.abspath(os.path.join(HERE,rel)); sys.path.insert(0,p) if p not in sys.path else None
import SOMA_CELL_0_6_5_pythonista as soma
PREREG=os.path.join(ROOT,'results','SOMA_CELL_0_6_5_PREREGISTRATION.json')
JSON_OUT=os.path.join(HERE,'soma_cell_0_6_5_experiment_consolidated.json'); CSV_OUT=os.path.join(HERE,'soma_cell_0_6_5_experiment_summary.csv'); REPORT=os.path.join(HERE,'SOMA_CELL_0_6_5_EXPERIMENT_REPORT.txt')

def active_count(h): return float(h['sentinel_frequency']+h['readiness_frequency']+h['organ_frequency']+h['prediction_frequency']+h['plasticity_frequency'])

def main():
 p=json.load(open(PREREG,encoding='utf-8')); evol=[]; errors=[]
 for env in p['environments']:
  for seed in p['evolution_seeds']:
   try:
    r=soma.run_grammar_evolution_assay(seed=seed,environment=env,generations=p['generations'],population=p['population'],mutation=True,seconds=p['body_assay_seconds'],standing_variation=True)
    f=r['history'][-1]; evol.append({'environment':env,'seed':seed,'best_score':r['best_score'],'best_hash':r['best_hash'],'best_traits':r['best_traits'],'final':f,'history':r['history'],'best_genome':r['best_genome'].tolist(),'events':r['event_totals'],'cache_size':r['cache_size']})
    print('EVOL',env,seed,'active',active_count(f),'organ',f['organ_frequency'],'score',f['mean_score'])
   except Exception as e: errors.append({'kind':'evolution','environment':env,'seed':seed,'error':repr(e),'traceback':traceback.format_exc()})
 # clonal mutation-disabled controls
 controls=[]
 for env in p['environments']:
  r=soma.run_grammar_evolution_assay(seed=p['mutation_disabled_control_seed'],environment=env,generations=p['generations'],population=8,mutation=False,seconds=2.0,standing_variation=False)
  controls.append({'environment':env,'final':r['history'][-1],'initial':r['history'][0]})
 # environment aggregates
 agg={}
 for env in p['environments']:
  rows=[x for x in evol if x['environment']==env]; finals=[x['final'] for x in rows]
  agg[env]={
   'replicates':len(rows),
   'mean_final_active_modules':float(np.mean([active_count(f) for f in finals])) if finals else 0.0,
   'stable_low_active_replicates':int(sum(active_count(f)<=1.5 for f in finals)),
   'mean_organ_frequency':float(np.mean([f['organ_frequency'] for f in finals])) if finals else 0.0,
   'organ_retained_replicates':int(sum(f['organ_frequency']>=0.6 and active_count(f)>=2.5 for f in finals)),
   'mean_score':float(np.mean([f['mean_score'] for f in finals])) if finals else 0.0,
  }
 # choose retained candidate env by mean active count among periodic/long_delay
 candidate=max((soma.ENV_PERIODIC,soma.ENV_LONG_DELAY),key=lambda e:agg[e]['mean_final_active_modules'])
 best=max([x for x in evol if x['environment']==candidate],key=lambda x:x['best_score'])
 seq=np.asarray(best['best_genome'],dtype=np.uint8)
 causal=soma.causal_test_genome(seq,candidate,seeds=tuple(p['causal_holdout_seeds']),seconds=p['causal_seconds'])
 ko_negative=sum(row['ko_diff']<0.0 for row in causal); reintro_positive=sum(row['reintro_diff']>0.0 for row in causal)
 stable_pass=agg[soma.ENV_STABLE]['stable_low_active_replicates']>=2
 complex_pass=agg[candidate]['organ_retained_replicates']>=2
 dep_pass=agg[candidate]['mean_final_active_modules']-agg[soma.ENV_STABLE]['mean_final_active_modules']>=1.0
 causal_ko=ko_negative>=4; causal_reintro=reintro_positive>=4
 # material check on causal rows is implicit through evaluator validation; run explicit panel
 material_rows=[]; max_res=0.0; finite=True
 for env in p['environments']:
  for seed in (7101,7102):
   rr=soma.evaluate_grammar_genome(seq,env,seed,2.0); material_rows.append({'environment':env,'seed':seed,'finite':rr['finite'],'residual':rr['material_residual']}); finite=finite and rr['finite']; max_res=max(max_res,abs(rr['material_residual']))
 material_pass=finite and max_res<3e-5
 passed=stable_pass and complex_pass and dep_pass and causal_ko and causal_reintro and material_pass and not errors
 decision='PASS_ENVIRONMENT_DEPENDENT_RETENTION' if passed else 'NEGATIVE_OR_INCOMPLETE_ENVIRONMENT_DEPENDENCE'
 out={'build':soma.BUILD,'schema':soma.SCHEMA_VERSION,'preregistration':p,'evolution':evol,'controls':controls,'aggregates':agg,'retained_candidate_environment':candidate,'causal':causal,'causal_knockout_negative_pairs':ko_negative,'causal_reintroduction_positive_pairs':reintro_positive,'material_rows':material_rows,'max_material_residual':max_res,'gates':{'stable_shrinkage':stable_pass,'complex_retention':complex_pass,'environment_dependence':dep_pass,'causal_knockout':causal_ko,'causal_reintroduction':causal_reintro,'material':material_pass},'passed':passed,'decision':decision,'errors':errors}
 json.dump(out,open(JSON_OUT,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
 with open(CSV_OUT,'w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['environment','replicates','mean_final_active_modules','mean_organ_frequency','retained_replicates','mean_score']);
  for env in p['environments']:
   a=agg[env]; w.writerow([env,a['replicates'],a['mean_final_active_modules'],a['mean_organ_frequency'],a['organ_retained_replicates'],a['mean_score']])
 lines=[soma.BUILD+' EXPERIMENT REPORT','Schema: '+soma.SCHEMA_VERSION,'Decision: '+decision,'','Preregistered evolution: {} environments x {} seeds = {} lineage assays'.format(len(p['environments']),len(p['evolution_seeds']),len(evol)),'']
 for env in p['environments']:
  a=agg[env]; lines.append('{}: active {:.3f}, organ {:.3f}, retained {}/{}, mean score {:.6f}'.format(env,a['mean_final_active_modules'],a['mean_organ_frequency'],a['organ_retained_replicates'],a['replicates'],a['mean_score']))
 lines += ['','Retained candidate: '+candidate,'Causal knockout negative pairs: {}/6'.format(ko_negative),'Causal reintroduction positive pairs: {}/6'.format(reintro_positive),'Max material residual: {:.3e}'.format(max_res),'','Gates: '+repr(out['gates']),'','Interpretation: selection is over material genome variants scored by identical-seed SOMA body-world assays. A failed gate remains a negative result; it is not post-hoc retuned.']
 open(REPORT,'w',encoding='utf-8').write('\n'.join(lines)+'\n'); print('\n'.join(lines)); return 0
if __name__=='__main__': raise SystemExit(main())
