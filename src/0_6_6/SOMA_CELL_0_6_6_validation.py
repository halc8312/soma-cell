# coding: utf-8
from __future__ import division
import csv, json, os, tempfile, traceback
import numpy as np
import SOMA_CELL_0_6_6_pythonista as soma

HERE=os.path.dirname(os.path.abspath(__file__))
CSV_PATH=os.path.join(HERE,'soma_cell_0_6_6_validation.csv')
TXT_PATH=os.path.join(HERE,'SOMA_CELL_0_6_6_VALIDATION_RESULTS.txt')

def step(w,n=10):
    for _ in range(int(n)): w.step(1.0/soma.SIM_HZ)

def exact(a,b):
    if isinstance(a,dict): return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray): return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,float): return a==b
    return a==b

def test_build_and_schema():
    assert soma.BUILD=='SOMA-CELL 0.6.6' and soma.SCHEMA_VERSION=='0.6.6-ECO1.0'; return 'build/schema frozen'

def test_three_material_founders():
    w=soma.Formal066World(seed=1,initial_cells=3,config=soma.Formal066Config(eco66_founder_prime=False))
    cs=sorted(w.living_cells(),key=lambda c:c.lineage)
    t=[soma.s65.grammar_traits_from_sequence(c.genomes[0]) for c in cs]
    assert t[0]['active_count']==5 and t[1]['active_count']==0 and t[1]['grammar_gene_count']==5 and t[2]['grammar_gene_count']==0
    return 'full/dormant/absent founders are physical genome states'

def test_no_external_fitness_channel():
    w=soma.Formal066World(seed=2,initial_cells=3); s=w.summary(); assert s['eco66_external_fitness_events']==0
    cfg=soma.Formal066Config().__dict__
    assert 'fitness' not in cfg and 'correct_direction' not in cfg
    assert not hasattr(w,'fitness') and not hasattr(w,'correct_direction')
    return 'shared ecology adds no organism fitness/correct-direction channel'

def test_founder_g2_prime_is_initial_condition_only():
    w=soma.Formal066World(seed=3,initial_cells=3); assert all(len(c.genomes)>=2 and c.age>=80 for c in w.living_cells())
    assert abs(w.matter_ledger_residual())<1e-12; return 'founders begin as explicit G2 initial material before ledger baseline'

def test_real_division_makes_fresh_daughters():
    # Exercise the world division handler so division losses/returns are
    # measured by the same material ledger used by the simulation.
    w=soma.Formal066World(seed=4,initial_cells=3,config=soma.Formal066Config(eco66_washout=False,eco66_mutation=False))
    p=sorted(w.living_cells(),key=lambda c:c.lineage)[0]
    parent_id=int(p.cell_id); parent_generation=int(p.generation)
    assert len(p.genomes)>=2
    p.eco66_actuator_estimate=-0.7; p.eco66_estimator_evidence=1.0
    p.eco66_prev_issued=np.array([0.8,-0.2],dtype=float)
    p.division_progress=1.0
    w._handle_divisions_and_deaths()
    daughters=[d for d in w.living_cells() if int(d.generation)==parent_generation+1 and int(d.cell_id)!=parent_id]
    assert len(daughters)>=2
    daughters=sorted(daughters,key=lambda d:int(d.cell_id))[-2:]
    for d in daughters:
        assert d.generation==parent_generation+1
        assert d.eco66_estimator_evidence==0.0
        assert d.eco66_actuator_estimate==1.0
        assert np.allclose(d.eco66_prev_issued,0.0)
    assert abs(w.matter_ledger_residual())<3e-5
    return 'physical world split resets learned actuator state and stays inside material ledger gate'

def test_daughter_grammar_mutation_is_material_accounted():
    found=False
    for seed in range(10,80):
        w=soma.Formal066World(seed=seed,initial_cells=3,config=soma.Formal066Config(eco66_mutation_rate=1.0,eco66_washout=False))
        for _ in range(180): w.step(0.1)
        if w.eco66_mutation_events:
            assert abs(w.matter_ledger_residual())<3e-5; found=True; break
    assert found
    return 'at least one real daughter grammar mutation occurred with conserved matter'

def test_insertion_rejected_without_nucleotide_or_atp():
    # A grammar-absent genome requests reacquisition at rate 1. With no budget,
    # any insertion proposal must be rejected rather than granted free matter.
    seen=False
    for seed in range(90,160):
        w=soma.Formal066World(seed=seed,initial_cells=3,config=soma.Formal066Config(eco66_mutation=False,eco66_founder_prime=False))
        c=sorted(w.cells,key=lambda x:x.lineage)[2]; c.pools[soma.s5.POOL_NUCLEOTIDE]=0.0; c.pools[soma.s5.POOL_ATP]=0.0
        r=soma._material_mutate_grammar(c,w,1.0)
        if r['attempted']:
            seen=True; assert r['accepted']==0 and r['rejected_material']==1; break
    assert seen; return 'grammar insertion cannot bypass nucleotide/ATP budget'

def test_deletion_returns_nucleotide_material():
    # Force candidates until a deletion occurs, then verify exact length-mass return.
    found=False
    for seed in range(160,300):
        w=soma.Formal066World(seed=seed,initial_cells=3,config=soma.Formal066Config(eco66_mutation=False,eco66_founder_prime=False))
        c=sorted(w.cells,key=lambda x:x.lineage)[0]; before_len=len(c.genomes[0]); before=float(c.pools[soma.s5.POOL_NUCLEOTIDE])
        r=soma._material_mutate_grammar(c,w,1.0)
        if r['events'].get('deletion',0):
            delta=before_len-len(c.genomes[0]); assert delta>0
            assert abs((c.pools[soma.s5.POOL_NUCLEOTIDE]-before)-delta*soma.s5.MONOMER_MASS)<1e-12; found=True; break
    assert found; return 'whole-gene deletion returns polymer matter as nucleotide pool'

def test_chemostat_is_external_particle_injection():
    w=soma.Formal066World(seed=5,initial_cells=3,config=soma.Formal066Config(eco66_chemostat_interval=0.2,eco66_washout=False))
    injected0=float(w.field.injected_material); step(w,5)
    assert w.eco66_chemostat_events>=2 and w.field.injected_material>injected0 and abs(w.matter_ledger_residual())<3e-5
    return 'chemostat material enters as accounted environmental particles'

def test_washout_is_same_step_material_conservative_death():
    w=soma.Formal066World(seed=6,initial_cells=4,config=soma.Formal066Config(eco66_washout_start=0.0,eco66_washout_interval=100,eco66_founder_prime=False))
    before=w.total_material(); w._washout(); after=w.total_material()
    assert w.eco66_washout_events==1 and len(w.cells)==3 and len(w.corpses)>=1 and abs(before-after)<3e-8
    return 'genotype-neutral washout becomes corpse in same ledger step'

def test_hgt_off_disables_material_hgt_path():
    w=soma.Formal066World(seed=7,initial_cells=3,config=soma.Formal066Config(eco66_hgt=False,eco66_founder_prime=False))
    assert not w.config.extracellular_dna and not w.config.competence and not w.config.recombination
    return 'HGT-off control disables eDNA uptake/recombination'

def test_hgt_grammar_acquisition_counter_requires_integration():
    w=soma.Formal066World(seed=8,initial_cells=3,config=soma.Formal066Config(eco66_founder_prime=False))
    c=sorted(w.cells,key=lambda x:x.lineage)[2]; assert not soma.grammar_module_set(c.genomes[0])
    c.genomes[0]=np.concatenate([c.genomes[0],soma.s65.make_grammar_gene(soma.s65.GRAMMAR_SENTINEL,0,promoter=5)]).astype(np.uint8)
    c.hgt_integrations+=1; w._track_hgt_grammar(); assert w.eco66_grammar_hgt_acquisitions==1
    return 'grammar-HGT counter requires an integration increment plus newly present module'

def test_stable_motor_law_never_reverses():
    w=soma.Formal066World(seed=9,initial_cells=3,config=soma.Formal066Config(ecology_environment=soma.ENV_STABLE,eco66_founder_prime=False))
    for age in (0,20,100): w.age=age; assert w._eco66_motor_sign()==1.0
    return 'stable physical actuator law remains fixed'

def test_periodic_motor_law_reverses_without_teacher_signal():
    w=soma.Formal066World(seed=10,initial_cells=3,config=soma.Formal066Config(ecology_environment=soma.ENV_PERIODIC,eco66_founder_prime=False,eco66_rule_first_age=2,eco66_rule_period=3))
    w.age=0; a=w._eco66_motor_sign(); w.age=2.1; b=w._eco66_motor_sign(); w.age=5.1; c=w._eco66_motor_sign(); assert (a,b,c)==(1.0,-1.0,1.0)
    assert 'correct_direction' not in repr(w.config.__dict__)
    return 'periodic physical rule changes; config exposes no correct direction'

def test_active_grammar_pays_atp_and_wear_is_material_conversion():
    w=soma.Formal066World(seed=11,initial_cells=3,config=soma.Formal066Config(eco66_founder_prime=False,eco66_washout=False))
    c=sorted(w.cells,key=lambda x:x.lineage)[0]; diss0=float(w.dissipated_energy); step(w,3)
    assert c.eco66_grammar_atp>0 and w.dissipated_energy>diss0
    assert c.eco66_grammar_wear>=0; return 'active grammar records explicit ATP dissipation and material wear'

def test_dormant_and_absent_have_no_grammar_activity_cost():
    w=soma.Formal066World(seed=12,initial_cells=3,config=soma.Formal066Config(eco66_founder_prime=False,eco66_washout=False))
    cs=sorted(w.cells,key=lambda x:x.lineage); step(w,3)
    assert cs[1].eco66_grammar_atp==0.0 and cs[2].eco66_grammar_atp==0.0
    return 'dormant/absent added grammar does not receive active-module ATP spending'

def test_clone_is_deterministic():
    w=soma.Formal066World(seed=13,initial_cells=3,config=soma.Formal066Config(eco66_washout=False)); step(w,20); c=w.clone(); step(w,12); step(c,12); assert exact(w.state_dict(),c.state_dict())
    return '0.6.6 clone remains deterministic'

def test_save_restore_is_deterministic():
    w=soma.Formal066World(seed=14,initial_cells=3,config=soma.Formal066Config(eco66_washout=False)); step(w,16)
    fd,path=tempfile.mkstemp(suffix='.pkl'); os.close(fd)
    try:
        w.save(path); c=soma.Formal066World.load(path); step(w,10); step(c,10); assert exact(w.state_dict(),c.state_dict())
    finally:
        try: os.remove(path)
        except OSError: pass
    return 'save/restore preserves ecology, genomes and RNG exactly'

def test_summary_exposes_ecology_metrics():
    s=soma.Formal066World(seed=15,initial_cells=3).summary()
    for k in ('eco66_max_generation','eco66_mean_active_modules','eco66_grammar_absent_frequency','eco66_external_fitness_events','eco66_grammar_hgt_acquisitions'): assert k in s
    return 'summary exposes generation, grammar and HGT metrics'

def test_absent_only_control_is_physically_absent_at_start():
    w=soma.Formal066World(seed=16,initial_cells=3,config=soma.Formal066Config(eco66_founder_mode='absent_only',eco66_founder_prime=False))
    assert all(soma.s65.grammar_traits_from_sequence(c.genomes[0])['grammar_gene_count']==0 for c in w.cells)
    return 'grammar-absent-only control starts with zero grammar genes in all founders'

def test_generation_two_short_positive_control():
    w=soma.Formal066World(seed=17,initial_cells=3,config=soma.Formal066Config(eco66_washout=False,eco66_replication_rate_scale=8.0,eco66_fuel_rate=0.32,eco66_mineral_rate=0.28))
    for _ in range(700): w.step(0.1)
    assert w.summary()['eco66_max_generation']>=2 and w.divisions>=4 and abs(w.matter_ledger_residual())<3e-5
    return 'endogenous parent→daughter→granddaughter cycle reaches generation 2'

def test_material_ledger_short_run():
    w=soma.Formal066World(seed=18,initial_cells=3); maxr=0.0
    for _ in range(80): w.step(0.1); maxr=max(maxr,abs(w.matter_ledger_residual()))
    assert maxr<3e-5; return 'short shared ecology remains inside material residual gate'

def test_finite_short_run():
    w=soma.Formal066World(seed=19,initial_cells=3); step(w,80); assert w.finite(); return 'all tracked ecology/neural/genetic values remain finite'

def test_compact_hud_and_descendant_reset_source_contract():
    src=open(os.path.join(HERE,'SOMA_CELL_0_6_6_pythonista.py'),encoding='utf-8').read()
    assert 's64.SomaCell064Scene.draw(self)' in src and 'rect(0.0,152.0,self.size.w,40.0)' in src and 'return Formal066World' in src
    return '0.6.6 replaces rather than stacks compact HUD and reset targets Formal066World'

def test_main_preregistration_exists_and_has_no_posthoc_seed_choice():
    path=os.path.abspath(os.path.join(HERE,'../../results/SOMA_CELL_0_6_6_PREREGISTRATION.json')); p=json.load(open(path,encoding='utf-8'))
    assert p['main_seeds']==[6601,6602,6603] and p['seconds']==80.0 and p['external_fitness_events_allowed']==0
    return 'main seeds, duration and zero-fitness rule are preregistered'

TESTS=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
def main():
    rows=[]
    for fn in TESTS:
        try: detail=fn(); rows.append((fn.__name__,'PASS',detail,'')); print('PASS',fn.__name__,detail,flush=True)
        except Exception as e: rows.append((fn.__name__,'FAIL','',repr(e))); traceback.print_exc(); print('FAIL',fn.__name__,repr(e),flush=True)
    with open(CSV_PATH,'w',newline='',encoding='utf-8') as f: csv.writer(f).writerows([('test','status','detail','error')]+rows)
    passed=sum(r[1]=='PASS' for r in rows)
    with open(TXT_PATH,'w',encoding='utf-8') as f:
        f.write('SOMA-CELL 0.6.6 VALIDATION RESULTS\n=====================================\n\n{} / {} PASS\n\n'.format(passed,len(rows)))
        for r in rows: f.write('{}: {} - {}{}\n'.format(r[1],r[0],r[2],(' | '+r[3]) if r[3] else ''))
    print('{} / {} PASS'.format(passed,len(rows))); return 0 if passed==len(rows) else 1
if __name__=='__main__': raise SystemExit(main())
