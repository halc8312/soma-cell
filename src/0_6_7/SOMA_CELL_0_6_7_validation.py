# coding: utf-8
from __future__ import division
import csv, hashlib, json, os, tempfile, traceback
import numpy as np
import SOMA_CELL_0_6_7_pythonista as soma

HERE=os.path.dirname(os.path.abspath(__file__))
CSV_PATH=os.path.join(HERE,'soma_cell_0_6_7_validation.csv')
TXT_PATH=os.path.join(HERE,'SOMA_CELL_0_6_7_VALIDATION_RESULTS.txt')


def result_path(name):
    flat=os.path.join(HERE,name)
    if os.path.exists(flat): return flat
    return os.path.abspath(os.path.join(HERE,'../../results',name))


def exact(a,b):
    if isinstance(a,dict): return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray): return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,float): return a==b
    return a==b


def step(world,n=10,dt=None):
    dt=1.0/soma.SIM_HZ if dt is None else float(dt)
    for _ in range(int(n)): world.step(dt)


def reset_ledger(world):
    world.dissipated_energy=0.0; world.exported_material=0.0; world.injected_material=0.0
    world.initial_total_material=world.current_material(); world.last_material_residual=0.0


def test_build_and_schema():
    assert soma.BUILD=='SOMA-CELL 0.6.7' and soma.SCHEMA_VERSION=='0.6.7-LH1.0'
    return 'build/schema frozen'


def test_external_fitness_is_rejected():
    try: soma.Formal067Config(external_fitness_selection=True)
    except ValueError: pass
    else: raise AssertionError('external fitness must fail closed')
    cfg=soma.Formal067Config().__dict__
    assert 'fitness' not in cfg and 'correct_direction' not in cfg
    return 'organism-facing fitness/correct-direction channels are absent'


def test_founders_are_material_full_dormant_absent_states():
    w=soma.Formal067World(seed=1,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    cs=sorted(w.cells,key=lambda c:c.lineage); raw=[c.traits_raw() for c in cs]
    assert raw[0]['active_count']==5 and raw[1]['active_count']==0 and raw[1]['grammar_gene_count']==5 and raw[2]['grammar_gene_count']==0
    assert cs[0].module_mass()>0 and cs[1].module_mass()==0 and cs[2].module_mass()==0
    assert abs(w.matter_ledger_residual())<1e-12
    return 'founder grammar and pretranslated protein are explicit initial material'


def test_descendants_must_translate_module_protein():
    w=soma.Formal067World(seed=2,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    p=sorted(w.cells,key=lambda c:c.lineage)[0]
    p.copy_symbols=len(p.genome);p.copy_complete=True;p.age=10.0
    p.pools[soma.P_STRUCTURE]=2.0;p.pools[soma.P_MEMBRANE]=1.0;p.pools[soma.P_ATP]=1.0
    reset_ledger(w); ds=w._divide(p); w.cells=[c for c in w.cells if c.alive]+ds
    d=ds[0]
    assert d.module_mass()==0.0 and not d.expressed_modules
    atp0=float(d.pools[soma.P_ATP]); structure0=float(d.pools[soma.P_STRUCTURE])
    w._module_expression(d,1.2)
    assert d.module_mass()>0 and d.pools[soma.P_ATP]<atp0 and d.pools[soma.P_STRUCTURE]<structure0
    assert d.traits()['active_count']>0
    assert abs(w.matter_ledger_residual())<1e-10
    return 'daughter function appears only after ATP/material-paid translation'


def test_long_delay_cue_is_accounted_material():
    w=soma.Formal067World(seed=3,config=soma.Formal067Config(ecology_environment=soma.ENV_LONG_DELAY,initial_per_founder=1,mutation=False,hgt=False))
    cue0=float(np.sum(w.patch_cue)); w._inject_resources(0.25)
    assert float(np.sum(w.patch_cue))>cue0 and w.injected_material>0
    assert abs(w.matter_ledger_residual())<1e-10
    return 'advance cue enters as extracellular material and is in the ledger'


def test_no_privileged_future_label_in_behavior_source():
    src=open(os.path.join(HERE,'SOMA_CELL_0_6_7_pythonista.py'),encoding='utf-8').read()
    assert 'desired=self.cue_niche()' not in src
    assert 'np.argmax(self.patch_cue)' in src and 'correct_direction' not in src
    return 'cells sense a physical cue reservoir rather than a future-state label'


def test_metabolism_conserves_material_and_dissipates_atp():
    w=soma.Formal067World(seed=4,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    c=sorted(w.cells,key=lambda x:x.lineage)[2]
    c.pools=np.array([1.0,1.0,1.0,0.3,1.0,0.5,0.0,0.0],dtype=float); reset_ledger(w)
    before=w.accounted_total(); w._metabolize(c,0.5); after=w.accounted_total()
    assert w.dissipated_energy>0 and abs(after-before)<1e-10 and abs(w.matter_ledger_residual())<1e-10
    return 'coarse metabolism is stoichiometric and ATP spending enters dissipation'


def test_replication_consumes_nucleotide_and_atp():
    w=soma.Formal067World(seed=5,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    c=sorted(w.cells,key=lambda x:x.lineage)[2]
    c.copy_symbols=0;c.copy_complete=False;c.pools[soma.P_NUCLEOTIDE]=1.0;c.pools[soma.P_ATP]=1.0;reset_ledger(w)
    n0=float(c.pools[soma.P_NUCLEOTIDE]);a0=float(c.pools[soma.P_ATP]);w._replicate(c,0.25)
    assert c.copy_symbols>0 and c.pools[soma.P_NUCLEOTIDE]<n0 and c.pools[soma.P_ATP]<a0
    assert c.copy_mass()>0 and abs(w.matter_ledger_residual())<1e-10
    return 'genome copy polymer is built from explicit nucleotide and ATP'


def test_division_is_material_and_resets_learned_state():
    w=soma.Formal067World(seed=6,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    p=sorted(w.cells,key=lambda x:x.lineage)[0]
    p.copy_symbols=len(p.genome);p.copy_complete=True;p.age=9.0;p.preference=0.91;p.uptake_ema=0.77
    p.pools[soma.P_STRUCTURE]=2.0;p.pools[soma.P_MEMBRANE]=1.0;p.pools[soma.P_ATP]=1.0;reset_ledger(w)
    before=w.accounted_total();ds=w._divide(p);w.cells=[c for c in w.cells if c.alive]+ds;after=w.accounted_total()
    assert len(ds)==2 and all(d.generation==1 and d.copy_symbols==0 and not d.copy_complete for d in ds)
    assert all(d.uptake_ema==0.0 and d.module_mass()==0.0 for d in ds)
    assert abs(after-before)<1e-10 and abs(w.matter_ledger_residual())<1e-10
    return 'one physical genome copy goes to each fresh daughter without learned-state copying'


def _patch_mutator(fn):
    old=soma.s65.mutate_grammar_sequence; soma.s65.mutate_grammar_sequence=fn; return old


def test_duplication_requires_polymer_and_atp():
    w=soma.Formal067World(seed=7,config=soma.Formal067Config(initial_per_founder=1,mutation=True,hgt=False,mutation_rate=1.0))
    c=sorted(w.cells,key=lambda x:x.lineage)[0]
    rec=soma.s65.grammar_records_from_sequence(c.genome)[0];frag=c.genome[rec['start']:rec['start']+soma.g2.GENE_SPAN].copy()
    def fake(seq,rng,rate=1.0,duplication=True): return np.concatenate([seq,frag]).astype(np.uint8),{'deletion':0,'duplication':1,'dormancy':0,'reactivation':0,'regulatory':0}
    c.pools[soma.P_NUCLEOTIDE]=1.0;c.pools[soma.P_ATP]=1.0;reset_ledger(w);n0=len(c.genome);nu0=float(c.pools[soma.P_NUCLEOTIDE]);a0=float(c.pools[soma.P_ATP])
    old=_patch_mutator(fake)
    try:w._material_mutate(c)
    finally:soma.s65.mutate_grammar_sequence=old
    assert len(c.genome)==n0+len(frag) and c.pools[soma.P_NUCLEOTIDE]<nu0 and c.pools[soma.P_ATP]<a0
    assert abs(w.matter_ledger_residual())<1e-10
    return 'whole-gene duplication pays nucleotide mass and ATP'


def test_deletion_returns_nucleotide_material():
    w=soma.Formal067World(seed=8,config=soma.Formal067Config(initial_per_founder=1,mutation=True,hgt=False,mutation_rate=1.0))
    c=sorted(w.cells,key=lambda x:x.lineage)[0]
    rec=soma.s65.grammar_records_from_sequence(c.genome)[0];a=rec['start'];b=a+soma.g2.GENE_SPAN
    def fake(seq,rng,rate=1.0,duplication=True): return np.concatenate([seq[:a],seq[b:]]).astype(np.uint8),{'deletion':1,'duplication':0,'dormancy':0,'reactivation':0,'regulatory':0}
    reset_ledger(w);n0=len(c.genome);nu0=float(c.pools[soma.P_NUCLEOTIDE])
    old=_patch_mutator(fake)
    try:w._material_mutate(c)
    finally:soma.s65.mutate_grammar_sequence=old
    assert len(c.genome)==n0-soma.g2.GENE_SPAN
    assert abs((c.pools[soma.P_NUCLEOTIDE]-nu0)-soma.g2.GENE_SPAN*soma.MONOMER_MASS)<1e-12
    assert abs(w.matter_ledger_residual())<1e-10
    return 'whole-gene deletion returns polymer matter to nucleotide pool'


def test_mutation_reentry_is_recorded_as_sequence_event():
    w=soma.Formal067World(seed=9,config=soma.Formal067Config(initial_per_founder=1,mutation=True,hgt=False,mutation_rate=1.0))
    c=sorted(w.cells,key=lambda x:x.lineage)[2]
    gene=soma.s65.make_grammar_gene(soma.s65.GRAMMAR_SENTINEL,0,promoter=5)
    def fake(seq,rng,rate=1.0,duplication=True): return np.concatenate([seq,gene]).astype(np.uint8),{'deletion':0,'duplication':0,'dormancy':0,'reactivation':1,'regulatory':0}
    c.pools[soma.P_NUCLEOTIDE]=1.0;c.pools[soma.P_ATP]=1.0;reset_ledger(w)
    old=_patch_mutator(fake)
    try:w._material_mutate(c)
    finally:soma.s65.mutate_grammar_sequence=old
    assert w.mutation_reentries==1 and w.grammar_reentries==1 and c.reentry_lineage and not c.grammar_absent()
    assert c.traits()['active_count']==0
    return 'mutation re-entry changes real sequence but still requires later translation'


def test_hgt_is_material_and_translation_gated():
    w=soma.Formal067World(seed=10,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=True,hgt_probability=100.0))
    absent=sorted(w.cells,key=lambda x:x.lineage)[2];full=sorted(w.cells,key=lambda x:x.lineage)[0]
    rec=soma.s65.grammar_records_from_sequence(full.genome)[0];frag=full.genome[rec['start']:rec['start']+soma.g2.GENE_SPAN].copy()
    w.cells=[absent];absent.niche=0;absent.pools[soma.P_ATP]=1.0;absent.pools[soma.P_STRUCTURE]=1.0
    w.edna=[soma.EDNAFragment(frag,0,full.lineage)];reset_ledger(w);n0=len(absent.genome)
    w._hgt(0.25)
    assert len(absent.genome)==n0+len(frag) and w.hgt_reentries==1 and not absent.grammar_absent()
    assert absent.traits()['active_count']==0
    w._module_expression(absent,1.2)
    assert absent.traits()['active_count']>=1 and abs(w.matter_ledger_residual())<1e-10
    return 'eDNA mass enters genome; function appears only after paid translation'


def test_hgt_off_blocks_integration():
    w=soma.Formal067World(seed=11,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False,hgt_probability=100.0))
    c=sorted(w.cells,key=lambda x:x.lineage)[2];gene=soma.s65.make_grammar_gene(soma.s65.GRAMMAR_SENTINEL,0,promoter=5)
    w.cells=[c];w.edna=[soma.EDNAFragment(gene,c.niche,0)];n0=len(c.genome);w._hgt(1.0)
    assert len(c.genome)==n0 and w.hgt_integrations==0 and len(w.edna)==1
    return 'HGT-OFF disables material integration path'


def test_edna_decay_returns_polymer_to_mineral():
    w=soma.Formal067World(seed=12,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False,edna_decay=100.0))
    gene=soma.s65.make_grammar_gene(soma.s65.GRAMMAR_SENTINEL,0,promoter=5);w.edna=[soma.EDNAFragment(gene,0,0)];reset_ledger(w);m0=float(w.patch_mineral[0]);w._decay_edna(0.25)
    assert not w.edna and abs((w.patch_mineral[0]-m0)-len(gene)*soma.MONOMER_MASS)<1e-12
    assert abs(w.matter_ledger_residual())<1e-10
    return 'decayed environmental DNA returns its polymer mass to mineral pool'


def test_death_returns_module_and_genome_material():
    w=soma.Formal067World(seed=13,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    c=sorted(w.cells,key=lambda x:x.lineage)[0];reset_ledger(w);before=w.accounted_total();w._die(c,'test');after=w.accounted_total()
    assert not c.alive and len(w.edna)>=1 and abs(after-before)<1e-10 and abs(w.matter_ledger_residual())<1e-10
    return 'death conserves body/module matter and releases grammar as eDNA'


def test_neutral_transfer_selection_is_genotype_blind():
    cfg=soma.Formal067Config(initial_per_founder=2,mutation=False,hgt=False,transfer_start=0.0,transfer_target=3,transfer_interval=100.0)
    a=soma.Formal067World(seed=14,config=cfg);b=a.clone()
    # Swap grammar genomes/labels while preserving cell IDs and RNG state.
    genomes=[c.genome.copy() for c in b.cells][::-1];labels=[c.founder_class for c in b.cells][::-1]
    for c,g,label in zip(b.cells,genomes,labels): c.genome=g;c.founder_class=label;c._refresh_expression(initial=False)
    a.age=b.age=0.0;a.next_transfer=b.next_transfer=0.0
    a._neutral_transfer();b._neutral_transfer()
    assert sorted(c.cell_id for c in a.living_cells())==sorted(c.cell_id for c in b.living_cells())
    return 'neutral transfer keeps identical IDs after genotype/label permutation'


def test_neutral_transfer_is_material_accounted():
    w=soma.Formal067World(seed=15,config=soma.Formal067Config(initial_per_founder=2,mutation=False,hgt=False,transfer_start=0.0,transfer_target=3,transfer_interval=100.0))
    reset_ledger(w);w.age=0.0;w.next_transfer=0.0;w._neutral_transfer()
    assert w.transfer_events==1 and len(w.living_cells())==3 and abs(w.matter_ledger_residual())<1e-10
    return 'genotype-blind bottleneck is lysis/outflow inside the material ledger'


def test_generation_fifteen_long_horizon_positive_control():
    r=soma.run_long_horizon_assay(seed=6703,environment=soma.ENV_STABLE,seconds=360.0,mutation=True,hgt=False)
    assert r['max_generation']>=15 and r['grammar_absent_frequency']>=0.70 and r['max_abs_material_residual']<1e-7
    return 'development control reaches actual generation 15 with stable grammar loss signal'


def test_clone_is_deterministic():
    w=soma.Formal067World(seed=16,config=soma.Formal067Config(initial_per_founder=1));step(w,30);c=w.clone();step(w,24);step(c,24)
    assert exact(w.state_dict(),c.state_dict())
    return 'clone preserves ecology, genomes, translated material and RNG exactly'


def test_save_restore_is_deterministic():
    w=soma.Formal067World(seed=17,config=soma.Formal067Config(initial_per_founder=1));step(w,24)
    fd,path=tempfile.mkstemp(suffix='.pkl');os.close(fd)
    try:
        w.save(path);c=soma.Formal067World.load(path);step(w,20);step(c,20);assert exact(w.state_dict(),c.state_dict())
    finally:
        try:os.remove(path)
        except OSError:pass
    return 'save/restore preserves long-horizon ecology and RNG exactly'


def test_summary_exposes_loss_reentry_and_material_metrics():
    s=soma.Formal067World(seed=18,config=soma.Formal067Config(initial_per_founder=1)).summary()
    for key in ('max_generation','mean_active_modules','mean_expressed_modules','grammar_absent_frequency','mutation_reentries','hgt_reentries','persistent_reentry_grammar_cells','module_protein_material','matter_residual','external_fitness_events'):
        assert key in s
    return 'summary exposes long-horizon grammar, re-entry, protein and ledger metrics'


def test_short_run_is_finite_and_material_conservative():
    w=soma.Formal067World(seed=19,config=soma.Formal067Config(initial_per_founder=1));maxr=0.0
    for _ in range(160):w.step(1.0/soma.SIM_HZ);maxr=max(maxr,abs(w.matter_ledger_residual()))
    assert w.finite() and maxr<1e-7 and w.external_fitness_events==0
    return 'short serial-transfer ecology stays finite, material-conservative and fitness-free'


def test_scene_contract_is_compact_and_resets_descendant_world():
    src=open(os.path.join(HERE,'SOMA_CELL_0_6_7_pythonista.py'),encoding='utf-8').read()
    assert "return LongHorizon067World" in src and "rect(0,h-top,w,top);rect(0,0,w,bottom)" in src
    assert "self.world=self._fresh_world()" in src
    return 'compact HUD is not stacked and double-tap reset targets LongHorizon067World'


def test_preregistration_matches_frozen_source_and_holdout_seeds():
    path=result_path('SOMA_CELL_0_6_7_R3_PREREGISTRATION.json');p=json.load(open(path,encoding='utf-8'))
    source=open(os.path.join(HERE,'SOMA_CELL_0_6_7_pythonista.py'),'rb').read();sha=hashlib.sha256(source).hexdigest()
    assert p['source_sha256']==sha and p['holdout_seeds']==[6731,6732,6733] and p['duration_seconds']==480.0 and p['registration']=='R3'
    return 'R3 canonical source hash, unseen holdout seeds and duration were frozen before main assay'


def test_prior_registrations_are_explicitly_invalidated():
    r1=result_path('SOMA_CELL_0_6_7_R1_PREREGISTRATION_INVALIDATION_20260808.txt')
    r2=result_path('SOMA_CELL_0_6_7_R2_PREREGISTRATION_INVALIDATION_20260808.txt')
    assert os.path.exists(r1) and os.path.exists(r2)
    text=open(r2,encoding='utf-8').read()
    assert 'not eligible as formal holdouts' in text and 'R3 uses unseen seeds 6731-6733' in text
    return 'R1/R2/intermediate registrations remain development-only and are not promoted post hoc'


def test_module_material_is_in_finite_state_audit():
    w=soma.Formal067World(seed=20,config=soma.Formal067Config(initial_per_founder=1,mutation=False,hgt=False))
    c=w.living_cells()[0]
    c.module_material[soma.s65.GRAMMAR_NAMES[0]]=float('nan')
    assert not c.finite() and not w.finite()
    return 'non-finite translated neural-module protein is caught by cell and world audits'


def test_r3_registered_rows_and_gate_report_are_complete():
    result_path=os.path.join(HERE,'soma_cell_0_6_7_r3_experiment_results.json')
    gate_path=os.path.join(HERE,'soma_cell_0_6_7_r3_gate_report.json')
    rows=json.load(open(result_path,encoding='utf-8')); report=json.load(open(gate_path,encoding='utf-8'))
    expected={(c,s) for c in ('stable_hgt_on','stable_hgt_off','long_delay_hgt_on','long_delay_hgt_off') for s in (6731,6732,6733)}
    assert {(r['condition'],r['seed']) for r in rows}==expected and len(rows)==12
    assert report['gates']['all_main_rows_reported'] and not report['overall_primary_gate_pass']
    return 'all 12 R3 rows and every positive/negative gate are reported without seed filtering'


TESTS=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]

def main():
    rows=[]
    for fn in TESTS:
        try:
            detail=fn();rows.append((fn.__name__,'PASS',detail,''));print('PASS',fn.__name__,detail,flush=True)
        except Exception as exc:
            rows.append((fn.__name__,'FAIL','',repr(exc)));traceback.print_exc();print('FAIL',fn.__name__,repr(exc),flush=True)
    with open(CSV_PATH,'w',newline='',encoding='utf-8') as handle:
        csv.writer(handle).writerows([('test','status','detail','error')]+rows)
    passed=sum(r[1]=='PASS' for r in rows)
    with open(TXT_PATH,'w',encoding='utf-8') as handle:
        handle.write('SOMA-CELL 0.6.7 VALIDATION RESULTS\n=====================================\n\n{} / {} PASS\n\n'.format(passed,len(rows)))
        for row in rows:handle.write('{}: {} - {}{}\n'.format(row[1],row[0],row[2],(' | '+row[3]) if row[3] else ''))
    print('{} / {} PASS'.format(passed,len(rows)))
    return 0 if passed==len(rows) else 1

if __name__=='__main__':raise SystemExit(main())
