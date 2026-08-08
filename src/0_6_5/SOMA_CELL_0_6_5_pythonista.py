# coding: utf-8
"""SOMA-CELL 0.6.5 — multigenerational neural-grammar economy.

The added neural grammar is dormant by default after the negative 0.6.4
preparedness boundary.  This version places sentinel/readiness/organ/
prediction/plasticity control modules in the ordinary material genome and
allows whole-gene deletion, duplication, regulatory dormancy and reactivation.
An accelerated multigenerational lineage assay uses the same genome sequences
and actual SOMA-CELL body trials to measure environment-dependent retention or
loss.  No reward or correct action is exposed to an organism.
"""
from __future__ import division

import copy
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_4', '../0_6_3', '../0_6_2', '../0_6_1', '../0_6',
            '../0_6_p2', '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_4_pythonista as s64

s63=s64.s63; s62=s64.s62; s61=s64.s61; f06=s64.f06; p2=s64.p2; p1=s64.p1; p0=s64.p0
s5=s64.s5; s4=s64.s4; g2=p2.g2

BUILD='SOMA-CELL 0.6.5'
BUILD_LONG=BUILD+' | multigenerational neural-grammar economy'
SCHEMA_VERSION='0.6.5-GE1.0'
SAVE_VERSION=1
SAVE_FILE='soma_cell_0_6_5.pkl'
LOG_FILE='soma_cell_0_6_5_longrun.csv'
REPORT_FILE='soma_cell_0_6_5_report.txt'
SESSION_FILE='soma_cell_0_6_5_sessions.csv'
SIM_HZ=s64.SIM_HZ
clamp=s64.clamp
_atomic_pickle=s64._atomic_pickle

GRAMMAR_EFFECT=s4.EFFECT_RESERVED
GRAMMAR_CHANNEL=s4.CONTROL_RESERVED_7
GRAMMAR_LOCALISATION=s4.LOC_RESERVED
GRAMMAR_SENTINEL=0
GRAMMAR_READINESS=1
GRAMMAR_ORGAN=2
GRAMMAR_PREDICTION=3
GRAMMAR_PLASTICITY=4
GRAMMAR_NAMES=('sentinel','readiness','organ','prediction','plasticity')
GRAMMAR_COUNT=len(GRAMMAR_NAMES)
READINESS_LEVELS=(0.0,0.25,0.50,0.75,0.85,0.90,0.95,1.0)
# organ code: 0/1 absent, 2 conditional two-cell, >=3 constitutive two-cell.
ORGAN_NONE=0; ORGAN_CONDITIONAL=2; ORGAN_CONSTITUTIVE=3
DORMANT_PROMOTER_MAX=1
ACTIVE_PROMOTER_MIN=2
GRAMMAR_BOOTSTRAP_PROTEIN=0.00045

ENV_STABLE='stable'
ENV_PERIODIC='periodic'
ENV_RARE_FAULT='rare_fault'
ENV_LONG_DELAY='long_delay'
EVOLUTION_ENVIRONMENTS=(ENV_STABLE,ENV_PERIODIC,ENV_RARE_FAULT,ENV_LONG_DELAY)


def _seq_hash(seq):
    return hashlib.sha1(np.asarray(seq,dtype=np.uint8).tobytes()).hexdigest()[:12]


def make_grammar_gene(module, value=0, promoter=1, efficiency=4, fidelity=6):
    return g2.make_gene(
        g2.ROLE_REGULATOR, parameter=GRAMMAR_EFFECT, regulator=GRAMMAR_CHANNEL,
        promoter=int(promoter), efficiency=int(efficiency), fidelity=int(fidelity),
        localisation=GRAMMAR_LOCALISATION,
        spare=(int(module), int(value), 6, 1),
    )


def is_grammar_spec(spec):
    return bool(
        spec.get('role')==g2.ROLE_REGULATOR
        and int(spec.get('localisation',-1))==GRAMMAR_LOCALISATION
        and int(spec.get('parameter',-1))==GRAMMAR_EFFECT
        and int(spec.get('regulator',-1))==GRAMMAR_CHANNEL
        and 0 <= int(spec['payload'][8]) < GRAMMAR_COUNT
    )


def grammar_records_from_sequence(sequence):
    out=[]
    for rec in g2.parse_genes(np.asarray(sequence,dtype=np.uint8)):
        if not is_grammar_spec(rec):
            continue
        payload=tuple(int(v) for v in rec['payload'])
        out.append({
            'module':int(payload[8])%GRAMMAR_COUNT,
            'value':int(payload[9])%8,
            'promoter':int(payload[3]),
            'efficiency':int(payload[4]),
            'fidelity':int(payload[5]),
            'start':int(rec['start']),
            'fingerprint':int(rec['fingerprint']),
        })
    return out


def grammar_traits_from_sequence(sequence):
    records=grammar_records_from_sequence(sequence)
    best={}
    counts={i:0 for i in range(GRAMMAR_COUNT)}
    for rec in records:
        m=rec['module']; counts[m]+=1
        strength=(rec['promoter']+1)*(rec['efficiency']+1)
        old=best.get(m)
        if old is None or strength>(old[0]+1)*(old[1]+1):
            best[m]=(rec['promoter'],rec['efficiency'],rec['value'])
    def active(m):
        return m in best and best[m][0]>=ACTIVE_PROMOTER_MIN and best[m][1]>=2
    readiness_code=best.get(GRAMMAR_READINESS,(0,0,0))[2]
    organ_code=best.get(GRAMMAR_ORGAN,(0,0,0))[2]
    return {
        'sentinel':active(GRAMMAR_SENTINEL),
        'readiness':active(GRAMMAR_READINESS),
        'readiness_fraction':READINESS_LEVELS[int(readiness_code)%8] if active(GRAMMAR_READINESS) else 0.0,
        'organ':active(GRAMMAR_ORGAN),
        'organ_code':int(organ_code),
        'organ_mode':('none' if not active(GRAMMAR_ORGAN) or organ_code<2 else ('conditional' if organ_code==2 else 'constitutive')),
        'prediction':active(GRAMMAR_PREDICTION),
        'plasticity':active(GRAMMAR_PLASTICITY),
        'counts':tuple(int(counts[i]) for i in range(GRAMMAR_COUNT)),
        'active_count':int(sum(active(i) for i in range(GRAMMAR_COUNT))),
        'grammar_gene_count':int(len(records)),
        'grammar_symbols':int(len(records)*g2.GENE_SPAN),
    }


def grammar_signature(sequence):
    t=grammar_traits_from_sequence(sequence)
    return (t['sentinel'],t['readiness'],round(t['readiness_fraction'],2),t['organ_mode'],
            t['prediction'],t['plasticity'],t['grammar_gene_count'],len(sequence))


def install_grammar_cassette(cell, dormant=True, full=False, bootstrap=True):
    if not cell.genomes:
        return False
    existing=grammar_records_from_sequence(cell.genomes[0])
    present={r['module'] for r in existing}
    genes=[]
    for module in range(GRAMMAR_COUNT):
        if module in present:
            continue
        if full:
            promoter=5
        else:
            promoter=1 if dormant else 4
        value=0
        if module==GRAMMAR_READINESS: value=6  # 0.95 readiness option
        elif module==GRAMMAR_ORGAN: value=2   # conditional two-cell organ
        genes.append(make_grammar_gene(module,value=value,promoter=promoter))
    if not genes:
        return False
    total_symbols=sum(len(g) for g in genes)
    if len(cell.genomes[0])+total_symbols>g2.MAX_GENOME_LENGTH:
        raise ValueError('no physical room for 0.6.5 grammar cassette')
    cell.genomes[0]=np.concatenate([cell.genomes[0]]+genes).astype(np.uint8)
    # Founder construction explicitly accounts added genome mass; after this,
    # ordinary replication carries the sequence and pays normal copying cost.
    cell.pools[s5.POOL_NUCLEOTIDE]+=total_symbols*s5.MONOMER_MASS
    cell._refresh_gene_cache()
    if bootstrap:
        for fp,spec in cell.gene_specs.items():
            if is_grammar_spec(spec):
                cell.proteins[fp]=cell.proteins.get(fp,0.0)+GRAMMAR_BOOTSTRAP_PROTEIN
        cell._sync_protein_pool()
    return True



def remove_grammar_module(sequence,module):
    seq=np.asarray(sequence,dtype=np.uint8).copy()
    records=[r for r in grammar_records_from_sequence(seq) if r['module']==int(module)]
    for rec in sorted(records,key=lambda r:r['start'],reverse=True):
        a=rec['start']; seq=np.concatenate([seq[:a],seq[a+g2.GENE_SPAN:]]).astype(np.uint8)
    return seq

def set_grammar_module_promoter(sequence,module,promoter):
    seq=np.asarray(sequence,dtype=np.uint8).copy()
    for rec in grammar_records_from_sequence(seq):
        if rec['module']==int(module): seq[rec['start']+5]=np.uint8(int(promoter)%8)
    return seq

def knockout_grammar(sequence):
    seq=np.asarray(sequence,dtype=np.uint8).copy()
    records=sorted(grammar_records_from_sequence(seq),key=lambda r:r['start'],reverse=True)
    for rec in records:
        a=rec['start']; b=a+g2.GENE_SPAN
        seq=np.concatenate([seq[:a],seq[b:]]).astype(np.uint8)
    return seq


def reintroduce_full_grammar(sequence):
    seq=knockout_grammar(sequence)
    genes=[
        make_grammar_gene(GRAMMAR_SENTINEL,0,promoter=5),
        make_grammar_gene(GRAMMAR_READINESS,6,promoter=5),
        make_grammar_gene(GRAMMAR_ORGAN,2,promoter=5),
        make_grammar_gene(GRAMMAR_PREDICTION,0,promoter=5),
        make_grammar_gene(GRAMMAR_PLASTICITY,0,promoter=5),
    ]
    return np.concatenate([seq]+genes).astype(np.uint8)


def mutate_grammar_sequence(sequence,rng,rate=0.22,duplication=True):
    """Whole-gene material mutations restricted to the added grammar.

    Core metabolism is frozen so the assay asks whether this added organ
    grammar itself is retained, silenced, duplicated or lost.  Insertions and
    deletions change actual genome length; regulatory mutations alter promoter
    or readiness/organ payload symbols in-place.
    """
    seq=np.asarray(sequence,dtype=np.uint8).copy()
    events={'deletion':0,'duplication':0,'dormancy':0,'reactivation':0,'regulatory':0}
    if rng.random()>=max(0.0,float(rate)):
        return seq,events
    records=grammar_records_from_sequence(seq)
    if not records:
        # Rare reacquisition by reintroducing one physically encoded module.
        module=int(rng.integers(0,GRAMMAR_COUNT)); value=6 if module==GRAMMAR_READINESS else (2 if module==GRAMMAR_ORGAN else 0)
        gene=make_grammar_gene(module,value=value,promoter=int(rng.integers(1,5)))
        if len(seq)+len(gene)<=g2.MAX_GENOME_LENGTH:
            seq=np.concatenate([seq,gene]).astype(np.uint8); events['reactivation']+=1
        return seq,events
    op=int(rng.integers(0,5 if duplication else 4))
    rec=records[int(rng.integers(0,len(records)))]
    a=rec['start']; b=a+g2.GENE_SPAN
    if op==0 and len(records)>1:
        seq=np.concatenate([seq[:a],seq[b:]]).astype(np.uint8); events['deletion']+=1
    elif op==1 and duplication and len(seq)+g2.GENE_SPAN<=g2.MAX_GENOME_LENGTH:
        frag=seq[a:b].copy(); pos=int(rng.integers(0,len(seq)+1))
        seq=np.concatenate([seq[:pos],frag,seq[pos:]]).astype(np.uint8); events['duplication']+=1
    elif op==2:
        # payload promoter index 3 => start + 2 + 3
        idx=a+5
        old=int(seq[idx]); seq[idx]=np.uint8(int(rng.integers(0,2)))
        if old>DORMANT_PROMOTER_MAX: events['dormancy']+=1
        else: events['regulatory']+=1
    elif op==3:
        idx=a+5
        old=int(seq[idx]); seq[idx]=np.uint8(int(rng.integers(3,8)))
        if old<=DORMANT_PROMOTER_MAX: events['reactivation']+=1
        else: events['regulatory']+=1
    else:
        # mutate the module's value code, not its identity
        idx=a+2+9
        seq[idx]=np.uint8(int(rng.integers(0,8))); events['regulatory']+=1
    return seq,events


class Formal065Config(s64.Formal064Config):
    def __init__(self, grammar_install=True, grammar_initial='dormant', grammar_mutation_rate=0.22,
                 evolution_environment=ENV_STABLE, **kwargs):
        self.grammar_install=bool(grammar_install)
        self.grammar_initial=str(grammar_initial)
        self.grammar_mutation_rate=float(grammar_mutation_rate)
        self.evolution_environment=str(evolution_environment)
        if self.evolution_environment not in EVOLUTION_ENVIRONMENTS:
            raise ValueError('unknown 0.6.5 evolution environment')
        # Added grammar is dormant by default after the 0.6.4 negative gate.
        kwargs.setdefault('amortization_policy',s64.AMORT_BARE)
        super(Formal065Config,self).__init__(**kwargs)
    @classmethod
    def from_state(cls,state):
        state=dict(state or {})
        allowed=set(cls().__dict__.keys()); allowed.discard('audit_active_cells')
        state.pop('audit_active_cells',None)
        return cls(**{k:v for k,v in state.items() if k in allowed})


class Formal065ProtoCell(s64.s63.Formal063ProtoCell):
    @classmethod
    def from_state(cls,rng,state):
        cell=s64.s63.Formal063ProtoCell.from_state(rng,state); cell.__class__=cls; return cell


class Formal065World(s64.Formal064World):
    def __init__(self,seed=101,initial_cells=1,config=None):
        config=config if config is not None else Formal065Config()
        if not isinstance(config,Formal065Config): config=Formal065Config.from_state(config.state_dict())
        self.grammar_mutation_events={'deletion':0,'duplication':0,'dormancy':0,'reactivation':0,'regulatory':0}
        super(Formal065World,self).__init__(seed=seed,initial_cells=initial_cells,config=config)
        self.config=config
        for cell in self.cells:
            cell.__class__=Formal065ProtoCell
            if config.grammar_install:
                install_grammar_cassette(cell,dormant=config.grammar_initial!='full',full=config.grammar_initial=='full')
        self.initial_total_material=self.total_material(); self.last_step_material_residual=0.0

    def summary(self):
        out=super(Formal065World,self).summary(); living=self.living_cells()
        traits=[grammar_traits_from_sequence(c.genomes[0]) for c in living if c.genomes]
        out.update({
            'build':BUILD,'grammar_schema':SCHEMA_VERSION,
            'grammar_cells':len(traits),
            'grammar_mean_gene_count':float(np.mean([t['grammar_gene_count'] for t in traits])) if traits else 0.0,
            'grammar_sentinel_frequency':float(np.mean([t['sentinel'] for t in traits])) if traits else 0.0,
            'grammar_readiness_frequency':float(np.mean([t['readiness'] for t in traits])) if traits else 0.0,
            'grammar_organ_frequency':float(np.mean([t['organ'] for t in traits])) if traits else 0.0,
            'grammar_prediction_frequency':float(np.mean([t['prediction'] for t in traits])) if traits else 0.0,
            'grammar_plasticity_frequency':float(np.mean([t['plasticity'] for t in traits])) if traits else 0.0,
            'grammar_default_dormant':1,
        })
        return out

    def state_dict(self):
        st=super(Formal065World,self).state_dict(); st.update({'save_version':SAVE_VERSION,'build':BUILD,'config':self.config.state_dict(),'grammar_mutation_events':dict(self.grammar_mutation_events)}); return st

    @classmethod
    def from_state(cls,state):
        base=dict(state); base['save_version']=s64.SAVE_VERSION; base['build']=s64.BUILD
        allowed=set(s64.Formal064Config().__dict__.keys()); base['config']={k:v for k,v in dict(state.get('config',{})).items() if k in allowed}
        world=s64.Formal064World.from_state(base); world.__class__=cls; world.config=Formal065Config.from_state(state.get('config',{})); world.grammar_mutation_events=dict(state.get('grammar_mutation_events',{}))
        for c in world.cells: c.__class__=Formal065ProtoCell
        return world
    def clone(self): return Formal065World.from_state(self.state_dict())


def _config_for_traits(traits,environment):
    common=dict(grammar_install=False,grammar_initial='dormant',evolution_environment=environment,
                p2_install_genes=True,p2_bootstrap_neural_protein=True,external_test_harness=True,
                task_label='065-'+environment)
    if environment==ENV_STABLE:
        common.update(rule_mode=s64.RULE_SINGLE,rule_first_change_age=1e9,p2_environment=p2.P2_ENV_MOVING_PATCH)
    elif environment==ENV_PERIODIC:
        common.update(rule_mode=s64.RULE_PERIODIC,rule_first_change_age=2.0,rule_change_period=3.0,p2_environment=p2.P2_ENV_CUE_REVERSAL,p2_cue_delay=1.5)
    elif environment==ENV_LONG_DELAY:
        common.update(rule_mode=s64.RULE_PERIODIC,rule_first_change_age=2.0,rule_change_period=5.0,p2_environment=p2.P2_ENV_CUE_REVERSAL,p2_cue_delay=3.0,p2_cue_duration=1.5)
    else:
        common.update(rule_mode=s64.RULE_SINGLE,rule_first_change_age=1e9,p2_environment=p2.P2_ENV_MOVING_PATCH,mechanism_fault_enabled=True,mechanism_fault_age=2.5,mechanism_fault_gain_scale=0.18)
    if not traits['sentinel']:
        common['amortization_policy']=s64.AMORT_BARE
    elif not traits['organ']:
        common['amortization_policy']=s64.AMORT_OPTION_NONE
        common['preparedness_option_fraction']=traits['readiness_fraction'] if traits['readiness'] else 0.0
    elif traits['organ_mode']=='constitutive':
        common['amortization_policy']=s64.AMORT_ALWAYS_E2
    else:
        common['amortization_policy']=s64.AMORT_CONDITIONAL
        common['preparedness_option_fraction']=traits['readiness_fraction'] if traits['readiness'] else 0.0
    common['p2_prediction']=bool(traits['prediction'])
    common['p2_plasticity']=bool(traits['plasticity'])
    common['p2_recurrence']=False
    return Formal065Config(**common)


def evaluate_grammar_genome(sequence,environment=ENV_STABLE,seed=101,seconds=18.0):
    seq=np.asarray(sequence,dtype=np.uint8).copy(); traits=grammar_traits_from_sequence(seq)
    cfg=_config_for_traits(traits,environment)
    world=Formal065World(seed=int(seed),initial_cells=1,config=cfg)
    cell=world.living_cells()[0]
    old_len=len(cell.genomes[0]); new_len=len(seq); delta=(new_len-old_len)*s5.MONOMER_MASS
    if delta>0.0:
        if cell.pools[s5.POOL_NUCLEOTIDE] < delta:
            return {'finite':True,'margin_auc':-1e6,'uptake_delta':0.0,'traits':traits,'material_residual':0.0,'invalid_material':1}
        cell.pools[s5.POOL_NUCLEOTIDE]-=delta
    else:
        cell.pools[s5.POOL_NUCLEOTIDE]+=-delta
    cell.genomes[0]=seq.copy(); cell._refresh_gene_cache(); cell._sync_protein_pool()
    world.initial_total_material=world.total_material(); world.last_step_material_residual=0.0
    dt=1.0/SIM_HZ; auc=0.0; uptake0=float(world.p2_reward_uptake_total)
    for _ in range(int(round(float(seconds)*SIM_HZ))):
        if not world.living_cells(): break
        world.step(dt); alive=world.living_cells(); auc+=(float(np.mean([c.autopoietic_margin() for c in alive])) if alive else -1.0)*dt
    return {'finite':bool(world.finite()),'margin_auc':float(auc),'uptake_delta':float(world.p2_reward_uptake_total-uptake0),'traits':traits,'material_residual':float(world.matter_ledger_residual()),'invalid_material':0}


def _founder_sequence(full=True):
    world=Formal065World(seed=17,initial_cells=1,config=Formal065Config(grammar_initial='full' if full else 'dormant',external_test_harness=True))
    return world.living_cells()[0].genomes[0].copy()


def run_grammar_evolution_assay(seed=101,environment=ENV_STABLE,generations=12,population=24,mutation=True,seconds=16.0,standing_variation=True):
    rng=np.random.default_rng(int(seed)); founder=_founder_sequence(full=True)
    if standing_variation:
        variants=[founder.copy(),_founder_sequence(full=False),knockout_grammar(founder)]
        for module in range(GRAMMAR_COUNT): variants.append(remove_grammar_module(founder,module))
        genomes=[variants[i%len(variants)].copy() for i in range(int(population))]
        rng.shuffle(genomes)
    else:
        genomes=[founder.copy() for _ in range(int(population))]
    cache={}; history=[]; event_totals={'deletion':0,'duplication':0,'dormancy':0,'reactivation':0,'regulatory':0}
    evaluation_seeds=(int(seed)*10+1,)
    def score(seq):
        sig=grammar_signature(seq)
        key=(sig,environment,evaluation_seeds)
        if key not in cache:
            rows=[evaluate_grammar_genome(seq,environment=environment,seed=s,seconds=seconds) for s in evaluation_seeds]
            cache[key]={'margin_auc':float(np.mean([r['margin_auc'] for r in rows])),'rows':rows}
        return float(cache[key]['margin_auc'])
    for gen in range(int(generations)+1):
        scores=np.asarray([score(g) for g in genomes],dtype=float); traits=[grammar_traits_from_sequence(g) for g in genomes]
        history.append({
            'generation':gen,'mean_score':float(np.mean(scores)),'max_score':float(np.max(scores)),
            'mean_grammar_genes':float(np.mean([t['grammar_gene_count'] for t in traits])),
            'sentinel_frequency':float(np.mean([t['sentinel'] for t in traits])),
            'readiness_frequency':float(np.mean([t['readiness'] for t in traits])),
            'organ_frequency':float(np.mean([t['organ'] for t in traits])),
            'prediction_frequency':float(np.mean([t['prediction'] for t in traits])),
            'plasticity_frequency':float(np.mean([t['plasticity'] for t in traits])),
            'mean_length':float(np.mean([len(g) for g in genomes])),
            'genome_types':len({_seq_hash(g) for g in genomes}),
        })
        if gen>=generations: break
        offspring=[]
        while len(offspring)<population:
            contestants=rng.integers(0,population,4); pi=int(contestants[np.argmax(scores[contestants])]); child=genomes[pi].copy()
            if mutation:
                child,ev=mutate_grammar_sequence(child,rng,rate=0.34,duplication=True)
                for k,v in ev.items(): event_totals[k]+=int(v)
            offspring.append(child)
        genomes=offspring
    final_scores=np.asarray([score(g) for g in genomes],dtype=float); bi=int(np.argmax(final_scores)); best=genomes[bi].copy(); best_traits=grammar_traits_from_sequence(best)
    return {'seed':int(seed),'environment':environment,'generations':int(generations),'population':int(population),'mutation':bool(mutation),'standing_variation':bool(standing_variation),'history':history,'best_genome':best,'best_hash':_seq_hash(best),'best_traits':best_traits,'best_score':float(final_scores[bi]),'final_mean_score':float(np.mean(final_scores)),'event_totals':event_totals,'cache_size':len(cache),'evaluation_seeds':evaluation_seeds}


def causal_test_genome(sequence,environment,seeds=(701,702,703,704,705,706),seconds=20.0):
    original=np.asarray(sequence,dtype=np.uint8); ko=knockout_grammar(original); reintro=reintroduce_full_grammar(ko)
    rows=[]
    for seed in seeds:
        a=evaluate_grammar_genome(original,environment,seed,seconds); k=evaluate_grammar_genome(ko,environment,seed,seconds); r=evaluate_grammar_genome(reintro,environment,seed,seconds)
        rows.append({'seed':int(seed),'original':a['margin_auc'],'knockout':k['margin_auc'],'reintroduced':r['margin_auc'],'ko_diff':k['margin_auc']-a['margin_auc'],'reintro_diff':r['margin_auc']-k['margin_auc']})
    return rows


def run_headless_trial(seed=101,seconds=60.0,initial_cells=1,config=None):
    world=Formal065World(seed=seed,initial_cells=initial_cells,config=config if config is not None else Formal065Config())
    dt=1.0/SIM_HZ; auc=0.0
    for _ in range(int(round(float(seconds)*SIM_HZ))):
        if not world.living_cells(): break
        world.step(dt); alive=world.living_cells(); auc+=(float(np.mean([c.autopoietic_margin() for c in alive])) if alive else -1.0)*dt
    out=world.summary(); out.update({'seed':int(seed),'seconds':float(seconds),'margin_auc':float(auc),'finite':bool(world.finite()),'material_residual':float(world.matter_ledger_residual())}); return out


LOG_FIELDS=tuple(list(s64.LOG_FIELDS)+['grammar_schema','grammar_cells','grammar_mean_gene_count','grammar_sentinel_frequency','grammar_readiness_frequency','grammar_organ_frequency','grammar_prediction_frequency','grammar_plasticity_frequency','grammar_default_dormant'])

class LongRunLogger(s64.LongRunLogger):
    def __init__(self,world,path=LOG_FILE,interval=10.0): super(LongRunLogger,self).__init__(world,path=path,interval=interval); self.fields=LOG_FIELDS


def generate_report(log_path=LOG_FILE,report_path=REPORT_FILE,session_path=SESSION_FILE):
    return s64.generate_report(log_path=log_path,report_path=report_path,session_path=session_path)

try:
    from scene import Scene,run,LANDSCAPE,background,fill,rect,text,ellipse,line,stroke,stroke_weight
    class SomaCell065Scene(s64.SomaCell064Scene):
        def _fresh_world(self): return Formal065World(seed=101,initial_cells=2,config=Formal065Config())
        def setup(self):
            background(0.006,0.012,0.022)
            try: self.world=Formal065World.load(SAVE_FILE); self.save_status='LOAD'
            except Exception: self.world=self._fresh_world(); self.save_status='NEW'
            self.accumulator=0.0; self.last_wall=time.time(); self.last_save_age=self.world.age; self.last_touch_wall=-10.0; self.paused=False; self.fps=0.0; self.sim_rate=0.0; self.telemetry_wall=time.time(); self.telemetry_age=self.world.age; self.telemetry_frames=0; self.logger=LongRunLogger(self.world); self.logger.log(self.world,reason='start',force=True); self.report_status='WAIT'
        def draw(self):
            # Reuse compact 0.6.4 HUD, then add one unobtrusive grammar line.
            super(SomaCell065Scene,self).draw(); s=self.world.summary(); fill(0.70,0.92,1.0)
            text('0.6.5 grammar dormant | genes {:.1f} S/R/O/P/L {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(s.get('grammar_mean_gene_count',0.0),s.get('grammar_sentinel_frequency',0.0),s.get('grammar_readiness_frequency',0.0),s.get('grammar_organ_frequency',0.0),s.get('grammar_prediction_frequency',0.0),s.get('grammar_plasticity_frequency',0.0)),x=18,y=166,font_size=8,alignment=4)
        def touch_began(self,touch):
            now=time.time()
            if now-self.last_touch_wall<0.42:
                try:
                    if os.path.exists(SAVE_FILE): os.remove(SAVE_FILE)
                except Exception: pass
                self.world=self._fresh_world(); self.logger=LongRunLogger(self.world); self.logger.log(self.world,reason='reset',force=True); self.report_status='WAIT'; self.save_status='NEW'; self.accumulator=0.0; self.paused=False; self.last_save_age=self.world.age; self.last_touch_wall=-10.0; gc.collect(); return
            self.last_touch_wall=now
            if touch.location.y>self.size.h-60:
                self.paused=not self.paused; self.logger.log(self.world,reason='pause' if self.paused else 'resume',force=True); return
            position=self._unit_position(touch.location)
            if not self.world.puncture_nearest(position): self.world.inject_cloud(position)
        def stop(self):
            try: self.world.save(SAVE_FILE); self.save_status='OK'
            except Exception: self.save_status='ERR'
            self.logger.log(self.world,reason='stop',force=True); self.report_status=generate_report()
except ImportError:
    Scene=None

if __name__=='__main__':
    if Scene is None: print(run_headless_trial(seed=101,seconds=12.0))
    else: run(SomaCell065Scene(),orientation=LANDSCAPE,show_fps=False,multi_touch=False)
