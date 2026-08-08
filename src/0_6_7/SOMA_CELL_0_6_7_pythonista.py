# coding: utf-8
"""SOMA-CELL 0.6.7 — long-horizon material serial-transfer ecology.

0.6.6 established endogenous birth/death in the full shared particle world but
only reached generation 2 in the registered short assays. 0.6.7 adds a clearly
labelled coarse-grained, material-conserving chemostat layer for generation-20+
questions. It does not replace 0.6.6 microphysics; it is calibrated as a second
research instrument whose units are explicit material pools, genomes, ATP,
corpses, eDNA, neutral outflow, and actual parent→daughter division events.

No cell receives fitness, reward, a correct direction, a switch-time flag, or an
experimenter-selected lineage copy. Population frequencies change only through
material uptake, DNA replication, physical division, death, neutral transfer,
mutation and material HGT.
"""
from __future__ import division

import csv
import gc
import hashlib
import json
import math
import os
import pickle
import sys
import time
import uuid

import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('.','../0_6_6','../0_6_5','../0_6_4','../0_6_3','../0_6_2','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
    path=os.path.abspath(os.path.join(HERE,rel))
    if path not in sys.path: sys.path.insert(0,path)

import SOMA_CELL_0_6_6_pythonista as s66
s65=s66.s65; s5=s66.s5; g2=s66.g2

BUILD='SOMA-CELL 0.6.7'
BUILD_LONG=BUILD+' | long-horizon material serial-transfer ecology'
SCHEMA_VERSION='0.6.7-LH1.0'
SAVE_VERSION=1
SAVE_FILE='soma_cell_0_6_7.pkl'
LOG_FILE='soma_cell_0_6_7_longrun.csv'
REPORT_FILE='soma_cell_0_6_7_report.txt'
SESSION_FILE='soma_cell_0_6_7_sessions.csv'
SIM_HZ=4.0

ENV_STABLE=s66.ENV_STABLE
ENV_PERIODIC=s66.ENV_PERIODIC
ENV_LONG_DELAY=s66.ENV_LONG_DELAY
ENVIRONMENTS=(ENV_STABLE,ENV_PERIODIC,ENV_LONG_DELAY)

FOUNDER_FULL=s66.FOUNDER_FULL
FOUNDER_DORMANT=s66.FOUNDER_DORMANT
FOUNDER_ABSENT=s66.FOUNDER_ABSENT
FOUNDER_NAMES=(FOUNDER_FULL,FOUNDER_DORMANT,FOUNDER_ABSENT)

# Material pools. Genome and partial-copy polymer are accounted separately.
P_FUEL=0
P_MINERAL=1
P_ATP=2
P_NUCLEOTIDE=3
P_STRUCTURE=4
P_MEMBRANE=5
P_WASTE=6
P_REACTIVE=7
POOL_COUNT=8
MONOMER_MASS=float(s5.MONOMER_MASS)

MODULE_COST={
    'sentinel':0.00026,
    'readiness':0.00018,
    'organ':0.00034,
    'prediction':0.00028,
    'plasticity':0.00028,
}
MODULE_WEAR=0.018
MODULE_PROTEIN_TARGET=0.00018
MODULE_TRANSLATION_ATP_PER_MATERIAL=1.80


def clamp(v,lo,hi):
    return lo if v<lo else hi if v>hi else v


def _atomic_pickle(path,payload):
    tmp=path+'.tmp'
    with open(tmp,'wb') as handle:
        pickle.dump(payload,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp,path)


def _seq_hash(seq):
    return hashlib.sha1(np.asarray(seq,dtype=np.uint8).tobytes()).hexdigest()[:12]


def active_module_names(traits):
    return tuple(name for name in s65.GRAMMAR_NAMES if bool(traits.get(name,False)))


def grammar_module_set(sequence):
    return frozenset(int(rec['module']) for rec in s65.grammar_records_from_sequence(sequence))


def shannon(values):
    values=np.asarray(values,dtype=float)
    total=float(np.sum(values))
    if total<=0.0:return 0.0
    p=values[values>0]/total
    return float(-np.sum(p*np.log(p)))


class Formal067Config(object):
    def __init__(self,
                 ecology_environment=ENV_STABLE,
                 mutation=True,
                 hgt=True,
                 mutation_rate=0.15,
                 initial_per_founder=2,
                 population_cap=24,
                 transfer_target=12,
                 transfer_interval=12.0,
                 transfer_start=24.0,
                 patch_capacity=7.0,
                 fuel_inflow=0.90,
                 mineral_inflow=0.72,
                 resource_outflow=0.10,
                 transfer_lysis_fraction=0.18,
                 rule_first_age=18.0,
                 rule_period=14.0,
                 cue_lead=3.5,
                 cue_inflow=0.12,
                 cue_capacity=0.60,
                 cue_decay=0.08,
                 division_min_age=3.0,
                 division_structure=1.05,
                 division_membrane=0.36,
                 division_atp=0.050,
                 replication_symbols_per_second=150.0,
                 replication_atp_per_symbol=0.000035,
                 division_atp_cost=0.018,
                 hgt_probability=0.020,
                 hgt_atp_cost=0.0045,
                 edna_decay=0.012,
                 max_age=120.0,
                 max_steps=None,
                 external_fitness_selection=False,
                 visual_seed=101):
        if ecology_environment not in ENVIRONMENTS: raise ValueError('unknown 0.6.7 environment')
        self.ecology_environment=str(ecology_environment)
        self.mutation=bool(mutation); self.hgt=bool(hgt)
        self.mutation_rate=float(mutation_rate)
        self.initial_per_founder=int(initial_per_founder)
        self.population_cap=int(population_cap)
        self.transfer_target=int(transfer_target)
        self.transfer_interval=float(transfer_interval)
        self.transfer_start=float(transfer_start)
        self.patch_capacity=float(patch_capacity)
        self.fuel_inflow=float(fuel_inflow)
        self.mineral_inflow=float(mineral_inflow)
        self.resource_outflow=float(resource_outflow)
        self.transfer_lysis_fraction=float(transfer_lysis_fraction)
        self.rule_first_age=float(rule_first_age)
        self.rule_period=float(rule_period)
        self.cue_lead=float(cue_lead)
        self.cue_inflow=float(cue_inflow)
        self.cue_capacity=float(cue_capacity)
        self.cue_decay=float(cue_decay)
        self.division_min_age=float(division_min_age)
        self.division_structure=float(division_structure)
        self.division_membrane=float(division_membrane)
        self.division_atp=float(division_atp)
        self.replication_symbols_per_second=float(replication_symbols_per_second)
        self.replication_atp_per_symbol=float(replication_atp_per_symbol)
        self.division_atp_cost=float(division_atp_cost)
        self.hgt_probability=float(hgt_probability)
        self.hgt_atp_cost=float(hgt_atp_cost)
        self.edna_decay=float(edna_decay)
        self.max_age=float(max_age)
        self.max_steps=None if max_steps is None else int(max_steps)
        self.external_fitness_selection=bool(external_fitness_selection)
        if self.external_fitness_selection:
            raise ValueError('organism-facing/external fitness selection is forbidden in 0.6.7')
        self.visual_seed=int(visual_seed)
    def state_dict(self): return dict(self.__dict__)
    @classmethod
    def from_state(cls,state): return cls(**dict(state or {}))


class LongHorizonCell(object):
    def __init__(self,cell_id,parent_id,lineage,founder_class,generation,genome,niche,pools=None,pretranslated=False):
        self.cell_id=int(cell_id); self.parent_id=int(parent_id); self.lineage=int(lineage)
        self.founder_class=str(founder_class); self.generation=int(generation)
        self.genome=np.asarray(genome,dtype=np.uint8).copy()
        self.niche=int(niche)%2
        if pools is None:
            pools=np.array([0.52,0.40,0.18,0.30,0.82,0.34,0.02,0.002],dtype=float)
        self.pools=np.asarray(pools,dtype=float).copy()
        self.age=0.0; self.damage=0.0; self.alive=True; self.death_reason=''
        self.copy_symbols=0; self.copy_complete=False
        self.expression_progress={name:0.0 for name in s65.GRAMMAR_NAMES}
        self.expressed_modules=set()
        self.module_material={name:0.0 for name in s65.GRAMMAR_NAMES}
        self.uptake_ema=0.0; self.previous_uptake=0.0; self.preference=float(self.niche)
        self.predicted_niche=int(self.niche); self.last_switch_age=-1e9
        self.grammar_atp=0.0; self.grammar_wear=0.0; self.movement_atp=0.0
        self.hgt_integrations=0; self.grammar_hgt_acquisitions=0; self.grammar_reentries=0
        self.mutation_reentries=0; self.hgt_reentries=0
        self.was_grammar_absent=(len(s65.grammar_records_from_sequence(self.genome))==0)
        self.reentry_origin_generation=-1
        self.reentry_lineage=False
        self.reentry_source=''
        self.mutation_events=0
        self.total_uptake=0.0
        self._refresh_expression(initial=bool(pretranslated))

    def traits_raw(self): return s65.grammar_traits_from_sequence(self.genome)
    def traits(self):
        raw=self.traits_raw(); out=dict(raw)
        for name in s65.GRAMMAR_NAMES:
            out[name]=bool(raw.get(name,False) and name in self.expressed_modules)
        out['active_count']=int(sum(out[name] for name in s65.GRAMMAR_NAMES))
        return out
    def _refresh_expression(self,initial=False):
        raw=self.traits_raw()
        for name in s65.GRAMMAR_NAMES:
            if not raw.get(name,False):
                # A deleted/dormant module cannot keep a massless functional flag.
                recycled=float(self.module_material.get(name,0.0))
                if recycled>0.0:
                    self.pools[P_STRUCTURE]+=recycled
                self.module_material[name]=0.0
                self.expression_progress[name]=0.0
                self.expressed_modules.discard(name)
            elif initial:
                # Founders may start pre-translated, but the protein mass is taken
                # from their explicit structural pool before the ledger baseline.
                need=max(0.0,MODULE_PROTEIN_TARGET-float(self.module_material.get(name,0.0)))
                made=min(need,float(self.pools[P_STRUCTURE]))
                self.pools[P_STRUCTURE]-=made
                self.module_material[name]=float(self.module_material.get(name,0.0))+made
                self.expression_progress[name]=clamp(self.module_material[name]/max(MODULE_PROTEIN_TARGET,1e-12),0.0,1.0)
                if self.expression_progress[name]>=1.0-1e-9:
                    self.expressed_modules.add(name)
    def genome_mass(self): return len(self.genome)*MONOMER_MASS
    def copy_mass(self): return self.copy_symbols*MONOMER_MASS
    def module_mass(self): return float(sum(float(v) for v in self.module_material.values()))
    def material_mass(self): return float(np.sum(self.pools))+self.module_mass()+self.genome_mass()+self.copy_mass()
    def grammar_absent(self): return len(s65.grammar_records_from_sequence(self.genome))==0
    def signature(self): return s65.grammar_signature(self.genome)
    def finite(self):
        module_values=tuple(float(self.module_material.get(name,0.0)) for name in s65.GRAMMAR_NAMES)
        return bool(
            np.isfinite(self.pools).all()
            and np.isfinite(self.damage)
            and np.isfinite(self.preference)
            and np.isfinite(np.asarray(module_values,dtype=float)).all()
        )
    def state_dict(self):
        return {
            'cell_id':self.cell_id,'parent_id':self.parent_id,'lineage':self.lineage,
            'founder_class':self.founder_class,'generation':self.generation,'genome':self.genome.copy(),
            'niche':self.niche,'pools':self.pools.copy(),'age':self.age,'damage':self.damage,
            'alive':self.alive,'death_reason':self.death_reason,'copy_symbols':self.copy_symbols,
            'copy_complete':self.copy_complete,'expression_progress':dict(self.expression_progress),
            'expressed_modules':sorted(self.expressed_modules),'module_material':dict(self.module_material),'uptake_ema':self.uptake_ema,
            'previous_uptake':self.previous_uptake,'preference':self.preference,
            'predicted_niche':self.predicted_niche,'last_switch_age':self.last_switch_age,
            'grammar_atp':self.grammar_atp,'grammar_wear':self.grammar_wear,
            'movement_atp':self.movement_atp,'hgt_integrations':self.hgt_integrations,
            'grammar_hgt_acquisitions':self.grammar_hgt_acquisitions,'grammar_reentries':self.grammar_reentries,
            'mutation_reentries':self.mutation_reentries,'hgt_reentries':self.hgt_reentries,
            'was_grammar_absent':self.was_grammar_absent,'reentry_origin_generation':self.reentry_origin_generation,'reentry_lineage':self.reentry_lineage,'reentry_source':self.reentry_source,
            'mutation_events':self.mutation_events,'total_uptake':self.total_uptake,
        }
    @classmethod
    def from_state(cls,state):
        c=cls(state['cell_id'],state['parent_id'],state['lineage'],state['founder_class'],state['generation'],state['genome'],state['niche'],state['pools'],pretranslated=False)
        c.pools=np.asarray(state['pools'],dtype=float).copy()
        for key in ('age','damage','alive','death_reason','copy_symbols','copy_complete','uptake_ema','previous_uptake','preference','predicted_niche','last_switch_age','grammar_atp','grammar_wear','movement_atp','hgt_integrations','grammar_hgt_acquisitions','grammar_reentries','mutation_reentries','hgt_reentries','was_grammar_absent','reentry_origin_generation','reentry_lineage','reentry_source','mutation_events','total_uptake'):
            if key in state:setattr(c,key,state[key])
        c.expression_progress={str(k):float(v) for k,v in dict(state.get('expression_progress',{})).items()}
        c.module_material={str(k):float(v) for k,v in dict(state.get('module_material',{})).items()}
        for name in s65.GRAMMAR_NAMES:
            c.expression_progress.setdefault(name,0.0); c.module_material.setdefault(name,0.0)
        c.expressed_modules=set(str(v) for v in state.get('expressed_modules',[]))
        return c


class EDNAFragment(object):
    def __init__(self,sequence,niche,origin_lineage,age=0.0):
        self.sequence=np.asarray(sequence,dtype=np.uint8).copy(); self.niche=int(niche)%2
        self.origin_lineage=int(origin_lineage); self.age=float(age)
    def mass(self): return len(self.sequence)*MONOMER_MASS
    def state_dict(self): return {'sequence':self.sequence.copy(),'niche':self.niche,'origin_lineage':self.origin_lineage,'age':self.age}
    @classmethod
    def from_state(cls,state): return cls(state['sequence'],state['niche'],state['origin_lineage'],state.get('age',0.0))


class LongHorizon067World(object):
    def __init__(self,seed=101,config=None):
        self.seed=int(seed); self.rng=np.random.default_rng(self.seed)
        self.config=config if isinstance(config,Formal067Config) else Formal067Config.from_state(config or {})
        self.age=0.0; self.steps=0; self.next_cell_id=0
        self.cells=[]; self.edna=[]
        self.patch_fuel=np.array([2.5,1.3],dtype=float)
        self.patch_mineral=np.array([2.0,1.1],dtype=float)
        self.patch_waste=np.zeros(2,dtype=float)
        self.patch_cue=np.zeros(2,dtype=float)
        self.dissipated_energy=0.0; self.exported_material=0.0; self.injected_material=0.0
        self.divisions=0; self.deaths=0; self.transfer_events=0; self.rule_switches=0
        self.next_transfer=float(self.config.transfer_start); self.last_rich_niche=self.rich_niche()
        self.mutation_events=0; self.mutation_rejections=0; self.mutation_breakdown={k:0 for k in ('deletion','duplication','dormancy','reactivation','regulatory')}
        self.hgt_integrations=0; self.grammar_hgt_acquisitions=0; self.grammar_reentries=0
        self.mutation_reentries=0; self.hgt_reentries=0
        self.first_complete_loss_age=None; self.first_reentry_age=None
        self.generation_milestones={}; self.history=[]
        self.external_fitness_events=0
        self._make_founders()
        self.initial_total_material=self.current_material()
        self.last_material_residual=0.0

    def _core_genome(self):
        # Reuse the fully validated 0.6.6 material genome as the common core.
        cfg=s66.Formal066Config(ecology_environment=ENV_STABLE,eco66_founder_prime=False,eco66_chemostat=False,eco66_washout=False)
        micro=s66.Formal066World(seed=self.seed,initial_cells=3,config=cfg)
        return s65.knockout_grammar(micro.cells[0].genomes[0])

    def _new_cell(self,parent_id,lineage,founder_class,generation,genome,niche,pools=None,pretranslated=False):
        c=LongHorizonCell(self.next_cell_id,parent_id,lineage,founder_class,generation,genome,niche,pools,pretranslated=pretranslated)
        self.next_cell_id+=1; return c

    def _make_founders(self):
        base=self._core_genome(); full=s65.reintroduce_full_grammar(base); dormant=s65.reintroduce_full_grammar(base)
        for module in range(s65.GRAMMAR_COUNT): dormant=s65.set_grammar_module_promoter(dormant,module,1)
        seqs=((FOUNDER_FULL,full),(FOUNDER_DORMANT,dormant),(FOUNDER_ABSENT,base))
        n=max(1,int(self.config.initial_per_founder))
        for lineage,(label,seq) in enumerate(seqs):
            for j in range(n):
                niche=(lineage+j)%2
                pools=np.array([0.65,0.50,0.28,0.50,1.25,0.48,0.02,0.001],dtype=float)
                c=self._new_cell(-1,lineage,label,0,seq,niche,pools,pretranslated=True)
                self.cells.append(c)

    def rich_niche(self,age=None):
        age=self.age if age is None else float(age)
        if self.config.ecology_environment==ENV_STABLE:return 0
        if age<self.config.rule_first_age:return 0
        epoch=int(math.floor((age-self.config.rule_first_age)/max(self.config.rule_period,1e-9)))
        return epoch%2

    def cue_niche(self):
        if self.config.ecology_environment!=ENV_LONG_DELAY:return self.rich_niche()
        return self.rich_niche(self.age+self.config.cue_lead)

    def _inject_resources(self,dt):
        rich=self.rich_niche(); poor=1-rich
        if self.config.ecology_environment==ENV_STABLE:
            fuel=np.array([1.00,0.52])*self.config.fuel_inflow*dt
            mineral=np.array([1.00,0.60])*self.config.mineral_inflow*dt
        else:
            fuel=np.zeros(2); mineral=np.zeros(2)
            fuel[rich]=self.config.fuel_inflow*dt; fuel[poor]=0.34*self.config.fuel_inflow*dt
            mineral[rich]=self.config.mineral_inflow*dt; mineral[poor]=0.48*self.config.mineral_inflow*dt
        room_f=np.maximum(0.0,self.config.patch_capacity-self.patch_fuel)
        room_m=np.maximum(0.0,self.config.patch_capacity-self.patch_mineral)
        fuel=np.minimum(fuel,room_f); mineral=np.minimum(mineral,room_m)
        self.patch_fuel+=fuel; self.patch_mineral+=mineral
        injected=float(np.sum(fuel)+np.sum(mineral))
        # Long-delay information is an explicit extracellular cue molecule,
        # not a privileged future-state flag delivered to the cell.
        if self.config.ecology_environment==ENV_LONG_DELAY:
            cue_target=self.rich_niche(self.age+self.config.cue_lead)
            cue_add=min(max(0.0,self.config.cue_capacity-self.patch_cue[cue_target]),self.config.cue_inflow*dt)
            self.patch_cue[cue_target]+=cue_add; injected+=cue_add
            decayed=np.minimum(self.patch_cue,np.maximum(0.0,self.config.cue_decay*dt*self.patch_cue))
            self.patch_cue-=decayed; self.patch_waste+=decayed
        self.injected_material+=injected
        if rich!=self.last_rich_niche:self.rule_switches+=1; self.last_rich_niche=rich

    def _module_expression(self,c,dt):
        raw=c.traits_raw()
        for name in s65.GRAMMAR_NAMES:
            if not raw.get(name,False):
                recycled=float(c.module_material.get(name,0.0))
                if recycled>0.0:
                    c.pools[P_STRUCTURE]+=recycled
                c.module_material[name]=0.0
                c.expression_progress[name]=0.0
                c.expressed_modules.discard(name)
                continue
            have=float(c.module_material.get(name,0.0))
            need=max(0.0,MODULE_PROTEIN_TARGET-have)
            make=min(need,0.00015*dt,float(c.pools[P_STRUCTURE]))
            atp_cost=make*MODULE_TRANSLATION_ATP_PER_MATERIAL
            if make>0.0 and c.pools[P_ATP]>=atp_cost:
                c.pools[P_ATP]-=atp_cost
                c.pools[P_STRUCTURE]-=make
                c.module_material[name]=have+make
                self.dissipated_energy+=atp_cost
            c.expression_progress[name]=clamp(float(c.module_material.get(name,0.0))/max(MODULE_PROTEIN_TARGET,1e-12),0.0,1.0)
            if c.expression_progress[name]>=1.0-1e-9:
                c.expressed_modules.add(name)
            else:
                c.expressed_modules.discard(name)

    def _behavior(self,c,dt):
        traits=c.traits(); current=c.niche; other=1-current
        local=float(self.patch_fuel[current]+0.75*self.patch_mineral[current])
        other_level=float(self.patch_fuel[other]+0.75*self.patch_mineral[other])
        delta=c.previous_uptake-c.uptake_ema
        desired=current
        # Non-neural chemotaxis is slow and local: it can only escape a depleted patch.
        if local<0.45 and self.rng.random()<0.045*dt: desired=other
        if traits.get('organ'):
            if (self.config.ecology_environment==ENV_LONG_DELAY and traits.get('prediction')
                    and traits.get('sentinel') and abs(float(self.patch_cue[0]-self.patch_cue[1]))>0.015):
                desired=int(np.argmax(self.patch_cue))
            elif traits.get('sentinel') and traits.get('plasticity'):
                if delta<-0.006 or (local+0.16<other_level): desired=other
                elif c.uptake_ema>0.025: desired=current
            elif traits.get('sentinel') and local<0.70: desired=other
        if desired!=current:
            readiness=0.35+0.65*float(traits.get('readiness',False))
            cost=0.0032*(1.0/readiness)
            if c.pools[P_ATP]>=cost and self.age-c.last_switch_age>0.65:
                c.pools[P_ATP]-=cost; self.dissipated_energy+=cost; c.movement_atp+=cost
                c.niche=desired; c.last_switch_age=self.age
        c.preference=0.88*c.preference+0.12*float(c.niche)

    def _allocate_uptake(self,dt):
        living=[c for c in self.cells if c.alive]
        demands={0:[],1:[]}
        for c in living:
            structure_need=max(0.0,1.45-c.pools[P_STRUCTURE])
            demand=(0.055+0.035*structure_need)*dt
            demands[c.niche].append((c,demand))
        for niche in (0,1):
            items=demands[niche]
            if not items:continue
            total=sum(v for _,v in items)
            fuel_avail=float(self.patch_fuel[niche]); min_avail=float(self.patch_mineral[niche])
            scale=min(1.0,fuel_avail/max(total,1e-12),min_avail/max(total*0.72,1e-12))
            for c,demand in items:
                fuel=demand*scale; mineral=demand*0.72*scale
                self.patch_fuel[niche]-=fuel; self.patch_mineral[niche]-=mineral
                c.pools[P_FUEL]+=fuel; c.pools[P_MINERAL]+=mineral
                uptake=fuel+mineral; c.previous_uptake=uptake; c.uptake_ema=0.88*c.uptake_ema+0.12*uptake
                c.total_uptake+=uptake

    def _metabolize(self,c,dt):
        self._module_expression(c,dt)
        fuel=min(c.pools[P_FUEL],0.12*dt); mineral=min(c.pools[P_MINERAL],0.09*dt)
        amount=min(fuel,mineral/0.75)
        if amount>0.0:
            c.pools[P_FUEL]-=amount; c.pools[P_MINERAL]-=0.75*amount
            total=1.75*amount
            c.pools[P_ATP]+=0.36*total
            c.pools[P_STRUCTURE]+=0.27*total
            c.pools[P_MEMBRANE]+=0.16*total
            c.pools[P_NUCLEOTIDE]+=0.15*total
            c.pools[P_WASTE]+=0.06*total
        traits=c.traits(); active=active_module_names(traits)
        module_cost=sum(MODULE_COST[n] for n in active)*dt
        paid=min(c.pools[P_ATP],module_cost); c.pools[P_ATP]-=paid; self.dissipated_energy+=paid; c.grammar_atp+=paid
        wear_budget=paid*MODULE_WEAR
        if wear_budget>0.0 and active:
            per=wear_budget/float(len(active))
            worn=0.0
            for name in active:
                loss=min(float(c.module_material.get(name,0.0)),per)
                c.module_material[name]=float(c.module_material.get(name,0.0))-loss
                worn+=loss
            c.pools[P_REACTIVE]+=worn; c.grammar_wear+=worn
        basal=0.0018*dt+0.00016*c.material_mass()*dt
        paid=min(c.pools[P_ATP],basal); c.pools[P_ATP]-=paid; self.dissipated_energy+=paid
        # Waste export and damage/repair are material transformations.
        exported=min(c.pools[P_WASTE],0.012*dt); c.pools[P_WASTE]-=exported; self.patch_waste[c.niche]+=exported
        reactive=float(c.pools[P_REACTIVE]); c.damage+=dt*(0.012*reactive+0.002*(1.0 if c.pools[P_ATP]<0.012 else 0.0))
        repair=min(c.damage,0.020*dt,c.pools[P_ATP]/0.18,c.pools[P_STRUCTURE]/0.08)
        if repair>0:
            c.damage-=repair; c.pools[P_ATP]-=0.18*repair; c.pools[P_STRUCTURE]-=0.08*repair
            c.pools[P_WASTE]+=0.08*repair; self.dissipated_energy+=0.18*repair
        c.age+=dt

    def _replicate(self,c,dt):
        if c.copy_complete:return
        remaining=len(c.genome)-c.copy_symbols
        if remaining<=0:c.copy_complete=True;return
        max_symbols=max(1,int(self.config.replication_symbols_per_second*dt))
        possible_nuc=int(c.pools[P_NUCLEOTIDE]/MONOMER_MASS+1e-9)
        possible_atp=int(c.pools[P_ATP]/max(self.config.replication_atp_per_symbol,1e-12)+1e-9)
        n=max(0,min(remaining,max_symbols,possible_nuc,possible_atp))
        if n<=0:return
        c.pools[P_NUCLEOTIDE]-=n*MONOMER_MASS
        cost=n*self.config.replication_atp_per_symbol; c.pools[P_ATP]-=cost; self.dissipated_energy+=cost
        c.copy_symbols+=n
        if c.copy_symbols>=len(c.genome):c.copy_symbols=len(c.genome);c.copy_complete=True

    def _can_divide(self,c):
        return bool(c.copy_complete and c.age>=self.config.division_min_age and c.pools[P_STRUCTURE]>=self.config.division_structure and c.pools[P_MEMBRANE]>=self.config.division_membrane and c.pools[P_ATP]>=self.config.division_atp)

    def _material_mutate(self,c):
        if not self.config.mutation:return
        old=c.genome.copy(); before=grammar_module_set(old); was_absent=(len(before)==0)
        proposed,events=s65.mutate_grammar_sequence(old,self.rng,rate=self.config.mutation_rate,duplication=True)
        if not any(events.values()):return
        delta=len(proposed)-len(old); mass=delta*MONOMER_MASS; cost=max(0,delta)*s66.MUTATION_ATP_PER_SYMBOL
        if delta>0 and (c.pools[P_NUCLEOTIDE]<mass-1e-12 or c.pools[P_ATP]<cost-1e-12):return
        if delta>0:
            c.pools[P_NUCLEOTIDE]-=mass; c.pools[P_ATP]-=cost; self.dissipated_energy+=cost
        elif delta<0:c.pools[P_NUCLEOTIDE]+=-mass
        c.genome=np.asarray(proposed,dtype=np.uint8).copy(); c.mutation_events+=1; self.mutation_events+=1
        for key,value in events.items():self.mutation_breakdown[key]+=int(value)
        after=grammar_module_set(c.genome)
        if was_absent and after:
            c.grammar_reentries+=1; c.mutation_reentries+=1
            c.reentry_origin_generation=c.generation; c.reentry_lineage=True; c.reentry_source='mutation'
            self.grammar_reentries+=1; self.mutation_reentries+=1
            if self.first_reentry_age is None:self.first_reentry_age=float(self.age)
        c._refresh_expression(initial=False)

    def _divide(self,c):
        if not self._can_divide(c):return []
        c.pools[P_ATP]-=self.config.division_atp_cost; self.dissipated_energy+=self.config.division_atp_cost
        # Learned control states are not copied. Existing module proteins are
        # unfolded into the shared structural pool before the physical split.
        module_mass=c.module_mass()
        if module_mass>0.0:
            c.pools[P_STRUCTURE]+=module_mass
            for name in s65.GRAMMAR_NAMES:c.module_material[name]=0.0
            c.expressed_modules.clear()
        pools_a=0.5*c.pools; pools_b=c.pools-pools_a
        # One completed polymer copy goes to each daughter; no genome material is created.
        a=self._new_cell(c.cell_id,c.lineage,c.founder_class,c.generation+1,c.genome,c.niche,pools_a)
        b=self._new_cell(c.cell_id,c.lineage,c.founder_class,c.generation+1,c.genome,1-c.niche if self.rng.random()<0.15 else c.niche,pools_b)
        for d in (a,b):
            d.copy_symbols=0; d.copy_complete=False; d.age=0.0; d.expressed_modules=set(); d.expression_progress={name:0.0 for name in s65.GRAMMAR_NAMES}
            d.reentry_lineage=bool(c.reentry_lineage); d.reentry_origin_generation=int(c.reentry_origin_generation); d.reentry_source=str(c.reentry_source)
            self._material_mutate(d)
        c.alive=False; self.divisions+=1
        gen=c.generation+1
        self.generation_milestones.setdefault(int(gen),float(self.age))
        return [a,b]

    def _die(self,c,reason):
        if not c.alive:return
        c.alive=False; c.death_reason=str(reason); self.deaths+=1
        # Soluble body pools return to the local material environment.
        body=float(np.sum(c.pools))+c.module_mass(); self.patch_fuel[c.niche]+=0.28*body; self.patch_mineral[c.niche]+=0.40*body; self.patch_waste[c.niche]+=0.32*body
        # Grammar genes are released as actual eDNA fragments; core polymer is recycled.
        records=s65.grammar_records_from_sequence(c.genome); grammar_symbols=0
        for rec in records:
            a=rec['start']; b=a+g2.GENE_SPAN; frag=c.genome[a:b].copy(); self.edna.append(EDNAFragment(frag,c.niche,c.lineage)); grammar_symbols+=len(frag)
        core_symbols=max(0,len(c.genome)-grammar_symbols+c.copy_symbols)
        self.patch_mineral[c.niche]+=core_symbols*MONOMER_MASS

    def _hgt(self,dt):
        if not self.config.hgt or not self.edna:return
        living=[c for c in self.cells if c.alive]
        if not living:return
        for c in living:
            candidates=[(i,f) for i,f in enumerate(self.edna) if f.niche==c.niche]
            if not candidates or self.rng.random()>=self.config.hgt_probability*dt:continue
            idx,frag=candidates[int(self.rng.integers(0,len(candidates)))]
            if c.pools[P_ATP]<self.config.hgt_atp_cost:continue
            if len(c.genome)+len(frag.sequence)>g2.MAX_GENOME_LENGTH:continue
            before=grammar_module_set(c.genome); was_absent=(len(before)==0)
            c.pools[P_ATP]-=self.config.hgt_atp_cost; self.dissipated_energy+=self.config.hgt_atp_cost
            # An HGT insertion invalidates any partial second-genome copy. Return
            # that polymer to nucleotide and require a fresh physical copy so the
            # acquired gene is never inherited for free.
            if c.copy_symbols>0:
                c.pools[P_NUCLEOTIDE]+=c.copy_symbols*MONOMER_MASS
                c.copy_symbols=0; c.copy_complete=False
            pos=int(self.rng.integers(0,len(c.genome)+1)); c.genome=np.concatenate([c.genome[:pos],frag.sequence,c.genome[pos:]]).astype(np.uint8)
            del self.edna[idx]; c.hgt_integrations+=1; self.hgt_integrations+=1
            after=grammar_module_set(c.genome); gained=after-before
            if gained:
                n=len(gained); c.grammar_hgt_acquisitions+=n; self.grammar_hgt_acquisitions+=n
                if was_absent:
                    c.grammar_reentries+=1; c.hgt_reentries+=1
                    c.reentry_origin_generation=c.generation; c.reentry_lineage=True; c.reentry_source='hgt'
                    self.grammar_reentries+=1; self.hgt_reentries+=1
                    if self.first_reentry_age is None:self.first_reentry_age=float(self.age)
            c._refresh_expression(initial=False)

    def _decay_edna(self,dt):
        keep=[]
        for frag in self.edna:
            frag.age+=dt
            if self.rng.random()<self.config.edna_decay*dt:
                self.patch_mineral[frag.niche]+=frag.mass()
            else:keep.append(frag)
        self.edna=keep

    def _neutral_transfer(self):
        if self.age+1e-12<self.next_transfer:return
        while self.age+1e-12>=self.next_transfer:
            living=[c for c in self.cells if c.alive]
            target=min(self.config.transfer_target,len(living))
            if len(living)>target:
                order=self.rng.permutation(len(living)); keep_ids=set(living[int(i)].cell_id for i in order[:target])
                for c in living:
                    if c.cell_id not in keep_ids:
                        # Transfer fate is genotype-blind. A fixed fraction lyses inside
                        # the vessel and feeds corpse/eDNA chemistry; the rest leaves as outflow.
                        if self.rng.random()<clamp(self.config.transfer_lysis_fraction,0.0,1.0):
                            self._die(c,'neutral_transfer_lysis')
                        else:
                            self.exported_material+=c.material_mass(); c.alive=False; c.death_reason='neutral_transfer_outflow'; self.deaths+=1
            # A fraction of extracellular reservoirs is likewise exported.
            frac=clamp(self.config.resource_outflow,0.0,0.95)
            out=float(np.sum(self.patch_fuel+self.patch_mineral+self.patch_waste+self.patch_cue))*frac
            self.patch_fuel*=1.0-frac; self.patch_mineral*=1.0-frac; self.patch_waste*=1.0-frac; self.patch_cue*=1.0-frac
            self.exported_material+=out; self.transfer_events+=1; self.next_transfer+=self.config.transfer_interval

    def _cap_population(self):
        living=[c for c in self.cells if c.alive]
        if len(living)<=self.config.population_cap:return
        order=self.rng.permutation(len(living)); keep=set(living[int(i)].cell_id for i in order[:self.config.population_cap])
        for c in living:
            if c.cell_id not in keep:
                self.exported_material+=c.material_mass(); c.alive=False; c.death_reason='neutral_capacity_outflow'; self.deaths+=1

    def step(self,dt):
        dt=float(clamp(float(dt),1e-4,0.5)); self._inject_resources(dt)
        living=[c for c in self.cells if c.alive]
        for c in living:self._behavior(c,dt)
        self._allocate_uptake(dt)
        newborn=[]
        for c in list(living):
            self._metabolize(c,dt); self._replicate(c,dt)
            if self._can_divide(c):newborn.extend(self._divide(c))
            elif c.damage>=1.0 or c.pools[P_STRUCTURE]<0.06 or c.pools[P_MEMBRANE]<0.035 or c.age>self.config.max_age:self._die(c,'structural_collapse')
        self.cells.extend(newborn)
        self._hgt(dt); self._decay_edna(dt)
        self.age+=dt; self.steps+=1
        self._neutral_transfer(); self._cap_population()
        self.cells=[c for c in self.cells if c.alive]
        if self.summary()['grammar_absent_frequency']>=0.95 and self.first_complete_loss_age is None:self.first_complete_loss_age=float(self.age)
        self.last_material_residual=self.matter_ledger_residual()
        if not self.finite():raise FloatingPointError('non-finite SOMA-CELL 0.6.7 state')

    def current_material(self):
        return float(sum(c.material_mass() for c in self.cells if c.alive)+np.sum(self.patch_fuel)+np.sum(self.patch_mineral)+np.sum(self.patch_waste)+np.sum(self.patch_cue)+sum(f.mass() for f in self.edna))
    def accounted_total(self): return self.current_material()+self.dissipated_energy+self.exported_material
    def matter_ledger_residual(self): return self.accounted_total()-(self.initial_total_material+self.injected_material)
    def finite(self):
        return bool(np.isfinite(self.patch_fuel).all() and np.isfinite(self.patch_mineral).all() and np.isfinite(self.patch_waste).all() and np.isfinite(self.patch_cue).all() and all(c.finite() for c in self.cells) and np.isfinite(self.matter_ledger_residual()))
    def living_cells(self): return [c for c in self.cells if c.alive]

    def summary(self):
        living=self.living_cells(); raw_traits=[c.traits_raw() for c in living]; expressed_traits=[c.traits() for c in living]
        lineage_counts=[sum(1 for c in living if c.lineage==i) for i in range(3)]
        active=np.array([t['active_count'] for t in raw_traits],dtype=float) if raw_traits else np.zeros(0)
        expressed=np.array([t['active_count'] for t in expressed_traits],dtype=float) if expressed_traits else np.zeros(0)
        absent=np.array([1.0 if c.grammar_absent() else 0.0 for c in living],dtype=float) if living else np.zeros(0)
        module_freq={name:(float(np.mean([1.0 if t.get(name,False) else 0.0 for t in raw_traits])) if raw_traits else 0.0) for name in s65.GRAMMAR_NAMES}
        sigs=len(set(c.signature() for c in living))
        reentry_cells=sum(1 for c in living if bool(c.reentry_lineage))
        reentry_grammar_cells=sum(1 for c in living if bool(c.reentry_lineage) and not c.grammar_absent())
        hgt_reentry_cells=sum(1 for c in living if c.reentry_source=='hgt')
        hgt_reentry_grammar_cells=sum(1 for c in living if c.reentry_source=='hgt' and not c.grammar_absent())
        mutation_reentry_cells=sum(1 for c in living if c.reentry_source=='mutation')
        maxgen=max([c.generation for c in living],default=0)
        return {
            'build':BUILD,'eco67_schema':SCHEMA_VERSION,'age':float(self.age),'steps':int(self.steps),'cells':len(living),
            'divisions':int(self.divisions),'deaths':int(self.deaths),'max_generation':int(maxgen),
            'mean_active_modules':float(np.mean(active)) if len(active) else 0.0,
            'mean_expressed_modules':float(np.mean(expressed)) if len(expressed) else 0.0,
            'grammar_absent_frequency':float(np.mean(absent)) if len(absent) else 0.0,
            'full_count':int(lineage_counts[0]),'dormant_count':int(lineage_counts[1]),'absent_count':int(lineage_counts[2]),
            'lineage_shannon':shannon(lineage_counts),'unique_grammar_signatures':int(sigs),
            'sentinel_frequency':module_freq['sentinel'],'readiness_frequency':module_freq['readiness'],'organ_frequency':module_freq['organ'],
            'prediction_frequency':module_freq['prediction'],'plasticity_frequency':module_freq['plasticity'],
            'mutation_events':int(self.mutation_events),'hgt_integrations':int(self.hgt_integrations),
            'grammar_hgt_acquisitions':int(self.grammar_hgt_acquisitions),'grammar_reentries':int(self.grammar_reentries),
            'mutation_reentries':int(self.mutation_reentries),'hgt_reentries':int(self.hgt_reentries),
            'persistent_reentry_cells':int(reentry_cells),'persistent_reentry_grammar_cells':int(reentry_grammar_cells),
            'persistent_hgt_reentry_cells':int(hgt_reentry_cells),'persistent_hgt_reentry_grammar_cells':int(hgt_reentry_grammar_cells),
            'persistent_mutation_reentry_cells':int(mutation_reentry_cells),
            'reentry_lineage_frequency':float(reentry_cells)/max(len(living),1),
            'edna_fragments':len(self.edna),'transfer_events':int(self.transfer_events),'rule_switches':int(self.rule_switches),
            'patch_fuel_total':float(np.sum(self.patch_fuel)),'patch_mineral_total':float(np.sum(self.patch_mineral)),
            'patch_cue_total':float(np.sum(self.patch_cue)),'cue_contrast':float(abs(self.patch_cue[0]-self.patch_cue[1])),
            'injected_material':float(self.injected_material),'exported_material':float(self.exported_material),'dissipated_energy':float(self.dissipated_energy),
            'matter_residual':float(self.matter_ledger_residual()),'external_fitness_events':int(self.external_fitness_events),
            'first_complete_loss_age':self.first_complete_loss_age,'first_reentry_age':self.first_reentry_age,
            'mean_generation':float(np.mean([c.generation for c in living])) if living else 0.0,
            'mean_damage':float(np.mean([c.damage for c in living])) if living else 0.0,
            'mean_atp':float(np.mean([c.pools[P_ATP] for c in living])) if living else 0.0,
            'module_protein_material':float(sum(c.module_mass() for c in living)),
        }

    def state_dict(self):
        return {
            'save_version':SAVE_VERSION,'build':BUILD,'schema':SCHEMA_VERSION,'seed':self.seed,'config':self.config.state_dict(),
            'rng_state':self.rng.bit_generator.state,'age':self.age,'steps':self.steps,'next_cell_id':self.next_cell_id,
            'cells':[c.state_dict() for c in self.cells],'edna':[f.state_dict() for f in self.edna],
            'patch_fuel':self.patch_fuel.copy(),'patch_mineral':self.patch_mineral.copy(),'patch_waste':self.patch_waste.copy(),'patch_cue':self.patch_cue.copy(),
            'dissipated_energy':self.dissipated_energy,'exported_material':self.exported_material,'injected_material':self.injected_material,
            'divisions':self.divisions,'deaths':self.deaths,'transfer_events':self.transfer_events,'rule_switches':self.rule_switches,
            'next_transfer':self.next_transfer,'last_rich_niche':self.last_rich_niche,'mutation_events':self.mutation_events,
            'mutation_rejections':self.mutation_rejections,'mutation_breakdown':dict(self.mutation_breakdown),
            'hgt_integrations':self.hgt_integrations,'grammar_hgt_acquisitions':self.grammar_hgt_acquisitions,
            'grammar_reentries':self.grammar_reentries,'mutation_reentries':self.mutation_reentries,'hgt_reentries':self.hgt_reentries,
            'first_complete_loss_age':self.first_complete_loss_age,'first_reentry_age':self.first_reentry_age,
            'generation_milestones':dict(self.generation_milestones),'history':list(self.history),
            'external_fitness_events':self.external_fitness_events,'initial_total_material':self.initial_total_material,
            'last_material_residual':self.last_material_residual,
        }
    @classmethod
    def from_state(cls,state):
        world=cls.__new__(cls); world.seed=int(state['seed']); world.rng=np.random.default_rng(world.seed); world.config=Formal067Config.from_state(state['config'])
        world.age=float(state['age']); world.steps=int(state['steps']); world.next_cell_id=int(state['next_cell_id'])
        world.cells=[LongHorizonCell.from_state(v) for v in state['cells']]; world.edna=[EDNAFragment.from_state(v) for v in state.get('edna',[])]
        world.patch_fuel=np.asarray(state['patch_fuel'],dtype=float).copy(); world.patch_mineral=np.asarray(state['patch_mineral'],dtype=float).copy(); world.patch_waste=np.asarray(state['patch_waste'],dtype=float).copy(); world.patch_cue=np.asarray(state.get('patch_cue',np.zeros(2)),dtype=float).copy()
        for key in ('dissipated_energy','exported_material','injected_material','divisions','deaths','transfer_events','rule_switches','next_transfer','last_rich_niche','mutation_events','mutation_rejections','hgt_integrations','grammar_hgt_acquisitions','grammar_reentries','mutation_reentries','hgt_reentries','first_complete_loss_age','first_reentry_age','external_fitness_events','initial_total_material','last_material_residual'):
            setattr(world,key,state.get(key,0))
        world.mutation_breakdown=dict(state.get('mutation_breakdown',{})); world.generation_milestones={int(k):float(v) for k,v in dict(state.get('generation_milestones',{})).items()}; world.history=list(state.get('history',[]))
        world.rng.bit_generator.state=state['rng_state']; return world
    def clone(self): return LongHorizon067World.from_state(self.state_dict())
    def save(self,path=SAVE_FILE): _atomic_pickle(path,self.state_dict())
    @classmethod
    def load(cls,path=SAVE_FILE):
        with open(path,'rb') as handle:return cls.from_state(pickle.load(handle))


Formal067World=LongHorizon067World


def long_horizon_config(environment=ENV_STABLE,mutation=True,hgt=True,**kwargs):
    return Formal067Config(ecology_environment=environment,mutation=mutation,hgt=hgt,**kwargs)


def run_long_horizon_assay(seed=6711,environment=ENV_STABLE,seconds=180.0,mutation=True,hgt=True,config_overrides=None,history_interval=6.0):
    cfg=long_horizon_config(environment=environment,mutation=mutation,hgt=hgt,**dict(config_overrides or {}))
    world=LongHorizon067World(seed=int(seed),config=cfg); dt=1.0/SIM_HZ; max_res=0.0; next_hist=0.0
    for _ in range(int(round(float(seconds)*SIM_HZ))):
        world.step(dt); max_res=max(max_res,abs(world.matter_ledger_residual()))
        if world.age+1e-12>=next_hist:
            s=world.summary(); world.history.append({k:s[k] for k in ('age','cells','max_generation','mean_active_modules','grammar_absent_frequency','lineage_shannon','grammar_reentries')}); next_hist+=history_interval
        if cfg.max_steps is not None and world.steps>=cfg.max_steps:break
    s=world.summary()
    return {
        'seed':int(seed),'environment':str(environment),'seconds':float(world.age),'mutation':int(bool(mutation)),'hgt':int(bool(hgt)),
        'cells':int(s['cells']),'divisions':int(s['divisions']),'deaths':int(s['deaths']),'max_generation':int(s['max_generation']),
        'mean_generation':float(s['mean_generation']),'mean_active_modules':float(s['mean_active_modules']),
        'mean_expressed_modules':float(s['mean_expressed_modules']),
        'grammar_absent_frequency':float(s['grammar_absent_frequency']),'lineage_shannon':float(s['lineage_shannon']),
        'unique_grammar_signatures':int(s['unique_grammar_signatures']),'mutation_events':int(s['mutation_events']),
        'hgt_integrations':int(s['hgt_integrations']),'grammar_hgt_acquisitions':int(s['grammar_hgt_acquisitions']),
        'grammar_reentries':int(s['grammar_reentries']),'mutation_reentries':int(s['mutation_reentries']),'hgt_reentries':int(s['hgt_reentries']),
        'persistent_reentry_cells':int(s['persistent_reentry_cells']),'persistent_reentry_grammar_cells':int(s['persistent_reentry_grammar_cells']),
        'persistent_hgt_reentry_cells':int(s['persistent_hgt_reentry_cells']),'persistent_hgt_reentry_grammar_cells':int(s['persistent_hgt_reentry_grammar_cells']),'persistent_mutation_reentry_cells':int(s['persistent_mutation_reentry_cells']),
        'reentry_lineage_frequency':float(s['reentry_lineage_frequency']),'transfer_events':int(s['transfer_events']),
        'sentinel_frequency':float(s['sentinel_frequency']),'readiness_frequency':float(s['readiness_frequency']),
        'organ_frequency':float(s['organ_frequency']),'prediction_frequency':float(s['prediction_frequency']),
        'plasticity_frequency':float(s['plasticity_frequency']),'first_complete_loss_age':s['first_complete_loss_age'],
        'first_reentry_age':s['first_reentry_age'],'material_residual':float(s['matter_residual']),
        'max_abs_material_residual':float(max_res),'finite':int(bool(world.finite())),'external_fitness_events':int(s['external_fitness_events']),
        'history':list(world.history),
    }


LOG_FIELDS=('session_id','reason','wall_time','age','cells','divisions','deaths','max_generation','mean_generation','mean_active_modules','grammar_absent_frequency','lineage_shannon','unique_grammar_signatures','mutation_events','hgt_integrations','grammar_hgt_acquisitions','grammar_reentries','transfer_events','matter_residual')
class LongRunLogger(object):
    def __init__(self,world,path=LOG_FILE,interval=5.0):
        self.path=path; self.interval=float(interval); self.last_age=-1e9; self.session_id='{}-{}'.format(int(time.time()),world.seed); self.status='WAIT'
    def log(self,world,reason='periodic',force=False):
        if not force and world.age-self.last_age<self.interval:return False
        s=world.summary(); row={k:s.get(k,'') for k in LOG_FIELDS}; row.update({'session_id':self.session_id,'reason':reason,'wall_time':time.time()})
        exists=os.path.exists(self.path) and os.path.getsize(self.path)>0
        with open(self.path,'a',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=LOG_FIELDS)
            if not exists:writer.writeheader()
            writer.writerow(row)
        self.last_age=world.age; self.status='OK'; return True


def generate_report(log_path=LOG_FILE,report_path=REPORT_FILE,session_path=SESSION_FILE):
    if not os.path.exists(log_path):return 'NO LOG'
    with open(log_path,'r',newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    sessions={}
    for row in rows:sessions.setdefault(row['session_id'],[]).append(row)
    fields=('session_id','rows','final_age','final_cells','max_generation','active','absent','reentries','ledger')
    with open(session_path,'w',newline='',encoding='utf-8') as handle:
        w=csv.DictWriter(handle,fieldnames=fields); w.writeheader()
        for sid,items in sessions.items():
            last=items[-1]; w.writerow({'session_id':sid,'rows':len(items),'final_age':last.get('age',''),'final_cells':last.get('cells',''),'max_generation':last.get('max_generation',''),'active':last.get('mean_active_modules',''),'absent':last.get('grammar_absent_frequency',''),'reentries':last.get('grammar_reentries',''),'ledger':last.get('matter_residual','')})
    lines=[BUILD_LONG,'sessions: {}'.format(len(sessions)),'']
    for sid,items in sessions.items():
        last=items[-1]; lines.append('{} age={} cells={} gen={} active={} absent={} reentry={} ledger={}'.format(sid,last.get('age',''),last.get('cells',''),last.get('max_generation',''),last.get('mean_active_modules',''),last.get('grammar_absent_frequency',''),last.get('grammar_reentries',''),last.get('matter_residual','')))
    with open(report_path,'w',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
    return 'OK'


try:
    from scene import Scene,run,LANDSCAPE,background,fill,rect,ellipse,line,stroke,stroke_weight,text
    class SomaCell067Scene(Scene):
        def setup(self):
            self.paused=False; self.last_touch=-10.0; self.last_wall=time.time(); self.accumulator=0.0; self.fps=0.0; self.sim_rate=0.0; self.telemetry_wall=time.time(); self.telemetry_age=0.0; self.telemetry_frames=0
            try:self.world=LongHorizon067World.load(SAVE_FILE);self.save_status='LOAD'
            except Exception:self.world=LongHorizon067World(seed=101,config=Formal067Config(ecology_environment=ENV_PERIODIC));self.save_status='NEW'
            self.logger=LongRunLogger(self.world);self.logger.log(self.world,reason='start',force=True);self.report_status='WAIT';self.last_save_age=self.world.age
        def _fresh_world(self):return LongHorizon067World(seed=101,config=Formal067Config(ecology_environment=ENV_PERIODIC))
        def update(self):
            now=time.time(); elapsed=min(0.25,max(0.0,now-self.last_wall)); self.last_wall=now; self.telemetry_frames+=1
            if not self.paused:
                self.accumulator+=elapsed; fixed=1.0/SIM_HZ; steps=0
                while self.accumulator>=fixed and steps<8:self.world.step(fixed);self.accumulator-=fixed;steps+=1
            td=now-self.telemetry_wall
            if td>=1.0:
                self.fps=self.telemetry_frames/td; self.sim_rate=(self.world.age-self.telemetry_age)/td; self.telemetry_wall=now;self.telemetry_age=self.world.age;self.telemetry_frames=0
            self.logger.log(self.world)
            if self.world.age-self.last_save_age>=60.0:
                try:self.world.save(SAVE_FILE);self.save_status='OK';self.last_save_age=self.world.age;gc.collect()
                except Exception:self.save_status='ERR'
        def draw(self):
            background(0.008,0.014,0.026); w=float(self.size.w);h=float(self.size.h)
            top=50.0;bottom=92.0;mid=h-top-bottom
            fill(0.015,0.030,0.046);rect(18,bottom,w-36,mid)
            # Two material niches.
            niche_w=(w-52)/2.0
            colours=((0.10,0.28,0.38),(0.20,0.16,0.34))
            for i in (0,1):
                x=22+i*(niche_w+8); fill(*colours[i]);rect(x,bottom+4,niche_w,mid-8)
                fill(0.60,0.85,0.95);text('niche {}  fuel {:.2f} mineral {:.2f}'.format(i,self.world.patch_fuel[i],self.world.patch_mineral[i]),x=x+10,y=h-top-18,font_size=9,alignment=4)
            living=self.world.living_cells()
            for idx,c in enumerate(living):
                col=(0.75,0.32,0.92) if c.founder_class==FOUNDER_FULL else ((0.32,0.75,0.96) if c.founder_class==FOUNDER_DORMANT else (0.32,0.90,0.48))
                local=[x for x in living if x.niche==c.niche];rank=local.index(c);cols=max(1,int(math.ceil(math.sqrt(len(local)))));row=rank//cols;cc=rank%cols
                x=34+c.niche*(niche_w+8)+cc*34; y=bottom+20+row*32; r=6.0+1.2*c.traits()['active_count']
                fill(col[0],col[1],col[2],0.86);ellipse(x-r,y-r,2*r,2*r);fill(0.95,0.98,1.0);text('G{} A{}'.format(c.generation,c.traits()['active_count']),x=x,y=y-12,font_size=7,alignment=5)
            fill(0.010,0.018,0.030,0.97);rect(0,h-top,w,top);rect(0,0,w,bottom)
            s=self.world.summary();fill(0.92,0.98,1.0);text(BUILD,x=18,y=h-26,font_size=18,alignment=4)
            fill(0.64,0.78,0.86);text('{} | SAVE {} | {:.1f} fps | x{:.2f}'.format(self.world.config.ecology_environment,self.save_status,self.fps,self.sim_rate),x=w-18,y=h-26,font_size=9,alignment=6)
            fill(0.84,0.92,0.97);text('age {:.1f} cells {} gen {} div/death {}/{} transfer {}'.format(s['age'],s['cells'],s['max_generation'],s['divisions'],s['deaths'],s['transfer_events']),x=18,y=74,font_size=10,alignment=4)
            text('active {:.2f} absent {:.2f} lineage H {:.2f} signatures {}'.format(s['mean_active_modules'],s['grammar_absent_frequency'],s['lineage_shannon'],s['unique_grammar_signatures']),x=18,y=54,font_size=9,alignment=4)
            text('mut {} HGT {} grammar-HGT {} reentry {} eDNA {} cue {:.2f}'.format(s['mutation_events'],s['hgt_integrations'],s['grammar_hgt_acquisitions'],s['grammar_reentries'],s['edna_fragments'],s.get('patch_cue_total',0.0)),x=18,y=34,font_size=9,alignment=4)
            text('ledger {:+.2e} injected {:.2f} exported {:.2f} dissipated {:.2f}'.format(s['matter_residual'],s['injected_material'],s['exported_material'],s['dissipated_energy']),x=18,y=14,font_size=9,alignment=4)
            if self.paused:fill(1.0,0.92,0.45);text('PAUSED',x=w*0.5,y=h-26,font_size=13,alignment=5)
        def touch_began(self,touch):
            now=time.time()
            if now-self.last_touch<0.42:
                try:
                    if os.path.exists(SAVE_FILE):os.remove(SAVE_FILE)
                except Exception:pass
                self.world=self._fresh_world();self.logger=LongRunLogger(self.world);self.logger.log(self.world,reason='reset',force=True);self.save_status='NEW';self.report_status='WAIT';self.accumulator=0.0;self.paused=False;self.last_touch=-10.0;gc.collect();return
            self.last_touch=now
            if touch.location.y>self.size.h-60:self.paused=not self.paused;return
        def stop(self):
            try:self.world.save(SAVE_FILE);self.save_status='OK'
            except Exception:self.save_status='ERR'
            self.logger.log(self.world,reason='stop',force=True);self.report_status=generate_report()
except ImportError:
    SomaCell067Scene=None


def _run_scene():
    if SomaCell067Scene is None:raise RuntimeError('Pythonista scene module required')
    run(SomaCell067Scene(),orientation=LANDSCAPE,show_fps=False,multi_touch=False)

if __name__=='__main__':_run_scene()
