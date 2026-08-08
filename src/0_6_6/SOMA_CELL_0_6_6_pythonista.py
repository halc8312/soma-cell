# coding: utf-8
"""SOMA-CELL 0.6.6 — shared chemical ecology with endogenous generations.

This release candidate removes the 0.6.5 external tournament reproduction loop.
Full, dormant and grammar-absent founders occupy one material world, consume the
same finite resources, replicate DNA, divide, die, become corpses, and exchange
material eDNA/HGT. Neural-grammar mutations occur only on physical daughter
genomes. No organism receives a fitness value, reward label, correct direction,
or environmental switch time.
"""
from __future__ import division

import csv
import gc
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ('.', '../0_6_5', '../0_6_4', '../0_6_3', '../0_6_2', '../0_6_1', '../0_6',
            '../0_6_p2', '../0_6_p1', '../0_6_p0', '../baseline'):
    path = os.path.abspath(os.path.join(HERE, rel))
    if path not in sys.path:
        sys.path.insert(0, path)

import SOMA_CELL_0_6_5_pythonista as s65

s64=s65.s64; s63=s65.s63; s62=s65.s62; s61=s65.s61; f06=s65.f06
p2=s65.p2; p1=s65.p1; p0=s65.p0; s5=s65.s5; s4=s65.s4; g2=s65.g2

BUILD='SOMA-CELL 0.6.6'
BUILD_LONG=BUILD+' | endogenous shared chemical ecology'
SCHEMA_VERSION='0.6.6-ECO1.0'
SAVE_VERSION=1
SAVE_FILE='soma_cell_0_6_6.pkl'
LOG_FILE='soma_cell_0_6_6_longrun.csv'
REPORT_FILE='soma_cell_0_6_6_report.txt'
SESSION_FILE='soma_cell_0_6_6_sessions.csv'
SIM_HZ=10.0
clamp=s65.clamp
_atomic_pickle=s65._atomic_pickle

ENV_STABLE=s65.ENV_STABLE
ENV_PERIODIC=s65.ENV_PERIODIC
ENV_LONG_DELAY=s65.ENV_LONG_DELAY
ECO66_ENVIRONMENTS=(ENV_STABLE,ENV_PERIODIC,ENV_LONG_DELAY)
FOUNDER_FULL='full'
FOUNDER_DORMANT='dormant'
FOUNDER_ABSENT='absent'
FOUNDER_NAMES=(FOUNDER_FULL,FOUNDER_DORMANT,FOUNDER_ABSENT)
FOUNDER_BY_LINEAGE={0:FOUNDER_FULL,1:FOUNDER_DORMANT,2:FOUNDER_ABSENT}

# Material mutation costs. Inserted genome symbols consume ordinary nucleotide
# matter and ATP; deleted symbols return nucleotide matter to the cytoplasm.
MUTATION_ATP_PER_SYMBOL=0.00011
GRAMMAR_ACTIVITY_ATP_RATES={
    'sentinel':0.000010,
    'readiness':0.000007,
    'organ':0.000013,
    'prediction':0.000009,
    'plasticity':0.000010,
}
GRAMMAR_ACTIVITY_WEAR_RATE=0.020


def torus_delta(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    d=(d+0.5)%1.0-0.5
    return d


def active_module_names(traits):
    return tuple(name for name in s65.GRAMMAR_NAMES if bool(traits.get(name,False)))


def grammar_module_set(sequence):
    return frozenset(int(rec['module']) for rec in s65.grammar_records_from_sequence(sequence))


def _material_mutate_grammar(cell, world, rate):
    """Mutate a daughter genome while conserving matter and paying insertion ATP."""
    if not cell.genomes or rate <= 0.0:
        return {'attempted':0,'accepted':0,'rejected_material':0,'events':{k:0 for k in ('deletion','duplication','dormancy','reactivation','regulatory')}}
    original=np.asarray(cell.genomes[0],dtype=np.uint8).copy()
    proposed,events=s65.mutate_grammar_sequence(original,world.rng,rate=rate,duplication=True)
    attempted=int(any(events.values()))
    if not attempted:
        return {'attempted':0,'accepted':0,'rejected_material':0,'events':events}
    delta=int(len(proposed)-len(original))
    nucleotide_delta=float(delta)*s5.MONOMER_MASS
    atp_cost=max(0,delta)*MUTATION_ATP_PER_SYMBOL
    if delta>0 and (cell.pools[s5.POOL_NUCLEOTIDE] < nucleotide_delta-1e-12 or cell.pools[s5.POOL_ATP] < atp_cost+1e-12):
        return {'attempted':1,'accepted':0,'rejected_material':1,'events':{k:0 for k in events}}
    if delta>0:
        cell.pools[s5.POOL_NUCLEOTIDE]-=nucleotide_delta
        cell.pools[s5.POOL_ATP]-=atp_cost
        world.dissipated_energy+=atp_cost
    elif delta<0:
        cell.pools[s5.POOL_NUCLEOTIDE]+=-nucleotide_delta
    cell.genomes[0]=np.asarray(proposed,dtype=np.uint8).copy()
    cell._refresh_gene_cache()
    cell._sync_protein_pool()
    return {'attempted':1,'accepted':1,'rejected_material':0,'events':events}


class Formal066Config(s65.Formal065Config):
    def __init__(self,
                 ecology_environment=ENV_STABLE,
                 shared_ecology=True,
                 eco66_founder_mode='mixed',
                 eco66_mutation=True,
                 eco66_hgt=True,
                 eco66_mutation_rate=0.12,
                 eco66_founder_prime=True,
                 eco66_chemostat=True,
                 eco66_chemostat_interval=0.75,
                 eco66_fuel_rate=0.250,
                 eco66_mineral_rate=0.220,
                 eco66_washout=True,
                 eco66_washout_interval=24.0,
                 eco66_washout_start=70.0,
                 eco66_rule_first_age=18.0,
                 eco66_rule_period=18.0,
                 eco66_motor_adaptation=True,
                 eco66_grammar_costs=True,
                 eco66_replication_rate_scale=5.0,
                 eco66_division_rate_scale=2.0,
                 **kwargs):
        if ecology_environment not in ECO66_ENVIRONMENTS:
            raise ValueError('unknown 0.6.6 ecology environment')
        self.ecology_environment=str(ecology_environment)
        self.shared_ecology=bool(shared_ecology)
        self.eco66_founder_mode=str(eco66_founder_mode)
        if self.eco66_founder_mode not in ('mixed','absent_only','full_only','dormant_only'):
            raise ValueError('unknown 0.6.6 founder mode')
        self.eco66_mutation=bool(eco66_mutation)
        self.eco66_hgt=bool(eco66_hgt)
        self.eco66_mutation_rate=float(eco66_mutation_rate)
        self.eco66_founder_prime=bool(eco66_founder_prime)
        self.eco66_chemostat=bool(eco66_chemostat)
        self.eco66_chemostat_interval=float(eco66_chemostat_interval)
        self.eco66_fuel_rate=float(eco66_fuel_rate)
        self.eco66_mineral_rate=float(eco66_mineral_rate)
        self.eco66_washout=bool(eco66_washout)
        self.eco66_washout_interval=float(eco66_washout_interval)
        self.eco66_washout_start=float(eco66_washout_start)
        self.eco66_rule_first_age=float(eco66_rule_first_age)
        self.eco66_rule_period=float(eco66_rule_period)
        self.eco66_motor_adaptation=bool(eco66_motor_adaptation)
        self.eco66_grammar_costs=bool(eco66_grammar_costs)
        self.eco66_replication_rate_scale=float(eco66_replication_rate_scale)
        self.eco66_division_rate_scale=float(eco66_division_rate_scale)
        # Per-cell grammar phenotype is implemented below; inherited global
        # neurogenesis is held at bare mode so one genotype cannot globally
        # change the phenotype of competitors sharing the same world.
        kwargs.setdefault('grammar_install',False)
        kwargs.setdefault('grammar_initial','dormant')
        kwargs.setdefault('evolution_environment',ecology_environment if ecology_environment in s65.EVOLUTION_ENVIRONMENTS else ENV_STABLE)
        kwargs.setdefault('amortization_policy',s64.AMORT_BARE)
        kwargs.setdefault('extracellular_dna',bool(eco66_hgt))
        kwargs.setdefault('competence',bool(eco66_hgt))
        kwargs.setdefault('recombination',bool(eco66_hgt))
        kwargs.setdefault('seed_mobile_fragments',False)
        kwargs.setdefault('external_test_harness',True)
        super(Formal066Config,self).__init__(**kwargs)
    @classmethod
    def from_state(cls,state):
        state=dict(state or {})
        allowed=set(cls().__dict__.keys()); allowed.discard('audit_active_cells')
        state.pop('audit_active_cells',None)
        return cls(**{k:v for k,v in state.items() if k in allowed})


class Formal066ProtoCell(s65.Formal065ProtoCell):
    def _init_066_state(self):
        self.eco66_founder_class=getattr(self,'eco66_founder_class',FOUNDER_BY_LINEAGE.get(int(getattr(self,'lineage',-1)),'derived'))
        self.eco66_prev_pos=np.asarray(getattr(self,'eco66_prev_pos',self.pos),dtype=float).copy()
        self.eco66_prev_issued=np.asarray(getattr(self,'eco66_prev_issued',np.zeros(2)),dtype=float).copy()
        self.eco66_actuator_estimate=float(getattr(self,'eco66_actuator_estimate',1.0))
        self.eco66_estimator_evidence=float(getattr(self,'eco66_estimator_evidence',0.0))
        self.eco66_grammar_atp=float(getattr(self,'eco66_grammar_atp',0.0))
        self.eco66_grammar_wear=float(getattr(self,'eco66_grammar_wear',0.0))
        self.eco66_mutation_count=int(getattr(self,'eco66_mutation_count',0))
        self.eco66_hgt_grammar_acquisitions=int(getattr(self,'eco66_hgt_grammar_acquisitions',0))
        self.eco66_last_hgt_integrations=int(getattr(self,'eco66_last_hgt_integrations',getattr(self,'hgt_integrations',0)))
        self.eco66_prev_modules=frozenset(getattr(self,'eco66_prev_modules',grammar_module_set(self.genomes[0]) if self.genomes else ()))

    def ready_for_division(self):
        # Ecology-only time compression: preserve the 0.3 material checkpoint
        # but shorten the chronological gate equally for every genotype.
        if (not self.alive or len(self.genomes)<2 or self.replication_template is not None):
            return False
        return bool(
            self.closure()>0.955 and self.worst_gap()<0.26
            and float(np.sum(self.membrane))>g2.INITIAL_MEMBRANE_MASS*1.12
            and self.pools[s5.POOL_CATALYST]>0.40
            and self.pools[s5.POOL_ATP]>0.015
            and self.pools[s5.POOL_MEM_PRECURSOR]>0.030
            and self.material_mass()>4.00
            and self.age>24.0
        )

    def metabolism(self,world,dt,config):
        super(Formal066ProtoCell,self).metabolism(world,dt,config)
        # Ecology kinetic support: mineral can be converted into membrane
        # precursor through an ATP-paid route. Matter is transferred between
        # existing pools; no membrane material is created for free.
        if not self.alive:
            return
        mineral=float(self.pools[s5.POOL_MINERAL]); atp=float(self.pools[s5.POOL_ATP])
        need=max(0.0,0.095-float(self.pools[s5.POOL_MEM_PRECURSOR]))
        amount=min(need,0.024*dt,max(0.0,mineral-0.08),max(0.0,atp-0.018)/0.30)
        if amount>0.0:
            self.pools[s5.POOL_MINERAL]-=amount
            self.pools[s5.POOL_MEM_PRECURSOR]+=amount
            self.pools[s5.POOL_ATP]-=0.30*amount
            world.dissipated_energy+=0.30*amount

    def _replicate_genome(self,world,dt,config):
        # Time-compressed ecology kinetics: polymerase turns over faster but every
        # copied symbol still consumes the inherited nucleotide + ATP costs.
        scale=max(0.25,float(getattr(world.config,'eco66_replication_rate_scale',1.0)))
        return super(Formal066ProtoCell,self)._replicate_genome(world,dt*scale,config)

    def update_division(self,dt,config):
        # Septum chemistry is likewise kinetically accelerated for the ecology
        # assay; precursor and ATP stoichiometry remain unchanged.
        scale=max(0.25,float(getattr(config,'eco66_division_rate_scale',1.0)))
        return super(Formal066ProtoCell,self).update_division(dt*scale,config)

    def _grammar_traits(self):
        if not self.genomes:
            return {'active_count':0}
        return s65.grammar_traits_from_sequence(self.genomes[0])

    def _pay_grammar_costs(self,world,dt,traits):
        if not getattr(world.config,'eco66_grammar_costs',True):
            return
        requested=sum(GRAMMAR_ACTIVITY_ATP_RATES[name] for name in active_module_names(traits))*dt
        if requested<=0.0:
            return
        paid=min(float(self.pools[s5.POOL_ATP]),requested)
        self.pools[s5.POOL_ATP]-=paid
        world.dissipated_energy+=paid
        self.eco66_grammar_atp+=paid
        converted=min(float(self.pools[s5.POOL_WASTE]),paid*GRAMMAR_ACTIVITY_WEAR_RATE)
        self.pools[s5.POOL_WASTE]-=converted
        self.pools[s5.POOL_REACTIVE]+=converted
        self.eco66_grammar_wear+=converted

    def _update_actuator_estimate(self,world,traits):
        if not getattr(world.config,'eco66_motor_adaptation',True):
            return
        if not (traits.get('sentinel') and traits.get('organ') and traits.get('plasticity')):
            return
        command=np.asarray(self.eco66_prev_issued,dtype=float)
        cn=float(np.linalg.norm(command))
        if cn<0.18:
            return
        disp=torus_delta(self.pos,self.eco66_prev_pos)
        proj=float(np.dot(disp,command/max(cn,1e-12)))
        if abs(proj)<1e-6:
            return
        obs=1.0 if proj>=0.0 else -1.0
        alpha=0.18 if traits.get('prediction') else 0.10
        if world.config.ecology_environment==ENV_LONG_DELAY and not traits.get('prediction'):
            alpha*=0.42
        self.eco66_actuator_estimate=(1.0-alpha)*self.eco66_actuator_estimate+alpha*obs
        self.eco66_actuator_estimate=float(clamp(self.eco66_actuator_estimate,-1.0,1.0))
        self.eco66_estimator_evidence=float(clamp(self.eco66_estimator_evidence+0.08,0.0,1.0))

    def apply_effectors(self,world,dt,config):
        self._init_066_state()
        traits=self._grammar_traits()
        self._update_actuator_estimate(world,traits)
        self._pay_grammar_costs(world,dt,traits)

        intended=np.asarray(self.last_control_vectors[s4.CONTROL_MOTOR],dtype=float).copy()
        compensate=1.0
        if (traits.get('sentinel') and traits.get('organ') and traits.get('plasticity')
                and self.eco66_estimator_evidence>=0.16):
            compensate=1.0 if self.eco66_actuator_estimate>=0.0 else -1.0
        issued=intended*compensate
        backup=np.asarray(self.last_control_vectors[s4.CONTROL_MOTOR],dtype=float).copy()
        surface_before=np.asarray(self.surface_flux,dtype=float).copy()
        self.last_control_vectors[s4.CONTROL_MOTOR]=issued
        try:
            super(Formal066ProtoCell,self).apply_effectors(world,dt,config)
        finally:
            self.last_control_vectors[s4.CONTROL_MOTOR]=backup
        # Physical actuator law belongs to the environment. The organism sees
        # only resulting motion, never the sign or switch time.
        sign=float(world._eco66_motor_sign()) if hasattr(world,'_eco66_motor_sign') else 1.0
        motor_delta=np.asarray(self.surface_flux,dtype=float)-surface_before
        self.surface_flux=surface_before+sign*motor_delta
        self.eco66_prev_pos=np.asarray(self.pos,dtype=float).copy()
        self.eco66_prev_issued=np.asarray(issued,dtype=float).copy()

    def split(self,world):
        daughters=super(Formal066ProtoCell,self).split(world)
        if daughters is None:
            return None
        for daughter in daughters:
            daughter.__class__=Formal066ProtoCell
            daughter._init_066_state()
            # Learned actuator state does not cross the generation boundary.
            daughter.eco66_prev_pos=np.asarray(daughter.pos,dtype=float).copy()
            daughter.eco66_prev_issued=np.zeros(2,dtype=float)
            daughter.eco66_actuator_estimate=1.0
            daughter.eco66_estimator_evidence=0.0
            daughter.eco66_grammar_atp=0.0
            daughter.eco66_grammar_wear=0.0
            daughter.eco66_hgt_grammar_acquisitions=0
            if getattr(world.config,'eco66_mutation',False):
                result=_material_mutate_grammar(daughter,world,float(world.config.eco66_mutation_rate))
                if result['accepted']:
                    daughter.eco66_mutation_count+=1
                    world.eco66_mutation_events+=1
                    for k,v in result['events'].items():
                        world.eco66_mutation_breakdown[k]=world.eco66_mutation_breakdown.get(k,0)+int(v)
                world.eco66_mutation_rejections+=int(result['rejected_material'])
            daughter.eco66_prev_modules=grammar_module_set(daughter.genomes[0]) if daughter.genomes else frozenset()
            daughter.eco66_last_hgt_integrations=int(getattr(daughter,'hgt_integrations',0))
        return daughters

    def state_dict(self):
        st=super(Formal066ProtoCell,self).state_dict()
        self._init_066_state()
        st.update({
            'cell_class':'Formal066ProtoCell',
            'eco66_founder_class':self.eco66_founder_class,
            'eco66_prev_pos':self.eco66_prev_pos.copy(),
            'eco66_prev_issued':self.eco66_prev_issued.copy(),
            'eco66_actuator_estimate':self.eco66_actuator_estimate,
            'eco66_estimator_evidence':self.eco66_estimator_evidence,
            'eco66_grammar_atp':self.eco66_grammar_atp,
            'eco66_grammar_wear':self.eco66_grammar_wear,
            'eco66_mutation_count':self.eco66_mutation_count,
            'eco66_hgt_grammar_acquisitions':self.eco66_hgt_grammar_acquisitions,
            'eco66_last_hgt_integrations':self.eco66_last_hgt_integrations,
            'eco66_prev_modules':sorted(int(v) for v in self.eco66_prev_modules),
        })
        return st

    @classmethod
    def from_state(cls,rng,state):
        cell=s65.Formal065ProtoCell.from_state(rng,state); cell.__class__=cls
        cell.eco66_founder_class=str(state.get('eco66_founder_class',FOUNDER_BY_LINEAGE.get(int(cell.lineage),'derived')))
        cell.eco66_prev_pos=np.asarray(state.get('eco66_prev_pos',cell.pos),dtype=float).copy()
        cell.eco66_prev_issued=np.asarray(state.get('eco66_prev_issued',np.zeros(2)),dtype=float).copy()
        cell.eco66_actuator_estimate=float(state.get('eco66_actuator_estimate',1.0))
        cell.eco66_estimator_evidence=float(state.get('eco66_estimator_evidence',0.0))
        cell.eco66_grammar_atp=float(state.get('eco66_grammar_atp',0.0))
        cell.eco66_grammar_wear=float(state.get('eco66_grammar_wear',0.0))
        cell.eco66_mutation_count=int(state.get('eco66_mutation_count',0))
        cell.eco66_hgt_grammar_acquisitions=int(state.get('eco66_hgt_grammar_acquisitions',0))
        cell.eco66_last_hgt_integrations=int(state.get('eco66_last_hgt_integrations',getattr(cell,'hgt_integrations',0)))
        cell.eco66_prev_modules=frozenset(int(v) for v in state.get('eco66_prev_modules',[]))
        if not cell.eco66_prev_modules and cell.genomes:
            cell.eco66_prev_modules=grammar_module_set(cell.genomes[0])
        return cell


class Formal066World(s65.Formal065World):
    def __init__(self,seed=101,initial_cells=3,config=None):
        config=config if config is not None else Formal066Config()
        if not isinstance(config,Formal066Config):
            config=Formal066Config.from_state(config.state_dict())
        self.eco66_mutation_events=0
        self.eco66_mutation_rejections=0
        self.eco66_mutation_breakdown={k:0 for k in ('deletion','duplication','dormancy','reactivation','regulatory')}
        self.eco66_grammar_hgt_acquisitions=0
        self.eco66_chemostat_events=0
        self.eco66_chemostat_material=0.0
        self.eco66_washout_events=0
        self.eco66_next_chemostat=float(config.eco66_chemostat_interval)
        self.eco66_next_washout=float(config.eco66_washout_start)
        self.eco66_founder_classes={}
        self.eco66_rule_switches=0
        self.eco66_last_motor_sign=1.0
        super(Formal066World,self).__init__(seed=seed,initial_cells=initial_cells,config=config)
        self.config=config
        for cell in self.cells:
            cell.__class__=Formal066ProtoCell
            cell._init_066_state()
        if config.shared_ecology and len(self.cells)>=3:
            self._configure_founders()
        if config.eco66_founder_prime:
            self._prime_founders_for_first_division()
        for cell in self.cells:
            self.eco66_founder_classes[int(cell.cell_id)]=str(cell.eco66_founder_class)
        self.initial_total_material=self.total_material()
        self.last_step_material_residual=0.0

    def _replace_genome_initial(self,cell,sequence):
        old=sum(len(g) for g in cell.genomes)*s5.MONOMER_MASS
        cell.genomes=[np.asarray(sequence,dtype=np.uint8).copy()]
        new=len(cell.genomes[0])*s5.MONOMER_MASS
        # This is founder composition, established before the material baseline.
        cell.pools[s5.POOL_NUCLEOTIDE]+=max(0.0,new-old)
        if new<old:
            cell.pools[s5.POOL_NUCLEOTIDE]+=old-new
        cell._refresh_gene_cache(); cell._sync_protein_pool()

    def _configure_founders(self):
        cells=sorted(self.cells,key=lambda c:int(c.cell_id))[:3]
        # Use the same core genome for all three founders.
        base=s65.knockout_grammar(cells[0].genomes[0])
        full=s65.reintroduce_full_grammar(base)
        dormant=s65.reintroduce_full_grammar(base)
        for module in range(s65.GRAMMAR_COUNT):
            dormant=s65.set_grammar_module_promoter(dormant,module,1)
        mode=str(self.config.eco66_founder_mode)
        if mode=='absent_only':
            seqs=(base,base,base); labels=FOUNDER_NAMES
        elif mode=='full_only':
            seqs=(full,full,full); labels=FOUNDER_NAMES
        elif mode=='dormant_only':
            seqs=(dormant,dormant,dormant); labels=FOUNDER_NAMES
        else:
            seqs=(full,dormant,base); labels=FOUNDER_NAMES
        for cell,seq,label in zip(cells,seqs,labels):
            self._replace_genome_initial(cell,seq)
            cell.eco66_founder_class=label
            cell.lineage=labels.index(label)
            cell.eco66_prev_modules=grammar_module_set(cell.genomes[0])

    def _prime_founders_for_first_division(self):
        for cell in self.cells:
            if len(cell.genomes)<2:
                copy_genome=cell.genomes[0].copy()
                cell.genomes.append(copy_genome)
                cell.pools[s5.POOL_NUCLEOTIDE]+=len(copy_genome)*s5.MONOMER_MASS
            cell.age=max(float(cell.age),81.5)
            # Founder-only G2 composition. All added mass is part of the initial
            # condition because the world material baseline is reset afterwards.
            cell.membrane*=1.58
            cell.pools[s5.POOL_FUEL]+=0.42
            cell.pools[s5.POOL_MINERAL]+=0.34
            cell.pools[s5.POOL_ATP]+=0.12
            cell.pools[s5.POOL_MEM_PRECURSOR]+=0.16
            cell.radius=max(cell.radius,0.043)
            cell._refresh_gene_cache(); cell._sync_protein_pool()

    def _eco66_motor_sign(self):
        if self.config.ecology_environment==ENV_STABLE:
            return 1.0
        age=float(self.age)
        first=float(self.config.eco66_rule_first_age)
        if age<first:
            return 1.0
        period=max(float(self.config.eco66_rule_period),1e-9)
        epoch=int(math.floor((age-first)/period))
        return -1.0 if epoch%2==0 else 1.0

    def _chemostat(self):
        if not self.config.eco66_chemostat:
            return
        interval=max(0.1,float(self.config.eco66_chemostat_interval))
        while self.age+1e-12>=self.eco66_next_chemostat:
            # External material enters as ordinary environmental particles.
            center=np.asarray([0.5,0.5])+self.rng.normal(0.0,0.20,2)
            p1=center%1.0; p2pos=(center+self.rng.normal(0.0,0.06,2))%1.0
            fuel=max(0.0,float(self.config.eco66_fuel_rate))*interval
            mineral=max(0.0,float(self.config.eco66_mineral_rate))*interval
            if fuel>0:
                self.field.add_particle(s5.PARTICLE_FUEL,p1,fuel,count_as_injection=True)
            if mineral>0:
                self.field.add_particle(s5.PARTICLE_MINERAL,p2pos,mineral,count_as_injection=True)
            self.eco66_chemostat_material+=fuel+mineral
            self.eco66_chemostat_events+=1
            self.eco66_next_chemostat+=interval

    def _washout(self):
        if not self.config.eco66_washout:
            return
        interval=max(1.0,float(self.config.eco66_washout_interval))
        while self.age+1e-12>=self.eco66_next_washout:
            alive=self.living_cells()
            if len(alive)>3:
                victim=alive[int(self.rng.integers(0,len(alive)))]
                victim.alive=False; victim.death_reason='chemostat_washout'
                # Release and remove in the same ledger step so washout cannot
                # create a one-frame material deficit.
                self._release_dead_cell(victim)
                self.cells=[c for c in self.cells if c is not victim]
                self.eco66_washout_events+=1
            self.eco66_next_washout+=interval

    def _track_hgt_grammar(self):
        for cell in self.living_cells():
            if not isinstance(cell,Formal066ProtoCell):
                cell.__class__=Formal066ProtoCell; cell._init_066_state()
            current=grammar_module_set(cell.genomes[0]) if cell.genomes else frozenset()
            integrations=int(getattr(cell,'hgt_integrations',0))
            if integrations>cell.eco66_last_hgt_integrations:
                gained=current-cell.eco66_prev_modules
                if gained:
                    count=len(gained)
                    cell.eco66_hgt_grammar_acquisitions+=count
                    self.eco66_grammar_hgt_acquisitions+=count
            cell.eco66_prev_modules=current
            cell.eco66_last_hgt_integrations=integrations

    def step(self,dt):
        previous_sign=self._eco66_motor_sign()
        self._chemostat()
        super(Formal066World,self).step(dt)
        # Daughters created by inherited split hooks are recast if required.
        for cell in self.cells:
            if not isinstance(cell,Formal066ProtoCell):
                cell.__class__=Formal066ProtoCell; cell._init_066_state()
        self._track_hgt_grammar()
        self._washout()
        current_sign=self._eco66_motor_sign()
        if current_sign!=previous_sign:
            self.eco66_rule_switches+=1
        self.eco66_last_motor_sign=current_sign
        if not self.finite():
            raise FloatingPointError('non-finite SOMA-CELL 0.6.6 state')

    def _lineage_counts(self):
        counts={name:0 for name in FOUNDER_NAMES}
        for c in self.living_cells():
            name=FOUNDER_BY_LINEAGE.get(int(c.lineage),getattr(c,'eco66_founder_class','derived'))
            if name in counts: counts[name]+=1
        return counts

    def summary(self):
        out=super(Formal066World,self).summary(); living=self.living_cells(); counts=self._lineage_counts()
        traits=[s65.grammar_traits_from_sequence(c.genomes[0]) for c in living if c.genomes]
        absent=[1.0 if t['grammar_gene_count']==0 else 0.0 for t in traits]
        out.update({
            'build':BUILD,'eco66_schema':SCHEMA_VERSION,'eco66_environment':self.config.ecology_environment,
            'eco66_external_fitness_events':0,
            'eco66_max_generation':max([int(c.generation) for c in living],default=0),
            'eco66_full_count':counts[FOUNDER_FULL],'eco66_dormant_count':counts[FOUNDER_DORMANT],'eco66_absent_count':counts[FOUNDER_ABSENT],
            'eco66_mean_active_modules':float(np.mean([t['active_count'] for t in traits])) if traits else 0.0,
            'eco66_grammar_absent_frequency':float(np.mean(absent)) if absent else 0.0,
            'eco66_mutation_events':int(self.eco66_mutation_events),
            'eco66_mutation_rejections':int(self.eco66_mutation_rejections),
            'eco66_grammar_hgt_acquisitions':int(self.eco66_grammar_hgt_acquisitions),
            'eco66_chemostat_events':int(self.eco66_chemostat_events),
            'eco66_chemostat_material':float(self.eco66_chemostat_material),
            'eco66_washout_events':int(self.eco66_washout_events),
            'eco66_motor_sign':float(self._eco66_motor_sign()),
            'eco66_rule_switches':int(self.eco66_rule_switches),
            'eco66_grammar_atp_total':float(sum(getattr(c,'eco66_grammar_atp',0.0) for c in living)),
            'eco66_grammar_wear_total':float(sum(getattr(c,'eco66_grammar_wear',0.0) for c in living)),
        })
        return out

    def state_dict(self):
        st=super(Formal066World,self).state_dict()
        st.update({
            'save_version':SAVE_VERSION,'build':BUILD,'config':self.config.state_dict(),
            'cells':[c.state_dict() if isinstance(c,Formal066ProtoCell) else c.state_dict() for c in self.cells],
            'eco66_mutation_events':self.eco66_mutation_events,'eco66_mutation_rejections':self.eco66_mutation_rejections,
            'eco66_mutation_breakdown':dict(self.eco66_mutation_breakdown),
            'eco66_grammar_hgt_acquisitions':self.eco66_grammar_hgt_acquisitions,
            'eco66_chemostat_events':self.eco66_chemostat_events,'eco66_chemostat_material':self.eco66_chemostat_material,
            'eco66_washout_events':self.eco66_washout_events,'eco66_next_chemostat':self.eco66_next_chemostat,
            'eco66_next_washout':self.eco66_next_washout,'eco66_founder_classes':dict(self.eco66_founder_classes),
            'eco66_rule_switches':self.eco66_rule_switches,'eco66_last_motor_sign':self.eco66_last_motor_sign,
        })
        return st

    @classmethod
    def from_state(cls,state):
        base=dict(state); base['save_version']=s65.SAVE_VERSION; base['build']=s65.BUILD
        allowed=set(s65.Formal065Config().__dict__.keys()); base['config']={k:v for k,v in dict(state.get('config',{})).items() if k in allowed}
        world=s65.Formal065World.from_state(base); world.__class__=cls
        world.config=Formal066Config.from_state(state.get('config',{}))
        for i,item in enumerate(state.get('cells',[])):
            if i<len(world.cells):
                world.cells[i]=Formal066ProtoCell.from_state(world.rng,item)
        world.eco66_mutation_events=int(state.get('eco66_mutation_events',0))
        world.eco66_mutation_rejections=int(state.get('eco66_mutation_rejections',0))
        world.eco66_mutation_breakdown=dict(state.get('eco66_mutation_breakdown',{}))
        world.eco66_grammar_hgt_acquisitions=int(state.get('eco66_grammar_hgt_acquisitions',0))
        world.eco66_chemostat_events=int(state.get('eco66_chemostat_events',0))
        world.eco66_chemostat_material=float(state.get('eco66_chemostat_material',0.0))
        world.eco66_washout_events=int(state.get('eco66_washout_events',0))
        world.eco66_next_chemostat=float(state.get('eco66_next_chemostat',world.age+world.config.eco66_chemostat_interval))
        world.eco66_next_washout=float(state.get('eco66_next_washout',world.age+world.config.eco66_washout_interval))
        world.eco66_founder_classes={int(k):str(v) for k,v in dict(state.get('eco66_founder_classes',{})).items()}
        world.eco66_rule_switches=int(state.get('eco66_rule_switches',0))
        world.eco66_last_motor_sign=float(state.get('eco66_last_motor_sign',1.0))
        # Replacing parent-restored cells via Formal066ProtoCell.from_state consumes RNG.
        # Restore the serialized world RNG *after* all daughter/cell reconstruction so
        # clone/save-load remain bit-for-bit deterministic.
        if 'rng_state' in state:
            world.rng.bit_generator.state=state['rng_state']
        return world

    def clone(self): return Formal066World.from_state(self.state_dict())


def shared_ecology_config(environment=ENV_STABLE,mutation=True,hgt=True,**kwargs):
    return Formal066Config(ecology_environment=environment,eco66_mutation=mutation,eco66_hgt=hgt,**kwargs)


def run_shared_ecology_assay(seed=6601,environment=ENV_STABLE,seconds=130.0,mutation=True,hgt=True,config_overrides=None):
    kw=dict(config_overrides or {})
    cfg=shared_ecology_config(environment=environment,mutation=mutation,hgt=hgt,**kw)
    world=Formal066World(seed=int(seed),initial_cells=3,config=cfg)
    dt=1.0/SIM_HZ; auc=0.0; max_res=0.0
    for _ in range(int(round(float(seconds)*SIM_HZ))):
        world.step(dt)
        alive=world.living_cells()
        if alive:
            auc+=float(np.mean([c.autopoietic_margin() for c in alive]))*dt
        max_res=max(max_res,abs(float(world.matter_ledger_residual())))
    s=world.summary()
    return {
        'seed':int(seed),'environment':str(environment),'seconds':float(seconds),'mutation':int(bool(mutation)),'hgt':int(bool(hgt)),
        'cells':int(s['cells']),'divisions':int(s['divisions']),'deaths':int(s['deaths']),
        'max_generation':int(s['eco66_max_generation']),'full_count':int(s['eco66_full_count']),
        'dormant_count':int(s['eco66_dormant_count']),'absent_count':int(s['eco66_absent_count']),
        'mean_active_modules':float(s['eco66_mean_active_modules']),'grammar_absent_frequency':float(s['eco66_grammar_absent_frequency']),
        'hgt_integrations':int(s.get('hgt_integrations',0)),'grammar_hgt_acquisitions':int(s['eco66_grammar_hgt_acquisitions']),
        'mutation_events':int(s['eco66_mutation_events']),'washout_events':int(s['eco66_washout_events']),
        'margin_auc':float(auc),'material_residual':float(world.matter_ledger_residual()),'max_abs_material_residual':float(max_res),
        'finite':int(bool(world.finite())),'external_fitness_events':0,
    }


LOG_FIELDS=tuple(list(s65.LOG_FIELDS)+[
    'eco66_schema','eco66_environment','eco66_external_fitness_events','eco66_max_generation',
    'eco66_full_count','eco66_dormant_count','eco66_absent_count','eco66_mean_active_modules',
    'eco66_grammar_absent_frequency','eco66_mutation_events','eco66_mutation_rejections',
    'eco66_grammar_hgt_acquisitions','eco66_chemostat_events','eco66_chemostat_material',
    'eco66_washout_events','eco66_motor_sign','eco66_rule_switches','eco66_grammar_atp_total','eco66_grammar_wear_total'])

class LongRunLogger(s65.LongRunLogger):
    def __init__(self,world,path=LOG_FILE,interval=10.0):
        super(LongRunLogger,self).__init__(world,path=path,interval=interval); self.fields=LOG_FIELDS


def generate_report(log_path=LOG_FILE,report_path=REPORT_FILE,session_path=SESSION_FILE):
    return s65.generate_report(log_path=log_path,report_path=report_path,session_path=session_path)

try:
    from scene import run, LANDSCAPE, fill, rect, text

    class SomaCell066Scene(s65.SomaCell065Scene):
        def _fresh_world(self):
            return Formal066World(seed=101,initial_cells=3,config=Formal066Config(ecology_environment=ENV_PERIODIC))
        def setup(self):
            try:
                self.world=Formal066World.load(SAVE_FILE); self.save_status='LOAD'
            except Exception:
                self.world=self._fresh_world(); self.save_status='NEW'
            self.accumulator=0.0; self.last_wall=time.time(); self.last_save_age=self.world.age; self.last_touch_wall=-10.0
            self.paused=False; self.fps=0.0; self.sim_rate=0.0; self.telemetry_wall=time.time(); self.telemetry_age=self.world.age; self.telemetry_frames=0
            self.logger=LongRunLogger(self.world); self.logger.log(self.world,reason='start',force=True); self.report_status='WAIT'
        def draw(self):
            # Draw the compact 0.6.4 base directly, intentionally skipping the
            # extra 0.6.5 line. 0.6.6 reuses the SAME 40-pixel HUD band rather
            # than stacking another panel over the world view.
            s64.SomaCell064Scene.draw(self); s=self.world.summary()
            fill(0.010,0.018,0.030,0.97); rect(0.0,152.0,self.size.w,40.0)
            fill(0.76,0.96,1.0)
            text('0.6.6 {} | gen {} | F/D/A {}/{}/{} | active {:.2f} absent {:.2f}'.format(
                s.get('eco66_environment','?'),s.get('eco66_max_generation',0),s.get('eco66_full_count',0),
                s.get('eco66_dormant_count',0),s.get('eco66_absent_count',0),s.get('eco66_mean_active_modules',0.0),
                s.get('eco66_grammar_absent_frequency',0.0)),x=18,y=178,font_size=8,alignment=4)
            text('div/death {}/{} | mut {} HGT {} grammar-HGT {} | ledger {:+.2e}'.format(
                s.get('divisions',0),s.get('deaths',0),s.get('eco66_mutation_events',0),
                s.get('hgt_integrations',0),s.get('eco66_grammar_hgt_acquisitions',0),s.get('matter_residual',0.0)),
                x=18,y=162,font_size=8,alignment=4)
        def stop(self):
            try: self.world.save(SAVE_FILE); self.save_status='OK'
            except Exception: self.save_status='ERR'
            self.logger.log(self.world,reason='stop',force=True); self.report_status=generate_report()
except ImportError:
    SomaCell066Scene=None


def _run_scene():
    if 'run' not in globals():
        raise RuntimeError('Pythonista scene module is required for interactive mode')
    run(SomaCell066Scene(),orientation=LANDSCAPE,show_fps=False,multi_touch=False)

if __name__=='__main__':
    _run_scene()
