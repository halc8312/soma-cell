# coding: utf-8
"""SOMA-CELL 0.6.2 — metabolic neural audit and minimal-brain profiles.

Freezes SOMA-CELL 0.6.1 (including UI1) and compares prediction,
recurrence, plasticity, diagnosis, and neural compartment count under explicit
ATP/material metering. Inactive compartments are not developed; when a loaded
8-cell tissue is reduced, retired matter is returned through the P0 body port.
"""
from __future__ import division
import csv, gc, math, os, pickle, sys, time, uuid
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
for rel in ('.','../0_6_1','../0_6','../0_6_p2','../0_6_p1','../0_6_p0','../baseline'):
    p=os.path.abspath(os.path.join(HERE,rel))
    if p not in sys.path: sys.path.insert(0,p)
import SOMA_CELL_0_6_1_pythonista as s61
f06=s61.f06; p2=s61.p2; p1=s61.p1; p0=s61.p0; s5=s61.s5; s4=s61.s4
BUILD='SOMA-CELL 0.6.2'; BUILD_LONG=BUILD+' | metabolic neural audit / minimal brain'
SCHEMA_VERSION='0.6.2-M1.1'; SAVE_VERSION=1
SAVE_FILE='soma_cell_0_6_2.pkl'; LOG_FILE='soma_cell_0_6_2_longrun.csv'
REPORT_FILE='soma_cell_0_6_2_report.txt'; SESSION_FILE='soma_cell_0_6_2_sessions.csv'
AUTO_SAVE_INTERVAL=60.0; SIM_HZ=s61.SIM_HZ; clamp=s61.clamp; _atomic_pickle=s61._atomic_pickle

PROFILE_FULL8='full8'; PROFILE_NO_PREDICTION='no_prediction'; PROFILE_NO_RECURRENCE='no_recurrence'
PROFILE_NO_DIAGNOSIS='no_diagnosis'; PROFILE_NO_PLASTICITY='no_plasticity'
PROFILE_SIZE4='size4'; PROFILE_SIZE2='size2'; PROFILE_SIZE1='size1'; PROFILE_NO_TISSUE='no_tissue'
PROFILE_MINIMAL4='minimal4'; PROFILE_EFFICIENT4='efficient4'; PROFILE_EFFICIENT2='efficient2'
AUDIT_SELECTED_PROFILE=PROFILE_NO_TISSUE
MINIMAL_NEURAL_PROFILE=PROFILE_EFFICIENT2
RESEARCH_DEFAULT_PROFILE=PROFILE_EFFICIENT2
PROFILES=frozenset((PROFILE_FULL8,PROFILE_NO_PREDICTION,PROFILE_NO_RECURRENCE,
 PROFILE_NO_DIAGNOSIS,PROFILE_NO_PLASTICITY,PROFILE_SIZE4,PROFILE_SIZE2,
 PROFILE_SIZE1,PROFILE_NO_TISSUE,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2))
MODULE_PREDICTION='prediction'; MODULE_RECURRENCE='recurrence'; MODULE_PLASTICITY='plasticity'
MODULES=(MODULE_PREDICTION,MODULE_RECURRENCE,MODULE_PLASTICITY)

def profile_active_count(profile):
    return {PROFILE_SIZE1:1,PROFILE_SIZE2:2,PROFILE_EFFICIENT2:2,PROFILE_SIZE4:4,PROFILE_MINIMAL4:4,PROFILE_EFFICIENT4:4,PROFILE_NO_TISSUE:0}.get(str(profile),8)

def even_active_indices(count):
    count=int(max(0,min(p2.P2_CELL_COUNT,count)))
    if count<=0:return np.zeros(0,dtype=np.int64)
    if count>=p2.P2_CELL_COUNT:return np.arange(p2.P2_CELL_COUNT,dtype=np.int64)
    return np.asarray([int(round(i*p2.P2_CELL_COUNT/float(count)))%p2.P2_CELL_COUNT for i in range(count)],dtype=np.int64)

class Formal062Config(s61.Formal061Config):
    def __init__(self,neural_profile=PROFILE_FULL8,metabolic_meter_enabled=True,
                 prediction_meter_atp_rate=1.0e-5,prediction_meter_wear_rate=3.5e-7,
                 recurrence_meter_atp_rate=3.0e-6,recurrence_meter_wear_rate=1.2e-7,
                 plasticity_meter_atp_rate=8.0e-6,plasticity_meter_wear_rate=2.2e-7,
                 module_meter_interval=0.5,
                 tissue_dormancy_return_escrow=True,tissue_dormancy_return_committed=True,
                 external_test_harness=False,**kwargs):
        profile=str(neural_profile)
        if profile not in PROFILES: raise ValueError('unknown profile '+profile)
        if profile==PROFILE_NO_TISSUE:
            kwargs.setdefault('p2_tissue_mode',p2.P2_MODE_NONE); kwargs.setdefault('p2_install_genes',False)
        if profile in (PROFILE_NO_PREDICTION,PROFILE_MINIMAL4): kwargs.setdefault('p2_prediction',False)
        if profile in (PROFILE_NO_RECURRENCE,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2): kwargs.setdefault('p2_recurrence',False)
        if profile==PROFILE_NO_PLASTICITY: kwargs.setdefault('p2_plasticity',False)
        if profile in (PROFILE_NO_DIAGNOSIS,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2):
            kwargs.setdefault('diagnosis_mode',s61.DIAGNOSIS_OFF); kwargs.setdefault('diagnosis_enabled',False)
        super(Formal062Config,self).__init__(**kwargs)
        if not bool(metabolic_meter_enabled) and not bool(external_test_harness):
            raise ValueError('unmetered 0.6.2 modules are allowed only in an explicit external test harness')
        self.neural_profile=profile; self.audit_active_cells=profile_active_count(profile)
        self.metabolic_meter_enabled=bool(metabolic_meter_enabled); self.external_test_harness=bool(external_test_harness)
        self.prediction_meter_atp_rate=float(prediction_meter_atp_rate); self.prediction_meter_wear_rate=float(prediction_meter_wear_rate)
        self.recurrence_meter_atp_rate=float(recurrence_meter_atp_rate); self.recurrence_meter_wear_rate=float(recurrence_meter_wear_rate)
        self.plasticity_meter_atp_rate=float(plasticity_meter_atp_rate); self.plasticity_meter_wear_rate=float(plasticity_meter_wear_rate)
        self.module_meter_interval=max(0.05,float(module_meter_interval))
        self.tissue_dormancy_return_escrow=bool(tissue_dormancy_return_escrow)
        self.tissue_dormancy_return_committed=bool(tissue_dormancy_return_committed)
    @classmethod
    def from_state(cls,state):
        # audit_active_cells is a derived field stored for telemetry, not a
        # constructor argument.  Never leak it down into the 0.5 config chain.
        state=dict(state or {})
        state.pop('audit_active_cells',None)
        allowed=set(cls().__dict__.keys())
        allowed.discard('audit_active_cells')
        return cls(**{k:v for k,v in state.items() if k in allowed})

class NeuralModuleLedger(object):
    def __init__(self):
        self.atp={m:0.0 for m in MODULES}; self.material={m:0.0 for m in MODULES}; self.steps={m:0 for m in MODULES}
        self.retired_cells=0; self.returned_material=0.0; self.returned_atp=0.0; self.profile_switches=0; self.last_profile=PROFILE_FULL8
    def note(self,module,atp=0.0,material=0.0,steps=1):
        if module in self.atp:
            self.atp[module]+=max(0.0,float(atp)); self.material[module]+=max(0.0,float(material)); self.steps[module]+=int(max(0,steps))
    def total_atp(self): return float(sum(self.atp.values()))
    def total_material(self): return float(sum(self.material.values()))
    def finite(self): return bool(np.all(np.isfinite(list(self.atp.values())+list(self.material.values())+[self.returned_material,self.returned_atp])))
    def state_dict(self): return {'atp':dict(self.atp),'material':dict(self.material),'steps':dict(self.steps),'retired_cells':self.retired_cells,'returned_material':self.returned_material,'returned_atp':self.returned_atp,'profile_switches':self.profile_switches,'last_profile':self.last_profile}
    @classmethod
    def from_state(cls,state):
        o=cls(); state=dict(state or {})
        for name in MODULES:
            o.atp[name]=float(state.get('atp',{}).get(name,0.0)); o.material[name]=float(state.get('material',{}).get(name,0.0)); o.steps[name]=int(state.get('steps',{}).get(name,0))
        o.retired_cells=int(state.get('retired_cells',0)); o.returned_material=float(state.get('returned_material',0.0)); o.returned_atp=float(state.get('returned_atp',0.0)); o.profile_switches=int(state.get('profile_switches',0)); o.last_profile=str(state.get('last_profile',PROFILE_FULL8)); return o

class MetabolicAuditTissue(s61.ActiveDiagnosticMaterialTissue):
    def __init__(self,gene_parameters,mode=p2.P2_MODE_FULL,rng_seed=0,formal_enabled=False,profile=PROFILE_FULL8):
        super(MetabolicAuditTissue,self).__init__(gene_parameters,mode=mode,rng_seed=rng_seed,formal_enabled=formal_enabled)
        self.module_ledger=NeuralModuleLedger(); self.audit_profile=str(profile); self.audit_active_count=profile_active_count(profile)
        self.audit_active_mask=np.zeros(p2.P2_CELL_COUNT,dtype=bool); self.audit_active_mask[even_active_indices(self.audit_active_count)]=True
        self.audit_retired_mask=np.zeros(p2.P2_CELL_COUNT,dtype=bool); self.audit_module_payments=0
        self.module_meter_elapsed=0.0; self.module_meter_due_dt=0.0; self.module_plasticity_pending=0
    def configure_profile(self,profile):
        profile=str(profile)
        if profile not in PROFILES: raise ValueError(profile)
        if profile!=self.audit_profile:self.module_ledger.profile_switches+=1
        self.audit_profile=profile; self.audit_active_count=profile_active_count(profile); self.audit_active_mask[:]=False; self.audit_active_mask[even_active_indices(self.audit_active_count)]=True; self.module_ledger.last_profile=profile
    def _flags(self,config):
        pred=bool(config.p2_prediction) and self.audit_profile not in (PROFILE_NO_PREDICTION,PROFILE_MINIMAL4)
        rec=bool(config.p2_recurrence) and self.audit_profile not in (PROFILE_NO_RECURRENCE,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2)
        plast=bool(config.p2_plasticity) and self.audit_profile!=PROFILE_NO_PLASTICITY
        diag=bool(config.diagnosis_enabled and config.diagnosis_mode!=s61.DIAGNOSIS_OFF and self.audit_profile not in (PROFILE_NO_DIAGNOSIS,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2))
        return pred,rec,plast,diag
    def masked_gene_activity(self,gene_activity):
        a=np.asarray(gene_activity,dtype=float).copy()
        if a.size>=p2.P2_CELL_COUNT:a[~self.audit_active_mask]=0.0
        return a
    def _retire_inactive(self,port,config):
        if self.audit_active_count>=p2.P2_CELL_COUNT:return
        for i in range(p2.P2_CELL_COUNT):
            if self.audit_active_mask[i] or self.audit_retired_mask[i]:continue
            tid=self.tissue_ids[i]
            try: status=port.attachment_status(tid)
            except KeyError:
                self.present[i]=False; self.mature[i]=False; self.audit_retired_mask[i]=True; continue
            stores=np.asarray(status['stores'],dtype=float); material=np.asarray(status['tissue_material'],dtype=float)
            ratp=0.0; rmat=0.0
            if config.tissue_dormancy_return_escrow and np.any(stores>1e-14):
                ret=port.return_unused_budget(tid); ratp+=float(ret.get('atp',0.0)); rmat+=sum(float(ret.get(n,0.0)) for n in ('protein','membrane','signal'))
            if config.tissue_dormancy_return_committed and np.any(material>1e-14):
                ret=port.return_dead_tissue(tid,reason='0.6.2-metabolic-retirement'); ratp+=float(ret.get('atp_dissipated',0.0)); rmat+=float(ret.get('material',0.0))
            else:
                try: port.detach(tid,return_unused=True)
                except Exception: pass
            self.present[i]=False; self.mature[i]=False; self.development[i]=0.0; self.hidden[i]=0.0; self.prev_hidden[i]=0.0; self.audit_retired_mask[i]=True
            self.module_ledger.retired_cells+=1; self.module_ledger.returned_atp+=ratp; self.module_ledger.returned_material+=rmat
    def _pay_meter(self,port,module,atp_rate,wear_rate,intensity,dt):
        active=np.flatnonzero(self.audit_active_mask)
        if active.size==0 or intensity<=0.0:return
        atp_total=max(0.0,float(atp_rate)*float(intensity)*dt); wear_total=max(0.0,float(wear_rate)*float(intensity)*dt)
        paid=0.0; worn=0.0
        for i in active:
            tid=self.tissue_ids[int(i)]
            try: port.attachment_status(tid)
            except KeyError: continue
            atp=atp_total/active.size; wear=wear_total/active.size
            assembly_atp=wear*float(p0.ASSEMBLY_ATP_PER_PROTEIN)
            port.allocate_budget(tid,{'atp':atp+assembly_atp,'protein':wear,'membrane':0.0,'signal':0.0},dt)
            state=port._attachment(tid)
            atp_paid,_=port._spend_energy_and_signal(state,atp,0.0)
            st=port.attachment_status(tid); stores=np.asarray(st['stores'],dtype=float)
            built=port.commit_material(tid,protein=min(wear,float(stores[p0.BUDGET_PROTEIN])),damaged_fraction=0.78,aggregate_fraction=0.12)
            paid+=float(atp_paid)+float(built.get('atp_spent',0.0)); worn+=float(built.get('damaged_protein',0.0)+built.get('aggregate',0.0))
        self.module_ledger.note(module,paid,worn); self.audit_module_payments+=1
    def _mask_backup(self):
        inactive=~self.audit_active_mask
        if not np.any(inactive):return None
        names=('w_sensor','w_rec','bias','homeo_gain','motor_gain','resource_lease')
        b={n:getattr(self,n).copy() for n in names}
        self.hidden[inactive]=0.0; self.prev_hidden[inactive]=0.0; self.w_sensor[inactive,:]=0.0; self.w_rec[inactive,:]=0.0; self.w_rec[:,inactive]=0.0; self.bias[inactive]=0.0; self.homeo_gain[inactive]=0.0; self.motor_gain[inactive]=0.0; self.resource_lease[inactive]=0.0
        return b
    def _restore_mask(self,b):
        if b is not None:
            for n,v in b.items():setattr(self,n,v)
    def pre_step(self,port,dt,config,gene_activity):
        self.configure_profile(getattr(config,'neural_profile',self.audit_profile)); self._retire_inactive(port,config)
        activity=self.masked_gene_activity(gene_activity); pred,rec,plast,diag=self._flags(config)
        old=(config.p2_prediction,config.p2_recurrence,config.p2_plasticity,config.diagnosis_enabled,config.diagnosis_mode)
        config.p2_prediction=pred; config.p2_recurrence=rec; config.p2_plasticity=plast
        if not diag:config.diagnosis_enabled=False; config.diagnosis_mode=s61.DIAGNOSIS_OFF
        backup=self._mask_backup()
        try: report=super(MetabolicAuditTissue,self).pre_step(port,dt,config,activity)
        finally:
            self._restore_mask(backup); config.p2_prediction,config.p2_recurrence,config.p2_plasticity,config.diagnosis_enabled,config.diagnosis_mode=old
        self.module_meter_due_dt=0.0
        if config.metabolic_meter_enabled:
            self.module_meter_elapsed+=max(0.0,float(dt))
            if self.module_meter_elapsed+1e-12>=config.module_meter_interval:
                paid_dt=self.module_meter_elapsed; self.module_meter_elapsed=0.0; self.module_meter_due_dt=paid_dt
                n=max(1,int(np.count_nonzero(self.audit_active_mask)))
                if pred:self._pay_meter(port,MODULE_PREDICTION,config.prediction_meter_atp_rate,config.prediction_meter_wear_rate,n,paid_dt)
                if rec:
                    e=int(np.count_nonzero(self.rec_mask & self.audit_active_mask[:,None] & self.audit_active_mask[None,:])); self._pay_meter(port,MODULE_RECURRENCE,config.recurrence_meter_atp_rate,config.recurrence_meter_wear_rate,max(1,e),paid_dt)
        return report
    def post_step(self,port,dt,config):
        pred,rec,plast,diag=self._flags(config); old=(config.p2_prediction,config.p2_recurrence,config.p2_plasticity,config.diagnosis_enabled,config.diagnosis_mode)
        config.p2_prediction=pred; config.p2_recurrence=rec; config.p2_plasticity=plast
        if not diag:config.diagnosis_enabled=False; config.diagnosis_mode=s61.DIAGNOSIS_OFF
        backup=self._mask_backup(); before=int(self.plasticity_updates); inactive=~self.audit_active_mask
        snap=None
        if np.any(inactive):snap={'w_sensor':self.w_sensor[inactive].copy(),'w_rec_rows':self.w_rec[inactive].copy(),'w_rec_cols':self.w_rec[:,inactive].copy(),'bias':self.bias[inactive].copy(),'motor_gain':self.motor_gain[inactive].copy(),'predict_w':self.predict_w[inactive].copy(),'maturity':self.maturity[inactive].copy(),'reopen':self.reopen_reserve[inactive].copy()}
        try: out=super(MetabolicAuditTissue,self).post_step(port,dt,config)
        finally:
            self._restore_mask(backup)
            if snap is not None:
                self.w_sensor[inactive]=snap['w_sensor']; self.w_rec[inactive]=snap['w_rec_rows']; self.w_rec[:,inactive]=snap['w_rec_cols']; self.bias[inactive]=snap['bias']; self.motor_gain[inactive]=snap['motor_gain']; self.predict_w[inactive]=snap['predict_w']; self.maturity[inactive]=snap['maturity']; self.reopen_reserve[inactive]=snap['reopen']
            config.p2_prediction,config.p2_recurrence,config.p2_plasticity,config.diagnosis_enabled,config.diagnosis_mode=old
        updates=max(0,int(self.plasticity_updates)-before); self.module_plasticity_pending+=updates
        if config.metabolic_meter_enabled and plast and self.module_meter_due_dt>0.0 and self.module_plasticity_pending>0:
            self._pay_meter(port,MODULE_PLASTICITY,config.plasticity_meter_atp_rate,config.plasticity_meter_wear_rate,self.module_plasticity_pending,self.module_meter_due_dt)
            self.module_plasticity_pending=0
        self.module_meter_due_dt=0.0
        return out
    def finite(self):return bool(super(MetabolicAuditTissue,self).finite() and self.module_ledger.finite())
    def state_dict(self):
        st=super(MetabolicAuditTissue,self).state_dict(); st.update({'audit_profile':self.audit_profile,'audit_active_count':self.audit_active_count,'audit_active_mask':self.audit_active_mask.copy(),'audit_retired_mask':self.audit_retired_mask.copy(),'module_ledger':self.module_ledger.state_dict(),'audit_module_payments':self.audit_module_payments,'module_meter_elapsed':self.module_meter_elapsed,'module_meter_due_dt':self.module_meter_due_dt,'module_plasticity_pending':self.module_plasticity_pending}); return st
    @classmethod
    def from_state(cls,state):
        o=s61.ActiveDiagnosticMaterialTissue.from_state(state); o.__class__=cls; o.audit_profile=str(state.get('audit_profile',PROFILE_FULL8)); o.audit_active_count=int(state.get('audit_active_count',profile_active_count(o.audit_profile))); o.audit_active_mask=np.asarray(state.get('audit_active_mask',np.isin(np.arange(p2.P2_CELL_COUNT),even_active_indices(o.audit_active_count))),dtype=bool).copy(); o.audit_retired_mask=np.asarray(state.get('audit_retired_mask',np.zeros(p2.P2_CELL_COUNT,dtype=bool)),dtype=bool).copy(); o.module_ledger=NeuralModuleLedger.from_state(state.get('module_ledger',{})); o.audit_module_payments=int(state.get('audit_module_payments',0)); o.module_meter_elapsed=float(state.get('module_meter_elapsed',0.0)); o.module_meter_due_dt=float(state.get('module_meter_due_dt',0.0)); o.module_plasticity_pending=int(state.get('module_plasticity_pending',0)); return o

class Formal062ProtoCell(s61.Formal061ProtoCell):
    @classmethod
    def from_state(cls,rng,state):
        cell=s61.Formal061ProtoCell.from_state(rng,state); cell.__class__=cls
        if state.get('p2_tissue') is not None:cell.p2_tissue=MetabolicAuditTissue.from_state(state['p2_tissue'])
        return cell

class Formal062World(s61.Formal061World):
    def __init__(self,seed=101,initial_cells=1,config=None):
        config=config if config is not None else Formal062Config()
        if not isinstance(config,Formal062Config):config=Formal062Config(**config.state_dict())
        super(Formal062World,self).__init__(seed=seed,initial_cells=initial_cells,config=config); self.config=config; self.metabolic_world_steps=0
        for cell in self.cells:
            cell.__class__=Formal062ProtoCell
            if cell.p2_tissue is not None and not isinstance(cell.p2_tissue,MetabolicAuditTissue):cell.p2_tissue=MetabolicAuditTissue.from_state(cell.p2_tissue.state_dict()); cell.p2_tissue.configure_profile(config.neural_profile)
        self._ensure_all_p2_tissues(); self.initial_total_material=self.total_material(); self.last_step_material_residual=0.0
    def _new_p2_tissue(self,cell):
        if self.config.neural_profile==PROFILE_NO_TISSUE or self.config.p2_tissue_mode==p2.P2_MODE_NONE:return None
        params=p2.p2_gene_parameters(cell); activity=p2.p2_gene_activity(cell)
        if params is None:return None
        mask=np.zeros(p2.P2_CELL_COUNT,dtype=bool); mask[even_active_indices(self.config.audit_active_cells)]=True
        ma=np.asarray(activity,dtype=float).copy(); ma[~mask]=0.0
        if np.count_nonzero(ma>=0.016)<max(1,np.count_nonzero(mask)):return None
        seed=((self.p2_seed*1000003)^(int(cell.cell_id)*9176)^(int(cell.generation)*7919)^0x061D1A)&0xffffffff
        tissue=MetabolicAuditTissue(params,mode=self.config.p2_tissue_mode,rng_seed=seed,formal_enabled=f06.formal_controller_activity(cell)>=0.016,profile=self.config.neural_profile)
        tissue.ensure_attachments(self.port_for(cell.cell_id),ma); cell.p2_tissue=tissue; cell.p2_tissue_births+=1; self.p2_tissue_creations+=1; self.formal_tissue_creations+=1; return tissue
    def _ensure_all_p2_tissues(self):
        super(Formal062World,self)._ensure_all_p2_tissues()
        for cell in self.living_cells():
            if not isinstance(cell,Formal062ProtoCell):cell.__class__=Formal062ProtoCell
            t=getattr(cell,'p2_tissue',None)
            if t is not None and not isinstance(t,MetabolicAuditTissue):cell.p2_tissue=MetabolicAuditTissue.from_state(t.state_dict())
            if isinstance(getattr(cell,'p2_tissue',None),MetabolicAuditTissue):cell.p2_tissue.configure_profile(self.config.neural_profile)
    def step(self,dt):super(Formal062World,self).step(dt); self.metabolic_world_steps+=1
    def finite(self):return bool(super(Formal062World,self).finite() and all(t.finite() for t in (getattr(c,'p2_tissue',None) for c in self.cells) if isinstance(t,MetabolicAuditTissue)))
    def summary(self):
        out=super(Formal062World,self).summary(); ts=[c.p2_tissue for c in self.living_cells() if isinstance(getattr(c,'p2_tissue',None),MetabolicAuditTissue)]
        total=lambda field,module:float(sum(getattr(t.module_ledger,field).get(module,0.0) for t in ts))
        out.update({'build':BUILD,'metabolic_schema':SCHEMA_VERSION,'metabolic_profile':self.config.neural_profile,'metabolic_active_cells':int(sum(np.count_nonzero(t.audit_active_mask) for t in ts)),'metabolic_retired_cells':int(sum(t.module_ledger.retired_cells for t in ts)),'metabolic_returned_material':float(sum(t.module_ledger.returned_material for t in ts)),'metabolic_returned_atp':float(sum(t.module_ledger.returned_atp for t in ts)),'metabolic_prediction_atp':total('atp',MODULE_PREDICTION),'metabolic_recurrence_atp':total('atp',MODULE_RECURRENCE),'metabolic_plasticity_atp':total('atp',MODULE_PLASTICITY),'metabolic_prediction_wear':total('material',MODULE_PREDICTION),'metabolic_recurrence_wear':total('material',MODULE_RECURRENCE),'metabolic_plasticity_wear':total('material',MODULE_PLASTICITY),'metabolic_module_atp_total':float(sum(t.module_ledger.total_atp() for t in ts)),'metabolic_module_material_total':float(sum(t.module_ledger.total_material() for t in ts)),'metabolic_world_steps':self.metabolic_world_steps,'metabolic_audit_selected_profile':AUDIT_SELECTED_PROFILE,'metabolic_minimal_neural_profile':MINIMAL_NEURAL_PROFILE}); return out
    def state_dict(self):
        st=super(Formal062World,self).state_dict(); st.update({'save_version':SAVE_VERSION,'build':BUILD,'config':self.config.state_dict(),'cells':[c.state_dict() for c in self.cells],'metabolic_world_steps':self.metabolic_world_steps}); return st
    @classmethod
    def from_state(cls,state):
        base=dict(state); base['save_version']=s61.SAVE_VERSION; base['build']=s61.BUILD; allowed=set(s61.Formal061Config().__dict__.keys()); base['config']={k:v for k,v in dict(state.get('config',{})).items() if k in allowed}
        w=s61.Formal061World.from_state(base); w.__class__=cls; w.config=Formal062Config.from_state(state.get('config',{})); w.cells=[Formal062ProtoCell.from_state(w.rng,item) for item in state['cells']]; w.rng.bit_generator.state=state['rng_state']; w.metabolic_world_steps=int(state.get('metabolic_world_steps',0)); return w
    def clone(self):return Formal062World.from_state(self.state_dict())

def set_neural_profile(world,profile):
    if profile not in PROFILES:raise ValueError(profile)
    world.config.neural_profile=profile; world.config.audit_active_cells=profile_active_count(profile); world.config.p2_prediction=profile not in (PROFILE_NO_PREDICTION,PROFILE_MINIMAL4); world.config.p2_recurrence=profile not in (PROFILE_NO_RECURRENCE,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2); world.config.p2_plasticity=profile!=PROFILE_NO_PLASTICITY
    if profile in (PROFILE_NO_DIAGNOSIS,PROFILE_MINIMAL4,PROFILE_EFFICIENT4,PROFILE_EFFICIENT2):world.config.diagnosis_mode=s61.DIAGNOSIS_OFF; world.config.diagnosis_enabled=False
    for c in world.living_cells():
        if isinstance(getattr(c,'p2_tissue',None),MetabolicAuditTissue):c.p2_tissue.configure_profile(profile)
    return world

def apply_062_common_disturbance_tape(world,seed,step,stream=0):return s61.apply_061_common_disturbance_tape(world,seed,step,stream=stream)

def run_headless_trial(seed=101,seconds=120.0,initial_cells=1,config=None):
    w=Formal062World(seed=seed,initial_cells=initial_cells,config=config or Formal062Config()); dt=1.0/SIM_HZ; auc=0.0; start=float(w.p2_reward_uptake_total); n=0
    for step in range(int(round(float(seconds)*SIM_HZ))):
        if not w.living_cells():break
        w.step(dt); living=w.living_cells(); m=float(np.mean([c.autopoietic_margin() for c in living])) if living else 0.0; auc+=m*dt; n+=1
    out=w.summary(); out.update({'seed':seed,'seconds':seconds,'margin_auc':auc,'uptake_delta':float(w.p2_reward_uptake_total-start),'living_steps':n,'finite':w.finite(),'material_residual':float(w.matter_ledger_residual())}); return out

LOG_FIELDS=tuple(list(s61.LOG_FIELDS)+['metabolic_profile','metabolic_active_cells','metabolic_retired_cells','metabolic_returned_material','metabolic_returned_atp','metabolic_prediction_atp','metabolic_recurrence_atp','metabolic_plasticity_atp','metabolic_prediction_wear','metabolic_recurrence_wear','metabolic_plasticity_wear','metabolic_module_atp_total','metabolic_module_material_total'])
class LongRunLogger(object):
    def __init__(self,world,path=LOG_FILE):self.path=path; self.session_id='{}-{}'.format(int(time.time()),uuid.uuid4().hex[:8]); self.last_age=-1e9; self.status='WAIT'
    def log(self,world,reason='periodic',force=False):
        if not force and world.age-self.last_age<10.0:return False
        s=world.summary(); row={k:s.get(k,'') for k in LOG_FIELDS}; row.update({'session_id':self.session_id,'reason':reason,'wall_time':time.time()}); exists=os.path.exists(self.path) and os.path.getsize(self.path)>0
        with open(self.path,'a',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=('session_id','reason','wall_time')+LOG_FIELDS)
            if not exists:w.writeheader()
            w.writerow(row)
        self.last_age=world.age; self.status='OK'; return True

def generate_report(log_path=LOG_FILE,report_path=REPORT_FILE,session_path=SESSION_FILE):
    if not os.path.exists(log_path):return 'NO LOG'
    with open(log_path,'r',newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
    sessions={}
    for r in rows:sessions.setdefault(r['session_id'],[]).append(r)
    with open(session_path,'w',newline='',encoding='utf-8') as h:
        f=('session_id','rows','final_age','final_cells','profile','module_atp'); w=csv.DictWriter(h,fieldnames=f); w.writeheader()
        for sid,items in sessions.items():
            last=items[-1]; w.writerow({'session_id':sid,'rows':len(items),'final_age':last.get('age',''),'final_cells':last.get('cells',''),'profile':last.get('metabolic_profile',''),'module_atp':last.get('metabolic_module_atp_total','')})
    lines=[BUILD_LONG,'sessions: {}'.format(len(sessions)),'']
    for sid,items in sessions.items():
        last=items[-1]; lines.append('{} age={} cells={} profile={} active={} moduleATP={} ledger={}'.format(sid,last.get('age',''),last.get('cells',''),last.get('metabolic_profile',''),last.get('metabolic_active_cells',''),last.get('metabolic_module_atp_total',''),last.get('matter_residual','')))
    with open(report_path,'w',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')
    return 'OK'

try:
    from scene import Scene,run,LANDSCAPE,background,fill,rect,text,ellipse,line,stroke,stroke_weight
    class SomaCell062Scene(s61.SomaCell061Scene):
        def setup(self):
            background(0.006,0.012,0.022)
            try:self.world=Formal062World.load(SAVE_FILE); self.save_status='LOAD'
            except Exception:self.world=Formal062World(seed=101,initial_cells=2,config=Formal062Config(neural_profile=RESEARCH_DEFAULT_PROFILE,p2_environment=p2.P2_ENV_CUE_REVERSAL)); self.save_status='NEW'
            self.accumulator=0.0; self.last_wall=time.time(); self.last_save_age=self.world.age; self.last_touch_wall=-10.0; self.paused=False; self.fps=0.0; self.sim_rate=0.0; self.telemetry_wall=time.time(); self.telemetry_age=self.world.age; self.telemetry_frames=0; self.logger=LongRunLogger(self.world); self.logger.log(self.world,reason='start',force=True); self.report_status='WAIT'
        def _fresh_world(self):return Formal062World(seed=101,initial_cells=2,config=Formal062Config(neural_profile=RESEARCH_DEFAULT_PROFILE,p2_environment=p2.P2_ENV_CUE_REVERSAL))
        def draw(self):
            super(SomaCell062Scene,self).draw(); s=self.world.summary()
            fill(0.010,0.018,0.030,0.97); rect(0.0,self.size.h-52.0,self.size.w,52.0)
            fill(0.92,0.98,1.0); text(BUILD,x=max(120.0,min(180.0,self.size.w*0.18)),y=self.size.h-26,font_size=18,alignment=4)
            fill(0.64,0.78,0.86); text('{} | SAVE {} | {:.1f} fps | x{:.2f}'.format(s.get('p2_environment','native'),self.save_status,self.fps,self.sim_rate),x=self.size.w-18,y=self.size.h-26,font_size=9,alignment=6)
            fill(0.010,0.018,0.030,0.94); rect(0.0,128.0,self.size.w,24.0); fill(0.92,0.87,1.0)
            text('0.6.2 run {} | audit winner {} | active {} retired {} module ATP {:.6f} P/R/L {:.5f}/{:.5f}/{:.5f}'.format(s.get('metabolic_profile','n/a'),s.get('metabolic_audit_selected_profile','n/a'),s.get('metabolic_active_cells',0),s.get('metabolic_retired_cells',0),s.get('metabolic_module_atp_total',0.0),s.get('metabolic_prediction_atp',0.0),s.get('metabolic_recurrence_atp',0.0),s.get('metabolic_plasticity_atp',0.0)),x=18,y=140,font_size=9,alignment=4)
        def stop(self):
            try:self.world.save(SAVE_FILE); self.save_status='OK'
            except Exception:self.save_status='ERR'
            self.logger.log(self.world,reason='stop',force=True); self.report_status=generate_report()
except ImportError:Scene=None

if __name__=='__main__':
    if Scene is None:print(run_headless_trial(seed=101,seconds=90.0,initial_cells=1,config=Formal062Config(neural_profile=RESEARCH_DEFAULT_PROFILE,p2_environment=p2.P2_ENV_CUE_REVERSAL)))
    else:run(SomaCell062Scene(),LANDSCAPE,show_fps=False)
