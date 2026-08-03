# coding: utf-8
"""Preregistered R2 holdout harness for SOMA-CELL 0.6.1-D1.4.

The primary assay is a body-centred microfluidic fault experiment.  It holds
local fuel/mineral availability constant while a neutral directional cue keeps
the motor tissue active.  Thus the primary endpoint tests whether a confirmed
actuator-fault conservation lease can recover its *material cost* without
confounding the result with spatial foraging geometry.

A separate native-spatial assay is exploratory and is never used to rescue a
failed primary gate.
"""
from __future__ import print_function
import argparse, concurrent.futures, csv, hashlib, json, math, os, sys, time, traceback
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
import SOMA_CELL_0_6_1_pythonista as soma
import SOMA_CELL_0_6_1_R2_development as r2
import SOMA_CELL_0_6_1_development as native

PREREG=os.path.join(HERE,'SOMA_CELL_0_6_1_R2_PREREGISTRATION.json')
RESULTS=os.path.join(HERE,'soma_cell_0_6_1_r2_holdout_results.csv')
SUMMARY=os.path.join(HERE,'soma_cell_0_6_1_r2_holdout_summary.json')
REPORT=os.path.join(HERE,'SOMA_CELL_0_6_1_R2_HOLDOUT_REPORT.txt')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_prereg():
    with open(PREREG,'r',encoding='utf-8') as f:data=json.load(f)
    expected=data['frozen_hashes']['model_source']
    actual=sha256(os.path.join(HERE,'SOMA_CELL_0_6_1_pythonista.py'))
    if actual!=expected:raise RuntimeError('frozen model hash mismatch: {} != {}'.format(actual,expected))
    return data


def _stable(seed):
    started=time.time();row=r2.run_tethered_stable_single(int(seed));row['wall']=time.time()-started;return row


def _fault(seed):
    started=time.time();row=r2.run_tethered_fault_twin(int(seed),True,True);row['kind']='fault_tethered';row['wall']=time.time()-started;return row


def _native(seed):
    started=time.time();row=native.fault_twin(int(seed),False);row['kind']='fault_native_exploratory';row['wall']=time.time()-started;return row


def _error(kind,seed,exc):
    return {'kind':kind,'seed':int(seed),'error':'{}: {}'.format(type(exc).__name__,exc),'traceback':traceback.format_exc()[-2000:]}


def run_job(kind,seed):
    try:
        if kind=='stable_tethered':return _stable(seed)
        if kind=='fault_tethered':return _fault(seed)
        if kind=='fault_native_exploratory':return _native(seed)
        raise ValueError(kind)
    except Exception as exc:return _error(kind,seed,exc)


def write_rows(rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    temp=RESULTS+'.tmp'
    with open(temp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,RESULTS)


def read_rows():
    if not os.path.exists(RESULTS):return []
    with open(RESULTS,'r',newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def _num(row,key,default=0.0):
    try:return float(row.get(key,default) or default)
    except Exception:return float(default)


def summarize(rows,prereg):
    stable=[r for r in rows if r.get('kind')=='stable_tethered']
    fault=[r for r in rows if r.get('kind')=='fault_tethered']
    native_rows=[r for r in rows if r.get('kind')=='fault_native_exploratory']
    errors=[r for r in rows if r.get('error')]
    issued=[r for r in fault if _num(r,'lease_t')>0]
    unissued=[r for r in fault if _num(r,'lease_t')<=0]
    residuals=[]
    for r in rows:
        for k in ('residual','residual_t','residual_c'):
            if r.get(k) not in (None,''):residuals.append(abs(_num(r,k)))
    positive=sum(_num(r,'auc_diff')>0 for r in issued)
    required_positive=int(math.ceil(prereg['acceptance']['fault_positive_auc_fraction']*max(len(issued),1)))
    summary={
        'build':soma.BUILD,'schema':soma.SCHEMA_VERSION,
        'source_sha256':sha256(os.path.join(HERE,'SOMA_CELL_0_6_1_pythonista.py')),
        'stable_n':len(stable),'stable_false_lease':sum(_num(r,'lease')>0 for r in stable),
        'stable_false_feedback':sum(_num(r,'feedback')>0 for r in stable),
        'stable_nonfinite':sum(_num(r,'finite')<0.5 for r in stable),
        'fault_n':len(fault),'fault_confirmed':sum(_num(r,'confirmed_t')>0 for r in fault),
        'fault_issued':len(issued),'fault_positive_auc':positive,
        'fault_mean_auc_issued':float(np.mean([_num(r,'auc_diff') for r in issued])) if issued else 0.0,
        'fault_mean_motor_atp_diff':float(np.mean([_num(r,'motor_atp_diff') for r in issued])) if issued else 0.0,
        'fault_mean_dissipated_diff':float(np.mean([_num(r,'dissipated_energy_diff') for r in issued])) if issued else 0.0,
        'fault_mean_uptake_diff':float(np.mean([_num(r,'uptake_diff') for r in issued])) if issued else 0.0,
        'fault_unissued_max_abs_auc':max([abs(_num(r,'auc_diff')) for r in unissued] or [0.0]),
        'native_n':len(native_rows),'native_issued':sum(_num(r,'lease_t')>0 for r in native_rows),
        'native_positive_auc':sum(_num(r,'auc_diff')>0 for r in native_rows),
        'native_mean_auc':float(np.mean([_num(r,'auc_diff') for r in native_rows])) if native_rows else 0.0,
        'max_abs_material_residual':max(residuals or [0.0]),'errors':len(errors),
    }
    a=prereg['acceptance']
    checks={
        'all_jobs_completed':len(stable)==len(prereg['seeds']['stable']) and len(fault)==len(prereg['seeds']['fault']) and len(native_rows)==len(prereg['seeds']['native_exploratory']) and not errors,
        'stable_false_lease':summary['stable_false_lease']<=a['stable_max_false_lease'],
        'stable_false_feedback':summary['stable_false_feedback']<=a['stable_max_false_feedback'],
        'stable_finite':summary['stable_nonfinite']==0,
        'fault_confirmed':summary['fault_confirmed']>=a['fault_min_confirmed'],
        'fault_issued':summary['fault_issued']>=a['fault_min_issued'],
        'fault_positive_fraction':positive>=required_positive,
        'fault_mean_auc':summary['fault_mean_auc_issued']>a['fault_min_mean_auc'],
        'fault_motor_cost_reduced':summary['fault_mean_motor_atp_diff']<a['fault_max_mean_motor_atp_diff'],
        'fault_dissipation_reduced':summary['fault_mean_dissipated_diff']<a['fault_max_mean_dissipated_diff'],
        'unissued_identical':summary['fault_unissued_max_abs_auc']<=a['unissued_max_abs_auc'],
        'material_residual':summary['max_abs_material_residual']<=a['max_abs_material_residual'],
    }
    summary['acceptance']=checks;summary['all_acceptance_pass']=bool(all(checks.values()))
    with open(SUMMARY,'w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,sort_keys=True)
    lines=['SOMA-CELL 0.6.1 R2 PREREGISTERED HOLDOUT REPORT',
           'Build: {} | Schema: {}'.format(soma.BUILD,soma.SCHEMA_VERSION),
           'Primary assay: body-centred microfluidic actuator-fault conservation',
           'Native spatial assay: exploratory only','',json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True),'']
    with open(REPORT,'w',encoding='utf-8') as f:f.write('\n'.join(lines))
    return summary


def run(max_workers=4,resume=True):
    prereg=load_prereg();existing=read_rows() if resume else []
    done={(r.get('kind'),int(float(r.get('seed',-1)))) for r in existing if not r.get('error')}
    jobs=[]
    for kind,key in (('stable_tethered','stable'),('fault_tethered','fault'),('fault_native_exploratory','native_exploratory')):
        for seed in prereg['seeds'][key]:
            if (kind,int(seed)) not in done:jobs.append((kind,int(seed)))
    rows=list(existing)
    print('R2 holdout jobs remaining:',len(jobs),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1,int(max_workers))) as ex:
        future_map={ex.submit(run_job,kind,seed):(kind,seed) for kind,seed in jobs}
        for future in concurrent.futures.as_completed(future_map):
            kind,seed=future_map[future];row=future.result();rows.append(row);write_rows(rows)
            print('[{}] {} seed={}{}'.format(len(rows),kind,seed,' ERROR' if row.get('error') else ''),flush=True)
    summary=summarize(rows,prereg);print(json.dumps(summary,sort_keys=True),flush=True);return summary


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--fresh',action='store_true');ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args(argv)
    prereg=load_prereg()
    if args.summarize:
        print(json.dumps(summarize(read_rows(),prereg),sort_keys=True));return 0
    summary=run(args.workers,not args.fresh);return 0 if summary['all_acceptance_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
