# coding: utf-8
from __future__ import print_function
import csv,json,os,subprocess,sys,time
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
import SOMA_CELL_0_6_7_pythonista as s

SEEDS=(6731,6732,6733)
SECONDS=480.0
TASKS=[]
for env,hgt,mutation,label in (
    (s.ENV_STABLE,True,True,'stable_hgt_on'),
    (s.ENV_STABLE,False,True,'stable_hgt_off'),
    (s.ENV_LONG_DELAY,True,True,'long_delay_hgt_on'),
    (s.ENV_LONG_DELAY,False,True,'long_delay_hgt_off'),
):
    for seed in SEEDS:TASKS.append((label,seed,env,hgt,mutation))

def one(task):
    label,seed,env,hgt,mutation=task;start=time.time()
    row=s.run_long_horizon_assay(seed=seed,environment=env,seconds=SECONDS,mutation=mutation,hgt=hgt)
    row['condition']=label;row['wall_seconds']=time.time()-start;return row

def child(task):
    label,seed,env,hgt,mutation=task
    args=[sys.executable,os.path.abspath(__file__),'--single',label,str(seed),env,'1' if hgt else '0','1' if mutation else '0']
    cp=subprocess.run(args,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    lines=[x for x in cp.stdout.splitlines() if x.strip()]
    if not lines:raise RuntimeError(cp.stderr)
    return json.loads(lines[-1])

def main():
    rows=[]
    for task in TASKS:
        row=child(task);rows.append(row)
        print(row['condition'],row['seed'],'gen',row['max_generation'],'active',round(row['mean_active_modules'],3),'absent',round(row['grammar_absent_frequency'],3),'hgt-reentry',row['hgt_reentries'],flush=True)
    rows.sort(key=lambda r:([x[0] for x in TASKS].index(r['condition']) if False else (r['condition'],r['seed'])))
    with open('soma_cell_0_6_7_r3_experiment_results.json','w',encoding='utf-8') as h:json.dump(rows,h,indent=2,sort_keys=True)
    scalar=[k for k,v in rows[0].items() if k!='history' and isinstance(v,(int,float,str,bool,type(None)))]
    with open('soma_cell_0_6_7_r3_experiment_results.csv','w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=scalar);w.writeheader();w.writerows([{k:r.get(k) for k in scalar} for r in rows])
    print('DONE',len(rows))

if __name__=='__main__':
    if len(sys.argv)>=2 and sys.argv[1]=='--single':
        _,_,label,seed,env,hgt,mutation=sys.argv
        print(json.dumps(one((label,int(seed),env,bool(int(hgt)),bool(int(mutation)))),sort_keys=True))
    else:main()
