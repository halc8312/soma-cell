import concurrent.futures, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import SOMA_CELL_0_6_2_pythonista as m
profiles=[m.PROFILE_FULL8,m.PROFILE_NO_PREDICTION,m.PROFILE_NO_RECURRENCE,m.PROFILE_NO_DIAGNOSIS,m.PROFILE_NO_PLASTICITY,m.PROFILE_SIZE4,m.PROFILE_SIZE2,m.PROFILE_MINIMAL4]
seeds=[101,202]
def job(ps):
 p,s=ps;c=m.Formal062Config(neural_profile=p,p2_environment=m.p2.P2_ENV_CUE_REVERSAL,p2_switch_age=10.0,p2_plasticity_warmup=5.0,mechanism_probe_min_active_age=6.0)
 return p,s,m.run_headless_trial(seed=s,seconds=20,initial_cells=1,config=c)
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(job,[(p,s) for p in profiles for s in seeds]))
 for p in profiles:
  rs=[r for pp,s,r in rows if pp==p];print(p,'auc',sum(x['margin_auc'] for x in rs)/len(rs),'uptake',sum(x['uptake_delta'] for x in rs)/len(rs),'atp',sum(x['metabolic_module_atp_total'] for x in rs)/len(rs))
 open('/mnt/data/soma062_dev_sweep_fast.json','w').write(json.dumps(rows,indent=2))
