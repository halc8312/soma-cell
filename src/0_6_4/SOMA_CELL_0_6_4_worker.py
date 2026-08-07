# coding: utf-8
import json, os, sys, traceback
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
import SOMA_CELL_0_6_4_experiment as exp

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value
spec_path,out_path=sys.argv[1],sys.argv[2]
spec=json.load(open(spec_path,encoding='utf-8'))
try:
    row=clean(exp.trial(spec))
except Exception as exc:
    row={'kind':spec.get('kind',''),'condition':spec.get('condition',''),'policy':spec.get('policy',''),'seed':spec.get('seed',''),'seconds':spec.get('seconds',''),'option_fraction':spec.get('option_fraction',''),'task_label':spec.get('task',{}).get('task_label',''),'error':'{}: {}'.format(type(exc).__name__,exc),'traceback':traceback.format_exc()[-2200:]}
with open(out_path,'w',encoding='utf-8') as h:json.dump(row,h,ensure_ascii=False,sort_keys=True)
print(json.dumps({k:row.get(k) for k in ('kind','seed','condition','margin_auc','uptake_delta','neurogenesis_developments','error')},ensure_ascii=False,sort_keys=True))
