#!/usr/bin/env python3
from pathlib import Path
import hashlib,py_compile,shutil,tempfile,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1]; S=ROOT/'src/0_6_6'; OUT=ROOT/'releases/SOMA_CELL_0_6_6_GOLD_20260808.zip'; NAME='SOMA_CELL_0_6_6_RELEASE_20260808'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def cp(a,b): b.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a,b)
def main():
 with tempfile.TemporaryDirectory(prefix='soma066rel_') as td:
  stage=Path(td)/NAME; stage.mkdir()
  for n in range(1,6): cp(ROOT/'src/baseline'/f'SOMA_CELL_0_{n}_pythonista.py',stage/f'SOMA_CELL_0_{n}_pythonista.py')
  deps=[('0_6_p0','SOMA_CELL_0_6_P0_pythonista.py'),('0_6_p1','SOMA_CELL_0_6_P1_pythonista.py'),('0_6_p2','SOMA_CELL_0_6_P2_pythonista.py'),('0_6','SOMA_CELL_0_6_pythonista.py'),('0_6_1','SOMA_CELL_0_6_1_pythonista.py'),('0_6_2','SOMA_CELL_0_6_2_pythonista.py'),('0_6_3','SOMA_CELL_0_6_3_pythonista.py'),('0_6_4','SOMA_CELL_0_6_4_pythonista.py'),('0_6_5','SOMA_CELL_0_6_5_pythonista.py')]
  for d,n in deps: cp(ROOT/'src'/d/n,stage/n)
  for n in ['SOMA_CELL_0_6_6_pythonista.py','SOMA_CELL_0_6_6_validation.py','SOMA_CELL_0_6_6_START_HERE.txt','SOMA_CELL_0_6_6_README_JA.md','SOMA_CELL_0_6_6_VALIDATION_RESULTS.txt','SOMA_CELL_0_6_6_REGRESSION_RESULTS.txt','SOMA_CELL_0_6_6_EXPERIMENT_REPORT.txt','soma_cell_0_6_6_validation.csv','soma_cell_0_6_6_experiment_summary.csv','soma_cell_0_6_6_experiment_consolidated.json']:
   cp(S/n,stage/n)
  for n in ['SOMA_CELL_0_6_6_FORMAL_CONTRACT.md','SOMA_CELL_0_6_6_FORMAL_SCHEMA.json']: cp(ROOT/'docs'/n,stage/n)
  cp(ROOT/'results/SOMA_CELL_0_6_6_PREREGISTRATION.json',stage/'SOMA_CELL_0_6_6_PREREGISTRATION.json')
  for p in stage.glob('*.py'): py_compile.compile(str(p),doraise=True)
  subprocess.check_call(['python3','-c',"import SOMA_CELL_0_6_6_pythonista as s; assert s.BUILD=='SOMA-CELL 0.6.6'; assert s.SCHEMA_VERSION=='0.6.6-ECO1.0'"],cwd=stage)
  shutil.rmtree(stage/'__pycache__',ignore_errors=True)
  files=sorted(p for p in stage.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt'); (stage/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  ./{p.name}' for p in files)+'\n')
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for p in sorted(stage.rglob('*')):
    if p.is_file(): z.write(p,Path(NAME)/p.relative_to(stage))
 with zipfile.ZipFile(OUT) as z: assert z.testzip() is None
 print(OUT); print(sha(OUT))
if __name__=='__main__': main()
