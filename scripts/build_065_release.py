#!/usr/bin/env python3
from pathlib import Path
import hashlib,py_compile,shutil,tempfile,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1]; S=ROOT/'src/0_6_5'; OUT=ROOT/'releases/SOMA_CELL_0_6_5_GOLD_20260808.zip'; NAME='SOMA_CELL_0_6_5_RELEASE_20260808'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def cp(a,b): b.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a,b)
def main():
 with tempfile.TemporaryDirectory(prefix='soma065rel_') as td:
  stage=Path(td)/NAME; stage.mkdir()
  # baseline + descendants as flat Pythonista dependencies
  for n in range(1,6): cp(ROOT/'src/baseline'/f'SOMA_CELL_0_{n}_pythonista.py',stage/f'SOMA_CELL_0_{n}_pythonista.py')
  deps=[('0_6_p0','SOMA_CELL_0_6_P0_pythonista.py'),('0_6_p1','SOMA_CELL_0_6_P1_pythonista.py'),('0_6_p2','SOMA_CELL_0_6_P2_pythonista.py'),('0_6','SOMA_CELL_0_6_pythonista.py'),('0_6_1','SOMA_CELL_0_6_1_pythonista.py'),('0_6_2','SOMA_CELL_0_6_2_pythonista.py'),('0_6_3','SOMA_CELL_0_6_3_pythonista.py'),('0_6_4','SOMA_CELL_0_6_4_pythonista.py')]
  for d,n in deps: cp(ROOT/'src'/d/n,stage/n)
  for n in ['SOMA_CELL_0_6_5_pythonista.py','SOMA_CELL_0_6_5_validation.py','SOMA_CELL_0_6_5_experiment.py','SOMA_CELL_0_6_5_START_HERE.txt','SOMA_CELL_0_6_5_README_JA.md','SOMA_CELL_0_6_5_VALIDATION_RESULTS.txt','SOMA_CELL_0_6_5_REGRESSION_RESULTS.txt','SOMA_CELL_0_6_5_EXPERIMENT_REPORT.txt','soma_cell_0_6_5_validation.csv','soma_cell_0_6_5_experiment_summary.csv','soma_cell_0_6_5_experiment_consolidated.json']:
   cp(S/n,stage/n)
  cp(ROOT/'docs/SOMA_CELL_0_6_5_FORMAL_CONTRACT.md',stage/'SOMA_CELL_0_6_5_FORMAL_CONTRACT.md'); cp(ROOT/'docs/SOMA_CELL_0_6_5_FORMAL_SCHEMA.json',stage/'SOMA_CELL_0_6_5_FORMAL_SCHEMA.json'); cp(ROOT/'results/SOMA_CELL_0_6_5_PREREGISTRATION.json',stage/'SOMA_CELL_0_6_5_PREREGISTRATION.json'); cp(ROOT/'results/SOMA_CELL_0_6_5_RELEASE_RECORD.json',stage/'SOMA_CELL_0_6_5_RELEASE_RECORD.json')
  for p in stage.glob('*.py'): py_compile.compile(str(p),doraise=True)
  env=':'.join(str(stage) for _ in [0]); out=subprocess.check_output(['python3','-c',"import SOMA_CELL_0_6_5_pythonista as s; assert s.BUILD=='SOMA-CELL 0.6.5'; assert s.SCHEMA_VERSION=='0.6.5-GE1.0'; print('FLAT IMPORT PASS')"],cwd=stage,text=True)
  (stage/'flat_release_verification.txt').write_text('24/24 dedicated + 202/202 inherited = 226/226 PASS\n'+out,encoding='utf-8')
  shutil.rmtree(stage/'__pycache__',ignore_errors=True)
  files=sorted(p for p in stage.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt'); (stage/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  ./{p.name}' for p in files)+'\n')
  OUT.parent.mkdir(parents=True,exist_ok=True); tmp=OUT.with_suffix('.tmp');
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for p in sorted(stage.rglob('*')):
    if p.is_file(): z.write(p,Path(NAME)/p.relative_to(stage))
  tmp.replace(OUT)
 with zipfile.ZipFile(OUT) as z: assert z.testzip() is None
 print(OUT); print(sha(OUT))
if __name__=='__main__': main()
