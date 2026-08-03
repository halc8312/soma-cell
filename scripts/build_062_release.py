#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,py_compile,shutil,subprocess,tempfile,zipfile,os
ROOT=Path(__file__).resolve().parents[1]
S62=ROOT/'src/0_6_2'; S61=ROOT/'src/0_6_1'; F06=ROOT/'src/0_6'; P2=ROOT/'src/0_6_p2';P1=ROOT/'src/0_6_p1';P0=ROOT/'src/0_6_p0';BASE=ROOT/'src/baseline'
OUT=ROOT/'releases/SOMA_CELL_0_6_2_GOLD_20260804.zip'; NAME='SOMA_CELL_0_6_2_RELEASE_20260804'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def run(args,cwd,timeout=300):
 p=subprocess.run(args,cwd=cwd,text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError('{}\n{}\n{}'.format(args,p.stdout,p.stderr))
 return p.stdout
def main(out=OUT):
 with tempfile.TemporaryDirectory(prefix='soma062_release_') as td:
  st=Path(td)/NAME;st.mkdir()
  for n in range(1,6):cp(BASE/f'SOMA_CELL_0_{n}_pythonista.py',st/f'SOMA_CELL_0_{n}_pythonista.py')
  for d,n in ((P0,'SOMA_CELL_0_6_P0_pythonista.py'),(P1,'SOMA_CELL_0_6_P1_pythonista.py'),(P2,'SOMA_CELL_0_6_P2_pythonista.py'),(F06,'SOMA_CELL_0_6_pythonista.py'),(S61,'SOMA_CELL_0_6_1_pythonista.py')):cp(d/n,st/n)
  names=['SOMA_CELL_0_6_2_pythonista.py','SOMA_CELL_0_6_2_validation.py','SOMA_CELL_0_6_2_experiment.py','SOMA_CELL_0_6_2_START_HERE.txt','SOMA_CELL_0_6_2_README_JA.md','SOMA_CELL_0_6_2_FORMAL_CONTRACT.md','SOMA_CELL_0_6_2_FORMAL_SCHEMA.json','SOMA_CELL_0_6_2_VALIDATION_RESULTS.txt','SOMA_CELL_0_6_2_REGRESSION_RESULTS.txt','SOMA_CELL_0_6_2_EXPERIMENT_REPORT.txt','SOMA_CELL_0_6_2_MANIFEST.txt','soma_cell_0_6_2_validation.csv','soma_cell_0_6_2_regression.csv','soma_cell_0_6_2_experiment_results.csv','soma_cell_0_6_2_experiment_summary.csv','soma_cell_0_6_2_experiment_consolidated.json']
  for n in names:cp(S62/n,st/n)
  for n in ('SOMA_CELL_0_6_2_DEVELOPMENT_PREREGISTRATION.json','SOMA_CELL_0_6_2_R2_PREREGISTRATION.json','SOMA_CELL_0_6_2_R3_PREREGISTRATION.json','SOMA_CELL_0_6_2_R4_RELEASE_RECORD.json'):cp(ROOT/'results'/n,st/n)
  for d,n in ((P0,'SOMA_CELL_0_6_P0_validation.py'),(P1,'SOMA_CELL_0_6_P1_validation.py'),(P2,'SOMA_CELL_0_6_P2_validation.py'),(F06,'SOMA_CELL_0_6_validation.py'),(S61,'SOMA_CELL_0_6_1_validation.py')):cp(d/n,st/n)
  for n in ('SOMA_CELL_0_6_INTEGRATION_CONTRACT.md','SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md','SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md','SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md','SOMA_CELL_0_6_FORMAL_CONTRACT.md','SOMA_CELL_0_6_1_FORMAL_CONTRACT.md'):cp(ROOT/'docs'/n,st/n)
  assert 'Passed: 24/24' in (st/'SOMA_CELL_0_6_2_VALIDATION_RESULTS.txt').read_text()
  assert 'Inherited total             : 127/127 PASS' in (st/'SOMA_CELL_0_6_2_REGRESSION_RESULTS.txt').read_text()
  for p in st.glob('*.py'):py_compile.compile(str(p),doraise=True)
  quick=['build_schema_and_selection_metadata','ui1_inheritance_and_safe_reset_source','unmetered_fails_closed','profile_active_counts_and_no_tissue','prediction_meter_and_ablation','recurrence_meter_and_ablation','inactive_cells_retired_and_material_returned','from_state_clone_and_derived_config_field','save_restore_exact','preregistered_confirmation_negative_neural_net_value','stable_safety_no_false_lease','full8_fault_conservation_positive','finite_and_material_residual']
  lines=[]
  for q in quick:
   o=run(['python3','SOMA_CELL_0_6_2_validation.py','--test',q],st,timeout=180);assert '"pass": 1' in o;lines.append(q+': PASS')
  o=run(['python3','-c',"import SOMA_CELL_0_6_2_pythonista as s; assert s.BUILD=='SOMA-CELL 0.6.2'; assert s.SCHEMA_VERSION=='0.6.2-M1.1'; assert s.RESEARCH_DEFAULT_PROFILE=='efficient2'; print('FLAT IMPORT PASS')"],st,timeout=60);lines.append(o.strip())
  (st/'flat_release_verification.txt').write_text('Canonical measured checks: 24/24 dedicated + 127/127 inherited = 151/151 PASS\n'+'\n'.join(lines)+'\n',encoding='utf-8')
  shutil.rmtree(st/'__pycache__',ignore_errors=True)
  files=sorted(p for p in st.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
  (st/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  ./{p.name}' for p in files)+'\n',encoding='utf-8')
  out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.zip.tmp');tmp.unlink(missing_ok=True)
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for p in sorted(st.rglob('*')):
    if p.is_file():z.write(p,Path(NAME)/p.relative_to(st))
  tmp.replace(out)
 with zipfile.ZipFile(out) as z:
  if z.testzip():raise RuntimeError(z.testzip())
 print(out);print(sha(out))
if __name__=='__main__':main()
