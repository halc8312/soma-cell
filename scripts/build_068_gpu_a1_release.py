#!/usr/bin/env python3
from pathlib import Path
import hashlib, os, py_compile, shutil, subprocess, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/0_6_8'
OUT=ROOT/'releases/SOMA_CELL_0_6_8_GPU_A1_CHECKPOINT_20260808.zip'
NAME='SOMA_CELL_0_6_8_GPU_A1_RELEASE_20260808'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def main():
    with tempfile.TemporaryDirectory(prefix='soma068gpu_') as td:
        stage=Path(td)/NAME;stage.mkdir()
        for n in range(1,6):
            cp(ROOT/'src/baseline'/f'SOMA_CELL_0_{n}_pythonista.py',stage/f'SOMA_CELL_0_{n}_pythonista.py')
        deps=[
            ('0_6_p0','SOMA_CELL_0_6_P0_pythonista.py'),('0_6_p1','SOMA_CELL_0_6_P1_pythonista.py'),
            ('0_6_p2','SOMA_CELL_0_6_P2_pythonista.py'),('0_6','SOMA_CELL_0_6_pythonista.py'),
            ('0_6_1','SOMA_CELL_0_6_1_pythonista.py'),('0_6_2','SOMA_CELL_0_6_2_pythonista.py'),
            ('0_6_3','SOMA_CELL_0_6_3_pythonista.py'),('0_6_4','SOMA_CELL_0_6_4_pythonista.py'),
            ('0_6_5','SOMA_CELL_0_6_5_pythonista.py'),('0_6_6','SOMA_CELL_0_6_6_pythonista.py'),
            ('0_6_7','SOMA_CELL_0_6_7_pythonista.py')]
        for d,n in deps:cp(ROOT/'src'/d/n,stage/n)
        own=[
            'SOMA_CELL_0_6_8_gpu.py','SOMA_CELL_0_6_8_validation.py',
            'SOMA_CELL_0_6_8_gpu_benchmark.py','SOMA_CELL_0_6_8_install_check.py',
            'SOMA_CELL_0_6_8_pythonista.py','SOMA_CELL_0_6_8_GPU_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_README_JA.md','SOMA_CELL_0_6_8_GPU_ENVIRONMENT_REPORT.json',
            'soma_cell_0_6_8_gpu_benchmark_cpu.json','soma_cell_0_6_8_gpu_validation.csv']
        for n in own:cp(SRC/n,stage/n)
        cp(ROOT/'docs/SOMA_CELL_0_6_8_GPU_CONTRACT.md',stage/'SOMA_CELL_0_6_8_GPU_CONTRACT.md')
        cp(ROOT/'docs/SOMA_CELL_0_6_8_GPU_SCHEMA.json',stage/'SOMA_CELL_0_6_8_GPU_SCHEMA.json')
        for n in ['SOMA_CELL_0_6_8_GPU_VALIDATION_RESULTS.txt','SOMA_CELL_0_6_8_GPU_REGRESSION_RESULTS.txt','SOMA_CELL_0_6_8_GPU_EXPERIMENT_REPORT.txt']:
            cp(ROOT/'results'/n,stage/n)
        (stage/'CUDA_INSTALL_NOTE.txt').write_text(
            'Install a CUDA-enabled PyTorch build appropriate for the NVIDIA driver.\n'
            'Run SOMA_CELL_0_6_8_install_check.py; cuda_available must be true.\n'
            'Do not infer CUDA support from the CPU-only development report.\n',encoding='utf-8')
        for p in stage.glob('*.py'):py_compile.compile(str(p),doraise=True)
        env=dict(os.environ);env['PYTHONPATH']=str(stage)
        subprocess.check_call(['python3','-c',"import SOMA_CELL_0_6_8_gpu as g; assert g.BUILD=='SOMA-CELL 0.6.8-GPU'; assert g.SCHEMA_VERSION=='0.6.8-GPU-A1.0'"],cwd=stage,env=env)
        subprocess.check_call(['python3','SOMA_CELL_0_6_8_validation.py'],cwd=stage,env=env)
        shutil.rmtree(stage/'__pycache__',ignore_errors=True)
        files=sorted(p for p in stage.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
        (stage/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  ./{p.name}' for p in files)+'\n',encoding='utf-8')
        OUT.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.write(p,Path(NAME)/p.relative_to(stage))
    with zipfile.ZipFile(OUT) as z:assert z.testzip() is None
    print(OUT);print(sha(OUT))

if __name__=='__main__':main()
