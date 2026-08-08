#!/usr/bin/env python3
"""Build and verify the SOMA-CELL 0.6.8-GPU A2 flat engineering release."""
from pathlib import Path
import hashlib
import os
import py_compile
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/0_6_8'
OUT = ROOT / 'releases/SOMA_CELL_0_6_8_GPU_A2_CHECKPOINT_20260808.zip'
NAME = 'SOMA_CELL_0_6_8_GPU_A2_RELEASE_20260808'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    with tempfile.TemporaryDirectory(prefix='soma068gpu_a2_') as td:
        stage = Path(td) / NAME
        stage.mkdir()

        # Full-detail material lineage required by the frozen 0.6.6 reference.
        for number in range(1, 6):
            cp(
                ROOT / 'src/baseline' / ('SOMA_CELL_0_{}_pythonista.py'.format(number)),
                stage / ('SOMA_CELL_0_{}_pythonista.py'.format(number)),
            )
        dependencies = [
            ('0_6_p0', 'SOMA_CELL_0_6_P0_pythonista.py'),
            ('0_6_p1', 'SOMA_CELL_0_6_P1_pythonista.py'),
            ('0_6_p2', 'SOMA_CELL_0_6_P2_pythonista.py'),
            ('0_6', 'SOMA_CELL_0_6_pythonista.py'),
            ('0_6_1', 'SOMA_CELL_0_6_1_pythonista.py'),
            ('0_6_2', 'SOMA_CELL_0_6_2_pythonista.py'),
            ('0_6_3', 'SOMA_CELL_0_6_3_pythonista.py'),
            ('0_6_4', 'SOMA_CELL_0_6_4_pythonista.py'),
            ('0_6_5', 'SOMA_CELL_0_6_5_pythonista.py'),
            ('0_6_6', 'SOMA_CELL_0_6_6_pythonista.py'),
            ('0_6_7', 'SOMA_CELL_0_6_7_pythonista.py'),
        ]
        for directory, filename in dependencies:
            cp(ROOT / 'src' / directory / filename, stage / filename)

        # A1 is the frozen tensor/RNG/diffusion foundation; A2 extends it.
        own_files = [
            'SOMA_CELL_0_6_8_gpu.py',
            'SOMA_CELL_0_6_8_validation.py',
            'SOMA_CELL_0_6_8_gpu_benchmark.py',
            'SOMA_CELL_0_6_8_install_check.py',
            'SOMA_CELL_0_6_8_pythonista.py',
            'SOMA_CELL_0_6_8_GPU_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_README_JA.md',
            'SOMA_CELL_0_6_8_GPU_ENVIRONMENT_REPORT.json',
            'soma_cell_0_6_8_gpu_benchmark_cpu.json',
            'soma_cell_0_6_8_gpu_validation.csv',
            'SOMA_CELL_0_6_8_gpu_a2.py',
            'SOMA_CELL_0_6_8_A2_validation.py',
            'SOMA_CELL_0_6_8_A2_benchmark.py',
            'SOMA_CELL_0_6_8_GPU_A2_pythonista.py',
            'SOMA_CELL_0_6_8_GPU_A2_START_HERE.txt',
            'SOMA_CELL_0_6_8_GPU_A2_README_JA.md',
            'SOMA_CELL_0_6_8_GPU_A2_ENVIRONMENT_REPORT.json',
            'soma_cell_0_6_8_gpu_a2_benchmark_cpu.json',
            'soma_cell_0_6_8_gpu_a2_validation.csv',
        ]
        for filename in own_files:
            cp(SRC / filename, stage / filename)

        for filename in (
            'SOMA_CELL_0_6_8_GPU_CONTRACT.md',
            'SOMA_CELL_0_6_8_GPU_SCHEMA.json',
            'SOMA_CELL_0_6_8_GPU_A2_CONTRACT.md',
            'SOMA_CELL_0_6_8_GPU_A2_SCHEMA.json',
        ):
            cp(ROOT / 'docs' / filename, stage / filename)
        for filename in (
            'SOMA_CELL_0_6_8_GPU_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_8_GPU_A2_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A2_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_8_GPU_A2_EXPERIMENT_REPORT.txt',
        ):
            cp(ROOT / 'results' / filename, stage / filename)

        (stage / 'CUDA_INSTALL_NOTE.txt').write_text(
            'Install a CUDA-enabled PyTorch build appropriate for the NVIDIA driver.\n'
            'Run SOMA_CELL_0_6_8_install_check.py; cuda_available must be true.\n'
            'Then run A2 validation and benchmark with --device cuda.\n'
            'The development environment used a CPU-only PyTorch build, so RTX/CUDA speed is NOT yet measured.\n',
            encoding='utf-8',
        )

        for path in stage.glob('*.py'):
            py_compile.compile(str(path), doraise=True)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(stage)
        subprocess.check_call(
            [
                'python3', '-c',
                "import SOMA_CELL_0_6_8_gpu_a2 as g; "
                "assert g.BUILD=='SOMA-CELL 0.6.8-GPU A2'; "
                "assert g.SCHEMA_VERSION=='0.6.8-GPU-A2.0'; "
                "assert g.PORT_STATUS['division']=='cpu-authoritative'",
            ],
            cwd=stage,
            env=env,
        )
        subprocess.check_call(['python3', 'SOMA_CELL_0_6_8_validation.py'], cwd=stage, env=env)
        subprocess.check_call(['python3', 'SOMA_CELL_0_6_8_A2_validation.py'], cwd=stage, env=env)
        subprocess.check_call(
            [
                'python3', 'SOMA_CELL_0_6_8_A2_benchmark.py',
                '--device', 'cpu', '--precision', 'float64', '--steps', '3',
                '--output', 'flat_release_a2_smoke_benchmark.json',
            ],
            cwd=stage,
            env=env,
        )
        shutil.rmtree(stage / '__pycache__', ignore_errors=True)
        files = sorted(path for path in stage.iterdir() if path.is_file() and path.name != 'SHA256SUMS.txt')
        (stage / 'SHA256SUMS.txt').write_text(
            '\n'.join('{}  ./{}'.format(sha(path), path.name) for path in files) + '\n',
            encoding='utf-8',
        )

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    archive.write(path, Path(NAME) / path.relative_to(stage))

    with zipfile.ZipFile(OUT) as archive:
        assert archive.testzip() is None
    print(OUT)
    print(sha(OUT))


if __name__ == '__main__':
    main()
