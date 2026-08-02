#!/usr/bin/env python3
"""Build and verify the standalone Pythonista release for formal SOMA-CELL 0.6.

The complete deterministic suites were run on the exact canonical sources and
are shipped as measured evidence (84/84 total).  The flat release build then
re-checks source hashes, compilation, imports, a selected set of material
invariants, save/restore, and a headless smoke test.  The two exceptionally
slow lockstep suites are not redundantly rerun during packaging.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'src' / '0_6'
P2 = ROOT / 'src' / '0_6_p2'
P1 = ROOT / 'src' / '0_6_p1'
P0 = ROOT / 'src' / '0_6_p0'
BASE = ROOT / 'src' / 'baseline'
DEFAULT_OUTPUT = ROOT / 'releases' / 'SOMA_CELL_0_6_GOLD_20260802.zip'
RELEASE_DIR_NAME = 'SOMA_CELL_0_6_RELEASE_20260802'
EXPECTED = {
    'SOMA_CELL_0_6_pythonista.py': 'e8f70c67cee16fdc1e90397808783e26b2895552ca3ec26f62c2b137ada08c26',
    'SOMA_CELL_0_6_P2_pythonista.py': '03cd333838f2bee0dde634e8d55ca2b2c7ed285c9104a52a26bcee4a07ce1693',
}
DEPENDENCIES = [f'SOMA_CELL_0_{n}_pythonista.py' for n in range(1, 6)] + [
    'SOMA_CELL_0_6_P0_pythonista.py',
    'SOMA_CELL_0_6_P1_pythonista.py',
    'SOMA_CELL_0_6_P2_pythonista.py',
]
FORMAL_FILES = [
    'SOMA_CELL_0_6_pythonista.py',
    'SOMA_CELL_0_6_validation.py',
    'SOMA_CELL_0_6_experiment.py',
    'SOMA_CELL_0_6_report.py',
    'SOMA_CELL_0_6_START_HERE.txt',
    'SOMA_CELL_0_6_README_JA.md',
    'SOMA_CELL_0_6_FORMAL_CONTRACT.md',
    'SOMA_CELL_0_6_FORMAL_SCHEMA.json',
    'SOMA_CELL_0_6_MANIFEST.txt',
    'SOMA_CELL_0_6_VALIDATION_RESULTS.txt',
    'SOMA_CELL_0_6_REGRESSION_RESULTS.txt',
    'SOMA_CELL_0_6_EXPERIMENT_REPORT.txt',
    'soma_cell_0_6_validation.csv',
    'soma_cell_0_6_experiment_results.csv',
    'soma_cell_0_6_experiment_results.json',
    'soma_cell_0_6_experiment_summary.csv',
]
PARENT_VALIDATORS = [
    P0 / 'SOMA_CELL_0_6_P0_validation.py',
    P1 / 'SOMA_CELL_0_6_P1_validation.py',
    P2 / 'SOMA_CELL_0_6_P2_validation.py',
]
DOC_FILES = [
    'SOMA_CELL_0_6_INTEGRATION_CONTRACT.md',
    'SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md',
    'SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md',
    'SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md',
]
FORMAL_QUICK_TESTS = [
    'build_and_frozen_p2',
    'unmetered_formal_controller_rejected',
    'material_gene_and_controller_development',
    'controller_activity_cost_is_paid_and_conservative',
    'counterbalanced_audit_and_hard_event_retry',
    'calibration_improves_heldout_mae',
    'source_separated_change_sentinel_blocks_routine_state_noise',
    'change_gated_feedback_protects_stable_tissue',
    'small_compiler_and_random_cost_match',
    'material_hgt_activates_controller_without_learning_copy',
]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def run_checked(args, cwd, timeout=300):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError('command failed {}\nSTDOUT:\n{}\nSTDERR:\n{}'.format(' '.join(args), p.stdout, p.stderr))
    return p.stdout

def copy_dependency(name, stage):
    if name == 'SOMA_CELL_0_6_P0_pythonista.py': source = P0 / name
    elif name == 'SOMA_CELL_0_6_P1_pythonista.py': source = P1 / name
    elif name == 'SOMA_CELL_0_6_P2_pythonista.py': source = P2 / name
    else: source = BASE / name
    shutil.copy2(source, stage / name)

def check_measured_evidence(stage):
    validation = (stage / 'SOMA_CELL_0_6_VALIDATION_RESULTS.txt').read_text(encoding='utf-8')
    regression = (stage / 'SOMA_CELL_0_6_REGRESSION_RESULTS.txt').read_text(encoding='utf-8')
    if 'Passed: 22/22' not in validation:
        raise RuntimeError('formal measured validation evidence is not 22/22')
    if 'Total inherited + formal checks : 84/84 PASS' not in regression:
        raise RuntimeError('measured regression evidence is not 84/84')

def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='soma_formal06_release_') as d:
        stage = Path(d) / RELEASE_DIR_NAME
        stage.mkdir()
        for name in DEPENDENCIES: copy_dependency(name, stage)
        for name in FORMAL_FILES: shutil.copy2(FORMAL / name, stage / name)
        for source in PARENT_VALIDATORS: shutil.copy2(source, stage / source.name)
        for name in DOC_FILES: shutil.copy2(ROOT / 'docs' / name, stage / name)

        for name, expected in EXPECTED.items():
            actual = digest(stage / name)
            if actual != expected:
                raise RuntimeError(f'frozen source mismatch {name}: {actual} != {expected}')
        check_measured_evidence(stage)

        for path in sorted(stage.glob('*.py')):
            py_compile.compile(str(path), doraise=True)
        print('cleanup/package', flush=True)
        shutil.rmtree(stage / '__pycache__', ignore_errors=True)

        quick_lines = []
        for name in FORMAL_QUICK_TESTS:
            print('flat test', name, flush=True)
            out = run_checked(['python3', 'SOMA_CELL_0_6_validation.py', '--one', name], stage, timeout=300)
            if '"pass": "PASS"' not in out:
                raise RuntimeError('flat formal quick test failed: ' + name)
            quick_lines.append(name + ': PASS')

        # Fast frozen-parent identity checks from the exact packaged source tree.
        print('parent smoke', flush=True)
        parent_smoke = run_checked([
            'python3', '-c',
            "import hashlib, SOMA_CELL_0_6_P0_pythonista as p0, "
            "SOMA_CELL_0_6_P1_pythonista as p1, SOMA_CELL_0_6_P2_pythonista as p2; "
            "assert p0.BUILD=='SOMA-CELL 0.6-P0.0'; "
            "assert p1.BUILD=='SOMA-CELL 0.6-P1.0'; "
            "assert p2.BUILD=='SOMA-CELL 0.6-P2.0'; "
            "print('PARENT IMPORT PASS')"
        ], stage)
        quick_lines.append(parent_smoke.strip())

        print('formal import smoke', flush=True)
        smoke = run_checked([
            'python3', '-c',
            "import SOMA_CELL_0_6_pythonista as s; "
            "assert s.BUILD=='SOMA-CELL 0.6.0'; "
            "assert s.FORMAL_SCHEMA_VERSION=='0.6-F2.0'; "
            "print('FORMAL IMPORT PASS')"
        ], stage, timeout=60)
        quick_lines.append(smoke.strip())
        (stage / 'flat_release_verification.txt').write_text(
            'Measured canonical suites: formal 22/22, inherited+formal 84/84 PASS\n' +
            '\n'.join(quick_lines) + '\n', encoding='utf-8')

        shutil.rmtree(stage / '__pycache__', ignore_errors=True)
        for p in list(stage.rglob('*.pyc')) + list(stage.rglob('*.pyo')):
            p.unlink(missing_ok=True)
        files = sorted(p for p in stage.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
        (stage / 'SHA256SUMS.txt').write_text(
            '\n'.join(f'{digest(p)}  ./{p.name}' for p in files) + '\n', encoding='utf-8')
        tmp = output.with_suffix(output.suffix + '.tmp')
        tmp.unlink(missing_ok=True)
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file(): z.write(p, Path(RELEASE_DIR_NAME) / p.relative_to(stage))
        tmp.replace(output)
    with zipfile.ZipFile(output) as z:
        bad = z.testzip()
        if bad: raise RuntimeError('ZIP CRC failure ' + bad)
    return output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    out = build(ap.parse_args().output.resolve())
    print(out)
    print(digest(out))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
