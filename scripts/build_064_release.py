#!/usr/bin/env python3
from pathlib import Path
import hashlib, py_compile, shutil, subprocess, tempfile, zipfile

ROOT = Path(__file__).resolve().parents[1]
S64 = ROOT / 'src/0_6_4'
S63 = ROOT / 'src/0_6_3'
S62 = ROOT / 'src/0_6_2'
S61 = ROOT / 'src/0_6_1'
F06 = ROOT / 'src/0_6'
P2 = ROOT / 'src/0_6_p2'
P1 = ROOT / 'src/0_6_p1'
P0 = ROOT / 'src/0_6_p0'
BASE = ROOT / 'src/baseline'
OUT = ROOT / 'releases/SOMA_CELL_0_6_4_GOLD_20260808.zip'
NAME = 'SOMA_CELL_0_6_4_RELEASE_20260808'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run(args, cwd, timeout=300):
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError('{}\nSTDOUT:\n{}\nSTDERR:\n{}'.format(args, proc.stdout, proc.stderr))
    return proc.stdout


def main(out=OUT):
    with tempfile.TemporaryDirectory(prefix='soma064_release_') as temp_dir:
        stage = Path(temp_dir) / NAME
        stage.mkdir()
        for number in range(1, 6):
            name = 'SOMA_CELL_0_{}_pythonista.py'.format(number)
            cp(BASE / name, stage / name)
        for directory, name in (
            (P0, 'SOMA_CELL_0_6_P0_pythonista.py'),
            (P1, 'SOMA_CELL_0_6_P1_pythonista.py'),
            (P2, 'SOMA_CELL_0_6_P2_pythonista.py'),
            (F06, 'SOMA_CELL_0_6_pythonista.py'),
            (S61, 'SOMA_CELL_0_6_1_pythonista.py'),
            (S62, 'SOMA_CELL_0_6_2_pythonista.py'),
            (S63, 'SOMA_CELL_0_6_3_pythonista.py'),
        ):
            cp(directory / name, stage / name)

        release_names = [
            'SOMA_CELL_0_6_4_pythonista.py',
            'SOMA_CELL_0_6_4_validation.py',
            'SOMA_CELL_0_6_4_experiment.py',
            'SOMA_CELL_0_6_4_worker.py',
            'SOMA_CELL_0_6_4_report.py',
            'SOMA_CELL_0_6_4_START_HERE.txt',
            'SOMA_CELL_0_6_4_README_JA.md',
            'SOMA_CELL_0_6_4_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_4_FORMAL_SCHEMA.json',
            'SOMA_CELL_0_6_4_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_4_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_4_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_4_MANIFEST.txt',
            'soma_cell_0_6_4_validation.csv',
            'soma_cell_0_6_4_regression.csv',
            'soma_cell_0_6_4_experiment_results.csv',
            'soma_cell_0_6_4_experiment_summary.csv',
            'soma_cell_0_6_4_experiment_consolidated.json',
        ]
        for name in release_names:
            cp(S64 / name, stage / name)
        cp(ROOT / 'results/SOMA_CELL_0_6_4_PREREGISTRATION.json', stage / 'SOMA_CELL_0_6_4_PREREGISTRATION.json')
        cp(ROOT / 'results/SOMA_CELL_0_6_4_RELEASE_RECORD.json', stage / 'SOMA_CELL_0_6_4_RELEASE_RECORD.json')

        for directory, name in (
            (P0, 'SOMA_CELL_0_6_P0_validation.py'),
            (P1, 'SOMA_CELL_0_6_P1_validation.py'),
            (P2, 'SOMA_CELL_0_6_P2_validation.py'),
            (F06, 'SOMA_CELL_0_6_validation.py'),
            (S61, 'SOMA_CELL_0_6_1_validation.py'),
            (S62, 'SOMA_CELL_0_6_2_validation.py'),
            (S63, 'SOMA_CELL_0_6_3_validation.py'),
        ):
            cp(directory / name, stage / name)
        for name in (
            'SOMA_CELL_0_6_INTEGRATION_CONTRACT.md',
            'SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md',
            'SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md',
            'SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md',
            'SOMA_CELL_0_6_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_1_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_2_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_3_FORMAL_CONTRACT.md',
        ):
            cp(ROOT / 'docs' / name, stage / name)

        assert 'Result: 25/25 PASS' in (stage / 'SOMA_CELL_0_6_4_VALIDATION_RESULTS.txt').read_text(encoding='utf-8')
        regression = (stage / 'SOMA_CELL_0_6_4_REGRESSION_RESULTS.txt').read_text(encoding='utf-8')
        assert 'Inherited total                       : 177/177 PASS' in regression
        assert 'Grand total                            : 202/202 PASS' in regression
        assert 'Decision: NEGATIVE_BOUNDARY_RESULT' in (stage / 'SOMA_CELL_0_6_4_EXPERIMENT_REPORT.txt').read_text(encoding='utf-8')

        for path in stage.glob('*.py'):
            py_compile.compile(str(path), doraise=True)

        quick = [0, 1, 3, 4, 6, 7, 11, 17, 19, 22, 23]
        verification = []
        for test_index in quick:
            output = run(['python3', 'SOMA_CELL_0_6_4_validation.py', '--test', str(test_index)], stage, timeout=300)
            if ' PASS ' not in output:
                raise RuntimeError(output)
            verification.append('test index {}: PASS'.format(test_index))
        flat = run([
            'python3', '-c',
            "import SOMA_CELL_0_6_4_pythonista as s; "
            "assert s.BUILD=='SOMA-CELL 0.6.4'; "
            "assert s.SCHEMA_VERSION=='0.6.4-AB1.1'; "
            "assert s.AMORT_CONDITIONAL=='conditional'; print('FLAT IMPORT PASS')"
        ], stage, timeout=120)
        verification.append(flat.strip())
        (stage / 'flat_release_verification.txt').write_text(
            'Canonical measured checks: 25/25 dedicated + 177/177 inherited = 202/202 PASS\n'
            'Scientific result: NEGATIVE_BOUNDARY_RESULT_WITH_LIMITS\n' + '\n'.join(verification) + '\n',
            encoding='utf-8',
        )

        shutil.rmtree(stage / '__pycache__', ignore_errors=True)
        files = sorted(path for path in stage.iterdir() if path.is_file() and path.name != 'SHA256SUMS.txt')
        (stage / 'SHA256SUMS.txt').write_text(
            '\n'.join('{}  ./{}'.format(sha(path), path.name) for path in files) + '\n',
            encoding='utf-8',
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        temp_zip = out.with_suffix('.zip.tmp')
        temp_zip.unlink(missing_ok=True)
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    archive.write(path, Path(NAME) / path.relative_to(stage))
        temp_zip.replace(out)

    with zipfile.ZipFile(out) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('bad ZIP member: ' + bad)
    print(out)
    print(sha(out))


if __name__ == '__main__':
    main()
