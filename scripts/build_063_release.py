#!/usr/bin/env python3
from pathlib import Path
import hashlib, py_compile, shutil, subprocess, tempfile, zipfile

ROOT = Path(__file__).resolve().parents[1]
S63 = ROOT / 'src/0_6_3'
S62 = ROOT / 'src/0_6_2'
S61 = ROOT / 'src/0_6_1'
F06 = ROOT / 'src/0_6'
P2 = ROOT / 'src/0_6_p2'
P1 = ROOT / 'src/0_6_p1'
P0 = ROOT / 'src/0_6_p0'
BASE = ROOT / 'src/baseline'
OUT = ROOT / 'releases/SOMA_CELL_0_6_3_GOLD_20260808.zip'
NAME = 'SOMA_CELL_0_6_3_RELEASE_20260808'


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
    with tempfile.TemporaryDirectory(prefix='soma063_release_') as temp_dir:
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
        ):
            cp(directory / name, stage / name)

        release_names = [
            'SOMA_CELL_0_6_3_pythonista.py',
            'SOMA_CELL_0_6_3_validation.py',
            'SOMA_CELL_0_6_3_experiment.py',
            'SOMA_CELL_0_6_3_report.py',
            'SOMA_CELL_0_6_3_START_HERE.txt',
            'SOMA_CELL_0_6_3_README_JA.md',
            'SOMA_CELL_0_6_3_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_3_FORMAL_SCHEMA.json',
            'SOMA_CELL_0_6_3_VALIDATION_RESULTS.txt',
            'SOMA_CELL_0_6_3_REGRESSION_RESULTS.txt',
            'SOMA_CELL_0_6_3_EXPERIMENT_REPORT.txt',
            'SOMA_CELL_0_6_3_MANIFEST.txt',
            'soma_cell_0_6_3_validation.csv',
            'soma_cell_0_6_3_regression.csv',
            'soma_cell_0_6_3_experiment_results.csv',
            'soma_cell_0_6_3_experiment_summary.csv',
            'soma_cell_0_6_3_experiment_consolidated.json',
        ]
        for name in release_names:
            cp(S63 / name, stage / name)

        cp(ROOT / 'results/SOMA_CELL_0_6_3_PREREGISTRATION.json',
           stage / 'SOMA_CELL_0_6_3_PREREGISTRATION.json')
        cp(ROOT / 'results/SOMA_CELL_0_6_3_RELEASE_RECORD.json',
           stage / 'SOMA_CELL_0_6_3_RELEASE_RECORD.json')

        # Include inherited validators and contracts needed to reproduce the regression chain.
        for directory, name in (
            (P0, 'SOMA_CELL_0_6_P0_validation.py'),
            (P1, 'SOMA_CELL_0_6_P1_validation.py'),
            (P2, 'SOMA_CELL_0_6_P2_validation.py'),
            (F06, 'SOMA_CELL_0_6_validation.py'),
            (S61, 'SOMA_CELL_0_6_1_validation.py'),
            (S62, 'SOMA_CELL_0_6_2_validation.py'),
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
        ):
            cp(ROOT / 'docs' / name, stage / name)

        assert 'Passed: 26/26' in (stage / 'SOMA_CELL_0_6_3_VALIDATION_RESULTS.txt').read_text(encoding='utf-8')
        assert 'Inherited total                : 151/151 PASS' in (stage / 'SOMA_CELL_0_6_3_REGRESSION_RESULTS.txt').read_text(encoding='utf-8')
        assert 'Grand total                    : 177/177 PASS' in (stage / 'SOMA_CELL_0_6_3_REGRESSION_RESULTS.txt').read_text(encoding='utf-8')
        assert 'Scientific status: NEGATIVE_RESULT_WITH_LIMITS' in (stage / 'SOMA_CELL_0_6_3_EXPERIMENT_REPORT.txt').read_text(encoding='utf-8')

        for path in stage.glob('*.py'):
            py_compile.compile(str(path), doraise=True)

        quick = [
            'build_schema_and_frozen_defaults',
            'preregistration_matches_source_hash',
            'unmetered_sentinel_fails_closed',
            'always_none_exact_parent_lockstep',
            'prepared_control_is_equal_initial_matter_and_no_tissue',
            'demand_trigger_matures_two_cell_tissue_and_uses_precursors',
            'development_only_prevents_premature_computation',
            'probe_freezes_learning_in_active_and_dormant_phases',
            'counterbalanced_probe_positive_leases_seed101',
            'nonpositive_probe_reabsorbs_and_archives_cost_seed202',
            'save_restore_exact_during_probe',
            'stable_holdout_has_no_false_development',
            'preregistered_primary_result_is_negative_by_pair_gate',
            'experiment_rows_are_finite_and_materially_conservative',
        ]
        verification = []
        for test_name in quick:
            output = run(['python3', 'SOMA_CELL_0_6_3_validation.py', '--test', test_name], stage, timeout=240)
            if '"pass": 1' not in output:
                raise RuntimeError(output)
            verification.append('{}: PASS'.format(test_name))
        flat = run([
            'python3', '-c',
            "import SOMA_CELL_0_6_3_pythonista as s; "
            "assert s.BUILD=='SOMA-CELL 0.6.3'; "
            "assert s.SCHEMA_VERSION=='0.6.3-NG1.0'; "
            "assert s.POLICY_DEMAND=='demand'; print('FLAT IMPORT PASS')"
        ], stage, timeout=90)
        verification.append(flat.strip())
        (stage / 'flat_release_verification.txt').write_text(
            'Canonical measured checks: 26/26 dedicated + 151/151 inherited = 177/177 PASS\n'
            'Scientific result: NEGATIVE_RESULT_WITH_LIMITS\n' + '\n'.join(verification) + '\n',
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
