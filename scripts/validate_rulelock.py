#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('PYTHON', 'python3')


def run(args, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    checks = []
    rc, out = run([PYTHON, 'scripts/soma_preflight.py', 'verify'])
    checks.append(('verify_clean_checkpoint', rc == 0, rc, out))

    before = set((ROOT/'work_sessions').glob('*_PREFLIGHT.json'))
    rc, out = run([
        PYTHON, 'scripts/soma_preflight.py', 'start',
        '--actor', 'validation', '--purpose', 'wrong ack test',
        '--milestone', 'SOMA-CELL 0.6-P0', '--ack', 'WRONG_ACK'
    ])
    after = set((ROOT/'work_sessions').glob('*_PREFLIGHT.json'))
    checks.append(('wrong_ack_fails_without_receipt', rc == 2 and before == after, rc, out))

    with tempfile.TemporaryDirectory(prefix='soma_rulelock_milestone_') as td:
        tmp = Path(td) / 'repo'
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns('.git'))
        rc, out = run([
            PYTHON, 'scripts/soma_preflight.py', 'start',
            '--actor', 'validation', '--purpose', 'wrong milestone test',
            '--milestone', 'SOMA-CELL 9.9', '--ack', 'READ_LATEST_SOURCES_AND_CONTRACTS'
        ], cwd=tmp)
        checks.append(('wrong_milestone_fails', rc == 3, rc, out))

    with tempfile.TemporaryDirectory(prefix='soma_rulelock_') as td:
        tmp = Path(td) / 'repo'
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns('.git'))
        baseline = tmp / 'src/baseline/SOMA_CELL_0_5_pythonista.py'
        baseline.write_bytes(baseline.read_bytes() + b'\n# tamper validation\n')
        rc, out = run([PYTHON, 'scripts/soma_preflight.py', 'verify'], cwd=tmp)
        checks.append(('hash_tamper_fails', rc == 1 and 'hash mismatch' in out, rc, out))

    receipt_exists = any((ROOT/'work_sessions').glob('*_PREFLIGHT.json'))
    checks.append(('valid_receipt_exists', receipt_exists, 0 if receipt_exists else 1, ''))

    rc, out = run([PYTHON, 'scripts/soma_preflight.py', 'check-receipt'])
    checks.append(('current_head_receipt_passes', rc == 0, rc, out))

    with tempfile.TemporaryDirectory(prefix='soma_rulelock_receipt_') as td:
        tmp = Path(td) / 'repo'
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns('.git'))
        receipts = sorted((tmp/'work_sessions').glob('*_PREFLIGHT.json'))
        data = __import__('json').loads(receipts[-1].read_text(encoding='utf-8'))
        data['milestone'] = 'STALE-MILESTONE'
        receipts[-1].write_text(__import__('json').dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        rc, out = run([PYTHON, 'scripts/soma_preflight.py', 'check-receipt'], cwd=tmp)
        checks.append(('stale_or_wrong_receipt_fails', rc == 1 and 'milestone' in out, rc, out))

    hook = ROOT/'.githooks/pre-commit'
    hook_ok = hook.is_file() and os.access(hook, os.X_OK)
    checks.append(('precommit_hook_present_executable', hook_ok, 0 if hook_ok else 1, ''))

    passed = sum(1 for _, ok, _, _ in checks if ok)
    lines = [
        'SOMA Project Rule Lock Validation',
        f'Passed: {passed}/{len(checks)}',
        ''
    ]
    for name, ok, rc, out in checks:
        lines.append(f"{'PASS' if ok else 'FAIL'} | {name} | rc={rc}")
        if not ok and out:
            lines.append(out)
    result = '\n'.join(lines) + '\n'
    (ROOT/'results/SOMA_RULELOCK_VALIDATION_RESULTS.txt').write_text(result, encoding='utf-8')
    print(result, end='')
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
