#!/usr/bin/env python3
"""Independent self-test for the SOMA RuleLock workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", "python3")
ACK = "READ_LATEST_SOURCES_AND_CONTRACTS"
TRANSITION_ACK = "COMPLETE_CURRENT_MILESTONE_AND_ADVANCE"


def run(args, cwd=ROOT):
    process = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return process.returncode, (process.stdout + process.stderr).strip()


def current_milestone(root=ROOT):
    state = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    return state["next_milestone"]["name"]


def new_receipts(root, before):
    after = set((root / "work_sessions").glob("*.json"))
    return sorted(after - before)


def main() -> int:
    checks = []
    milestone = current_milestone()

    rc, out = run([PYTHON, "scripts/soma_preflight.py", "verify"])
    checks.append(("verify_clean_checkpoint", rc == 0, rc, out))

    before = set((ROOT / "work_sessions").glob("*.json"))
    rc, out = run([
        PYTHON, "scripts/soma_preflight.py", "start",
        "--actor", "validation", "--purpose", "wrong ack test",
        "--milestone", milestone, "--ack", "WRONG_ACK",
    ])
    checks.append((
        "wrong_ack_fails_without_receipt",
        rc == 2 and not new_receipts(ROOT, before), rc, out,
    ))

    with tempfile.TemporaryDirectory(prefix="soma_rulelock_milestone_") as directory:
        temp = Path(directory) / "repo"
        shutil.copytree(ROOT, temp, ignore=shutil.ignore_patterns(".git"))
        rc, out = run([
            PYTHON, "scripts/soma_preflight.py", "start",
            "--actor", "validation", "--purpose", "wrong milestone test",
            "--milestone", "SOMA-CELL 9.9", "--ack", ACK,
        ], cwd=temp)
        checks.append(("wrong_milestone_fails", rc == 3, rc, out))

    with tempfile.TemporaryDirectory(prefix="soma_rulelock_tamper_") as directory:
        temp = Path(directory) / "repo"
        shutil.copytree(ROOT, temp, ignore=shutil.ignore_patterns(".git"))
        canonical = json.loads((temp / "PROJECT_STATE.json").read_text(encoding="utf-8"))["current_baseline"]["canonical_source"]
        target = temp / canonical
        target.write_bytes(target.read_bytes() + b"\n# tamper validation\n")
        rc, out = run([PYTHON, "scripts/soma_preflight.py", "verify"], cwd=temp)
        checks.append(("hash_tamper_fails", rc == 1 and "hash mismatch" in out, rc, out))

    with tempfile.TemporaryDirectory(prefix="soma_rulelock_fresh_") as directory:
        temp = Path(directory) / "repo"
        shutil.copytree(ROOT, temp, ignore=shutil.ignore_patterns(".git"))
        run(["git", "init", "-q"], cwd=temp)
        run(["git", "config", "user.email", "validation@example.invalid"], cwd=temp)
        run(["git", "config", "user.name", "SOMA RuleLock Validation"], cwd=temp)
        run(["git", "add", "-A"], cwd=temp)
        rc_commit, out_commit = run(["git", "commit", "-q", "--no-verify", "-m", "validation baseline"], cwd=temp)
        if rc_commit != 0:
            checks.append(("temporary_git_baseline", False, rc_commit, out_commit))
        else:
            run(["git", "config", "core.hooksPath", ".githooks"], cwd=temp)
            temp_milestone = current_milestone(temp)
            existing = set((temp / "work_sessions").glob("*.json"))
            rc1, out1 = run([
                PYTHON, "scripts/soma_preflight.py", "start",
                "--actor", "validation", "--purpose", "fresh receipt test",
                "--milestone", temp_milestone, "--ack", ACK,
            ], cwd=temp)
            rc2, out2 = run([PYTHON, "scripts/soma_preflight.py", "check-receipt"], cwd=temp)
            checks.append((
                "fresh_current_head_receipt_passes",
                rc1 == 0 and rc2 == 0, max(rc1, rc2), out1 + "\n" + out2,
            ))

            rc_hook, out_hook = run([str(temp / ".githooks/pre-commit")], cwd=temp)
            checks.append(("precommit_accepts_fresh_receipt", rc_hook == 0, rc_hook, out_hook))

            (temp / "DIRTY_VALIDATION.tmp").write_text("dirty\n", encoding="utf-8")
            rc3, out3 = run([
                PYTHON, "scripts/soma_preflight.py", "start",
                "--actor", "validation", "--purpose", "dirty worktree test",
                "--milestone", temp_milestone, "--ack", ACK,
            ], cwd=temp)
            checks.append(("dirty_worktree_start_fails", rc3 == 4, rc3, out3))
            (temp / "DIRTY_VALIDATION.tmp").unlink()

            generated = new_receipts(temp, existing)
            for path in generated:
                path.unlink()
            # A transition receipt is issued from a clean validated checkpoint.
            rc_bad, out_bad = run([
                PYTHON, "scripts/soma_preflight.py", "transition",
                "--actor", "validation", "--purpose", "wrong ack",
                "--from-milestone", temp_milestone,
                "--to-milestone", "SOMA-CELL 0.6-P1",
                "--baseline-after", "SOMA-CELL 0.6-P0.0",
                "--evidence", "results/SOMA_CELL_0_5_VALIDATION_RESULTS.txt",
                "--ack", "WRONG_ACK",
            ], cwd=temp)
            checks.append(("wrong_transition_ack_fails", rc_bad == 2, rc_bad, out_bad))

            rc5, out5 = run([
                PYTHON, "scripts/soma_preflight.py", "transition",
                "--actor", "validation", "--purpose", "validated transition",
                "--from-milestone", temp_milestone,
                "--to-milestone", "SOMA-CELL 0.6-P1",
                "--baseline-after", "SOMA-CELL 0.6-P0.0",
                "--evidence", "results/SOMA_CELL_0_5_VALIDATION_RESULTS.txt",
                "--ack", TRANSITION_ACK,
            ], cwd=temp)
            state_path = temp / "PROJECT_STATE.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["current_baseline"]["name"] = "SOMA-CELL 0.6-P0.0"
            state["next_milestone"]["name"] = "SOMA-CELL 0.6-P1"
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rc6, out6 = run([PYTHON, "scripts/soma_preflight.py", "check-receipt"], cwd=temp)
            checks.append((
                "transition_receipt_authorizes_one_state_advance",
                rc5 == 0 and rc6 == 0,
                max(rc5, rc6), out5 + "\n" + out6,
            ))

            state["next_milestone"]["name"] = "SOMA-CELL 0.6-P2"
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rc7, out7 = run([PYTHON, "scripts/soma_preflight.py", "check-receipt"], cwd=temp)
            checks.append((
                "transition_receipt_wrong_target_fails",
                rc7 == 1 and "transition target" in out7, rc7, out7,
            ))

    hook = ROOT / ".githooks/pre-commit"
    hook_ok = hook.is_file() and os.access(hook, os.X_OK)
    checks.append(("precommit_hook_present_executable", hook_ok, 0 if hook_ok else 1, ""))

    passed = sum(1 for _, ok, _, _ in checks if ok)
    lines = [
        "SOMA Project Rule Lock Validation",
        f"Passed: {passed}/{len(checks)}",
        "",
    ]
    for name, ok, rc, out in checks:
        lines.append(f"{'PASS' if ok else 'FAIL'} | {name} | rc={rc}")
        if not ok and out:
            lines.append(out)
    result = "\n".join(lines) + "\n"
    (ROOT / "results/SOMA_RULELOCK_VALIDATION_RESULTS.txt").write_text(result, encoding="utf-8")
    print(result, end="")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
