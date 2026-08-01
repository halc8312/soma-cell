#!/usr/bin/env python3
"""Fail-closed preflight for SOMA project work.

This script cannot prove comprehension, but it mechanically prevents silent work
against missing, stale, or hash-mismatched project state and creates a durable
receipt of the exact sources that were checked before implementation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple

ACK = "READ_LATEST_SOURCES_AND_CONTRACTS"
ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    Path("00_MANDATORY_STARTUP_GATE_JA.md"),
    Path("MANDATORY_WORKFLOW.json"),
    Path("PROJECT_STATE.json"),
    Path("CURRENT_BASELINE.txt"),
    Path("docs/SOMA_CONTEXT_HANDOFF_JA.md"),
    Path("docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md"),
    Path("src/baseline/SOMA_CELL_0_5_pythonista.py"),
    Path("results/SOMA_CELL_0_5_VALIDATION_RESULTS.txt"),
    Path("results/SOMA_CELL_0_5_EXPERIMENT_REPORT.txt"),
    Path("planning/ROADMAP_0_6_JA.md"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_top_manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, rel = parts
        rel = rel.strip()
        if rel.startswith("./"):
            rel = rel[2:]
        # The manifest's own digest is intentionally ignored because rewriting
        # the manifest changes that digest recursively.
        if rel == "manifests/SHA256SUMS.txt":
            continue
        result[rel] = digest
    return result


def parse_baseline_manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 3:
            continue
        digest = parts[0]
        filename = parts[-1]
        result[f"src/baseline/{filename}"] = digest
    return result


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unavailable"


def load_state() -> dict:
    return json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))


def verify() -> Tuple[dict, List[str]]:
    errors: List[str] = []
    checked: Dict[str, dict] = {}

    for rel in REQUIRED:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing required file: {rel}")
            continue
        checked[str(rel)] = {"sha256": sha256(path), "size": path.stat().st_size}

    try:
        state = load_state()
    except Exception as exc:
        errors.append(f"cannot parse PROJECT_STATE.json: {exc}")
        state = {}

    try:
        workflow = json.loads((ROOT / "MANDATORY_WORKFLOW.json").read_text(encoding="utf-8"))
        if not workflow.get("mandatory") or not workflow.get("fail_closed"):
            errors.append("MANDATORY_WORKFLOW is not mandatory/fail_closed")
    except Exception as exc:
        errors.append(f"cannot parse MANDATORY_WORKFLOW.json: {exc}")
        workflow = {}

    top_manifest = parse_top_manifest(ROOT / "manifests/SHA256SUMS.txt")
    baseline_manifest = parse_baseline_manifest(ROOT / "src/baseline/SHA256SUMS.txt")
    expected = {**top_manifest, **baseline_manifest}
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.exists():
            errors.append(f"manifest target missing: {rel}")
            continue
        actual = sha256(path)
        if actual != digest:
            errors.append(f"hash mismatch: {rel}: expected {digest}, got {actual}")

    canonical = state.get("current_baseline", {}).get("canonical_source")
    if canonical != "src/baseline/SOMA_CELL_0_5_pythonista.py":
        errors.append(f"unexpected canonical source: {canonical!r}")
    if canonical and not (ROOT / canonical).exists():
        errors.append(f"canonical source missing: {canonical}")

    next_name = state.get("next_milestone", {}).get("name")
    if next_name != "SOMA-CELL 0.6-P0":
        errors.append(f"unexpected next milestone: {next_name!r}")

    report = {
        "rule_id": workflow.get("rule_id"),
        "checkpoint_id": state.get("checkpoint_id"),
        "baseline": state.get("current_baseline", {}).get("name"),
        "next_milestone": next_name,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "checked_files": checked,
        "errors": errors,
    }
    return report, errors


def print_summary(report: dict) -> None:
    print("SOMA mandatory preflight")
    print(f"  rule:       {report.get('rule_id')}")
    print(f"  checkpoint: {report.get('checkpoint_id')}")
    print(f"  baseline:   {report.get('baseline')}")
    print(f"  next:       {report.get('next_milestone')}")
    print(f"  git:        {report.get('git_commit')}")
    print(f"  files:      {len(report.get('checked_files', {}))} required files read/hash-checked")
    if report.get("errors"):
        print("  RESULT: FAIL")
        for err in report["errors"]:
            print(f"    - {err}")
    else:
        print("  RESULT: PASS")


def cmd_verify(_: argparse.Namespace) -> int:
    report, errors = verify()
    print_summary(report)
    return 1 if errors else 0


def cmd_start(args: argparse.Namespace) -> int:
    if args.ack != ACK:
        print(f"Refusing to start: exact --ack value required: {ACK}", file=sys.stderr)
        return 2
    report, errors = verify()
    print_summary(report)
    if errors:
        print("Fail-closed: no work-session receipt created.", file=sys.stderr)
        return 1
    if report.get("git_status") not in ("", "unavailable"):
        print("Refusing to start: Git worktree is not clean. Commit, restore, or explicitly recover first.", file=sys.stderr)
        print(report.get("git_status"), file=sys.stderr)
        return 4

    state = load_state()
    expected_milestone = state["next_milestone"]["name"]
    if args.milestone != expected_milestone:
        print(
            f"Refusing to start: requested milestone {args.milestone!r} "
            f"does not match PROJECT_STATE {expected_milestone!r}",
            file=sys.stderr,
        )
        return 3

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    safe_milestone = "".join(c if c.isalnum() else "_" for c in args.milestone).strip("_")
    receipt_dir = ROOT / "work_sessions"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "1.0",
        "session_id": f"{stamp}-{safe_milestone}",
        "started_utc": now.isoformat().replace("+00:00", "Z"),
        "actor": args.actor,
        "purpose": args.purpose,
        "milestone": args.milestone,
        "acknowledgement": args.ack,
        "rule_id": report["rule_id"],
        "checkpoint_id": report["checkpoint_id"],
        "baseline": report["baseline"],
        "git_commit": report["git_commit"],
        "git_status_at_start": report["git_status"],
        "required_sources_read_and_hash_checked": report["checked_files"],
        "declared_reviews": [
            "latest PROJECT_STATE",
            "current canonical source",
            "context handoff",
            "0.6 integration contract",
            "baseline validation and experiment results",
            "known failures and open risks",
            "next milestone roadmap",
        ],
        "fail_closed": True,
    }
    out = receipt_dir / f"{stamp}_{safe_milestone}_PREFLIGHT.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight receipt created: {out.relative_to(ROOT)}")
    print("Implementation may now begin for the declared milestone only.")
    return 0


def latest_receipt() -> Path | None:
    receipts = sorted((ROOT / "work_sessions").glob("*_PREFLIGHT.json"))
    return receipts[-1] if receipts else None


def cmd_check_receipt(_: argparse.Namespace) -> int:
    path = latest_receipt()
    if path is None:
        print("No preflight receipt found.", file=sys.stderr)
        return 1
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        state = load_state()
    except Exception as exc:
        print(f"Cannot read receipt/state: {exc}", file=sys.stderr)
        return 1
    current_head = git_output("rev-parse", "HEAD")
    expected_milestone = state.get("next_milestone", {}).get("name")
    errors = []
    if current_head != "unavailable" and receipt.get("git_commit") != current_head:
        errors.append(
            f"stale receipt: receipt HEAD {receipt.get('git_commit')} != current HEAD {current_head}"
        )
    if receipt.get("milestone") != expected_milestone:
        errors.append(
            f"receipt milestone {receipt.get('milestone')!r} != PROJECT_STATE {expected_milestone!r}"
        )
    if receipt.get("acknowledgement") != ACK:
        errors.append("receipt acknowledgement is invalid")
    if not receipt.get("fail_closed"):
        errors.append("receipt is not fail_closed")
    if errors:
        print(f"Receipt check FAIL: {path.relative_to(ROOT)}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"Receipt check PASS: {path.relative_to(ROOT)}")
    print(f"  HEAD: {current_head}")
    print(f"  milestone: {expected_milestone}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SOMA fail-closed project preflight")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="verify required sources and hashes")
    v.set_defaults(func=cmd_verify)
    c = sub.add_parser("check-receipt", help="require the newest receipt to match the current Git HEAD and milestone")
    c.set_defaults(func=cmd_check_receipt)
    s = sub.add_parser("start", help="verify and create a durable work-session receipt")
    s.add_argument("--actor", required=True)
    s.add_argument("--purpose", required=True)
    s.add_argument("--milestone", required=True)
    s.add_argument("--ack", required=True)
    s.set_defaults(func=cmd_start)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
