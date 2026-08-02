#!/usr/bin/env python3
"""Fail-closed preflight and milestone-transition gate for SOMA work.

This script cannot prove comprehension.  It does, however, prevent silent work
against missing, stale, hash-mismatched, or wrong-milestone project state.  A
normal work receipt is tied to the exact Git HEAD and active next milestone.
A transition receipt is separately issued from a clean checkpoint and permits
one commit that promotes a completed milestone to the next baseline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple

ACK = "READ_LATEST_SOURCES_AND_CONTRACTS"
TRANSITION_ACK = "COMPLETE_CURRENT_MILESTONE_AND_ADVANCE"
ROOT = Path(__file__).resolve().parents[1]

STATIC_REQUIRED = [
    Path("00_MANDATORY_STARTUP_GATE_JA.md"),
    Path("MANDATORY_WORKFLOW.json"),
    Path("PROJECT_STATE.json"),
    Path("CURRENT_BASELINE.txt"),
    Path("docs/SOMA_CONTEXT_HANDOFF_JA.md"),
    Path("docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md"),
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
        # A manifest cannot stably contain its own hash.
        if rel == "manifests/SHA256SUMS.txt":
            continue
        result[rel] = digest
    return result


def parse_baseline_manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
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


def load_workflow() -> dict:
    return json.loads((ROOT / "MANDATORY_WORKFLOW.json").read_text(encoding="utf-8"))


def dynamic_required(state: dict) -> List[Path]:
    """Resolve the latest baseline source/results from PROJECT_STATE.

    This avoids hard-coding 0.5 forever: once P0 is promoted, P1 preflight must
    read P0 itself and its measured results before any edit can begin.
    """
    baseline = state.get("current_baseline", {})
    paths = list(STATIC_REQUIRED)
    for key, fallback in (
        ("body_port_contract_file", None),
        ("parent_tissue_contract_file", None),
        ("tissue_contract_file", None),
        ("tissue_schema_file", None),
        ("formal_contract_file", None),
        ("formal_schema_file", None),
        ("canonical_source", None),
        ("validation_result_file", "results/SOMA_CELL_0_5_VALIDATION_RESULTS.txt"),
        ("regression_result_file", None),
        ("experiment_report_file", "results/SOMA_CELL_0_5_EXPERIMENT_REPORT.txt"),
    ):
        value = baseline.get(key, fallback)
        if value:
            candidate = Path(str(value))
            if candidate not in paths:
                paths.append(candidate)
    return paths


def parse_current_baseline_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def verify() -> Tuple[dict, List[str]]:
    errors: List[str] = []
    checked: Dict[str, dict] = {}

    try:
        state = load_state()
    except Exception as exc:
        errors.append(f"cannot parse PROJECT_STATE.json: {exc}")
        state = {}

    try:
        workflow = load_workflow()
        if not workflow.get("mandatory") or not workflow.get("fail_closed"):
            errors.append("MANDATORY_WORKFLOW is not mandatory/fail_closed")
    except Exception as exc:
        errors.append(f"cannot parse MANDATORY_WORKFLOW.json: {exc}")
        workflow = {}

    for rel in dynamic_required(state):
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing required file: {rel}")
            continue
        checked[str(rel)] = {"sha256": sha256(path), "size": path.stat().st_size}

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

    baseline = state.get("current_baseline", {})
    baseline_name = baseline.get("name")
    canonical = baseline.get("canonical_source")
    next_name = state.get("next_milestone", {}).get("name")
    if not baseline_name:
        errors.append("current baseline name is missing")
    if not canonical:
        errors.append("current canonical source is missing")
    elif not (ROOT / canonical).is_file():
        errors.append(f"canonical source missing: {canonical}")
    if not next_name:
        errors.append("next milestone is missing")

    baseline_text = parse_current_baseline_file(ROOT / "CURRENT_BASELINE.txt")
    if baseline_name and baseline_text.get("CURRENT_BASELINE") != baseline_name:
        errors.append(
            "CURRENT_BASELINE.txt mismatch: {!r} != {!r}".format(
                baseline_text.get("CURRENT_BASELINE"), baseline_name
            )
        )
    if next_name and baseline_text.get("NEXT_MILESTONE") != next_name:
        errors.append(
            "CURRENT_BASELINE.txt next mismatch: {!r} != {!r}".format(
                baseline_text.get("NEXT_MILESTONE"), next_name
            )
        )
    if canonical and baseline_text.get("CANONICAL_SOURCE") != canonical:
        errors.append(
            "CURRENT_BASELINE.txt canonical mismatch: {!r} != {!r}".format(
                baseline_text.get("CANONICAL_SOURCE"), canonical
            )
        )

    report = {
        "rule_id": workflow.get("rule_id"),
        "checkpoint_id": state.get("checkpoint_id"),
        "baseline": baseline_name,
        "canonical_source": canonical,
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
    print(f"  canonical:  {report.get('canonical_source')}")
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


def require_clean_verified(ack: str) -> Tuple[dict | None, int]:
    if ack != ACK:
        print(f"Refusing to start: exact --ack value required: {ACK}", file=sys.stderr)
        return None, 2
    report, errors = verify()
    print_summary(report)
    if errors:
        print("Fail-closed: no work-session receipt created.", file=sys.stderr)
        return None, 1
    if report.get("git_status") not in ("", "unavailable"):
        print(
            "Refusing to start: Git worktree is not clean. Commit, restore, "
            "or explicitly recover first.", file=sys.stderr
        )
        print(report.get("git_status"), file=sys.stderr)
        return None, 4
    return report, 0


def receipt_path(stamp: str, milestone: str, suffix: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in milestone).strip("_")
    directory = ROOT / "work_sessions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stamp}_{safe}_{suffix}.json"


def base_receipt(report: dict, args: argparse.Namespace, receipt_type: str) -> dict:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return {
        "schema_version": "1.1",
        "receipt_type": receipt_type,
        "session_id": "{}-{}".format(
            now.strftime("%Y%m%dT%H%M%SZ"), receipt_type.upper()
        ),
        "started_utc": now.isoformat().replace("+00:00", "Z"),
        "actor": args.actor,
        "purpose": args.purpose,
        "acknowledgement": args.ack,
        "rule_id": report["rule_id"],
        "checkpoint_id": report["checkpoint_id"],
        "baseline": report["baseline"],
        "canonical_source": report["canonical_source"],
        "git_commit": report["git_commit"],
        "git_status_at_start": report["git_status"],
        "required_sources_read_and_hash_checked": report["checked_files"],
        "declared_reviews": [
            "latest PROJECT_STATE",
            "current canonical source",
            "context handoff",
            "0.6 integration contract",
            "current baseline validation and experiment results",
            "known failures and open risks",
            "next milestone roadmap",
        ],
        "fail_closed": True,
    }


def cmd_start(args: argparse.Namespace) -> int:
    report, rc = require_clean_verified(args.ack)
    if report is None:
        return rc
    state = load_state()
    expected_milestone = state["next_milestone"]["name"]
    if args.milestone != expected_milestone:
        print(
            f"Refusing to start: requested milestone {args.milestone!r} "
            f"does not match PROJECT_STATE {expected_milestone!r}",
            file=sys.stderr,
        )
        return 3

    receipt = base_receipt(report, args, "preflight")
    receipt["milestone"] = args.milestone
    stamp = receipt["started_utc"].replace("-", "").replace(":", "").replace("Z", "Z")
    # ISO compacting above leaves a T and is lexically sortable.
    out = receipt_path(stamp, args.milestone, "PREFLIGHT")
    receipt["session_id"] = out.stem
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight receipt created: {out.relative_to(ROOT)}")
    print("Implementation may now begin for the declared milestone only.")
    return 0


def validate_recorded_files(records: dict) -> List[str]:
    errors: List[str] = []
    for rel, record in dict(records or {}).items():
        path = ROOT / rel
        if not path.is_file():
            errors.append("receipt source missing: {}".format(rel))
            continue
        if sha256(path) != record.get("sha256"):
            errors.append("receipt source changed: {}".format(rel))
    return errors


def cmd_transition(args: argparse.Namespace) -> int:
    """Authorize exactly one clean-checkpoint milestone promotion commit."""
    if args.ack != TRANSITION_ACK:
        print(
            "Refusing transition: exact --ack value required: {}".format(TRANSITION_ACK),
            file=sys.stderr,
        )
        return 2
    report, errors = verify()
    print_summary(report)
    if errors:
        print("Fail-closed: no transition receipt created.", file=sys.stderr)
        return 1
    if report.get("git_status") not in ("", "unavailable"):
        print("Refusing transition: Git worktree is not clean.", file=sys.stderr)
        print(report.get("git_status"), file=sys.stderr)
        return 4
    state = load_state()
    expected_from = state.get("next_milestone", {}).get("name")
    if args.from_milestone != expected_from:
        print(
            f"Refusing transition: --from-milestone {args.from_milestone!r} "
            f"does not match PROJECT_STATE {expected_from!r}", file=sys.stderr,
        )
        return 3
    if not args.to_milestone or args.to_milestone == args.from_milestone:
        print("Refusing transition: target milestone must be different and non-empty.", file=sys.stderr)
        return 3
    if not args.baseline_after:
        print("Refusing transition: --baseline-after is required.", file=sys.stderr)
        return 3

    evidence = {}
    for raw in args.evidence:
        rel = Path(raw)
        if rel.is_absolute():
            try:
                rel = rel.resolve().relative_to(ROOT.resolve())
            except ValueError:
                print("Refusing transition: evidence must be inside the project: {}".format(raw), file=sys.stderr)
                return 6
        path = ROOT / rel
        if not path.is_file():
            print("Refusing transition: missing evidence {}".format(rel), file=sys.stderr)
            return 6
        evidence[rel.as_posix()] = {"sha256": sha256(path), "size": path.stat().st_size}
    if not evidence:
        print("Refusing transition: at least one --evidence file is required.", file=sys.stderr)
        return 6

    receipt = base_receipt(report, args, "transition")
    receipt.update({
        "milestone": args.from_milestone,
        "from_milestone": args.from_milestone,
        "to_milestone": args.to_milestone,
        "baseline_before": report["baseline"],
        "baseline_after": args.baseline_after,
        "authorization_scope": "one commit promoting the declared completed milestone",
        "evidence_files_hash_checked": evidence,
    })
    stamp = receipt["started_utc"].replace("-", "").replace(":", "").replace("Z", "Z")
    out = receipt_path(stamp, args.from_milestone, "TRANSITION")
    receipt["session_id"] = out.stem
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Transition receipt created: {out.relative_to(ROOT)}")
    print(
        "One clean-checkpoint commit may now change baseline {!r} -> {!r} and "
        "next milestone {!r} -> {!r}.".format(
            report["baseline"], args.baseline_after,
            args.from_milestone, args.to_milestone,
        )
    )
    return 0


def latest_receipt() -> Path | None:
    directory = ROOT / "work_sessions"
    candidates = list(directory.glob("*_PREFLIGHT.json"))
    candidates.extend(directory.glob("*_TRANSITION.json"))
    return sorted(candidates)[-1] if candidates else None


def cmd_check_receipt(_: argparse.Namespace) -> int:
    path = latest_receipt()
    if path is None:
        print("No preflight or transition receipt found.", file=sys.stderr)
        return 1
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        state = load_state()
        workflow = load_workflow()
    except Exception as exc:
        print(f"Cannot read receipt/state/workflow: {exc}", file=sys.stderr)
        return 1

    current_head = git_output("rev-parse", "HEAD")
    expected_milestone = state.get("next_milestone", {}).get("name")
    expected_baseline = state.get("current_baseline", {}).get("name")
    receipt_type = receipt.get("receipt_type", "preflight")
    errors: List[str] = []

    if current_head != "unavailable" and receipt.get("git_commit") != current_head:
        errors.append(
            f"stale receipt: receipt HEAD {receipt.get('git_commit')} != current HEAD {current_head}"
        )

    if receipt_type == "transition":
        if receipt.get("to_milestone") != expected_milestone:
            errors.append(
                f"transition target {receipt.get('to_milestone')!r} != PROJECT_STATE {expected_milestone!r}"
            )
        if receipt.get("baseline_after") != expected_baseline:
            errors.append(
                f"transition baseline {receipt.get('baseline_after')!r} != PROJECT_STATE {expected_baseline!r}"
            )
        if receipt.get("from_milestone") == receipt.get("to_milestone"):
            errors.append("transition source and target are identical")
    else:
        if receipt.get("milestone") != expected_milestone:
            errors.append(
                f"receipt milestone {receipt.get('milestone')!r} != PROJECT_STATE {expected_milestone!r}"
            )

    expected_ack = TRANSITION_ACK if receipt_type == "transition" else ACK
    if receipt.get("acknowledgement") != expected_ack:
        errors.append("receipt acknowledgement is invalid")
    # A preflight receipt records what was read at session start.  Those files
    # may legitimately be edited by the declared milestone, so their old hash
    # is historical evidence rather than a commit-time invariant.  Canonical
    # tampering is still caught by verify()/the project manifest.
    if receipt_type == "transition":
        evidence = receipt.get("evidence_files_hash_checked", {})
        if not evidence:
            errors.append("transition receipt has no validation/release evidence")
        errors.extend(validate_recorded_files(evidence))
    else:
        required = {path.as_posix() for path in dynamic_required(state)}
        recorded = set(receipt.get("required_sources_read_and_hash_checked", {}))
        missing = sorted(required - recorded)
        if missing:
            errors.append("receipt lacks current required sources: " + ", ".join(missing))
    if receipt.get("rule_id") != workflow.get("rule_id"):
        errors.append(
            f"receipt rule {receipt.get('rule_id')!r} != workflow {workflow.get('rule_id')!r}"
        )
    if not receipt.get("fail_closed"):
        errors.append("receipt is not fail_closed")

    if errors:
        print(f"Receipt check FAIL: {path.relative_to(ROOT)}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"Receipt check PASS: {path.relative_to(ROOT)}")
    print(f"  type: {receipt_type}")
    print(f"  HEAD: {current_head}")
    print(f"  baseline: {expected_baseline}")
    print(f"  milestone: {expected_milestone}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOMA fail-closed project preflight")
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify", help="verify required latest sources and hashes")
    verify_parser.set_defaults(func=cmd_verify)

    check_parser = sub.add_parser(
        "check-receipt",
        help="require the newest receipt to match current Git HEAD and project state",
    )
    check_parser.set_defaults(func=cmd_check_receipt)

    start_parser = sub.add_parser("start", help="create a normal milestone work receipt")
    start_parser.add_argument("--actor", required=True)
    start_parser.add_argument("--purpose", required=True)
    start_parser.add_argument("--milestone", required=True)
    start_parser.add_argument("--ack", required=True)
    start_parser.set_defaults(func=cmd_start)

    transition_parser = sub.add_parser(
        "transition", help="authorize one milestone-promotion commit from a clean checkpoint"
    )
    transition_parser.add_argument("--actor", required=True)
    transition_parser.add_argument("--purpose", required=True)
    transition_parser.add_argument("--from-milestone", required=True)
    transition_parser.add_argument("--to-milestone", required=True)
    transition_parser.add_argument("--baseline-after", required=True)
    transition_parser.add_argument("--evidence", action="append", default=[], required=True)
    transition_parser.add_argument("--ack", required=True)
    transition_parser.set_defaults(func=cmd_transition)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
