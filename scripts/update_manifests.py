#!/usr/bin/env python3
"""Regenerate deterministic file registry and SHA-256 manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SHA_PATH = ROOT / "manifests/SHA256SUMS.txt"
REGISTRY_PATH = ROOT / "manifests/FILE_REGISTRY.csv"
EXCLUDE_REGISTRY = {
    "manifests/SHA256SUMS.txt",
    "manifests/FILE_REGISTRY.csv",
}
EXCLUDE_SHA = {"manifests/SHA256SUMS.txt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def project_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    values = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = ROOT / rel
        if path.is_file() and "__pycache__" not in path.parts and not rel.endswith((".pyc", ".pyo")):
            values.append(rel)
    return sorted(set(values))


def main() -> int:
    files = project_files()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "size_bytes", "sha256"))
        for rel in files:
            if rel in EXCLUDE_REGISTRY:
                continue
            path = ROOT / rel
            writer.writerow((rel, path.stat().st_size, sha256(path)))

    # FILE_REGISTRY now has its final content and can itself be hashed.
    files = project_files()
    lines = []
    for rel in files:
        if rel in EXCLUDE_SHA:
            continue
        lines.append("{}  ./{}".format(sha256(ROOT / rel), rel))
    SHA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("updated {} files".format(len(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
