#!/usr/bin/env python3
"""Build and verify the standalone Pythonista release for SOMA-CELL 0.6-P0."""
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
P0 = ROOT / "src" / "0_6_p0"
BASE = ROOT / "src" / "baseline"
DEFAULT_OUTPUT = ROOT / "releases" / "SOMA_CELL_0_6_P0_GOLD_20260801.zip"
RELEASE_DIR_NAME = "SOMA_CELL_0_6_P0_RELEASE_20260801"

DEPENDENCIES = [f"SOMA_CELL_0_{n}_pythonista.py" for n in range(1, 6)]
P0_FILES = [
    "SOMA_CELL_0_6_P0_pythonista.py",
    "SOMA_CELL_0_6_P0_validation.py",
    "SOMA_CELL_0_6_P0_experiment.py",
    "SOMA_CELL_0_6_P0_report.py",
    "SOMA_CELL_0_6_P0_START_HERE.txt",
    "SOMA_CELL_0_6_P0_README_JA.md",
    "SOMA_CELL_0_6_P0_API_CONTRACT.md",
    "SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md",
    "SOMA_CELL_0_6_P0_PORT_SCHEMA.json",
    "SOMA_CELL_0_6_P0_MANIFEST.txt",
    "SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt",
    "SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt",
    "soma_cell_0_6_p0_validation.csv",
    "soma_cell_0_6_p0_experiment_results.csv",
    "soma_cell_0_6_p0_experiment_summary.csv",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checked(args: list[str], cwd: Path) -> str:
    process = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            "command failed ({}):\n{}\n{}".format(
                " ".join(args), process.stdout, process.stderr
            )
        )
    return process.stdout


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soma_p0_release_") as directory:
        stage = Path(directory) / RELEASE_DIR_NAME
        stage.mkdir()
        for name in DEPENDENCIES:
            shutil.copy2(BASE / name, stage / name)
        for name in P0_FILES:
            shutil.copy2(P0 / name, stage / name)
        shutil.copy2(
            ROOT / "docs" / "SOMA_CELL_0_6_INTEGRATION_CONTRACT.md",
            stage / "SOMA_CELL_0_6_INTEGRATION_CONTRACT.md",
        )

        for path in sorted(stage.glob("*.py")):
            py_compile.compile(str(path), doraise=True)
        shutil.rmtree(stage / "__pycache__", ignore_errors=True)

        # Run the exact packaged sources, not repository imports.
        validation_out = run_checked(
            ["python3", "SOMA_CELL_0_6_P0_validation.py"], stage
        )
        if "[PASS] port_surface_hides_mutable_body_and_attachment_state" not in validation_out:
            raise RuntimeError("packaged validation did not run the final 22-test suite")
        smoke = run_checked(
            [
                "python3", "-c",
                "import SOMA_CELL_0_6_P0_pythonista as p; "
                "r=p.run_headless_trial(seed=101,seconds=3,initial_cells=1); "
                "assert r['build']=='SOMA-CELL 0.6-P0.0'; "
                "assert r['cells']>=0; print('SMOKE PASS',r['age'],r['matter_residual'])",
            ],
            stage,
        )
        (stage / "release_smoke_test.txt").write_text(smoke, encoding="utf-8")
        shutil.rmtree(stage / "__pycache__", ignore_errors=True)
        for bytecode in list(stage.rglob("*.pyc")) + list(stage.rglob("*.pyo")):
            bytecode.unlink(missing_ok=True)
        for cache_dir in sorted(stage.rglob("__pycache__"), reverse=True):
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Validation rewrites its measured result files; package those exact outputs.
        files = sorted(path for path in stage.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
        lines = ["{}  ./{}".format(digest(path), path.name) for path in files]
        (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        leaked = [path for path in stage.rglob("*") if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo")]
        if leaked:
            raise RuntimeError("bytecode leaked into release staging: {}".format(leaked))

        temp_zip = output.with_suffix(output.suffix + ".tmp")
        if temp_zip.exists():
            temp_zip.unlink()
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(RELEASE_DIR_NAME) / path.relative_to(stage))
        temp_zip.replace(output)

    # CRC check after atomic replacement.
    with zipfile.ZipFile(output, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError("ZIP CRC failure: {}".format(bad))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(result)
    print(digest(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
