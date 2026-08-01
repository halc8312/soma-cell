#!/usr/bin/env python3
"""Build and verify the standalone Pythonista release for SOMA-CELL 0.6-P2."""
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
P2 = ROOT / "src" / "0_6_p2"
P1 = ROOT / "src" / "0_6_p1"
P0 = ROOT / "src" / "0_6_p0"
BASE = ROOT / "src" / "baseline"
DEFAULT_OUTPUT = ROOT / "releases" / "SOMA_CELL_0_6_P2_GOLD_20260802.zip"
RELEASE_DIR_NAME = "SOMA_CELL_0_6_P2_RELEASE_20260802"

DEPENDENCIES = [f"SOMA_CELL_0_{n}_pythonista.py" for n in range(1, 6)]
DEPENDENCIES += [
    "SOMA_CELL_0_6_P0_pythonista.py",
    "SOMA_CELL_0_6_P1_pythonista.py",
]
P2_FILES = [
    "SOMA_CELL_0_6_P2_pythonista.py",
    "SOMA_CELL_0_6_P2_validation.py",
    "SOMA_CELL_0_6_P2_experiment.py",
    "SOMA_CELL_0_6_P2_report.py",
    "SOMA_CELL_0_6_P2_START_HERE.txt",
    "SOMA_CELL_0_6_P2_README_JA.md",
    "SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md",
    "SOMA_CELL_0_6_P2_TISSUE_SCHEMA.json",
    "SOMA_CELL_0_6_P2_MANIFEST.txt",
    "SOMA_CELL_0_6_P2_VALIDATION_RESULTS.txt",
    "SOMA_CELL_0_6_P2_REGRESSION_RESULTS.txt",
    "SOMA_CELL_0_6_P2_EXPERIMENT_REPORT.txt",
    "soma_cell_0_6_p2_validation.csv",
    "soma_cell_0_6_p2_experiment_results.csv",
    "soma_cell_0_6_p2_experiment_results.json",
    "soma_cell_0_6_p2_experiment_summary.csv",
]
DOC_FILES = [
    "SOMA_CELL_0_6_INTEGRATION_CONTRACT.md",
    "SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md",
    "SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md",
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


def copy_dependency(name: str, stage: Path) -> None:
    if name == "SOMA_CELL_0_6_P0_pythonista.py":
        source = P0 / name
    elif name == "SOMA_CELL_0_6_P1_pythonista.py":
        source = P1 / name
    else:
        source = BASE / name
    shutil.copy2(source, stage / name)


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soma_p2_release_") as directory:
        stage = Path(directory) / RELEASE_DIR_NAME
        stage.mkdir()
        for name in DEPENDENCIES:
            copy_dependency(name, stage)
        for name in P2_FILES:
            shutil.copy2(P2 / name, stage / name)
        for name in DOC_FILES:
            shutil.copy2(ROOT / "docs" / name, stage / name)

        for path in sorted(stage.glob("*.py")):
            py_compile.compile(str(path), doraise=True)
        shutil.rmtree(stage / "__pycache__", ignore_errors=True)

        validation_out = run_checked(
            ["python3", "SOMA_CELL_0_6_P2_validation.py"], stage
        )
        if "[PASS] cue_reversal_full_vs_equal_fixed_twin" not in validation_out:
            raise RuntimeError("packaged validation did not run the final 22-test P2 suite")
        if validation_out.count("[PASS]") != 22:
            raise RuntimeError("packaged P2 validation did not report 22 PASS tests")

        smoke = run_checked(
            [
                "python3", "-c",
                "import SOMA_CELL_0_6_P2_pythonista as p; "
                "r=p.run_headless_trial(seed=101,seconds=3,initial_cells=1); "
                "assert r['build']=='SOMA-CELL 0.6-P2.0'; "
                "assert r['finite']; "
                "print('SMOKE PASS',r['age'],r['cells'],r['matter_residual'])",
            ],
            stage,
        )
        (stage / "release_smoke_test.txt").write_text(smoke, encoding="utf-8")

        # Ensure frozen parent suites still pass from the same flat package.
        p0_validation = ROOT / "src" / "0_6_p0" / "SOMA_CELL_0_6_P0_validation.py"
        p1_validation = ROOT / "src" / "0_6_p1" / "SOMA_CELL_0_6_P1_validation.py"
        shutil.copy2(p0_validation, stage / p0_validation.name)
        shutil.copy2(p1_validation, stage / p1_validation.name)
        p0_out = run_checked(["python3", p0_validation.name], stage)
        p1_out = run_checked(["python3", p1_validation.name], stage)
        if p0_out.count("[PASS]") != 22 or p1_out.count("[PASS]") != 18:
            raise RuntimeError("flat P0/P1 regression did not fully pass")
        (stage / "flat_regression_test.txt").write_text(
            "P0 PASS count: {}\nP1 PASS count: {}\n".format(
                p0_out.count("[PASS]"), p1_out.count("[PASS]")
            ), encoding="utf-8"
        )

        shutil.rmtree(stage / "__pycache__", ignore_errors=True)
        for bytecode in list(stage.rglob("*.pyc")) + list(stage.rglob("*.pyo")):
            bytecode.unlink(missing_ok=True)
        for cache_dir in sorted(stage.rglob("__pycache__"), reverse=True):
            shutil.rmtree(cache_dir, ignore_errors=True)

        files = sorted(
            path for path in stage.iterdir()
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (stage / "SHA256SUMS.txt").write_text(
            "\n".join("{}  ./{}".format(digest(path), path.name) for path in files) + "\n",
            encoding="utf-8",
        )

        leaked = [
            path for path in stage.rglob("*")
            if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo")
        ]
        if leaked:
            raise RuntimeError("bytecode leaked into release staging: {}".format(leaked))

        temp_zip = output.with_suffix(output.suffix + ".tmp")
        if temp_zip.exists():
            temp_zip.unlink()
        with zipfile.ZipFile(
            temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(RELEASE_DIR_NAME) / path.relative_to(stage))
        temp_zip.replace(output)

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
