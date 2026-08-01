#!/usr/bin/env python3
"""Build the SOMA-CELL 0.6-P2 continuity vault from the promoted state."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD_VAULT = ROOT / "archives/SOMA_CONTINUITY_VAULT_GOLD_20260801_0_6_P1.zip"
OUT = ROOT / "archives/SOMA_CONTINUITY_VAULT_GOLD_20260802_0_6_P2.zip"
VAULT_NAME = "SOMA_CONTINUITY_VAULT_GOLD_20260802_0_6_P2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_required(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def latest_receipt(pattern: str) -> Path:
    values = sorted((ROOT / "work_sessions").glob(pattern))
    if not values:
        raise FileNotFoundError(pattern)
    return values[-1]


def main() -> int:
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if state["current_baseline"]["name"] != "SOMA-CELL 0.6-P2.0":
        raise RuntimeError("P2 vault requires promoted P2 baseline")
    if state["next_milestone"]["name"] != "SOMA-CELL 0.6":
        raise RuntimeError("P2 vault requires formal 0.6 as next milestone")

    with tempfile.TemporaryDirectory(prefix="soma_p2_vault_") as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(OLD_VAULT) as archive:
            archive.extractall(temporary)
        old_root = next(path for path in temporary.iterdir() if path.is_dir())
        vault_root = temporary / VAULT_NAME
        old_root.rename(vault_root)

        mapping = {
            ROOT / "docs/SOMA_CONTEXT_HANDOFF_JA.md": vault_root / "docs/SOMA_CONTEXT_HANDOFF_JA.md",
            ROOT / "docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md": vault_root / "docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md",
            ROOT / "docs/SOMA_LINEAGE_INDEX_JA.md": vault_root / "docs/SOMA_LINEAGE_INDEX_JA.md",
            ROOT / "docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md": vault_root / "docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md",
            ROOT / "docs/SOMA_CELL_0_6_P2_TISSUE_SCHEMA.json": vault_root / "docs/SOMA_CELL_0_6_P2_TISSUE_SCHEMA.json",
            ROOT / "docs/SOMA_CELL_0_6_P2_README_JA.md": vault_root / "docs/SOMA_CELL_0_6_P2_README_JA.md",
            ROOT / "src/0_6_p2/SOMA_CELL_0_6_P2_pythonista.py": vault_root / "canonical/SOMA_CELL_0_6_P2_pythonista.py",
            ROOT / "src/0_6_p2/SOMA_CELL_0_6_P2_validation.py": vault_root / "canonical/SOMA_CELL_0_6_P2_validation.py",
            ROOT / "src/0_6_p2/SOMA_CELL_0_6_P2_experiment.py": vault_root / "canonical/SOMA_CELL_0_6_P2_experiment.py",
            ROOT / "src/0_6_p2/SOMA_CELL_0_6_P2_report.py": vault_root / "canonical/SOMA_CELL_0_6_P2_report.py",
            ROOT / "results/SOMA_CELL_0_6_P2_VALIDATION_RESULTS.txt": vault_root / "results/SOMA_CELL_0_6_P2_VALIDATION_RESULTS.txt",
            ROOT / "results/SOMA_CELL_0_6_P2_REGRESSION_RESULTS.txt": vault_root / "results/SOMA_CELL_0_6_P2_REGRESSION_RESULTS.txt",
            ROOT / "results/SOMA_CELL_0_6_P2_EXPERIMENT_REPORT.txt": vault_root / "results/SOMA_CELL_0_6_P2_EXPERIMENT_REPORT.txt",
            ROOT / "results/soma_cell_0_6_p2_validation.csv": vault_root / "results/soma_cell_0_6_p2_validation.csv",
            ROOT / "results/soma_cell_0_6_p2_experiment_results.csv": vault_root / "results/soma_cell_0_6_p2_experiment_results.csv",
            ROOT / "results/soma_cell_0_6_p2_experiment_results.json": vault_root / "results/soma_cell_0_6_p2_experiment_results.json",
            ROOT / "results/soma_cell_0_6_p2_experiment_summary.csv": vault_root / "results/soma_cell_0_6_p2_experiment_summary.csv",
            ROOT / "releases/SOMA_CELL_0_6_P2_GOLD_20260802.zip": vault_root / "releases/SOMA_CELL_0_6_P2_GOLD_20260802.zip",
            ROOT / "PROJECT_STATE.json": vault_root / "governance/PROJECT_STATE.json",
            ROOT / "CURRENT_BASELINE.txt": vault_root / "governance/CURRENT_BASELINE.txt",
            ROOT / "PROJECT_STATUS_JA.md": vault_root / "governance/PROJECT_STATUS_JA.md",
            ROOT / "TEST_MATRIX.csv": vault_root / "governance/TEST_MATRIX.csv",
            ROOT / "DECISION_LOG_JA.md": vault_root / "governance/DECISION_LOG_JA.md",
            ROOT / "CHANGELOG_JA.md": vault_root / "governance/CHANGELOG_JA.md",
            ROOT / "00_MANDATORY_STARTUP_GATE_JA.md": vault_root / "governance/00_MANDATORY_STARTUP_GATE_JA.md",
            ROOT / "MANDATORY_WORKFLOW.json": vault_root / "governance/MANDATORY_WORKFLOW.json",
            ROOT / "planning/ROADMAP_0_6_JA.md": vault_root / "governance/ROADMAP_0_6_JA.md",
            latest_receipt("*_TRANSITION.json"): vault_root / "governance/P2_TRANSITION_RECEIPT.json",
            latest_receipt("*_PREFLIGHT.json"): vault_root / "governance/LATEST_PREFLIGHT_RECEIPT.json",
        }
        for source, target in mapping.items():
            copy_required(source, target)
        copy_required(ROOT / "manifests/SOMA_LINEAGE.json", vault_root / "SOMA_LINEAGE.json")

        (vault_root / "README_FIRST.txt").write_text(
            """SOMA CONTINUITY VAULT — GOLD 20260802 / SOMA-CELL 0.6-P2
================================================================
現在の凍結基準: SOMA-CELL 0.6-P2.0
P2組織契約: 0.6-P2.2
身体ポート契約: 0.6-P0.2
次のマイルストーン: 正式SOMA-CELL 0.6

このVaultの目的
----------------
会話の記憶や一時sandboxに依存せず、0.5化学身体、P0身体ポート、
P1一物質神経、P2八細胞物質再帰組織、旧SOMA因果監査系を復元し、
正式0.6の物質的ABBA/BAAB監査・校正・変化ゲートを正しく続ける。

最初に読むもの
--------------
1. governance/00_MANDATORY_STARTUP_GATE_JA.md
2. governance/PROJECT_STATE.json
3. docs/SOMA_CONTEXT_HANDOFF_JA.md
4. docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md
5. docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md
6. results/SOMA_CELL_0_6_P2_VALIDATION_RESULTS.txt
7. results/SOMA_CELL_0_6_P2_EXPERIMENT_REPORT.txt

凍結する科学的限定
------------------
- holdout全機構対固定は5/6正、平均AUC +1.872678だがn=6。
- 開発seedは試験環境の校正に使った。
- 予測器の正味身体価値は未証明。
- 再帰の追加価値は極小で未証明。
- 正式0.6でもno-prediction/no-recurrenceを外さない。
- 生命、意識、オープンエンド進化を宣言しない。
""",
            encoding="utf-8",
        )
        (vault_root / "RESTORE_INSTRUCTIONS_JA.txt").write_text(
            """SOMA継続Vault 復元手順 — SOMA-CELL 0.6-P2
================================================

1. このZIP全体を新しい会話へ添付する。
2. 次を伝える。

   「governance/00_MANDATORY_STARTUP_GATE_JA.mdを最初に読み、
    governance/PROJECT_STATE.json、docs/SOMA_CONTEXT_HANDOFF_JA.md、
    docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md、
    docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.mdを正本として、
    正式SOMA-CELL 0.6を続行してください。」

3. SHA256SUMS.txtを照合する。
4. canonical/SOMA_CELL_0_6_P2_pythonista.pyとP2検証・実験結果を再読する。
5. P0 22/22、P1 18/18、P2 22/22を回帰検査として維持する。
6. 正式0.6開始前に新しいGit HEADでプリフライト証跡を発行する。
""",
            encoding="utf-8",
        )

        for name in ("SHA256SUMS.txt", "VAULT_MANIFEST.json"):
            path = vault_root / name
            if path.exists():
                path.unlink()
        files = sorted(path for path in vault_root.rglob("*") if path.is_file())
        records = [
            {
                "path": path.relative_to(vault_root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ]
        manifest = {
            "vault": VAULT_NAME,
            "created": "2026-08-02",
            "current_baseline": "SOMA-CELL 0.6-P2.0",
            "p2_tissue_contract": "0.6-P2.2",
            "body_port_contract": "0.6-P0.2",
            "next_milestone": "SOMA-CELL 0.6",
            "validation": "P0 22/22; P1 18/18; P2 22/22 PASS",
            "comparison_trials": 36,
            "holdout_result": "5/6 positive, mean AUC +1.872678",
            "file_count_excluding_manifest_and_checksum": len(records),
            "files": records,
        }
        (vault_root / "VAULT_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        files = sorted(
            path for path in vault_root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (vault_root / "SHA256SUMS.txt").write_text(
            "\n".join(
                "{}  ./{}".format(sha256(path), path.relative_to(vault_root).as_posix())
                for path in files
            ) + "\n",
            encoding="utf-8",
        )

        temporary_zip = OUT.with_suffix(".zip.tmp")
        if temporary_zip.exists():
            temporary_zip.unlink()
        if OUT.exists():
            OUT.unlink()
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(vault_root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(VAULT_NAME) / path.relative_to(vault_root))
        temporary_zip.replace(OUT)
        with zipfile.ZipFile(OUT) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError("corrupt vault member: {}".format(bad))
    print("built {} ({} bytes, sha256={})".format(
        OUT.relative_to(ROOT), OUT.stat().st_size, sha256(OUT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
