#!/usr/bin/env python3
"""Build the SOMA-CELL 0.6-P0 continuity vault from the frozen project state."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD_VAULT = ROOT / "archives/SOMA_CONTINUITY_VAULT_GOLD_20260801_0_5.zip"
OUT = ROOT / "archives/SOMA_CONTINUITY_VAULT_GOLD_20260801_0_6_P0.zip"
VAULT_NAME = "SOMA_CONTINUITY_VAULT_GOLD_20260801_0_6_P0"


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
    if state["current_baseline"]["name"] != "SOMA-CELL 0.6-P0.0":
        raise RuntimeError("P0 continuity vault requires the promoted P0 baseline")
    if state["next_milestone"]["name"] != "SOMA-CELL 0.6-P1":
        raise RuntimeError("P0 continuity vault requires P1 as next milestone")

    with tempfile.TemporaryDirectory(prefix="soma_p0_vault_") as temporary:
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
            ROOT / "docs/SOMA_CELL_0_6_P0_API_CONTRACT.md": vault_root / "docs/SOMA_CELL_0_6_P0_API_CONTRACT.md",
            ROOT / "docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md": vault_root / "docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md",
            ROOT / "docs/SOMA_CELL_0_6_P0_PORT_SCHEMA.json": vault_root / "docs/SOMA_CELL_0_6_P0_PORT_SCHEMA.json",
            ROOT / "docs/SOMA_CELL_0_6_P0_README_JA.md": vault_root / "docs/SOMA_CELL_0_6_P0_README_JA.md",
            ROOT / "src/0_6_p0/SOMA_CELL_0_6_P0_pythonista.py": vault_root / "canonical/SOMA_CELL_0_6_P0_pythonista.py",
            ROOT / "src/0_6_p0/SOMA_CELL_0_6_P0_validation.py": vault_root / "canonical/SOMA_CELL_0_6_P0_validation.py",
            ROOT / "src/0_6_p0/SOMA_CELL_0_6_P0_experiment.py": vault_root / "canonical/SOMA_CELL_0_6_P0_experiment.py",
            ROOT / "src/0_6_p0/SOMA_CELL_0_6_P0_report.py": vault_root / "canonical/SOMA_CELL_0_6_P0_report.py",
            ROOT / "results/SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt": vault_root / "results/SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt",
            ROOT / "results/SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt": vault_root / "results/SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt",
            ROOT / "results/soma_cell_0_6_p0_validation.csv": vault_root / "results/soma_cell_0_6_p0_validation.csv",
            ROOT / "results/soma_cell_0_6_p0_experiment_results.csv": vault_root / "results/soma_cell_0_6_p0_experiment_results.csv",
            ROOT / "results/soma_cell_0_6_p0_experiment_summary.csv": vault_root / "results/soma_cell_0_6_p0_experiment_summary.csv",
            ROOT / "releases/SOMA_CELL_0_6_P0_GOLD_20260801.zip": vault_root / "releases/SOMA_CELL_0_6_P0_GOLD_20260801.zip",
            ROOT / "PROJECT_STATE.json": vault_root / "governance/PROJECT_STATE.json",
            ROOT / "CURRENT_BASELINE.txt": vault_root / "governance/CURRENT_BASELINE.txt",
            ROOT / "00_MANDATORY_STARTUP_GATE_JA.md": vault_root / "governance/00_MANDATORY_STARTUP_GATE_JA.md",
            ROOT / "MANDATORY_WORKFLOW.json": vault_root / "governance/MANDATORY_WORKFLOW.json",
            ROOT / "planning/ROADMAP_0_6_JA.md": vault_root / "governance/ROADMAP_0_6_JA.md",
            latest_receipt("*_TRANSITION.json"): vault_root / "governance/P0_TRANSITION_RECEIPT.json",
            latest_receipt("*_PREFLIGHT.json"): vault_root / "governance/LATEST_PREFLIGHT_RECEIPT.json",
        }
        for source, target in mapping.items():
            copy_required(source, target)
        copy_required(ROOT / "manifests/SOMA_LINEAGE.json", vault_root / "SOMA_LINEAGE.json")

        (vault_root / "README_FIRST.txt").write_text(
            """SOMA CONTINUITY VAULT — GOLD 20260801 / SOMA-CELL 0.6-P0
================================================================
現在の凍結基準: SOMA-CELL 0.6-P0.0
身体ポート契約: 0.6-P0.2
次のマイルストーン: SOMA-CELL 0.6-P1

このVaultの目的
----------------
会話コンテキスト、sandboxリンク、一時添付に依存せず、P0の検証済み
化学身体ポートとSOMA-0〜0.5の系譜を復元し、P1の1物質神経細胞を
正しい身体境界上で実装できる状態を保存する。

新しい会話で再開するとき
------------------------
1. このZIP全体を添付する。
2. governance/00_MANDATORY_STARTUP_GATE_JA.mdを最初に読む。
3. governance/PROJECT_STATE.json、docs/SOMA_CONTEXT_HANDOFF_JA.md、
   docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md、
   docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.mdを正本として扱う。
4. SHA256SUMS.txtを照合する。
5. P1実装前にP0本体・22項目検証・15試行レポートを再読し、
   最新Git HEADで新しいプリフライト証跡を発行する。

凍結ルール
----------
- P0は情報処理神経を含まない。適応利益を主張しない。
- P1組織はChemicalBodyPortの公開API以外から身体へアクセスしない。
- ATP・材料・信号・損傷・死体化学・eDNA・HGTを無料化しない。
- 未接続／Null時の0.5ロックステップとP0 22/22検証を回帰試験として維持する。
- 旧SOMA-5〜7（Rなし）は統合元にしない。

重要
----
ChatGPTの会話コンテキストやsandboxは永久保管を保証しない。このZIPを
ユーザー自身のGoogle Drive、iCloud Drive、Filesアプリ、PC等にも保存する。
""",
            encoding="utf-8",
        )
        (vault_root / "RESTORE_INSTRUCTIONS_JA.txt").write_text(
            """SOMA継続Vault 復元手順 — SOMA-CELL 0.6-P0
================================================

1. SOMA_CONTINUITY_VAULT_GOLD_20260801_0_6_P0.zipを添付します。
2. 次を伝えます。

   「governance/00_MANDATORY_STARTUP_GATE_JA.mdを最初に読み、
    governance/PROJECT_STATE.json、docs/SOMA_CONTEXT_HANDOFF_JA.md、
    docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md、
    docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.mdを正本として、
    SOMA-CELL 0.6-P1を続行してください。」

3. SHA256SUMS.txtを照合します。
4. P1開始前にcanonical/SOMA_CELL_0_6_P0_pythonista.py、
   results/SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt、
   results/SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txtを再読します。
5. P0 22/22回帰検査と0.5ロックステップを維持します。
6. 実機ログ、画像、Pythonista保存ファイルがあれば追加で添付します。

P1の目的
---------
P0ポートだけを通る1個の物質神経細胞を、同量・同費用の非情報処理
ダミー組織と比較する。神経が動くだけでは合格せず、少なくとも一つの
非定常環境で純身体利益がダミーを上回るか、負なら原因を確定する。
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
            "created": "2026-08-01",
            "current_baseline": "SOMA-CELL 0.6-P0.0",
            "body_port_contract": "0.6-P0.2",
            "next_milestone": "SOMA-CELL 0.6-P1",
            "validation": "22/22 PASS",
            "comparison_trials": 15,
            "contains_information_processing_neuron": False,
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
    print("built {} ({} bytes, sha256={})".format(OUT.relative_to(ROOT), OUT.stat().st_size, sha256(OUT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
