#!/usr/bin/env python3
"""Build continuity vault for frozen formal SOMA-CELL 0.6.0."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, shutil, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='SOMA_CONTINUITY_VAULT_GOLD_20260802_0_6'
OUT=ROOT/'archives'/(NAME+'.zip')

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def copy_file(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def copy_tree(src:Path,dst:Path):
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='soma06_vault_') as td:
        base_zip=ROOT/'archives/SOMA_CONTINUITY_VAULT_GOLD_20260802_0_6_P2.zip'
        with zipfile.ZipFile(base_zip) as base:
            base.extractall(Path(td)/'base')
        old_root=next((Path(td)/'base').iterdir())
        v=Path(td)/NAME
        shutil.copytree(old_root,v)
        for d in ('0_6_p0','0_6_p1','0_6_p2','0_6'):
            copy_tree(ROOT/'src'/d,v/'source'/d)
        for name in (
            'SOMA_CONTEXT_HANDOFF_JA.md','SOMA_CELL_0_6_INTEGRATION_CONTRACT.md',
            'SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md','SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md',
            'SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md','SOMA_CELL_0_6_FORMAL_CONTRACT.md',
            'SOMA_CELL_0_6_FORMAL_SCHEMA.json','SOMA_LINEAGE_INDEX_JA.md'):
            copy_file(ROOT/'docs'/name,v/'docs'/name)
        for name in (
            '00_MANDATORY_STARTUP_GATE_JA.md','MANDATORY_WORKFLOW.json','PROJECT_STATE.json',
            'CURRENT_BASELINE.txt','PROJECT_STATUS_JA.md','DECISION_LOG_JA.md','CHANGELOG_JA.md',
            'TEST_MATRIX.csv','README.md','RESUME_PROMPT_JA.txt'):
            copy_file(ROOT/name,v/'governance'/name)
        for p in sorted((ROOT/'results').glob('SOMA_CELL_0_6*'))+sorted((ROOT/'results').glob('soma_cell_0_6*')):
            if p.is_file(): copy_file(p,v/'results'/p.name)
        copy_file(ROOT/'releases/SOMA_CELL_0_6_GOLD_20260802.zip',v/'release/SOMA_CELL_0_6_GOLD_20260802.zip')
        copy_file(ROOT/'manifests/SOMA_LINEAGE.json',v/'governance/SOMA_LINEAGE.json')
        copy_file(ROOT/'planning/ROADMAP_0_6_JA.md',v/'planning/ROADMAP_0_6_JA.md')
        (v/'README_FIRST.txt').write_text(
            'SOMA Continuity Vault — SOMA-CELL 0.6.0\n\n'
            'Current baseline: SOMA-CELL 0.6.0 (engineering PASS, science PASS_WITH_LIMITS)\n'
            'Next milestone: SOMA-CELL 0.6.1\n'
            'First read governance/00_MANDATORY_STARTUP_GATE_JA.md, governance/PROJECT_STATE.json, '
            'docs/SOMA_CONTEXT_HANDOFF_JA.md and docs/SOMA_CELL_0_6_FORMAL_CONTRACT.md.\n',encoding='utf-8')
        (v/'RESTORE_INSTRUCTIONS_JA.txt').write_text(
            '1. ZIP全体を新しい会話へ添付。\n'
            '2. governance/PROJECT_STATE.jsonと必須ゲート、formal contractを正本として0.6.1を続行と指定。\n'
            '3. SHA256SUMS.txtを照合。\n'
            '4. 自然な自動因果循環が0.6で未達だった事実を成功済みにしない。\n'
            '5. 新しいGit HEADでプリフライト証跡を発行。\n',encoding='utf-8')
        files=sorted(p for p in v.rglob('*') if p.is_file())
        manifest={'vault':NAME,'created':'2026-08-02','current_baseline':'SOMA-CELL 0.6.0',
                  'next_milestone':'SOMA-CELL 0.6.1','engineering':'PASS','science':'PASS_WITH_LIMITS',
                  'validation':'formal 22/22; total 84/84','comparison_trials':33,
                  'automatic_chain':'not established','files':[]}
        for p in files:
            manifest['files'].append({'path':p.relative_to(v).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size})
        (v/'VAULT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        files=sorted(p for p in v.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
        (v/'SHA256SUMS.txt').write_text('\n'.join(f'{sha256(p)}  ./{p.relative_to(v).as_posix()}' for p in files)+'\n',encoding='utf-8')
        tmp=OUT.with_suffix('.zip.tmp'); tmp.unlink(missing_ok=True); OUT.unlink(missing_ok=True)
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(v.rglob('*')):
                if p.is_file(): z.write(p,Path(NAME)/p.relative_to(v))
        tmp.replace(OUT)
    with zipfile.ZipFile(OUT) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(bad)
    print(OUT); print(sha256(OUT)); return 0

if __name__=='__main__': raise SystemExit(main())
