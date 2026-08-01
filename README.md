# STOP — Mandatory preflight before any SOMA work

SOMAを会話記憶だけから実装・修正・評価してはならない。まず`00_MANDATORY_STARTUP_GATE_JA.md`を読み、`PROJECT_STATE.json`の正確な次マイルストーンに対して失敗閉鎖プリフライトを実行する。

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<name>" \
  --purpose "<work>" \
  --milestone "<PROJECT_STATE next_milestone>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

検証または復元に失敗した場合、欠落実装を推測せず停止する。

---

# SOMA Project Checkpoint — SOMA-CELL 0.6-P1 baseline

現在の凍結基準版は **SOMA-CELL 0.6-P1.0**、次の未実装マイルストーンは **SOMA-CELL 0.6-P2**。

## P1 canonical artifacts

- Runtime: `src/0_6_p1/SOMA_CELL_0_6_P1_pythonista.py`
- Frozen tissue contract: `docs/SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md`
- Tissue schema: `docs/SOMA_CELL_0_6_P1_TISSUE_SCHEMA.json` (`0.6-P1.1`)
- Frozen body-port contract: `docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md` (`0.6-P0.2`)
- Detailed README: `docs/SOMA_CELL_0_6_P1_README_JA.md`
- Validation: `results/SOMA_CELL_0_6_P1_VALIDATION_RESULTS.txt` — 18/18 PASS
- P0 regression: 22/22 PASS
- Engineering experiment: `results/SOMA_CELL_0_6_P1_EXPERIMENT_REPORT.txt` — 24 trials
- Pythonista release: `releases/SOMA_CELL_0_6_P1_GOLD_20260801.zip`
- Integration contract: `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
- Machine state: `PROJECT_STATE.json`

P1 contains one gene-built material neural compartment and an equal-material, equal-maintenance non-informational dummy. In moving-patch tests, the informational compartment beat the dummy in 3/3 paired seeds. The local predictor learned its series but did not establish positive net behavioural value; P2 must preserve the no-prediction ablation.

## Authority order

1. Validated P1 canonical source and measured results
2. Frozen P1 tissue contract
3. Frozen P0 body-port contract
4. `00_MANDATORY_STARTUP_GATE_JA.md`
5. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
6. `PROJECT_STATE.json`
7. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
8. Conversation memory and summaries
