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

# SOMA Project Checkpoint — SOMA-CELL 0.6-P0 baseline

現在の凍結基準版は **SOMA-CELL 0.6-P0.0**、次の未実装マイルストーンは **SOMA-CELL 0.6-P1**。

## P0 canonical artifacts

- Runtime: `src/0_6_p0/SOMA_CELL_0_6_P0_pythonista.py`
- Frozen body-port contract: `docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md`
- Detailed API: `docs/SOMA_CELL_0_6_P0_API_CONTRACT.md`
- Port schema: `docs/SOMA_CELL_0_6_P0_PORT_SCHEMA.json` (`0.6-P0.2`)
- Detailed README: `docs/SOMA_CELL_0_6_P0_README_JA.md`
- Validation: `results/SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt` — 22/22 PASS
- Engineering experiment: `results/SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt` — 15 trials
- Pythonista release: `releases/SOMA_CELL_0_6_P0_GOLD_20260801.zip`
- Integration contract: `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
- Machine state: `PROJECT_STATE.json`

P0 contains no information-processing neuron. It freezes read-only physical sensing, finite chemical budgets, bounded physical effectors, conservative tissue return, a non-mutable public port surface, and exact 0.5 compatibility for P1.

## Authority order

1. Validated canonical source and measured results
2. Frozen P0 body-port contract
3. `00_MANDATORY_STARTUP_GATE_JA.md`
4. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
5. `PROJECT_STATE.json`
6. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
7. Conversation memory and summaries
