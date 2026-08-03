# STOP — Mandatory preflight before any SOMA work

会話記憶だけから実装・修正・評価してはならない。`00_MANDATORY_STARTUP_GATE_JA.md`を読み、最新版のSHA-256確認とプリフライト証跡を作成する。

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<name>" \
  --purpose "<work>" \
  --milestone "<PROJECT_STATE next_milestone>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

# SOMA Project Checkpoint — SOMA-CELL 0.6.1

現在の凍結基準版は **SOMA-CELL 0.6.1**、次は **SOMA-CELL 0.6.2**。

- Runtime: `src/0_6_1/SOMA_CELL_0_6_1_pythonista.py`
- Formal contract: `docs/SOMA_CELL_0_6_1_FORMAL_CONTRACT.md`
- Formal schema: `docs/SOMA_CELL_0_6_1_FORMAL_SCHEMA.json`
- Preregistration: `results/SOMA_CELL_0_6_1_R2_PREREGISTRATION.json`
- Holdout: `results/SOMA_CELL_0_6_1_R2_HOLDOUT_REPORT.txt`
- Validation: `results/SOMA_CELL_0_6_1_VALIDATION_RESULTS.txt` — 43/43 PASS
- Inherited regressions: `results/SOMA_CELL_0_6_1_REGRESSION_RESULTS.txt` — 84/84 PASS
- Pythonista release: `releases/SOMA_CELL_0_6_1_GOLD_20260803.zip`

0.6.1の強い結果は、事前登録された物理的運動器故障での有料診断と神経休眠による代謝保全です。自然空間の短期物質取得、外部分子の意味診断、局所予測、再帰の純価値は未確立です。

## Authority order

1. Validated 0.6.1 source, R2 preregistration and measured results
2. 0.6.1 formal contract/schema
3. Formal 0.6 and P2/P1/P0 frozen contracts
4. Mandatory startup gate and integration contract
5. PROJECT_STATE and context handoff
6. Conversation memory
