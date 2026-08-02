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

# SOMA Project Checkpoint — SOMA-CELL 0.6.0

現在の凍結基準版は **SOMA-CELL 0.6.0**、次は **SOMA-CELL 0.6.1**。

- Runtime: `src/0_6/SOMA_CELL_0_6_pythonista.py`
- Formal contract: `docs/SOMA_CELL_0_6_FORMAL_CONTRACT.md`
- Formal schema: `docs/SOMA_CELL_0_6_FORMAL_SCHEMA.json`
- Validation: `results/SOMA_CELL_0_6_VALIDATION_RESULTS.txt` — 22/22 PASS
- Total regressions: `results/SOMA_CELL_0_6_REGRESSION_RESULTS.txt` — 84/84 PASS
- Experiment: `results/SOMA_CELL_0_6_EXPERIMENT_REPORT.txt` — 33 trials
- Pythonista release: `releases/SOMA_CELL_0_6_GOLD_20260802.zip`

0.6は工学統合PASSだが、自然な自動監査→反証→再可塑化→利益の循環は未達。0.6.1はこの一点を扱い、概念・文化・大規模生態系はまだ戻さない。

## Authority order

1. Validated formal 0.6 source and measured results
2. Formal 0.6 material causal contract/schema
3. P2/P1/P0 frozen contracts
4. Mandatory startup gate and integration contract
5. PROJECT_STATE and context handoff
6. Conversation memory
