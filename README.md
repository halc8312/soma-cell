# STOP — Mandatory preflight before any SOMA work

Do not implement, modify, evaluate, or transition SOMA from memory alone. Read `00_MANDATORY_STARTUP_GATE_JA.md`, then run the fail-closed preflight for the exact milestone in `PROJECT_STATE.json`.

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<name>" \
  --purpose "<work>" \
  --milestone "<PROJECT_STATE next_milestone>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

Baseline transitions require the separate one-commit `transition` receipt described in the startup gate. If verification or restoration fails, stop instead of inferring missing details.

---

# SOMA Project Checkpoint — SOMA-CELL 0.6-P0

SOMA-CELL 0.6-P0.0 has been implemented and validated as the **pre-neural chemical-body contract layer**. The repository transition record determines whether `PROJECT_STATE.json` still names 0.5/P0 during the completion commit or has already advanced to P0/P1.

## P0 canonical artifacts

- Runtime: `src/0_6_p0/SOMA_CELL_0_6_P0_pythonista.py`
- API contract: `docs/SOMA_CELL_0_6_P0_API_CONTRACT.md`
- Detailed README: `docs/SOMA_CELL_0_6_P0_README_JA.md`
- Validation: `results/SOMA_CELL_0_6_P0_VALIDATION_RESULTS.txt` — 21/21 PASS
- Engineering experiment: `results/SOMA_CELL_0_6_P0_EXPERIMENT_REPORT.txt` — 15 trials
- Pythonista release: `releases/SOMA_CELL_0_6_P0_GOLD_20260801.zip`
- Integration contract: `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
- Machine-readable state: `PROJECT_STATE.json`

P0 contains no information-processing neuron. It freezes read-only physical sensing, finite chemical budgets, bounded physical effectors, conservative tissue return, and exact 0.5 compatibility for P1.

## Authority order

1. Validated canonical source and measured results
2. `00_MANDATORY_STARTUP_GATE_JA.md`
3. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
4. `PROJECT_STATE.json`
5. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
6. Conversation memory and summaries
