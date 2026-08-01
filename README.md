# STOP — Mandatory preflight before any SOMA work

Do not implement, modify, or evaluate SOMA from memory alone. First read `00_MANDATORY_STARTUP_GATE_JA.md`, then run:

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start --actor "<name>" --purpose "<work>" --milestone "SOMA-CELL 0.6-P0" --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

If verification fails or no preflight receipt is created, stop. Restore the latest checkpoint from the project archive rather than inferring missing details.

---

# SOMA Project Checkpoint — 2026-08-01

This repository is the continuity source of truth for the SOMA artificial-life project at the transition from **SOMA-CELL 0.5** to **SOMA-CELL 0.6-P0**.

## Current baseline

- Canonical runtime: `src/baseline/SOMA_CELL_0_5_pythonista.py`
- Baseline release: `releases/SOMA_CELL_0_5_GOLD_20260801.zip`
- Continuity vault: `archives/SOMA_CONTINUITY_VAULT_GOLD_20260801_0_5.zip`
- Integration contract: `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
- Machine-readable state: `PROJECT_STATE.json`
- Next implementation plan: `planning/ROADMAP_0_6_JA.md`
- Resume prompt: `RESUME_PROMPT_JA.txt`

## Authority order

When prose, memory, and code disagree, use this order:

1. Canonical baseline source and its tests/results
2. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
3. `PROJECT_STATE.json`
4. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
5. Conversation summaries

## Current next step

Implement **SOMA-CELL 0.6-P0**, a read-only chemical-body sensor API plus conserved neural budget/effector interfaces, while preserving exact 0.5 behavior when no neural tissue is attached.
