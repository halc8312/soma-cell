# SOMA簡略変更履歴

- SOMA-0: 単体・再帰神経・生涯学習
- SOMA-1: 因果栄養組織、複数身体通貨
- SOMA-2/2.1: 実停止監査、形態実験、期限付き器官
- SOMA-3: 内生的問い生成
- SOMA-4/4.1/4.2: 反証コンパイラ、ツイン評価、変化ゲート
- SOMA-5R/6R/7R: 内的計画、共有文化、生態系の品質再構築
- SOMA-CELL 0.1: 動的膜と自己生産
- 0.2: 物質ゲノムと変異
- 0.3: 損傷・修復・機能年齢
- 0.4: 物質感覚運動と局所可塑性
- 0.5: 死体化学、環境DNA、HGT、可動配列
- 次: 0.6-P0 化学身体API凍結

## 2026-08-01 — Project Rule Lock 1

- `00_MANDATORY_STARTUP_GATE_JA.md`を追加。
- `MANDATORY_WORKFLOW.json`を追加。
- `scripts/soma_preflight.py`を追加し、必須ファイル・SHA-256・マイルストーンを失敗閉鎖で検査。
- `work_sessions/*_PREFLIGHT.json`による作業開始証跡を導入。
- `.githooks/pre-commit`を追加し、実装変更にプリフライト証跡を要求。
- `PROJECT_STATE.json`へ`workflow_guard`を追加。
- READMEと復元手順へプリフライトを最優先として追記。
