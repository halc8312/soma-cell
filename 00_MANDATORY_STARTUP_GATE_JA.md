# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 失敗閉鎖

次のどれかが欠ける場合は作業を始めない。

1. `PROJECT_STATE.json`、`CURRENT_BASELINE.txt`、継続ハンドオフ、統合契約を読めない。
2. 0.6.1基準ソース、契約、スキーマ、事前登録、検証、回帰、holdout結果を読めない。
3. SHA-256照合に失敗する。
4. 次マイルストーン`SOMA-CELL 0.6.2`と目的が一致しない。
5. 既知の負の結果・未達を確認していない。
6. 現在Git HEADに結び付く新しいプリフライト証跡を作れない。

## 必須読み順 — 0.6.2開始時

1. `00_MANDATORY_STARTUP_GATE_JA.md`
2. `MANDATORY_WORKFLOW.json`
3. `PROJECT_STATE.json`
4. `CURRENT_BASELINE.txt`
5. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
6. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
7. `docs/SOMA_CELL_0_6_1_FORMAL_CONTRACT.md`
8. `docs/SOMA_CELL_0_6_1_FORMAL_SCHEMA.json`
9. `src/0_6_1/SOMA_CELL_0_6_1_pythonista.py`
10. 0.6.1検証・回帰・R2事前登録・R2 holdout・実験レポート
11. `planning/ROADMAP_0_6_JA.md`
12. 親0.6/P2/P1/P0契約と基準ソース
13. 移植元SOMA-2.1、SOMA-4.2

## 機械ゲート

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<作業者>" \
  --purpose "<目的>" \
  --milestone "<PROJECT_STATEのnext_milestone>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## 0.6.2の凍結条件

- 0.6.1の物質会計、安定安全性、故障確認、期限付き神経保全リースを落とさない。
- 予測・再帰・区画数・診断機構を「賢そうだから」残さず、費用込み純価値で判定する。
- 0.6.1の主結果が身体中心の制御浴であり、自然空間の短期取得差が0だった事実を出発点にする。
- 局所予測と再帰の正味価値は未確立。アブレーションを必ず残す。
- 神経ATP／信号エスクロー上限と保存的返却を維持する。
- 外部reward、正解方向、無料計算、無料継承を再導入しない。
- モジュールの費用を回収できなければ縮小・休眠・棄却する。

## 禁止

- 最新版確認なしの旧コードコピー。
- reward/fitness/正解方向の再導入。
- 神経出力による位置・ATP・膜・DNA直接変更。
- 学習状態、神経物質、遺伝子の無料コピー。
- R1失敗、会計混入、未達を隠した完成宣言。
- 生命・意識・新規性の未検証断定。

## 完了時の必須更新

`PROJECT_STATE.json`、状態文書、テスト行列、判断ログ、変更履歴、SHA-256、Gitコミット・タグ、Continuity Vault、Google Driveアーカイブ。

## 権威順位

1. 検証済み0.6.1基準ソース、R2事前登録、実測結果
2. 0.6.1正式契約・スキーマ
3. 0.6/P2/P1/P0凍結契約
4. 本ゲートと統合契約
5. PROJECT_STATE、継続ハンドオフ
6. 会話記憶

## Rule Lock R2

証跡は現在HEADと次マイルストーンに一致する必要がある。pre-commitは`verify`と`check-receipt`を実行し、コミット後は旧証跡が失効する。
