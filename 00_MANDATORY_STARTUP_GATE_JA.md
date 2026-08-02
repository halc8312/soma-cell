# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 失敗閉鎖

次のどれかが欠ける場合は作業を始めない。

1. `PROJECT_STATE.json`、`CURRENT_BASELINE.txt`、継続ハンドオフ、統合契約を読めない。
2. 現在基準のソース、契約、スキーマ、検証、実験結果を読めない。
3. SHA-256照合に失敗する。
4. 次マイルストーンと目的が一致しない。
5. 既知の負の結果・未達を確認していない。
6. 現在Git HEADに結び付くプリフライト証跡を作れない。

## 必須読み順 — 0.6.1開始時

1. `00_MANDATORY_STARTUP_GATE_JA.md`
2. `MANDATORY_WORKFLOW.json`
3. `PROJECT_STATE.json`
4. `CURRENT_BASELINE.txt`
5. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
6. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
7. `docs/SOMA_CELL_0_6_FORMAL_CONTRACT.md`
8. `docs/SOMA_CELL_0_6_FORMAL_SCHEMA.json`
9. `src/0_6/SOMA_CELL_0_6_pythonista.py`
10. 0.6検証・回帰・33試行レポート
11. `planning/ROADMAP_0_6_JA.md`
12. P2/P1/P0契約と基準ソース
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

## 0.6.1の凍結条件

- 0.6の物質的監査・校正・変化源分離・費用・死体/HGTを落とさない。
- `state`変動だけで強い再可塑化を開かない。
- 0.6の自動短期反転ではフィードバック0件だった事実を出発点にする。
- 既知反証の身体余裕利益は3/6・微小であり、成功済みと扱わない。
- 予測・再帰の正味価値は未確立。アブレーションを残す。
- 診断行動にもATP・材料・身体危険を課す。
- 自然な自動循環が費用込みツイン利益を示すまで0.6.1を完成扱いにしない。

## 禁止

- 最新版確認なしの旧コードコピー。
- reward/fitness/正解方向の再導入。
- 神経出力による位置・ATP・膜・DNA直接変更。
- 学習状態、神経物質、遺伝子の無料コピー。
- 安定安全性や負の結果を隠した完成宣言。
- 生命・意識・新規性の未検証断定。

## 完了時の必須更新

`PROJECT_STATE.json`、状態文書、テスト行列、判断ログ、変更履歴、SHA-256、Gitコミット・タグ、Continuity Vault、Google Driveアーカイブ。

## 権威順位

1. 検証済み0.6基準ソースと実測結果
2. 0.6正式契約・スキーマ
3. P2/P1/P0凍結契約
4. 本ゲートと統合契約
5. PROJECT_STATE、継続ハンドオフ
6. 会話記憶

## Rule Lock R2

証跡は現在HEADと次マイルストーンに一致する必要がある。pre-commitは`verify`と`check-receipt`を実行し、コミット後は旧証跡が失効する。

マイルストーン昇格にはクリーンな完了コミット上で`transition`証跡を発行する。
