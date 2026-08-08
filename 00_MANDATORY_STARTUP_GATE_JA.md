# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 失敗閉鎖

次のどれかが欠ける場合は作業を始めない。

1. `PROJECT_STATE.json`、`CURRENT_BASELINE.txt`、継続ハンドオフ、統合契約を読めない。
2. 0.6.5基準ソース、契約、スキーマ、事前登録、検証、回帰、実験結果を読めない。
3. SHA-256照合に失敗する。
4. 次マイルストーン`SOMA-CELL 0.6.6`と目的が一致しない。
5. 0.6.5の部分成功と失敗（stable shrinkage 1/3）を確認していない。
6. 現在Git HEADに結び付く新しいプリフライト証跡を作れない。

## 必須読み順 — 0.6.6開始時

1. `00_MANDATORY_STARTUP_GATE_JA.md`
2. `MANDATORY_WORKFLOW.json`
3. `PROJECT_STATE.json`
4. `CURRENT_BASELINE.txt`
5. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
6. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
7. `docs/SOMA_CELL_0_6_5_FORMAL_CONTRACT.md`
8. `docs/SOMA_CELL_0_6_5_FORMAL_SCHEMA.json`
9. `src/0_6_5/SOMA_CELL_0_6_5_pythonista.py`
10. 0.6.5事前登録・検証・回帰・実験レポート・release record
11. `planning/ROADMAP_0_6_JA.md`
12. 親0.6.4契約と基準ソース

## 機械ゲート

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<作業者>" \
  --purpose "<目的>" \
  --milestone "<PROJECT_STATEのnext_milestone>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## 0.6.6の凍結条件

- 加速トーナメント選択を自然選択と呼ばない。
- 実際の分裂・構造的死・資源競争・死体化学を共有世界で起こす。
- 個体へ外部fitness/reward/正解方向を与えない。
- stableでの文法縮小とcomplexでの保持を未調整複数系列で比較する。
- HGT on/off、mutation off、grammar knockout/reintroductionを必須対照にする。
- 0.6.5のstable shrinkage失敗1/3を隠さない。
- 無料遺伝・無料神経・無料物質・学習状態の無料コピーを再導入しない。

## 禁止

- 最新版確認なしの旧コードコピー。
- reward/fitness/正解方向の個体への再導入。
- 実験者のトーナメント置換を自然淘汰と呼ぶ。
- 神経出力による位置・ATP・膜・DNA直接変更。
- 学習状態、神経物質、遺伝子の無料コピー。
- 負の結果や未達を隠した完成宣言。
- 生命・意識・オープンエンド進化の未検証断定。

## 完了時の必須更新

`PROJECT_STATE.json`、状態文書、テスト行列、判断ログ、変更履歴、SHA-256、Gitコミット・タグ、Continuity Vault。Drive容量が不足する場合はローカル完全チェックポイントを正本として明記する。

## 権威順位

1. 検証済み0.6.5基準ソース、事前登録、実測結果
2. 0.6.5正式契約・スキーマ
3. 0.6.4以前の凍結契約
4. 本ゲートと統合契約
5. PROJECT_STATE、継続ハンドオフ
6. 会話記憶

## Rule Lock R2

証跡は現在HEADと次マイルストーンに一致する必要がある。pre-commitは`verify`と`check-receipt`を実行し、コミット後は旧証跡が失効する。
