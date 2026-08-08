# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 現在地
- 凍結基準: `SOMA-CELL 0.6.7`
- 次: `SOMA-CELL 0.6.8`
- 0.6.7は長期世代用の粗視化物質計器であり、0.6.6粒子物理の代替ではない。

## 失敗閉鎖
次のどれかが欠ける場合は開始しない。
1. PROJECT_STATE、CURRENT_BASELINE、handoff、統合契約を読めない。
2. 0.6.7ソース、正式契約、R3事前登録、12行holdout、28検証、251回帰を読めない。
3. SHA-256照合に失敗する。
4. 次マイルストーン0.6.8と目的が一致しない。
5. R3の失敗（stable loss 0/3、strict HGT rescue 1/3、long-delay差+0.388889）を確認していない。
6. 現在HEADに結び付く新しいpreflightを発行できない。

## 必須読み順
`MANDATORY_WORKFLOW.json`の`required_read_order`を厳守する。

## 機械ゲート
```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start --actor "<作業者>" --purpose "<目的>" --milestone "SOMA-CELL 0.6.8" --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## 0.6.8凍結条件
- 同一系統史のstable burn-in→regime shiftを用いる。
- HGT/変異によるloss→re-entry→翻訳→子孫持続を実配列と物質費で追う。
- 外部fitness、reward、正解方向、切替通知、実験者系統コピーを導入しない。
- HGT OFF、mutation OFF、grammar absent、no-switch対照を事前登録する。
- 結果がnegativeでも全系列を報告する。

## 禁止
- 0.6.7の失敗を隠して神経有利に再調整する。
- 粗視化0.6.7を完全粒子0.6.6と同一視する。
- 学習状態、遺伝子、ATP、物質の無料コピー。
- 未検証の生命・意識・開放進化断定。

## Rule Lock
証跡は現在HEADとnext milestoneに一致する必要がある。コミット後、旧証跡は意図的に失効する。
