# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 現在地
- 工学凍結基準: `SOMA-CELL 0.6.8-GPU A1`
- 次: `SOMA-CELL 0.6.8-GPU A2`
- 詳細物理の科学正本: `SOMA-CELL 0.6.6`
- 長期粗視化計器: `SOMA-CELL 0.6.7`
- A1はfull GPU worldではない。CUDA実機未検証。

## 失敗閉鎖
次のどれかが欠ける場合は開始しない。
1. PROJECT_STATE、CURRENT_BASELINE、handoff、統合契約を読めない。
2. A1 source、GPU契約、32検証、279継承回帰、engineering reportを読めない。
3. 0.6.6 particle source/contractと0.6.7 coarse scopeを読めない。
4. SHA-256照合に失敗する。
5. 次マイルストーンA2と目的が一致しない。
6. 現在HEADに結び付く新しいpreflightを発行できない。

## 必須読み順
`MANDATORY_WORKFLOW.json`の`required_read_order`を厳守する。

## 機械ゲート
```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start --actor "<作業者>" --purpose "<目的>" --milestone "SOMA-CELL 0.6.8-GPU A2" --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## A2凍結条件
- surface exchange、waste export、leak、radius/motionをNumPy/Torch fp64で照合する。
- spatial hash/neighbor listはdense参照と一致する。
- 0.6.6の物質・イベント順序を保持する。
- CPU、Torch CPU、CUDA（利用可能時）を区別して報告する。
- fp32はfp64誤差を明記する。
- full GPU world未完成ならfalse表示を維持する。

## 禁止
- 詳細物理を粗視化へ置換して高速化と呼ぶ。
- 容量超過した粒子・ゲノムを黙って切り捨てる。
- 未移植機構をGPU実装済みと表示する。
- 学習状態、遺伝子、ATP、物質の無料コピー。
- 外部fitness、reward、正解方向、切替時刻。
- CUDA未実測性能、生命、意識、開放進化の断定。

## Rule Lock
証跡は現在HEADとnext milestoneに一致する必要がある。コミット後、旧証跡は意図的に失効する。
