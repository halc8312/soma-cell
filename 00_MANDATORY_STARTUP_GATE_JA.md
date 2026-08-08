# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 現在地
- 工学凍結基準: `SOMA-CELL 0.6.8-GPU A2`
- 次: `SOMA-CELL 0.6.8-GPU A3`
- 詳細物理の科学正本: `SOMA-CELL 0.6.6`
- 長期粗視化計器: `SOMA-CELL 0.6.7`
- A2はfull GPU worldではない。CUDA/RTX実機未検証。
- CPU-only比較ではA2 hybridは凍結CPU版より約1.524倍遅い。速度結果として扱わない。

## 失敗閉鎖
次のどれかが欠ける場合は開始しない。
1. PROJECT_STATE、CURRENT_BASELINE、handoff、統合契約を読めない。
2. A2 source、A2契約、32検証、311継承記録、engineering reportを読めない。
3. A1 source/contract、0.6.6 particle source/contract、0.6.7 coarse scopeを読めない。
4. SHA-256照合に失敗する。
5. 次マイルストーンA3と目的が一致しない。
6. 現在HEADに結び付く新しいpreflightを発行できない。

## 必須読み順
`MANDATORY_WORKFLOW.json`の`required_read_order`を厳守する。

## 機械ゲート
```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start --actor "<作業者>" --purpose "<目的>" --milestone "SOMA-CELL 0.6.8-GPU A3" --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## A3凍結条件
- gene-coded metabolism、損傷・修復、膜酸化、aggregate composition、damage segregationを独立NumPy/Torch fp64で照合する。
- A2表面層とのexport/leak/radius/motion二重呼出しを統合event schedulerで防ぐ。
- CPU正本の反応順序、ATP・材料費、物質台帳、RNG、clone/save-restoreを維持する。
- CUDA利用可能時のみRTX/CUDA性能を測定し、CPU-only結果と区別する。
- fp32はfp64との差を保存する。
- genome、division、death/eDNA/HGT、neural等が未移植なら`full_gpu_world_step=false`を維持する。

## 禁止
- 詳細物理・損傷化学を粗視化へ置換して高速化と呼ぶ。
- 容量超過した粒子・ゲノム・タンパク質辞書を黙って切り捨てる。
- 未移植機構をGPU実装済みと表示する。
- 学習状態、遺伝子、ATP、修復材料の無料コピー。
- 外部fitness、reward、正解方向、切替時刻。
- CUDA未実測性能、生命、意識、開放進化の断定。

## Rule Lock
証跡は現在HEADとnext milestoneに一致する必要がある。コミット後、旧証跡は意図的に失効する。
