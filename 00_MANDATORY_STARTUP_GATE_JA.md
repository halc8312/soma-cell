# SOMA 必須スタートアップ・ゲート（最優先）

**実装・修正・評価の前に必ず読む。会話記憶だけから開始しない。**

## 現在地
- 工学凍結基準: `SOMA-CELL 0.6.8-GPU A3`
- 次: `SOMA-CELL 0.6.8-GPU A4`
- 詳細物理の科学正本: `SOMA-CELL 0.6.6`
- 長期粗視化計器: `SOMA-CELL 0.6.7`
- A3はfull GPU worldではない。
- NVIDIA GeForce RTX 4060 Ti 16GB上でCUDA fp64を実測し、正確性ゲートを通過した。
- fp32はcandidate-onlyであり、fp64の意味論正本を置き換えない。
- 正式world sweepではA3 CUDA fp64は凍結0.6.6 CPU正本より570.096〜590.276倍遅い。速度向上を主張しない。

## 失敗閉鎖
次のどれかが欠ける場合は開始しない。
1. PROJECT_STATE、CURRENT_BASELINE、handoff、統合契約を読めない。
2. A3 core、scheduler、契約、schema、event order、36専用検証、343継承検証、benchmark、fp32 discrepancy、engineering reportを読めない。
3. A2 source/contract/results、A1 source/contract、0.6.6 particle source/contract、0.6.7 coarse scopeを読めない。
4. SHA-256照合に失敗する。
5. 次マイルストーンA4と目的が一致しない。
6. 現在HEADに結び付く新しいpreflightを発行できない。

## 必須読み順
`MANDATORY_WORKFLOW.json`の`required_read_order`を厳守する。

## 機械ゲート
```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start --actor "<作業者>" --purpose "<目的>" --milestone "SOMA-CELL 0.6.8-GPU A4" --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

## A4凍結条件
- variable-length genomeを黙って固定長化せず、losslessなdevice-neutral ragged genome buffersへ移す。
- translation、replication、proofreading、mutation、material mutationを独立NumPy/Torch fp64で照合する。
- genome symbol順、dict挿入順、ATP・monomer・材料支払い、proofreading/mutation RNG順をCPU正本から変えない。
- A3 exactly-once schedulerへ統合し、A3/A2処理との二重実行を防ぐ。
- exact-capacityは通し、capacity+1、malformed offsets、tail不整合はstate mutation前にfail closedとする。
- 対象機はGPU強・CPU制約である。A4の主実行経路はdevice-residentなragged stateとcell batchに置き、per-cell Python pack/commit、host-device往復、小kernel逐次起動を削減する。
- CPU実装は意味論正本・照合oracleとして保持するが、GPU移植済みphaseの通常実行をCPUへ戻して性能を成立させない。
- CPU frozen worldとの1/10 step、stressed、pre-division、clone、save/restore、物質台帳、RNGを照合する。
- CUDA fp64合格後にだけfp32候補を測り、差を独立artifactへ保存する。
- formal benchmarkでdevice residency、転送、kernel launch、GPU utilization、A3比性能を測り、GPU主経路として実用化できるかを負の結果も含めて判定する。
- actual division、death/corpse/eDNA/HGT、neural等が未移植なら`full_gpu_world_step=false`を維持する。

## 禁止
- 詳細物理・損傷化学を粗視化へ置換して高速化と呼ぶ。
- 容量超過した粒子・ゲノム・タンパク質辞書を黙って切り捨てる。
- 未移植機構をGPU実装済みと表示する。
- 学習状態、遺伝子、ATP、修復材料の無料コピー。
- genome sequence、ragged offset、translation productをclip、sort、mergeして続行する。
- DNA、タンパク質、ATP、monomer、proofreading、mutationの無料生成・無料コピー。
- CPU正本のreplication/translation/mutation RNG順をdevice reduction都合で変更する。
- 外部fitness、reward、正解方向、切替時刻。
- CUDA未実測性能、生命、意識、開放進化の断定。

## Rule Lock
証跡は現在HEADとnext milestoneに一致する必要がある。コミット後、旧証跡は意図的に失効する。
