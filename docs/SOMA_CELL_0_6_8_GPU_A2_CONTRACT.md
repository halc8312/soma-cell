# SOMA-CELL 0.6.8-GPU A2 正式工学契約

## 目的

SOMA-CELL 0.6.6の詳細粒子世界を粗視化せず、A1のtensor基盤へ表面交換・漏出・半径・運動を段階移植する。A2は完全GPU世界ではない。

## 凍結公理

1. 局所膜材料と膜酸化から閉鎖度が生じる。
2. 内外輸送は粒子位置、膜区画、輸送体、ATP、濃度勾配から計算する。
3. 燃料・鉱物・廃棄物・代替基質の処理順序を0.6.6から変更しない。
4. ATP不足時も促進拡散は残るが、能動ポンプ分は実ATPを必要とする。
5. 漏出した物質は環境粒子へ移り、ATP損失は散逸へ入る。
6. Brownian乱数はA2ではCPU正本のRNG順序を維持する。
7. GPU化を理由に膜、粒子、DNA、死体、HGT、神経を削除しない。
8. 未移植機構がある限り `full_gpu_world_step=false` と表示する。
9. fp32実験ではfp64との差を測定し、暗黙に同等と扱わない。
10. 外部fitness、報酬、正解方向を追加しない。

## A2でhybrid統合した範囲

- A1 particle diffusion / patch drift
- A1 membrane-local ligand profiles
- toroidal spatial candidate index
- surface exchange（fuel/mineral/waste/alt）
- waste export
- leak fraction / target segment / material plan
- radius relaxation
- motion

漏出粒子の生成と乱数消費は、イベント順序を守るためCPU正本を使う。

## CPU正本に残る範囲

- 代謝・損傷系の全統合
- genome replication
- translation / expression
- division
- death / corpse / eDNA / HGT
- neural / causal system

## 合格条件

- A2専用32/32 PASS
- A1 32/32再実行PASS
- inherited 279/279維持
- fp64で一歩・十歩・周期環境のイベントロックステップ
- RNG一致
- clone/save-restore一致
- 物質台帳許容範囲
- 完全GPU・CUDA性能を誤表示しない

## 科学的限定

A2は工学基盤であり、生命、進化、神経優位、CUDA高速化を証明しない。
