# SOMA-CELL 0.6.8-GPU A1 正式移行契約

## 目的

SOMA-CELL 0.6.6の局所膜・粒子化学・実DNA・死体・環境DNA・HGT・神経物質制約を削らず、RTX 4060 Tiおよび将来のオンラインGPUで並列実行できるデータ構造と決定論的検証経路を作る。

## 位置づけ

0.6.8-GPU A1はGPU移行の工学基盤であり、フルGPU世界の完成版ではない。0.6.6のCPU世界が引き続き意味論上の正本である。0.6.7の粗視化長期計器とも別物である。

## 凍結不変条件

1. 膜境界は局所膜材料から生じる。
2. 内外輸送の非対称性を保つ。
3. DNA複製にはmonomer、ATP、replicaseを必要とする。
4. 死はhealthフラグではなく構造崩壊から生じる。
5. 死体・環境DNA・HGTを物質台帳へ含める。
6. 神経・記憶・監査・運動はATPと材料を消費する。
7. reward、教師ラベル、正解方向、切替時刻、外部fitnessを追加しない。
8. 学習済み重み・遺伝子・ATP・物質を無料コピーしない。
9. 粗視化で詳細物理を置換しない。
10. GPU未移植機構はCPU正本と明記し、フルGPUと表示しない。

## A1でGPU/テンソル化した範囲

- 0.6.6世界の固定容量テンソルスナップショット
- 可変長ゲノムのpadded byte mirrorと長さ配列
- 粒子位置・種類・量・マスク
- 細胞位置・速度・膜36区画・膜酸化・輸送体・13内部プール
- 決定論的counter random stream
- 粒子拡散とgeochemical drift
- 膜局所particle ligand profile
- circular smoothing
- core metabolismの独立NumPy/Torchロックステップ版
- batched independent-world kernel runner
- 0.6.6 CPU世界への安全なhybrid diffusion/profile offload

## A1でCPU正本のまま残す範囲

- surface exchange全体
- export/leak/motion/division/viabilityの統合
- variable genome replicationとtranslation
- 死亡・死体・eDNA・HGT
- 神経組織、学習、因果監査
- 完全なworld.stepのGPU移植

## 決定論と精度

- 検証精度はfloat64。
- CPU authoritative RNGを使うhybrid kernelでは、同一RNG streamを維持する。
- device-neutral counter RNGも別途ロックステップ検証する。
- fp32は速度用候補であり、fp64との誤差を明示してから使用する。
- CUDA未搭載CIではTorch CPUで意味論を検証し、CUDA性能・実機一致は未検証と明記する。

## 合格条件

- 専用検証32/32 PASS。
- 親0.6.7専用検証28/28 PASS。
- diffusion/profileのNumPy/Torch fp64ロックステップ。
- hybrid 0.6.6の短時間イベント一致。
- clone/save-restore決定論的一致。
- 可変ゲノムや容量超過を黙って切り捨てない。
- summaryがfull GPU worldと誤表示しない。

## 科学的限定

A1は速度基盤であり、新しい生命・進化・神経優位の科学結果を主張しない。CUDA実機検証がない環境ではRTX性能も主張しない。
