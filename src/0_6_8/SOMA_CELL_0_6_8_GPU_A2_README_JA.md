# SOMA-CELL 0.6.8-GPU A2

A2はSOMA-CELL 0.6.6詳細粒子世界のGPU移植第二段階です。粗視化0.6.7を生命本体へ置き換えず、表面交換、廃棄物排出、漏出、浸透圧半径、運動をNumPy/Torch fp64で照合し、CPU正本hybridへ接続しました。

## 現在GPU/Torchへ移した処理

- 粒子拡散とpatch drift
- 膜周囲のligand profile
- torus近傍候補
- fuel/mineral/waste/alt surface exchange
- ATP有料のwaste export
- 膜gapからのleak plan
- radius relaxation
- surface-flux/Brownian motion

表面交換は内部poolとATPが粒子順序に依存するため、A2では正確さを優先した逐次scanです。CPU-only環境では凍結CPU版より約1.52倍遅く、速度向上を主張しません。

## 未移植

- 代謝・損傷の完全統合
- DNA複製
- 翻訳
- 分裂
- 死亡、死体、環境DNA、HGT
- 神経、因果監査

従って `gpu_full_world_step` は常にFalseです。

## RTX 4060 Tiでの確認

CUDA対応PyTorchを入れ、まず以下を実行します。

```bash
python SOMA_CELL_0_6_8_install_check.py
python SOMA_CELL_0_6_8_A2_validation.py
python SOMA_CELL_0_6_8_A2_benchmark.py --device cuda --precision float64
```

A2のCUDA性能はこの開発環境では未検証です。4060 Tiではfp64が高速とは限らないため、fp32へ進む場合もfp64との差を必ず保存します。

## Pythonista

Pythonista companionは0.6.6詳細CPU世界の観察用です。iPhone上でCUDAは使用しません。
