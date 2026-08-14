# SOMA-CELL 0.6.8-GPU A3

A3 は、0.6.6 詳細粒子世界の現象を削らずに、A2 の表面・物理層へ gene-coded metabolism と damage/repair の Torch 計算を追加する correctness checkpoint です。

## A3 の範囲

- ATP generation と有料 precursor/surface synthesis
- generic gene reactions
- active/damaged protein と aggregate composition
- reactive byproduct、membrane oxidation、genome-lesion scalar chemistry
- antioxidant、chaperone、protease、genome、membrane repair
- damage segregation planning
- A2 export/leak/radius/motion を含む exactly-once scheduler

NumPy 転記を独立正本、Torch fp64 を GPU candidate として相互照合します。legacy aggregate は過去の fingerprint を推定せず、由来不明量として保存します。

## A3 でも CPU に残るもの

- variable genome translation mechanics
- replication/mutation と live-genome hydrolysis RNG
- actual division、death、corpse、eDNA、HGT
- neural/tissue/causal system
- particle emission と Brownian RNG

したがって `full_gpu_world_step` は False です。

## RTX 4060 Ti での実行

プロジェクト root から、CUDA 対応 PyTorch の環境で次を実行します。

```bash
python src/0_6_8/SOMA_CELL_0_6_8_install_check.py
python src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py --require-cuda
python src/0_6_8/SOMA_CELL_0_6_8_A3_benchmark.py \
  --device cpu --device cuda --precision both \
  --worlds 1 8 32 128 --cells 3 --cell-sweep 1 3 8 \
  --particle-sweep 216 288 360 --capacity-sweep 64 128 256 \
  --steps 1 --world-warmup 1 --world-repeats 2 \
  --other-axis-warmup 0 --other-axis-repeats 1 \
  --correctness-steps 1 10 20
```

validation は独立 NumPy/Torch fp64、strict lossless schema、CPU world lockstep、CUDA fp64、fp32 差を一つの gate で実行します。`--require-cuda` 指定時に CUDA が利用不能なら CPU fallback せず失敗します。benchmark は各 requested device の fp64 correctness probe が通った場合だけ timing を開始し、fp32 は candidate-only の差分付きで保存します。13 sweep spec / 65 measurement の欠落・重複、非有限時間、CUDA peak VRAM、source hash も fail closed で検査します。

## 数値上の注意

protein/gene map は Python dict 挿入順で逐次反応します。translation、generic reaction、repair は ATP と材料を前段から引き継ぐため、勝手に一括 reduction しません。この correctness-first 設計では、小規模 world の CUDA が CPU より遅くても異常ではありません。

## Pythonista

`SOMA_CELL_0_6_8_GPU_A3_pythonista.py` は凍結 0.6.6 詳細 CPU Scene の観察用です。iPhone 上で Torch/CUDA kernel を実行せず、GPU 性能を主張しません。

## 科学的限定

A3 は生命、意識、オープンエンド進化を証明しません。負の結果、速度低下、fp32 差は結果として保存します。
