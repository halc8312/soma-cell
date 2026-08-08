# SOMA-CELL 0.6.8-GPU A1

## 何を作ったか

0.6.6の詳細粒子世界を薄くせずにGPUへ移すための、第一段階のテンソル基盤です。0.6.7の粗視化チェモスタットをGPUで速くした版ではありません。

現在、凍結0.6.6世界を固定容量のTorchテンソルへ写し、可変長ゲノム・36区画膜・輸送体・内部物質プールを保持できます。粒子拡散と膜局所リガンド計算は、0.6.6のworld.step内からTorchへ安全にオフロードできます。

## なぜ一気に全部GPU化しないか

分裂、HGT、可変長DNA、死体化学は分岐と可変長データを多く含みます。ここを一度に書き換えると、速くても別の生命モデルになりかねません。

そのため、各カーネルを次の順で移します。

1. 独立NumPy参照を作る。
2. Torch版とfp64でロックステップする。
3. CPU正本worldへ一部だけ接続する。
4. clone/save/restoreとイベント一致を確認する。
5. その後にCPU経路を外す。

A1では1〜4を、粒子拡散とリガンドプロファイルについて完了しています。

## 主要ファイル

- `SOMA_CELL_0_6_8_gpu.py`: tensor schema、adapter、kernels、hybrid runner
- `SOMA_CELL_0_6_8_validation.py`: 32項目の検証
- `SOMA_CELL_0_6_8_gpu_benchmark.py`: batched kernel benchmarkとVRAM容量推定
- `SOMA_CELL_0_6_8_install_check.py`: PyTorch/CUDA環境確認
- `SOMA_CELL_0_6_8_pythonista.py`: iPhone上の0.6.6詳細CPU参照viewer

## RTX 4060 Tiでの開始手順

CUDA対応PyTorchを導入したPC環境で、展開フォルダから次を実行します。

```bash
python SOMA_CELL_0_6_8_install_check.py
python SOMA_CELL_0_6_8_validation.py
python SOMA_CELL_0_6_8_gpu_benchmark.py --device cuda --precision float64 --worlds 8 --particles 512 --cells 8 --steps 20
```

まずfloat64で意味論を照合します。その後、同じ条件をfloat32で実行し、誤差と速度を比較します。

## 現在の実測

開発環境はCPU-only PyTorch 2.10.0でした。CUDA実機は未検証です。

小型CPU benchmark:

- worlds: 4
- particles/world: 128
- cells/world: 4
- steps: 5
- precision: float64
- kernel world-steps/s: 約4278

これはdiffusion/profileカーネルだけの値であり、フル0.6.6 world-stepの速度ではありません。

## 32項目の検証

- device-neutral counter RNG
- circular smoothing
- diffusion fp64 lockstep
- ligand profile fp64 lockstep
- full state adapter
- genome byte mirror
- capacity fail-closed
- metabolism core NumPy/Torch lockstep
- hybrid 0.6.6 one-step/short-step lockstep
- hybrid clone/save-restore
- batch isolation
- fp32 error disclosure
- no external fitness
- full-GPU誤表示防止

すべてPASSしています。

## 次の移植順序

A2:
- surface exchange
- waste export
- leak
- radius/motion

A3:
- core metabolismをworld.stepへ統合
- membrane repair/damage layers

A4:
- genome buffer、replication、translation

A5:
- division/death/corpse/eDNA/HGT

A6:
- neural tissue and causal audit
- full CPU/GPU event lockstep

## 限定

この版はGPU移行基盤です。フルGPU版ではなく、新しい進化結果や生命認定を示しません。
