# SOMA-CELL 0.6.8-GPU A4.1 事前登録

状態: 開発用の最小slice。A4昇格判定ではない。

## 仮説

A3が遅い主因はGPU演算能力ではなく、cell/phaseごとのpack、host-device
往復、同期、CPU object commitである。最初から全生物相を作り直さず、
まず可変長ゲノムを一度だけGPUへ載せて保持できる境界を作る。

## 今回実装するもの

- 一世界のcell列を対象とするfixed-capacity ragged symbol arena
- complete genome、replication template、partial copyのlossless表現
- 短いgenome lesion vectorの保存
- NumPy/Torch fp64間の明示変換
- 全件検証後だけ行う原子的CPU復元
- 明示CUDA上の短いresident checksum smoke

## 今回実装しないもの

- gene translation、replication、mutation、RNG tape
- A3 scheduler/world-stepの差し替え
- 汎用ragged allocator、自動capacity拡張
- `torch.compile`、CUDA Graph、Triton、独自CUDA
- fp32、multi-GPU、online GPU
- A3の4時間formal benchmark再実行

## 固定テスト

1. API、schema、`full_gpu_world_step=false`。
2. 0/1/2/3 complete genomes、active template、empty/partial copy、短いlesion
   vectorを混ぜたroundtrip。
3. NumPy -> Torch fp64 -> NumPyのdtype、値、順序一致。
4. offset、dtype、alphabet、unused tail、replication relationの破損拒否。
5. cell、sequence、symbol capacityのexact PASSと+1 atomic FAIL。
6. target ID順序違反、duplicate IDの拒否。
7. CUDA必須実行では明示device確認、入力pointer安定、初回upload後の
   checksum反復、反復後だけ行う明示readback phase。
8. A3のpack/clone/saveおよび短いworld lockstepをfocused regression。

## 判定

- 上記が全PASSならA4.1 representation foundationとして保持する。
- A3を正式baselineのまま維持し、A4完成や高速化とは呼ばない。
- CUDA不可、capacity clip、順序変化、atomicity違反があればFAIL。
- resident checksumの速度は合否に使わない。実生物kernelを載せる前に
  速度向上を主張しない。

## 次の小さいslice

A4.1合格後、complete genomeのbatched gene decodeとgene-cache挿入順を
CPU正本へ照合し、その後にRNGを使わないpaid translationを移す。
replication/mutationはさらに後とする。
