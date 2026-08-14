# SOMA-CELL 0.6.8-GPU A3 事前登録

登録日: 2026-08-14  
正式 baseline: SOMA-CELL 0.6.8-GPU A2  
対象 device: NVIDIA GeForce RTX 4060 Ti 16GB  
目的: full-fidelity gene-coded metabolism と damage/repair の correctness-first hybrid 移植

## 事前仮説

主仮説は「A3 Torch fp64 plan と凍結 CPU 0.6.6 が、反応順序・RNG・材料支払いを変えずに lockstep 一致できる」である。速度向上は主仮説ではない。逐次依存と小さい kernel が残るため、A3 が A2/CPU より遅い結果も受理し、そのまま保存する。

帰無側の扱い:

- fp64 差、RNG 差、ledger 差、scheduler duplicate、capacity truncation のいずれかが閾値外なら A3 不合格。
- CUDA 不可または未測定なら A3 不合格。
- fp32 が速くても、fp64 正確性を代替しない。

## 固定 scenario

独立 kernel:

1. clean
2. stressed/reactive
3. oxidized membrane
4. aggregate-heavy（typed と unresolved を含む）
5. ATP-poor
6. repair-disabled
7. exact-capacity
8. capacity+1

world lockstep:

1. one step
2. ten steps
3. periodic/reversal 20 steps
4. stressed / repair-heavy
5. pre-division と actual split 境界
6. clone continuation
7. save/restore continuation

性能 sweep は correctness 合格後に 1/8/32/128 worlds と cells/particles/capacity を測る。OOM や実行不能も結果であり、shape を縮小して成功扱いにしない。

## 固定判定

- NumPy vs Torch fp64 pure kernel: 各 float field 最大絶対差 `<= 2e-12`
- frozen CPU vs A3 fp64 world: 各 float field 最大絶対差 `<= 5e-12`
- material ledger residual 差: `<= 5e-10`
- integer、bool、fingerprint、sequence、dict key/order、RNG state: 完全一致
- input mutation: pure plan 呼出し前後の semantic hash 完全一致
- capacity+1: 明示例外、例外前後の input/CPU world hash 完全一致
- scheduler: canonical event が executed/skipped のいずれかで一回。duplicate/out-of-order は state mutation 前に例外
- CUDA request: unavailable 時は CPU fallback せず明示失敗

閾値を変更する場合、既存結果を上書きせず新しい事前登録と理由を追加する。

## fp32

CUDA fp64 合格後にのみ測定する。全 field の fp64 差、RNG、ledger、非有限値を JSON に保存する。fp32 は性能候補であり、A3 の意味論正本にはしない。

## 報告

GPU 名、driver、Torch、Torch CUDA、VRAM、Git commit、source SHA256、dtype/device、warmup、repeats、step 数、world/cell/particle capacity、peak allocated/reserved VRAM を保存する。負の結果、遅い結果、CUDA未達を削除または言い換えない。

## 科学的限定

本試験は生命、意識、オープンエンド進化を検定しない。A3 は GPU 移植の工学検証である。
