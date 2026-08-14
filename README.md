# SOMA-CELL

SOMA-CELLは、膜・粒子・物質ゲノム・損傷・死体・eDNA・HGT・神経因果系を削らずに扱う、物質制約付き人工生命シミュレーションです。GPU化は現象を簡略化するためではなく、同じ詳細現象を大規模に計算するために進めています。

> 現在の工学baseline: **SOMA-CELL 0.6.8-GPU A3**
> 次のmilestone: **SOMA-CELL 0.6.8-GPU A4**

A3はgene-coded metabolism、ATP有料合成・維持、protein damage/aggregation、reactive byproduct、膜酸化、damage repair、segregation planningと、A2/A3処理のexactly-once schedulerを統合したengineering checkpointです。

## 現在の実測状態

- A3専用検証: 36/36 PASS
- A2・A1・Windows historical chainを含む合計: 379/379 PASS
- RTX 4060 Ti CUDA fp64 full-world最大差: `3.552713678800501e-15`
- fp32 full-world最大差: `3.598134365018738e-6`（candidate-only）
- 正式benchmark: 13仕様・65測定を完走
- A3 CUDA fp64は凍結0.6.6 CPU正本より約570.096〜590.276倍遅く、速度向上は未達
- `full_gpu_world_step=false`

性能が遅い主因は、per-cell Python orchestration、CPU object pack/commit、host-device転送、小kernelの逐次起動です。A4ではGPUを主計算器とし、device-resident ragged genome、batched cell state、kernel統合を優先します。CPU実装は意味論正本・照合oracle・未移植部分のfallbackとして保持します。

## 作業開始前の必須ゲート

会話記憶やこのREADMEだけから編集を始めないでください。権威は`PROJECT_STATE.json`、`CURRENT_BASELINE.txt`、現在source、測定済みresultsです。

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<name>" \
  --purpose "<work>" \
  --milestone "SOMA-CELL 0.6.8-GPU A4" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

必須読み順は`MANDATORY_WORKFLOW.json`にあります。

## 主要ファイル

- A3 core: `src/0_6_8/SOMA_CELL_0_6_8_gpu_a3.py`
- A3 scheduler: `src/0_6_8/SOMA_CELL_0_6_8_gpu_a3_scheduler.py`
- A3 contract/schema/event order: `docs/SOMA_CELL_0_6_8_GPU_A3_*`
- CPU科学正本: `src/0_6_6/SOMA_CELL_0_6_6_pythonista.py`
- 状態: `PROJECT_STATE.json`
- 検証: `results/SOMA_CELL_0_6_8_GPU_A3_VALIDATION_RESULTS.txt`
- 工学報告: `results/SOMA_CELL_0_6_8_GPU_A3_EXPERIMENT_REPORT.txt`
- benchmark: `results/soma_cell_0_6_8_gpu_a3_benchmark.json`

CUDA環境でA3検証を再実行する場合:

```bash
python3 src/0_6_8/SOMA_CELL_0_6_8_A3_validation.py --require-cuda --no-write
```

## Checkout上の注意

Git履歴には大文字・小文字だけが異なるpathがあります。通常のcase-insensitive Windows checkoutでは片方を失う可能性があるため、WSL ext4などcase-sensitiveなfilesystemへcloneしてください。

## 科学的な限界

A3は工学的移植checkpointです。生命、意識、開放進化、GPU高速化を証明したものではありません。正・負の結果、CUDA未達、platform sensitivityを削除せず保存します。

## License

特記のないSOMA-CELLのproject-authored code・documentation・resultsは、[Apache License 2.0](LICENSE)で公開します。

凍結済みA3 release ZIPはcheckpoint hashを維持するため再生成していません。ZIPを単独再配布する場合はroot `LICENSE`も添付してください。今後のrelease builderではライセンスをarchiveへ同梱します。
