# SOMA-CELL 0.6.8-GPU A3 正式工学契約

## 目的

SOMA-CELL 0.6.6 の詳細粒子世界を科学的正本として維持したまま、0.2 の gene-coded metabolism と 0.3 の損傷・修復化学を NumPy/Torch の独立した fp64 計算へ転記し、A2 hybrid world-step に統合する。A3 は計算場所を移す段階であり、生命機構を省略する段階ではない。

## 正本と継承

- 世界・細胞の意味論と RNG 順序: `src/0_6_6/SOMA_CELL_0_6_6_pythonista.py` とその継承元
- gene-coded metabolism: `src/0_6_2/SOMA_CELL_0_6_2_pythonista.py`
- damage / repair / segregation: `src/baseline/SOMA_CELL_0_3_pythonista.py`
- sensorimotor wrapper: `src/baseline/SOMA_CELL_0_4_pythonista.py`
- ecology wrapper: `src/baseline/SOMA_CELL_0_5_pythonista.py`
- 表面・物理層: `src/0_6_8/SOMA_CELL_0_6_8_gpu_a2.py`

A1 の `metabolism_core_*` は凍結 0.1 standalone core であり、A3 の gene-coded metabolism の正本としては使用しない。

## 凍結公理

1. 反応、支払い、辞書反復、RNG、world/cell wrapper の順序を 0.6.6 から変えない。
2. ATP、燃料、鉱物、前駆体、膜、輸送体、タンパク質、DNA は無料生成・無料コピーしない。
3. active protein と damaged protein の fingerprint identity を保存する。
4. legacy aggregate の fingerprint 組成は推定しない。既存 scalar は `aggregate_unresolved` として保持し、A3 以降に生成した aggregate だけを typed composition に記録する。
5. Python dict の挿入順を packed order tensor に保存し、key sort や device reduction で置き換えない。
6. live genome の重度 lesion による symbol hydrolysis は、元の cell 順・genome 順で CPU world RNG を消費する。
7. translation、replication、actual division、death/corpse/eDNA/HGT、neural/causal system は A3 では CPU-authoritative のままでよい。
8. A2 の surface/export/leak/radius/motion と A3 phase は、各 canonical site で一度だけ実行する。duplicate claim は例外とする。
9. capacity 超過、非有限値、順序不正、pool/dict 不一致は commit 前に fail closed とし、切り捨てない。
10. 未移植機構があるため `full_gpu_world_step=false` を固定する。
11. fp64 を正確性基準とし、fp32 は fp64 合格後にのみ性能候補として測定する。
12. 外部 fitness、reward、正解方向、環境切替時刻を個体へ渡さない。

## A3 で Torch へ移す範囲

- generic gene reactions（固定順 0→3）
- gene-coded ATP generation
- membrane/transporter/nucleotide precursor synthesis と paid maintenance/assembly
- active protein damage
- damaged-protein aggregation と A3 typed composition ledger
- waste 由来 reactive byproduct
- membrane oxidation と severe hydrolysis
- genome-lesion scalar gain と repair allocation
- antioxidant、chaperone、protease、genome、membrane repair（固定順）
- ordinary pool/membrane/transporter housekeeping
- Formal066 mineral→membrane-precursor paid route
- actual split 時だけ消費される damage segregation plan

GPU kernel は pure plan を返し、検査済み plan だけを CPU object graph へ commit する。

## CPU-authoritative のまま残る範囲

- variable genome の translation mechanics と辞書挿入
- genome replication、proofreading RNG、mutation
- live genome symbol hydrolysis の RNG と sequence edit
- actual division、daughter allocation、death release
- corpse、eDNA、HGT
- neural、tissue、causal audit
- particle emission と Brownian RNG

## exactly-once scheduler

ledger key は `(world_step_id, cell_id, event)` とする。各 entry は少なくとも `status`（`executed` または `skipped`）、ordinal、backend、metadata を保存する。`status` は物量ではなく canonical invocation の有無を表す。pure plan、backend method、CPU-authoritative operation を canonical site で呼び出した場合は、feature disabled または物量 0 の no-work return でも `executed` とする。死亡、同 step daughter、phase 到達前終了などにより operation 自体を呼ばなかった場合だけ `skipped` とする。`enabled`、`work_performed`、`amount` は metadata で別々に記録する。

- 同じ key の二回目の claim は state mutation 前に例外。
- predecessor を満たさない claim は state mutation 前に例外。
- cell が代謝開始時に死亡している場合は、必須 phase を明示的に skipped として閉じる。
- world step 完了時、参加した各 cell の必須 phase と全 world phase を検査する。
- A2 backend の work-positive counter は証明に使わず、scheduler receipt を正本とする。
- raw `Formal066World.from_state` から hook を装着し、古い A1/A2 closure を別 backend へ再利用しない。

## packed-state 不変条件

- `pools[4] == sum(active_amount)`
- `pools[10] == sum(damaged_amount)`
- `pools[11] == aggregate_unresolved + sum(aggregate_typed_amount)`
- `active_order`、`damaged_order`、`gene_order`、`aggregate_order` は有効 slot の permutation で、未使用は `-1`
- `species_mask[i] == (i < species_count)`、`genome_mask[i] == (i < genome_count)` を厳密に満たす。unused fingerprint/order/gene metadata と count 外の amount/genome tail は規定 sentinel/zero でなければならない
- active/damaged/typed aggregate の非零 slot 集合は各 order の used slot 集合と完全一致する。used active は `>1e-10`、used damaged は `>1e-11`、used aggregate は `>0` とし、CPU mapping にこれ以下の entry が存在する場合は pack 時に拒否して黙って落とさない
- species union は実在する active、damaged、typed aggregate、gene specs の fingerprint の和集合とする。負の reserved debris fingerprint も通常の key として lossless に扱う
- gene metadata が非 default の slot 集合は `gene_order[:gene_count]` と完全一致する。非 gene slot は role/parameter/reaction/localisation=`-1`、promoter/efficiency/copy number=`0` とする
- `0 <= genome_lesion_count <= genome_count`。凍結 save に存在する一時的な不足を pack/unpack で保存し、damage phase が lesion gain の直前にだけ不足要素を 0 で拡張する
- complete genome の length は count 外 0、lesion は `genome_lesion_count` 外 0、hydrolysis hazard は `genome_count` 外 0 とし、`total_genome_symbols == sum(genome_lengths[:genome_count])` を満たす。inactive replication は template/copy length と template lesion が全て 0、active replication は `0 < template_length <= max_genome_symbols` かつ `0 <= copy_length <= template_length` とする
- 全 array は NumPy または Torch のどちらか一方に統一する。float array は同一 dtype（float64 reference または明示的 float32 candidate）、Torch array/scalar は同一 device、integer は int64、mask は bool とする。mixed backend/device/dtype、float16/bfloat16 は fail closed とする
- genome capacity は 3 以上、protein species と genome length capacity は config で明示
- A3 の公開 packed 単位は一細胞の移植対象フィールドを lossless に mirror する `A3PackedState` である。CPU-authoritative の実 genome sequence、translation/replication object、神経・生態 state は既存 cell object に保持する。現 checkpoint の hybrid scheduler はこれを細胞ごとに処理し、未実装の batched world tensor を主張しない
- pack は一細胞の全対象を先に検証し、一件でも超過・subthreshold mapping・不正 sidecar があれば tensor も CPU state も生成途中で公開しない。unpack も tail を含む全変換と整合性検査を終えてから CPU object graph を更新する

## エネルギー台帳

凍結実装と同じく、translation、precursor synthesis、repair、proofreading の ATP 支払いは ATP pool から差し引くが、通常 `world.dissipated_energy` には加えない。明示的に散逸へ加えるのは、少なくとも A2 leak、damage segregation、Formal066 supplemental route である。この区別を A1 standalone core の delta で上書きしない。

## 合格条件

- 独立 NumPy/Torch fp64 kernel parity
- CPU frozen vs A3 hybrid の 1/10 step、periodic/reversal、stressed、repair-heavy、pre-division lockstep
- RNG、material ledger、全離散状態一致
- protein/damaged/aggregate/lesion を含む全状態比較
- clone と save/restore の継続一致
- exact-capacity PASS、capacity+1 atomic failure
- duplicate/out-of-order scheduler failure と exactly-once receipt
- validation/benchmark 証跡の `source_sha256_map` が final core、scheduler、validator/runner、A2、0.6.6 frozen source と一致
- A1 32/32、A2 32/32、historical regression の維持
- RTX 4060 Ti 上の CUDA fp64 実測保存
- fp32 discrepancy の項目別保存

いずれかが未達なら A2 を正式 baseline とし、A3 を完成または昇格と呼ばない。

## 科学的限定

A3 は工学的正確性 checkpoint である。生命、意識、オープンエンド進化、CUDA高速化を証明しない。
