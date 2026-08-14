# SOMA-CELL 0.6.8-GPU A4.2 事前登録

状態: 開発用の最小slice。A4昇格判定ではない。

## 継承するA4.1基盤

A4.1は、0/1/2/3 complete genomes、active template、empty/partial copy、
短いlesion vectorをlosslessなfixed-capacity ragged arenaへ格納し、明示CUDA
deviceへ一度uploadして保持できることを確認した。A3正式baseline、CPU正本、
`full_gpu_world_step=false`は変更しない。

## A4.2仮説

complete genome列をcellごとにCPUへ戻さず、resident arena上でまとめてdecode
しても、凍結CPUの`parse_genes`と`_refresh_gene_cache`の結果を完全に保存できる。
性能主張ではなく、次のpaid translation sliceに必要な正確な派生cache境界を
作ることが目的である。

## 今回実装するもの

- complete genomeだけを対象とするbatched delimiter decode
- valid gene後に16進む凍結greedy parserの固定round再現
- cell -> genome list -> local startの初出順
- same-cell fingerprint重複のcopy number集計
- 12-symbol payload、fingerprint、first start、copy numberのfixed-shape arena
- NumPy独立実装とTorch CPU/CUDA実装
- 明示readback後の全CPU `gene_specs`復元

派生cacheは保存・外部復元の正本にしない。次sliceのtranslationも、同じvalidated
resident genome arenaから直前に生成したcacheだけを受け取り、外部cacheを信用しない。

Torch decoder内では`.item()`、`.cpu()`、`.numpy()`、`.tolist()`、`nonzero`、
masked select、可変長`unique`を使わない。出力shapeは入力capacityから決まり、
decode中にCPU cell、RNG、material、protein、scheduler receiptを変更しない。
複合int64 keyの予約sentinel衝突を避けるため、`cell_capacity`は
`134217727`以下を必須とし、超過は配列確保前に拒否する。

## 今回実装しないもの

- paid translation、protein生成、ATP/precursor commit
- replication、proofreading、mutation、hydrolysis、RNG tape
- A3 scheduler/world-stepの差し替え
- dirty/incremental cache、汎用allocator、別world state/save形式
- `torch.compile`、CUDA Graph、Triton、独自CUDA
- fp32、multi-GPU、online GPU
- A3の4時間formal benchmarkと速度向上主張

## 固定fixture

一cellに3 complete genomesとactive template/copyを置く。

- malformed start/stop
- valid outerとvalid nestedが重なるcase
- invalid outerの1-symbol後にvalid innerが現れるcase
- generic/non-generic/0.6.5 grammar gene
- complete genomes間のduplicate fingerprint
- geneを含むtemplateとpartial copy

期待するcomplete-genome occurrenceは6、unique cacheは3。

```text
fingerprint order = [40116138652, 55715248503, 68679218097]
copy numbers      = [2, 3, 1]
first starts      = [19, 35, 19]
```

template-only fingerprint `9105062057`はcacheへ入らない。grammar payloadの
module/value `(1,6)`を保持する。genericだけが`reaction`/`reaction_name`を持つ。

## 固定テスト

1. A4.1のAPI、lossless roundtrip、corruption、capacity、ID、CUDA residency。
2. 上記fixtureのhard-coded occurrence/order/count/startとCPU正本の全辞書一致。
3. NumPy、Torch CPU、RTX CUDAのfixed-shape cache完全一致。
4. Torch decoder sourceにhost scalar/readback/dynamic-output operationがないこと。
5. decode前後でCPU world、ragged arrays、CUDA input pointersが不変。
6. payload/fingerprint、mask、offset、copy、unused tail破損の拒否。
7. 6個の連続geneがderived capacity exactでPASSし、7個目は既存fixed symbol
   boundaryでpack前にatomic FAIL。zero-entry capacityもPASS。
8. A3 pack/clone/saveと短いworld lockstepのfocused regression。

## 判定

- 全testがNumPy/Torch CPUと明示CUDAでPASSすればA4.2 decode foundationとして
  保持する。
- A3正式baseline、CPU translation authority、`full_gpu_world_step=false`を維持。
- nested marker、first insertion order、template exclusion、payload、capacity、
  source nonmutationのいずれかが違えばFAIL。
- decode timingは合否に使わず、速度向上を主張しない。

## 次の小さいslice

A4.2合格後、このderived cacheを使うRNG不要のpaid translationだけを移す。
replication/mutationとscheduler authority変更はさらに後とする。
