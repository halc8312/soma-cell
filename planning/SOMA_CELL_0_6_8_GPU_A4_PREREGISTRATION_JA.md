# SOMA-CELL 0.6.8-GPU A4.4a 事前登録（A4.3継承）

状態: A4.3までを継承する開発用の最小slice。A4昇格判定ではない。

## 継承する基盤

A4.1のlossless fixed-capacity ragged genome arenaと、A4.2のcomplete-genome
resident gene decodeをそのまま使う。A3正式baseline、凍結0.6.6 CPU正本、
`full_gpu_world_step=false`は変更しない。派生gene cacheは保存・外部復元の
正本にせず、同じvalidated ragged arenaから内部生成したものだけを使う。

## A4.3仮説

gene cache、物質pool、active/damaged proteinの独立挿入順、損傷・膜・sensor・
effector・ecology信号、およびCPU正本のneural attachment osmolyteをfixed-shape
fp64 stateへ一度packすれば、RNGを使わないpaid translationをresident
NumPy/Torch計画として再現できる。

ここでいう再現対象はA3のpure helperだけではなく、Formal066の実MROである。

- 0.3: paid translation、quiescence、misfold、protein sync
- 0.4: LOC_REPAIR activity、sensor/effector need、behavioural quiescence
- 0.5: ecology-localised need
- P0: neural attachment osmolyteを含むinherited tension
- 0.2: 非regulator needと物質支払い

## 今回実装するもの

- `A4TranslationStateBatch`のfixed `[cell, protein]` active/damaged行
- 元のPython dict挿入順を保つfingerprint、mass、count
- ragged由来のgenome count、lesion mean、replication flag、material symbols
- sensor/control/ecology/behavioural signal snapshot
- neural attachment valuesの順序付きosmolyte合計（read-only snapshot）
- 内部decodeだけを許す非serializable `A4TranslationBinding`
- ragged SHA-256 provenance、resident tensor pointer/version、およびbinding内
  derived cache lifetime attestation
- 独立literal NumPy計画とfixed-shape Torch CPU/CUDA計画
- ATP reserve `0.042`とgene順の逐次材料枯渇
- active `>1e-10`、damaged `>1e-11`の凍結sync threshold

全weight、translator、chaperone、quiescence、misfold率は生産前stateから計算する。
その後だけgene-cache順に支払い、1翻訳単位につきfuel `0.64`、mineral `0.36`、
ATP `0.52`を消費する。既存protein keyは位置を維持し、新規keyはthresholdを
越えた場合だけgene順に末尾追加する。

Torch計画内では`.item()`、`.cpu()`、`.numpy()`、`.tolist()`、`nonzero`、
masked select、可変長`unique`を使わない。支払いはhost定数のgene-rank loopを
cell batchへ適用し、凍結CPUの逐次fp64減算順を保つ。入力ragged/cache/physiology
tensorとCPU cell、world、RNG、scheduler receiptを変更しない。

凍結CPUの逐次支払いが生成し得るsigned fp64 residualだけはfuel/mineral/ATPで
`-2e-12`まで許容する。他poolはnonnegativeのまま維持する。これは一般の
negative biology許容やclipではなく、より大きい負値はfail closedとする。

## 今回実装しないもの

- A3 schedulerの`translation_cpu`差し替え
- CPU cell/worldへのprotein/material commit
- replication、proofreading、mutation、hydrolysis、RNG tape
- dirty cache、device allocator、別save authority
- `torch.compile`、CUDA Graph、Triton、独自CUDA
- fp32、multi-GPU、online GPU、formal 65-row benchmark、速度向上主張

## 固定fixture

4つのFormal066 cellを同一batchで使う。

1. rich material、非zero sensor/control/ecology signal、および実際の
   `NeuralAttachmentState`（osmolyte `29.3`）。
2. ATPをreserveと等しい`0.042`にし、weight gate後に合成量0となるcell。
3. mineralを`0.36e-6`にし、先頭geneが使い切って後続geneが0となるcell。
4. translator geneを残したままactive translator proteinだけ除き、早期returnするcell。

全cellにLOC_REPAIR/LOC_SENSOR/LOC_EFFECTOR/LOC_ECOLOGY regulatorを含める。
`quiescence=false`かつ`quiescence_effector=true`のcaseではbehavioural値が
`last_quiescence`へ反映されることも固定する。

## 固定テスト

1. A4.1/A4.2の13テストを継続PASS。
2. 上記4 cellを直接`Formal066ProtoCell.translate`へ通し、planのpools、
   `last_translation`、条件付き`last_quiescence`、active/damaged amountsをfp64照合。
3. `state_dict()`とは別にactive/damaged/gene dict key順をexact照合。
4. fuel/mineral/ATP支払いとcatalyst/damaged ledgerを照合。
5. NumPy、Torch CPU、明示RTX CUDAのplan parity。
6. resident入力pointer/value不変、全outputが同じdevice、明示readbackは最後だけ。
7. protein union capacity exact PASS、capacity + 1はupload前atomic FAIL。
8. 外部translatorのself-need、`dt=1e-8`の極小resource枯渇、signed ledger residualを固定。
9. `source_provenance`不一致、caller binding/cache、upload後tensor mutationをreject。
10. ledger toleranceを超えるnegative、nonfinite、mixed dtype、count、tail、identity
   mismatchをfail closed。
11. A3 pack/clone/saveと短いworld lockstepのfocused regression。

## 判定

- 全17テストが明示CUDAを含めPASSすれば、A4.3を「full Formal066 paid
  translationのpure resident plan、未統合」として保持する。
- CPU正本、material ledger、dict order、early-return境界、MRO dispatchの
  いずれかが違えばFAIL。
- A3 `translation_cpu`とreplication authorityは維持し、速度向上を主張しない。

## A4.4a追加仮説

既にactiveなreplication templateとpartial copyがあり、mutation、proofreading、
external replicase、quiescence、quiescence effectorを全て無効にし、このcallで
template completionへ到達しない場合、DNA伸長の物質支払いをfixed-shape
NumPy/Torch fp64 planとしてCPU正本どおり再現できる。

これはreplication全体の移植ではない。template選択、completion、新genome追加、
lesion継承、gene cache refresh、突然変異、hydrolysis、RNG、scheduler commitは
今回の対象外とする。

## A4.4aで実装するもの

- Formal066の`eco66_replication_rate_scale`によるdt倍率
- post-translation active protein挿入順からのendogenous replicase activity
- nucleotide/ATP saturationとfractional carry
- integer requestをresource gate前にfractionalから除く凍結順序
- template byte順のコピー
- 1 symbolあたり`MONOMER_MASS` nucleotideと`0.0012 ATP`の支払い
- nucleotide exact gateとATP reserve `0.022`を含む逐次resource gate
- `last_replication_symbols`と、mutation無効時にも更新される
  `last_effective_error_rate`
- fixed capacity事前検査、入力非変更、CPU/Torch CPU/CUDA独立照合

## A4.4aで実装しないもの

- inactive cellのtemplate選択
- proofreading、external replicase、inherited/behavioural quiescence
- substitution、insertion、deletion、duplication、inversion、transposition
- copy completion、新complete genome、lesion/cycle/cache更新
- genome hydrolysisとRNG tape
- ragged/CPU cellへのworld commit、A3 scheduler差し替え
- fp32、compile、custom CUDA、formal benchmark、速度向上主張

CUDAの有効経路ではhidden D2Hを禁止する。device上でscope外を検出した場合は、
fixed error codeを結果へ保持し、明示readbackまたは将来のcommit境界で必ずrejectする。
scope外のrowを成功扱いせず、途中だけGPUで進めて同じreplication eventをCPUで
再実行することも禁止する。

## A4.4a固定テスト

1. Active partial-copy Formal066 cellをCPU `_replicate_genome`へ直接通し、append
   bytes/count、nucleotide/ATP、fractional carry、replication telemetryを照合する。
2. requested=0、nucleotide/ATPのexact thresholdと`nextafter`直下、途中枯渇を固定する。
3. CPU oracleのRNG bit-generator state、world dissipated energy、complete genomes、
   lesions、gene cache、入力ragged/cache/physiologyが完全不変であることを確認する。
4. NumPy、Torch CPU、明示RTX CUDA fp64を照合し、resident入力pointerとdeviceを確認する。
5. exact symbol capacityはPASS、capacity+1、inactive template、completion、延期flagは
   atomic fail closedとする。
6. A4.1〜A4.3の17テストとfocused A3 regressionを継続PASSさせる。

## A4.4a判定

- 全テストが明示CUDAを含めPASSした場合だけ、A4.4aを「既存templateの
  mutation-free/proofreading-free/non-completing paid elongation pure plan、未統合」
  と記録する。
- CPU支払順、fractional carry、RNG不変、material ledger、CUDA fp64のいずれかが
  違えばA4.3を維持する。
- `replication_cpu`、promoted A3、`full_gpu_world_step=false`は変更しない。

## 次の小さいslice

A4.4bでは同じnon-completion境界にproofreadingとquiescence/external replicaseを
追加する。その後にtemplate/RNG、completion/structural mutation、hydrolysisを
別sliceで進める。scheduler置換は、連続したresident chainをhost/device往復なしで
atomic commitできる段階まで延期する。
