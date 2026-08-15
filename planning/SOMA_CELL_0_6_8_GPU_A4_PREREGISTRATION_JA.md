# SOMA-CELL 0.6.8-GPU A4.7b 事前登録（A4.1〜A4.7a継承）

状態: A4.7aまでを継承するA4.7b開発slice。A4昇格判定ではない。

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

## A4.4b追加仮説

A4.4aと同じpre-existing active template / non-completion境界であれば、
proofreading、inherited quiescence、behavioural quiescence effector、および
external replicaseを、RNGを導入せずFormal066の順序どおりfixed-shape
NumPy/Torch fp64 planへ追加できる。

このsliceでもmutationは無効、template選択済み、completion未到達を必須とする。
replication入口のATPが負の場合は、A4.3のsigned payment residual範囲内であっても
rate計算へ流さず明示scope errorでfail closedとする。成功planのATP出力も
nonnegativeを必須とし、signed residualを維持できるのは未変更のfuel/mineralだけとする。
また、正のprogress increment後のfractional totalが正整数から
`4096 * eps64 * max(1, abs(total))`以内なら、CPU/CUDA除算の数ULP差を
DNA symbol数の差へ変換しないためscope code 6でfail closedとする。これは
software fp64 divisionを新設せずCPU正本を維持するための狭い境界である。
同じ係数のrelative bandをderived effective-replicase `>1e-6` と、
proofreading有効時のderived ATP gateにも適用する。ATP bandがsymbol loop途中で
検出された場合は、そのrowの先行append/payment/telemetryを全て入力値へrollbackする。

## A4.4bで実装するもの

- active protein dict順のendogenous replicaseと、その減衰後に加えるexternal `0.85`
- LOC_REPAIR regulatorだけを使うproofreading/quiescence signal
- CPU literalのpayload decode、`(mass * efficiency) * promoter`、fp64 left fold
- aggregate inhibition、protein/genome機能係数のCPUと同じ演算結合順
- proofreading speed penaltyと1 symbolあたり追加ATP `0.00075 * proof_fraction`
- inherited quiescenceと、その後のbehavioural max/clamp
- proofreadingで減衰した`last_effective_error_rate`
- 入力`cumulative_proofreading_atp`と、accepted symbolごとに逐次加算した
  `cumulative_proofreading_atp_after`
- ragged/cache/physiologyのtensor pointer/versionに加え、bind時host scalar metadataの
  lifetime attestation
- current PyTorch/CUDAでdeterministic algorithmsを有効にした直接CUDA照合
- fp64 integer-request、effective-replicase、proofreading ATP gateの曖昧帯を
  device-resident scope code 6とし、明示readback前のrequested/payment/telemetryを
  commit authorityにしない

累積proofreading ATPは0始点deltaを後で一括加算しない。既存値からsymbol loop内で
逐次`+= extra_atp`し、大きい既存値を含むCPUのfp64丸め結果をそのまま照合する。

## A4.4bで実装しないもの

- inactive cellのtemplate選択とRNG
- substitutionおよび全structural/material mutation
- copy completion、新complete genome、lesion/cycle/cache更新
- genome hydrolysis RNG
- ragged/CPU cellへのcommit、A3 scheduler authorityの置換
- fp32、compile、CUDA Graph、Triton、custom CUDA、formal benchmark
- 速度向上主張（A4.4aの負のtimingを維持する）

## A4.4b固定テスト

1. Proofreading on/off、inherited/behavioural quiescence、external on/off、全route
   combinedを直接`Formal066ProtoCell._replicate_genome`と照合する。
2. Complete-genome lesionとtemplate lesionを別方向に振り、quiescence burdenと
   effective-error入力の誤配線を検出する。
3. Proofreading追加ATP込みのexact ATP gateと`nextafter`直下、exact monomer gate、
   requested=0、resource exhaustionを固定する。
4. 非zeroかつ大きい既存`cumulative_proofreading_atp`からの逐次after値をCPUと照合する。
5. NumPy、Torch CPU、明示RTX CUDA fp64を照合し、入力値/pointer、world、RNG、
   dissipated energyが不変であることを確認する。
6. Bind後のragged/cache/physiology tensorおよびscalar metadata変更、negative ATP、
   capacity+1、inactive、completion、mutationをatomic fail closedとする。
7. 2種類のcrafted integer-boundary（pairwise由来とscalar division由来）、
   derived replicase比較、およびproofreading ATP exact/nextafter比較を
   NumPy/Torch CPU/CUDAすべてcode 6でrejectし、`1e-10`離したcontrolは同じ
   requested countで支持する。NumPy pairwise fold自体もdevice上でbit-exact照合する。
8. A4.1〜A4.4aとfocused A3 regressionを継続PASSさせ、A3の`replication_cpu`を
   exactly onceの正本として維持する。

## A4.4b判定

- 全テストがdeterministic algorithms有効の明示CUDAを含めPASSした場合だけ、
  A4.4bを「既存templateのdeterministic proofreading/quiescence/external-replicase
  paid elongation pure plan、未統合」と記録する。
- CPU演算順、累積ATP、material ledger、scope trust、CUDA fp64のいずれかが違えば
  A4.4aを維持する。
- 4096-epsilonの各離散帯は、50,000件・最大639回のproofreading支払い境界集中
  測定で観測した最大2987.54倍を次の2冪へ切り上げたengineering boundであり、
  通常位相1,000万件ではreject 0だった。ただし全実数への
  数学的証明とは主張しない。帯内rowは移植済みに数えずCPU authorityを維持する。
  software divisionや汎用数値frameworkの導入が必要になった場合はこのsliceで行わない。
- `replication_cpu`、promoted A3、`full_gpu_world_step=false`は変更しない。

## A4.5a追加仮説

既にactiveなtemplateを使い、このcallでcompletionへ到達しないA4.4bの支払schedule
なら、Formal066のper-symbol substitution RNG順を、live world RNGを動かさない
transactional event tapeとして固定できる。今回支持するmutationは同じ長さの
substitutionだけである。

inactive template startは同じCPU call内でtemplate/copyのragged topologyを新設して
直ちに伸長する別transactionなのでA4.5bへ分離する。completion後のstructural/
material mutationとhydrolysis RNGも同梱しない。

## A4.5aで実装するもの

- 明示されたcanonical NumPy PCG64 before-stateのclone
- cell入力順、paid append順のscalar `random()` 1回
- strict `uniform < effective_error`成立時だけ直後にscalar `integers(0,7)`
- `effective_error=0`でもpaid appendごとにthreshold drawを消費し、resource stop後は
  drawを消費しない凍結順序
- `state/inc/has_uint32/uinteger`を含む完全before/after state
- fixed-shape draw/replacement prefix、count、effective error、source/config/dt/schedule
  digestを持つprivate-factory `A4SubstitutionRngTape`
- hostの高水準PCG64 replayによる全draw、decision、bounded integer、after-state照合
- scalar/PCG64 metadataとresident array pointer/versionのlifetime attestation
- CPUで確定したreplacement maskのNumPy/Torch CPU/CUDA適用
- device上でcell/count/effective-error bits/decisionを再照合し、foreign physiology
  scheduleをcode 7でbatch全体rollback
- uniform/effective-errorが4096-epsilon帯内ならcode 6でbatch全体rollback
- live world RNG、source ragged/cache/physiology、CPU cell、scheduler receiptの不変

bounded integerが内部で消費するraw draw数は仮定しない。全uniformを先にvector生成、
非hit分のinteger先取り、Torch RNG、counter RNG、独自software RNGは使わない。
`rng_after_state`は将来のatomic commit候補であり、このsliceではlive authorityでない。

## A4.5aで実装しないもの

- inactive template startとtemplate/copy arena topology transaction
- completion、新complete genome、lesion/cycle/cache更新
- insertion/deletion/duplication/inversion/transpositionとmaterial mutation
- hydrolysis RNG
- world RNGへのafter-state commit、CPU cell/raggedへのcommit
- A3 scheduler authority置換、global interleaved RNG event tape
- device RNG kernel、汎用RNG framework、fp32、compile/custom CUDA、速度向上主張

現schedulerは各cellのreplication後に別RNG eventを挟むため、A4.5aのmulti-cell tapeは
direct-call oracleだけに使う。このglobal順序をtransactionとして表現する前に
world-stepへ接続しない。

## A4.5a固定テスト

1. 3つのFormal066 cellでsuffix byte、substitution event count、完全PCG64 after-stateを
   direct CPU oracleとexact照合する。
2. error=0でもthreshold drawを消費しinteger drawは0、mixed hit/missではhit直後だけ
   integer、requestedよりresource-paid数が少ない場合はpaid数だけdrawすることを固定する。
3. NumPy、Torch CPU、明示RTX CUDAで同じtapeのsuffix/event/pools/telemetryを照合し、
   source値、pointer、tape、live RNGが不変であることを確認する。
4. forged factory token、PCG64 after-state、tail、dt/config、非PCG64 state、
   resident array mutationをfail closedとする。
5. source/config/schedule digestとbefore/after PCG64 dictをupload後に変更した場合、
   resident plan前にattestation errorとする。
6. 同じragged/countでも別reactive physiologyから作ったvalid tapeをcode 7とし、
   requested/append/eventを全row rollbackする。
7. device effective errorを記録uniformの4096-epsilon帯へ調整したcaseはcode 6を
   code 7より優先し、同じく全row rollbackする。
8. completion、inactive start、structural mutationは成功扱いせず、A3
   `replication_cpu` exactly onceと`full_gpu_world_step=false`を維持する。
9. A4.1〜A4.4bの25テストを変更せず継続し、合計28/28をCUDA必須でPASSさせる。

## A4.5a判定

- 28/28、direct Formal066 RNG state、NumPy/Torch CPU/CUDA、trust/capacity/scopeが
  全てPASSした場合だけ「active-template/noncompletion substitution RNG tape pure
  plan、未統合」と記録する。
- PCG64高水準call順、full state、suffix、material ledger、device schedule照合の
  どれかが違えばA4.4bを維持する。
- schedule mismatchをrow-local成功へ縮退したり、live RNGを部分的に進めたりしない。
- A4.5a単独の速度測定・scheduler統合・A4昇格は行わない。

## A4.5b追加仮説

mutation有効、complete genomeがexactly 1本、replicase gate通過という凍結境界なら、
inactive template startを同じreplication call内のproofreading/quiescence/支払い/
substitutionまで含めたpure transaction planとして固定できる。startだけを中間commitせず、
cellごとに `integers(0,1) -> paid symbolのrandom -> hit直後integers(0,7)` を完結して
から次cellへ進む。

現NumPy PCG64ではscalar `integers(0,1)`がstate/inc/has_uint32/uintegerを変えないが、
この挙動をindex直接代入の根拠にはしない。clone generatorで高水準callを実行し、完全
before/after stateをA4.5aと同じreplayで照合する。

## A4.5bで実装するもの

- mutation有効のsubstitution-tape経路だけで、inactiveかつcomplete genome exactly 1本の
  index 0 template start
- complete genome 0のbyte-exact template参照、対応lesion（欠損時0）、empty copy、
  fractional 0からの同一call継続
- planの`template_start_events`、`selected_template_indices`、
  `template_storage_symbols`
- tapeの`template_start_mask`、`template_selection_indices`とschedule digest/replay
- cell入力順にselectionとそのcellのsubstitution drawをinterleaveする完全PCG64順
- future sequence capacity `Q + 2*start_count` とfuture symbol capacity
  `S + sum(template length) + sum(paid append)` の同時事前検証
- dt=0でも2 sequence slotとtemplate storageを必要とし、threshold drawは0という境界
- NumPy/Torch CPU/CUDA parity、source/tape/live RNG不変、start metadata trust

## A4.5bで実装しないもの

- mutation無効時のinactive start
- complete genome 0本/2本以上やreplicase-gated cellを通常no-opとして混載するworld commit
- actual ragged template/copy arena mutation、provenance再発行、live RNG after-state commit
- completion、新complete genome、lesion/cycle/cache/novel-path更新
- structural/material mutation、hydrolysis RNG
- A3 scheduler置換、global interleaved world RNG transaction
- device RNG kernel、汎用allocator/RNG framework、fp32、compile/custom CUDA、速度向上主張

start後のpaid appendがtemplate長へ到達するrowは、容量に余裕があればcode 3、同じbatchの
future capacityも超える場合は継承した後段のcode 4優先でfail closedとし、startだけをGPUで
行ってcompletionをCPUで二重実行しない。0本/2本以上やreplicase gateの凍結no-opは、
prior `last_effective_error_rate`をinput stateに持たない現pure planではcommit authorityに
しない。

## A4.5b固定テスト

1. inactive/active/inactiveの3 Formal066 cellで、selection call、suffix、lesion、pools、
   fractional/effective-error/proof ATP、substitution count、完全PCG64 after-stateをdirect
   CPU oracleと照合する。primed uint32 cacheを含める。
2. dt=0 startで+2 sequenceとtemplate bytesをexact capacity PASS、各1不足FAILとする。
   非zero paid appendではtemplate+appendのfuture symbol exact/1不足もatomicに固定する。
3. NumPy/Torch CPU/明示RTX CUDAでplan/tape/start metadataを照合し、pointer/source/live RNG
   不変を確認する。host/resident start mask/index改ざんを拒否する。
4. genome 0本/2本、replicase gate、mutation off、同一call completionをscope外とし、
   A3 `replication_cpu` exactly onceと`full_gpu_world_step=false`を維持する。
5. A4.1〜A4.5aの28テストを変更せず継続し、合計31/31をCUDA必須でPASSさせる。

## A4.5b判定

- 31/31、direct Formal066 state/PCG64、NumPy/Torch CPU/CUDA、future capacity、trust/scopeが
  全てPASSした場合だけ「mutation-enabled index-zero template-start + noncompletion
  substitution RNG tape pure plan、未統合」と記録する。
- start call順、lesion、arena容量、PCG64 full state、material ledgerのどれかが違えば
  A4.5aを維持する。
- actual arena/RNG commit、通常no-op混載、completionをこのsliceへ追加しない。
- A4.5b単独の速度測定・scheduler統合・A4昇格は行わない。

## A4.6a追加仮説

既にactiveで非emptyなtemplateと、templateより短い既存partial copyがあり、
`mutation=false`のまま今回のordered paymentで全used rowがcompletionへ到達する場合、
Formal066の完了payload・台帳・lesion・cycle・topology差分をresident pure descriptorとして
再現できる。実ragged arena、gene cache、CPU cell、live RNG、schedulerはまだcommitしない。

過去callでsubstitutionされたpartial prefixをtemplateから作り直してはならない。完成polymerは
`existing partial + current paid append`とする。`mutation=false`では
`mutate_sequence`がRNG/structural/material処理前にreturnするため、RNG stateとmutation
counterは不変である。

## A4.6aで実装するもの

- 新public dispatch `paid_replication_completion_plan`。legacy elongation/substitution APIの
  completion code 3は変更しない。public mode flagは公開しない。
- pre-existing active、entry時incomplete、mutation off、全used row same-call completionだけ。
- `completion_events`、fixed `completed_symbols/completed_lengths`。
- `new_genome_lesions = template_lesion * (0.28 + 0.22 *
  (1 - proof_fraction)) + effective_error * completed_length * 0.06`。
- `replication_cycle_deltas=1`、`topology_sequence_deltas=-1`、
  `topology_symbol_deltas=append_count-completed_length`。
- completion時`replication_fractional_after=0`。既存A4.4bのpools、last count、
  effective error、cumulative proofreading ATPのordered結果を維持する。
- 一rowでもnoncompletion/scope failureならbatch全体を成功descriptorにしない。
- NumPy/Torch CPU/CUDA fp64 parity、input pointer/value、world、RNG不変。

storage topology deltaはmaterial paymentではない。active時の`template + partial copy`を
`completed genome`へ置換するarena差分であり、future symbol capacityへpaid appendを二重加算
しない。exact current capacityが通るcaseを固定する。

## A4.6aで実装しないもの

- inactive template start、mutation-enabled completion、structural/material mutation。
- actual ragged complete-genome/lesion append、template/copy消去、provenance再発行。
- live replication cycle、gene cache、`novel_path_first_age`、CPU cellへのcommit。
- world RNG commit、hydrolysis、A3 scheduler authority置換。
- grammar daughter material mutation（actual division後のA5対象）。
- fp32、compile、custom CUDA、速度向上主張。

## A4.6a固定テスト

1. 2つのFormal066 cellで過去置換済みpartial prefix、今回paid suffix、pools、fractional reset、
   last count/error、逐次proof ATP、new lesion、cycleをdirect CPU oracleと照合する。
2. Conceptual CPU post-stateをpackし、sequence/symbol/lesion countがplan deltaと一致し、
   A4.2再decode cacheがCPU `_refresh_gene_cache` と一致することを確認する。ただしlive authority
   にはしない。
3. NumPy、Torch CPU、明示RTX CUDAで全descriptorを照合し、input tensor pointer/value、
   source cell、world、RNG不変を確認する。
4. legacy code 3、noncompletion、mixed batch、mutation on、inactive start、および10件の
   structural/schema/tail/suffix/range改ざんをfail closedとする。exact current symbol capacityで
   append二重計上がないことを固定する。
5. A4.1〜A4.5bの31テストを継続し、合計34/34をCUDA必須でPASSさせる。

## A4.6a判定

- 34/34、direct Formal066 completion、NumPy/Torch CPU/CUDA、topology/cache概念照合、
  trust/scopeが全てPASSした場合だけ「mutation-free completion payload/ledger/topology pure
  descriptor、未commit・未統合」と記録する。
- completed payload、lesion丸め順、material ledger、topology差分、CUDA fp64のどれかが
  違えばA4.5bを維持する。
- actual arena/cache/RNG/CPU commit、structural mutationをA4.6aへ追加しない。
- standalone plan validatorはcompleted prefixとlesionをsource bindingから再導出しない。
  future atomic commitではinternally generated planだけを受け、両semantic relationをattested
  sourceへ再照合する。A4.6a planを外部復元/commit authorityにしない。
- A4.6a単独の速度測定・scheduler統合・A4昇格は行わない。

## A4.6b1追加仮説

pre-existing active templateが同一callでcompletionへ到達するmutation-enabled経路なら、
paid append substitutionとcompletion直後のstructural mutationを、cellごとに連続した
combined PCG64 tapeとして固定できる。A4.6b1ではdraw/payload/material scheduleの確定だけを
行い、device上の変形適用はA4.6b2へ分離する。

## A4.6b1で実装するもの

- private-factory `A4CompletionMutationRngTape`
- pre-existing active、all-row completion、mutation enabledの限定scope
- binding/config/dtと完全PCG64 before/after stateのattestation
- cell入力順の paid append `random()` とhit直後のscalar `integers(0,7)`
- 同じcellで直ちに続く insertion/deletion/duplication/inversion/transposition の
  threshold draw mask、decision、count/position/ordinal
- insertionとminimum-length paddingのhigh-level `uint8` vector payload
- post-elongation nucleotideから一度だけ計算し、結果同値なfrozen最大長でbounded化した
  effective material budget
- pre-budget/final length、committed material delta、pre-trim event telemetry
- binding-aware host replay、host array digest、resident pointer/version attestation、
  resident cloneへ継承するexpected digestと明示readback時のcontent再照合
- exact future capacity PASSとone-short atomic failure
- NumPy tapeとTorch CPU/CUDA storage/readback parity

`MAX_GENOME_LENGTH`、`MIN_GENOME_LENGTH`、`GENE_SPAN`、alphabetは正本定数から取得し、
384や640をliteral biologyとして埋め込まない。bounded integer/vector callの内部raw draw数は
仮定せず、NumPy high-level callをそのままreplayする。

## A4.6b1で実装しないもの

- tapeからのNumPy/Torch structural/material変形（A4.6b2）
- CPUで作ったfinal genomeのtape格納とGPUへの単純copy
- mutation-free inactive start、mixed completion/noncompletion event
- actual ragged/cache/lesion/cycle/novel-path/CPU-cell commit
- live RNG after-state commit、A3 scheduler置換、world-step authority
- hydrolysis RNG（A4.7）
- division後daughter grammar mutation（A5）
- generic RNG framework、software fp64、custom CUDA/Triton、速度向上主張

material-budget tail trimは凍結CPUの物質則として保持する。全structural RNGとevent計数後に
final positive deltaをbudgetまで末尾trimし、event countersはtrim前値のまま、物質支払/返却は
final length deltaだけを使う。arena capacity不足を追加trimで隠すことは禁止する。
CPUでfinite integerへ変換できる大きなnucleotide budgetは、構造的な正のdelta上限である
frozen `MAX_GENOME_LENGTH`へeffective budgetだけをcapする。raw fp64 quotientがnonfiniteに
なるpathological poolは、CPU正本の変換例外をA4だけ成功へ変えず明示scope errorにする。

## A4.6b1固定テスト

1. primed PCG64の3 Formal066 rowでappend miss/hit、conditional integer、insertion、
   deletion、duplication、inversion、transposition、padding、budget trim、negative delta refund、
   complete after-stateをexact照合する。
2. NumPy tapeをTorch CPUと明示RTX CUDAへupload/readbackし、全dtype/shape/value、pointer、
   version、scalar metadata、expected content digest、`.data` version-bypass拒否、
   source/world/live RNG不変を確認する。
3. schema/source/config/dt/PCG64、append/structural payload、bounds、padding tail、material
   budget/delta/event改ざんをfail closedとし、exact future capacity PASS、one-short FAILを固定する。
4. A4.1〜A4.6aの34 testsを変更せず継続し、合計37/37をCUDA必須でPASSさせる。
5. A3 `replication_cpu` exactly once、promoted A3、`full_gpu_world_step=false`を維持する。

## A4.6b1判定

- 37/37、Formal066 high-level RNG order、full PCG64 state、material/event schedule、
  NumPy/Torch CPU/CUDA tape attestationが全てPASSした場合だけ「A4.6b1 combined
  completion-mutation RNG tape、未適用・未統合」と記録する。
- CPU final genomeをtapeへ保存してdevice applyを省略しない。
- A4.6b1だけでA4.6b完了、structural mutation移植済み、scheduler authorityとは呼ばない。

次はA4.6b2としてattested operation tapeをNumPy/Torch CPU/CUDAの固定shape変形・材料・
lesion・topology descriptorへ適用する。その後hydrolysisをA4.7として分離する。scheduler
置換は、連続したresident chainをhost/device往復なしでatomic commitできる段階まで延期する。

## A4.6b2追加仮説

A4.6b1で高水準PCG64 callと全structural payloadをbinding-awareに固定できたなら、
CPUで作ったfinal genomeをtapeへ追加せず、同じoperation列をNumPy/Torch CPU/CUDAの
fixed-shape配列へ適用して、Formal066と同じfinal polymer、材料、lesion、cycle、topology、
次state用derived値をpure descriptorとして再現できる。

## A4.6b2で実装するもの

- private-factory A4.6b1 tapeだけを受ける`A4CompletionMutationPlan`
- paid append substitution後、insertion、deletion、gene duplication、inversion、
  transposition、minimum-length padding、frozen maximum/material-budget tail trimを
  evolving polymerへ順に適用するNumPy/Torch CPU/CUDA固定shape経路
- final symbols/length、pools、requested/append/last/error/proof telemetry、
  substitution/5 structural event counts、material delta
- template lesionとordered proofreading signalからinherited lesion成分を直接再計算し、
  effective-error length項へfinal mutated lengthを使うnew lesion。丸め済みA4.6a値からの
  subtractive recoveryは禁止
- cycle/topology delta、replication active/template lesion/fractional reset、
  post genome count/material-symbol count/ordered lesion mean
- b1 tapeのscalar/pointer/version/digestに加え、32 public resident arraysをvalidated upload時の
  private expected tensorsとdevice上exact比較して`.data` version bypassを拒否するtrust境界
- 一rowでもtape/operation/bounds/material/capacity/derived-state不一致ならcode 7で
  used batch全体rollback
- NumPy/Torch CPU/明示RTX CUDA parity、input/tape/world/live RNG不変

event countはmaterial trim前、物質支払/返却はfinal length deltaだけを正本とする。
host tapeのPython float64/int budgetをtrim authorityとし、resident quotientは同じinteger
bucketを4096-epsilon帯込みで検証する。pre-structural長がMIN未満なら、padding draw後に
zero budgetで元の短い長さへtrimされることを許す。structural mutationはATPを消費しない。
lesion meanはold+newのcombined prefixを一度にreduceし、NumPy正本のpairwise groupingを
Torch CPU/CUDAで再現する。arena容量不足を追加trimで隠さない。

## A4.6b2で実装しないもの

- actual ragged completed-genome/lesion append、template/copy消去、provenance再発行
- live replication cycle/material/lesion state、gene cache、novel-path、CPU cellへのcommit
- live PCG64 after-state commit、world-step interleave、A3 scheduler authority置換
- inactive-template completion、mixed completion/noncompletion、mutation-free start
- hydrolysis RNG（A4.7）
- division後daughter grammar mutation（A5）
- generic mutation/RNG framework、software fp64、custom CUDA/Triton、速度向上主張

## A4.6b2固定テスト

1. 3 Formal066 rowでappend substitution、5 structural operations、padding、material trim/refund、
   final polymer、pools、lesion、cycle/reset、topology、post count/material/lesion meanをdirect
   CPU oracleと照合する。
2. NumPy、Torch CPU、明示RTX CUDAで全fixed plan arraysを照合し、全outputが同device、
   source ragged/cache/physiology/tape pointer/valueが不変であることを確認する。
3. pre長8/padding/zero-budget/final長8、exact host budget 19、old lesion 7→post 8の
   pairwise境界、大きなfinite template-lesionでのsubtractive-cancellation反例、および
   lesion count 1〜8/複数seed/configでfinal polymer、material、lesionを照合する。
4. 9 plan field corruptionsとCPU/CUDA tape `.data` version bypassを拒否し、scope code 7、
   zeroed success telemetry、全row rollback、live RNG/A3 authority不変を固定する。
5. A4.1〜A4.6b1の37 testsを継続し、合計40/40をCUDA必須でPASSさせる。

## A4.6b2判定

- 40/40、direct Formal066 final state、NumPy/Torch CPU/CUDA、resident trust、capacity、
  source/RNG非変更が全てPASSした場合だけ「A4.6b2 structural/material completion mutation
  pure resident descriptor、未commit・未統合」と記録する。
- CPUで作ったfinal genomeをtapeへ格納したり、arena overflowをmaterial trimへ混同しない。
- plan単体を外部復元/commit authorityにせず、将来commit直前にbinding/source semanticを
  再照合する。
- A4.6b2だけでA4完了、scheduler authority、full GPU world-step、速度向上とは呼ばない。

次はA4.7としてsymbol hydrolysis RNGを別event tape/planに分離する。その後にだけ、
resident replication chainのatomic arena/cache/RNG commitとscheduler統合を検討する。

## A4.7a追加仮説

Formal066実MROで凍結0.3の`_decay_information_and_proteins`へ到達するlive-genome
hydrolysisは、lesion gain後のcomplete genome polymer/lesionを同時に拘束する一cell
`A4TranslationBinding`があれば、arenaを変更せず高水準PCG64 callだけを独立tapeとして
固定できる。world RNG interleaveを飛ばさないため、multi-cell連続tapeにはしない。

## A4.7aで実装するもの

- 新規core `SOMA_CELL_0_6_8_gpu_a4_hydrolysis.py`
- private-factory `A4HydrolysisRngTape`
- post-lesion-gain / pre-hydrolysis、exactly one cell、complete genomeごとにlesionが
  exactly oneのNumPy `A4TranslationBinding`限定scope
- `sequence_capacity`に整列した`genome_slot_mask`、`draw_mask`、`uniform_draws`、
  `hit_mask`、`deletion_positions`
- cell ID、sequence/genome count、source provenance、exact `dt_hex`、draw/hit count、
  schedule SHA-256、完全なPCG64 before/after state
- cloned NumPy PCG64によるscalar `random()`と、hit直後だけの
  scalar `integers(0,length)` high-level replay
- host array digest、resident scalar/pointer/version、private expected tensor、明示D2Hでの
  actual/expected content digest照合
- clone、Torch CPU/CUDA upload、明示NumPy readback、state dict。ただしrestore constructorや
  live authorityは作らない

凍結literal gateは`lesion > 0.75 and length > MIN_GENOME_LENGTH`である。このgateを
通ったcomplete genomeは、`dt==0`で確率が0でも`random()`を1回消費する。hit判定は
strict `draw < dt * 0.00065 * lesion`であり、hitした場合だけ直ちにlengthを上限とする
bounded integerを1回呼ぶ。template/copyは走査しない。

通常のresident require/clone/state-dictはmetadata、pointer、versionだけを検証し、
`.cpu()`、`.numpy()`、`.tolist()`、`.item()`、`torch.equal`によるhidden D2H/syncを行わない。
contentの実値検査は明示`to_numpy()`だけで行い、Torch `.data` version bypassはそこで
original host digest不一致としてrejectする。

現A3 `genome_hydrolysis_plan`は`probability<=0`をskipするため、eligibleかつ`dt==0`で
凍結CPUと異なりdrawを消費しない。この差は隠さず既知bridge差として固定する。A4.7aでは
A3 schedulerを変更せず、後続integration前にbridgeとlockstep testを別sliceで修正する。

## A4.7aで実装しないもの

- deletionをpolymerへ適用するNumPy/Torch plan（A4.7b）
- ragged arena再compact、lesion `*=0.80`、wasteへの`MONOMER_MASS`返却、
  `genome_damage_events`更新、gene-cache refresh
- completion planとの連結、actual arena/cache/CPU-cell/world commit
- live PCG64 after-state commit、A3 hydrolysis bridge/scheduler authority置換
- device RNG kernel、generic RNG framework、software fp64、custom CUDA/Triton
- division後daughter grammar mutation（A5）、速度向上主張、A4昇格

## A4.7a固定テスト

1. lesion `==0.75`、length `==MIN_GENOME_LENGTH`はdrawなし、eligible lesion/lengthかつ
   `dt==0`はdraw exactly one / hit zeroとして凍結literal orderを固定する。
2. miss/hitを含むcomplete-genome順のscalar random/conditional integer、完全PCG64
   before/after、source/live RNG不変をdirect frozen CPU oracleと照合する。
3. Torch CPUと明示RTX CUDA upload/readback、pointer/version/content trust、scalar/tail/source/
   dt/RNG改ざん、`.data` version bypass、single-cell/post-gain scopeを検証する。
4. A4.1〜A4.6b2の40 testsを変更せず継続し、合計43/43をCUDA必須でPASSさせる。

## A4.7a判定

- 43/43、凍結gate/call order、完全PCG64 state、NumPy/Torch CPU/CUDA tape storage、
  trust/scope、source/live RNG非変更が全てPASSした場合だけ「A4.7a single-cell
  post-gain hydrolysis RNG tape、未適用・未統合」と記録する。
- tapeだけでsymbol hydrolysis移植済み、live RNG authority、scheduler authority、
  full GPU world-step、速度向上とは呼ばない。

## A4.7b追加仮説

A4.7aのattested tapeと同じpost-lesion-gain / pre-hydrolysis一cell bindingがあれば、
凍結0.3のsymbol deletion、waste返却、lesion減衰、damage event、cache refresh意味を、
compact arenaやlive stateを変更せずfixed row-padded descriptorとして独立照合できる。
actual ragged再compactとatomic commitを同時に行う必要はない。

## A4.7bで実装するもの

- coreは既存`SOMA_CELL_0_6_8_gpu_a4_hydrolysis.py`だけを拡張する。
- pure `A4HydrolysisDeletionPlan`。scalar identityは`sequence_capacity Q`、
  `max_sequence_symbols W`、sequence/genome/source-symbol count、cell ID、source
  provenance、A4.7a tape schedule digest。
- public APIは`genome_hydrolysis_deletion_numpy`、
  `genome_hydrolysis_deletion_torch`、`genome_hydrolysis_deletion_plan`、および
  host binding-aware `validate_a4_hydrolysis_deletion_plan`。
- fixed arraysは`scope_valid[1]`、`scope_error_code[1]`、
  `final_symbols[Q,W]`、`final_lengths[Q]`、`genome_lesions_after[Q]`、
  `pools_after[1,POOL_COUNT]`、`symbol_count_after[1]`、
  `topology_symbol_delta[1]`、`genome_damage_event_delta[1]`、
  `gene_cache_dirty[1]`、`gene_cache_refresh_count[1]`、
  `genome_material_symbols_after[1]`、`genome_lesion_mean_after[1]`。
- 全used sequenceをsource順のrow-padded表現にする。hit complete genomeだけattested
  positionを1 symbol除去し、missとactive template/copyはbyte-exact、全tailはzero。

各hitはgenome順に、delete、wasteへ`MONOMER_MASS`をfp64で1回加算、当該lesionを
`*=0.80`、damage eventとcache refresh countを各1増加、の順を保つ。wasteを
`initial + hit_count * MONOMER_MASS`で一括計算しない。CPUの各hit直後cache refreshは
このloop中にreaderがないため、A4.7bではliteral refresh countとdirtyだけを保持する。
final gene cacheはconceptual final complete genomesを既存A4.2 decoderへ渡して照合するが、
live cacheにはしない。

`genome_lesion_mean_after`は更新後complete lesion全prefixを1配列としてNumPy contiguous
pairwise groupingでreduceする。CPU/CUDAでsumはexactでも最後のgenome-count divisionだけ
1 ULP異なる実測edgeがあるため、このfieldだけfinite/nonnegative expectedの直前・直後
`nextafter`までを許容する。他のsymbols、lengths、各lesion、sequential waste、event/
material ledgerはexactを維持する。software fp64 divisionは追加しない。

NumPyはsource/dt/tapeをfull replayしてから適用する。Torch CPU/CUDAはbinding provenance、
capacity、dt、schedule、backend/deviceを照合し、tape public tensorとprivate expected tensorを
device上で全field比較する。`.data` bypassまたはsemantic mismatchはsingle global failureとし、
polymer/lesion/pools/derived stateをsource値へrollback、event telemetryをzeroにする。
通常経路に`.item()`、`.cpu()`、`.numpy()`、`.tolist()`、`torch.equal`、hidden syncを入れない。

## A4.7bで実装しないもの

- row-padded outputからactual compact `A4RaggedGenomeBatch`を再構築・差替するcommit
- actual gene-cache refresh/差替、CPU cell/world material/event更新
- live PCG64 after-state commit、A3 hydrolysis bridge/scheduler authority置換
- A3のeligible `dt==0` draw欠落修正、multi-cell tape連結/interleave
- completion planとのatomic連結、device RNG、generic allocator/RNG framework
- division/death/corpse/eDNA/HGT（A5）、neural/causal（A6）、速度向上主張、A4昇格

## A4.7b固定テスト

1. A4.7a hit/miss/hit tapeをdirect Formal066へ適用し、final polymers、genome順の
   sequential waste、lesion `*0.80`、event/topology/material、cache dirty/refresh count、
   conceptual final cacheを照合する。active template/copyは不変、`dt==0`はidentity。
2. NumPy、Torch CPU、明示RTX CUDAで全plan arrays、device residency、source/tape/world/live
   RNG purityを照合する。更新後lesion prefix groupingを固定し、3-genome巨大finite fixtureで
   sum exactかつfinal divisionだけ1 ULPを許すことを示す。
3. exact capacity、position bound、tail、plan field、source/dt/schedule、resident public/private
   tape `.data`を検査する。失敗時はsingle global rollback、明示readback後host replay reject、
   no hidden D2H、A3 authority不変を固定する。
4. A4.1〜A4.7aの43 testsを変更せず継続し、合計46/46をCUDA必須でPASSさせる。

## A4.7b判定

- 46/46、direct Formal066、NumPy/Torch CPU/CUDA、sequential ledger、row-padded polymer、
  derived state、trust/rollback、source/RNG非変更が全てPASSした場合だけ「A4.7b
  single-cell hydrolysis deletion/ledger pure descriptor、未compact・未commit・未統合」と
  記録する。
- row-padded planをactual arena、live cache/RNG、CPU/world commit、scheduler authority、
  full GPU world-step、速度向上とは呼ばない。
- 次はresident chain全体のsource/tape/planをcommit直前に再attestし、compact arena、cache、
  live RNG、CPU/worldとの境界をatomicに扱う別sliceとする。A3 dt0 bridge修正とmulti-cell
  event interleaveもその統合前に別途固定する。

## A4.8a追加仮説

A4.7a/bで、post-lesion-gainの一cell source、literal PCG64 call列、row-padded deletion/
ledger結果をbinding-awareに再演算できるようになった。そこで既存A3 event順とevent名を
変えず、`genome_hydrolysis_cpu_rng` 一eventだけを、外部planを受けない内部生成・即時検証・
原子的publish bridgeへ置換できるはずである。

これはA4.7b末尾の「scheduler統合は後段」という一般則に対する、hydrolysis一event限定の
correctness exceptionである。per-cell host/device往復を含み、速度候補、resident world-step、
A4主経路、A4昇格とは扱わない。translation、replication、multi-cell RNG batchを同時に
置換しない。

## A4.8aで実装するもの

- promoted A3 source/schedulerを変更せず、新規A4 integration moduleだけに置く
  `A3EventScheduler` subclass。既存event IDとcell event順を維持する。
- A4 scheduler/config/deviceをstep境界で保存・復元する最小Hybrid world wrapper。
  active schedulerまたはpending transactionのsave/cloneは拒否し、完了eventのtape、plan、
  bindingは永続化しない。
- post-`genome_lesion_gain`のlive cellを一cellNumPy A4 bindingへpackし、live PCG64 before-stateの
  cloneからA4.7a tapeを作り、明示Torch CPU/CUDA resident plan、明示readback、binding-aware
  host replay、compact CPU candidate、fresh A4 bindingまでclaim前に完成させるprivate one-shot
  transaction。
- live cell identity/source、exact `dt.hex()`、A4 capacities、world config、完全PCG64 before-stateを
  commit直前に再照合してから、既存`genome_hydrolysis_cpu_rng`をexactly once claimする。
- 成功時だけcomplete genomes、ordered lesions、pools、gene cache、`genome_damage_events`、完全
  PCG64 after-stateを一括publishする。active replication template/copyと他stateは不変にする。
- hitごとのsequential waste、lesion `*=0.80`、damage-event delta、final cacheをA4.7b正本から
  commitする。no-hitおよびeligible `dt==0`もeventはexecuted、delete workはfalseだが、凍結
  literalどおりeligible genomeごとにrandom drawを消費する。
- preclaim failureはreceipt、生物状態、live RNGを完全不変とする。claim後publish exceptionは
  touched CPU/RNG stateを局所rollbackし、outer schedulerにはaborted receiptを残す。
- scheduler receiptへsource/final provenance、draw/hit count、device、A4.8a authority、
  `work_performed`を記録する。`full_gpu_world_step=false`を維持する。

live commitは内部factoryで作ったcandidateだけを受ける。public APIからtape/plan/candidateを
注入したり、scope/capacity/trust failureを旧CPU bridgeへfallbackしたりしない。旧A3 hazard
tupleは呼出形だけ検証しても、生物/RNG authorityには使わず、post-gain sourceからliteral gateを
再導出する。

## A4.8aで実装しないもの

- A4.3 translation commitまたは`translation_cpu` scheduler置換
- A4.4〜A4.6 replication commit、inactive/no-op/completion fallback分類、live mutation ledger
- multi-cell連続tape、cell順の並べ替え、device RNG、persistent cross-step arena authority
- generic transaction manager、generic allocator/RNG framework、custom CUDA/Triton、fp32
- division/death/corpse/eDNA/HGT（A5）、neural/causal system（A6）
- promoted A3 file、A3 baseline/results、event ID/orderの変更
- performance、speedup、full GPU world-step、A4完成または昇格の主張

## A4.8a固定テスト

1. direct Formal066のhit/miss/hit、no-hit、eligible `dt==0`を照合し、compact genomes/offsets、
   lesions、sequential waste、cache、damage counter、完全PCG64 after-stateをexactに固定する。
2. Torch CPUと明示RTX CUDA candidate/commitを照合し、source/live RNG purity、fresh bindingの
   full validation、変更後sourceに対するold binding/candidateの拒否を確認する。
3. wrong cell/dt/source/config/RNG、tape/plan/public・private `.data` tamper、Q/S/W exactと
   one-short、duplicate/out-of-order、one-shot再利用、claim前失敗、注入publish失敗の局所
   rollbackをfail closedとして固定する。
4. 2-cellのper-cell replication→hydrolysis→later RNG interleave、旧CPU hydrolysis bridge未呼出し、
   1-step/10-step/stress、step-boundary save/load/clone継続、A3 authority非重複を固定する。
5. A4.1〜A4.7bの46 testsを変更せず継続し、合計最大50 testsをCUDA必須でPASSさせる。

## A4.8a判定

- 上記全test、direct CPU semantics、PCG64 interleave、atomic rollback、fresh source/cache binding、
  save/clone、promoted A3 byte不変がPASSした場合だけ「A4.8a single-cell genome hydrolysis
  atomic commit bridge」と記録する。
- A4.8aだけでtranslation/replication統合、A4 world-step完成、persistent GPU authority、speedup、
  A4昇格とは呼ばない。
- 次はA4.8b1としてtranslationのbinding-aware output replayとCPU commitを先に作る。
  replicationはさらにA4.8b2へ分離し、single-cell event orderとlive PCG64 commitを独立固定する。

## A4.8b追加仮説

A4.3のpaid translation planは、凍結CPUの遺伝子挿入順、逐次FUEL/MINERAL/ATP支払い、
active/damaged protein辞書、`last_translation`、条件付き`last_quiescence`をNumPyとresident
Torch fp64で再現する。A4.8aで確立したclaim前のsource/resident/output再attestとclaim後の局所
rollbackをtranslation一eventへ限定して適用すれば、既存`translation_cpu` eventを外部planや
CPU fallbackなしで原子的に置換できるはずである。

commit値は明示readbackしたresident Torch出力とする。独立NumPy replayはdiscrete値と辞書順を
exact、floatを最大`2e-12`でclaim前に検証するoracleであり、host値を代わりにcommitしない。
通常有効fixtureでもCPU/CUDAに数ULP差があるためbit-exactを偽装しない。resident差が直後のCPU
replicationのgate、RNG、離散結果を変える反例が出た場合はCPUへfallbackせずA4.8bをSTOPする。

## A4.8bで実装するもの

- A4.8a schedulerを継承する新schedulerとworld wrapper。promoted A3、A4.3 pure core、A4.8a
  classを変更せず、既存event ID/rankの`translation_cpu`だけを追加置換する。hydrolysisは
  A4.8a、replication以下はA3 authorityを維持する。
- post-maintenance/live-translation境界の一cellをfresh host A4 bindingへpackし、独立NumPy
  replay、明示Torch CPU/CUDA source/cache/state、resident paid plan、全resident値の明示readback、
  fresh CPU candidateとfresh bindingまでclaim前に作るprivate one-shot candidate。
- live cell object/ID/generation、exact `dt.hex()`、world config bytes、A4 capacity/device、binding
  provenance、resident pointer/version/content、candidate physiologyをcommit直前に再照合する。
  Torch `.data` version bypassも全値readbackで拒否する。
- NumPyとresidentのcell/mask/fingerprint/count/orderをexact比較し、全float64配列を`2e-12`
  以内で比較する。signed fp64支払残差はclipせず、schemaで許すpaid poolだけ保持する。
- 検証後だけ`translation_cpu`をexactly once claimし、resident候補の`pools`、active `proteins`、
  `damaged_proteins`、`last_translation`、条件付き`last_quiescence`を一括publishする。CPUの
  `_sync_protein_pool`/`_sync_damage_pool`と同じ閾値・挿入順を保つ。
- genomes、lesions、gene specs/cache、replication state、world state、live PCG64 identity/stateは
  不変にする。no-op、disabled、no-genome、translator gateもeventはexecutedでwork=falseとする。
- preclaim failureはreceipt/biology/RNGを完全不変とし旧CPU translateへfallbackしない。claim後
  publish例外は元のpool/dictionary/scalar object identityと値へrollbackし、outer schedulerへ
  aborted receiptを残す。
- scheduler/worldのA4.8b schema/config/device/typeをsave/load/cloneで保持し、active/pending
  serializationを拒否する。`full_gpu_world_step=false`、速度主張なしを維持する。

## A4.8bで実装しないもの

- replication template start、paid elongation、substitution/structural completionのlive commit
- translationとreplicationを一candidateへ束ねるgeneric transaction manager
- persistent cross-step arena/cache、multi-cell translation batch、device RNG、CPU fallback
- promoted A3 file、A4.3 pure plan、A4.8a class、event ID/rank、baseline/resultsの変更
- division/death/corpse/eDNA/HGT（A5）、neural/causal system（A6）
- fp32、compile/graph/Triton/custom CUDA、performance、speedup、full GPU world-step、A4昇格

## A4.8b固定テスト

1. rich/exhaustion/reserve/no-translator、disabled/no-genome、`dt==0`、実signed residualをdirect
   Formal066 translateと照合する。辞書順・離散値・genotype/RNGはexact、floatは`2e-12`以内、
   no-opもreceipt exactly onceとする。
2. 明示Torch CPU/CUDA candidate/commit、independent NumPy replay、resident device/pointer、source
   purity、fresh binding、resident値publish、one-shotを照合する。genome/cache/RNGは不変とする。
3. Q/S/W/P exactと各one-short、wrong cell/dt/source/config/device、host replay、resident
   ragged/state/cache/plan `.data`、candidate/fresh binding tamper、duplicate/out-of-order、claim前
   failure、注入publish failureのrollbackをfail closedとして固定する。
4. 1-step/10-step/translation-heavy stress/2-cell event interleave、旧CPU translate bridge未呼出し、
   save/load/clone継続、active/pending拒否、A3 byte不変を固定する。resident translation commit直後の
   CPU replicationについてreplicase/resource exact/nextafter境界、RNG state、離散copy結果をdirect
   CPU-translation後と照合する。
5. A4.1〜A4.8aの50 testsを変更せず継続し、合計最大54 testsをCUDA必須でPASSさせる。

## A4.8b判定

- 54/54、direct CPU semantics、resident/NumPy association、signed ledger、dict order、直後CPU
  replicationのgate/RNG、atomic rollback、save/clone、promoted A3 byte不変が全てPASSした場合だけ
  「A4.8b paid translation atomic commit bridge」と記録する。
- CPU/CUDA差が直後replicationの離散結果を変える、hidden D2H、外部plan authority、host値commit、
  CPU fallback、scope拡大が必要になった場合はA4.8aを維持してSTOPする。
- 次sliceはreplication一eventをinactive/no-op/partial/completion/mutationごとに明示分類して扱う。
  A4.8bだけでA4完成、persistent GPU world、speedup、昇格とは呼ばない。

## A4.8c1追加仮説

A4.4b/A4.5aは、pre-existing active templateの非完了paid elongationについて、mutation
offのdeterministic planとmutation onのPCG64 substitution tape/planを既に持つ。A4.8bで
確立したresident authority、claim前再attest、CPU-cellへの原子的publishをこの1分岐だけへ
適用すれば、既存`replication_cpu` eventを外部planやCPU fallbackなしで安全に置換できる
はずである。

ただし既存pure APIは、mutation-free inactive start、inactive startから同一callでのcompletion、
およびdisabled/no-genome/replicase-gate等の通常early no-opをcommit descriptorとして閉じて
いない。したがってA4.8cを一括実装せず、A4.8c1はalive、`config is world.config`、
`genome_replication=true`、pre-existing active nonempty template、entry copyがtemplate未満、
replicase gate通過、非負ATP、same-call noncompletionに限定する。

## A4.8c1で実装するもの

- A4.8b scheduler/worldを継承する新規replication integration module。promoted A3、A4.4〜
  A4.6 pure core、A4.8a/bを変更せず、既存event ID/rankの`replication_cpu`だけを追加置換する。
- mutation offは`paid_replication_elongation_plan`、mutation onはlive PCG64 before-stateのcloneから
  `prepare_substitution_rng_tape`を作り、resident tapeと`paid_replication_substitution_plan`を
  使用する。mutation onのscalar random→conditional integer順と完全PCG64 after-stateを保つ。
- requested zero、`dt==0`、resource stopでappend zeroでも、active gate通過後のfractional、
  effective error、last count更新を含むvalid noncompletionとして扱う。単純early no-opへ畳まない。
- 一cellhost binding、独立NumPy oracle、明示Torch CPU/CUDA binding/cache/state/tape/plan、全resident
  値の明示readback、resident結果からのprivate deep-copied candidate、fresh bindingをclaim前に作る。
- commit直前にcell/object/generation/alive、exact `dt.hex()`、world config bytes、A4 capacity/device、
  live PCG64、host source/oracle/tape、resident artifact object/backend/device/pointer/content、candidate
  全stateを再attestする。Torch `.data`、whole-member、host-plan authority差替えを拒否する。
- NumPyとresidentはcell/scope/append bytes/count/discrete順をexact、float64を最大`2e-12`で照合する。
  commit値はresident readbackであり、host oracle値への差替えは禁止する。
- 検証後だけ`replication_cpu`をexactly once claimし、元`pools` arrayへNUCLEOTIDE/ATP結果、元
  `replication_copy` listへsuffix、`replication_fractional`、`last_replication_symbols`、
  `last_effective_error_rate`、`cumulative_proofreading_atp`、元`mutation_events` dictの
  substitution deltaをpublishする。mutation onだけ同じlive PCG64 objectへattested after-stateを
  適用する。
- genomes/lesions/template/gene specs outer+nested/proteins/damaged proteins/replication cycles/
  novel-path/world energyのidentityと値を不変にする。mutation offでは完全PCG64 stateも不変にする。
- preclaim failureはreceipt/biology/RNGを完全不変とし、旧CPU replicationへfallbackしない。
  claim後publish例外はpool/copy/mutation-event/telemetry/RNGを元object identity・順序・値へ
  rollbackし、outer schedulerへaborted receiptを残す。
- scheduler/worldのA4.8c1 schema/config/device/typeをsave/load/cloneで保持し、active/pending
  serializationを拒否する。candidate/binding/tape/planは保存しない。

## A4.8c1で実装しないもの

- inactive template start、startと同一callの伸長・completion
- pre-existing active templateのsame-call completion（mutation off/on）
- disabled、no-genome、replicase gate、inactive two-genome等のearly no-op authority
- completion topology、new genome/lesion/cycle/cache/novel-pathのlive commit
- caller supplied tape/plan/candidate、CPU fallback、multi-cell replication batch、device RNG
- persistent cross-step arena/cache、generic transaction/RNG framework、A4.4〜A4.6の横断refactor
- promoted A3/A4.8a/b、event ID/order、baseline/resultsの変更
- division/death/corpse/eDNA/HGT（A5）、neural/causal system（A6）
- fp32、compile/graph/Triton/custom CUDA、performance、speedup、full GPU world-step、A4完成/昇格

## A4.8c1固定テスト

1. mutation offのrich active partialとmutation onのforced miss/hit、`dt==0`、requested zero、
   resource stopをdirect Formal066と照合する。copy suffix、paid pools、fractional/error/proof ledger、
   substitution counter、完全PCG64、CPU object identityを固定する。
2. 明示Torch CPU/CUDA candidate/commit、independent NumPy replay、resident device/pointer/content、
   prepare purity、fresh binding/cache、mutation-off RNG不変、mutation-on after-state、one-shotを照合する。
3. exact Q/S/W/Pと各one-short、wrong cell/dt/source/config/device/RNG、host oracle/tape、resident
   ragged/state/cache/tape/plan public・private・whole-member・`.data`、candidate/fresh binding tamper、
   packed/unpacked alias、duplicate/out-of-order、claim前失敗、注入publish失敗のidentity rollbackを
   fail closedとして固定する。inactive/start/completion/disabled/no-genome/replicase/negative ATP/
   fp64 code6はno fallbackでscope外とする。
4. active noncompletionの1-step/10-step、2-cell非batch replication→surface→hydrolysis RNG順、旧CPU
   replication bridge未呼出し、save/load/clone、active/pending拒否を固定する。直後surface assemblyの
   ATP gate exact/nextbelow/nextaboveと後続PCG64/discrete stateをdirect CPU twinへ照合する。
5. A4.1〜A4.8bの54 testsを変更せず継続し、合計最大58 testsをCUDA必須でPASSさせる。

## A4.8c1判定

- 58/58、A3 36/36、direct CPU semantics、resident/NumPy association、copy/ledger/object identity、
  live PCG64、atomic rollback、save/clone、後続surface gate、promoted A3/A4.8a/b byte不変がPASSした
  場合だけ「A4.8c1 pre-existing-active noncompletion replication atomic commit bridge」と記録する。
- completion/start/early no-op、離散/RNG不一致、CPU fallback、host値commit、scope拡大が必要なら
  A4.8bを維持してSTOPする。tolerance拡大やsilent clipで通さない。
- 次は同じ4 testを拡張し、pre-existing active completion、その後inactive start/early no-opの順に
  別sliceで閉じる。A4.8c1だけでA4完成、persistent GPU world、speedup、昇格とは呼ばない。

## A4.8c2追加仮説

Rule Lock receipt `20260815T054018Z` の範囲は、pre-existing active nonempty template、
entry時incomplete copy、`mutation=false`、same-call completionの原子的commitだけとする。
A4.8c1 coreは変更せず、新規integration moduleのscheduler/world subclassでこの分岐だけを
追加すれば、A4.6a completion descriptorをresident authorityとして凍結CPU semanticsどおり
publishできるはずである。

## A4.8c2で実装するもの

- A4.8c1 scheduler/worldを継承する新規integration module。既存event ID/rankの
  `replication_cpu`をexactly once claimし、A4.8c1 source/coreとpromoted A3は変更しない。
- claim前にA4.6a `paid_replication_completion_plan`を内部生成し、resident Torch CPU/CUDA
  fp64 outputを明示readbackする。独立NumPy replayはdiscrete値をexact、float64を最大
  `2e-12`で照合するoracleに限定し、commit値はresident readbackとする。
- completion payloadが`existing partial + paid suffix`であること、pools、proofreading台帳、
  last count/error、fractional reset、new lesion、cycle/topology差分をsourceから再attestする。
- 検証後だけcomplete genome/lesion追加、template/template lesion消去、copy reset、fractional、
  cycle、gene cache、条件付き`novel_path_first_age`、paid pools/telemetryを一括publishする。
- preclaim failureはreceipt/biology/live PCG64を不変にし、claim後publish exceptionは触れた
  object identity・順序・値を局所rollbackする。`mutation=false`のため成功時も完全PCG64
  identity/stateとmutation countersを不変にする。旧CPU bridgeへのfallbackは禁止する。
- scheduler/worldのA4.8c2 type/schema/config/deviceをstep境界のsave/load/cloneで保持し、
  active/pending serializationを拒否する。`full_gpu_world_step=false`を維持する。

## A4.8c2で実装しないもの

- mutation-on same-call completion、inactive template start、およびdisabled/no-genome/
  replicase-gate等のpre-active-gate early-return authority。これらへ到達したらno fallbackでSTOPする。
  active gate通過後の`dt==0`、requested-zero、resource stop noncompletionはA4.8c1へ明示delegateする。
- caller supplied descriptor/candidate、multi-cell batch、device RNG、persistent arena/cache、
  generic transaction/RNG framework、A4.6aまたはA4.8c1 coreの変更。
- fp32、compile/graph/Triton/custom CUDA、performance、speedup、A4完成/昇格、A5/A6。

## A4.8c2固定テスト

1. mutation-free active completionをdirect Formal066と照合し、completed payload、paid pools、
   proof/fractional/error、new lesion、cycle/template/copy/cache/novel-path、RNG不変を固定する。
2. 明示Torch CPU/CUDA candidate/commit、independent NumPy oracle、resident device/pointer/content、
   fresh binding/cache、one-shot、A4.6a source relationを照合する。
3. exact capacityと各one-short、source/dt/config/device/descriptor/candidate tamper、duplicate/
   out-of-order、claim前失敗、注入publish失敗のidentity rollbackをfail closedで固定する。
   mutation-on completion、inactive start、pre-active-gate early returnはscope外かつno fallbackとする。
4. completion後の後続event、save/load/clone、旧CPU replication未呼出しを照合し、既存58 testsを
   変更せず継続して既存58 + 新規最大4 = 合計最大62 testsをCUDA必須でPASSさせる。

## A4.8c2判定

- 62/62とA3 regression、direct CPU semantics、resident/NumPy association、atomic rollback、
  RNG不変、save/clone、promoted A3/A4.8c1 core不変がPASSした場合だけ「A4.8c2
  mutation-free pre-existing-active completion atomic commit bridge」と記録する。
- mutation-on completion、inactive start、pre-active-gate early return、CPU fallback、host値commit、scope拡大が
  必要ならA4.8c1を維持してSTOPする。A4.8c2でもA4は未完であり、
  `full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c3追加仮説

Rule Lock receipt `20260815T064751Z` の範囲は、pre-existing active nonempty template、
entry時incomplete copy、`mutation=true`、same-call completionの原子的commitだけとする。
A4.6b1のcombined PCG64 tapeとA4.6b2のresident final descriptorを、一cellの凍結CPU順序で
claim前に再生成・再attestすれば、substitution、5種のstructural mutation、padding、material
trim/refund、completion topologyを外部planやCPU fallbackなしでpublishできるはずである。

## A4.8c3で実装するもの

- A4.8c2 scheduler/worldを継承する新規integration module。既存event ID/rankの
  `replication_cpu`だけを追加置換し、A4.6b1/b2、A4.8c1/c2、promoted A3を変更しない。
- claim前にlive PCG64 before-stateのcloneから
  `prepare_completion_mutation_rng_tape`を生成し、binding-aware resident tapeと
  `paid_replication_completion_mutation_plan`をTorch CPU/CUDA fp64で作る。明示readbackをcommit
  authorityとし、独立NumPy planはdiscrete exact、通常float64最大`2e-12`のoracleに限定する。
  派生`genome_lesion_mean_after`とfresh `genome_lesion_mean`だけは、final divisionで測定済みの
  非負finite最大1 ULPを許し、2 ULP以上をclaim前に拒否する。
- RNG順はcell event内でappend各symbolのscalar random、hit時scalar bounded integer、直後に
  insertion、deletion、conditional gene duplication、inversion、transposition、MIN padding、
  MAX trim、material-budget tail trimを凍結CPU高水準callどおり逐次再生する。完全PCG64
  `state/inc/has_uint32/uinteger`を保持する。
- commit直前にcurrent live bindingと同じbefore-stateからtapeを新規再生成し、retained host/resident
  tape、schedule、after-stateをexact照合する。caller supplied tape/plan/candidateは受け取らない。
- resident final polymer/lesionを元genomes/lesions outerへappendし、元pools arrayのNUCLEOTIDE/ATP、
  substitutionと5 structural attempted counters、template/template lesion/copy/fractional reset、cycle、
  proof/last/error telemetry、final polymerによるgene cache、条件付きnovel-path ageを一括publishする。
  material trim後もattempted event counterを補正しない。最後に同じlive Generatorへattested
  after-stateを一度だけ適用する。
- padding後にmaterial budgetでfinal lengthがMIN未満へ戻る凍結挙動を許容する。材料trimをarena
  overflowのclipに流用せず、Q/S/W/P不足はclaim前にfail closedとする。lesionはtemplate lesionと
  proof fractionから正本演算順で直接再計算し、old+new lesion meanはcombined prefixを一回reduceする。
- mutation offのactive completionはA4.8c2、mutation off/onのactive noncompletionとactive zero-workは
  A4.8c2経由でA4.8c1へ明示delegateする。例外catchによるbranch分類は禁止する。
- preclaim failureはreceipt/biology/RNGを完全不変にする。claim後publish例外は、このbridgeのwrite-set
  であるpools、genome/lesion outerと既存array、template/copy、mutation ledger、gene cache outer+nested、
  cycle/telemetry/novel age、world energy、live RNG object/stateを元identity・順序・値へrollbackし、
  outer schedulerへaborted receiptを残す。
- scheduler/worldのA4.8c3 type/schema/config/deviceをsave/load/cloneで保持し、c1/c2/c3のactive/pending
  serializationを拒否する。candidate/binding/tape/planは保存しない。`full_gpu_world_step=false`を維持する。

## A4.8c3で実装しないもの

- inactive template start、startと同一callの伸長・completion、disabled/no-genome/replicase-gate/
  negative ATP/dead等のpre-active-gate early-return authority。これらはno fallbackでscope外とする。
- caller supplied tape/plan/candidate、multi-cell replication batch、device RNG、persistent arena/cache、
  generic transaction/RNG framework、A4.6b1/b2またはA4.8c1/c2 coreの変更。
- bridge write-set外の任意whole-cell mutationを復元する汎用transaction claim。
- fp32、compile/graph/Triton/custom CUDA、performance、speedup、A4完成/昇格、A5/A6。

## A4.8c3固定テスト

1. A4.6b1の3-row fixed fixtureをcell event順にdirect Formal066へ照合する。substitution、5 structural
   counters、final polymer 577/638/32、material charge/refund/padding trim、pools、lesion/topology/cache/
   novel-path、完全PCG64 before/afterを固定する。
2. 明示Torch CPU/CUDA candidate/commit、host tape、independent NumPy oracle、resident binding/tape/final
   plan、device/pointer/version/content、prepare purity、fresh binding/cache、same Generator after-state、
   one-shotを照合する。
3. exact Q/S/W/Pと各one-short、wrong cell/dt/source/config/device/RNG、host tape/oracle、resident
   ragged/state/cache/tape/final planのpublic/private/whole-member/`.data`、candidate/fresh binding/alias
   tamper、duplicate/out-of-order、claim前失敗、注入publish失敗のwrite-set rollbackをfail closedで
   固定する。inactive/start/pre-active early returnは旧CPU bridgeへfallbackせずno claimとする。
4. c1 mutation off/on noncompletion、requested-zero、c2 mutation-free completionの明示delegate、c3
   completion後のsurface/hydrolysis/motion RNG順、旧CPU replication未呼出し、save/load/clone、active/
   pending拒否を固定する。A4.1〜A4.8c2の62 testsを変更せず、既存62 + 新規最大4 = 合計最大66
   testsをCUDA必須でPASSさせる。

## A4.8c3判定

- 66/66とA3 regression、3-row direct CPU semantics、resident/NumPy association、完全PCG64、material/
  lesion/topology ledger、atomic rollback、delegate、save/clone、promoted A3/A4.8c1/c2 byte不変がPASSした
  場合だけ「A4.8c3 mutation-enabled pre-existing-active completion atomic commit bridge」と記録する。
- fresh tape/after-stateまたはresident-vs-NumPy discreteが一致しない、host値commit、capacity clip、
  inactive/start/early-returnへのscope拡大、A4.8c1/c2編集、CPU fallbackが必要ならA4.8c2を維持してSTOPする。
  tolerance拡大や独自RNGで通さない。
- A4.8c3でもA4は未完である。次はinactive template startとpre-active early-return authorityを別sliceで
  閉じ、`full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c4追加仮説

Rule Lock receipt `20260815T085714Z` の範囲は、aliveな一cell、`mutation=true`、
entry時はreplication inactive、exactly oneかつnonemptyのcomplete genomeを持ち、
A4.5b resident planが`template_start_events=true`かつ`completion_events=false`と判定する
template-start/noncompletionの原子的commitだけとする。同じlive PCG64 before-stateから
selection callの後にappend symbolごとのthreshold/integer callを凍結CPU順で再生し、
resident readbackとfresh tapeが一致すれば、CPU replication fallbackなしでtemplate開始と
そのcallの有料伸長を一括publishできるはずである。

## A4.8c4で実装するもの

- A4.8c3 scheduler/worldを継承する新規integration module。既存event ID/rankの
  `replication_cpu`だけをこの開始分岐に追加置換し、A4.5b pure core、A4.8c1/c2/c3、
  promoted A3を変更しない。
- claim前にlive NumPy PCG64 Generatorのbefore-state cloneから
  `prepare_substitution_rng_tape`を新規生成し、binding-awareの
  `paid_replication_substitution_plan`をTorch CPU/CUDA fp64で実行する。resident planの
  明示readbackだけをcommit authorityとし、NumPy planはdiscrete exact、float64最大
  `2e-12`の独立oracleに限定する。
- RNG順をtemplate selection `integers(0, 1)`→appendごとのscalar `random()`→hit時の
  scalar `integers(0, 7)`に固定し、PCG64の`state/inc/has_uint32/uinteger`と同じ
  Generator object identityを保持する。claim直前にcurrent live binding/before-stateからfresh
  tapeを作り直し、retained host/resident tape、schedule、after-stateとexact照合する。
- entryのtemplateが`None`でcopyがemptyであること、exactly one nonempty complete genome、
  resident planの`scope_valid=true`、`template_start_events=true`、selected index `0`、
  `template_storage_symbols=template length`、`completion_events=false`を必須とする。
  `dt==0`、requested-zero、nucleotide/ATP resource-stopでappendが0でも、template開始自体を
  成功workとしてcommitする。
- capacityをこの開始で増えるsequence `Q + 2`、symbol `S + template length + append`に
  対しclaim前に検査する。source/final sequenceを支えるmax-sequence width `W`と
  無変更のgene/protein rowsを支えるprotein capacity `P`も同じbinding trustに含める。不足を
  clipせず、Q/S/W/Pのexact capacityだけを通す。genome/lesion countとderived gene cacheは
  開始時に変更しない。
- 元pools arrayのNUCLEOTIDE/ATP、元mutation ledgerのsubstitution値、元telemetryと
  fractional carry、元RNG objectのafter-stateをselective publishする。新しいtemplateはselected
  complete genomeとaliasしないNumPy copyとし、新しいreplication copy listにappend symbolsを
  保持する。template lesionはselected genome lesion、対応lesionが無い凍結CPUの
  fallbackは`0.0`とする。
- preclaim failureはreceipt/biology/RNGを完全不変にする。claim後publish例外は、開始時に
  entry templateが`None`だったことも含め、pools、template/copyのouter identityと値、fractional/
  telemetry、mutation ledger、live RNG object/state、world energyを元のidentity・順序・値へ
  rollbackし、outer schedulerにaborted receiptを残す。
- mutation trueのactive sourceはc3からc2/c1へ明示delegateする。scheduler/worldのA4.8c4
  type/schema/config/deviceをstep境界のsave/load/cloneで保持し、c1/c2/c3/c4のactive/pending
  serializationを拒否する。candidate/binding/tape/planは保存しない。

## A4.8c4で実装しないもの

- `mutation=false`のinactive start、mutation off/onのstartと同一callのcompletion、disabled/
  no-genome/replicase-gate/negative ATP/dead/inactive two-or-more genomesといったpre-active
  ordinary early no-op authority。これらはclaimせずSTOPし、凍結CPUへfallbackしない。
- caller-supplied tape/plan/candidate、multi-cell replication batch、device RNG、persistent arena/cache、
  generic transaction/RNG framework、A4.5bまたはA4.8c1/c2/c3 coreの変更。
- bridge write-set外のwhole-cell transaction claim、fp32、compile/graph/Triton/custom CUDA、
  performance/speedup、A4完成/昇格、A5/A6。`full_gpu_world_step=false`を維持する。

## A4.8c4固定テスト

1. mutation true、inactive、one nonempty genomeのselection→append RNGとdirect Formal066を照合し、
   template/copyのnonalias、lesion/fallback-zero、pools/fractional/telemetry/substitution、完全
   PCG64 before/after、`Q+2`/`S+template+append`を固定する。`dt=0`、requested=0、
   resource-stopもtemplate-start成功かつ`work_performed=true`とする。
2. NumPy/Torch CPU/明示CUDA candidate/commit、resident readback authority、fresh tape/after-state、
   resident device/pointer/version/content、fresh binding/cache、same Generator object、one-shotを照合する。
3. exact Q/S/W/P capacityと各one-short、wrong cell/dt/source/config/device/RNG、host/resident tape/plan/
   candidate/fresh binding tamper、duplicate/out-of-order、claim前失敗、注入publish失敗の
   `None` template復元を含むwrite-set rollbackをfail closedで固定する。
4. active c3→c2→c1 delegate、scope外inactive分岐のno CPU fallback、開始後の後続event、
   save/load/clone、active/pending拒否、既存66 tests無変更を固定する。既存66 + 新規最大4 =
   合計最大70 testsをCUDA必須でPASSさせる。

## A4.8c4判定

- 最大70/70とA3 regression、direct CPU semantics、resident/NumPy association、完全PCG64、
  topology/material/lesion ledger、atomic rollback、delegate、save/clone、promoted A3/A4.5b/A4.8c1/c2/c3
  byte不変がPASSした場合だけ「A4.8c4 mutation-enabled inactive-template-start
  noncompletion atomic commit bridge」と記録する。
- resident planがstart/noncompletionでない、fresh tape/after-stateが一致しない、host値commit、
  capacity clip、scope外分岐への拡大、A4.5b/c1/c2/c3編集、CPU fallbackが必要なら
  A4.8c3を維持してSTOPする。tolerance拡大や独自RNGで通さない。
- A4.8c4でもA4は未完である。mutation-free inactive start、same-call completion、pre-active
  ordinary early-return authorityを後続sliceで閉じ、`full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c5追加仮説

Rule Lock receipt `20260815T102600Z` の範囲は、aliveな一cell、`mutation=false`、entry時は
replication inactive、exactly oneかつnonemptyのcomplete genomeを持ち、既存A4.5b内部の
deterministic NumPy/Torch fp64 kernelが`template_start_events=true`かつ
`completion_events=false`と判定するtemplate-start/noncompletionの原子的commitだけとする。
凍結Formal066がreplicase gate後に実行する高水準`integers(0, 1)`を省略せず、selection-onlyの
private PCG64 evidenceとして完全before/after stateを再生すれば、mutation用threshold/integer drawを
誤って消費せず、CPU replication fallbackなしで開始とそのcallの有料伸長を一括publishできるはずである。

## A4.8c5で実装するもの

- A4.8c4 scheduler/worldを継承する新規integration module
  `SOMA_CELL_0_6_8_gpu_a4_replication_start_mutation_free_integration.py`を追加する。
  既存event ID/rankの`replication_cpu`をこの分岐に追加置換し、A4 pure core、A4.8c1/c2/c3/c4、
  promoted A3、既存resultsは変更しない。canonical validator/contract/schema/manifestsは、c5の
  evidence、API、source hash、最大74 testsの登録・検証に必要な範囲だけを更新する。
- integration-local private dispatcherだけが既存
  `_paid_replication_elongation_numpy/torch(..., allow_template_start=True)`を呼ぶ。resident planの明示
  readbackをcommit authorityとし、独立NumPy planはdiscrete exact、float64最大`2e-12`のoracleとする。
  public mode flagや新しいpure tape/APIを追加しない。
- source binding identity、cell ID、exact `dt_hex`、mutation-free replication config digest、call bounds
  `0/1`、selected index `0`、call count `1`、schedule digest、完全なPCG64 before/afterを持つprivate-factory
  selection evidenceをevent-localに生成する。clone Generatorで実際に`integers(0, 1)`を一回呼び、
  `state/inc/has_uint32/uinteger`をexact replayする。現NumPyでbefore==afterでも直接代入で省略しない。
- entry template `None`、empty copy、exactly one nonempty complete genome、planのscope-valid/start/index-0/
  template-storage/noncompletion、append lengthがtemplate未満であることを必須とする。`dt==0`、
  requested-zero、nucleotide/ATP resource-stopでappendが0でもselectionとtemplate startを成功workとする。
- capacityは`Q + 2`、`S + template length + paid append`をclaim前に検査し、source/final row幅`W`と
  gene/protein capacity `P`を同じbinding trustに含める。exact Q/S/W/Pだけを通し、one-shortをclipしない。
  genome/lesion countとgene cacheは不変、`genome_material_symbols`はpaid appendだけ増やす。
- 元pools arrayのNUCLEOTIDE/ATP、new non-alias template ndarray、new replication-copy list、template lesion
  またはmissing-lesion fallback `0.0`、fractional/last-symbol/effective-error/proofreading telemetry、および同じ
  live Generator objectのevidence after-stateだけをselective publishする。mutation ledgerのobject、挿入順、
  keys、valuesは完全不変とし、world dissipated energyも変更しない。
- claim直前にlive binding/RNGからNumPy planとselection evidenceを再生成し、retained host/resident plan、
  source/fresh binding、schedule、after-stateを再照合する。preclaim failureはreceipt/biology/RNGを完全不変にし、
  claim後publish例外はpools、entryの`None` template、元empty copy-list object/value、telemetry、mutation ledger、
  RNG object/state、world energyを元identity・順序・値へrollbackしてaborted receiptを残す。
- active sourceはA4.8c4からA4.8c3/c2/c1へ、inactive `mutation=true`はA4.8c4へ明示delegateする。
  scheduler/worldのA4.8c5 type/schema/config/deviceをsave/load/cloneで保持し、c1〜c5のactive commitと
  pending stepのserializationを拒否する。candidate/binding/evidence/planは保存しない。

## A4.8c5で実装しないもの

- mutation off/onのinactive startと同一callのcompletion、disabled/no-genome/replicase-gate/negative ATP/
  dead/inactive two-or-more genomes等のpre-active ordinary early no-op authority。claimせずCPU fallbackなしで
  scope外とする。
- caller-supplied evidence/plan/candidate、multi-cell batch、device RNG、persistent arena/cache、generic
  transaction/RNG framework、pure coreまたはA4.8c1/c2/c3/c4の変更。
- fp32、compile/graph/Triton/custom CUDA、performance/speedup、A4完成/昇格、A5/A6。
  `full_gpu_world_step=false`を維持する。

## A4.8c5固定テスト

1. mutation false inactive one-genomeのdirect Formal066 oracleをnormal/missing-lesion/`dt=0`/
   requested-zero/resource-stop、primed/unprimed PCG64で照合し、実selection call、new template/copy nonalias、
   pools/telemetry、mutation-ledger不変、完全PCG64、`Q+2`/`S+T+A`、zero append work=trueを固定する。
2. NumPy/Torch CPU/明示CUDA candidate/commit、resident readback authority、fresh selection evidence/after-state、
   resident device/pointer/version/content、fresh binding/cache、same Generator object、one-shotを照合する。
3. exact Q/S/W/Pと各one-short、wrong cell/dt/source/config/device/RNG、host evidence/oracle、resident plan/
   binding、candidate/fresh binding tamper、excluded scopes、duplicate/out-of-order、preclaim不変、注入publish失敗の
   `None` templateと元copy-list identityを含むrollbackをfail closedで固定する。
4. inactive mutation trueのc4、active c3/c2/c1 delegate、旧CPU fallback不使用、開始後successor event order、
   save/load/clone、active/pending拒否、API/private非公開、pure/c1〜c4 hash、既存70 tests無変更を固定する。
   既存70 + 新規最大4 = 合計最大74 testsをCUDA必須でPASSさせる。

## A4.8c5 GO/STOP判定

- fresh Rule Lockに結び付く実装開始はGOとする。最大74/74、A3 regression、direct CPU semantics、resident/
  NumPy association、実高水準selection call、完全PCG64、Q/S/W/P、material/topology/lesion/cache、atomic rollback、
  delegate、save/clone、promoted A3/A4 pure/A4.8c1〜c4 byte不変が全てPASSした場合だけ「A4.8c5
  mutation-free inactive-template-start noncompletion atomic commit bridge」と記録する。
- private deterministic start kernel、fresh evidence、resident-vs-NumPy discrete、direct CPU stateのいずれかが
  一致しない、selection call省略、余分なmutation RNG draw、host plan commit、capacity clip、scope外拡大、
  pure/c1〜c4編集、CPU fallbackが必要ならA4.8c4を正式authorityとして維持してSTOPする。
- A4.8c5でもA4は未完である。startと同一callのcompletionおよびpre-active ordinary early-return authorityを
  後続sliceで閉じ、`full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c6追加仮説

Rule Lock receipt `20260815T115505Z` の範囲は、aliveな一cell、`mutation=false`、entry時は
replication inactive、exactly oneかつnonemptyのcomplete genomeを持ち、template選択後の同一callで
複製copyがtemplate長へ到達するstart/completionの原子的commitだけとする。元inactive sourceから
event-localに、selected genomeのnon-alias template copy、対応lesion（欠損時`0.0`）、empty copy、
fractional `0.0`を持つsynthetic active-start bindingを構築し、そのbindingに既存public
`paid_replication_completion_plan`を適用すれば、pure coreやc1〜c5を変更せず、凍結Formal066の
selection後のmutation-free completionをresident authorityで再現できるはずである。

## A4.8c6で実装するもの

- A4.8c5 scheduler/worldを継承する新規integration module
  `SOMA_CELL_0_6_8_gpu_a4_replication_start_completion_integration.py`を追加する。既存event ID/rankの
  `replication_cpu`をこの一分岐だけ追加置換し、A4 pure、A4.8c1〜c5、promoted A3、既存resultsは変更しない。
- 元bindingに対しA4.8c5と同じprivate selection evidenceを生成し、clone PCG64 Generatorで高水準
  `integers(0, 1)`を実際に一回呼ぶ。selectionの完全before/after state、source binding identity、cell ID、
  exact dt/config digest、selected index `0`、call count `1`を固定し、mutation用RNG drawは一切行わない。
- 元inactive sourceからevent-local candidateを作り、selected genomeをnon-alias template、template lesionを
  対応lesionまたは`0.0`、copyをempty、fractionalを`0.0`としてpackしたsynthetic active-start bindingを
  作る。このsynthetic bindingをsource identityとは混同せず、元source→synthetic sourceの完全な変換証跡を
  host/resident artifactとして保持する。
- synthetic bindingにpublic `paid_replication_completion_plan`をNumPy fp64 oracleおよびTorch CPU/CUDA
  resident authorityとして適用する。scope-valid、completion true、template-start false、appendがtemplate長、
  completed polymerがselected genomeとexact一致、cycle delta `1`、topology delta `-1`、mutation/substitution `0`
  を必須とし、resident readbackだけをcommit authority、NumPyはdiscrete exactかつfloat64最大`2e-12`のoracleとする。
- capacityは元inactive sourceから最終completed sourceへの`Q + 1`、`S + template length`、row幅`W`、
  gene/protein capacity `P`をclaim前に検査する。synthetic intermediateの`Q + 2`、`S + template length`も同じ
  fixed capacity内に収まることを必須とし、Q/S/W/P不足をclipしない。
- resident planから元pools arrayのNUCLEOTIDE/ATP、新しいnon-alias completed genome、元lesion listへのnew lesion、
  replication cycle、template/copy/fractional reset、last-symbol/effective-error/proofreading telemetry、derived gene
  cache、条件付きnovel-path ageを一括publishする。同じlive Generator objectへselection after-stateだけを適用し、
  mutation ledgerのidentity/order/keys/valuesとworld dissipated energyは変更しない。
- claim直前にlive元binding/RNGからselection evidence、synthetic binding、NumPy planを全て再生成し、retained
  host/resident source、plan、schedule、after-state、fresh final bindingとexact照合する。preclaim failureはreceipt/
  biology/RNGを完全不変にし、claim後publish例外はpools、genome/lesion outerと既存array、entryの`None` template、
  元empty copy-list object、cycle/telemetry/gene cache/novel age、mutation ledger、RNG object/state、world energyを
  元identity・順序・値へrollbackしてaborted receiptを残す。
- active sourceおよびinactive `mutation=true`、mutation-free start/noncompletionはA4.8c5からc4/c3/c2/c1へ
  明示delegateする。scheduler/worldのA4.8c6 type/schema/config/deviceをsave/load/cloneで保持し、c1〜c6のactive
  commitとpending stepのserializationを拒否する。candidate/binding/evidence/planは保存しない。

## A4.8c6で実装しないもの

- mutation-enabled inactive start/completion、disabled/no-genome/replicase-gate/negative ATP/dead/inactive
  two-or-more genomes等のpre-active ordinary early no-op authority。claimせずCPU fallbackなしでscope外とする。
- caller-supplied evidence/plan/candidate、multi-cell batch、device RNG、persistent arena/cache、generic transaction/
  RNG framework、pure coreまたはA4.8c1〜c5の変更。
- fp32、compile/graph/Triton/custom CUDA、performance/speedup、A4完成/昇格、A5/A6。
  `full_gpu_world_step=false`を維持する。

## A4.8c6固定テスト

1. mutation false inactive one-genome start/completionをdirect Formal066へnormal/missing-lesion、primed/unprimed
   PCG64で照合し、実selection call exactly one、mutation RNGなし、completed genome/lesion/pools/cycle/telemetry/
   cache/novel age、mutation ledger不変、完全PCG64、元→synthetic→final topology/materialを固定する。
2. synthetic active-start bindingとpublic completion planのNumPy/Torch CPU/明示CUDA candidate/commit、resident
   readback authority、fresh evidence/plan、resident device/pointer/version/content、fresh binding、same Generator object、
   one-shotを照合する。
3. exact Q/S/W/Pと各one-short、wrong cell/dt/source/config/device/RNG、source/synthetic/final binding、host oracle、
   resident plan/candidate tamper、duplicate/out-of-order、preclaim不変、注入publish失敗の`None` templateと元copy-list/
   genome/lesion/gene-cache identityを含むrollbackをfail closedで固定する。
4. active、inactive mutation true、mutation-free start/noncompletionのc5以下delegate、旧CPU fallback不使用、後続event
   order、save/load/clone、active/pending拒否、pure/c1〜c5 hash、既存74 tests無変更を固定する。既存74 + 新規最大4 =
   合計最大78 testsをCUDA必須でPASSさせる。

## A4.8c6 GO/STOP判定

- fresh Rule Lockに結び付く実装開始はGOとする。最大78/78、A3 regression、direct CPU semantics、
  元→synthetic→resident/final association、実高水準selection call、完全PCG64、Q/S/W/P、material/topology/lesion/
  cache、atomic rollback、delegate、save/clone、promoted A3/A4 pure/A4.8c1〜c5 byte不変が全てPASSした場合だけ
  「A4.8c6 mutation-free inactive-template-start same-call-completion atomic commit bridge」と記録する。
- synthetic bindingまたはfresh evidence/planが一致しない、resident-vs-NumPy discrete/direct CPU stateが一致しない、
  selection call省略、余分なmutation RNG draw、host plan commit、capacity clip、scope外拡大、pure/c1〜c5編集、
  CPU fallbackが必要ならA4.8c5を正式authorityとして維持してSTOPする。tolerance拡大や独自RNGで通さない。
- A4.8c6でもA4は未完である。mutation-enabled start/completionとpre-active ordinary early-return authorityを
  後続sliceで閉じ、`full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c7追加仮説

Rule Lock receipt `20260815T133443Z` の範囲は、aliveな一cell、`mutation=true`、entry時は
replication inactive、exactly oneかつnonemptyのcomplete genomeを持ち、template選択後の同一callで
複製copyがtemplate長へ到達し、その後のstructural mutationまで完了するstart/completionの原子的commitだけとする。
元sourceで高水準`integers(0, 1)`を一回だけ再生し、そのafter-stateをA4.8c6と同じevent-local synthetic
active-start bindingに対する既存A4.6b1 completion-mutation tapeのbefore-stateへexactに接続すれば、selection、
append substitution、structural threshold/edit、padding、MAX長trim、nucleotide-budget trimの凍結PCG64順序を
二重消費せず、A4.6b2 resident final planをcommit authorityとして再現できるはずである。

## A4.8c7で実装するもの

- A4.8c6 scheduler/worldを継承する新規integration module
  `SOMA_CELL_0_6_8_gpu_a4_replication_start_completion_mutation_integration.py`を追加する。public surfaceは
  `A4ReplicationStartCompletionMutationCommitError`、
  `A4ReplicationStartCompletionMutationEventScheduler`、
  `Hybrid066WorldA4ReplicationStartCompletionMutation`とbuild/schema/status constantsだけとする。既存event
  ID/rankの`replication_cpu`をこの一分岐だけ追加置換し、A4 pure、A4.8c1〜c6、promoted A3、既存resultsを
  変更しない。
- dispatchだけは元`config`を変更せずdeep-copyした`mutation=false` deterministic configとsynthetic
  active-start resident bindingへpublic `paid_replication_completion_plan`を適用し、resident tensorのscope codeで
  completion/noncompletionを分類する。A4.5b start tapeはsame-call completionをscope code 3で拒否し、A4.6b1
  tapeと直列に置くとappend RNGを二重消費するため、dispatchにもcommit chainにも使わない。例外をbranch分類に
  使用しない。
- 元inactive bindingに結び付くc7専用private selection evidenceを作り、clone PCG64 Generatorで高水準
  `integers(0, 1)`を実際に一回呼ぶ。source/synthetic binding identity、cell ID、exact dt、mutation-enabled
  completion config digest、low/high `0/1`、selected index `0`、call count `1`、完全before/after state、schedule/
  chain digestを保持する。現在before==afterでも直接代入でcallを省略しない。
- selected genomeのnon-alias template copy、対応lesionまたは欠損時`0.0`、empty copy、fractional `0.0`を持つ
  A4.8c6と同じevent-local synthetic active-start cell/bindingを構築する。selection evidence after-stateを
  `prepare_completion_mutation_rng_tape`のbefore-stateへexactに渡し、tape after-stateを最終live RNG state候補とする。
  selection以外のRNGは全てA4.6b1 tapeだけが一回ずつ記録する。
- synthetic NumPy binding上のA4.6b1 tapeとA4.6b2 NumPy final planを独立oracle、同じtapeを明示H2Dした
  synthetic Torch CPU/CUDA binding上のA4.6b2 plan readbackをcommit authorityとする。append threshold/replacement、
  insertion/deletion/duplication/inversion/transposition、padding symbols、MAX長trim、nucleotide-budget trim、final
  polymer、event counters、material delta、lesion、topologyをdiscrete exact、float64最大`2e-12`で照合する。
- 元genome長を`T`、structural/budget処理後のfinal genome長を`L=T+delta`とし、capacityはexact
  `C=1`、synthetic transient `Q=3`、`S=max(2T, T+L)=2T+max(delta,0)`、`W=max(T,L)`、
  `P=max(source, synthetic, finalのactive/damaged proteinとdecoded gene-cache entry requirement)`をclaim前に
  検査する。各one-shortをclipせず拒否する。net topologyは元sourceからsequence `+1`、symbol `+L`、complete
  genome `+1`、lesion entry `+1`、`genome_material_symbols +L`とする。
- resident final planから元pools arrayのNUCLEOTIDE/ATP、新しいnon-alias final genome、元lesion listへのnew lesion、
  replication cycle、template/copy/fractional reset、last-symbol/effective-error/proofreading telemetry、final genomeから
  再decodeしたgene cache、条件付きnovel-path ageを一括publishする。NUCLEOTIDEはappend `T`の支払後にstructural
  `delta`を追加支払または返金し、net新genome `L`分だけ減る。ATPはpaid appendだけに従い、無料生成しない。
- 元mutation ledger object/order/既存keysを保持し、resident planのsubstitution countと5種structural event countだけを
  対応keyへ加算する。未知keyを削除・並べ替えず、world dissipated energyとprotein/damaged stateを変更しない。同じ
  live Generator objectへtape after-stateだけを適用する。
- claim直前にlive元binding/config/dt/RNGからselection evidence、synthetic binding、A4.6b1 tape、NumPy planを全て
  再生成し、retained host/resident tape/planのfactory、digest、device、pointer、version、content、fresh final bindingを
  exact照合する。preclaim failureはreceipt/biology/RNGを完全不変とし、claim後publish例外はpools、genome/lesion
  outerと既存array、entryの`None` template、元empty copy-list object、cycle/telemetry/gene cache/novel age、mutation
  ledger、RNG object/state、world energyを元identity・順序・値へrollbackしてaborted receiptを残す。
- active sourceはA4.8c6以下へ、inactive `mutation=false`はA4.8c6へ、inactive `mutation=true`でcompletion probeが
  noncompletionならA4.8c6からA4.8c5/A4.8c4へ明示delegateする。scheduler/worldのA4.8c7 type/schema/config/deviceを
  save/load/cloneで保持し、c1〜c7のactive commitとpending stepのserializationを拒否する。candidate/binding/
  evidence/tape/planは保存しない。

## A4.8c7で実装しないもの

- disabled/no-genome/empty-genome/replicase-gate/negative ATP/dead/inactive two-or-more genomes等のpre-active ordinary
  early no-op authority。claimせずCPU fallbackなしでscope外とする。
- A4.5b substitution tapeとの二重append replay、caller-supplied evidence/tape/plan/candidate、multi-cell batch、device
  RNG、persistent arena/cache、generic transaction/RNG framework、pure coreまたはA4.8c1〜c6の変更。
- fp32、compile/graph/Triton/custom CUDA、performance/speedup、A4完成/昇格、A5/A6。
  `full_gpu_world_step=false`を維持する。

## A4.8c7固定テスト

1. mutation true inactive one-genome start/completionをdirect Formal066へdeletion/refund、expansion、missing-lesion、
   primed/unprimed、short padding/budget-trim fixtureで照合し、selection exactly oneからtape afterまでの完全PCG64、
   final polymer/lesion/pools/counters/material/cache/novel ageを固定する。
2. 元→synthetic→A4.6b1 tape→A4.6b2 NumPy/Torch CPU/明示CUDA plan→fresh final bindingのcandidate/commit、
   resident readback authority、tape device/pointer/version/content、one-shotを照合する。
3. exact Q/S/W/Pと各one-short、wrong cell/dt/source/config/device/RNG、selection/tape chain、source/synthetic/final
   binding、host oracle、resident tape/plan/candidate tamper、duplicate/out-of-order、preclaim不変、注入publish失敗の
   `None` templateと元copy-list/genome/lesion/gene-cache/ledger identityを含むrollbackをfail closedで固定する。
4. active、inactive mutation false、mutation-enabled start/noncompletionのc6以下delegate、旧CPU fallback不使用、後続event
   order、save/load/clone、active/pending拒否、pure/c1〜c6 hash、既存78 tests無変更を固定する。既存78 + 新規最大4 =
   合計最大82 testsをCUDA必須でPASSさせる。

## A4.8c7 GO/STOP判定

- fresh Rule Lockに結び付く実装開始はGOとする。最大82/82、A3 regression、direct CPU semantics、selection after→
  A4.6b1 before→A4.6b2 afterの完全PCG64 chain、resident/NumPy association、Q/S/W/P、material/topology/lesion/
  cache/ledger、atomic rollback、delegate、save/clone、promoted A3/A4 pure/A4.8c1〜c6 byte不変が全てPASSした場合だけ
  「A4.8c7 mutation-enabled inactive-template-start same-call-completion atomic commit bridge」と記録する。
- A4.5bとA4.6b1によるappend draw二重消費、selection call省略、tape chain不一致、synthetic/final bindingまたは
  resident-vs-NumPy/direct CPU state不一致、host plan commit、free material、capacity clip、tolerance拡大、scope外拡大、
  pure/c1〜c6編集、CPU fallbackが必要ならA4.8c6を正式authorityとして維持してSTOPする。
- A4.8c7でもA4は未完である。pre-active ordinary early-return authorityを後続sliceで閉じ、
  `full_gpu_world_step=false`、速度向上なしを維持する。

## A4.8c8追加仮説

Rule Lock receipt `20260815T152726Z` の範囲は、凍結Formal066が0.4親実装へ委譲する
`_replicate_genome`のうち、実際に`last_replication_symbols = 0`以外のbiology、material、telemetry、
RNGを変更しない順序付きearly returnだけとする。CPU正本の順序は、(1) `genome_replication=false`、
(2) complete genomeが0本、(3) replicase `<= 1e-6`、(4) template inactiveかつcomplete genomeが2本以上、
である。これらを新しいresident NumPy/Torch planで明示分類し、A4.8c7の上でexactly-once commitすれば、
旧CPU replicationへfallbackせず通常no-op authorityを閉じられるはずである。

空のcomplete genomeは独立no-opではない。replicase gate通過時、mutation falseでは空genomeをもう1本追加して
cycleを進め、mutation trueではminimum paddingとmaterial移動を伴う。負ATPのone-genome sourceはtemplate選択後に
負のfractional progressとeffective-error telemetryを作り、dead cellへのdirect callもreplication workを行い得る。
したがってempty genome、entry ATP `< 0`、deadはA4.8c8成功scopeへ混ぜない。

## A4.8c8で実装するもの

- 既存A4 pure coreとA4.8c1〜c7をbyte不変に保ち、新規module
  `SOMA_CELL_0_6_8_gpu_a4_replication_early_noop_integration.py`へpure descriptorとintegrationを同居させる。
  public surfaceは`A4ReplicationEarlyNoopPlan`、`paid_replication_early_noop_numpy`、
  `paid_replication_early_noop_torch`、`A4ReplicationEarlyNoopCommitError`、
  `A4ReplicationEarlyNoopEventScheduler`、`Hybrid066WorldA4ReplicationEarlyNoop`、build/schema/status constantsとする。
- pure planはfixed-capacity binding、exact `dt`、strict boolean configに結び付き、`cell_ids/cell_mask`、
  `scope_valid/scope_error_code`、順序付き`branch_code`、resident fp64 `replicase_activity`、
  `last_replication_symbols_after=0`、`rng_call_count=0`を返す。NumPyは凍結dict挿入順、Torchはresident
  cache/stateから同じordered replicase和、proteostasis、genome-lesion factor、external replicaseを計算する。
  CPU/CUDA fp64比較guard上のreplicaseは成功へ丸めずscope外とする。
- 成功branchは優先順に`disabled`、`no-genome`、`replicase-gate`、`inactive-multi-genome`の4種だけとする。
  後ろの条件が前の条件と同時に成立しても、CPU正本と同じ先行branch codeだけを記録する。
  empty complete genome、negative ATP、非finite/malformed topology/cache/config、capacity不足はplan成功前に
  fail closedとする。mutation flag、active/inactive topologyはCPUが当該early returnより後で初めて使用する範囲では
  no-opの意味論を変えないが、normal replication workへ到達するrowは`not-early-noop`としてc7以下へdelegateする。
- integrationはA4.8c7 scheduler/worldを継承し、`prepare/ready/revalidate/commit/publish`のprivate one-shot seamを持つ。
  live cell、world/config identity、generation、exact `dt.hex()`、canonical host binding/gene cache、A4 capacity/device、
  resident pointer/version/content、NumPy oracle、live PCG64 object/state、world energy、source object identity/stateを
  claim直前に再attestする。caller supplied plan/candidateをauthorityにしない。
- 検証後だけ既存`replication_cpu`を1回claimし、元cellの`last_replication_symbols`を整数`0`へ更新する。
  genomes/lesions/template/copy/fractional/pools/proteins/damaged proteins/gene cache/mutation ledger/cycle/
  effective-error/proofreading/novel-path/world energyとlive RNG object/stateは完全不変とする。receiptは
  `amount=0`、`work_performed=false`、`rng_call_count=0`、branch、device、source provenanceを記録する。
  preclaimではcanonical cell state全体を再attestする一方、claim後rollbackはgeneric whole-cell transactionを
  主張しない。実publisherのwrite-setである元last-symbol値と、明示snapshotしたpools、genome/lesion、
  template/copy、mutation ledger、proteins/damaged proteins、gene cache、replication telemetry、age/alive、
  world energy、RNGのidentity・順序・値だけを復元し、outer receiptをabortedにする。未列挙fieldを変更する
  悪性publisher overrideの汎用rollbackはscope外とする。
- `cpu_replication`はduplicate/order/nested guardを先に行い、c8 planが4成功branchのどれかならc8 commit、
  `not-early-noop`ならA4.8c7へ明示delegateする。scope error、empty/negative/dead、resident/NumPy不一致を
  c7や凍結CPUへfallbackして成功扱いにしない。save/load/cloneはc8 scheduler/world type/schema/config/deviceを保持し、
  c1〜c8 active commitとpending stepのserializationを拒否する。
- CPU正本はdisabled/no-genomeをgene-cache validationより前にreturnするが、A3正式world pathはreplication直前に
  `gene_refresh`を完了する。A4.8c8はcanonical bindingを要求し、stale/corrupt cacheをCPU短絡に合わせて受理せず
  preclaimでfail closedとする。この狭まりは有効world semanticsの変更ではなくintegration trust boundaryである。

## A4.8c8で実装しないもの

- empty-genome completion/padding、negative-ATP start、dead direct replication、replicase fp64 comparison boundary、
  normal start/noncompletion/completion/mutation workの新規置換。normal workは既存c1〜c7 authorityだけへdelegateする。
- 既存`SOMA_CELL_0_6_8_gpu_a4_replication.py`、A4.8c1〜c7、promoted A3、validator、contract/schema、results、
  manifestsの同時変更。事前登録と新moduleの実装後、検証器・文書・manifest更新は別の監査段階で行う。
- caller-supplied plan/candidate、CPU fallback、persistent arena/cache、multi-cell live commit、device RNG、generic transaction
  framework、fp32、compile/graph/Triton/custom CUDA、性能・speedup・A4完成/昇格、A5/A6。
  `full_gpu_world_step=false`を維持する。

## A4.8c8固定テスト

1. 4成功branchをsentinel `last_replication_symbols`付きdirect Formal066へ照合し、変更fieldがlast-symbolだけ、
   high-level RNG call 0、PCG64 state/biology/material/telemetry/object identity不変であることを固定する。
   条件重複fixtureでdisabled→no-genome→replicase-gate→inactive-multiの優先順も固定する。
2. mixed-row pure planをNumPy/Torch CPU/明示CUDA fp64で照合し、branch/error/discrete exact、replicase fp64許容差、
   resident device/pointer/version/content、clone/to_numpy/to_torch/state schema、入力非変更を固定する。
3. empty mutation off/on、negative ATP、dead、replicase exact/nextafter boundary、normal c1〜c7 row、stale cache、
   malformed plan/config/source、capacity one-short、wrong cell/dt/device/RNG、resident/host/candidate tamper、
   duplicate/out-of-order、preclaim不変、注入publish失敗rollbackをfail closedで固定する。
4. c1〜c7 CPU/CUDA delegate、旧CPU fallback不使用、full-step successor event order、save/load/clone、active/pending拒否、
   既存82 testsのname/function-ASTとpromoted A3/A4 pure/c1〜c7 byte hashを固定する。既存82 + 新規最大4 =
   合計最大86 testsをCUDA必須でPASSさせる。

## A4.8c8 GO/STOP判定

- fresh Rule Lockに結び付く実装開始はGOとする。最大86/86、A3 regression、4 branch direct CPU semantics、
  NumPy/Torch CPU/CUDA resident plan、zero RNG、last-symbol-only atomic publish/rollback、delegate、save/clone、
  promoted A3/A4 pure/A4.8c1〜c7 byte不変が全てPASSした場合だけ
  「A4.8c8 pre-active ordinary early-noop atomic commit bridge」と記録する。
- empty/negative/deadをno-opへ畳む、replicase境界を丸める、host値をcommitする、CPU fallback、capacity clip、
  tolerance拡大、既存pure/c1〜c7編集、RNG callまたはlast-symbol以外のstate変化が必要ならA4.8c7を正式authorityとして
  維持してSTOPする。
- A4.8c8でもA4は未完である。次はpersistent resident ragged arena/cacheとmulti-cell GPU-primary pathを別sliceで進め、
  `full_gpu_world_step=false`、速度向上なしを維持する。

## A4.9a追加仮説

Rule Lock receipt `20260815T173703Z`（`work_sessions/20260815T173703Z_SOMA_CELL_0_6_8_GPU_A4_PREFLIGHT.json`、
clean-start HEAD `2f8a6a52914ccada9336b890e85a2ef82b53c7fe`）の範囲は、schedulerが非activeなCPU正本の
world-wide ordered cell setを一度だけexact H2Dして作るimmutable shadowに限定する。shadowは既存
`A4RaggedGenomeBatch`、`A4TranslationStateBatch`と、その同じresident raggedから内部decodeした`A4GeneCacheBatch`を
同じarena generationへ束縛する。CPU正本が変化しない間だけprivate read leaseを発行し、変化後は旧storageを
再同期せず拒否し、CPU正本から別storageへworld全体を再構築してcallerが原子的にswapすれば、c1〜c8の
CPU durable authority、event order、RNG、rollbackを一切変えずに、後続resident化へ必要なcoherence境界だけを
先に閉じられるはずである。

## A4.9aで実装するもの

- 新規module `SOMA_CELL_0_6_8_gpu_a4_resident_arena.py`だけを追加する。既存A4 pure core、A3 scheduler/world、
  A4.8a、A4.8c1〜c8はbyte不変とし、scheduler hook、world subclass、event commitを追加しない。public surfaceは
  `A4ResidentArenaError`、`A4ResidentArenaScopeError`、`A4ResidentArenaEpochs`、`A4ResidentArenaOwner`、build/schema/status
  constantsだけとする。ownerのpublic operationは`from_cpu`、`audit_cpu`、`prepare_rebuild`、`swap_rebuild`、`invalidate`と
  read-onlyな`arena_id/lifecycle/epochs`だけに固定し、tensor/binding/leaseをpublic authorityとして返さない。
- `from_cpu`はexactな`tuple(world.cells)`相当のordered sourceを一つのworld-wide arenaとしてpackする。cell count、
  cell ID、cell order、complete genome order、genome symbol、lesion、template、copy、fractional progressと既存
  `A4TranslationStateBatch`全fieldを失わず、CPU `gene_specs`はcomplete genomeからのdecodeとのexact照合だけに使う。
  caller cache、caller binding、caller tensorを受理せず、resident cacheは同じcandidate resident raggedからだけ
  decodeする。build前後にCPU sourceを再attestし、途中変化を成功arenaへしない。
- arenaのresident ragged/state/cache tensor、offset、mask、metadata、pointer、storage、Torch versionはarena lifetime中
  immutableとする。CPU NumPy/list/dict、既存event-local binding、別arena、cloneとのstorage aliasを禁止する。
  lifecycle metadataとownerのactive-arena pointerだけを可変とし、resident storageへのin-place copy、resize、append、
  compact、slot reuse、pointer差替えを行わない。
- lifecycleは`COHERENT`、`CPU_NEWER`、`INVALID`の3値だけとする。fresh buildのdouble attestation後だけ`COHERENT`、同じ
  membershipでshadow対象R/S/CのCPU sourceまたはsource-object provenanceが変化したら`CPU_NEWER`、membership/order/
  cell identity変化、resident pointer/version/content tamper、malformed source、capacity違反、epoch不整合なら`INVALID`とする。
  旧arenaは`CPU_NEWER`または`INVALID`から`COHERENT`へ戻さず、auditのたびに自動H2Dもしない。
- ownerは単調非減少かつ再利用しない`membership_epoch`、`ragged_epoch`、`state_epoch`、`cache_epoch`を保持し、arenaは
  build時の同じ4 epochをimmutable snapshotする。same-membershipのragged driftはragged/state/cache epochを進め、
  state-only driftはstate epoch、CPU gene-cache/provenance driftはcache epochを進める。membership/count/order/identity driftは
  membership epochを進めて`INVALID`にする。cacheは独立biology authorityを持たず、各arenaの
  `cache_source_ragged_epoch`がそのarenaの`ragged_epoch`とexact一致し、`cache_epoch`がそのdecode generationへ一致する時だけ
  validとする。epoch rollback、wrap、手動代入、同一owner内のarena generation間再利用を拒否する。epoch namespaceはowner-local
  とし、fresh load/clone owner間で同じ数値から始まってもarena IDとowner tokenで区別する。
- privateなevent-free read leaseはarena ID、lifecycle、4 epoch、`cache_source_ragged_epoch`、device、全resident pointer/
  version/provenanceを取得時に束縛し、取得直前にCPU sourceを再auditする。`COHERENT`かつ全値一致時だけ一回発行し、
  lease中もstorageを変更しない。`CPU_NEWER/INVALID`、epoch差、cache-source差、stale/duplicate/cross-owner leaseを
  `A4TranslationBinding`またはkernelへ渡す前に拒否する。active lease中のswap/invalidateを拒否し、lease終了後も同じ
  tokenを再利用しない。
- `prepare_rebuild`は現在のCPU正本を別のfresh storageへworld-wide full pack/H2D/decodeしてprivate candidateを作る。
  old arena/storage/lifecycle/epochを一切変更せず、candidate作成後にもCPUをexact再attestする。`swap_rebuild`はexpected
  old arena ID、expected lifecycle/epochs、zero active lease、candidateのfresh CPU attestationをCAS guardとし、成功時だけ
  ownerのactive pointerを一回差し替える。raggedだけ、stateだけ、cacheだけのincremental refreshや旧storage再利用はしない。
- private creation sealはbiology/source authorityではなく、記録済みstrong referenceとdetached valueの照合recordとする。同じ
  reference/valueを保持するseal record自体の等値copy identityはtrust条件にせず、記録対象のsource/candidate/epoch/tensor/
  digestのidentity、order、value、associationが一つでも変わればclaim前に拒否する。
- capacityはcandidate全体について既存fixed C/Q/S/W/Pとcell-count requirementをallocation/publish前にexact検査する。
  一つでもone-short、offset/mask/tail/topology不正、duplicate cell ID、nonfinite、cache decode不足ならclip、drop、truncate、
  silent grow、CPU fallbackをせずcandidateを破棄し、old arenaとowner状態を不変に保つ。ただしCPUが既にdriftしたold arenaを
  fallback readしてstepを続けることも禁止する。

## A4.9a dirty-domainとCPU境界の固定

- M（membership）はworld cell count/order/identityとarena参加可否、R（ragged）はgenomes/lesions/template/copy/
  fractional、S（state）は既存`A4TranslationStateBatch`全field、C（cache）はcomplete genomeから導くgene cacheとする。
  R driftはgenome-count、lesion-mean、replication-active、material-symbol等のderived stateとbinding provenanceも変えるため
  R/S/Cを同時にstaleとする。Cのcontentが変わらないlesion/template/copyだけのR変化でも、旧cache leaseを新ragged
  epochへ付け替えない。S-onlyはS、CPU `gene_specs`だけの置換/汚染はCをstaleにし、再構築時のdecode exact照合に失敗すれば
  fail closedとする。
- A4 translation commitはS、A4 hydrolysis commitはR/S/C、replication c1 noncompletion、c4 start、c5 mutation-free startは
  R/S/C（cache content不変でもragged provenance失効）、c2 completion、c3 completion-mutation、c6 start-completion、
  c7 start-completion-mutationはR/S/Cをstaleにする。c8 ordered early-noopはshadow外のlast-replication-symbolだけなので
  M/R/S/Cをstaleにしない。全c1〜c8 publishは従来どおりCPUへ行い、A4.9a arenaへpublishしない。
- CPU gene refreshはS/Cとする。A3 adapterのgeneric/precursor/maintenance/surface assembly/protein damage/aggregation/
  reactive/membrane oxidation/ordinary decay/export/radius/division-progress/repair/learning/mobile-export/supplemental metabolism/
  segregation planningは、共通unpackが`genome_lesions` outer listとtemplate-lesion provenanceを再束縛するため、contentが
  等値でもR/S/Cを一回staleにする。A4 translation単独commitはSだけをstaleにする。lesion gain、genome-lesion repair、
  replication-copy leakはR/S/C、hydrolysis deletion、eDNA/HGT integration、Formal066 grammar mutationはR/S/Cをstaleにする。
  pre/post P2、current-stress assignment、sense/surface exchange/effector/corpse-contactは対象pool、damage、signal、behavior、
  neural-osmolyteを変更した時Sをstaleにする。位置、age、mutation ledger等のshadow外fieldだけの変化はM/R/S/C epochを
  進めないが、同じroutineがpool等を変更した時はSを進める。
- alive/dead participationの変化、actual split、daughter replacement、death removal、washout、cell reorderはMを変えて旧arenaを
  `INVALID`にする。A5 CPU segregation/split、death release、washout、HGTは常にCPU正本だけを読むため、A4.9aにはD2H boundaryが
  存在しない。A5前にflushを装ってはならず、A5後の旧leaseを拒否し、必要なら安定CPU worldから全arenaを別途再構築する。
  corpse/eDNA/environmentだけの変化はcell R/S/Cへ反映されるinteractionが起きるまでarenaをdirtyにしない。
- 既存world event order、cell event order、exactly-once claim、PCG64 call order、receipt、rollbackを完全に維持する。
  arena build/audit/rebuild/swapはscheduler active/pending commit中に実行せず、A4.9a leaseをc1〜c8 live eventへ接続しない。

## A4.9a save/load/cloneとsource-of-truth

- CPU world/cellだけをdurable source-of-truthとする。arena、candidate、epoch-bound lease、resident cache、pointer/provenance、
  lifecycleは`state_dict`へ保存せず、既存c1〜c8のactive/pending save拒否を変更しない。`CPU_NEWER`または`INVALID` shadowがあっても
  resident-newer biologyは存在しないため、CPU saveをD2H待ちにせずshadowを破棄できる。
- load/cloneはCPU durable stateと既存RNGだけを復元/複製し、必要時に`from_cpu`で新しいarena ID、fresh storage、fresh cache、
  fresh epochsから再構築する。元world/clone/load先のtensor、cache、candidate、lease、ownerを共有せず、epoch値やpointer identityの
  一致をclone correctness条件にしない。biologyと次のCPU event/RNG resultは従来どおりexactでなければならない。
- residentをsource-of-truthとするwrite、device-dirty、`RESIDENT_NEWER`、D2H、CPU overwrite、live commit、multi-cell GPU-primary
  authorityをpublic/privateを問わずscope errorで拒否する。これらの状態、flag、APIを成功pathとして先回り実装しない。

## A4.9aで実装しないもの

- CPU drift後の自動またはincremental H2D refresh、old storageへのcopy、per-cell dirty upload、allocator/compaction、slot reuse、
  partial arena、resident write、device-dirty/`RESIDENT_NEWER`/D2H、A5 flush、live c1〜c8 binding、scheduler/world integration。
- c1〜c8 authority/dispatch/publish/rollbackの変更、CPU fallback置換、device RNG、multi-cell biological kernel、event fusion、
  fp32、compile/graph/Triton/custom CUDA、performance/speedup claim、A4完成/昇格、A5/A6。
  `full_gpu_world_step=false`を維持する。

## A4.9a固定テスト

1. heterogeneous world-wide fixtureをfresh CPU sourceからbuildし、ragged/stateのexact readback、resident-only cache decode、
   cell/genome order、M/R/S/C/cache-source epoch、CPU/Torch CPU/明示CUDA fp64一致、pointer/version不変、全入力非変更、
   CPU/event-local/clone/arena間non-aliasを固定する。
2. 各dirty categoryとc1〜c8 write-setをfixtureで再現し、`COHERENT`からsame-membership `CPU_NEWER`またはmembership `INVALID`への
   一方向遷移、RからS/Cへの波及、c8/out-of-scope-only不変、private leaseのfresh/one-shot/cross-owner/stale拒否、resident
   pointer/version/content tamper拒否を固定する。
3. `CPU_NEWER/INVALID`からの別storage `prepare_rebuild`とexpected-ID/epoch CAS `swap_rebuild`、build途中CPU drift、active lease、duplicate/
   reordered cell、C/Q/S/W/P/cell-count各one-short、malformed ragged/cache/stateを注入し、失敗時old arena/epoch/lifecycle/
   CPU biology不変、clip/grow/fallbackなしを固定する。
4. existing c1〜c8/A3 event order、RNG、receipt、save/load/clone、CPU A5 split/death/washout結果とbyte hashが不変であること、
   arena非serializationとload/clone fresh identity、device-dirty/`RESIDENT_NEWER`/D2H/live-authority operationのscope拒否を固定する。
   既存最大86 + 新規最大4 = 合計最大90 testsとし、performanceを測定しない。

## A4.9a GO/STOP判定

- 上記fresh Rule Lockに結び付く新module実装開始はGOとする。最大90/90、world-wide lossless pack/readback、resident-derived
  cache、全epoch/lifecycle/lease、immutable pointer/storage、separate rebuild/CAS swap、capacity fail-closed、save/load/clone、
  A5 CPU boundary、promoted A3/A4 pure/A4.8c1〜c8 byte不変が全てPASSした場合だけ
  「A4.9a immutable world-wide H2D shadow/cache coherence foundation」と記録する。
- stale lease受理、CPU drift見逃し、old arenaの再cohere、`cache_source_ragged_epoch`不一致、resident in-place mutation、alias、
  partial/incremental refresh、capacity clip/grow、CPU fallback、D2H/device-dirty/`RESIDENT_NEWER`、scheduler/event/RNG/save semantics
  変更、c1〜c8またはA4 pure編集が必要ならA4.8c8を正式authorityとして維持してSTOPする。
- A4.9aでもresident biology authority、multi-cell GPU-primary path、性能向上は未達であり、A4完成または昇格と呼ばない。
  `full_gpu_world_step=false`、baseline A3、CPU durable authorityを維持する。
