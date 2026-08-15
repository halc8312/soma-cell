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
