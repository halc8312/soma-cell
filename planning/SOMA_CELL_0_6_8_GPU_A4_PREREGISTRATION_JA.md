# SOMA-CELL 0.6.8-GPU A4.6a 事前登録（A4.1〜A4.5b継承）

状態: A4.5bまでを継承する開発用の最小slice。A4昇格判定ではない。

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

次はA4.6bとしてcompletion後のstructural/material mutation RNGを別sliceで扱う。その後
hydrolysisをA4.7として分離する。scheduler置換は、連続したresident chainをhost/device往復
なしでatomic commitできる段階まで延期する。
