# SOMA簡略変更履歴

- SOMA-0: 単体・再帰神経・生涯学習
- SOMA-1: 因果栄養組織、複数身体通貨
- SOMA-2/2.1: 実停止監査、形態実験、期限付き器官
- SOMA-3: 内生的問い生成
- SOMA-4/4.1/4.2: 反証コンパイラ、ツイン評価、変化ゲート
- SOMA-5R/6R/7R: 内的計画、共有文化、生態系の品質再構築
- SOMA-CELL 0.1: 動的膜と自己生産
- 0.2: 物質ゲノムと変異
- 0.3: 損傷・修復・機能年齢
- 0.4: 物質感覚運動と局所可塑性
- 0.5: 死体化学、環境DNA、HGT、可動配列
- 0.6-P0: 化学身体API凍結（22/22検証完了、基準昇格待ち）
- 次: 0.6-P1 1個の物質神経細胞

## 2026-08-01 — Project Rule Lock 1

- `00_MANDATORY_STARTUP_GATE_JA.md`を追加。
- `MANDATORY_WORKFLOW.json`を追加。
- `scripts/soma_preflight.py`を追加し、必須ファイル・SHA-256・マイルストーンを失敗閉鎖で検査。
- `work_sessions/*_PREFLIGHT.json`による作業開始証跡を導入。
- `.githooks/pre-commit`を追加し、実装変更にプリフライト証跡を要求。
- `PROJECT_STATE.json`へ`workflow_guard`を追加。
- READMEと復元手順へプリフライトを最優先として追記。

## 2026-08-01 — Project Rule Lock R2

- プリフライト証跡を現在のGit HEADへ結合。
- `check-receipt`コマンドを追加。
- pre-commitで`verify`と`check-receipt`を必須化。
- 作業開始時にクリーンなGit worktreeを要求。
- 古い証跡の使い回しを防止。

## 2026-08-01 — SOMA-CELL 0.6-P0.0

- 凍結SOMA-CELL 0.5を変更せず、`ChemicalBodyPort`を追加。
- 実化学量だけを返す読み取り専用・RNG非消費センサーを追加。
- ATP、タンパク質基質、膜前駆体、有限信号の保存的予算エスクローを追加。
- 前駆体から組織物質への変換に明示的ATP費を追加。
- 運動、輸送体極性、修復極性、休眠だけを既存身体物理経由で要求可能にした。
- 位置、膜量、ATP、物質プール、DNA、ゲノムの直接書換えを失敗閉鎖で拒否。
- 組織死、宿主死、分裂前に神経物質を保存的に回収し、無料の組織・記憶継承を禁止。
- 0.5保存状態からP0への無変更移行と、P0保存復元を実装。
- 22/22の契約検証をPASS。
- 3 seed × 5条件＝15試行の工学比較を実施。未接続、Null接続、予算往復は0.5と3/3完全一致。
- P0には情報処理神経、学習、因果監査はまだ存在しない。

## 2026-08-01 — Project Rule Lock R2 milestone-transition extension

- 必須読込対象を`PROJECT_STATE.json`と`MANDATORY_WORKFLOW.json`から動的解決するようプリフライトを更新。
- `transition`コマンドと一回限りの`*_TRANSITION.json`を追加。
- 基準版・次マイルストーン更新も証跡なしではコミットできない設計へ変更。
- 遷移コミット後は証跡が自動的に古くなり、次実装前の再読込を強制する。

## 2026-08-01 — P0.2 ポート表面の硬化

- ポートの公開`cell`／`world`参照を廃止。
- `attach()`は可変な内部エスクローを返さず、読み取り専用診断コピーだけを返すよう変更。
- 物質・予算・エフェクタ契約は維持し、検証を22/22へ拡張。
- ビルド名は0.6-P0.0のまま、身体ポート契約版を0.6-P0.2へ更新。

## 2026-08-01 — SOMA-CELL 0.6-P0を凍結基準版へ昇格

- Rule Lockの一回限り遷移証跡でcurrent baselineをSOMA-CELL 0.6-P0.0へ更新。
- next milestoneをSOMA-CELL 0.6-P1へ更新。
- P1開始時にP0身体ポート契約、P0本体、22項目検証、15試行レポートの再読込を必須化。
- P0には情報処理神経がなく、適応利益を主張しないという限定を基準状態へ固定。

## 2026-08-01 — SOMA-CELL 0.6-P1.0

- P0化学身体ポートだけを使用する1個の物質神経区画を追加。
- 既存物質ゲノム文法へ16記号の神経発生カセットを追加。
- ATP、タンパク質基質、膜材、有限シグナルを使う区画発生・維持・摩耗を実装。
- 実リガンドの36膜区画分布から局所勾配を読むセンサーを追加。
- 1つの漏れ積分状態と、次濃度・勾配・流入を予測する局所予測器を追加。
- 運動と輸送体偏在をP0の有料物理エフェクタ要求として接続。
- 同物質・同維持・同出力強度の非情報処理ダミーを主対照として追加。
- 損傷タンパク質・凝集体による神経区画交換と保存的物質返却を追加。
- 分裂前に区画を回収し、学習済み数値状態の無料娘継承を禁止。
- カセットなしでP0と3 seed × 240 stepの状態・RNG完全一致を確認。
- P0 22/22回帰検証を再実行しPASS。
- P1 18/18決定論的検証をPASS。
- 3 seed × 8条件＝24試行を実施。
- moving-patchで神経区画が等費用ダミーを3/3 seedで上回り、P1情報価値ゲートをPASS。
- 局所予測器は予測誤差を低下させたが、身体利益は一貫せず、未証明として記録。


## 2026-08-01 — SOMA-CELL 0.6-P1を凍結基準版へ昇格

- Rule Lockの一回限り遷移証跡でcurrent baselineをSOMA-CELL 0.6-P1.0へ更新。
- next milestoneをSOMA-CELL 0.6-P2へ更新。
- P2開始時にP1本体、P1組織契約、P1検証、24試行レポート、P0身体ポート契約の再読込を必須化。
- P1予測器の純身体利益が未証明であることを凍結し、P2のno-prediction対照を必須化。
- P2は等物質8細胞固定組織から始め、再帰・予測・可塑性を段階的に導入する。


## 2026-08-01 — P1配布アーカイブR1

- 初回P1アーカイブの`SHA256SUMS.txt`に自己参照行が含まれる包装不具合を検出。
- SHA256SUMS自身を検査対象から除外し、全対象ファイルの照合をPASS。
- P1検証スクリプトをフラット配布フォルダでもP0ソースを検出できるよう修正。
- R1配布物を新名で作成し、旧アーカイブは遷移証拠として保存。
- シミュレーション本体、P1 18/18結果、24試行結果は変更なし。

## 2026-08-02 — SOMA-CELL 0.6-P2.0

- 16記号×8本の物質神経発生カセットを追加。
- 8個のATP・タンパク質・膜・有限信号を持つ神経区画を追加。
- 20本の有料・有限信号・摩耗付き疎な再帰接続を追加。
- 12物理特徴入力と細胞別局所予測器を追加。
- 適格度トレースと4物理負債に基づく三因子可塑性を追加。
- 発達費誤帰属を防ぐ34秒の発達安定期間を追加。
- 分子エピソード記憶へ最大急性害の記録を追加。
- 成熟、再可塑化、保存的な結線再編・細胞回収を追加。
- 規則反転用の実粒子環境と全条件共通の物理マイクロ流体トラップを追加。
- ステップ番号由来の共通外乱テープを追加。
- P2検証22/22、P0回帰22/22、P1回帰18/18をPASS。
- 36比較試行を実施。holdout全機構対固定8区画は5/6正、平均AUC +1.872678。
- 可塑性は限定的正信号。予測の正味価値は負／未確立、再帰の効果は極小として凍結。
- Pythonista平坦配布からP2/P1/P0全検証を再実行する正式ZIPを作成。

## 2026-08-02 — SOMA-CELL 0.6-P2を凍結基準へ昇格

- Rule Lock遷移証跡によりcurrent baselineをSOMA-CELL 0.6-P2.0へ更新。
- next milestoneを正式SOMA-CELL 0.6へ更新。
- 正式0.6開始時にP2契約・ソース・検証・36試行レポート、P1/P0契約、SOMA-2.1/4.2を再読することを必須化。
- 正式0.6では物質的ABBA/BAAB監査、因果校正、変化ゲート、小型反証、神経死体/HGTを段階導入する。

## 2026-08-02 — SOMA-CELL 0.6.0

- P2へ物質的ABBA/BAAB監査、偽対照、証拠品質、因果校正、変化源分離を統合。
- 反証区画の期限付き資源リース、小型反証、神経死体化・神経遺伝子HGTを追加。
- Formal 22/22、全回帰84/84 PASS。33比較試行。
- 安定holdoutで誤フィードバック0件。既知反証で方向と平均取り込みを改善。
- 自然な短期反転では自動フィードバック0件。科学状態をPASS_WITH_LIMITSとして凍結。
- 次を0.6.1の自律因果循環に限定。

## 2026-08-03 — SOMA-CELL 0.6.1 R2正式凍結

- 証拠不足と変化なしを分離する証拠充足台帳を追加。
- ATP・材料・危険費を払う低危険な診断曝露を追加。
- 命令→推進力の有料反復校正と故障確認を追加。
- 神経／監査コントローラーのATP・信号エスクロー上限を追加し、未使用分を身体へ保存的に返却。
- 故障確認後に運動支配区画を一時休眠させる期限付き保全リースを追加。
- 休眠対象を絶対運動量ではなく、現在運動方向への正の仕事で選ぶよう修正。
- R1事前登録失敗を保存し、R1 seedを最終評価から除外。
- R2をソースハッシュと未使用seedで事前登録。
- R2安定20、故障12、自然空間探索6の38試行を完了。全一次基準PASS。
- 0.6.1専用43/43、P0/P1/P2/0.6回帰84/84、合計127/127 PASS。
- 外部分子の能動意味診断、予測、再帰、自然空間取得の純利益は未確立として残す。

## 2026-08-04 — SOMA-CELL 0.6.1 UI1 保守修正

- Pythonistaで親SceneのHUDが多重描画され、世界領域の約半分が隠れる問題を修正。
- 0.6.1 Sceneは0.5の世界描画とP2神経オーバーレイを直接描き、上52px・下128px以内のコンパクトHUDへ統合。
- 表示サマリは`.get()`で安全に参照し、リセット直後の一時的なキー不一致で落ちないようにした。
- ダブルタップ時に親クラスのFormal06Worldへ戻る問題を修正し、必ずFormal061Worldを再生成する。
- 科学機構、R2事前登録結果、0.6.2計画は変更しない。

## 2026-08-04 — SOMA-CELL 0.6.2

- 0.6.1 UI1のコンパクトHUDと安全な子Worldリセットを継承。
- 予測、再帰、可塑性へ独立したATP・材料摩耗台帳を追加。
- 通常実行の無料モジュールを失敗閉鎖。
- 8/4/2/1/0区画プロファイルを追加。
- 非活動区画のエスクローと組織物質をP0身体へ保存的に返却。
- no_prediction / no_recurrence / no_diagnosis / no_plasticity / efficient4 / efficient2 / no_tissue比較を追加。
- clone設定の派生フィールド漏出をR3事前登録修正として公開。
- 94件の事前登録比較を実施。
- 予測は研究プロファイルへ保持、再帰と能動意味診断は既定休眠。
- efficient2を最小神経研究プロファイルとしたが、no_tissueが費用込み身体余裕で上回った。
- 0.6.2専用24/24、継承回帰127/127、合計151/151 PASS。
- 科学結果を`NEGATIVE_RESULT_WITH_LIMITS`として凍結。
- 次を需要連動型ニューロジェネシス0.6.3へ変更。

## 2026-08-08 — SOMA-CELL 0.6.3

- 0.6.2の`no_tissue`勝利を受け、常設追加神経を既定から外した。
- 有料の非神経センチネルと実前駆体予約を追加。
- 物理的な情報不足・意味変化・運動変換変化・身体低下から発生需要を推定。
- 推定価値が発生・維持・再吸収費を上回る場合だけ2〜4区画を物質発生。
- 発達中の神経計算を禁止し、成熟後のみ活動開始。
- ACTIVE/DORMANTのABBA/BAAB器官価値監査を追加。
- 監査中は予測・可塑性の学習を凍結し、比較中の組織変化を防止。
- 正効果なら期限付きリース、非正ならATP・膜・タンパク質・信号物質を保存的に再吸収。
- random-cost-matched、prepared-no-tissue、bare-no-tissue、always-efficient2/4等の対照を追加。
- 事前登録74試行を実施。stable誤発生0/12。
- 主要holdoutのdemand対preparedは平均差+0.272332だが正1/6で再現性ゲートFAIL。
- demand対bare no-tissueは平均AUC差-0.621635、正1/6。
- 0.6.3専用26/26、継承151/151、合計177/177 PASS。
- 工学状態PASS、科学状態`NEGATIVE_RESULT_WITH_LIMITS`として凍結。
- 次を費用償却境界と準備投資ゲートの0.6.4へ変更。

## 2026-08-08 — SOMA-CELL 0.6.4 費用償却境界

- 0.6.3の需要連動型ニューロジェネシスを親に、早期前駆体オプションを実タンパク質・膜・信号物質として追加。
- 完全2区画器官に対する早期予約率0.25/0.50/0.75/0.85/0.90/0.95/1.00を事前登録して比較。
- 条件付き完全準備、低需要時の拡張分返却、オプション床の保存を実装。
- 周期的な物理規則変化を個体へ時刻通知せず実行。
- `option_no_tissue`、bare、eager、常設efficient2、費用整合ランダムを主要対照に追加。
- 物理的発生境界は0.95。0.90以下は3/3で器官を形成できず、0.95以上は3/3で形成。
- 安定12系列で誤完全準備0、誤器官発生0。
- 未調整6系列でconditionalは全5対照との平均身体余裕AUC差が負。主要対照option-no-tissueに対して正1/6。
- 0.6.4専用25/25、継承177/177、合計202/202 PASS。
- 科学判定を`NEGATIVE_BOUNDARY_RESULT_WITH_LIMITS`として凍結し、追加神経文法を既定休眠へ移す方針を採用。

## 2026-08-08 — SOMA-CELL 0.6.5

- 0.6.4の負の境界結果を受け、追加神経文法を既定休眠へ移した。
- sentinel / readiness / organ / prediction / plasticityを通常の物質ゲノム上の区切り遺伝子として実装。
- whole-gene deletion/duplication、promoter dormancy/reactivation、payload regulationを追加。
- 3世代×8個体の加速系統選択アッセイと、stable/periodic/rare_fault/long_delay環境を追加。
- lineage内の競合ゲノムは同一body-world seedで評価し、genotype固有seed混入を防止。
- periodic/long_delayで3/3保持、stable shrinkageは1/3で事前登録ゲートFAIL。
- periodic候補の実配列ノックアウト6/6低下、再導入6/6回復。
- 0.6.5専用24/24、継承202/202、合計226/226 PASS。
- 科学状態を`PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETE`として凍結。


## SOMA-CELL 0.6.6 — 2026-08-08
- 外部トーナメント型世代選択を廃止し、共有化学生態系の実出生・実死亡へ移行。
- generation 2を主要9/9試行で内生的に確認。
- 神経文法の欠失・重複・休眠・再活性化を実分裂時の物質ゲノム変異へ接続。
- 死体化学・環境DNA・HGTを共有生態系へ統合。
- stableの活動縮小はPASSしたが完全文法喪失はFAIL。periodic/long-delayの保持優位もFAIL。
- 0.6.5の複雑環境保持信号が外部トーナメント選択に依存した可能性を正式な負の結果として凍結。


## 2026-08-08 — SOMA-CELL 0.6.7
- generation 15〜21用の粗視化物質serial-transfer ecologyを追加。
- genome/copy polymer/module protein/ATP/death/eDNA/HGT/neutral outflowを台帳化。
- R1/R2/中間登録をsource driftまたはseed再使用で無効化し、R3未使用seed 6731〜6733を事前登録。
- R3 12/12行を報告。長期世代・物質ゲートPASS。stable loss、strict HGT rescue、long-delay保持ゲートFAIL。
- 専用28/28、継承251/251、合計279/279 PASS。
- 次を0.6.8 regime-shift evolutionary memoryへ変更。

## 2026-08-08 — SOMA-CELL 0.6.8-GPU A1 engineering checkpoint

- 0.6.6完全粒子世界用の固定容量Torch tensor schemaを追加。
- 粒子、細胞、36区画膜、輸送体、13内部プール、可変長ゲノムbyte mirrorを追加。
- opaque CPU stateを保持するlossless adapterを追加。
- device-neutral counter RNGを追加。
- particle diffusion + patch driftのNumPy/Torch fp64 lockstepを追加。
- membrane-local particle ligand profileのNumPy/Torch fp64 lockstepを追加。
- core metabolismの独立NumPy/Torch lockstep版を追加。
- frozen 0.6.6 worldへdiffusion/profileだけを接続するHybrid066Worldを追加。
- batched independent-world benchmarkとVRAM容量推定を追加。
- 0.6.8-GPU専用32/32 PASS、親0.6.7専用28/28 PASS。
- 開発環境はTorch CPU-onlyであり、CUDA実機性能は未検証として固定。
- full world-step未移植のため正式0.6.8科学版には昇格せず、A1 engineering checkpointとして保存。

## 2026-08-08 — SOMA-CELL 0.6.8-GPU A2 engineering checkpoint

- A1を変更せず、A2 extension moduleを追加。
- toroidal spatial candidate indexと将来のCUDA segmented scan用grid keyを追加。
- fuel/mineral/waste/alt surface exchangeをNumPy/Torch fp64へ移植。
- 0.4 sensorimotor uptake trace、eligibility、contamination後処理までhybrid統合。
- ATP有料waste exportを移植。
- damage-aware closureとpolymer-aware osmosisを使うleak planを移植。
- 漏出粒子生成とRNG順序はCPU正本に保持。
- polymer-aware radius relaxationとsurface-flux/Brownian motionを移植。
- 一歩、十歩、周期環境、clone、save/restoreでCPU/Torch event lockstepを確認。
- A2専用32/32、A1専用32/32再実行、継承279/279を維持。
- CPU-only benchmarkではA2 hybridが約1.524倍遅く、速度結果ではないことを明記。
- genome/translation/division/death/eDNA/HGT/neuralはCPU正本のまま、full_gpu_world_step=false。

## 2026-08-14 — SOMA-CELL 0.6.8-GPU A3 engineering candidate

- 0.6.6詳細世界のgene-coded metabolism、ATP生成、有料前駆体合成、維持・組立を独立NumPy/Torch kernelへ移植。
- active/damaged proteinのfingerprintと辞書挿入順を保持する固定容量packed stateを追加し、容量超過をatomicにfail closed化。
- damaged-protein aggregateをA3生成分のtyped compositionと、推定禁止のlegacy unresolved量に分離。
- reactive byproduct、membrane oxidation、genome-lesion chemistry、antioxidant/chaperone/protease/genome/membrane repairをfp64で照合。
- damage segregationはactual divisionを変更しないpure planとして追加。translation、replication、hydrolysis RNG、actual divisionはCPU正本を維持。
- A2 surface/export/leak/radius/motionとA3 metabolism/repairをcanonical orderで一度だけ実行するreceipt付きschedulerを追加。
- `maintenance_shortfall`を含むscalar telemetryのatomic unpackを修正し、ATP不足時の膜・輸送体劣化をCPU正本と統合lockstep化。
- mask/count/order tail、mass完全被覆、gene/genome/replication、backend/device/dtypeを厳密検査し、隠れたstateとmixed tensorをcommit前に拒否。
- validation/benchmarkへ正本source SHA-256 mapを付与し、release builderが36 test ID、13 spec/65 measurement、fp32親artifactと現行sourceを独立照合するよう強化。
- 独立kernel、strict lossless schema、1/10 step、ATP不足、transport無効、stress、pre-division、periodic/reversal、clone、save/restore、schedulerを含むA3専用検証36/36 PASS。
- RTX 4060 Ti 16GB上でCUDA fp64 kernel parityをPASSし、fp64合格後のfp32最大差1.8013e-07を候補精度として記録。
- 同一source hashに結合した正式benchmarkを約4時間3分で完走。13 spec/65 measurement、CPU/CUDA fp64 correctness 1/10/20 stepを全て通し、fp64 state最大差3.553e-15、ledger差0、RNG完全一致を記録。
- world 1/8/32/128でA3 CUDA fp64は凍結CPU正本より約570.096〜590.276倍遅く、CUDA fp32も約573.308〜589.486倍遅かった。速度向上なしという負の性能結果を保存。
- fp32 full-world最大差3.5982e-06、ledger最大差1.7114e-06をcandidate-onlyとして保存。structured CUDA peak allocatedはfp64 215,552 bytes、fp32 190,976 bytes、reservedは双方2,097,152 bytes。
- A1 32/32、A2 32/32、Windows nativeの歴史11 validator 279/279を再実行し、合計379/379 PASS。WSL/NumPy 2.5.2だけでP2が21/22となる数値stack感度も負の互換性証拠として保存。
- 未移植機構が残るため`full_gpu_world_step=false`を維持し、CUDA高速化・生命・意識・オープンエンド進化を主張しない。
