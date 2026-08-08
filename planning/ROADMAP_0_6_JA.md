# SOMA-CELL 0.6 系統ロードマップ

更新日: 2026-08-03

## 0.6-P0〜P2 — 凍結済み基盤

- P0: 化学身体ポート、有限ATP・材料予算、失敗閉鎖の物理エフェクタ
- P1: 一個の物質神経細胞、等物質ダミー比較
- P2: 8区画、有限信号再帰、局所予測、三因子可塑性、保存的再編

凍結した負の結果:

- P1/P2の局所予測器は予測誤差を下げたが、純身体利益は未確立。
- P2再帰の必要性は未確立。

## 正式0.6 — 凍結済み

- 物質的ABBA/BAAB停止監査
- 活動・運動影響・物質費を合わせた偽対照
- 証拠品質、因果校正、変化源分離
- 期限付き因果リース、小型反証コンパイラ
- 神経死体化、環境DNA、HGT

工学統合はPASS。自然な短期監査→反証→再可塑化→利益の全循環は未証明。

## 0.6.1 — 凍結済み・PASS_WITH_LIMITS

### 実装

1. `変化なし`と`証拠不足`を分離する証拠充足台帳
2. ATP・材料・危険費を払う低危険診断
3. 命令→実推進力を測る有料反復パルス
4. 神経／監査コントローラーの再利用ATP・信号エスクロー上限
5. 未使用物質の身体への保存的返却
6. 物理的運動器故障後の期限付き神経休眠リース
7. 現在の運動命令へ正の仕事をする区画の対象選択
8. R1失敗の保存とseed除外、R2事前登録holdout

### R2一次結果

- 安定20系列: 誤リース0、誤feedback 0
- 運動器故障12双子: 故障確認12、リース12、費用込み身体余裕AUC正12
- 平均AUC差 +0.0294916
- 平均運動ATP差 -0.000803644
- 平均散逸差 -0.000774693
- 自然空間探索6双子: 身体余裕AUC正6、短期取得差0

### 限定

- 強い証拠は制御された運動器故障の代謝保全。
- 自然空間の取得改善は未支持。
- 外部分子の能動意味診断、予測、再帰の純価値は未解決。

## 次: SOMA-CELL 0.6.2 — 神経モジュール代謝監査と最小化

目的は脳を大きくすることではなく、各モジュールが自分の足代を払えるかを測り、価値のない機構を縮小・休眠・削除することである。

### 0.6.2-A: モジュール別費用台帳

- 感覚処理
- 再帰信号
- 局所予測
- 可塑性
- 因果監査
- 能動診断
- 故障保全リース

ATP、信号、タンパク質摩耗、修復、散逸をモジュール別に記録する。

### 0.6.2-B: 同一外乱アブレーション

- 0.6.1 full
- no prediction
- no recurrence
- no plasticity
- no causal audit
- no active diagnosis
- no conservation feedback
- 4 / 2 / 1区画
- no tissue

### 0.6.2-C: 条件付き休眠と廃止

安定環境と非定常環境で純価値が非正なら、該当モジュールを縮小・休眠・削除する。見栄えの良い機構を理由に残さない。

### 0.6.2合格条件

- 0.6.1の安定安全性と故障時保全利益を維持
- 事前登録holdoutで、残す各モジュールが安価な対照に費用込みで優位
- 価値がないモジュールを実際に縮小または棄却
- P0〜0.6.1全回帰、物質台帳、保存復元を維持

## 0.6.x以降へ保留

- 多段内的世界
- 概念、夢／再生
- 物質通信・文化
- 大規模多個体生態系
- 固定48細胞脳

これらは、0.6.2で最小神経組織の純価値を確定した後にだけ段階復帰する。

## 0.6.2 — 凍結候補・神経代謝監査の負の結果

### 実装

- 予測、再帰、可塑性の明示的ATP・材料摩耗台帳
- 8/4/2/1/0区画プロファイル
- 非活動区画の保存的物質返却
- no-prediction / no-recurrence / no-diagnosis / no-plasticity対照
- efficient4 / efficient2研究プロファイル
- 0.6.1 UI1表示・リセット修正の継承

### 94試行の主要結果

- screening: no_tissue AUC 19.305746 > full8 18.102215
- confirmation: no_tissue 23.661983 > efficient2 23.159490
- 予測棄却基準は未達のため研究プロファイルに保持
- 再帰保持基準は未達のため既定休眠
- 安定20試行で誤保全リース0
- full8故障保全は6/6で正のAUC差を維持

### 判定

```text
代謝監査の選択       : no_tissue
最小神経研究プロファイル: efficient2
科学状態             : NEGATIVE_RESULT_WITH_LIMITS
```

現在の追加神経層は短いcue-reversal課題で発生・維持費を回収しなかった。この負の結果を理由に、固定脳の拡大と多世代進化を一旦延期する。

## 次: SOMA-CELL 0.6.3 — 需要連動型ニューロジェネシス

通常時に脳を維持せず、必要性を物質的に推定して神経組織を発生・休眠・再吸収する。

### 0.6.3-A: 最小センチネル

- 神経組織なし、または1個の低費用センチネル
- 環境非定常性、情報不足、運動変換故障を読む
- 正解方向・外部報酬は受け取らない

### 0.6.3-B: 発生価値予測

```text
期待される情報利益
- 神経発生ATP
- 神経材料
- 維持費
- 発生遅延
- 誤発生危険
```

が正の場合だけ2〜4区画を発生する。

### 0.6.3-C: 期限付き神経器官リース

- 発生した神経組織は期限付き
- 共通外乱ツインまたは可逆休眠で利益を再検査
- 利益がなければ組織を身体へ再吸収
- 利益が再現すれば期限延長

### 必須対照

- 常時no_tissue
- 常時efficient2
- 常時efficient4
- 費用整合ランダム発生
- 需要連動発生
- 発生後の再吸収なし

### 合格条件

安定、移動資源、意味反転、運動器故障を含む未調整タスク群で、需要連動型が常時神経・常時無神経・ランダム発生より総合費用込み身体余裕を改善し、誤発生を抑えること。

## 0.6.4 — 多世代の神経発生規則進化

0.6.3が合格した場合だけ、発生閾値、区画数、予測、可塑性、器官リース期間を物質ゲノムの進化対象にする。

## 0.6.3 — 凍結済み・需要連動ニューロジェネシスの負の結果

### 実装

- 有料の非神経センチネル
- タンパク質・膜・信号前駆体の実予約
- 需要スコアと明示的発生費の比較
- 2〜4区画の発達中計算遮断
- ACTIVE/DORMANTの反転器官価値監査
- 監査中の学習凍結
- 正なら期限付きリース、非正なら保存的再吸収
- prepared-no-tissue / bare-no-tissue / random-cost-matched / 常設神経対照

### 74試行の主要結果

- stable誤発生: 0/12
- demand−prepared平均AUC: +0.272332、正1/6（事前登録基準4/6をFAIL）
- demand−bare no-tissue平均AUC: -0.621635、正1/6
- demand−prepared物質取得: 平均+0.064033、正3/6
- screeningではno-predictionとno-plasticityがfull demandを上回った

### 判定

```text
engineering: PASS
science: NEGATIVE_RESULT_WITH_LIMITS
```

機構は動作し、誤発生を抑えたが、短い意味反転課題で準備・発生・維持・再吸収費を一貫して回収できなかった。

## 次: SOMA-CELL 0.6.4 — 費用償却境界と準備投資ゲート

### 目的

一時器官が不利だった理由を、課題時間不足、変化頻度、準備費、器官機能の純価値へ分解する。

### 事前登録する軸

- 課題継続時間
- 規則変化頻度
- 予測が必要な遅延長
- 非神経反射だけでは解けない部分観測性
- センチネル・前駆体準備の有無
- 2区画／4区画／無組織

### 主要対照

- bare no_tissue
- prepared no_tissue
- demand transient organ
- always efficient2
- 費用整合ランダム発生
- 条件付き準備発現

### 合格条件

未調整holdoutで、需要連動器官が費用を回収する明確な課題領域を再現性付きで示す。該当領域が見つからなければ、ニューロジェネシス文法を縮小または休眠する。

## 0.6.4 — 凍結済み・準備オプション境界の負の結果

### 実装と結果

- 実物質の早期前駆体オプション
- 証拠後の完全準備拡張と、低需要時の拡張分返却
- 0.25〜1.00の予約率境界
- option-no-tissue / bare / eager / 常設2区画 / random対照
- 最小物理発生境界0.95
- stable誤完全準備・誤発生0/12
- conditionalは未調整holdoutで全5対照との平均差が負

### 判定

```text
engineering: PASS
science: NEGATIVE_BOUNDARY_RESULT_WITH_LIMITS
```

後から器官を作るにはほぼ全材料を早期予約する必要があり、条件付き投資の節約余地は小さかった。追加神経文法は既定休眠にする。

## 次: SOMA-CELL 0.6.5 — 多世代の器官経済と遺伝的保持・欠失

- センチネル、準備率、器官発生、予測、可塑性を物質ゲノム上の可変モジュールにする
- 重複、欠失、調節低下、休眠、HGTを許可
- 神経文法なし／最小文法／完全文法を同じ化学生態系で競争させる
- 安定・周期変化・稀な故障の環境別に保持頻度を測る
- ノックアウトと再導入で因果を確認する
- 神経文法が費用を払えない場合は自然に欠失することを許容する

0.6.5では神経を有利にすることを目的とせず、環境依存で保持されるか失われるかを結果として受け入れる。

## 0.6.5 — 凍結済み・神経文法の加速多世代アッセイ

- 5つの神経文法モジュールを物質ゲノム化。
- 欠失、重複、休眠、再活性化、調節変異を実装。
- stable / periodic / rare_fault / long_delayで3世代×8個体×3系列を事前登録。
- periodic/long_delayは3/3保持、stable shrinkageは1/3でFAIL。
- periodic候補はノックアウト6/6低下、再導入6/6回復。
- 判定: `PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETE`。

## 次: 0.6.6 — 共有化学生態系での実多世代選択

- トーナメント世代置換を廃止。
- 実際の分裂、構造的死、死体、資源競争、環境DNA、HGTを共有世界で用いる。
- lineage IDと神経文法頻度を長時間追跡。
- stable / periodic / long-delayを主要環境にする。
- HGT on/off、mutation off、grammar knockout/reintroductionを必須対照にする。
- stableで欠失が再現しない場合は、欠失コスト、突然変異供給、世代数、背景感覚運動の強さを分離する。


## SOMA-CELL 0.6.7 — 凍結済み・長期物質系統計器

- 粗視化物質チェモスタットでgeneration 15〜21を達成。
- R3 12/12 finite、external fitness 0、物質残差<1e-7。
- stable loss 0/3、strict HGT rescue 1/3、long-delay差+0.388889で主要ゲートFAIL。
- HGT再侵入とHGT由来grammar子孫はHGT-ON stable 3/3で成立。
- 判定: `PARTIAL_LONG_HORIZON_HGT_REENTRY_WITHOUT_PREREGISTERED_LOSS_OR_DELAY_RETENTION`。

## 次: 0.6.8 — regime shift下の物質的進化記憶

1. stable burn-inを20世代以上継続。
2. 外部通知なしでlong-delayまたはperiodicへ物理規則を切替。
3. loss→mutation/HGT re-entry→翻訳→子孫持続を同一系統史で追跡。
4. HGT OFF、mutation OFF、no-switch、grammar absentを事前登録。
5. 再侵入遺伝子をノックアウト／再導入し、保持の因果を確認。
6. 0.6.7の粗視化結果を0.6.6粒子世界の短い確認実験で交差検証する。

## 0.6.8-GPU — フル詳細GPU移行

ユーザーの計算基盤方針により、0.6.8の主軸を同一系統史の粗視化regime-shiftから、0.6.6完全粒子世界のGPU移行へ変更した。regime-shift仮説はGPU基盤後の詳細モデル試験へ繰り下げる。

### A1 — 凍結候補engineering checkpoint

- full 0.6.6 tensor snapshot schema
- deterministic counter RNG
- particle diffusion / patch drift
- particle ligand profiles
- circular smoothing
- standalone core metabolism lockstep
- hybrid CPU-authoritative world
- batched kernel benchmark

A1はfull GPU worldではなく、CUDA実機未検証。

### A2

- surface exchange
- waste export
- leak
- radius and motion
- spatial hash / neighbor list

### A3

- core metabolismのworld.step統合
- 0.3損傷修復層
- membrane oxidation / damage segregation

### A4

- ragged genome buffers
- replication and translation
- material mutation

### A5

- division / death / corpse / eDNA / HGT

### A6

- neural tissue / causal audit
- full CPU/GPU event lockstep
- RTX 4060 Ti and online GPU scaling

### GPU後の科学試験

- 0.6.7で計画した同一系統史regime shift
- 0.6.6完全粒子物理でloss/re-entry/persistenceを再検証
