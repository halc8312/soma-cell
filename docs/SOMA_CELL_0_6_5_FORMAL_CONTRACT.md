# SOMA-CELL 0.6.5 多世代神経文法経済・遺伝保持契約

## 目的
0.6.4では条件付き神経器官が必要材料の95%を早期予約しなければ成立せず、登録比較をすべて失った。0.6.5は追加神経文法を既定休眠とし、センチネル・準備率・器官発生・予測・可塑性を通常の物質ゲノムへ置き、環境別の多世代選択で保持・休眠・欠失・重複が起きるかを測る。

## 凍結公理
1. 0.6.4までの膜、代謝、物質ゲノム、死体、eDNA、HGT、神経台帳を壊さない。
2. 神経文法は通常のdelimited geneであり、ゲノム長と複製材料を持つ。
3. 学習済み重みや器官状態を遺伝子として無料コピーしない。
4. 文法変異はwhole-gene deletion/duplication、promoter dormancy/reactivation、payload regulationとして実配列を変更する。
5. organismへreward、fitness、正解方向、環境変更時刻を渡さない。
6. 選択実験は同じ環境seedで異なる遺伝型を比較し、遺伝型ごとに別の幸運なseedを割り当てない。
7. ノックアウトと再導入は実配列の欠失・再挿入で行う。

## 遺伝モジュール
- sentinel
- readiness
- organ
- prediction
- plasticity

promoter 0–1は調節休眠、2以上を発現候補とする。readiness payloadは0–1.0の物質予約率、organ payloadは欠失／条件付き2区画／構成的2区画を表す。

## 多世代試験
凍結core metabolismは変更せず、追加神経文法だけにwhole-gene変異を許すaccelerated lineage assayを用いる。各遺伝型の選択値は、同じseedのSOMA-CELL身体世界を実行して得る費用込みautopoietic-margin AUCである。これはorganism内部のrewardではなく、実験者側の世代交代測定である。

環境:
- stable
- periodic physical rule change
- rare actuator fault
- long-delay cue/reward chemistry

対照:
- grammar absent
- regulatory-dormant grammar
- full grammar
- mutation-disabled clonal full grammar
- HGTなし
- unchanged stable environment

## 事前登録ゲート
- stable: 3系列中2以上で最終平均active grammar modules <=1.5
- complex retention: periodicまたはlong_delayで3系列中2以上がorgan frequency >=0.6かつactive modules >=2.5
- retained complex envとstableのactive-module差 >=1.0
- retained genotype knockoutでholdout 4/6以上AUC低下
- full grammar再導入でholdout 4/6以上knockoutより改善
- 全body assay有限、|material residual| <3e-5

## 結果
- 専用検証 24/24 PASS。
- 親0.6.4 dedicated 25/25をfresh rerun、0.6.3以前177/177 frozen regressionを維持。継承計202/202。
- stable final active modules: 平均3.042、near-dormant shrinkageは1/3のみで事前登録gate FAIL。
- periodic: active平均4.625、organ frequency 1.000、retention 3/3。
- long_delay: active平均4.625、organ frequency 1.000、retention 3/3。
- periodic-stable差 +1.583。
- periodic retained genotypeの実配列knockout: AUC低下 6/6。
- full grammar再導入: knockoutより改善 6/6。
- 最大物質残差 約5.475e-6。

したがって判定は `PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETE` とする。

## 科学的限定
複雑環境での保持と因果的価値は短い登録試験で支持されたが、安定環境での一貫した縮小・欠失は3系列中1系列しか成立せず、完全な環境依存選択gateは不通過である。これは自然な長期共有生態系でのダーウィン進化、オープンエンド進化、生命・意識の証明ではない。
