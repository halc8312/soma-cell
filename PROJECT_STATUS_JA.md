# SOMAプロジェクト進行状況

更新日: 2026-08-08

## 現在地

**SOMA-CELL 0.6.5を工学的PASS・科学的PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETEの凍結基準へ昇格。**

```text
Baseline: SOMA-CELL 0.6.5
Canonical: src/0_6_5/SOMA_CELL_0_6_5_pythonista.py
Schema: 0.6.5-GE1.0
Engineering: PASS
Science: PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETE
Next: SOMA-CELL 0.6.6
```

## 検証

```text
0.6.5専用: 24/24 PASS
継承回帰: 202/202 PASS
合計: 226/226 PASS
```

## 0.6.5の科学結果

物質ゲノム上に`sentinel / readiness / organ / prediction / plasticity`の5モジュールを置き、欠失、重複、プロモータ休眠・再活性化、調節変異を許可した。追加神経文法は既定休眠で、学習済み状態は遺伝しない。

3世代×8個体×3系列の加速系統アッセイでは、periodicとlong_delayで発現神経文法が3/3系列保持された。stableでは近休眠まで縮小したのは1/3系列だけで、事前登録したstable shrinkage gateはFAILした。

```text
stable: active modules mean 3.042, organ freq 0.708, near-dormant 1/3
periodic: active modules mean 4.625, organ freq 1.000, retained 3/3
rare_fault: active modules mean 3.042, organ freq 0.708
long_delay: active modules mean 4.625, organ freq 1.000, retained 3/3
periodic - stable active modules: +1.583
```

periodicで選ばれた候補ゲノムは、未調整6系列で実配列ノックアウトにより身体余裕AUCが6/6低下し、完全文法再導入で6/6回復した。最大物質残差は約5.475e-06。

ただし、これは実験者側の加速系統トーナメントであり、共有世界の自然な分裂・死亡・競争だけによる選択ではない。そのため完全な環境依存自然選択とは主張しない。

## 次: SOMA-CELL 0.6.6

加速トーナメントをやめ、有限資源を共有する化学生態系で実際の分裂、構造的死、死体、環境DNA、HGTを通して神経文法頻度を追跡する。stableでの欠失が不完全だった理由を解き、complex環境での保持とstableでの縮小が自然集団でも再現するかを検証する。
