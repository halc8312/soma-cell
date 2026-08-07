# SOMAプロジェクト進行状況

更新日: 2026-08-08

## 現在地

**SOMA-CELL 0.6.4を工学的PASS・科学的NEGATIVE_BOUNDARY_RESULT_WITH_LIMITSの凍結基準へ昇格。後から2区画神経器官を作るには完全前駆体の95%を需要判明前に予約する必要があり、条件付き器官は全ての事前登録対照へ平均で敗れた。**

```text
Baseline: SOMA-CELL 0.6.4
Canonical: src/0_6_4/SOMA_CELL_0_6_4_pythonista.py
Schema: 0.6.4-AB1.1
Engineering: PASS
Science: NEGATIVE_BOUNDARY_RESULT_WITH_LIMITS
Next: SOMA-CELL 0.6.5
```

## 検証

```text
0.6.4専用: 25/25 PASS
継承回帰: 177/177 PASS
合計: 202/202 PASS
登録試行: 69
```

## 科学結果

```text
最小物理発生オプション: 0.95
stable誤完全準備/誤発生: 0/12, 0/12
conditional - option-no-tissue: -0.918171, 正1/6
conditional - bare: -2.394640, 正0/6
conditional - eager: -1.136416, 正1/6
conditional - always-efficient2: -1.794204, 正2/6
conditional - random: -1.280801, 正1/6
```

## 次: SOMA-CELL 0.6.5

追加神経文法を既定休眠にし、センチネル、準備率、器官発生、予測、可塑性の物質遺伝子が多世代選択で保持・縮小・欠失するかを測る。神経を有利に調整せず、費用を払えない環境では自然に失われることを許容する。
