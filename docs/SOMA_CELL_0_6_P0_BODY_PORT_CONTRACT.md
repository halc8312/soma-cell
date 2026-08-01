# SOMA-CELL 0.6-P0 Chemical Body Port Contract

版: `0.6-P0.1`  
対象実装: `src/0_6_p0/SOMA_CELL_0_6_P0_pythonista.py`  
詳細仕様: `SOMA_CELL_0_6_P0_API_CONTRACT.md`

## 1. 目的

将来の物質神経組織がSOMA-CELL 0.5化学身体へ接続する際に、身体状態を特権的に直接変更できないよう、読み取り、予算、物理エフェクタ、死組織返却の境界を固定する。

## 2. 不変条件

1. ポート未使用時の物理軌道とRNGは凍結0.5と一致する。
2. センサー呼出しは状態とRNGを変更しない。
3. センサーは実化学量のみを返し、reward・fitness・正解行動を返さない。
4. 配分予算は実在量、時間率、身体予備量を超えない。
5. 未使用予算は同じ物質プールへ保存的に返却できる。
6. 組織材料の組立にはATPを支払う。
7. 神経要求は既存の身体物理を通し、位置・膜・ATP・DNAを直接書き換えない。
8. 組織死は物質返却として扱い、抽象記憶・遺伝子を捏造しない。
9. 宿主死亡・分裂前に組織物質を回収し、娘へ無料コピーしない。
10. 保存復元・ログで決定論を壊さない。

## 3. 読み取りAPI

```python
frame = port.raw_sensor_fluxes(tissue_id)
```

返値は再帰的な読み取り専用スナップショット。絶対位置、世界／細胞年齢、`reward`、`fitness`、`correct_action`、`food_direction`、`autopoietic_margin`は含めない。

## 4. 予算API

```python
grant = port.allocate_budget(tissue_id, request, dt)
returned = port.return_unused_budget(tissue_id)
built = port.commit_material(...)
```

区分は`atp`, `protein`, `membrane`, `signal`。実在量、身体予備、時間当たり上限に制約され、組立にもATPを払う。

## 5. エフェクタAPI

許可: `motor`, `transporter_polarity`, `repair_polarity`, `quiescence`。

禁止: 位置、速度、膜量、ATP、物質プール、DNA、ゲノム、配列、cell IDの直接書換え、および未知命令。

## 6. 死組織API

```python
result = port.return_dead_tissue(tissue_id, reason)
```

組織タンパク質・膜・信号・毒を既存の身体分解／前駆体／廃棄物プールへ返し、残存ATPは散逸させる。

## 7. P1への拘束

P1の1神経細胞はこのポート以外から身体へアクセスしてはならない。同量・同費用の非情報処理ダミーと比較し、学習済み状態や組織を娘へPythonオブジェクトとして無料コピーしない。
