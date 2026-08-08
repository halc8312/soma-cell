# SOMA-CELL 0.6.5

0.6.4の負の結果を受け、追加神経文法を既定休眠にし、5つの神経制御モジュールを通常の物質ゲノムへ移した版です。

## 何が新しいか

- sentinel/readiness/organ/prediction/plasticityを実ゲノムのdelimited geneとして保持
- whole-gene deletion/duplication
- promoterによる休眠・再活性化
- readiness/organ payloadの調節変異
- 同一seed身体世界を使う多世代lineage competition
- 実配列knockout/reintroductionによる因果確認

## 主要結果
periodicとlong_delayでは発現神経文法が3/3系列で保持され、periodicの代表遺伝型はknockoutで6/6 AUC低下、再導入で6/6回復しました。一方stableではnear-dormant化が1/3だけで、事前登録した安定環境縮小gateは失敗しました。

よって科学判定は `PARTIAL_ENVIRONMENT_DEPENDENCE_STABLE_LOSS_INCOMPLETE` です。

## Pythonista
全ファイルを同じフォルダへ置き `SOMA_CELL_0_6_5_pythonista.py` を実行してください。追加神経文法は画面上の通常個体では既定休眠です。0.6.1 Visual Fix由来のコンパクトHUDと子孫版安全リセットを継承します。
