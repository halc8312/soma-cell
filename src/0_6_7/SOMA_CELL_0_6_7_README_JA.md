# SOMA-CELL 0.6.7
## 長期物質serial-transfer生態系

状態: **engineering PASS / scientific PARTIAL_LONG_HORIZON_HGT_REENTRY_WITHOUT_PREREGISTERED_LOSS_OR_DELAY_RETENTION**

### 何を追加したか
0.6.6の完全粒子共有生態系をそのまま凍結し、generation 15〜21を現実的な計算時間で調べる粗視化物質チェモスタットを追加した。ゲノム、途中コピー、翻訳済みモジュール、ATP、構造物質、死体、eDNA、HGT、neutral outflowを明示的に会計する。

### 何を追加していないか
fitness、reward、正解方向、環境切替時刻、成績上位系統の外部コピーは追加していない。

### R3結果
- 12/12登録試行を報告。
- 全条件3/3でgeneration >=15、generation 20/21も観測。
- 物質保存・有限値・外部fitness 0: PASS。
- stable完全喪失: FAIL（0/3が厳密条件）。
- stable HGT救済: FAIL（厳密条件1/3）。
- long-delay保持: FAIL（active差+0.388889、要求+0.50）。
- ただしHGT-ONの全stable系列でHGT再侵入とgrammarを保持するHGT由来子孫が観測された。

### 正確な解釈
0.6.7は、外部fitnessなしで長期世代と物質HGT再侵入が成立することを示した。一方、不要環境での完全遺伝子喪失、HGTによる適応的救済、long-delay固有の保持は再現性が不足した。

### Pythonista
`SOMA_CELL_0_6_7_pythonista.py`を実行する。コンパクトHUDと安全なダブルタップ子孫版リセットを継承する。

### iPhone上の注意
この長期計器は粗視化されているが、端末のFPS、温度、長時間メモリは実機で確認する必要がある。
