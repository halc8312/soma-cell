# SOMA系譜インデックス

このVaultは、会話の記憶ではなく**検証済みソースを基準に0.6統合を再現する**ためのものです。

| 段階 | 基準ファイル | 統合時に使う要素 |
|---|---|---|
| SOMA-0 | `canonical/SOMA_0_pythonista.py` | 単体身体学習の原型 |
| SOMA-1 | `canonical/SOMA_1_pythonista.py` | 疎な再帰組織、局所予測、三因子可塑性、細胞代謝 |
| SOMA-2 | `canonical/SOMA_2_pythonista.py` | 神経群の因果監査、形態仮説 |
| SOMA-2.1 | `canonical/SOMA_2_1_pythonista.py` | ABBA/BAAB監査、証拠品質、可撤回リース |
| SOMA-3 | `canonical/SOMA_3_pythonista.py` | 内生的な問い生成 |
| SOMA-4 | `canonical/SOMA_4_pythonista.py` | 反証コンパイラ |
| SOMA-4.1 | `canonical/SOMA_4_1_pythonista.py` | 共通乱数ツイン評価 |
| SOMA-4.2 | `canonical/SOMA_4_2_pythonista.py` | 変化監視、因果校正、再可塑化ゲート |
| SOMA-5R | `canonical/SOMA_5R_pythonista.py` | 状態依存内的計画、概念、思考費用 |
| SOMA-6R | `canonical/SOMA_6R_pythonista.py` | 共有世界、局所文化、信頼校正 |
| SOMA-7R 1.0.2 | `canonical/SOMA_7R_1_0_2_pythonista.py` | 資源回復性を持つ生態系・系譜 |
| SOMA-CELL 0.1 | `canonical/SOMA_CELL_0_1_pythonista.py` | 動的膜、物質的自己生産、構造的死 |
| SOMA-CELL 0.2 | `canonical/SOMA_CELL_0_2_pythonista.py` | 物質ゲノム、内部複製、遺伝可能変異、反応辺進化 |
| SOMA-CELL 0.3 | `canonical/SOMA_CELL_0_3_pythonista.py` | 物質損傷、費用付き修復、複製精度、損傷分配、生活史 |
| SOMA-CELL 0.4 | `canonical/SOMA_CELL_0_4_pythonista.py` | 物質受容体、ATP支払い膜運動、輸送体極性、自己生産由来可塑性 |
| SOMA-CELL 0.5 | `canonical/SOMA_CELL_0_5_pythonista.py` | 空間死体、環境DNA、物質HGT、死体利用、可動配列と宿主費用 |

## 置き換え済みの版

旧SOMA-5、6、7（Rなし）は統合元にしません。運動学習の巻き戻し、個別世界を共有世界のように扱う問題、短すぎる進化試験などがあり、5R〜7Rで再構築されています。

SOMA-7Rは1.0.0/1.0.1ではなく、資源ゼロからの再生不能問題を修正した1.0.2を使います。

## 0.6へ戻す順序

```text
SOMA-CELL 0.5化学身体
  ↓
0.4の物質受容体・膜エフェクタ
  ↓
SOMA-1の8〜12細胞小型神経組織
  ↓
SOMA-2/2.1の実停止因果監査
  ↓
SOMA-4/4.1/4.2の反証・ツイン・変化ゲート
  ↓
神経死体・神経遺伝子の物質HGT検査
  ↓
純利益が示された後だけ5R、6R、7Rを段階追加
```

詳細は`SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`を優先してください。

## SOMA-CELL 0.6-P0 — 化学身体契約層

- 基準身体: **SOMA-CELL 0.6-P0.0（凍結済み）**。内部物理はSOMA-CELL 0.5.0を変更しない。
- 新規: 読み取り専用実化学センサー、有限ATP／材料予算、ATP有料組立、物理エフェクタ要求、死組織返却
- 互換性: 未接続・Null接続・予算往復で凍結0.5と状態／RNG完全一致
- 検証: 22/22 PASS、工学比較15試行
- 限定: 情報処理神経・学習・因果監査は未導入
- P1への回帰基盤として継続


## SOMA-CELL 0.6-P1 — 1物質神経区画

- 基準: **SOMA-CELL 0.6-P1.0（凍結済み）**
- 組織契約: `docs/SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md`
- 組織スキーマ: `0.6-P1.1`
- 新規: 16記号発生カセット、1物質神経区画、局所実化学入力、漏れ積分、局所予測、P0有料エフェクター
- 主対照: 同物質・同維持費・非情報処理ダミー
- 検証: P1 18/18 PASS、P0回帰22/22 PASS、24比較試行
- 結果: moving-patchで神経が等費用ダミーを3/3 seedで上回る
- 限定: 局所予測器の身体利益は未証明
- 次: **0.6-P2**で等物質8細胞固定組織から段階的に再帰・予測・可塑性を追加

## SOMA-CELL 0.6-P2 — 8細胞物質再帰組織

- 基準: **SOMA-CELL 0.6-P2.0（凍結済み）**
- 組織契約: `docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md`
- 組織スキーマ: `0.6-P2.2`
- 新規: 8物質神経区画、20有限再帰、局所予測、三因子可塑性、発達安定期間、分子エピソード、成熟・再編
- 検証: P2 22/22、P1回帰18/18、P0回帰22/22、36比較試行
- 主結果: holdout全機構対固定8区画は5/6正、平均AUC +1.872678
- 限定: 予測の正味価値未証明、再帰効果極小、holdout n=6
- 次: **正式SOMA-CELL 0.6**で物質的因果監査・校正・変化ゲート・神経死体/HGT

### SOMA-CELL 0.6
P2の8区画物質神経組織へ、物質的ABBA/BAAB監査、偽対照、因果校正、変化源分離、期限付き資源リース、小型反証、神経死体・神経遺伝子HGTを統合。工学PASS、科学的自動因果循環はPASS_WITH_LIMITS。

### SOMA-CELL 0.6.1

証拠不足台帳、有料運動器校正、神経ATP／信号エスクロー修正、正の運動仕事に基づく対象選択、期限付き故障時神経休眠。R2事前登録で安定誤リース0/20、故障身体余裕AUC正12/12。自然空間取得、意味診断、予測、再帰の純価値は未確立。

