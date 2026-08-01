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
- 次: **0.6-P1**で1個の物質神経細胞と同量・同維持費ダミーを比較
