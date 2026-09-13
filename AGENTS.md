# SOMA 現在の作業指示 — 2026-09-13修復枝

最初に `README.md`、`REPAIR_STATE.json`、`PROJECT_STATE.json`、`REPAIR_R0_README_JA.md` を読む。
**この修復枝ではA3を直ちに開始しない。** 古い引き継ぎや歴史的READMEより、このファイルとREPAIR_STATEを現在作業の指示として優先する。

- 現在段階: SOMA-REPAIR R0。
- 次: R1。計画は `planning/REPAIR_PRIORITY_JA.md`。
- 旧A2/A3および0.6.xのソースとタグは履歴。不整合を既知としたうえで保存し、勝手に上書きしない。
- 修復差分は `audits/20260913/grammar.patch` と `audits/20260913/particle.patch` に記録。
- R0の専用検査結果は `results/repair_r0/green/results.json`。
- Google Drive停止指示を維持。クラウドGPU契約・費用発生は新たな承認なしに行わない。
- RTX 4060 Ti 16GBを使う最終方針は維持。ただし本段階は参照モデルの修復。

R0の主張は限定修復の検証だけ。旧不具合の再現・30/30の専用検査を、過去の343件へ足して総保証件数にしない。

R1では同じ物質予算・同じ観測機会・同じ身体法則を揃え、同一状態から学習ON/OFFだけを分岐する。生存・繁殖・子孫は評価へ含めても、個体への外部報酬としては渡さない。

Codex移行後のユーザーPC側変更を強制resetしない。mainへ統合する前に差分を確認し、必要ならrepair branch上でrebase/mergeする。
