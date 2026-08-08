# SOMA-CELL 0.6.7 長期物質系統契約

## 目的
0.6.6の完全粒子共有生態系を変更せず、generation 15〜20以上の遺伝子喪失・物質HGT再侵入・環境依存保持を調べる粗視化長期チェモスタットを第二の研究装置として追加する。

## 凍結公理
1. 個体へfitness、reward、正解方向、規則切替通知を渡さない。
2. 集団頻度は物質取込、ATP代謝、DNA複製、実分裂、死、neutral outflow、変異、HGTだけで変わる。
3. ゲノム・途中コピー鎖・翻訳済み神経モジュールタンパク質は物質台帳へ含める。
4. 遺伝子重複はnucleotideとATPを消費し、欠失はpolymer物質をnucleotideへ返す。
5. HGTはeDNA断片を実配列へ組み込み、取得直後は機能せず、構造物質とATPを使う翻訳後にだけ機能する。
6. 娘へ学習状態・予測状態・環境履歴を無料コピーしない。
7. neutral transferは遺伝型を参照せず、流出・lysis・eDNA化を物質会計へ含める。
8. long-delay cueは将来ラベルではなく、外部reservoirから入る会計済みの化学物質である。
9. 本装置は0.6.6粒子物理の代替ではなく、長期世代用の粗視化計器である。

## R3合格判定
- 長期世代・物質保存: PASS。
- stable HGT-OFF完全喪失: FAIL。
- stable HGT救済: FAIL（厳密ペア1/3）。
- long-delay保持: FAIL（差+0.388889、ペア2/3）。

## 科学判定
`PARTIAL_LONG_HORIZON_HGT_REENTRY_WITHOUT_PREREGISTERED_LOSS_OR_DELAY_RETENTION`

物質HGT再侵入と持続子孫は成立したが、適応的救済、stable完全喪失、long-delay固有保持の事前登録条件は満たしていない。
