# SOMA 必須スタートアップ・ゲート（最優先）

**このファイルは、SOMAの実装・修正・評価を始める前に必ず読む。**

会話上の記憶、直前の要約、推測だけで作業を開始してはならない。最新のプロジェクト状態、現在の基準ソース、マイルストーン固有契約、実測結果を実際に読み、SHA-256を確認し、作業証跡を作成してから編集する。

## 失敗閉鎖ルール

次のどれか一つでも満たせない場合、実装を開始しない。

1. `PROJECT_STATE.json`、`CURRENT_BASELINE.txt`、継続ハンドオフ、0.6統合契約を読めない。
2. `PROJECT_STATE.json`が指す基準ソース、身体ポート契約、神経組織契約、検証結果、実験レポートを読めない。
3. SHA-256照合に失敗する。
4. 現在のマイルストーンと作業目的が一致しない。
5. 前版の既知の失敗・未解決事項を確認していない。
6. `work_sessions/`へ当該作業の新鮮なプリフライト証跡を作れない。

不足や矛盾がある場合は推測で補わず、Git、Continuity Vault、Google Driveから復元する。復元できなければ不足をユーザーへ明示して停止する。

## 必須の読み順

1. `00_MANDATORY_STARTUP_GATE_JA.md`
2. `MANDATORY_WORKFLOW.json`
3. `PROJECT_STATE.json`
4. `CURRENT_BASELINE.txt`
5. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
6. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
7. `docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md`
8. `docs/SOMA_CELL_0_6_P1_TISSUE_CONTRACT.md`
9. `docs/SOMA_CELL_0_6_P2_TISSUE_CONTRACT.md`
10. `docs/SOMA_CELL_0_6_P2_TISSUE_SCHEMA.json`
11. `src/0_6_p2/SOMA_CELL_0_6_P2_pythonista.py`
12. P2検証結果・36試行レポート・P0/P1回帰結果
13. `planning/ROADMAP_0_6_JA.md`
14. 因果監査の移植元となるSOMA-2.1、SOMA-4.2の基準ソースと検証結果

## 実装開始前の機械的確認

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<作業者名>" \
  --purpose "<今回の目的>" \
  --milestone "<PROJECT_STATEのnext_milestone名>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

`start`が成功すると、現在のGit HEADと次マイルストーンへ結び付いた`work_sessions/*_PREFLIGHT.json`が生成される。証跡がない作業は正式なSOMA作業として扱わない。

## 正式SOMA-CELL 0.6追加ゲート

正式0.6へ入る前に、P2で確定した次の事実を凍結条件として確認する。

- P0身体ポート以外から身体へアクセスしない。
- P2は8個のゲノム由来物質区画、20本の有限再帰、局所予測、三因子可塑性、保存的再編を持つ。
- holdout全機構対固定8区画は6 seed中5 seedで正、平均累積余裕AUC差は+1.872678だった。
- 開発seedは物理トラップ、手掛かり時間、運動スケールの調整に使ったためholdoutではない。
- 局所予測は誤差を下げるが、正味身体価値は未証明である。
- 再帰の平均paired効果は+0.016963と極小で、必要性は未証明である。
- 可塑性は限定的な正の信号を示したが、holdout単独アブレーションは未実施である。
- 正式0.6は予測・再帰を成功済み前提にせず、必ず`no prediction`と`no recurrence`を残す。
- 因果監査は神経細胞群への信号・ATP流を可逆遮断して行い、位置・ATP・膜・DNAを直接変更しない。
- ABBA/BAAB順序、偽対照、接触汚染分離、証拠品質、費用台帳を実装する。
- 因果効果が校正され、監査・再可塑化の純利益が示されるまで正式0.6を完成扱いにしない。

## 作業中の禁止事項

- 旧コードを最新版確認なしでコピーする。
- 会話の説明を基準ソースより優先する。
- 0.5の物質台帳、死体化学、環境DNA、HGT、0.6-P0の身体ポート契約を黙って落とす。
- P1/P2の負の結果を無視して、予測や再帰を有効と断定する。
- 神経出力から位置、ATP、膜量、DNAを直接書き換える。
- 無料の報酬、学習、神経物質、遺伝を追加する。
- 検証前に「生命」「意識」「新規性」「適応利益」を断定する。

## 作業完了時の必須更新

1. `PROJECT_STATE.json`
2. `PROJECT_STATUS_JA.md`
3. `TEST_MATRIX.csv`
4. `DECISION_LOG_JA.md`
5. `CHANGELOG_JA.md`
6. SHA-256とファイル台帳
7. Gitコミットと版タグ
8. Continuity Vault
9. Google DriveのポータブルZIP、Git Bundle、進捗文書

## 権威順位

1. 検証済み基準ソースと実測結果
2. P2神経組織契約、P1神経組織契約、P0身体ポート契約
3. 本スタートアップ・ゲート
4. 0.6再統合契約
5. `PROJECT_STATE.json`
6. 継続ハンドオフ
7. 会話上の記憶・要約

## ルールロックR2

プリフライト証跡は現在のGit HEADと`PROJECT_STATE.json`の次マイルストーンに一致しなければならない。

```bash
python3 scripts/soma_preflight.py check-receipt
```

pre-commitフックは`verify`と`check-receipt`を実行する。一度コミットすると旧証跡は意図的に失効するため、次の作業前に新HEADで最新版を読み直し、新証跡を発行する。

## マイルストーン遷移ゲート

完了版の昇格には、クリーンな完了コミット上で一回限りの遷移証跡を作る。

```bash
python3 scripts/soma_preflight.py transition \
  --actor "<作業者名>" \
  --purpose "<遷移理由>" \
  --from-milestone "<完了マイルストーン>" \
  --to-milestone "<次マイルストーン>" \
  --baseline-after "<昇格後基準名>" \
  --evidence "<検証結果>" \
  --evidence "<配布アーカイブ>" \
  --ack "COMPLETE_CURRENT_MILESTONE_AND_ADVANCE"
```

遷移コミット後は証跡が失効し、P2実装前にあらためてP2プリフライトが必要である。
