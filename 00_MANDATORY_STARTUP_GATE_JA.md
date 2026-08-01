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
9. `docs/SOMA_CELL_0_6_P1_TISSUE_SCHEMA.json`
10. `src/0_6_p1/SOMA_CELL_0_6_P1_pythonista.py`
11. P1検証結果・実験レポート
12. `planning/ROADMAP_0_6_JA.md`
13. 移植対象となる旧SOMAの基準ソースと検証結果

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

## SOMA-CELL 0.6-P2追加ゲート

P2へ入る前に、P1で確定した次の事実を凍結条件として確認する。

- P0身体ポート以外から身体へアクセスしない。
- P1の1神経区画は、等物質・等維持費ダミーよりmoving-patch環境で3/3 seedの純利益を示した。
- P1局所予測器は予測RMSを改善したが、予測なし条件より身体利益が高いとは示せなかった。
- P2は予測の有用性を前提にせず、必ず`no prediction`を比較する。
- P2の主対照は、等物質の固定8細胞組織とする。
- 必須アブレーションは、神経なし、再帰なし、予測なし、可塑性なし、等物質固定組織。
- 8細胞化によるATP・膜材・タンパク質・信号分子・損傷費をすべて台帳へ含める。
- 学習済み数値状態を娘へ無料コピーしない。

## 作業中の禁止事項

- 旧コードを最新版確認なしでコピーする。
- 会話の説明を基準ソースより優先する。
- 0.5の物質台帳、死体化学、環境DNA、HGT、0.6-P0の身体ポート契約を黙って落とす。
- P1の負の結果を無視して、予測機構を有効と断定する。
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
2. P1神経組織契約とP0身体ポート契約
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
