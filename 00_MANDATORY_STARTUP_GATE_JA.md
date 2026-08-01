# SOMA 必須スタートアップ・ゲート（最優先）

**このファイルは、SOMAの実装・修正・評価を始める前に必ず読む。**

会話上の記憶、直前の要約、推測だけで作業を開始してはならない。最新のプロジェクト状態と基準ソースを実際に読み、整合性を確認し、プリフライト記録を作成してから編集する。

## 失敗閉鎖ルール

次のどれか一つでも満たせない場合、実装を開始しない。

1. `PROJECT_STATE.json` を読めない。
2. `CURRENT_BASELINE.txt` を読めない。
3. `docs/SOMA_CONTEXT_HANDOFF_JA.md` を読めない。
4. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md` を読めない。
5. `PROJECT_STATE.json`が指す現在の基準ソース、固有契約、基準検証結果を読めない。
6. SHA-256照合に失敗する。
7. 現在のマイルストーンと作業目的が一致しない。
8. 前版の既知の失敗・未解決事項を確認していない。
9. `work_sessions/` に当該作業のプリフライト記録を作れない。

不足や矛盾がある場合は、推測で補わず、アーカイブまたはGoogle Driveから復元する。復元できなければユーザーへ不足を明示する。

## 必須の読み順

1. `00_MANDATORY_STARTUP_GATE_JA.md`
2. `PROJECT_STATE.json`
3. `CURRENT_BASELINE.txt`
4. `docs/SOMA_CONTEXT_HANDOFF_JA.md`
5. `docs/SOMA_CELL_0_6_INTEGRATION_CONTRACT.md`
6. 現在の基準ソース
7. 現在の基準版に固有の凍結契約（P1ではP0身体ポート契約）
8. 現在の基準版の検証結果・実験レポート
9. 次マイルストーンのロードマップ
10. 移植対象となる旧SOMAの基準ソースと検証結果

## 実装開始前の機械的確認

リポジトリ直下で次を実行する。

```bash
python3 scripts/soma_preflight.py verify
python3 scripts/soma_preflight.py start \
  --actor "<作業者名>" \
  --purpose "<今回の目的>" \
  --milestone "<PROJECT_STATEのnext_milestone名>" \
  --ack "READ_LATEST_SOURCES_AND_CONTRACTS"
```

`start` が成功すると、`work_sessions/*_PREFLIGHT.json` が生成される。この記録がない作業は、正式なSOMA作業として扱わない。

## 作業中の禁止事項

- 旧コードを、最新版確認なしでコピーする。
- 会話の説明を基準ソースより優先する。
- 0.5の物質台帳、死体化学、環境DNA、HGTを黙って落とす。
- 神経出力から位置、ATP、膜量、DNAを直接書き換える。
- 無料の報酬、無料の学習、無料の遺伝を追加する。
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

矛盾がある場合は次を優先する。

1. 検証済み基準ソースと実測結果
2. 本スタートアップ・ゲート
3. 0.6再統合契約
4. `PROJECT_STATE.json`
5. 継続ハンドオフ
6. 会話上の記憶・要約

この規則自体を会話の記憶へ依存させないため、README、PROJECT_STATE、復元プロンプト、プリフライトスクリプト、Google Driveにも重複保存する。

## ルールロックR2：古い証跡の使い回しを禁止

プリフライト証跡は、単に一つ存在すればよいのではない。最新の`*_PREFLIGHT.json`が、**現在のGit HEAD**および`PROJECT_STATE.json`の**次マイルストーン**と一致しなければならない。

```bash
python3 scripts/soma_preflight.py check-receipt
```

Git pre-commitフックは`verify`と`check-receipt`の両方を実行する。したがって、一度コミットした後に次の変更へ進む場合は、その新しいHEADであらためて最新版を読み、プリフライトを作り直す。

### コミット後の状態

プリフライト証跡は「作業を開始したHEAD」に結び付く。コミット後はHEADが変わるため、以前の証跡が古くなるのは正常である。次の作業を始める直前に、クリーンな新HEADで`start`を再実行する。最終配布物に常時「有効な作業中証跡」が入っている必要はない。むしろ、作業開始前に都度発行することが重要である。

## マイルストーン遷移ゲート

完了したマイルストーンを次の基準版へ昇格する変更も、会話記憶だけで行ってはならない。クリーンな完了コミット上で、検証結果と配布物を証拠として次を実行する。

```bash
python3 scripts/soma_preflight.py transition \
  --actor "<作業者名>" \
  --purpose "<遷移理由>" \
  --from-milestone "<完了した現在マイルストーン>" \
  --to-milestone "<次マイルストーン>" \
  --baseline-after "<昇格後の基準版名>" \
  --evidence "<検証結果ファイル>" \
  --evidence "<配布アーカイブ>" \
  --ack "COMPLETE_CURRENT_MILESTONE_AND_ADVANCE"
```

`*_TRANSITION.json`は、宣言済みの基準版と次マイルストーンへ進める一回のコミットだけを許可する。別の遷移先や証拠改変は拒否される。遷移コミット後はGit HEADが変わるため証跡は古くなり、次マイルストーンの実装前に最新版を再読して新しい`*_PREFLIGHT.json`を発行しなければならない。

## P0昇格後のP1追加ゲート

SOMA-CELL 0.6-P1を始める際は、`docs/SOMA_CELL_0_6_P0_BODY_PORT_CONTRACT.md`を必ず読み、神経組織がP0ポート以外から身体へアクセスしないことを確認する。同量・同維持費の非情報処理ダミーを主対照とし、P0の22契約検証と0.5ロックステップを回帰試験として維持する。
