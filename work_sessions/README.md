# Work Session Receipts

`scripts/soma_preflight.py`が、実装と状態遷移の前に読み込み・整合性確認を行った証跡を保存する。

- `*_PREFLIGHT.json`: 作業開始時の基準版、必須ファイルSHA-256、目的、マイルストーン、Git HEAD。
- `*_TRANSITION.json`: 完了版を次の基準版へ昇格させる一回限りのコミット許可。検証／配布証拠のSHA-256を含む。
- `*_FINAL.json`: 作業完了時の検証、コミット、成果物、未解決事項。

プリフライトまたは正当な遷移証跡がない変更は正式版へマージしない。どちらの証跡もGit HEADに結び付き、コミット後は古くなる。
