# 社内ナレッジ検索AIシステム 仕様

更新日: 2026-07-02
リポジトリ: `https://github.com/naokoni24/rag-search.git`
ブランチ: `master`（GitHub側は `main`）
対象ファイル: `app.py` / `rag_core.py`
本番URL: `https://naokoni-rag-search.hf.space`（Hugging Face Space。iframe無しの直接URL）

## 運用ルール

このプロジェクトを修正する時は、毎回このObsidianノートを読み込んでから作業します。

作業後は以下を行います。

- 実装内容に合わせてこの `ai-search.md` を更新
- リポジトリ内の `ai-search.md`（このファイル自身のコピー）も同様に更新
- `rag-search.git` の `master` ブランチへpush（GitHub Actionsが自動でHugging Face Spaceに同期する）

push先は必ず `https://github.com/naokoni24/rag-search.git` です（旧 `rag-sample.git` はStreamlit版の名残で、現在は使用していません）。

## 概要

社内ナレッジ検索AIシステムは、社内のPDF文書（規定・マニュアル等）をAIで検索し、
質問に対して出典付きで回答するWebアプリです。

中小企業の社員が、社内規定・マニュアル・規則などを自然な日本語で質問するだけで、
該当箇所と出典PDF・ページ番号を提示しながら回答します。

**2026-07-02にStreamlitからFlask + htmxへ全面移行しました。** 理由は、Streamlitでは
`<head>`を制御できず、iPhoneの「ホーム画面に追加」用アイコン（apple-touch-icon）を
確実に設定できなかったため。あわせてホスティングもStreamlit Community Cloudから
Hugging Face Spaces（Docker）へ移行しました。

## 基本構成

- フレームワーク: Flask + Jinja2 + htmx（部分更新、`static/htmx`はCDN読み込み）
- AI（埋め込み・回答生成）: Google Gemini
  - 埋め込みモデル: `gemini-embedding-001`（ベクトル次元 3072）
  - 生成モデル: `gemini-2.5-flash`
- ベクトルDB: Qdrant
  - ローカル開発: `./qdrant_data`（ファイルベース、`QDRANT_URL`未設定時のフォールバック）
  - 本番: Qdrant Cloud（`QDRANT_URL` / `QDRANT_API_KEY` 必須）
- ホスティング: Hugging Face Spaces（Docker SDK、Private、CPU Basic無料枠）
  - デプロイ経路: GitHubの`main`にpush → GitHub Actions（`.github/workflows/sync-to-hf-space.yml`）が`HF_TOKEN`を使って自動でSpaceにforce push
  - アプリサーバー: gunicorn（`--workers 1 --threads 4 --timeout 600`。ワーカーを1つに絞っているのはプロセス内メモリキャッシュを共有するため）
- PDF処理: PyMuPDF（`fitz`）
- Markdown整形: `markdown`ライブラリ（GeminiはMarkdown箇条書きの前に空行を入れないことが多く、`_normalize_markdown()`で補正してから変換）

## 環境変数（Hugging Face SpaceのSettings → Variables and secretsで設定）

| 変数 | 必須 | 内容 |
| --- | --- | --- |
| `GEMINI_API_KEY` | 必須 | Gemini APIキー（埋め込み・回答生成） |
| `QDRANT_URL` | 必須（本番） | Qdrant CloudのURL。**末尾に`:6333`のポート番号が必要**（無いと接続がハングする） |
| `QDRANT_API_KEY` | 必須（本番） | Qdrant Cloud APIキー |
| `ADMIN_PASSWORD` | 文書管理に必須 | 「文書を管理」タブのログインパスワード |
| `FLASK_SECRET_KEY` | 必須 | セッションCookie署名用のランダム文字列 |

`get_secret()`（`rag_core.py`）で`os.getenv()`から取得し、**前後の空白・改行を自動でstrip**する
（コピペ時に混入した改行がQdrantクライアントの「Illegal header value」エラーの原因になったため）。

## データモデル（Qdrantコレクション）

| コレクション | 用途 | ベクトル |
| --- | --- | --- |
| `documents` | PDFチャンク本文（filename, page, text） | 3072次元（Gemini埋め込み） |
| `pdf_files` | PDF本体（base64で保存、ダウンロード・再表示用） | ダミー1次元 |
| `search_logs` | 検索ログ（クエリ正規化キーごとの件数・ラベル） | ダミー1次元、単一ポイントに集約保存 |

- チャンクサイズ: 400文字、オーバーラップ80文字
- 検索ログには回答本文・チャンク本文は保存しない（機密情報保護のため件数とラベルのみ）
- StreamlitのStreamlit版から移行時、Qdrant Cloudのデータ・コレクション構造はそのまま流用（データ移行不要）

## 画面構成

Flaskルーティングでタブを分離: `/`（検索） / `/manage`（管理）

### UIデザイン方針

- 2026-07-02にUIを「ミニマルSaaS + プレミアムライト」寄りに調整
  - 白〜淡いグレー基調、青アクセント、控えめなボーダーと影で、業務ツールとしての信頼感を優先
  - ヘッダー・タブ・検索フォーム・回答カード・管理パネルを半透明の軽い面表現に統一
  - 操作UIは絵文字主体の表現を抑えつつ、検索結果の会話アバターは人・ロボット絵文字で質問/回答の区別を維持
  - 回答本文は読みやすさを優先し、通常ウェイト・広めの行間・控えめな左アクセントのカードで表示
  - フォントはMac/iOSで自然に見えるシステムフォント寄りにし、本文・ボタン・ラベルのサイズ階層を控えめに統一
  - 検索画面は中央の検索体験を主役にし、管理画面は登録済みドキュメント一覧を広めに取る2カラム配置
  - `style.css` は更新時刻をクエリ文字列に付けて読み込み、ブラウザ/Space側のCSSキャッシュでUI変更が反映されない問題を防止
  - モバイル幅ではタブ・検索フォーム・ドキュメント行が横スクロールしないようレスポンシブ調整済み

### 文書を検索（`/` , `/search`）

- 自然言語で質問を入力 → htmxで`/search`にPOST、`#search-results`を部分更新
- 処理フロー:
  1. `sanitize_query` でプロンプトインジェクション・長文を検査
  2. `expand_query` でクエリを検索向けに書き換え（元クエリと併用）
  3. 元クエリ・拡張クエリ両方で `documents` をベクトル検索し、スコア降順・閾値0.55以上をマージ
  4. ヒットしたチャンクをコンテキストとして `generate_answer` で回答生成
     - 参考ドキュメントに回答がない場合は「【参照】」を付けないようプロンプトで明示（付けると存在しない参照文書が表示されてしまうバグがあったため）
  5. 回答末尾の `【参照】ファイル名 p.ページ番号` を検出し、`linkify_answer` でリンク表示
     - **PDFダウンロードリンクを表示**（`allow_download=True`。Space自体がHugging Face認証で保護されているため、Streamlit版にあった「検索タブは非認証だからダウンロード禁止」という制限は撤廃）
  6. 引用付きの有効回答のみ `record_search` でログ記録
- サーバー全体で共有するTTLキャッシュ（`_answer_cache`、24時間）で同一クエリの再生成を回避
  - Streamlit版はブラウザセッション単位のキャッシュだったが、Flaskはgunicornワーカー1プロセスなのでプロセス全体で共有する方式に変更（他ユーザーの同じ質問にも即答できる）
  - PDFの登録・削除時（`ingest_pdf` / `delete_document`）に`_answer_cache`を明示的にクリアし、TTLを長く取りつつもドキュメント更新後は即座に反映されるようにしている
- 「よく検索されています」（Top3ピル）: 検索ログ上位3件を表示。クリックすると検索欄にも同じ文言が入る。ログが無い場合は静的サジェスト4件をフォールバック表示

### 文書を管理（`/manage`、管理者ログイン必須）

- `ADMIN_PASSWORD` によるログイン（5回失敗でロック、30分操作なしで自動ログアウト）
- セッションはFlaskの署名付きCookie。`SESSION_COOKIE_SAMESITE=None` + `SESSION_COOKIE_SECURE=True`
  （Hugging FaceのSpaceページ経由だとiframe埋め込みになりCookieがブロックされるためNoneが必要。本番URL `https://naokoni-rag-search.hf.space` を直接開けばiframeを回避できる）
- PDFアップロード（最大10MB、最大50件登録）
  - 同名ファイル再登録時は既存チャンクを削除してから再登録
  - テキスト抽出0件（スキャンPDF等）は警告表示
  - 登録中の例外は画面にエラーメッセージとして表示し、サーバーログ（`app.logger.exception`）にも記録（以前は例外を握りつぶしていて原因不明のまま失敗していた）
- 登録済みドキュメント一覧（登録日時降順、htmxで部分更新）
  - チェックボックスで複数選択 → 一括削除（`documents` / `pdf_files` 両方から削除）

## セキュリティ対策

- プロンプトインジェクション対策: `_INJECTION_PATTERNS` で既知パターンを検出して拒否、`` ``` ``/`###`/`---` を除去
- システムプロンプトとユーザー入力を明確に分離し、ドキュメント外の情報での回答を禁止
- Hugging Face Spaceは Private 設定（閲覧・pushとも自分のHFアカウントのみ）。社内共有する場合はOrganization招待が別途必要
- 管理者パスワードの総当たり防止（試行回数制限）とセッションタイムアウト
- Secrets（APIキー等）はHugging Face SpaceのRepository secretsで管理。GitHub側は`HF_TOKEN`（Space同期専用）のみをActions secretsに保持

## デプロイ手順

1. コードを修正し、ローカルの`.env`（`GEMINI_API_KEY`, `ADMIN_PASSWORD`, 任意で`QDRANT_URL`/`QDRANT_API_KEY`）で動作確認
2. `git push rag-search-github master:main`（このリモートは `https://github.com/naokoni24/rag-search.git`）
3. GitHub Actions（`sync-to-hf-space.yml`）が自動的に`https://huggingface.co/spaces/naokoni/rag-search`へforce push
4. `app.py`やDockerfileの変更はイメージ再ビルドが走るため反映まで数分かかる。コードのみの変更は比較的早い

## 既知の課題 / 今後の検討事項

- スキャンPDF・画像PDFは非対応（OCR未対応）
- 検索ログは単一ポイントに集約保存のため、同時書き込み時はマージ処理で競合緩和しているが完全ではない
- `MAX_DOCS = 50` 件、`MAX_UPLOAD_MB = 10` はハードコード（運用規模に応じて見直しの可能性）
- gunicornを`--workers 1`にしているため、同時アクセスが多い場合はスレッド（4つ）の範囲でしか並列処理できない。将来的にアクセスが増えたらQdrant Cloud接続を前提に`--workers`を増やすことを検討（ローカル埋め込みQdrantはマルチプロセスでファイルロック競合を起こすため増やせないが、本番は外部Qdrant Cloudなので制約なし）
- Hugging Face Spacesの無料枠はコンテナのファイルシステムが再起動のたびに初期化されるが、検索データはQdrant Cloud（外部）にあるため影響なし
