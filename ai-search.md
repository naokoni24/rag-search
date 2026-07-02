# 社内ナレッジ検索AIシステム 仕様

更新日: 2026-07-03
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
- ホスティング: Hugging Face Spaces（Docker SDK、Public、CPU Basic無料枠）
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
| `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` | 本番で必須 | アプリ全体を保護するHTTP Basic認証。両方未設定ならローカル開発用に認証をスキップ |

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

- 2026-07-03にUIを「モダンSaaS」方向へ再整理
  - Linear / Notion / Vercelに近い白基調の中密度UIへ寄せ、インディゴ（`#4F46E5`）をアクセントとして統一
  - `static/style.css` のカラー・角丸・シャドウをトークン化し、角丸は8px/12px/999px、shadowはsm/md/lgのみを使用
  - ヘッダーは境界線と影を控えめに整理し、タブはpill型から下線型へ変更
  - 検索画面は検索パネル構成を維持し、検索結果はフェードイン、参照元detailsはインディゴの開閉アイコン付きに変更
  - 検索結果の会話アバターは人・ロボット絵文字で質問/回答の区別を維持
  - `#search-results` は `aria-live="polite"` を付け、更新結果をスクリーンリーダーにも通知
  - 「よく検索されています」のサジェストは初期の青い見やすさを残しつつ、白地・インディゴ境界線・軽いhover浮き上がりのピルに調整
  - 管理画面はアップロードドロップゾーン、登録済みドキュメントの「ファイル名」「登録日時」列見出し、成功トースト通知を追加
  - トースト通知は右下表示・3秒で自動消去・`role="status"`/`aria-live="polite"` を付与。エラー/警告はインライン表示を維持
  - アップロード中はスケルトン風プログレスバーで処理中状態を表示（実進捗%はFlask側未変更）
  - 2026-07-03に全体のフォントウェイトと見出しサイズを調整し、強い太字を抑えてやわらかい読み心地に変更
  - 2026-07-03にアクセントカラーをインディゴ（`#4F46E5`、紫寄り）から、アプリアイコン
    （`static/apple-touch-icon.png`）実測の青（`#1987FC`前後）に変更。
    `--color-accent` / `--color-accent-hover` / `--color-accent-soft` / `--color-hover` /
    `--color-selected` / `--color-info` の各CSS変数のみを`:root`で差し替えており、
    サイト全体に変数経由で伝播するためテンプレート側の変更は不要だった
  - `style.css` は更新時刻をクエリ文字列に付けて読み込み、ブラウザ/Space側のCSSキャッシュでUI変更が反映されない問題を防止
  - モバイル幅ではタブ・検索フォーム・ドキュメント行が横スクロールしないようレスポンシブ調整済み
  - 2026-07-03に回答本文（`.bubble-answer p/li`）の`max-width: 72ch`を撤廃。この単位は
    半角英字基準のため日本語の全角文字だと想定の半分程度しか収まらず、吹き出しの枠幅に対して
    本文が不自然に狭く見えていた（右側に大きな余白ができていた）

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
  - 2026-07-03にクライアントサイドJS（`static/doc-list.js`）で、列ソート（ファイル名/登録日時）、ファイル名絞り込み、20件単位のページネーションを追加
  - ソート・絞り込み・ページングはDOM上の`.doc-card`を操作するだけで、Flask側のルーティングやQdrant取得ロジックは変更しない
  - 絞り込みやページ外でチェックした行も選択状態を保持する（削除は選択済み全件が対象。
    2026-07-03に画面上の説明文「削除時は選択済み全件が対象です。」は冗長として削除し、
    現在は選択中に「N件選択中」とだけ表示する）
  - 2026-07-03にファイル名を`GET /manage/download/<filename>`へのダウンロードリンクにした
    （`_is_admin()`必須、`core.get_pdf_bytes()`を利用）。検索結果の引用リンクのようにbase64を
    HTMLへ全件埋め込むと登録数が増えた時にページが肥大化するため、専用ルートに分離。
    `Content-Disposition`は非ASCII文字を直接ヘッダーに書けない問題（`WWW-Authenticate`の教訓）を
    踏まえ、ASCIIフォールバック名＋RFC 5987形式のUTF-8エンコード名を併記している。
    ファイル名の`<a>`は`.doc-card`（`<label>`）内にあるためクリックが行のチェックボックス選択にも
    伝播してしまう問題があり、`event.stopPropagation()`で回避している

### 細かいUI調整（2026-07-03）

- アップロードのドロップゾーン内で「PDF」という単語が3回（アイコンバッジ・見出し・補足文）連続し
  くどかったため、アイコンバッジをアップロードアイコン（SVG）に差し替え、テキストを
  「ここにドラッグ&ドロップ」「またはクリックして選択」に簡素化（セクション見出しと
  「PDF形式のみ」表記に情報は残るため欠落なし）
- 管理者ログインカードのアイコンバッジが「Admin」という文字表示だったのを、見出し
  「管理者ログイン」と重複するためロックアイコン（SVG）に変更
- スマートフォン幅（520px以下）で、チャットのアバターと吹き出しが横並びのままだと
  吹き出しの左端が内側に寄って窮屈だったため、`.chat-message`を1カラムのグリッドに変更し、
  アバターを吹き出しの上に縦積み表示するようにした
- アップロードのドロップゾーンでファイルを選択/ドロップした際、「N件のファイルを選択中」という
  件数だけの表示だったのを、実際のファイル名を表示するように変更（`base.html`の
  `hydrateDropZones()`内`updateMeta()`）。1〜3件はファイル名を「、」区切りで列挙、
  4件以上は先頭2件＋「ほかN件」に省略。未選択時の案内文言も他箇所と合わせて統一

## セキュリティ対策

- プロンプトインジェクション対策: `_INJECTION_PATTERNS` で既知パターンを検出して拒否、`` ``` ``/`###`/`---` を除去
- システムプロンプトとユーザー入力を明確に分離し、ドキュメント外の情報での回答を禁止
- Hugging Face Spaceは Public 設定（コードのpush権限は自分のHFアカウントのみ）
  - Private時代はiPhoneの直接URL（`https://naokoni-rag-search.hf.space`）が署名付き一時URLでしか機能せずホーム画面追加に使えなかったため、Public化して解決した経緯がある
  - Public化に伴い、アプリ全体を**HTTP Basic認証**（`BASIC_AUTH_USER`/`BASIC_AUTH_PASS`、`app.py`の`before_request`）で保護し、URLを知っているだけではアクセスできないようにしている
  - PDFダウンロードは検索結果から（Basic認証を通過した人なら）誰でも可能。この点は許容した上での判断
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
