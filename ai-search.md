# 社内ナレッジ検索AIシステム 仕様

更新日: 2026-07-19
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

## Geminiモデル移行（2026-07-19）→ 同日中に3.5へ移行

- 新規Gemini APIキーでは`gemini-2.5-flash`が利用できず404になるため、回答生成とOCRを
  一時的に`gemini-3.5-flash`へ移行した。
- Gemini 3系の推奨設定に合わせ、回答生成の思考設定を`thinking_budget=0`から
  `thinking_level="minimal"`へ変更した。
- `google-genai`を`2.12.1`へ更新した。クエリ拡張・リランク・ラベル生成は
  `gemini-2.5-flash-lite`を継続利用する。

**→ 同日中にAPIキー側の404問題が解決したため`gemini-2.5-flash`へ差し戻した。**
`gemini-3.5-flash`は入力$1.50・出力$9.00 / 1Mトークンと`gemini-2.5-flash`（入力$0.30・
出力$2.50）の5〜3.6倍の単価で、直前に行った[Geminiコスト削減](#geminiコスト削減2026-07-19-続きその2)
の効果を打ち消してしまうため。回答生成の思考設定も`thinking_budget=0`に戻した
（`thinking_level`は`google-genai`のバージョンは2.12.1のまま据え置いても動作する）。
`gemini-2.5-flash`が今後またAPIキー側で利用不可になった場合は、この節の移行手順を再度参照すること。

## `.env`のGEMINI_API_KEY重複バグ、および回答生成のLite化（2026-07-19 続きその3）

### `.env`に無効なキーが混入し、Lite系呼び出しが全滅していた

ローカルの`.env`に`GEMINI_API_KEY`が2行あり（正しい`AIzaSy...`形式のキーと、`AQ.Ab8...`という
OAuth認可コードらしき無効な値）、`python-dotenv`は同一キーが複数行ある場合ファイル内の
**後勝ち**で読み込むため、無効な`AQ.Ab8...`の方が実際に使われていた。

この状態で`gemini-2.5-flash-lite`を試したところ`404 NOT_FOUND`になり、原因調査の過程で発覚した。
`expand_query`/`rerank_chunks`/`_generate_label`はいずれも例外を握りつぶして安全側にフォールバック
する実装だったため画面のエラーにはならなかったが、狙った機能（クエリ拡張・リランクによる精度向上、
ラベル生成）が無効な間は効いていなかったことになる。**本番（Hugging Face SpaceのSecrets）で同種の
誤ったキーが設定されていないか、合わせて確認すること。**

`.env`の重複行を削除して修正。正しい`AIzaSy...`キーでは`gemini-2.5-flash`・`gemini-2.5-flash-lite`
とも問題なく利用できることを確認した。

### 回答生成もLiteモデルへ変更

`generate_answer`（最終回答生成）を`GEN_MODEL`（`gemini-2.5-flash`）から`GEN_MODEL_LITE`
（`gemini-2.5-flash-lite`）に変更した。`eval_search.py`で比較し、検索ヒット率100%・回答忠実性95%
とflash版から変化がないことを確認済み。

- OCR（`_ocr_page_with_gemini`）は画像認識精度が未検証のため、`GEN_MODEL`を`GEN_MODEL_OCR`に
  改名した上で`gemini-2.5-flash`を維持し、回答生成とは別モデルにした
- これで応答に関わるGemini呼び出し（クエリ拡張・リランク・ラベル生成・回答生成）が全て
  `gemini-2.5-flash-lite`になった。OCRのみ`gemini-2.5-flash`
- 副次的なバグ修正: `_gemini_error_message`が旧`GEN_MODEL`定数名をエラーメッセージに直接埋め込んで
  おり、モデルが複数になったことで参照先の定数がなくなって`NameError`を起こす状態になっていた
  （404発生時に本来の「モデルが見つかりません」という案内の代わりに未定義エラーで落ちる）。
  モデル名を埋め込まない汎用メッセージに変更して修正
- **注意**: `eval_search.py`の20問はいずれも社員就業規則.pdf1文書からの直接的な事実確認質問。
  複数チャンクをまたぐ複雑な質問や文章のニュアンスを汲む必要がある質問は未検証

## Geminiコスト削減(2026-07-19 続きその2)

1検索あたりのGemini API概算コストを試算した上で、以下を実施。

### 実施したこと

- `generate_answer`に`thinking_config=ThinkingConfig(thinking_budget=0)`を追加。他の3呼び出し(`expand_query`/`rerank_chunks`/`_generate_label`)には既に付いていたが、最終回答生成だけ抜けており、gemini-2.5-flashのデフォルト思考トークンが出力扱いで課金されていた
- クエリ拡張(`expand_query`)・リランク(`rerank_chunks`)・ラベル生成(`_generate_label`)を`gemini-2.5-flash`から`gemini-2.5-flash-lite`（`GEN_MODEL_LITE`定数、入力$0.10・出力$0.40 / 1Mトークン。通常のFlashは入力$0.30・出力$2.50）に変更。いずれも変換・分類程度の軽いタスクのため精度への影響は小さいと判断
  - 回答生成(`generate_answer`)とOCR(`_ocr_page_with_gemini`)は精度優先で`GEN_MODEL`(gemini-2.5-flash)のまま維持
  - `eval_search.py`（`eval_queries.json`の20問）で変更前後を比較し、検索ヒット率100%・回答忠実性95%とも変化なしを確認

### 検討したが見送ったこと: クエリ拡張の廃止

ハイブリッド検索(疎ベクトル検索)導入によりクエリ拡張の必要性が下がっている可能性を検証するため、`expand_query`を無効化した状態で`eval_search.py`を実行して比較した。

- 検索ヒット率: 有効/無効とも20/20(100%)で差なし
- 回答忠実性: 有効時19/20・無効時18/20と表面上は差が出たが、無効時に唯一増えた不一致（「昇給は年に1回」という質問に対し回答が「年に1回」と表記し、期待キーワード「年1回」との完全一致に失敗しただけ）は**内容は正しく、eval_search.pyの単純な部分文字列一致による偽陰性**と判明。実質的な精度差はなかった
- ただし`eval_queries.json`の20問はいずれも文書中の正式な用語に近い直接的な言い回しで、クエリ拡張が本来効果を発揮する「口語的・曖昧な言い回し→正式用語への変換」というシナリオを検証できていない
- 上記の理由により、**このテスト結果だけではクエリ拡張の廃止が安全とは判断できないため、見送った**。曖昧な言い回しの質問を`eval_queries.json`に追加した上で再検証すれば、より確度の高い判断ができる

## バグ修正・運用機能追加(2026-07-19 続き)

### バグ修正

- Basic認証(`app.py`の`require_basic_auth`)と管理者ログイン(`manage_login`)のパスワード比較を`==`から`secrets.compare_digest`に変更(タイミング攻撃対策)
- `ingest.py`（Streamlit時代の残骸で、`multilingual-e5-small`384次元を使っており現行の3072次元コレクションと非互換だった）を`rag_core.ingest_pdf`を呼ぶだけのCLIラッパーに書き換え。Web UIと同じ埋め込み・チャンク分割ロジックを使うため非互換の心配がなくなった
- ローカルのファイルベースQdrant(`./qdrant_data`)で、検索ログ記録をバックグラウンドスレッド化した際に`sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`が発生する不具合を修正。macOS/WindowsのSQLiteビルドは`CHECK_SAME_THREAD`が既定で有効なため、`QdrantClient(path=..., force_disable_check_same_thread=True)`を指定して無効化した（本番はQdrant Cloud/HTTP接続のため影響なし。Linuxの多くのビルドも影響なし）

### 運用機能: ヒットしなかった質問のログ

検索してもチャンクが1件もヒットしなかった質問、および「チャンクは取得できたが出典付きで回答できなかった質問」（実質ヒットなし）を、`search_logs`コレクションの別ポイント（`NO_HIT_LOG_POINT_ID`）に集計記録するようにした。

- `record_no_hit_search()` / `load_no_hit_log()` / `save_no_hit_log()` / `get_top_no_hit_queries()`（`rag_core.py`）
- 成功ログ(`record_search`、「よく検索されています」ピル用)とは完全に別集計にし、成功ログの質を落とさない
- `app.py`の`do_search()`で、`chunks`が空、または`chunks`はあるが`_has_citation(answer)`がFalseの場合にバックグラウンドスレッドで記録
- 管理画面（`/manage`）の登録済みドキュメント一覧の下に「ヒットしなかった質問」セクションを追加（`templates/partials/admin_panel.html`、データが無い場合は非表示）。関連文書の登録漏れの参考にする想定

### 運用機能: 検索精度の評価スクリプト

`eval_search.py` + `eval_queries.json` を追加。想定質問・正解ファイル名・回答に含まれるべきキーワードのセットに対して`core.search()`/`core.generate_answer()`を実行し、検索ヒット率と回答忠実性(キーワード充足率)を集計する回帰確認ツール。

- 使い方: `python eval_search.py [質問セットのJSONパス]`（省略時は`eval_queries.json`）
- 同梱の`eval_queries.json`は`社員就業規則.pdf`の実際の条文から作成した20問（有給休暇・懲戒・給与・テレワーク等）。ローカルでこの文書を登録した状態で実行し、検索ヒット率100%・回答忠実性95%を確認済み（1件は「いつから」という質問に対し回答が日数まで言及しなかっただけで、内容自体は正しい）
- チャンク分割・ハイブリッド検索・リランクなど検索まわりを変更した際に、このスクリプトを前後で実行して精度への影響を数値で比較できる
- 他の文書についても同じ形式でエントリを追加すれば評価範囲を拡張できる

### 運用機能: スキャンPDF・画像PDFのOCR対応

`extract_pages()`（`rag_core.py`）で、PDFページにテキスト層がない場合（スキャンPDF・画像PDF）はページを画像化(`page.get_pixmap(dpi=150)`)し、Geminiに書き起こしさせる(`_ocr_page_with_gemini`)フォールバックを追加。追加ライブラリ依存なし（既存のGemini API呼び出しのみ）。

- 通常のテキスト層があるPDFは従来通り高速なテキスト抽出のみ(OCR呼び出しは発生しない。実測: 16ページで0.05秒)
- テキスト層のないページのみ個別にOCR呼び出しが発生する(1ページ=1回のGemini呼び出し)ため、スキャンPDFの登録は従来より時間がかかる
- OCR失敗時は空文字を返し、そのページは従来通りスキップされる（登録自体は失敗しない）
- 合成テスト画像（日本語フォント非対応で文字化けした画像、および正しく描画した英語画像）で書き起こし精度を確認済み

## 検索精度・速度の改善(2026-07-19)

### 検索パイプライン刷新: ハイブリッド検索 + リランク

`search()`(`rag_core.py`)を以下の流れに刷新した。

1. 元クエリ・`expand_query`で拡張したクエリの両方を**1回のGemini埋め込みAPI呼び出しにまとめて**embed(以前は2回に分けていた)
2. 各クエリについて、Qdrantに対し
   - 密ベクトル検索(Gemini embedding、コサイン類似度)
   - 疎ベクトル検索(Janomeで内容語をトークン化 → feature hashingでスパースベクトル化。Qdrant側`Modifier.IDF`でBM25風のIDF重み付け)
   を実行し、各20件(`CANDIDATE_LIMIT`)取得
3. 複数の結果リストをReciprocal Rank Fusion(`_rrf_merge`)でクライアント側マージ(密・疎はスコアのスケールが異なるため、コサイン類似度の閾値ではなく順位ベースで統合)
4. マージ後の候補を`rerank_chunks()`でGeminiに関連度順に並べ替えさせ、無関係な候補を除いてtop_k(5件)に絞る
   - 従来の`SCORE_THRESHOLD`(0.55)による足切りはハイブリッド対応コレクションでは使わなくなった(意味的な関連性判定はリランカーが担う)
   - リランクAPI呼び出しが失敗した場合はコサイン類似度順にフォールバック

新規依存: `janome`(純Python実装の形態素解析器。MeCab等の外部バイナリ不要でDockerビルドに影響しない)

### チャンク分割の改善

`split_chunks()`を固定400文字での機械的な分割から、**文単位(句点区切り)でCHUNK_SIZE(400文字)前後・OVERLAP(80文字)重複で詰める方式**に変更。「第◯条」のような条文見出しが出現した場合はそこでチャンクを区切り、条文と直前の内容が混在しないようにした。

あわせて`ingest_pdf`のチャンク分割をページ単位からドキュメント全体単位に変更(`_build_document_text`で全ページを連結し、`_page_for_offset`でチャンクの開始位置から所属ページを逆引き)。ページをまたぐ条文が分断される問題を解消した。

**既存の登録済みPDFはチャンク分割の恩恵を受けない**(再アップロードして再チャンク化するまで旧チャンクのまま)。

### Qdrantスキーマ移行(ハイブリッド検索の有効化)

Qdrantはコレクション作成後にベクトル設定を変更できないため、ハイブリッド検索(名前付きdense/sparseベクトル、`DENSE_VECTOR_NAME="dense"` / `SPARSE_VECTOR_NAME="text_sparse"`)を有効にするには`documents`コレクションの再作成と全PDFの再インデックスが必要。

- `_collection_supports_hybrid()`でコレクションのスキーマを判定し、**旧スキーマ(無名ベクトルのみ)のままでも`search()`/`ingest_pdf()`はレガシー互換の密ベクトルのみのモードに自動フォールバック**する(移行前でも検索・登録は壊れない。動作確認済み)
- 移行は`migrate_hybrid_search.py`を実行(`QDRANT_URL`/`QDRANT_API_KEY`を対象環境に向けて`python migrate_hybrid_search.py`)。`pdf_files`コレクションに保存済みのPDF本体(base64)を使って全件再インデックスするため元ファイルの再アップロードは不要。Gemini埋め込みAPIを再呼び出しするため課金・時間が発生する点に注意
- **本番(Qdrant Cloud)は本対応をpushしただけでは未移行のまま**。移行スクリプトを実行するまでは新チャンク分割・ハイブリッド検索・リランクは効かず、レガシー動作(旧チャンク・密ベクトルのみ)が続く

### 速度改善

- クエリ埋め込みを1回のAPI呼び出しにまとめた(上記)
- `expand_query`に`thinking_config=ThinkingConfig(thinking_budget=0)`を追加(付け忘れで無駄な思考時間が発生していた)
- 検索ログ記録(`record_search`。初回クエリはラベル生成のGemini呼び出しを伴う)をレスポンス返却後にバックグラウンドスレッドで実行するよう変更(`app.py`の`_record_search_async`)。検索の体感速度から切り離し、失敗しても回答自体は返るようにした(以前は失敗すると検索全体がエラーになっていた)
- `_answer_cache`のTTLが実際にはチェックされていなかったバグを修正(`get_cached_result`が`_store`を直接読んでいて期限切れでも古い回答を返し続けていた)。`_TTLCache`に`get`/`set`メソッドを追加して修正

以上により、1クエリあたりのLLM呼び出しは「クエリ拡張→リランク→回答生成」の3回(以前は2回)になった。リランクの追加は精度向上とのトレードオフとして許容(いずれも`thinking_budget=0`の短いクラシフィケーション呼び出しのため、実測ではローカル環境で1クエリ2〜3秒程度)。

## 基本構成

- フレームワーク: Flask + Jinja2 + htmx（部分更新、`static/htmx`はCDN読み込み）
- AI（埋め込み・回答生成）: Google Gemini
  - 埋め込みモデル: `gemini-embedding-001`（ベクトル次元 3072）
  - 生成モデル: 回答生成・クエリ拡張・リランク・ラベル生成は`gemini-2.5-flash-lite`
    （`GEN_MODEL_LITE`）、OCRのみ`gemini-2.5-flash`（`GEN_MODEL_OCR`）
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
| `documents` | PDFチャンク本文（filename, page, text） | ハイブリッド対応コレクション: 名前付き`dense`（3072次元、Gemini埋め込み）+ `text_sparse`（疎ベクトル、Modifier.IDF）。未移行の旧コレクションは無名3072次元ベクトルのみ（[検索精度・速度の改善](#検索精度速度の改善2026-07-19)参照） |
| `pdf_files` | PDF本体（base64で保存、ダウンロード・再表示用） | ダミー1次元 |
| `search_logs` | 検索ログ（クエリ正規化キーごとの件数・ラベル） | ダミー1次元、単一ポイントに集約保存 |

- チャンクサイズ: 目標400文字（文単位で詰めるため前後する）、オーバーラップ80文字
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
  3. 元クエリ・拡張クエリを1回のAPI呼び出しでembedし、`documents` をハイブリッド検索（密ベクトル+疎ベクトル、RRF統合）→ Geminiによるリランクでtop_k件に絞り込み（詳細は[検索精度・速度の改善](#検索精度速度の改善2026-07-19)参照）
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
- 2026-07-03にアップロードUIを複数選択前提に調整。`input multiple` と既存の複数ファイル処理は
  そのまま維持しつつ、選択後は「N件のファイルを選択中」とファイル名チップをドロップゾーン内に表示し、
  複数アップロード対象が分かりやすい見た目にした
- 管理者ログインカードのアイコンバッジが「Admin」という文字表示だったのを、見出し
  「管理者ログイン」と重複するためロックアイコン（SVG）に変更
- スマートフォン幅（520px以下）で、チャットのアバターと吹き出しが横並びのままだと
  吹き出しの左端が内側に寄って窮屈だったため、`.chat-message`を1カラムのグリッドに変更し、
  アバターを吹き出しの上に縦積み表示するようにした
- アップロードのドロップゾーンでファイルを選択/ドロップした際、「N件のファイルを選択中」という
  件数だけの表示だったのを、実際のファイル名を表示するように変更（`base.html`の
  `hydrateDropZones()`内`updateMeta()`）。1〜3件はファイル名を「、」区切りで列挙、
  4件以上は先頭2件＋「ほかN件」に省略。未選択時の案内文言も他箇所と合わせて統一
- 2026-07-03に検索トップのサブタイトルを「社内文書に関する質問を入力してください。AIが最適な
  回答を提供します。」から「知りたいことをそのまま質問するだけ。AIが文書から探してお答え
  します。」に変更（`templates/index.html`の`.search-hero-sub`）。操作の手軽さを伝える文言に調整
- 2026-07-03にヘッダーの小見出し（`base.html`の`.header-title`/`.header-subtitle`、
  768px以下では非表示）を「社内ナレッジ検索システム」「社内文書をAIで即座に検索・回答」から
  「社内ナレッジAI」「質問するだけで、AIが文書から回答」に変更。トップページのサブタイトルと
  表現のトーンを揃えた

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

- **本番Qdrant Cloudの`documents`コレクションは`migrate_hybrid_search.py`未実行の間、ハイブリッド検索スキーマに移行されない**（レガシーモードにフォールバックし続ける。動作は壊れないが新チャンク分割・ハイブリッド検索・リランクの恩恵を受けられない）
- 疎ベクトルのトークン化（Janome）は形態素解析の基本形抽出のみで、同義語展開などは行わない
- スキャンPDF・画像PDFはGemini OCRで対応済み（テキスト層がないページのみ）。ページ数が多いスキャンPDFはページ単位でOCR呼び出しが発生するため登録に時間がかかる
- 検索ログ・ヒットしなかった質問ログは単一ポイントに集約保存のため、同時書き込み時はマージ処理で競合緩和しているが完全ではない
- `MAX_DOCS = 50` 件、`MAX_UPLOAD_MB = 10` はハードコード（運用規模に応じて見直しの可能性）
- gunicornを`--workers 1`にしているため、同時アクセスが多い場合はスレッド（4つ）の範囲でしか並列処理できない。将来的にアクセスが増えたらQdrant Cloud接続を前提に`--workers`を増やすことを検討（ローカル埋め込みQdrantはマルチプロセスでファイルロック競合を起こすため増やせないが、本番は外部Qdrant Cloudなので制約なし）
- Hugging Face Spacesの無料枠はコンテナのファイルシステムが再起動のたびに初期化されるが、検索データはQdrant Cloud（外部）にあるため影響なし
