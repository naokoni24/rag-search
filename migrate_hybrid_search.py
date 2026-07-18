"""
既存の Qdrant `documents` コレクションをハイブリッド検索スキーマ
(名前付き dense + sparse ベクトル)へ移行するワンタイムスクリプト。

背景:
  Qdrant はコレクション作成後にベクトル設定を変更できないため、ハイブリッド検索
  (密ベクトル + 疎ベクトルの RRF 統合)を有効にするにはコレクションを再作成し、
  登録済み PDF を再インデックスする必要がある。PDF 本体は `pdf_files` コレクションに
  base64 で保存済みのため、元ファイルを探し直す必要はない。

使い方:
  python migrate_hybrid_search.py
  ※ .env (または環境変数)の QDRANT_URL / QDRANT_API_KEY / GEMINI_API_KEY を
    対象環境に向けて実行すること。ローカル Qdrant(./qdrant_data)が対象なら
    QDRANT_URL は未設定のままでよい。

  移行前は documents コレクションが旧スキーマ(無名ベクトルのみ)のままでも
  rag_core.search() はレガシー互換の密ベクトル検索にフォールバックするため、
  このスクリプトを実行するまで検索機能が止まることはない。ただし新規 PDF の
  アップロードは(旧スキーマのコレクションに新形式の点を書き込もうとして)
  エラーになるため、早めに実行することを推奨する。

処理内容:
  1. documents コレクションのスキーマを確認し、既にハイブリッド対応済みなら何もしない
  2. pdf_files から登録済み全 PDF の base64 を取得
  3. documents コレクションを削除し、新スキーマで再作成
  4. 各 PDF を新しいチャンク分割ロジック(文単位)で再インデックス
     (Gemini 埋め込み API を再度呼び出すため課金・時間が発生する)
"""
import base64
import time

import rag_core as core


def main():
    client = core.get_qdrant()

    if core._collection_supports_hybrid(client):
        print("documentsコレクションは既にハイブリッド検索スキーマです。何もしません。")
        return

    names = [c.name for c in client.get_collections().collections]
    if core.COLLECTION not in names:
        print("documentsコレクションが存在しません。新スキーマで新規作成します。")
        core.ensure_collection(client)
        print("完了。")
        return

    core._ensure_pdf_collection(client)
    pdf_records, _ = client.scroll(
        collection_name=core.PDF_COLLECTION,
        limit=1000,
        with_payload=["filename", "b64"],
        with_vectors=False,
    )
    docs = [(r.payload["filename"], r.payload["b64"]) for r in pdf_records if r.payload.get("b64")]

    if not docs:
        print("登録済みPDFがありません。documentsコレクションを再作成するだけで移行を完了します。")
        client.delete_collection(core.COLLECTION)
        core.ensure_collection(client)
        print("完了。")
        return

    print(f"{len(docs)}件のPDFを再インデックスします(Gemini埋め込みAPIを再呼び出しするため課金・時間が発生します)。")
    answer = input("続行しますか？ [y/N]: ").strip().lower()
    if answer != "y":
        print("中止しました。")
        return

    print("documentsコレクションを削除して新スキーマで再作成します...")
    client.delete_collection(core.COLLECTION)
    core.ensure_collection(client)
    core._schema_cache.clear()

    failed = []
    for i, (filename, b64) in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {filename} を再インデックス中...")
        try:
            n = core.ingest_pdf(base64.b64decode(b64), filename)
            print(f"  -> {n} チャンク登録")
        except Exception as e:
            print(f"  失敗: {e}")
            failed.append(filename)
        time.sleep(1)  # Gemini APIのレート制限に配慮

    print("\n移行完了。")
    if failed:
        print(f"以下のファイルは失敗したため再度「文書を管理」画面から登録し直してください: {', '.join(failed)}")


if __name__ == "__main__":
    main()
