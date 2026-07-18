"""
PDF をコマンドラインからベクトルDBに登録するスクリプト。
rag_core.ingest_pdf を呼ぶだけの薄いCLIラッパー(Web UIの「文書を管理」タブと同じ
Gemini埋め込み・チャンク分割ロジックを使うため、登録済みデータと非互換になる心配はない)。

使い方:
  python ingest.py 規程集.pdf 就業規則.pdf
"""

import sys
from pathlib import Path

import rag_core as core


def main():
    if len(sys.argv) < 2:
        print("使い方: python ingest.py <PDF1> [<PDF2> ...]")
        sys.exit(1)

    core.setup_genai_check()

    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"  ❌ ファイルが見つかりません: {path}")
            continue
        print(f"処理中: {p.name}")
        data = p.read_bytes()
        err = core.validate_pdf(data, p.name)
        if err:
            print(f"  ❌ {err}")
            continue
        n = core.ingest_pdf(data, p.name)
        if n == 0:
            print("  ⚠️ テキストを抽出できませんでした(スキャンPDF・画像PDFは非対応)")
        else:
            print(f"  → {n} チャンクを登録しました")

    print("\n完了。python app.py で起動してください。")


if __name__ == "__main__":
    main()
