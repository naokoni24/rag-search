"""
検索精度の回帰確認用スクリプト。

想定質問と正解(該当ファイル名・回答に含まれるべきキーワード)のセットに対して
core.search() / core.generate_answer() を実行し、
- 検索ヒット率: 期待したファイルが検索候補に含まれたか
- 回答忠実性: 期待キーワードが生成された回答に含まれたか
を集計する。チャンク分割・ハイブリッド検索・リランクなどを変更した前後で
実行して比較すれば、精度への影響を数値で確認できる。

使い方:
  python eval_search.py [質問セットのJSONパス]
  (省略時は eval_queries.json を使用)

注意: 質問セットに書かれた expected_filename の文書は、あらかじめ
「文書を管理」から登録しておく必要がある(eval_queries.json は
社員就業規則.pdf を登録した状態を前提にしたサンプル)。
"""
import json
import sys
from pathlib import Path

import rag_core as core


def run_eval(path: str):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    core.setup_genai_check()

    hit_count = 0
    keyword_results = []

    for i, entry in enumerate(entries, 1):
        query = entry["query"]
        expected_filename = entry.get("expected_filename")
        expected_keywords = entry.get("expected_keywords", [])

        chunks = core.search(query)
        hit_filenames = [c["filename"] for c in chunks]
        hit = (expected_filename in hit_filenames) if expected_filename else bool(chunks)
        rank = hit_filenames.index(expected_filename) + 1 if hit and expected_filename else None
        if hit:
            hit_count += 1

        answer = ""
        keyword_ok = None
        missing = []
        if chunks:
            answer = core.generate_answer(query, chunks)
            if expected_keywords:
                missing = [kw for kw in expected_keywords if kw not in answer]
                keyword_ok = not missing
                keyword_results.append(keyword_ok)

        status = "OK " if hit else "MISS"
        kw_status = "-" if keyword_ok is None else ("OK " if keyword_ok else "MISS")
        print(f"[{i:2}/{len(entries)}] 検索:{status} 忠実性:{kw_status} rank={rank or '-'}  {query}")
        if not hit:
            print(f"          期待ファイル: {expected_filename}  実際の候補: {hit_filenames}")
        if missing:
            print(f"          回答に不足しているキーワード: {missing}")
            print(f"          回答: {answer[:150]}")

    n = len(entries)
    print("\n" + "=" * 60)
    print(f"検索ヒット率: {hit_count}/{n} ({hit_count / n * 100:.0f}%)")
    if keyword_results:
        ok = sum(keyword_results)
        print(f"回答忠実性  : {ok}/{len(keyword_results)} ({ok / len(keyword_results) * 100:.0f}%)")


if __name__ == "__main__":
    query_path = sys.argv[1] if len(sys.argv) > 1 else "eval_queries.json"
    run_eval(query_path)
