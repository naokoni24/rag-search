"""
社内文書検索AI - Flaskアプリ

起動方法:
  python app.py
  ※ .env ファイルに GEMINI_API_KEY / ADMIN_PASSWORD を設定してください
"""

import secrets
import time

from flask import Flask, render_template, request, redirect, url_for, session

import rag_core as core

app = Flask(__name__)
app.secret_key = core.get_secret("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 複数PDF一括アップロードを許容(単体上限は別途10MBを検証)
# Hugging Face SpacesはアプリをiframeでHTTPS配信するため、既定のSameSite=Laxだと
# セッションCookieがサードパーティ扱いでブロックされログインが保持されない。
# SameSite=None(+Secure必須)にしてiframe内でもCookieが送信されるようにする。
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True

MAX_LOGIN_ATTEMPTS = 5

_STATIC_SUGGESTIONS = [
    "勤務体系について",
    "見積もりルールについて",
    "経費申請の手続き",
    "有給休暇の申請方法",
]


def _get_pills():
    top_queries = core.get_top_queries(3)
    if top_queries:
        return top_queries
    return [(q, q) for q in _STATIC_SUGGESTIONS]


def _is_admin() -> bool:
    """管理者セッションが有効か確認し、タイムアウトしていれば自動ログアウトする"""
    if not session.get("admin_authenticated"):
        return False
    last = session.get("admin_last_active", 0)
    if time.time() - last > core.ADMIN_TIMEOUT_SEC:
        session.pop("admin_authenticated", None)
        session.pop("admin_last_active", None)
        session["_was_timed_out"] = True
        return False
    return True


def _touch_admin_session():
    session["admin_last_active"] = time.time()


@app.route("/")
def index():
    try:
        docs = core.get_registered_docs()
    except Exception:
        docs = []
    return render_template(
        "index.html",
        active_tab="search",
        has_docs=bool(docs),
        pills=_get_pills(),
        is_admin_header=False,
    )


def _search_message(level: str, text: str):
    return render_template("partials/search_result.html", kind="message", level=level, text=text)


@app.route("/search", methods=["POST"])
def do_search():
    query = request.form.get("query", "")

    try:
        docs = core.get_registered_docs()
    except Exception:
        docs = []
    if not docs:
        return _search_message("info", "「文書を管理」タブからPDFを登録してください。")

    safe_query = core.sanitize_query(query)
    if safe_query is None:
        return _search_message("warning", "入力内容を確認できませんでした。500文字以内の質問を入力してください。")

    cached = core.get_cached_result(safe_query)
    if cached:
        answer, chunks = cached
    else:
        try:
            chunks = core.search(safe_query)
            answer = None
            if chunks:
                answer = core.generate_answer(safe_query, chunks)
                core.record_search(safe_query, answer, chunks)
                core.set_cached_result(safe_query, answer, chunks)
        except Exception as e:
            app.logger.exception("search failed for query=%r", safe_query)
            return _search_message("error", core._gemini_error_message(e))

    if not chunks:
        return _search_message("warning", "関連するドキュメントが見つかりませんでした。別のキーワードで試してください。")

    answer_html = core.render_answer_html(answer, allow_download=False)

    display_chunks = []
    if core._has_citation(answer):
        for c in chunks:
            excerpt = c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
            display_chunks.append({
                "filename": c["filename"],
                "page": c["page"],
                "score_pct": int(c["score"] * 100),
                "excerpt": excerpt,
            })

    return render_template(
        "partials/search_result.html",
        kind="result",
        query=safe_query,
        answer_html=answer_html,
        chunks=display_chunks,
        top3_oob=True,
        pills=_get_pills(),
    )


@app.route("/manage")
def manage():
    is_admin = _is_admin()
    flash_message = None
    flash_level = "success"
    if not is_admin:
        if session.pop("_show_logout_msg", False):
            flash_message, flash_level = "ログアウトしました", "success"
        elif session.pop("_was_timed_out", False):
            flash_message, flash_level = "セッションがタイムアウトしました。再度ログインしてください。", "info"
        return render_template(
            "manage.html",
            active_tab="manage",
            is_admin=False,
            flash_message=flash_message,
            flash_level=flash_level,
            login_error=None,
        )

    _touch_admin_session()
    docs_with_dates = core.get_registered_docs_with_dates()
    return render_template(
        "manage.html",
        active_tab="manage",
        is_admin=True,
        is_admin_header=True,
        docs=docs_with_dates,
        doc_count=len(docs_with_dates),
        max_docs=core.MAX_DOCS,
        messages=[],
    )


@app.route("/manage/login", methods=["POST"])
def manage_login():
    password = request.form.get("password", "")
    admin_password = core.get_secret("ADMIN_PASSWORD")
    attempts = session.get("_login_attempts", 0)

    login_error = None
    if not admin_password:
        login_error = "⚠️ 管理者パスワードが設定されていません。ADMIN_PASSWORDを設定してください。"
    elif attempts >= MAX_LOGIN_ATTEMPTS:
        login_error = "⛔ ログイン試行回数が上限（5回）に達しました。ページを再読み込みしてください。"
    elif password == admin_password:
        session["_login_attempts"] = 0
        session["admin_authenticated"] = True
        _touch_admin_session()
        return redirect(url_for("manage"))
    else:
        session["_login_attempts"] = attempts + 1
        remain = MAX_LOGIN_ATTEMPTS - (attempts + 1)
        login_error = f"パスワードが違います（残り {remain} 回）"

    return render_template(
        "manage.html",
        active_tab="manage",
        is_admin=False,
        flash_message=None,
        login_error=login_error,
    )


@app.route("/manage/logout", methods=["POST"])
def manage_logout():
    session.pop("admin_authenticated", None)
    session.pop("admin_last_active", None)
    session["_show_logout_msg"] = True
    return redirect(url_for("manage"))


def _render_admin_panel(messages):
    docs_with_dates = core.get_registered_docs_with_dates()
    return render_template(
        "partials/admin_panel.html",
        docs=docs_with_dates,
        doc_count=len(docs_with_dates),
        max_docs=core.MAX_DOCS,
        messages=messages,
    )


@app.route("/manage/upload", methods=["POST"])
def manage_upload():
    if not _is_admin():
        return redirect(url_for("manage"))
    _touch_admin_session()

    uploaded_files = [f for f in request.files.getlist("files") if f and f.filename]
    docs = core.get_registered_docs()
    messages = []

    if uploaded_files:
        already = [f.filename for f in uploaded_files if f.filename in docs]
        new_files = [f for f in uploaded_files if f.filename not in docs]
        if already:
            messages.append(("error", f"すでに登録済み: {', '.join(already)}"))

        if new_files:
            remaining = core.MAX_DOCS - len(docs)
            if remaining <= 0:
                messages.append(("error", f"登録上限（{core.MAX_DOCS}件）に達しています。不要なファイルを削除してください。"))
            else:
                addable, over = new_files[:remaining], new_files[remaining:]
                if over:
                    messages.append(("warning", f"上限のため {len(over)} 件は登録できません（残り {remaining} 件まで登録可）"))

                success_names = []
                for uf in addable:
                    data = uf.read()
                    err = core.validate_pdf(data, uf.filename)
                    if err:
                        messages.append(("error", err))
                        continue
                    try:
                        n = core.ingest_pdf(data, uf.filename)
                    except Exception as e:
                        app.logger.exception("ingest_pdf failed for %r", uf.filename)
                        messages.append(("error", f"{uf.filename}：登録中にエラーが発生しました（{core._gemini_error_message(e)}）"))
                        continue
                    if n == 0:
                        messages.append(("warning", f"⚠️ {uf.filename}：テキストを抽出できませんでした（スキャンPDF・画像PDFは非対応）"))
                    else:
                        success_names.append(uf.filename)
                if success_names:
                    messages.append(("success", f"✅ {len(success_names)}件を登録しました: {', '.join(success_names)}"))

    return _render_admin_panel(messages)


@app.route("/manage/delete", methods=["POST"])
def manage_delete():
    if not _is_admin():
        return redirect(url_for("manage"))
    _touch_admin_session()

    filenames = request.form.getlist("filenames")
    messages = []
    if filenames:
        for name in filenames:
            core.delete_document(name)
        messages.append(("success", f"{len(filenames)} 件削除しました"))

    return _render_admin_panel(messages)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(core.get_secret("PORT", "7860")), debug=True)
