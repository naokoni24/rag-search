---
title: 社内ナレッジ検索AI
emoji: 🔍
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 社内ナレッジ検索AI

社内PDF文書をGemini + Qdrantで検索・回答するFlaskアプリ。

## ローカル起動

```
pip install -r requirements.txt
python app.py
```

`.env` に以下を設定してください（`.env.example` 参照）。

- `GEMINI_API_KEY`
- `ADMIN_PASSWORD`
- `QDRANT_URL` / `QDRANT_API_KEY`（未設定時はローカル埋め込みQdrantにフォールバック）
- `FLASK_SECRET_KEY`（セッション署名用。本番では必ず設定する）

## Hugging Face Spacesへのデプロイ

1. huggingface.co で Docker SDK の新規Spaceを作成
2. Space の Settings → Repository secrets に上記の環境変数を設定
3. このリポジトリをSpaceのgitリモートにpush
