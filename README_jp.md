# Gemini File Search Store Manager & RAG Chat

Google Gemini API の [File Search Tool (RAG)](https://ai.google.dev/gemini-api/docs/file-search) を管理・活用するためのローカル Streamlit アプリケーションです。

ドキュメントのアップロード、Vector Storeのライフサイクル管理（Blue/Greenデプロイ）、および履歴保存機能付きのチャットGUIを提供します。

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B)
![Gemini API](https://img.shields.io/badge/Google%20GenAI%20SDK-v1.0-4285F4)

## 🌟 主な機能

### 📄 ドキュメント管理

- **スマートアップロード**: PDF, コード, テキストファイルをドラッグ＆ドロップで登録。MD5ハッシュによる重複排除を行い、APIコストと時間を節約します。
- **Blue/Green リフレッシュ**: 稼働中のStoreに影響を与えずに知識ベースを更新します。新規Store作成→ファイル再インデックス→切り替え→旧Store削除を自動で行います。
- **ローカル管理**: SQLite (`gemini_store.db`) でメタデータを管理し、実ファイルは `./input_files/` に保存されるため、手元での編集・管理が容易です。

### 💬 RAG チャット

- **ストリーミング回答**: `gemini-2.5-flash-lite`（設定可）による高速なリアルタイム生成。
- **履歴の永続化**: チャットセッションはローカルDBに保存され、いつでも過去の会話を再開できます。
- **根拠の提示 (Citations)**: モデルが回答に使用したファイルやWebソースを自動的に表示します。
- **自動タイトル生成**: 会話の内容に基づいて、セッション名を自動で要約・設定します。
- **エクスポート**: チャット履歴をJSON形式でダウンロード可能です。

### ⚙️ 管理機能

- **History Manager**: 古くなったチャットセッションを個別に削除できます。
- **堅牢なエラー処理**: APIのレートリミットや503エラーに対する自動リトライ機能を搭載。

## 🚀 インストール方法

### 前提条件

- Python 3.12以上
- Gemini API が有効な Google Cloud プロジェクトおよび APIキー
- [uv](https://github.com/astral-sh/uv) (推奨) または pip

### 1. リポジトリのクローン

```bash
git clone https://github.com/joelgoldschmidt0214/gemini_file_search_tool_mgmt.git
cd gemini_file_search_tool_mgmt
```

### 2. 依存関係のインストール

`uv` を使用する場合 (推奨):

```bash
uv sync
```

標準の `pip` を使用する場合:

```bash
pip install streamlit -r requirements.txt
```

### 3. 環境設定

プロジェクトルートに `.env` ファイルを作成してください:

```env
# 必須
GOOGLE_API_KEY="your-gemini-api-key"

# 任意 (デフォルトは gemini-2.5-flash-lite)
GEMINI_MODEL="gemini-2.5-flash-lite"
```

## 💻 使い方

アプリを起動します:

```bash
uv run streamlit run app.py
# または
streamlit run app.py
```

### 画面構成

サイドバーのナビゲーションで機能を切り替えます。

1. **Documents (ドキュメント管理):**
    - Storeの選択または新規作成を行います。
    - ファイルをアップロードして同期します。
    - ローカルファイルを直接編集した場合などは、**"🔄 Refresh Store"** を押すことでインデックスを再作成・クリーンアップできます。
    - 最下部の "Danger Zone" からStoreを削除できます。
2. **RAG Chat (チャット):**
    - 対話したいStoreを選択します。
    - "✨ New Session" で新規会話を開始、または履歴から再開します。
    - 入力欄の右上にあるボタンから履歴をJSON保存できます。
3. **History Manager (履歴管理):**
    - 保存されたチャットセッションの一覧表示と削除が可能です。

## 🏗️ アーキテクチャ

### SQLiteによるステート管理

Gemini API単体では「特定のStore内のファイルメタデータ」を管理・取得するのが難しいため、本アプリは `SQLAlchemy` を用いてローカルDB (`gemini_store.db`) で状態を管理します。

- `store_records`: Store IDと表示名を管理。
- `file_records`: ファイルパス、MD5ハッシュ、同期状態を管理。
- `chat_sessions` & `chat_messages`: 会話履歴を保存。

### ファイル更新フロー (Blue/Green)

ローカルにあるMarkdownファイル等をエディタで編集した後、反映させるフローは以下の通りです：

1. **Create**: 新規Storeを作成。
2. **Migrate**: `./input_files/` にある最新の実体ファイルを新Storeへアップロード。
3. **Swap**: DBの参照先を新Store IDに切り替え。
4. **Cleanup**: 古いStoreをAPIから削除。

## 📁 ディレクトリ構成

```text
.
├── app.py                # メインアプリケーション
├── gemini_store.db       # SQLiteデータベース (自動生成)
├── input_files/          # アップロードファイルの保存先 (自動生成)
├── .env                  # 環境変数設定
└── pyproject.toml        # 依存ライブラリ定義
```

## 📄 ライセンス

[MIT](https://choosealicense.com/licenses/mit/)
