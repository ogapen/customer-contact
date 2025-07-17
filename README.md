# 顧客問い合わせ対応AIアプリケーション

このアプリケーションは、顧客からの問い合わせに対してAIが自動で回答するシステムです。

## 🚀 クイックスタート

### 1. アプリケーションの起動
```bash
./start.sh
```

### 2. ブラウザでアクセス
```
http://localhost:8504
```

## 📁 ファイル構成

```
customer-contact/
├── main.py              # メイン処理（アプリケーションの起動ポイント）
├── initialize.py        # 初期化処理（AIエージェントの設定）
├── utils.py            # ユーティリティ関数（RAG処理、データベース操作）
├── components.py       # UI部品（画面表示の処理）
├── constants.py        # 設定値（定数、メッセージ、プロンプト）
├── .env               # 環境変数（APIキーの設定）
├── start.sh           # 起動スクリプト
└── data/              # 学習データ
    ├── company/       # 会社情報
    ├── service/       # サービス情報
    └── customer/      # 顧客情報
```

## 🔧 設定

### 環境変数の設定
`.env`ファイルに以下を設定してください：

```
# 必須：OpenAI APIキー
OPENAI_API_KEY=your_openai_api_key_here

# オプション：Web検索機能を使用する場合
SERPAPI_API_KEY=your_serpapi_key_here
```

## 📋 機能

- **AIチャット機能**: 顧客からの質問に自動回答
- **RAG検索**: 社内文書を参照した正確な回答
- **Web検索**: 必要に応じてリアルタイム情報の検索
- **フィードバック機能**: 回答品質の改善
- **会話履歴管理**: 長い会話の自動管理

## 🛠️ トラブルシューティング

### アプリケーションが起動しない場合
1. `.env`ファイルにOPENAI_API_KEYが設定されているか確認
2. 仮想環境が正しく作成されているか確認
3. `logs/application.log`でエラーログを確認

### 初期化エラーが発生する場合
1. `data/`フォルダに必要なファイルがあるか確認
2. OpenAI APIキーが有効か確認
3. ログファイルで詳細なエラーを確認

## 📝 ログ確認

```bash
# リアルタイムログ表示
tail -f logs/application.log

# 最新のエラーログ確認
grep ERROR logs/application.log | tail -10
```