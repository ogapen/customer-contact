# 問い合わせ対応自動化AIエージェント

このアプリケーションは、顧客からの問い合わせに対してAIが自動で回答するシステムです。問い合わせモード機能により、担当者への自動通知も可能です。

## 🚀 機能

### 基本機能
- **AIチャットボット**: 顧客の質問に対してAIが自動回答
- **AIエージェント機能**: より高度な回答生成のための試行錯誤機能
- **RAG（Retrieval-Augmented Generation）**: 社内文書を参照した回答生成
- **ユーザーフィードバック**: 回答の品質向上のためのフィードバック機能

### 問い合わせモード機能
- **ON/OFF切り替え**: サイドバーから簡単に切り替え可能
- **自動担当者割り振り**: 問い合わせ内容に基づいて適切な担当者を選出
- **メール通知**: 指定されたメールアドレスに問い合わせ内容を通知
- **回答案の自動生成**: 問い合わせに対する複数の解決策を提案

## 📋 設定方法

### ローカル環境
1. 必要なパッケージをインストール：
   ```bash
   pip install -r requirements.txt
   ```

2. 環境変数を設定（`.env`ファイルを作成）：
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   INQUIRY_EMAIL_RECIPIENTS=admin@yourcompany.com,support@yourcompany.com
   ```

3. アプリケーションを起動：
   ```bash
   streamlit run customer-contact/main.py
   ```

### Streamlit Cloud
1. GitHubリポジトリをStreamlit Cloudに接続
2. Secrets設定で以下を追加：
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   INQUIRY_EMAIL_RECIPIENTS = "admin@yourcompany.com,support@yourcompany.com"
   ```

## 📧 問い合わせモード

### メールアドレス設定
`constants.py`の`INQUIRY_EMAIL_RECIPIENTS`または環境変数で設定：

```python
INQUIRY_EMAIL_RECIPIENTS = [
    "admin@yourcompany.com",
    "support@yourcompany.com"
]
```

### 使用方法
1. サイドバーで問い合わせモードを「ON」に設定
2. チャット欄からメッセージを送信
3. システムが自動的に担当者に通知メールを送信

## 📁 プロジェクト構造

```
customer-contact/
├── main.py                 # メインアプリケーション
├── initialize.py           # 初期化処理
├── components.py           # UI コンポーネント
├── constants.py            # 定数定義
├── utils.py                # ユーティリティ関数
├── requirements.txt        # 依存パッケージ
└── data/                   # RAG用データ
    ├── company/           # 会社情報
    ├── service/           # サービス情報
    └── customer/          # 顧客情報
```

## 🛠️ 技術スタック

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT-4, LangChain
- **Vector Database**: ChromaDB
- **Document Processing**: PyMuPDF, python-docx
- **Email**: SMTP (Gmail対応)

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
