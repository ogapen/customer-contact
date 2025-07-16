"""
# =============================================================================
# ユーティリティ関数群
# =============================================================================
# このファイルには、アプリケーション全体で使用される汎用的な関数が含まれています。
"""

############################################################
# 必要なライブラリの読み込み
############################################################
import os
import logging
import unicodedata
import platform
import streamlit as st
import smtplib
import csv
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

############################################################
# エラーメッセージ処理
############################################################
def build_error_message(message):
    """
    # エラーメッセージの構築
    エラーメッセージに共通の管理者問い合わせ情報を追加します。
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])

############################################################
# RAGチェーンの作成
############################################################
def create_rag_chain(db_name):
    """
    # RAG（Retrieval-Augmented Generation）チェーンの作成
    指定されたデータベースを参照するRAGチェーンを作成します。
    
    Args:
        db_name: データベース名（.db_all, .db_company, .db_service, .db_customer）
    
    Returns:
        作成されたRAGチェーン
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"RAGチェーンを作成中: {db_name}")

    # ドキュメントを格納するリスト
    docs_all = []
    
    # 全データベースモードの場合
    if db_name == ct.DB_ALL_PATH:
        logger.info(f"全データベースモードで処理中: {ct.RAG_TOP_FOLDER_PATH}")
        if os.path.exists(ct.RAG_TOP_FOLDER_PATH):
            folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
            for folder_path in folders:
                if folder_path.startswith("."):
                    continue
                full_path = f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}"
                logger.info(f"フォルダを処理中: {full_path}")
                add_docs(full_path, docs_all)
    else:
        # 個別データベースモードの場合
        folder_path = ct.DB_NAMES[db_name]
        logger.info(f"個別データベースモードで処理中: {db_name} -> {folder_path}")
        add_docs(folder_path, docs_all)

    # 文字列の正規化処理（Windows環境対応）
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # テキストを適切なサイズに分割
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # OpenAIの埋め込みモデルを使用
    embeddings = OpenAIEmbeddings()

    # データベースの作成または読み込み
    if os.path.isdir(db_name):
        # 既存のデータベースを読み込み
        db = Chroma(persist_directory=db_name, embedding_function=embeddings)
        logger.info(f"既存のデータベースを読み込み: {db_name}")
    else:
        # 新しいデータベースを作成
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=db_name)
        logger.info(f"新しいデータベースを作成: {db_name}")

    # 検索機能の設定
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    # 質問生成用のプロンプトテンプレート
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", question_generator_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 回答生成用のプロンプトテンプレート
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", question_answer_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 履歴を考慮した検索機能の作成
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    # 質問回答チェーンの作成
    question_answer_chain = create_stuff_documents_chain(
        st.session_state.llm, question_answer_prompt
    )
    
    # 最終的なRAGチェーンの作成
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

############################################################
# ドキュメントの追加処理
############################################################
def add_docs(folder_path, docs_all):
    """
    # フォルダ内のドキュメントをリストに追加
    指定されたフォルダ内のサポートされているファイルを読み込みます。
    
    Args:
        folder_path: 読み込むフォルダのパス
        docs_all: ドキュメントを格納するリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # フォルダの存在確認
    if not os.path.exists(folder_path):
        logger.warning(f"フォルダが存在しません: {folder_path}")
        return
    
    if not os.path.isdir(folder_path):
        logger.warning(f"パスがディレクトリではありません: {folder_path}")
        return
    
    try:
        files = os.listdir(folder_path)
    except PermissionError:
        logger.error(f"フォルダへのアクセス権限がありません: {folder_path}")
        return
    except Exception as e:
        logger.error(f"フォルダの読み込みに失敗: {folder_path}, エラー: {e}")
        return
    
    # 各ファイルを処理
    for file in files:
        # ヒドンファイルをスキップ
        if file.startswith('.'):
            continue
            
        # ファイル拡張子の確認
        file_extension = os.path.splitext(file)[1]
        
        # サポートされているファイル形式の場合のみ処理
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            try:
                # 適切なローダーを使用してファイルを読み込み
                loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
                docs = loader.load()
                docs_all.extend(docs)
                logger.info(f"ファイルを読み込み: {folder_path}/{file}")
            except Exception as e:
                logger.error(f"ファイルの読み込みに失敗: {folder_path}/{file}, エラー: {e}")
        else:
            logger.debug(f"サポートされていないファイル形式: {folder_path}/{file}")

############################################################
# 文字列の正規化処理
############################################################
def adjust_string(text):
    """
    # 文字列の正規化処理
    Windows環境での文字化け防止のための文字列処理を行います。
    
    Args:
        text: 処理対象の文字列
    
    Returns:
        正規化された文字列
    """
    if isinstance(text, str):
        # Unicode正規化
        text = unicodedata.normalize("NFKC", text)
        # Windows環境の場合、cp932で表現できない文字を除去
        if platform.system() == "Windows":
            text = text.encode("cp932", errors="ignore").decode("cp932")
    return text

############################################################
# AIエージェントの実行
############################################################
def execute_agent_or_chain(message):
    """
    # AIエージェントまたはRAGチェーンの実行
    ユーザーの質問に対してAIが回答を生成します。
    
    Args:
        message: ユーザーからの質問
    
    Returns:
        AIの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # AIエージェントを使用して回答を生成
        result = st.session_state.agent_executor.invoke({
            "input": message,
            "chat_history": st.session_state.chat_history
        })
        
        # 回答部分を取得
        answer = result.get("output", "")
        logger.info(f"AI回答生成成功: {len(answer)}文字")
        
        return answer
        
    except Exception as e:
        logger.error(f"AI回答生成エラー: {e}")
        return "申し訳ございません。システムエラーが発生しました。しばらく時間をおいて再度お試しください。"

############################################################
# 会話履歴の管理
############################################################
def delete_old_conversation_log(result):
    """
    # 古い会話履歴の削除
    トークン数が上限を超えた場合に古い会話を削除します。
    
    Args:
        result: 最新のAI回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # 新しい回答のトークン数を計算
    result_tokens = len(st.session_state.enc.encode(result))
    st.session_state.total_tokens += result_tokens
    
    # トークン数が上限を超えた場合、古い会話を削除
    if st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS and st.session_state.messages:
            # 最古のメッセージを削除
            removed_message = st.session_state.messages.pop(0)
            removed_tokens = len(st.session_state.enc.encode(removed_message["content"]))
            st.session_state.total_tokens -= removed_tokens
            
            # チャット履歴からも削除
            if st.session_state.chat_history:
                st.session_state.chat_history.pop(0)
        
        logger.info(f"古い会話履歴を削除しました。現在のトークン数: {st.session_state.total_tokens}")

############################################################
# ツール実行関数群
############################################################
def run_company_doc_chain(param):
    """
    # 会社情報検索ツールの実行
    会社に関する情報を検索します。
    """
    result = st.session_state.company_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })
    return result["answer"]

def run_service_doc_chain(param):
    """
    # サービス情報検索ツールの実行
    サービスに関する情報を検索します。
    """
    result = st.session_state.service_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })
    return result["answer"]

def run_customer_doc_chain(param):
    """
    # 顧客情報検索ツールの実行
    顧客とのやり取りに関する情報を検索します。
    """
    result = st.session_state.customer_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })
    return result["answer"]

############################################################
# 新しいツール関数群
############################################################

def run_all_documents_chain(param):
    """
    # 全文書横断検索ツールの実行
    全社文書から横断的に情報を検索します。
    """
    result = st.session_state.rag_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })
    return result["answer"]

def get_current_time(param=None):
    """
    # 現在時刻取得ツールの実行
    現在の日時を返します。
    """
    import datetime
    import pytz
    
    # 日本時間で現在時刻を取得
    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    
    formatted_time = now.strftime("%Y年%m月%d日 %H:%M:%S")
    return f"現在の日時: {formatted_time} (日本時間)"

def calculate_expression(param):
    """
    # 計算ツールの実行
    数値計算を実行します。
    """
    try:
        # 安全な計算のため、evalの代わりに制限された計算を行う
        import re
        
        # 数字と基本的な演算子のみを許可
        if re.match(r'^[\d+\-*/().\s]+$', param):
            result = eval(param)
            return f"計算結果: {param} = {result}"
        else:
            return "計算エラー: 数字と基本的な演算子（+, -, *, /, (), .）のみ使用可能です。"
    except Exception as e:
        return f"計算エラー: {str(e)}"

def validate_email(param):
    """
    # メールアドレス検証ツールの実行
    メールアドレスの形式をチェックします。
    """
    import re
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, param):
        return f"✅ メールアドレス '{param}' は有効な形式です。"
    else:
        return f"❌ メールアドレス '{param}' は無効な形式です。正しい形式で入力してください。"

def check_order_status(param):
    """
    # 注文状況確認ツールの実行
    注文番号から注文状況を確認します（模擬データ）。
    """
    import random
    
    # 模擬データ
    order_statuses = [
        "注文受付済み",
        "商品準備中",
        "発送準備中",
        "発送済み",
        "配送中",
        "配達完了"
    ]
    
    # 注文番号の形式チェック
    if len(param) >= 6 and param.isalnum():
        status = random.choice(order_statuses)
        return f"注文番号 {param} の状況: {status}"
    else:
        return "注文番号の形式が正しくありません。6文字以上の英数字で入力してください。"

def search_faq(param):
    """
    # FAQ検索ツールの実行
    よくある質問から関連する情報を検索します。
    """
    # 模擬FAQデータ
    faqs = {
        "配送": "配送は通常3-5営業日で完了します。お急ぎの場合は翌日配送（追加料金）もご利用いただけます。",
        "返品": "商品到着後30日以内であれば返品可能です。未開封・未使用品に限ります。",
        "支払い": "クレジットカード、銀行振込、代金引換、コンビニ決済をご利用いただけます。",
        "サイズ": "サイズ表は各商品ページに記載されています。ご不明な場合はお気軽にお問い合わせください。",
        "会員登録": "会員登録は無料です。ポイント還元やお得な情報をお届けします。",
        "キャンセル": "発送前であればキャンセル可能です。発送後のキャンセルは承れません。",
        "保証": "商品には1年間の保証が付いています。不具合がございましたら無料で交換いたします。"
    }
    
    # キーワードマッチング
    for keyword, answer in faqs.items():
        if keyword in param:
            return f"【FAQ】{keyword}について: {answer}"
    
    # マッチしない場合
    return "該当するFAQが見つかりませんでした。以下のキーワードで検索できます: " + ", ".join(faqs.keys())

def get_contact_info(param=None):
    """
    # 連絡先情報取得ツールの実行
    会社の連絡先情報を返します。
    """
    contact_info = """
    【株式会社EcoTee 連絡先情報】
    
    📞 電話番号: 03-1234-5678
    📧 メールアドレス: info@ecotee.co.jp
    🌐 ウェブサイト: https://www.ecotee.co.jp
    📍 住所: 〒100-0001 東京都千代田区千代田1-1-1 EcoTeeビル
    
    【サポート窓口】
    📞 カスタマーサポート: 0120-123-456 (フリーダイヤル)
    📧 サポートメール: support@ecotee.co.jp
    """
    return contact_info

def get_business_hours(param=None):
    """
    # 営業時間取得ツールの実行
    営業時間とサポート時間を返します。
    """
    business_hours = """
    【営業時間・サポート時間】
    
    🏢 営業時間: 平日 9:00 - 18:00 (土日祝日休み)
    📞 電話サポート: 平日 9:00 - 17:00 (土日祝日休み)
    📧 メールサポート: 24時間受付 (返信は営業時間内)
    🌐 オンラインサポート: 24時間利用可能
    
    【緊急時対応】
    重要な案件については営業時間外でも対応可能です。
    緊急連絡先: emergency@ecotee.co.jp
    """
    return business_hours

def calculate_price(param):
    """
    # 価格計算ツールの実行
    商品価格の計算や割引適用の計算を行います。
    """
    try:
        # 基本価格設定
        base_prices = {
            "basic": 1000,
            "standard": 2000,
            "premium": 3000,
            "enterprise": 5000
        }
        
        # 割引率設定
        discount_rates = {
            "student": 0.2,  # 学生割引20%
            "senior": 0.15,  # シニア割引15%
            "member": 0.1,   # 会員割引10%
            "bulk": 0.25     # まとめ買い割引25%
        }
        
        # パラメータ解析
        param_lower = param.lower()
        total_price = 0
        discount = 0
        
        # 商品価格の計算
        for product, price in base_prices.items():
            if product in param_lower:
                quantity = 1
                # 数量の検出
                import re
                qty_match = re.search(r'(\d+)', param_lower)
                if qty_match:
                    quantity = int(qty_match.group(1))
                
                subtotal = price * quantity
                total_price += subtotal
                
                # 割引適用
                for disc_type, rate in discount_rates.items():
                    if disc_type in param_lower:
                        discount = subtotal * rate
                        break
        
        if total_price > 0:
            final_price = total_price - discount
            result = f"""
            【価格計算結果】
            商品小計: ¥{total_price:,}
            割引額: ¥{discount:,}
            最終価格: ¥{final_price:,}
            """
            return result
        else:
            return "価格計算のためには商品名（basic, standard, premium, enterprise）を含めてください。"
    
    except Exception as e:
        return f"価格計算エラー: {str(e)}"

def get_weather_info(param=None):
    """
    # 天気情報取得ツールの実行
    天気情報を返します（模擬データ）。
    """
    import random
    
    weather_conditions = ["晴れ", "曇り", "雨", "小雨", "雪"]
    temperature = random.randint(5, 35)
    condition = random.choice(weather_conditions)
    
    weather_info = f"""
    【現在の天気情報】
    🌤️ 天気: {condition}
    🌡️ 気温: {temperature}°C
    
    【配送への影響】
    """
    
    if condition in ["雨", "雪"]:
        weather_info += "⚠️ 悪天候により配送に遅延が生じる可能性があります。"
    elif condition == "晴れ":
        weather_info += "✅ 良好な天候で配送に影響はありません。"
    else:
        weather_info += "☁️ 通常通りの配送が可能です。"
    
    return weather_info

def detect_language(param):
    """
    # 言語検出ツールの実行
    入力テキストの言語を検出します。
    """
    try:
        # 簡単な言語検出ロジック
        import re
        
        # 日本語の検出（ひらがな、カタカナ、漢字）
        if re.search(r'[あ-ん]', param) or re.search(r'[ア-ン]', param) or re.search(r'[一-龯]', param):
            return f"検出言語: 日本語 (Japanese)\n入力テキスト: {param}"
        
        # 韓国語の検出
        elif re.search(r'[가-힣]', param):
            return f"検出言語: 韓国語 (Korean)\n입력 텍스트: {param}"
        
        # 中国語の検出（簡体字・繁体字）
        elif re.search(r'[一-龯]', param) and not re.search(r'[あ-んア-ン]', param):
            return f"検出言語: 中国語 (Chinese)\n输入文本: {param}"
        
        # 英語の検出
        elif re.search(r'^[a-zA-Z\s.,!?]+$', param):
            return f"検出言語: 英語 (English)\nInput text: {param}"
        
        else:
            return f"言語を特定できませんでした。\n入力テキスト: {param}"
    
    except Exception as e:
        return f"言語検出エラー: {str(e)}"

############################################################
# メール設定テスト機能
############################################################
def test_email_settings():
    """
    メール設定をテストする関数
    
    Returns:
        str: テスト結果メッセージ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # SMTP設定の取得
        smtp_host = os.getenv("SMTP_HOST", ct.SMTP_SERVER)
        smtp_port = int(os.getenv("SMTP_PORT", ct.SMTP_PORT))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        logger.info(f"SMTP設定確認: {smtp_host}:{smtp_port}")
        logger.info(f"ユーザー名: {smtp_username}")
        logger.info(f"パスワード設定: {'設定済み' if smtp_password else '未設定'}")
        
        if not smtp_username:
            return "❌ SMTP_USERNAMEが設定されていません"
        
        if not smtp_password:
            return "❌ SMTP_PASSWORDが設定されていません"
        
        # 送信先の確認
        recipients = ct.get_inquiry_email_recipients()
        logger.info(f"送信先: {recipients}")
        
        if not recipients:
            return "❌ 送信先メールアドレスが設定されていません"
        
        return f"✅ 設定確認完了 - 送信先: {', '.join(recipients)}"
        
    except Exception as e:
        logger.error(f"設定確認エラー: {e}")
        return f"❌ 設定確認エラー: {str(e)}"

############################################################
# 会話履歴の問い合わせ機能
############################################################
def send_conversation_inquiry():
    """
    会話履歴をまとめて問い合わせメールを送信する
    
    Returns:
        str: 送信結果メッセージ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # 会話履歴の取得
        conversation_history = get_conversation_summary()
        logger.info(f"会話履歴を取得: {len(conversation_history)} 文字")
        
        # メール内容の作成
        email_content = f"""
問い合わせ日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

【会話履歴】
{conversation_history}

【問い合わせ内容】
上記の会話に関して、さらに詳しい情報が必要です。
担当者からの連絡をお待ちしています。

---
このメールは自動送信されました。
"""
        
        # メール送信
        recipients = ct.get_inquiry_email_recipients()
        logger.info(f"メール送信先: {recipients}")
        send_email(recipients, ct.EMAIL_SUBJECT, email_content)
        
        logger.info("会話履歴問い合わせ送信完了")
        return f"問い合わせを送信しました。担当者から連絡があるまでお待ちください。"
        
    except Exception as e:
        logger.error(f"会話履歴問い合わせ送信エラー: {e}")
        logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
        return f"問い合わせの送信に失敗しました。エラー: {str(e)}"

def get_conversation_summary():
    """
    会話履歴をまとめて読みやすい形式で返す
    
    Returns:
        str: 整形された会話履歴
    """
    if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
        return "会話履歴がありません。"
    
    summary = []
    for i, message in enumerate(st.session_state.messages, 1):
        role = "👤 ユーザー" if message["role"] == "user" else "🤖 AI"
        content = message["content"]
        summary.append(f"{i}. {role}: {content}\n")
    
    return "\n".join(summary)

def send_email(recipients, subject, content):
    """
    メール送信関数
    
    Args:
        recipients: 送信先メールアドレスのリスト
        subject: メール件名
        content: メール本文
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # SMTP設定の取得
        smtp_host = os.getenv("SMTP_HOST", ct.SMTP_SERVER)
        smtp_port = int(os.getenv("SMTP_PORT", ct.SMTP_PORT))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        logger.info(f"SMTP設定: {smtp_host}:{smtp_port}, ユーザー: {smtp_username}")
        
        if not smtp_username or not smtp_password:
            raise Exception("SMTP認証情報が設定されていません。環境変数SMTP_USERNAMEとSMTP_PASSWORDを設定してください。")
        
        # メール作成
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        
        # メール本文の追加
        msg.attach(MIMEText(content, 'plain', 'utf-8'))
        
        # メール送信
        logger.info("SMTP接続を開始")
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            logger.info("SMTP認証を開始")
            server.login(smtp_username, smtp_password)
            logger.info("メール送信を開始")
            server.send_message(msg)
            logger.info("メール送信完了")
            
    except Exception as e:
        logger.error(f"メール送信エラー: {e}")
        logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
        raise
