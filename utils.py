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
# 問い合わせ処理とメール送信機能
############################################################

def process_inquiry(message, user_name="お客様"):
    """
    問い合わせ処理を実行し、メール通知を送信します。
    
    Args:
        message: 問い合わせ内容
        user_name: 問い合わせ者名
    
    Returns:
        処理結果メッセージ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # 問い合わせ内容の分析
        inquiry_info = analyze_inquiry(message)
        
        # 担当者の割り振り
        assigned_staff = assign_staff_member(inquiry_info)
        
        # 回答案の生成
        response_suggestions = generate_response_suggestions(inquiry_info)
        
        # メール送信
        send_inquiry_email(message, user_name, inquiry_info, assigned_staff, response_suggestions)
        
        logger.info(f"問い合わせ処理完了: {inquiry_info['category']}")
        
        return "問い合わせを受け付けました。担当者にメール通知を送信しました。"
        
    except Exception as e:
        logger.error(f"問い合わせ処理エラー: {e}")
        return "問い合わせの処理中にエラーが発生しました。"

def analyze_inquiry(message):
    """
    問い合わせ内容を分析してカテゴリを判定します。
    
    Args:
        message: 問い合わせ内容
    
    Returns:
        分析結果（カテゴリ、緊急度など）
    """
    # 問い合わせカテゴリの判定
    categories = {
        "技術的なトラブル対応": ["ログイン", "エラー", "動かない", "使えない", "不具合", "システム", "接続"],
        "商品・サービス問い合わせ": ["商品", "サービス", "料金", "価格", "プラン", "機能", "仕様"],
        "契約・請求関連": ["契約", "請求", "支払い", "料金", "解約", "更新", "変更"],
        "その他の問い合わせ": []
    }
    
    message_lower = message.lower()
    determined_category = "その他の問い合わせ"
    
    for category, keywords in categories.items():
        if any(keyword in message_lower for keyword in keywords):
            determined_category = category
            break
    
    return {
        "category": determined_category,
        "content": message,
        "urgency": "普通",  # 緊急度の判定ロジックを追加可能
        "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    }

def assign_staff_member(inquiry_info):
    """
    問い合わせ内容に基づいて担当者を割り振ります。
    
    Args:
        inquiry_info: 問い合わせ情報
    
    Returns:
        割り振られた担当者情報
    """
    # 模擬的な担当者データ
    staff_members = {
        "技術的なトラブル対応": [
            {
                "name": "棚橋由香里",
                "department": "技術部",
                "position": "課長",
                "specialty": "品質管理とテスト実施",
                "experience": "過去に同様の問い合わせに対応した経験があり"
            },
            {
                "name": "山本和也",
                "department": "技術部",
                "position": "エンジニア",
                "specialty": "製品の技術サポート",
                "experience": "過去に同様のログイン問題に迅速に対応した実績があり"
            }
        ],
        "商品・サービス問い合わせ": [
            {
                "name": "佐藤美咲",
                "department": "営業部",
                "position": "主任",
                "specialty": "商品企画とマーケティング",
                "experience": "商品に関する豊富な知識を持ち"
            }
        ],
        "契約・請求関連": [
            {
                "name": "田中健太",
                "department": "経理部",
                "position": "係長",
                "specialty": "契約管理と請求処理",
                "experience": "契約関連の問い合わせに精通しており"
            }
        ]
    }
    
    category = inquiry_info["category"]
    if category in staff_members:
        return staff_members[category]
    else:
        # デフォルト担当者
        return [{
            "name": "総合受付",
            "department": "カスタマーサービス",
            "position": "担当者",
            "specialty": "総合的なカスタマーサポート",
            "experience": "幅広い問い合わせに対応可能で"
        }]

def generate_response_suggestions(inquiry_info):
    """
    問い合わせ内容に基づいて回答案を生成します。
    
    Args:
        inquiry_info: 問い合わせ情報
    
    Returns:
        回答案のリスト
    """
    category = inquiry_info["category"]
    
    response_templates = {
        "技術的なトラブル対応": [
            {
                "content": "パスワードリセットを行うための手順を案内します。",
                "reasoning": "過去の問い合わせ履歴から、パスワードリセットが多くのログイン問題の解決に寄与しているため。"
            },
            {
                "content": "システムのサポートチームに連絡し、直接サポートを受けることを提案します。",
                "reasoning": "技術的なトラブルの場合、専門のサポートが必要なケースが多いため。"
            },
            {
                "content": "ログイン試行時のエラーメッセージを確認し、それに基づいた対応を行います。",
                "reasoning": "エラーメッセージは問題の特定に役立つ情報を提供するため。"
            }
        ],
        "商品・サービス問い合わせ": [
            {
                "content": "商品の詳細な仕様書をお送りし、ご要望に最適なプランをご提案します。",
                "reasoning": "商品について詳しく知ることで、適切な選択ができるため。"
            },
            {
                "content": "無料トライアルをご利用いただき、実際にサービスを体験していただくことを提案します。",
                "reasoning": "実際の使用感を確認することで、満足度の高い導入が可能なため。"
            },
            {
                "content": "類似の導入事例をご紹介し、具体的な活用方法をご説明します。",
                "reasoning": "他社の成功事例を参考にすることで、効果的な利用方法が分かるため。"
            }
        ],
        "契約・請求関連": [
            {
                "content": "契約内容の詳細をご説明し、ご不明な点を解決いたします。",
                "reasoning": "契約内容の理解不足が問題の原因となるケースが多いため。"
            },
            {
                "content": "請求書の内訳を詳しく説明し、疑問点を解消いたします。",
                "reasoning": "請求内容の透明性を高めることで、信頼関係の構築につながるため。"
            },
            {
                "content": "支払い方法の変更や分割払いなど、柔軟な対応を検討します。",
                "reasoning": "顧客の状況に応じた支払い方法の提案により、継続的な関係を維持できるため。"
            }
        ]
    }
    
    if category in response_templates:
        return response_templates[category]
    else:
        return [
            {
                "content": "お問い合わせの内容を詳しく確認し、最適な解決策をご提案します。",
                "reasoning": "個別の事情に応じた対応が最も効果的なため。"
            },
            {
                "content": "関連部署と連携し、包括的なサポートを提供します。",
                "reasoning": "複数の部署が連携することで、より充実したサポートが可能なため。"
            },
            {
                "content": "追加の資料やサポートツールをご提供し、問題解決をサポートします。",
                "reasoning": "適切な情報提供により、お客様自身での問題解決が促進されるため。"
            }
        ]

def send_inquiry_email(message, user_name, inquiry_info, assigned_staff, response_suggestions):
    """
    問い合わせ通知メールを送信します。
    
    Args:
        message: 問い合わせ内容
        user_name: 問い合わせ者名
        inquiry_info: 問い合わせ情報
        assigned_staff: 割り振られた担当者
        response_suggestions: 回答案
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # メール本文の作成
        email_body = create_email_body(message, user_name, inquiry_info, assigned_staff, response_suggestions)
        
        # 環境変数からメールアドレスを取得
        email_recipients = ct.get_inquiry_email_recipients()
        
        # メール送信（実際の実装では、適切なSMTP設定が必要）
        logger.info("問い合わせメール送信（模擬）")
        logger.info(f"送信先: {email_recipients}")
        logger.info(f"件名: {ct.EMAIL_SUBJECT}")
        logger.info(f"本文: {email_body}")
        
        # 実際のメール送信コード（コメントアウト）
        # send_email_smtp(email_recipients, ct.EMAIL_SUBJECT, email_body)
        
    except Exception as e:
        logger.error(f"メール送信エラー: {e}")
        raise

def create_email_body(message, user_name, inquiry_info, assigned_staff, response_suggestions):
    """
    メール本文を作成します。
    
    Args:
        message: 問い合わせ内容
        user_name: 問い合わせ者名
        inquiry_info: 問い合わせ情報
        assigned_staff: 割り振られた担当者
        response_suggestions: 回答案
    
    Returns:
        メール本文
    """
    # 担当者情報の整理
    staff_info = ""
    for staff in assigned_staff:
        staff_info += f"{staff['name']}さんは、{staff['department']}の{staff['position']}として、{staff['specialty']}に関する専門知識を持っています。\n\n"
        staff_info += f"{staff['experience']}ます。\n\n"
    
    # 回答案の整理
    suggestions_text = ""
    for i, suggestion in enumerate(response_suggestions, 1):
        suggestions_text += f"＜{i}つ目＞\n\n"
        suggestions_text += f"●内容: {suggestion['content']}\n\n"
        suggestions_text += f"●根拠: {suggestion['reasoning']}\n\n"
    
    # メール本文のテンプレート
    email_body = f"""こちらは顧客問い合わせに対しての「担当者割り振り」と「回答・対応案の提示」を自動で行うAIアシスタントです。担当者は問い合わせ内容を確認し、対応してください。

================================

【問い合わせ情報】

・問い合わせ内容: {message}

・カテゴリ: {inquiry_info['category']}

・問い合わせ者: {user_name}

・日時: {inquiry_info['timestamp']}

-------------------

【メンション先の選定理由】

{staff_info}

-------------------

【回答・対応案】

{suggestions_text}

-------------------

【参照資料】

・従業員情報.csv

・問い合わせ履歴.csv
"""
    
    return email_body

def send_email_smtp(recipients, subject, body):
    """
    SMTPを使用してメールを送信します。
    
    Args:
        recipients: 送信先メールアドレスのリスト
        subject: 件名
        body: 本文
    """
    # 実際の実装では、環境変数からSMTP設定を読み込む
    # smtp_server = os.getenv("SMTP_SERVER", ct.SMTP_SERVER)
    # smtp_port = int(os.getenv("SMTP_PORT", ct.SMTP_PORT))
    # smtp_username = os.getenv("SMTP_USERNAME")
    # smtp_password = os.getenv("SMTP_PASSWORD")
    
    # 実際のメール送信処理（コメントアウト）
    # msg = MIMEMultipart()
    # msg['From'] = smtp_username
    # msg['Subject'] = subject
    # msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    # try:
    #     server = smtplib.SMTP(smtp_server, smtp_port)
    #     server.starttls()
    #     server.login(smtp_username, smtp_password)
    #     
    #     for recipient in recipients:
    #         msg['To'] = recipient
    #         server.send_message(msg)
    #         del msg['To']
    #     
    #     server.quit()
    # except Exception as e:
    #     raise e
    
    pass  # 実際の実装では上記のコードを有効化
