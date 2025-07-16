"""
# =============================================================================
# アプリケーション初期化処理
# =============================================================================
# このファイルは、Streamlitアプリケーションの初期化処理を行います。
# 画面を最初に読み込む際に実行される重要な処理が含まれています。
"""

############################################################
# 必要なライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct

# =============================================================================
# 環境変数の読み込み（Streamlit Cloud対応）
# =============================================================================
def load_environment_variables():
    """
    環境変数を読み込む関数（Streamlit Cloud対応）
    """
    import os
    
    try:
        # .envファイルから読み込み（ローカル環境用）
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                        
    except Exception as e:
        print(f"⚠️  環境変数の読み込みでエラーが発生しました: {e}")

# 環境変数の読み込み実行
load_environment_variables()

############################################################
# メイン初期化関数
############################################################
def initialize():
    """
    # アプリケーションの初期化を実行
    この関数は以下の処理を順番に実行します：
    1. セッション状態の初期化
    2. セッションIDの生成
    3. ログシステムの設定
    4. AIエージェントの作成
    """
    try:
        # 1. セッション状態の初期化
        initialize_session_state()
        
        # 2. セッションIDの生成
        initialize_session_id()
        
        # 3. ログシステムの設定
        initialize_logger()
        
        # 4. AIエージェントの作成
        initialize_agent_executor()
        
    except Exception as e:
        # 初期化処理でエラーが発生した場合の詳細なエラー情報を表示
        error_message = f"初期化処理でエラーが発生しました: {str(e)}"
        print(f"🚨 {error_message}")
        # エラーを再発生させて上位の処理に伝える
        raise Exception(error_message)

############################################################
# セッション状態の初期化
############################################################
def initialize_session_state():
    """
    # Streamlitのセッション状態を初期化
    チャット機能に必要な変数を初期化します。
    """
    if "messages" not in st.session_state:
        # チャット履歴を保存するリスト
        st.session_state.messages = []
        st.session_state.chat_history = []
        
        # トークン数の管理
        st.session_state.total_tokens = 0
        
        # フィードバック機能のフラグ
        st.session_state.feedback_yes_flg = False
        st.session_state.feedback_no_flg = False
        st.session_state.answer_flg = False
        st.session_state.dissatisfied_reason = ""
        st.session_state.feedback_no_reason_send_flg = False
        
        # 問い合わせモードの初期化
        st.session_state.inquiry_mode = ct.INQUIRY_MODE_OFF

############################################################
# セッションID生成
############################################################
def initialize_session_id():
    """
    # ユニークなセッションIDを生成
    各ユーザーセッションを識別するためのIDを作成します。
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

############################################################
# ログシステムの設定
############################################################
def initialize_logger():
    """
    # ログ出力システムの設定
    アプリケーションの動作を記録するログシステムを設定します。
    """
    # ログ保存用のディレクトリを作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # すでにログハンドラーが設定されている場合はスキップ
    if logger.hasHandlers():
        return
    
    # ログファイルの設定（日付ごとにローテーション）
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    
    # ログの出力形式を設定
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

############################################################
# AIエージェントの作成
############################################################
def initialize_agent_executor():
    """
    # AIエージェント（ChatGPT）の設定
    質問に答えるAIエージェントを作成し、必要なツールを設定します。
    """
    try:
        logger = logging.getLogger(ct.LOGGER_NAME)
        
        # すでにエージェントが作成済みの場合はスキップ
        if "agent_executor" in st.session_state:
            return
        
        # OpenAI APIキーの確認
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or not openai_api_key.strip():
            raise Exception("OPENAI_API_KEYが設定されていません。環境変数を確認してください。")
        
        # トークン数をカウントするためのエンコーダーを設定
        st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
        
        # ChatGPTのLLMを設定
        st.session_state.llm = ChatOpenAI(
            model_name=ct.MODEL, 
            temperature=ct.TEMPERATURE, 
            streaming=True
        )
        
        # 各種データベースのRAGチェーンを作成
        st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
        st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
        st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
        st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)
        
        # エージェントが使用できるツールの一覧を作成
        tools = [
            # 会社情報検索ツール
            Tool(
                name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
                func=utils.run_company_doc_chain,
                description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
            ),
            # サービス情報検索ツール
            Tool(
                name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
                func=utils.run_service_doc_chain,
                description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
            ),
            # 顧客情報検索ツール
            Tool(
                name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
                func=utils.run_customer_doc_chain,
                description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
            ),
            # 全文書横断検索ツール
            Tool(
                name=ct.SEARCH_ALL_DOCUMENTS_TOOL_NAME,
                func=utils.run_all_documents_chain,
                description=ct.SEARCH_ALL_DOCUMENTS_TOOL_DESCRIPTION
            ),
            # 現在時刻取得ツール
            Tool(
                name=ct.CURRENT_TIME_TOOL_NAME,
                func=utils.get_current_time,
                description=ct.CURRENT_TIME_TOOL_DESCRIPTION
            ),
            # 計算ツール
            Tool(
                name=ct.CALCULATE_TOOL_NAME,
                func=utils.calculate_expression,
                description=ct.CALCULATE_TOOL_DESCRIPTION
            ),
            # メールアドレス検証ツール
            Tool(
                name=ct.EMAIL_VALIDATION_TOOL_NAME,
                func=utils.validate_email,
                description=ct.EMAIL_VALIDATION_TOOL_DESCRIPTION
            ),
            # 注文状況確認ツール
            Tool(
                name=ct.ORDER_STATUS_TOOL_NAME,
                func=utils.check_order_status,
                description=ct.ORDER_STATUS_TOOL_DESCRIPTION
            ),
            # FAQ検索ツール
            Tool(
                name=ct.FAQ_SEARCH_TOOL_NAME,
                func=utils.search_faq,
                description=ct.FAQ_SEARCH_TOOL_DESCRIPTION
            ),
            # 連絡先情報取得ツール
            Tool(
                name=ct.CONTACT_INFO_TOOL_NAME,
                func=utils.get_contact_info,
                description=ct.CONTACT_INFO_TOOL_DESCRIPTION
            ),
            # 営業時間取得ツール
            Tool(
                name=ct.BUSINESS_HOURS_TOOL_NAME,
                func=utils.get_business_hours,
                description=ct.BUSINESS_HOURS_TOOL_DESCRIPTION
            ),
            # 価格計算ツール
            Tool(
                name=ct.PRICE_CALCULATOR_TOOL_NAME,
                func=utils.calculate_price,
                description=ct.PRICE_CALCULATOR_TOOL_DESCRIPTION
            ),
            # 天気情報取得ツール
            Tool(
                name=ct.WEATHER_TOOL_NAME,
                func=utils.get_weather_info,
                description=ct.WEATHER_TOOL_DESCRIPTION
            ),
            # 言語検出ツール
            Tool(
                name=ct.LANGUAGE_DETECTOR_TOOL_NAME,
                func=utils.detect_language,
                description=ct.LANGUAGE_DETECTOR_TOOL_DESCRIPTION
            )
        ]
        
        # Web検索ツールをオプションで追加（APIキーが設定されている場合のみ）
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key and serpapi_key.strip():
            try:
                search = SerpAPIWrapper()
                tools.append(
                    Tool(
                        name=ct.SEARCH_WEB_INFO_TOOL_NAME,
                        func=search.run,
                        description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
                    )
                )
                logger.info("Web検索ツールを有効化しました")
            except Exception as e:
                logger.warning(f"Web検索ツールの初期化に失敗: {e}")
        else:
            logger.info("SERPAPI_API_KEYが未設定のため、Web検索ツールは無効化されました")
        
        # AIエージェントを作成
        st.session_state.agent_executor = initialize_agent(
            llm=st.session_state.llm,
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        logger.info("AIエージェントの初期化が完了しました")
        
    except Exception as e:
        error_message = f"AIエージェントの初期化に失敗しました: {str(e)}"
        print(f"🚨 {error_message}")
        raise Exception(error_message)