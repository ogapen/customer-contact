"""
# =============================================================================
# 問い合わせ対応自動化AIエージェント - メイン処理
# =============================================================================
# このファイルは、Streamlitを使用したWebアプリケーションのメイン処理です。
# 顧客からの問い合わせに対してAIが自動で回答するシステムです。
"""

############################################################
# 必要なライブラリの読み込み
############################################################
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct

# =============================================================================
# アプリケーションの基本設定
# =============================================================================
# Streamlitページの設定
st.set_page_config(page_title=ct.APP_NAME)

# 環境変数の読み込み（Streamlit Cloud対応）
import os

def load_environment_variables():
    """
    環境変数を読み込む関数（Streamlit Cloud対応）
    """
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
            print("✅ .envファイルから環境変数を読み込みました。")
        else:
            print("⚠️  .envファイルが見つかりません。システムの環境変数またはStreamlit Secretsを使用します。")
            
    except Exception as e:
        print(f"⚠️  環境変数の読み込みでエラーが発生しました: {e}")
        print("システムの環境変数を使用します。")

# 環境変数の読み込み実行
load_environment_variables()

# ログシステムの準備
logger = logging.getLogger(ct.LOGGER_NAME)

# =============================================================================
# アプリケーションの初期化処理
# =============================================================================
try:
    # アプリケーションの初期化を実行
    initialize()
except Exception as e:
    # 初期化に失敗した場合のエラー処理
    error_message = str(e)
    
    # OpenAI APIキーのエラーの場合、特別な処理を行う
    if "401" in error_message and "invalid_api_key" in error_message:
        st.error("""
        🚨 **OpenAI APIキーが無効です**
        
        以下の手順でAPIキーを設定してください：
        
        1. [OpenAI API Keys](https://platform.openai.com/account/api-keys) にアクセス
        2. 新しいAPIキーを作成
        3. `.env`ファイルの`OPENAI_API_KEY`に設定
        4. アプリケーションを再起動
        
        現在のAPIキーの確認：
        - 環境変数の設定を確認してください
        - `.env`ファイルが正しく読み込まれているか確認してください
        """, icon=ct.ERROR_ICON)
    else:
        # その他のエラーの場合
        st.error(f"""
        🚨 **初期化処理でエラーが発生しました**
        
        エラーの詳細：
        {error_message}
        
        このエラーが繰り返し発生する場合は、管理者にお問い合わせください。
        """, icon=ct.ERROR_ICON)
    
    logger.error(f"初期化エラー: {error_message}")
    st.stop()

# アプリケーション起動ログの出力（初回のみ）
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

# =============================================================================
# 画面の初期表示
# =============================================================================
# アプリケーションのタイトル表示
cn.display_app_title()

# サイドバーの表示
cn.display_sidebar()

# AIからの初期メッセージ表示
cn.display_initial_ai_message()

# =============================================================================
# 画面のスタイリング
# =============================================================================
# CSSスタイルの適用
st.markdown(ct.STYLE, unsafe_allow_html=True)

# =============================================================================
# ユーザー入力の受け付け
# =============================================================================
# チャット入力フィールドの表示
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)

# =============================================================================
# 過去の会話履歴の表示
# =============================================================================
try:
    cn.display_conversation_log(chat_message)
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

# =============================================================================
# ユーザーからの新しいメッセージの処理
# =============================================================================
if chat_message:
    # トークン数の制限チェック
    input_tokens = len(st.session_state.enc.encode(chat_message))
    if input_tokens > ct.MAX_ALLOWED_TOKENS:
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            st.error(ct.INPUT_TEXT_LIMIT_ERROR_MESSAGE)
            st.stop()
    
    # トークン数を会話履歴に追加
    st.session_state.total_tokens += input_tokens

    # 1. ユーザーメッセージの画面表示
    logger.info({"message": chat_message})
    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(chat_message)
    
    # 2. AIからの回答生成
    try:
        with st.spinner(ct.SPINNER_TEXT):
            result = utils.execute_agent_or_chain(chat_message)
    except Exception as e:
        logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
        st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
        st.stop()
    
    # 3. 古い会話履歴の削除（メモリ管理）
    utils.delete_old_conversation_log(result)

    # 4. AIの回答を画面に表示
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        try:
            cn.display_llm_response(result)
            logger.info({"message": result})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()
    
    # 5. 会話履歴への保存
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": result})

# =============================================================================
# 問い合わせボタン
# =============================================================================
# 問い合わせボタンの表示
cn.display_inquiry_button()

# =============================================================================
# ユーザーフィードバック機能
# =============================================================================
# フィードバックボタンの表示
cn.display_feedback_button()