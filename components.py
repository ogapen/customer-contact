"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import logging
import streamlit as st
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_sidebar():
    """
    サイドバーの表示
    """
    with st.sidebar:
        st.markdown("## AIエージェント機能の利用有無")

        col1, col2 = st.columns([100, 1])
        with col1:
            st.session_state.agent_mode = st.selectbox(
                label="AIエージェント機能の選択",
                options=[ct.AI_AGENT_MODE_ON, ct.AI_AGENT_MODE_OFF],
                label_visibility="collapsed"
            )
        
        st.divider()

        st.markdown("**【AIエージェントとは】**")
        st.code("質問に対して適切と考えられる回答を生成できるまで、生成AIロボット自身に試行錯誤してもらえる機能です。自身の回答に対して評価・改善を繰り返すことで、より優れた回答を生成できます。", wrap_lines=True)
        st.warning("AIエージェント機能を利用する場合、回答生成により多くの時間を要する可能性が高いです。", icon=":material/warning:")


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        st.success("こちらは弊社に関する質問にお答えする生成AIチャットボットです。AIエージェントの利用有無を選択し、画面下部のチャット欄から質問してください。")
        st.warning("具体的に入力したほうが期待通りの回答を得やすいです。", icon=ct.WARNING_ICON)


def display_conversation_log(chat_message):
    """
    会話ログの一覧表示
    """
    # 会話ログの最後を表示する時のみ、フィードバック後のメッセージ表示するために「何番目のメッセージか」を取得
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=ct.AI_ICON_FILE_PATH):
                st.markdown(message["content"])
                # フィードバックエリアの表示
                display_after_feedback_message(index, chat_message)
        else:
            with st.chat_message(message["role"], avatar=ct.USER_ICON_FILE_PATH):
                st.markdown(message["content"])
                # フィードバックエリアの表示
                display_after_feedback_message(index, chat_message)


def display_after_feedback_message(index, chat_message):
    """
    ユーザーフィードバック後のメッセージ表示

    Args:
        result: LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # フィードバックで「いいえ」を選択するとno_flgがTrueになるため、再度フィードバックの入力エリアが表示されないようFalseにする
    if st.session_state.feedback_no_flg and chat_message:
        st.session_state.feedback_no_flg = False

    # 会話ログの最後のメッセージに対しての処理
    if index == len(st.session_state.messages) - 1:
        # フィードバックで「はい」が選択されたらThanksメッセージを表示し、フラグを下ろす
        if st.session_state.feedback_yes_flg:
            st.caption(ct.FEEDBACK_YES_MESSAGE)
            st.session_state.feedback_yes_flg = False
        # フィードバックで「いいえ」が選択されたら、入力エリアを表示する
        if st.session_state.feedback_no_flg:
            st.caption(ct.FEEDBACK_NO_MESSAGE)
            st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
            # 送信ボタンの表示
            if st.button(ct.FEEDBACK_BUTTON_LABEL):
                # 回答への不満足理由をログファイルに出力
                if st.session_state.dissatisfied_reason:
                    logger.info({"dissatisfied_reason": st.session_state.dissatisfied_reason})
                # 送信ボタン押下後、再度入力エリアが表示されないようにするのと、Thanksメッセージを表示するためにフラグを更新
                st.session_state.feedback_no_flg = False
                st.session_state.feedback_no_reason_send_flg = True
                st.rerun()
        # 入力エリアから送信ボタンが押された後、再度Thanksメッセージが表示されないようにフラグを下ろし、Thanksメッセージを表示
        if st.session_state.feedback_no_reason_send_flg:
            st.session_state.feedback_no_reason_send_flg = False
            st.caption(ct.FEEDBACK_THANKS_MESSAGE)


def display_llm_response(result):
    """
    LLMからの回答表示

    Args:
        result: LLMからの回答
    """
    st.markdown(result)
    # フィードバックボタンを表示する場合のみ、メッセージ表示
    if st.session_state.answer_flg:
        st.caption(ct.FEEDBACK_REQUIRE_MESSAGE)


def display_feedback_button():
    """
    フィードバックボタンの表示
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # LLMによる回答後のみにフィードバックボタンを表示
    if st.session_state.answer_flg:
        col1, col2, col3 = st.columns([1, 1, 5])
        # 良い回答が得られたことをフィードバックするためのボタン
        with col1:
            if st.button(ct.FEEDBACK_YES):
                # フィードバックをログファイルに出力
                logger.info({"feedback": ct.SATISFIED})
                # 再度フィードバックボタンが表示されないよう、フラグを下ろす
                st.session_state.answer_flg = False
                # 「はい」ボタンを押下後、Thanksメッセージを表示するためのフラグ立て
                st.session_state.feedback_yes_flg = True
                # 画面の際描画
                st.rerun()
        # 回答に満足できなかったことをフィードバックするためのボタン
        with col2:
            if st.button(ct.FEEDBACK_NO):
                # フィードバックをログファイルに出力
                logger.info({"feedback": ct.DISSATISFIED})
                # 再度フィードバックボタンが表示されないよう、フラグを下ろす
                st.session_state.answer_flg = False
                # 「はい」ボタンを押下後、フィードバックの入力エリアを表示するためのフラグ立て
                st.session_state.feedback_no_flg = True
                # 画面の際描画
                st.rerun()


def display_inquiry_button():
    """
    問い合わせボタンの表示
    """
    import utils
    
    # 会話履歴がある場合のみボタンを表示
    if hasattr(st.session_state, 'messages') and st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 さらに詳しい情報が必要な場合")
        st.markdown(ct.INQUIRY_BUTTON_TEXT)
        
        if st.button(ct.INQUIRY_BUTTON_LABEL, type="primary"):
            with st.spinner("問い合わせを送信中..."):
                result = utils.send_conversation_inquiry()
                if "送信しました" in result:
                    st.success(result)
                else:
                    st.error(result)