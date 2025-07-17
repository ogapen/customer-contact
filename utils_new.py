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
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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
