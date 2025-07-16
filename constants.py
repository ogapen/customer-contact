"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "問い合わせ対応自動化AIエージェント"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
APP_BOOT_MESSAGE = "アプリが起動されました。"
USER_ICON_FILE_PATH = os.path.join(os.path.dirname(__file__), "images", "user_icon.jpg")
AI_ICON_FILE_PATH = os.path.join(os.path.dirname(__file__), "images", "ai_icon.jpg")
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."


# ==========================================
# ユーザーフィードバック関連
# ==========================================
FEEDBACK_YES = "はい"
FEEDBACK_NO = "いいえ"

SATISFIED = "回答に満足した"
DISSATISFIED = "回答に満足しなかった"

FEEDBACK_REQUIRE_MESSAGE = "この回答はお役に立ちましたか？フィードバックをいただくことで、生成AIの回答の質が向上します。"
FEEDBACK_BUTTON_LABEL = "送信"
FEEDBACK_YES_MESSAGE = "ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！"
FEEDBACK_NO_MESSAGE = "ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。"
FEEDBACK_THANKS_MESSAGE = "ご回答いただき誠にありがとうございます。"


# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5


# ==========================================
# トークン関連
# ==========================================
MAX_ALLOWED_TOKENS = 1000
ENCODING_KIND = "cl100k_base"


# ==========================================
# RAG参照用のデータソース系
# ==========================================
RAG_TOP_FOLDER_PATH = "./data"

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8")
}

DB_ALL_PATH = "./.db_all"
DB_COMPANY_PATH = "./.db_company"


# ==========================================
# AIエージェント関連
# ==========================================
AI_AGENT_MAX_ITERATIONS = 5

DB_SERVICE_PATH = "./.db_service"
DB_CUSTOMER_PATH = "./.db_customer"

DB_NAMES = {
    DB_COMPANY_PATH: f"{RAG_TOP_FOLDER_PATH}/company",
    DB_SERVICE_PATH: f"{RAG_TOP_FOLDER_PATH}/service",
    DB_CUSTOMER_PATH: f"{RAG_TOP_FOLDER_PATH}/customer"
}

AI_AGENT_MODE_ON = "利用する"
AI_AGENT_MODE_OFF = "利用しない"

SEARCH_COMPANY_INFO_TOOL_NAME = "search_company_info_tool"
SEARCH_COMPANY_INFO_TOOL_DESCRIPTION = "自社「株式会社EcoTee」に関する情報を参照したい時に使う"
SEARCH_SERVICE_INFO_TOOL_NAME = "search_service_info_tool"
SEARCH_SERVICE_INFO_TOOL_DESCRIPTION = "自社サービス「EcoTee」に関する情報を参照したい時に使う"
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME = "search_customer_communication_tool"
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION = "顧客とのやりとりに関する情報を参照したい時に使う"
SEARCH_WEB_INFO_TOOL_NAME = "search_web_tool"
SEARCH_WEB_INFO_TOOL_DESCRIPTION = "自社サービス「HealthX」に関する質問で、Web検索が必要と判断した場合に使う"

# 新しいツールの定義
SEARCH_ALL_DOCUMENTS_TOOL_NAME = "search_all_documents_tool"
SEARCH_ALL_DOCUMENTS_TOOL_DESCRIPTION = "全社文書から横断的に情報を検索したい時に使う（会社情報、サービス情報、顧客情報をまとめて検索）"
CURRENT_TIME_TOOL_NAME = "get_current_time_tool"
CURRENT_TIME_TOOL_DESCRIPTION = "現在の日時を取得したい時に使う"
CALCULATE_TOOL_NAME = "calculate_tool"
CALCULATE_TOOL_DESCRIPTION = "数値計算を行いたい時に使う（料金計算、期間計算など）"
EMAIL_VALIDATION_TOOL_NAME = "email_validation_tool"
EMAIL_VALIDATION_TOOL_DESCRIPTION = "メールアドレスの形式をチェックしたい時に使う"
ORDER_STATUS_TOOL_NAME = "check_order_status_tool"
ORDER_STATUS_TOOL_DESCRIPTION = "注文状況や配送状況を確認したい時に使う（模擬データを使用）"
FAQ_SEARCH_TOOL_NAME = "faq_search_tool"
FAQ_SEARCH_TOOL_DESCRIPTION = "よくある質問（FAQ）から関連する情報を検索したい時に使う"
CONTACT_INFO_TOOL_NAME = "get_contact_info_tool"
CONTACT_INFO_TOOL_DESCRIPTION = "会社の連絡先情報を取得したい時に使う"
BUSINESS_HOURS_TOOL_NAME = "get_business_hours_tool"
BUSINESS_HOURS_TOOL_DESCRIPTION = "営業時間やサポート時間を確認したい時に使う"
PRICE_CALCULATOR_TOOL_NAME = "price_calculator_tool"
PRICE_CALCULATOR_TOOL_DESCRIPTION = "商品価格の計算や割引適用の計算を行いたい時に使う"
WEATHER_TOOL_NAME = "get_weather_info_tool"
WEATHER_TOOL_DESCRIPTION = "天気情報を取得したい時に使う（配送に関する問い合わせ等で使用）"
LANGUAGE_DETECTOR_TOOL_NAME = "detect_language_tool"
LANGUAGE_DETECTOR_TOOL_DESCRIPTION = "入力テキストの言語を検出したい時に使う"


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

NO_DOC_MATCH_MESSAGE = "回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。"

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内文書を基に、顧客からの問い合わせに対応するアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。

    {context}
"""


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
GET_LLM_RESPONSE_ERROR_MESSAGE = "回答生成に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
INPUT_TEXT_LIMIT_ERROR_MESSAGE = f"入力されたテキストの文字数が受付上限値（{MAX_ALLOWED_TOKENS}）を超えています。受付上限値を超えないよう、再度入力してください。"


# ==========================================
# スタイリング
# ==========================================
STYLE = """
<style>
    .stHorizontalBlock {
        margin-top: -14px;
    }
    .stChatMessage + .stHorizontalBlock {
        margin-left: 56px;
    }
    .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
        margin-left: -24px;
    }
    @media screen and (max-width: 480px) {
        .stChatMessage + .stHorizontalBlock {
            flex-wrap: nowrap;
            margin-left: 56px;
        }
        .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
            margin-left: -206px;
        }
    }
</style>
"""


# ==========================================
# 問い合わせモード関連
# ==========================================
INQUIRY_MODE_ON = "ON"
INQUIRY_MODE_OFF = "OFF"

# メール設定
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SUBJECT = "【問い合わせ自動通知】新しい問い合わせが届きました"

# 従業員情報ファイル
EMPLOYEE_INFO_FILE = "従業員情報.csv"
INQUIRY_HISTORY_FILE = "問い合わせ履歴.csv"

# 問い合わせメール送信先
INQUIRY_EMAIL_RECIPIENTS = [
    "kenta@ogawara-tosouten.com",        # 管理者メール（実際のメールアドレスに変更してください）
    "ogapen@gmail.com",      # サポートメール（実際のメールアドレスに変更してください）
    "manager@yourcompany.com"       # マネージャーメール（必要に応じて追加）
]

# 環境変数からメールアドレスを取得（設定されている場合）
def get_inquiry_email_recipients():
    """
    環境変数またはデフォルト設定から問い合わせメール送信先を取得
    """
    env_recipients = os.getenv("INQUIRY_EMAIL_RECIPIENTS", "")
    if env_recipients:
        # カンマ区切りで複数のメールアドレスを分割
        return [email.strip() for email in env_recipients.split(",")]
    else:
        # デフォルトのメールアドレス
        return INQUIRY_EMAIL_RECIPIENTS