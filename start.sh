#!/bin/bash
# =============================================================================
# アプリケーション起動スクリプト
# =============================================================================
# このスクリプトは、顧客問い合わせ対応AIアプリケーションを起動します。

echo "🚀 顧客問い合わせ対応AIアプリケーションを起動します..."

# 仮想環境のPythonパス
PYTHON_PATH="/Users/taken/Documents/python/生成AIエンジニア_Lesson23_サンプルアプリ2/ダウンロード用/Slack連携なし/customer-contact/.venv/bin/python"

# 既存のStreamlitプロセスを停止
echo "📋 既存のプロセスを停止中..."
pkill -f "streamlit run" 2>/dev/null || true

# 少し待機
sleep 2

# アプリケーションを起動
echo "🎯 アプリケーションを起動中..."
echo "📍 URL: http://localhost:8504"
echo "⏹️  停止するには Ctrl+C を押してください"

$PYTHON_PATH -m streamlit run main.py --server.port 8504
