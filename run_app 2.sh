#!/bin/bash

# 作業ディレクトリに移動
cd "$(dirname "$0")"

# 正しい仮想環境のPythonを使用してStreamlitを実行
exec /Users/taken/Documents/python/生成AIエンジニア_Lesson23_サンプルアプリ2/ダウンロード用/Slack連携なし/customer-contact/.venv/bin/python -m streamlit run main.py "$@"
