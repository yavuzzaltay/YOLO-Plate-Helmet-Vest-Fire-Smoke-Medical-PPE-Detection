#!/usr/bin/env bash
# YOLO Çoklu Tespit Platformu — Linux/macOS başlatıcı
# Kullanım:  bash scripts/run.sh
set -e
cd "$(dirname "$0")/.."
if [ -x "PlateDetection/.venv/bin/python" ]; then
  PYTHON="PlateDetection/.venv/bin/python"
else
  PYTHON="python3"
fi
"$PYTHON" -m streamlit run app.py --browser.gatherUsageStats=false
