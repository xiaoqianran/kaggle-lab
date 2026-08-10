#!/bin/sh
# Source this to use modern Kaggle CLI (KGAT token) in this sandbox.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null)}"
if [ -x /workspace/.venv-kaggle/bin/kaggle ]; then
  export PATH="/workspace/.venv-kaggle/bin:$PATH"
fi
