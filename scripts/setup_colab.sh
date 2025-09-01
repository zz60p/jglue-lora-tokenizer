#!/bin/bash
set -e
python -V
pip -q install -U pip
pip -q install -r requirements.txt
python - << 'PY'
import fugashi, sudachipy, transformers, datasets, peft, bitsandbytes
print("Sanity check passed.")
PY
echo "Environment ready."
