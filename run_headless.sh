#!/usr/bin/env bash

export PATH="/home/ubuntu/.local/bin:$PATH"
export OPTUNA_URL="mysql://optuna:{{ lookup('env', 'OPTUNA_DB_PASS') }}@{{ lookup('file', 'inventory-db-private').splitlines()[0] }}/optuna"

# run main notebook
jupyter nbconvert --to notebook --execute main_notebook.ipynb --output=output.ipynb --ExecutePreprocessor.timeout=-1 1>stdout.txt 2>stderr.txt || true

logger "all done"
sudo shutdown
