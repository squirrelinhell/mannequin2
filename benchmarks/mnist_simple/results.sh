#!/bin/bash

export PYTHONPATH="$(readlink -m ../..):$PYTHONPATH" || exit 1
export PATH="../_utils:$PATH"

[ -e __data.npz ] || ./get_data.py || exit 1

code-variants --run ./solve.py __variants || exit 1

cat __variants/*.out | grep -v ^test > __variants/train || exit 1
cat __variants/*.out | grep -v ^train > __variants/test || exit 1

PLOT_FILE=__train_error.png \
    marginal-plot --mean __variants/train 'log(1.0-accuracy)' 'log(batch)' 'run_id' || exit 1

PLOT_FILE=__test_error.png \
    marginal-plot --ci __variants/test 'log(1.0-accuracy)' 'batch' || exit 1
