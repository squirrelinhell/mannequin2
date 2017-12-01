#!/bin/bash

export PYTHONPATH="$(readlink -m ../..):$PYTHONPATH" || exit 1
export PATH="../_utils:$PATH"

code-variants --copies 100 --run ./solve.py __variants || exit 1

cat __variants/*.out > __variants/all || exit 1

PLOT_FILE=__reward.png \
    marginal-plot --ci __variants/all reward episode || exit 1
