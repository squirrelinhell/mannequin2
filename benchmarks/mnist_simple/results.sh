#!/bin/bash

export PYTHONPATH="$(readlink -m ../..):$PYTHONPATH" || exit 1
export PATH="../_utils:$PATH"

[ -e __data.npz ] || ./get_data.py || exit 1

VARIANTS=$(code-variants --print --run ./solve.py __variants) || exit 1

for v in $VARIANTS; do
    ( read -r line && echo "$line variant" && \
        while read -r line; do echo "$line $v"; done ) < "__variants/$v.out"
done > __variants/all

PLOT_FILE=__log_error.png \
    marginal-plot --mean __variants/all 'log(1.0-accuracy)' 'log(samples/10000)' type variant || exit 1
