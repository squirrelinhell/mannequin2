#!/bin/bash

export PYTHONPATH="$(readlink -m ../..):$PYTHONPATH" || exit 1
export PATH="../_utils:$PATH"

VARIANTS=$(code-variants --print --copies 8 --run ./solve.py __variants) || exit 1

for v in $VARIANTS; do
    ( read -r line && echo "$line variant" && \
        while read -r line; do echo "$line $v"; done ) < "__variants/$v.out"
done > __variants/all

PLOT_FILE=__reward.png \
    marginal-plot --mean __variants/all reward step variant || exit 1
