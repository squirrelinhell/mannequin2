#!/bin/bash

export PYTHONPATH="$(readlink -m ../..):$PYTHONPATH" || exit 1
export PATH="../_utils:$PATH"

VARIANTS=$(code-variants --print --copies 16 --run ./solve.py __variants) || exit 1

for v in $VARIANTS; do
    info="${v%_*}"
    ( read -r line && echo "$line variant" && \
        while read -r line; do echo "$line $v"; done ) < "__variants/$v.out"
done > __variants/all

PLOT_FILE=__reward.png \
    marginal-plot --mean \
        --xmin=0 --xmax=420000 \
        --ymin=-200 --ymax=400 \
        __variants/all reward steps || exit 1
