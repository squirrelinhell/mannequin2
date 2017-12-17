#!/bin/bash

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$(pwd):$(pwd)/benchmarks:$PYTHONPATH"
export PATH="$(pwd)/scripts:$PATH"

FILES=
for pattern in "${@:-}"; do
    FILES="$FILES"$'\n'$(find benchmarks -maxdepth 1 \
        -name "$pattern"'*.py' -not -name '_*')
done
FILES=$(sort -u <<<"$FILES" | grep -v '^$')

for file in $FILES; do
    [ -f "$file" ] || file="benchmarks/$file.py"
    NAME="${file##*/}"
    NAME="${NAME%.*}"
    echo "Running $NAME... "

    case "$NAME" in
        cartpole*) COPIES=100 ;;
        *) COPIES=16 ;;
    esac

    case "$NAME" in
        walker*) PLOT_OPTS=(--mean --xmin=0 --xmax=420000
            --ymin=-200 --ymax=300) ;;
        *) PLOT_OPTS=(--mean) ;;
    esac

    OUT_DIR="benchmarks/__results_$NAME"

    VARIANTS=$(code-variants --print \
        --run --copies "$COPIES" "$file" "$OUT_DIR") || exit

    PLOT_COLUMS="reward steps"
    [ "x${VARIANTS::4}" = xcopy ] || PLOT_COLUMS="$PLOT_COLUMS variant"

    for dir in $OUT_DIR $(find benchmarks -maxdepth 1 -type d \
            -name "results_${NAME}_"'*'); do
        BASENAME="${dir##*/}"
        BASENAME="${BASENAME#__}"
        BASENAME="${BASENAME#results_}"
        PLOT_FILE="benchmarks/__$BASENAME.png"
        [ ! -e "$PLOT_FILE" ] || continue
        echo "Plotting $BASENAME..."
        HEADER=$(grep '^#' "$dir/$(head -n 1 <<<"$VARIANTS").out")
        echo "$HEADER variant" > "$TMPDIR/data"
        for v in $VARIANTS; do
            DATA=$(grep -v '^#' "$dir/$v.out") || exit 1
            while read -r line; do
                echo "$line ${v%_*}"
            done <<<"$DATA"
        done >> "$TMPDIR/data"
        PLOT_FILE="$PLOT_FILE" marginal-plot "${PLOT_OPTS[@]}" \
            "$TMPDIR/data" $PLOT_COLUMS || exit 1
    done
done
