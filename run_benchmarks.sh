#!/bin/bash

export PYTHONPATH="$(pwd):$(pwd)/benchmarks:$PYTHONPATH"
export PATH="$(pwd)/scripts:$PATH"

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

mkdir "$TMPDIR/scripts"
find benchmarks -maxdepth 1 -type f -name '*.py' -not -name '_*' | \
    while read f; do
        NAME="${f##*/}"
        NAME="${NAME%.py}"
        if grep -F -q '###' "$f"; then
            code-variants "$f" "$TMPDIR/scripts/${NAME}_"
        else
            cat "$f" > "$TMPDIR/scripts/${NAME}"
        fi
    done

FILES=
for pattern in "${@:-}"; do
    FILES="$FILES"$'\n'$(find "$TMPDIR/scripts" -mindepth 1 \
        -maxdepth 1 -name "$pattern"'*')
done
FILES=$(sort -u <<<"$FILES" | grep -v '^$')

for file in $FILES; do
    NAME="${file##*/}"
    echo "Running $NAME... "

    case "$NAME" in
        cartpole*) COPIES=100 ;;
        *) COPIES=16 ;;
    esac

    OUT_DIR="benchmarks/__results_$NAME"
    VARIANTS=$(code-variants --print \
        --run --copies "$COPIES" "$file" "$OUT_DIR/") || exit
done

mkdir "$TMPDIR/plots"
for dir in $(find benchmarks -mindepth 1 -maxdepth 1 -type d \
        -name '*results_*' | sort); do
    PLOT="${dir##*/}"
    PLOT="${PLOT#__}"
    PLOT="${PLOT#results_}"
    VARIANT="${PLOT#*_}"
    PLOT="${PLOT%%_*}"
    [ -e "$TMPDIR/plots/$PLOT" ] || \
        echo "# steps reward variant" > "$TMPDIR/plots/$PLOT"
    cat $(find "$dir" -maxdepth 1 -type f -name '*.out') | \
        grep -v '^#' | \
        while read -r steps reward x; do
            echo "$steps $reward $VARIANT"
        done \
        >> "$TMPDIR/plots/$PLOT"
done

for plot in $(ls "$TMPDIR/plots"); do
    echo "Plotting $plot..."
    case "$plot" in
        walker*) PLOT_OPTS=(--mean --xmin=0 --xmax=400000
            --ymin=-200 --ymax=300) ;;
        *) PLOT_OPTS=(--mean) ;;
    esac
    PLOT_FILE="benchmarks/__$plot.png" \
        marginal-plot "${PLOT_OPTS[@]}" "$TMPDIR/plots/$plot" \
            reward steps variant || exit 1
done
