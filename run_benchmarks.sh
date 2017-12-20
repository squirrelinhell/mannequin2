#!/bin/bash

export PYTHONPATH="$(pwd):$(pwd)/benchmarks:$PYTHONPATH"
export PATH="$(pwd)/scripts:$PATH"

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

mkdir "$TMPDIR/scripts"
mkdir "$TMPDIR/problems"
find benchmarks -maxdepth 1 -type f -name '*.py' -not -name '_*' | \
    while read f; do
        NAME="${f##*/}"
        NAME="${NAME%.py}"
        if grep -q '###' "$f"; then
            code-variants --print "$f" "$TMPDIR/scripts/${NAME}_" | \
                while read v; do
                    echo "${v%%_*}" > "$TMPDIR/problems/${NAME}_$v"
                done
        else
            cat "$f" > "$TMPDIR/scripts/${NAME}"
            echo "${NAME##*_}" > "$TMPDIR/problems/${NAME}"
        fi
    done
FILES=
for pattern in "$@"; do
    FILES="$FILES"$'\n'$(find "$TMPDIR/scripts" -mindepth 1 \
        -maxdepth 1 -name "$pattern"'*')
done
FILES=$(sort -u <<<"$FILES" | grep -v '^$')

if [ "x$FILES" = x ]; then
    echo "Available benchmarks:"
    for file in $(ls "$TMPDIR/scripts"); do
        echo " * $file"
    done
else
    echo "Running benchmarks:"
    for file in $FILES; do
        NAME="${file##*/}"
        echo " * $NAME"
    done

    for file in $FILES; do
        NAME="${file##*/}"
        PROBLEM=$(cat "$TMPDIR/problems/${NAME}") || exit 1
        NAME=${NAME/${PROBLEM}_/}
        NAME=${NAME/_${PROBLEM}/}
        OUT_DIR="benchmarks/__results_${PROBLEM}_$NAME"
        case "$PROBLEM" in
            cartpole*) COPIES=100 ;;
            *) COPIES=16 ;;
        esac
        code-variants --run --copies "$COPIES" \
            "$file" "$OUT_DIR/" || exit 1
    done
fi

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
    if [ -e "$TMPDIR/done_${PLOT}_$VARIANT" ]; then
        echo "Error: conflicting names: '${PLOT}_$VARIANT'" 1>&2
        exit 1
    fi
    touch "$TMPDIR/done_${PLOT}_$VARIANT"
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
            --ymin=-200 --ymax=350) ;;
        *) PLOT_OPTS=(--mean) ;;
    esac
    PLOT_FILE="benchmarks/__$plot.png" \
        marginal-plot "${PLOT_OPTS[@]}" "$TMPDIR/plots/$plot" \
            reward steps variant || exit 1
done
