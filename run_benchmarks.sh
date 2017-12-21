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
        if grep -q '###' "$f"; then
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

if [ "x$FILES" = x ]; then
    echo "Available benchmarks:"
    for file in $(ls "$TMPDIR/scripts"); do
        echo " * $file"
    done
else
    mkdir "$TMPDIR/results"
    TARGETS=""
    echo "Running benchmarks:"
    for file in $FILES; do
        NAME="${file##*/}"
        PROBLEM=$(PRINT_ENV_ONLY=1 python3 "$file" </dev/null 2>/dev/null)
        if [ "x${PROBLEM::9}" != "xproblem: " ]; then
            echo "Error: could not determine environment: $NAME" 1>&2
            exit 1
        fi
        PROBLEM="${PROBLEM:9}"
        ALGO="${NAME/${PROBLEM}_/}"
        ALGO="${ALGO/_${PROBLEM}/}"
        OUT_DIR="benchmarks/__results_${PROBLEM}_${ALGO}"
        case "$PROBLEM" in
            cartpole*) COPIES=100 ;;
            *) COPIES=16 ;;
        esac
        echo " * $NAME (x$COPIES) on $PROBLEM"
        COPIES="$(seq -w $COPIES)" || exit 1
        {
            echo "$OUT_DIR/%.out:"
            echo $'\t@echo Running '"$NAME"'/$*...'
            echo $'\t@LOG_FILE='"$TMPDIR"'/results/'"$NAME"'_$*.out python3 '"$file"
            echo $'\t@mkdir -p $(dir $@)'
            echo $'\t@cp '"$TMPDIR"'/results/'"$NAME"'_$*.out $@'
            echo
        } >> "$TMPDIR/makefile"
        for c in $COPIES; do
            TARGETS="$TARGETS $OUT_DIR/copy$c.out"
        done
    done
    echo "all: $TARGETS" >> "$TMPDIR/makefile"
    NUM_THREADS=$(grep -c ^processor /proc/cpuinfo) || exit 1
    echo "Max threads: $NUM_THREADS" 1>&2
    make -f "$TMPDIR/makefile" -j "$NUM_THREADS" 1>&2 || exit 1
fi

mkdir "$TMPDIR/plots"
for dir in $(find benchmarks -mindepth 1 -maxdepth 1 -type d \
        -name '*results_*' | sort); do
    PLOT="${dir##*/}"
    [ "x${PLOT::2}" != x__ ] || PLOT="${PLOT:2}_new"
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
