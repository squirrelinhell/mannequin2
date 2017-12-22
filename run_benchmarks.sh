#!/bin/bash

export PYTHONPATH="$(pwd):$(pwd)/benchmarks:$PYTHONPATH"
export PATH="$(pwd)/scripts:$PATH"

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

mkdir "$TMPDIR/algo"
find benchmarks -maxdepth 1 -type f -name '*.py' -not -name '_*' | \
    while read f; do
        NAME="${f##*/}"
        NAME="${NAME%.py}"
        if grep -q '###' "$f"; then
            code-variants "$f" "$TMPDIR/algo/${NAME}_"
        else
            cat "$f" > "$TMPDIR/algo/${NAME}"
        fi
    done

ALGO=
ENV_PATTERNS=()
while [ "x$1" != x ]; do
    MATCHES=$(find "$TMPDIR/algo" -mindepth 1 \
        -maxdepth 1 -name "$1"'*')
    if [ "x$MATCHES" = x ]; then
        ENV_PATTERNS+=("$1")
    else
        ALGO="$ALGO"$'\n'"$MATCHES"
    fi
    shift
done
ALGO=$(sort -u <<<"$ALGO" | grep -v '^$')

ALL_ENVS=$(grep -F '": c("' benchmarks/_env.py | cut -d '"' -f 2 | sort)
ENVS=
while read env; do
    for pattern in "${ENV_PATTERNS[@]}"; do
        [[ $env == $pattern* ]] && ENVS="$ENVS"$'\n'"$env"
    done
done <<<"$ALL_ENVS"
ENVS=$(sort -u <<<"$ENVS" | grep -v '^$')
[ "x$ENVS" != x ] || ENVS="$ALL_ENVS"

n_copies() {
    if [ "x$N_COPIES" != x ]; then
        echo "$N_COPIES"
        return 0
    fi
    case "$1" in
        cartpole) echo 100 ;;
        acrobot) echo 100 ;;
        *) echo 16 ;;
    esac
}

if [ "x$ALGO" = x ]; then
    echo "Available algorithms:"
    for algo in $(ls "$TMPDIR/algo"); do
        echo " * $algo"
    done
    echo "Available environments:"
    for env in $ALL_ENVS; do
        echo " * $env (x$(n_copies $env))"
    done
else
    mkdir "$TMPDIR/results"
    TARGETS=""
    echo "Running algorithms:"
    for script in $ALGO; do
        echo " * ${script##*/}"
        for env in $ENVS; do
            NAME="${env}_${script##*/}"
            OUT_DIR="benchmarks/__results_${NAME}"
            COPIES="$(seq $(n_copies $env))" || exit 1
            {
                echo "$OUT_DIR/%.out:"
                echo $'\t@echo Running "'"${script##*/}"' on '"$env"' ($*)..."'
                echo $'\t@ENV='"$env" \
                    'LOG_FILE='"$TMPDIR"'/results/'"$NAME"'_$*.out' \
                        "python3 $script"
                echo $'\t@mkdir -p $(dir $@)'
                echo $'\t@mv '"$TMPDIR"'/results/'"$NAME"'_$*.out $@'
                echo
            } >> "$TMPDIR/makefile"
            for c in $COPIES; do
                TARGETS="$TARGETS $OUT_DIR/copy$c.out"
            done
        done
    done
    echo "On environments:"
    for env in $ENVS; do
        echo " * $env (x$(n_copies $env))"
    done
    echo "all: $TARGETS" >> "$TMPDIR/makefile"
    NUM_THREADS=$(grep -c ^processor /proc/cpuinfo) || exit 1
    echo "Max threads: $NUM_THREADS" 1>&2
    make --keep-going -j "$NUM_THREADS" \
        -f "$TMPDIR/makefile" 1>&2 || true
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
        walker*) PLOT_OPTS=(--mean --ymin=-200 --ymax=350) ;;
        lander*) PLOT_OPTS=(--mean --ymin=-300 --ymax=300) ;;
        *) PLOT_OPTS=(--mean) ;;
    esac
    PLOT_FILE="benchmarks/__$plot.png" \
        marginal-plot "${PLOT_OPTS[@]}" "$TMPDIR/plots/$plot" \
            reward steps variant || exit 1
done
