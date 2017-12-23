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
        echo '
            export ENV="$1"
            SCRIPT="'"$script"'"
            INFO="${SCRIPT##*/} on $ENV ($2)"
            export LOG_FILE="'"$TMPDIR"'/results/${ENV}_${SCRIPT##*/}_$2.out"
            OUT_FILE="benchmarks/results/${ENV}/__${SCRIPT##*/}/$2.out"
            echo "Running $INFO..."
            TIME_MSG=$({ time { python3 "$SCRIPT" 1>&3 2>&3; } } 3>&2 2>&1) || exit 1
            echo "Finished $INFO: $(echo $TIME_MSG)"
            mkdir -p "$(dirname $OUT_FILE)" || exit 1
            mv "$LOG_FILE" "$OUT_FILE" || exit 1
        ' > "$script.sh"
        for env in $ENVS; do
            OUT_DIR="benchmarks/results/${env}/__${script##*/}"
            echo "$OUT_DIR/%.out:"$'\n\t@'"/bin/bash $script.sh $env" \
                $' $*\n' >> "$TMPDIR/makefile"
            for c in $(seq $(n_copies $env)); do
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
        -f "$TMPDIR/makefile" 2>&1 || true
fi

mkdir "$TMPDIR/plots"
for env in $(ls benchmarks/results); do
    echo "# steps reward variant" > "$TMPDIR/plots/$env"
    for file in $(find "benchmarks/results/$env" -mindepth 1 -maxdepth 1); do
        variant="${file##*/}"
        [ "x${variant::2}" != x__ ] || PLOT="${variant:2}_new"
        cat $(find "$file" -maxdepth 1 -type f -name '*.out') | \
            grep -v '^#' | \
            while read -r steps reward x; do
                echo "$steps $reward $variant"
            done \
            >> "$TMPDIR/plots/$env"
    done
done

for plot in $(ls "$TMPDIR/plots"); do
    echo "Plotting $plot..."
    case "$plot" in
        walker*) PLOT_OPTS=(--mean --ymin=-200 --ymax=350) ;;
        lander*) PLOT_OPTS=(--mean --ymin=-300 --ymax=300) ;;
        *) PLOT_OPTS=(--mean) ;;
    esac
    PLOT_FILE="benchmarks/results/__$plot.png" \
        marginal-plot "${PLOT_OPTS[@]}" "$TMPDIR/plots/$plot" \
            reward steps variant || exit 1
done
