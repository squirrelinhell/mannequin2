#!/bin/bash

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$(pwd):$PYTHONPATH"

DEBUG_SETUP='
def debug_setup():
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)
debug_setup()
del debug_setup
'

TEST_SETUP='
def setup():
    import numpy as np
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    import time
    start_time = time.time()
    def get_time(print_info=True):
        total_time = time.time() - start_time
        import os
        import sys
        if print_info:
            sys.stderr.write("Time: %.3fs\n" % total_time)
            sys.stderr.flush()
        if "STOPTIME" in os.environ:
            return 0.0
        return total_time
    return get_time
timer = setup()
'

FILES=
for pattern in "${@:-}"; do
    FILES="$FILES"$'\n'$(find tests -maxdepth 1 \
        -name "$pattern"'*.py' -not -name '_*')
done
FILES=$(sort -u <<<"$FILES" | grep -v '^$')

if [ $(wc -l <<<"$FILES") = 1 ]; then
    echo "Running $FILES... " 1>&2
    export DEBUG=1
    export STOPTIME=1
    [ -f "$FILES" ] || TEST_FILE="tests/$FILES.py"
    cat "$FILES" > "$TMPDIR/run" || exit 1
    echo "$DEBUG_SETUP" "$TEST_SETUP" > "$TMPDIR/test_setup.py"
    exec python3 "$TMPDIR/run"
fi

for file in $FILES; do
    [ -f "$file" ] || file="tests/$file.py"
    NAME="${file##*/}"
    NAME="${NAME%.*}"
    echo -n "Running $NAME... "

    echo "$TEST_SETUP" > "$TMPDIR/test_setup.py"
    cat "$file" > "$TMPDIR/run" || exit 1

    OUT_FILE="${file%.*}.out"
    if [ -f "$OUT_FILE" ]; then
        cat "$OUT_FILE"
    fi > "$TMPDIR/ans"

    ( \
        cd "$TMPDIR" \
        && python3 ./run \
    ) </dev/null >"$TMPDIR/out" 2>"$TMPDIR/dbg"
    RESULT=$?

    if ! [ "x$RESULT" = x0 ]; then
        echo FAIL
        cat "$TMPDIR/dbg"
        echo
        echo "EXIT CODE $RESULT: $file"
        echo
    elif ! diff -b -q "$TMPDIR/ans" "$TMPDIR/out" >/dev/null; then
        echo FAIL
        cat "$TMPDIR/dbg"
        echo
        echo "INCORRECT OUTPUT: $file"
        echo
        diff -b --color=auto "$TMPDIR/ans" "$TMPDIR/out"
        echo
    else
        TIME=$(grep '^Time: ' <"$TMPDIR/dbg" | cut -d ' ' -f 2)
        [ "x$TIME" = x ] || TIME=" ("$(echo $TIME)")"
        echo "OK$TIME"
    fi
done
