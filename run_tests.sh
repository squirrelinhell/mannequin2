#!/bin/bash

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$(pwd):$PYTHONPATH"

DEBUG_SETUP='
import sys
import IPython.core.ultratb
sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)
'

TEST_SETUP='
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def timer():
    import time
    start_time = time.time()
    def get_time(print_info=True):
        total_time = time.time() - start_time
        import os
        import sys
        sys.stderr.write("Time: %.3fs\n" % total_time)
        sys.stderr.flush()
        if "STOPTIME" in os.environ:
            return 0.0
        return total_time
    return get_time
timer = timer()
'

if [ "x$1" != x -a "x$2" = x ]; then
    export DEBUG=1
    TEST_FILE="$1"
    [ -f "$TEST_FILE" ] || TEST_FILE="tests/$TEST_FILE"
    [ -f "$TEST_FILE" ] || TEST_FILE="$TEST_FILE.py"
    echo "$DEBUG_SETUP" "$TEST_SETUP" > "$TMPDIR/test_setup.py"
    cat "$TEST_FILE" > "$TMPDIR/run" || exit 1
    exec python3 "$TMPDIR/run"
fi

TESTS="$@"
if [ "x$TESTS" = x ]; then
    TESTS=$(find tests -maxdepth 1 -name '*.py' | sort)
fi

for test in $TESTS; do
    [ -f "$test" ] || test="tests/$test"
    [ -f "$test" ] || test="$test.py"

    echo "$TEST_SETUP" > "$TMPDIR/test_setup.py"
    cat "$test" > "$TMPDIR/run" || exit 1
    echo -n "Test: $test... "

    OUT_FILE="${test%.*}.out"
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
        echo "EXIT CODE $RESULT: $test"
        echo
    elif ! diff -b -q "$TMPDIR/ans" "$TMPDIR/out" >/dev/null; then
        echo FAIL
        cat "$TMPDIR/dbg"
        echo
        echo "INCORRECT OUTPUT: $test"
        echo
        diff -b --color=auto "$TMPDIR/ans" "$TMPDIR/out"
        echo
    else
        echo OK
    fi
done
