#!/bin/bash
#------------------------------------------------------
# simple bash script to run unit and integration tests
#------------------------------------------------------

while getopts ::aiuv flag
do
    case "${flag}" in
        a) all=true;;
        i) integration=true;;
        u) unit=true;;
        v) validation=true;;
    esac
done

if [ "$all" = true ]; then
    integration=true
    unit=true
    validation=true
fi

if [ "$integration" = true ]; then
    echo "====================================================================="
    echo "      Running integration tests"
    echo "====================================================================="
    cd test/integration/
    time ./integration_tests
    cd ../../
fi
if [ "$unit" = true ]; then
    echo "Running unit tests"
fi
if [ "$validation" = true ]; then
    echo "====================================================================="
    echo "      Running validation tests (serial vs CPU references)"
    echo "====================================================================="
    python3 test/validation/vtest.py run --exe ./vampire-serial --label serial --jobs 4
    python3 test/validation/vtest.py compare --test serial --ref cpu
fi
