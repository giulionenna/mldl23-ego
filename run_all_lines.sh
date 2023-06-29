#!/usr/bin/env bash
# Un semplice esempio di variabili


for VARIABLE in 7
do
    echo Executing script n: $VARIABLE
    python3 run_single_row.py $VARIABLE
    sleep 150
done