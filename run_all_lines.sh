#!/usr/bin/env bash
# Un semplice esempio di variabili


for VARIABLE in 0 1 2 3 4 5 6 7 
do
    echo Executing script n: $VARIABLE
    python3 run_single_row.py $VARIABLE
    sleep 250
done

#say RUN Finita ye ye ye ye Alè