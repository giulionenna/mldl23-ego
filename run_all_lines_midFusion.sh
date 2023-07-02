#!/usr/bin/env bash
# Un semplice esempio di variabili


for VARIABLE in 1  
do
    echo Executing script n: $VARIABLE
    python3 run_single_row_midFusion.py $VARIABLE
    #sleep 150
done

say RUN Finita RUN Finita RUN Finita RUN Finita RUN Finita