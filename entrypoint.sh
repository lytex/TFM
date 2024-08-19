#!/bin/bash

python3.7 optuna_trial.py
head -n +1 $(for file in *.csv; do echo $file; done | head -n 1) > all.csv; tail -q -n 1 *.csv >> all.csv
