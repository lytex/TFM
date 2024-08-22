#!/bin/bash


mkdir -p /home/ovh/code/logs
touch /home/ovh/code/logs/main.log
python3.7 optuna_trial.py >> /home/ovh/code/logs/main.log 2>&1 &
python_proc=$!
while (( $(pgrep python3.7 | wc -l)  < 2)); do
    sleep 60
done


(setsid tail -fq /home/ovh/code/logs/*.log &)

wait $python_proc
