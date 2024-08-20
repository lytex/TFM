#!/bin/bash

fallocate -l 400G /home/swapfile
chmod 600 /home/swapfile
mkswap /home/swapfile
swapon /home/swapfile
su - ovh -c 'cd /home/ovh/code && exec python3.7 optuna_trial.py'
