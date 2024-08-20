#!/bin/bash

fallocate -l 400G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
exec python3.7 optuna_trial.py
