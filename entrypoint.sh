#!/bin/bash

fallocate -l 400G /home/ovh/swapfile
chmod 600 /home/ovh/swapfile
mkswap /home/ovh/swapfile
swapon /home/ovh/swapfile
chown root:root -R /home/ovh
chmod 777 -R /home/ovh
sudo -H -u ovh bash -c 'cd /home/ovh/code && exec python3.7 optuna_trial.py'
