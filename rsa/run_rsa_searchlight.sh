#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH --mem=64G

python rsa_searchlight_WIP.py
