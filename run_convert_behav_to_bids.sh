#!/bin/bash
# --time=10:00
# --mem=2GB

# convert psychopy outputs to BIDS format
python convert_behav_to_bids.py --sub $1 --task $2
