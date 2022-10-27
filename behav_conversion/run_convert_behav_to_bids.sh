#!/bin/bash
# --time=10:00
# --mem=2GB

# convert psychopy outputs to BIDS format
for sub in FLT01 FLT02 FLT03 FLT04 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10; do
  python convert_behav_to_bids.py --sub $sub --task ToneLearning
done