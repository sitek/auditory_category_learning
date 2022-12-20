#!/bin/bash
# --time=10:00
# --mem=2GB

# convert psychopy outputs to BIDS format
for sub in FLT14 FLT15 FLT17 FLT18 FLT19 FLT21 FLT24 FLT25; do
  python convert_behav_to_bids.py --sub $sub --task ToneLearning
done
