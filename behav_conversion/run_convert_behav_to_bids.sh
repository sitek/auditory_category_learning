#!/bin/bash
# --time=10:00
# --mem=2GB

# convert psychopy outputs to BIDS format
#for sub in FLT02 FLT03 FLT04 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT17 FLT18 FLT19 #FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
sub=FLT15
  python convert_behav_to_bids.py --sub $sub --task ToneLearning
#done
