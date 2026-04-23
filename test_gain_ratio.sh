#!/bin/bash
make || exit 1

set -e

mkdir -p results

echo "Running Test 1: Gain Ratio baseline"
./risk_aware_shift_trees \
  --method gain_ratio \
  --results results/test1_gain_ratio.csv