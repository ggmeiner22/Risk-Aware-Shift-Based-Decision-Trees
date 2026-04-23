#!/bin/bash
make || exit 1

set -e

mkdir -p results

echo "Running Test 3: Proposed Risk-Aware Shift method"
./risk_aware_shift_trees \
  --method risk_aware_shift \
  --betas 0.6 \
  --results results/test3_risk_aware_shift_beta_0_6.csv