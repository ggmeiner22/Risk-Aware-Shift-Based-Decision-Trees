#!/bin/bash
make || exit 1

set -e

mkdir -p results

echo "Running Test 4: Proposed Risk-Aware Shift beta sweep"
./risk_aware_shift_trees \
  --method risk_aware_shift \
  --betas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --results results/test4_risk_aware_shift_beta_sweep.csv