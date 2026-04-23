#!/bin/bash
make || exit 1

set -e

mkdir -p results

echo "Running Test 2: Class Confidence comparison"
./risk_aware_shift_trees \
  --method class_confidence \
  --results results/test2_class_confidence.csv