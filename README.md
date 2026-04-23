# Risk-Aware Shift-Based Decision Trees

This project implements and evaluates a **novel decision tree splitting criterion** that balances:

- **Distributional shift** (informativeness of a split)
- **Worst-case branch risk** (uncertainty in resulting partitions)

The goal is to produce decision trees that are not only accurate, but also **more reliable and robust**, avoiding splits that create highly uncertain branches.

---

## Features

- Implements three splitting criteria:
  - **Gain Ratio** (C4.5 baseline)
  - **Class Confidence**
  - **Risk-Aware Shift (proposed method)**

- Supports:
  - Numeric and categorical features
  - Missing value handling (median / "MISSING" imputation)
  - Stratified **5x2 cross-validation**
  - Evaluation metrics:
    - Accuracy
    - Average shift
    - Average risk
    - Tree depth
    - Number of nodes

---

## Project Structure  
```
.
├── Makefile
├── README.md
├── data
│   ├── BreastCancerWisconsin.csv
│   ├── diabetes.csv
│   └── heartDisease.csv
├── include
│   └── core.h
├── results
│   ├── test1_gain_ratio.csv
│   ├── test2_class_confidence.csv
│   ├── test3_risk_aware_shift_beta_0_3.csv
│   └── test4_risk_aware_shift_beta_sweep.csv
├── src
│   ├── data.cpp
│   ├── experiment.cpp
│   ├── main.cpp
│   └── tree.cpp
├── test_confidence.sh
├── test_gain_ratio.sh
├── test_risk_shift.sh
└── test_risk_shift_beta_sweep.sh
```
---

## Execution

Build:
```bash
make
```

Permissions:
```bash
chmod +x test_gain_ratio.sh test_confidence.sh test_risk_shift.sh test_risk_shift_beta_sweep.sh risk_aware_shift_trees
```

This produces the `risk_aware_shift_trees` executable in the repository root. Run it with:

```bash
./risk_aware_shift_trees [options]
```

Common options:
- `--data-root <path>`    : Path to dataset folder (default: `data`)
- `--results <path>`      : Output CSV file (default: `results/experiment_results.csv`)
- `--seed <int>`          : Random seed (default: `1337`)
- `--betas <list>`        : Comma-separated beta values for the risk-aware method (e.g. `0.0,0.1,0.2`)
- `--help`                : Show command-line usage

Examples:

Run experiments on all datasets with several beta values and save results:

```bash
./risk_aware_shift_trees --data-root data --betas 0.0,0.1,0.2,0.5,1.0 --results results/test_run.csv
```

Run a single experiment with a fixed seed:

```bash
./risk_aware_shift_trees --data-root data --results results/diabetes.csv --seed 42
```

Provided convenience scripts:

```bash
./test_gain_ratio.sh
./test_confidence.sh
./test_risk_shift.sh
./test_risk_shift_beta_sweep.sh
```

Notes:
- Ensure the executable has execute permissions (`chmod +x risk_aware_shift_trees`).
- The program prints CSV-formatted results to stdout and writes to the path passed via `--results` (default: `results/experiment_results.csv`).

## Output
The program prints results to the console:
```bash
Dataset,Method,Beta,Accuracy,AvgShift,AvgRisk,MaxDepth,TotalNodes
...
```
It also saves results to:
> results/experiment_results.csv

## Method Overview
1. Distributional Shift
Measures how much class distributions change after a split.

2. Risk
Measures the worst-case uncertainty among child nodes:
> risk = max(1 - class_dominance)

3. Risk-Aware Shift Criterion
The proposed split score:
- score = shift × (1 - β × risk)
where β controls the trade-off:
- β = 0 → pure shift (like information-based methods)
> higher β → penalizes risky splits more

## Evaluation
Uses Repeated Stratified 5x2 Cross Validation
Compares:
- Gain Ratio
- Class Confidence
- Risk-Aware Shift (multiple β values)

## Datasets
The project uses:
1. Pima Indians Diabetes
2. Breast Cancer Wisconsin
3. UCI Heart Disease

## Implementation Details
Language: C++11
No external dependencies
Efficient:
- O(n log n) numeric split evaluation
- Median-based imputation
- Recursive tree construction
