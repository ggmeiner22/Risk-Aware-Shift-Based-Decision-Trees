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
├── Makefile
├── README.md
├── data
│   ├── BreastCancerWisconsin.csv
│   ├── diabetes.csv
│   └── heartDisease.csv
├── include
│   └── core.h
├── results
│   └── experiment_results.csv
└── src
    ├── data.cpp
    ├── experiment.cpp
    ├── main.cpp
    └── tree.cpp
```
---

## Execution

Compile using:

```bash
make
```
This produces the executable to run:
```bash
./risk_aware_shift_trees
```

Optional Arguments
`--data-root <path>`    # Path to dataset folder (default: current directory)
`--results <path>`      # Output CSV file (default: results/experiment_results.csv)
`--seed <int>`          # Random seed (default: 1337)
`--betas <list>`        # Comma-separated beta values for risk-aware method

Example:
```bash
./risk_aware_shift_trees --data-root data --betas 0.0,0.1,0.2,0.5,1.0
```

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
