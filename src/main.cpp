#include "core.h"

#include <exception>
#include <iostream>
#include <vector>

/*
 * Entry point for the experiment pipeline.
 *
 * Workflow:
 * 1. Parse command-line arguments
 * 2. Load datasets
 * 3. Run experiments (cross-validation + tree training)
 * 4. Print results to console
 * 5. Save results to CSV
 */
int main(int argc, char** argv) {
    try {
        // Parse CLI arguments into experiment configuration
        const auto config = risk_aware_shift::parse_args(argc, argv);

        // Load all datasets used for evaluation
        std::vector<risk_aware_shift::Table> datasets;
        datasets.push_back(risk_aware_shift::load_pima(
            risk_aware_shift::join_path(config.data_root, "diabetes.csv")));
        datasets.push_back(risk_aware_shift::load_breast(
            risk_aware_shift::join_path(config.data_root, "BreastCancerWisconsin.csv")));
        datasets.push_back(risk_aware_shift::load_heart(
            risk_aware_shift::join_path(config.data_root, "heartDisease.csv")));

        // Run full experiment suite across datasets and methods
        const auto results = risk_aware_shift::run_experiments(datasets, config);

        // Display results in terminal
        risk_aware_shift::print_results(results);  

        // Save results for later analysis (plots, tables, etc.)
        risk_aware_shift::write_results_csv(results, config.results_path);

        std::cout << "\nSaved CSV results to: " << config.results_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        // Catch and report any runtime errors 
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}