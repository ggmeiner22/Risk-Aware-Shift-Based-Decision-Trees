#include "core.h"

#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    try {
        const auto config = risk_aware_shift::parse_args(argc, argv);

        std::vector<risk_aware_shift::Table> datasets;
        datasets.push_back(risk_aware_shift::load_pima(
            risk_aware_shift::join_path(config.data_root, "diabetes.csv")));
        datasets.push_back(risk_aware_shift::load_breast(
            risk_aware_shift::join_path(config.data_root, "BreastCancerWisconsin.csv")));
        datasets.push_back(risk_aware_shift::load_heart(
            risk_aware_shift::join_path(config.data_root, "heartDisease.csv")));

        const auto results = risk_aware_shift::run_experiments(datasets, config);
        risk_aware_shift::print_results(results);
        risk_aware_shift::write_results_csv(results, config.results_path);

        std::cout << "\nSaved CSV results to: " << config.results_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}