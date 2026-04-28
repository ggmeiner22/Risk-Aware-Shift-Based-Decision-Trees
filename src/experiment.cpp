#include "core.h"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

namespace risk_aware_shift {
namespace {

/*
 * Trim whitespace from both ends of a string.
 * Used for parsing CLI inputs and CSV-like strings.
 */
std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

/*
 * Check if a file or directory exists.
 */
bool path_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

/*
 * Create a single directory if it does not already exist.
 * Handles cross-platform behavior.
 */
bool create_single_directory(const std::string& path) {
    if (path.empty() || path_exists(path)) {
        return true;
    }
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

/*
 * Parse a comma-separated list of beta values from CLI.
 * Used for sweeping the risk-aware-shift parameter.
 */
std::vector<double> parse_betas(const std::string& raw) {
    std::vector<double> betas;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            betas.push_back(std::stod(token));
        }
    }
    if (betas.empty()) {
        throw std::runtime_error("At least one beta value is required.");
    }
    return betas;
}

/*
 * Convert method string into enum selection.
 * Controls which learning algorithm(s) to run.
 */
MethodSelection parse_method(const std::string& raw) {
    const std::string method = trim(raw);
    if (method == "all") {
        return MethodSelection::All;
    }
    if (method == "gain_ratio") {
        return MethodSelection::GainRatioOnly;
    }
    if (method == "class_confidence") {
        return MethodSelection::ClassConfidenceOnly;
    }
    if (method == "risk_aware_shift") {
        return MethodSelection::RiskAwareShiftOnly;
    }
    throw std::runtime_error(
        "Unknown method: " + method +
        " (expected: all, gain_ratio, class_confidence, risk_aware_shift)");
}

/*
 * Compute the mean of metrics across all folds.
 * Aggregates results from cross-validation.
 */
TreeMetrics mean_metrics(const std::vector<TreeMetrics>& folds) {
    TreeMetrics result;
    if (folds.empty()) {
        return result;
    }

    for (const auto& fold : folds) {
        result.accuracy += fold.accuracy;
        result.avg_shift += fold.avg_shift;
        result.avg_risk += fold.avg_risk;
        result.max_depth += fold.max_depth;
        result.total_nodes += fold.total_nodes;
    }

    const double count = static_cast<double>(folds.size());
    result.accuracy /= count;
    result.avg_shift /= count;
    result.avg_risk /= count;
    result.max_depth /= count;
    result.total_nodes /= count;
    return result;
}

}  // namespace ends

/*
 * Default configuration for experiment runs.
 * Includes:
 * - dataset location
 * - output path
 * - random seed
 * - beta sweep values
 * - selected method(s)
 */
ExperimentConfig::ExperimentConfig()
    : data_root(join_path(current_working_directory(), "data")),
      results_path(join_path(join_path(current_working_directory(), "results"), "experiment_results.csv")),
      seed(1337),
      betas(std::vector<double>(1, 0.0)),
      method(MethodSelection::All) {
    betas.push_back(0.05);
    betas.push_back(0.1);
    betas.push_back(0.15);
    betas.push_back(0.2);
    betas.push_back(0.25);
    betas.push_back(0.3);
    betas.push_back(0.4);
    betas.push_back(0.5);
    betas.push_back(0.6);
    betas.push_back(0.7);
    betas.push_back(0.75);
    betas.push_back(0.8);
    betas.push_back(0.9);
    betas.push_back(1.0);
}

/*
 * Get the current working directory (cross-platform).
 */
std::string current_working_directory() {
#ifdef _WIN32
    char buffer[_MAX_PATH];
    if (_getcwd(buffer, sizeof(buffer)) == NULL) {
        throw std::runtime_error("Unable to determine current working directory.");
    }
#else
    char buffer[4096];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        throw std::runtime_error("Unable to determine current working directory.");
    }
#endif
    return std::string(buffer);
}

/*
 * Join two path components safely.
 */
std::string join_path(const std::string& lhs, const std::string& rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    const char last = lhs[lhs.size() - 1];
    if (last == '/' || last == '\\') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

/*
 * Ensure all directories in a file path exist.
 * Creates missing directories recursively.
 */
void ensure_parent_directories(const std::string& file_path) {
    std::string normalized = file_path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    const std::size_t separator = normalized.find_last_of('/');
    if (separator == std::string::npos) {
        return;
    }

    const std::string directory = normalized.substr(0, separator);
    if (directory.empty()) {
        return;
    }

    std::string current;
    std::size_t start = 0;

    if (directory.size() > 1 && directory[1] == ':') {
        current = directory.substr(0, 2);
        start = 2;
        if (start < directory.size() && directory[start] == '/') {
            current += "/";
            start++;
        }
    } else if (directory[0] == '/') {
        current = "/";
        start = 1;
    }

    while (start < directory.size()) {
        const std::size_t next = directory.find('/', start);
        const std::string part = directory.substr(
            start,
            next == std::string::npos ? std::string::npos : next - start);
        if (!part.empty()) {
            current = join_path(current, part);
            if (!create_single_directory(current) && !path_exists(current)) {
                throw std::runtime_error("Unable to create directory: " + current);
            }
        }
        if (next == std::string::npos) {
            break;
        }
        start = next + 1;
    }
}

/*
 * Parse command-line arguments into an ExperimentConfig.
 * Supports:
 * --data-root
 * --results
 * --seed
 * --betas
 * --method
 */
ExperimentConfig parse_args(int argc, char** argv) {
    ExperimentConfig config;

    for (int index = 1; index < argc; index++) {
        const std::string arg = argv[index];

        auto require_value = [&](const std::string& flag) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[index++];
        };

        if (arg == "--data-root") {
            config.data_root = require_value(arg);
        } else if (arg == "--results") {
            config.results_path = require_value(arg);
        } else if (arg == "--seed") {
            config.seed = std::stoi(require_value(arg));
        } else if (arg == "--betas") {
            config.betas = parse_betas(require_value(arg));
        } else if (arg == "--method") {
            config.method = parse_method(require_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return config;
}

/*
 * Create repeated stratified 5x2 cross-validation splits.
 *
 * Key idea:
 * - Maintain class balance (stratification)
 * - Repeat 5 times with different shuffles
 * - Each repeat produces 2 folds (swap train/test)
 *
 * Total: 10 folds
 */
std::vector<SplitSpec> make_repeated_stratified_5x2(const Table& table, int seed) {
    std::unordered_map<int, std::vector<int>> by_class;
    for (std::size_t index = 0; index < table.labels.size(); index++) {
        by_class[table.labels[index]].push_back(static_cast<int>(index));
    }

    std::vector<SplitSpec> splits;
    for (int repeat = 0; repeat < 5; repeat++) {
        std::vector<int> half_a;
        std::vector<int> half_b;

        for (auto& entry : by_class) {
            const int label = entry.first;
            auto& indices = entry.second;
            std::mt19937 rng(seed + repeat * 101 + label * 997);
            std::shuffle(indices.begin(), indices.end(), rng);
            const std::size_t midpoint = indices.size() / 2;
            half_a.insert(half_a.end(), indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(midpoint));
            half_b.insert(half_b.end(), indices.begin() + static_cast<std::ptrdiff_t>(midpoint), indices.end());
        }

        splits.push_back({half_a, half_b});
        splits.push_back({half_b, half_a});
    }

    return splits;
}

/*
 * Core experiment runner.
 *
 * For each dataset:
 * 1. Generate stratified splits
 * 2. Train and evaluate trees on each split
 * 3. Aggregate metrics across folds
 *
 * Supports:
 * - Gain Ratio
 * - Class Confidence
 * - Risk-Aware Shift (with beta sweep)
 */
std::vector<AggregateMetrics> run_experiments(const std::vector<Table>& datasets, const ExperimentConfig& config) {
    std::vector<AggregateMetrics> results;

    for (const auto& raw_dataset : datasets) {
        const auto splits = make_repeated_stratified_5x2(raw_dataset, config.seed);

        auto run_method = [&](const std::string& method_name, Criterion criterion, double beta) {
            std::vector<TreeMetrics> fold_metrics;
            fold_metrics.reserve(splits.size());

            for (const auto& split : splits) {
                Table prepared = impute_with_training_statistics(raw_dataset, split.train_indices);
                std::vector<bool> categorical_available(prepared.features.size(), true);
                auto tree = build_tree(prepared, split.train_indices, categorical_available, criterion, beta);
                fold_metrics.push_back(evaluate_tree(*tree, prepared, split.test_indices));
            }

            // Creates Metric summary object
            AggregateMetrics summary;
            summary.dataset = raw_dataset.name;
            summary.method = method_name;
            summary.beta = beta;
            summary.mean = mean_metrics(fold_metrics);
            results.push_back(summary);
        };

        // Runs the expierment

        if (config.method == MethodSelection::All || config.method == MethodSelection::GainRatioOnly) {
            run_method("gain_ratio", Criterion::GainRatio, std::numeric_limits<double>::quiet_NaN());
        }

        if (config.method == MethodSelection::All || config.method == MethodSelection::ClassConfidenceOnly) {
            run_method("class_confidence", Criterion::ClassConfidence, std::numeric_limits<double>::quiet_NaN());
        }

        if (config.method == MethodSelection::All || config.method == MethodSelection::RiskAwareShiftOnly) {
            for (double beta : config.betas) {
                run_method("risk_aware_shift", Criterion::RiskAwareShift, beta);
            }
        }
    }

    return results;
}

/*
 * Write experiment results to CSV file.
 *
 * Output includes:
 * dataset, method, beta, accuracy, shift, risk, tree stats
 */
void write_results_csv(const std::vector<AggregateMetrics>& results, const std::string& path) {
    ensure_parent_directories(path);

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to write results file: " + path);
    }

    output << "dataset,method,beta,accuracy,avg_shift,avg_risk,max_depth,total_nodes\n";
    output << std::fixed << std::setprecision(6);

    for (const auto& result : results) {
        output << '"' << result.dataset << '"' << ','
               << result.method << ',';
        if (std::isnan(result.beta)) {
            output << ',';
        } else {
            output << result.beta << ',';
        }
        output << result.mean.accuracy << ','
               << result.mean.avg_shift << ','
               << result.mean.avg_risk << ','
               << result.mean.max_depth << ','
               << result.mean.total_nodes << '\n';
    }
}

/*
 * Print experiment results to console in readable format.
 */
void print_results(const std::vector<AggregateMetrics>& results) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Dataset,Method,Beta,Accuracy,AvgShift,AvgRisk,MaxDepth,TotalNodes\n";
    for (const auto& result : results) {
        std::cout << result.dataset << ','
                  << result.method << ',';
        if (std::isnan(result.beta)) {
            std::cout << '-';
        } else {
            std::cout << result.beta;
        }
        std::cout << ','
                  << result.mean.accuracy << ','
                  << result.mean.avg_shift << ','
                  << result.mean.avg_risk << ','
                  << result.mean.max_depth << ','
                  << result.mean.total_nodes << '\n';
    }
}

} // namespace risk_aware_shift ends