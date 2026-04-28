#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace risk_aware_shift {

/*
 * Type of feature in dataset.
 * Numeric features use thresholds, categorical features use branching.
 */
enum class FeatureType {
    Numeric,
    Categorical,
};

/*
 * Splitting criteria used in decision tree.
 * - GainRatio: standard C4.5
 * - ClassConfidence: dominance-based heuristic
 * - RiskAwareShift: proposed method combining shift + risk
 */
enum class Criterion {
    GainRatio,
    ClassConfidence,
    RiskAwareShift,
};

/*
 * Controls which method(s) to run in experiments.
 */
enum class MethodSelection {
    All,
    GainRatioOnly,
    ClassConfidenceOnly,
    RiskAwareShiftOnly,
};

/*
 * Metadata describing a feature.
 */
struct FeatureSpec {
    std::string name;
    FeatureType type;
};

/*
 * Core dataset structure.
 * Stores:
 * - feature definitions
 * - numeric and categorical values
 * - labels and class names
 */
struct Table {
    std::string name;
    std::vector<FeatureSpec> features;
    std::vector<std::vector<double>> numeric_values;
    std::vector<std::vector<std::string>> categorical_values;
    std::vector<int> labels;
    std::vector<std::string> class_names;
};

/*
 * Configuration for experiment execution.
 * Includes paths, randomness, parameter sweeps, and method selection.
 */
struct ExperimentConfig {
    std::string data_root;
    std::string results_path;
    int seed;
    std::vector<double> betas;  // parameter sweep for risk-aware shift
    MethodSelection method;

    ExperimentConfig();
};

/*
 * Train/test split specification.
 */
struct SplitSpec {
    std::vector<int> train_indices;
    std::vector<int> test_indices;
};

/*
 * Node in decision tree.
 *
 * Stores:
 * - prediction (for leaf)
 * - split info (feature, threshold)
 * - evaluation metrics (score, shift, risk)
 * - child pointers
 */
struct TreeNode {
    bool is_leaf = true;
    int predicted_class = 0;
    int sample_count = 0;

    int feature_index = -1;
    bool is_numeric = false;
    double threshold = 0.0;

    double split_score = 0.0;
    double shift = 0.0;
    double risk = 0.0;

    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    std::map<std::string, std::unique_ptr<TreeNode>> children;
};

/*
 * Metrics computed for a trained tree.
 * Used for evaluating model performance and structure.
 */
struct TreeMetrics {
    double accuracy = 0.0;
    double avg_shift = 0.0;
    double avg_risk = 0.0;
    double max_depth = 0.0;
    double total_nodes = 0.0;
};

/*
 * Aggregated metrics across cross-validation folds.
 */
struct AggregateMetrics {
    std::string dataset;
    std::string method;
    double beta = 0.0;
    TreeMetrics mean;
};

/* ===== Utility Functions ===== */

/*
 * Get current working directory.
 */
std::string current_working_directory();

/*
 * Join two file paths safely.
 */
std::string join_path(const std::string& lhs, const std::string& rhs);

/*
 * Ensure directories exist for a given file path.
 */
void ensure_parent_directories(const std::string& file_path);

/* ===== Data Loading ===== */

/*
 * Load datasets into Table format.
 */
Table load_pima(const std::string& path);
Table load_breast(const std::string& path);
Table load_heart(const std::string& path);

/*
 * Impute missing values using training data statistics.
 */
Table impute_with_training_statistics(const Table& raw, const std::vector<int>& train_indices);

/* ===== Experiment Pipeline ===== */

/*
 * Parse command-line arguments into configuration.
 */
ExperimentConfig parse_args(int argc, char** argv);

/*
 * Create repeated stratified 5x2 cross-validation splits.
 */
std::vector<SplitSpec> make_repeated_stratified_5x2(const Table& table, int seed);

/*
 * Run experiments across datasets and methods.
 */
std::vector<AggregateMetrics> run_experiments(const std::vector<Table>& datasets, const ExperimentConfig& config);

/*
 * Output results to CSV file.
 */
void write_results_csv(const std::vector<AggregateMetrics>& results, const std::string& path);

/*
 * Print results to console.
 */
void print_results(const std::vector<AggregateMetrics>& results);


/* ===== Decision Tree ===== */

/*
 * Build decision tree using specified criterion.
 */
std::unique_ptr<TreeNode> build_tree(
    const Table& table,
    const std::vector<int>& indices,
    const std::vector<bool>& categorical_available,
    Criterion criterion,
    double beta,
    int depth = 0);

/*
 * Predict class label for a single sample.
 */
int predict(const TreeNode& node, const Table& table, int row_index);

/*
 * Evaluate tree on test data and compute metrics.
 */
TreeMetrics evaluate_tree(const TreeNode& tree, const Table& table, const std::vector<int>& test_indices);

}  