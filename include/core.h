#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace risk_aware_shift {

enum class FeatureType {
    Numeric,
    Categorical,
};

enum class Criterion {
    GainRatio,
    ClassConfidence,
    RiskAwareShift,
};

struct FeatureSpec {
    std::string name;
    FeatureType type;
};

struct Table {
    std::string name;
    std::vector<FeatureSpec> features;
    std::vector<std::vector<double>> numeric_values;
    std::vector<std::vector<std::string>> categorical_values;
    std::vector<int> labels;
    std::vector<std::string> class_names;
};

struct ExperimentConfig {
    std::string data_root;
    std::string results_path;
    int seed;
    std::vector<double> betas;

    ExperimentConfig();
};

struct SplitSpec {
    std::vector<int> train_indices;
    std::vector<int> test_indices;
};

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

struct TreeMetrics {
    double accuracy = 0.0;
    double avg_shift = 0.0;
    double avg_risk = 0.0;
    double max_depth = 0.0;
    double total_nodes = 0.0;
};

struct AggregateMetrics {
    std::string dataset;
    std::string method;
    double beta = 0.0;
    TreeMetrics mean;
};

std::string current_working_directory();
std::string join_path(const std::string& lhs, const std::string& rhs);
void ensure_parent_directories(const std::string& file_path);

Table load_pima(const std::string& path);
Table load_breast(const std::string& path);
Table load_heart(const std::string& path);
Table impute_with_training_statistics(const Table& raw, const std::vector<int>& train_indices);

ExperimentConfig parse_args(int argc, char** argv);
std::vector<SplitSpec> make_repeated_stratified_5x2(const Table& table, int seed);
std::vector<AggregateMetrics> run_experiments(const std::vector<Table>& datasets, const ExperimentConfig& config);
void write_results_csv(const std::vector<AggregateMetrics>& results, const std::string& path);
void print_results(const std::vector<AggregateMetrics>& results);

std::unique_ptr<TreeNode> build_tree(
    const Table& table,
    const std::vector<int>& indices,
    const std::vector<bool>& categorical_available,
    Criterion criterion,
    double beta,
    int depth = 0);

int predict(const TreeNode& node, const Table& table, int row_index);
TreeMetrics evaluate_tree(const TreeNode& tree, const Table& table, const std::vector<int>& test_indices);

}  // namespace risk_aware_shift
