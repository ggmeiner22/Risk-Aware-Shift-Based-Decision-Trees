#include "core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace risk_aware_shift {
namespace {

/*
 * Small constant to avoid floating-point instability.
 */
constexpr double kEpsilon = 1e-12;

/*
 * Represents a candidate split during tree construction.
 * Stores both the split parameters and evaluation metrics.
 */
struct CandidateSplit {
    bool valid = false;
    int feature_index = -1;
    bool is_numeric = false;
    double threshold = 0.0;
    double score = -std::numeric_limits<double>::infinity();
    double shift = 0.0;
    double risk = 0.0;
};

/*
 * Count class occurrences for a subset of data.
 */
std::vector<int> class_counts(const Table& table, const std::vector<int>& indices) {
    std::vector<int> counts(table.class_names.size(), 0);
    for (int index : indices) {
        counts[table.labels[index]]++;
    }
    return counts;
}

/*
 * Check if all samples belong to the same class.
 * Used as a stopping condition for tree growth.
 */
bool is_pure(const Table& table, const std::vector<int>& indices) {
    if (indices.empty()) {
        return true;
    }
    const int label = table.labels[indices.front()];
    return std::all_of(indices.begin(), indices.end(), [&](int index) {
        return table.labels[index] == label;
    });
}

/*
 * Return the majority class label for a subset.
 * Used for leaf predictions.
 */
int majority_class(const Table& table, const std::vector<int>& indices) {
    const auto counts = class_counts(table, indices);
    return static_cast<int>(std::distance(
        counts.begin(),
        std::max_element(counts.begin(), counts.end())));
}

/*
 * Compute entropy from class counts.
 * Used for gain ratio calculations.
 */
double entropy_from_counts(const std::vector<int>& counts) {
    const double total = static_cast<double>(std::accumulate(counts.begin(), counts.end(), 0));
    if (total <= 0.0) {
        return 0.0;
    }

    double result = 0.0;
    for (int count : counts) {
        if (count == 0) {
            continue;
        }
        const double probability = static_cast<double>(count) / total;
        result -= probability * std::log2(probability);
    }
    return result;
}

/*
 * Normalize class counts into probability distribution.
 */
std::vector<double> normalized_distribution(const std::vector<int>& counts) {
    const double total = static_cast<double>(std::accumulate(counts.begin(), counts.end(), 0));
    std::vector<double> result(counts.size(), 0.0);
    if (total <= 0.0) {
        return result;
    }

    for (std::size_t index = 0; index < counts.size(); index++) {
        result[index] = static_cast<double>(counts[index]) / total;
    }
    return result;
}

/*
 * Compute distributional shift between parent and child nodes.
 * Measures how much class distributions change after a split.
 */
double compute_shift(const std::vector<int>& parent_counts, const std::vector<std::vector<int>>& child_counts) {
    const auto parent_distribution = normalized_distribution(parent_counts);
    const double parent_total = static_cast<double>(std::accumulate(parent_counts.begin(), parent_counts.end(), 0));
    if (parent_total <= 0.0) {
        return 0.0;
    }

    double shift = 0.0;
    for (const auto& child : child_counts) {
        const double child_total = static_cast<double>(std::accumulate(child.begin(), child.end(), 0));
        if (child_total <= 0.0) {
            continue;
        }
        const auto child_distribution = normalized_distribution(child);
        double distance = 0.0;
        for (std::size_t label = 0; label < child_distribution.size(); label++) {
            distance += std::fabs(child_distribution[label] - parent_distribution[label]);
        }
        shift += (child_total / parent_total) * distance;
    }

    return shift;
}

/*
 * Compute worst-case risk across branches.
 * Risk is defined as 1 - dominant class proportion.
 */
double compute_risk(const std::vector<std::vector<int>>& child_counts) {
    double risk = 0.0;
    for (const auto& child : child_counts) {
        const double total = static_cast<double>(std::accumulate(child.begin(), child.end(), 0));
        if (total <= 0.0) {
            continue;
        }
        const int branch_majority = *std::max_element(child.begin(), child.end());
        const double dominance = static_cast<double>(branch_majority) / total;
        risk = std::max(risk, 1.0 - dominance);
    }
    return risk;
}

/*
 * Standard Gain Ratio criterion (C4.5).
 */
double compute_gain_ratio_score(const std::vector<int>& parent_counts, const std::vector<std::vector<int>>& child_counts) {
    const double parent_total = static_cast<double>(std::accumulate(parent_counts.begin(), parent_counts.end(), 0));
    if (parent_total <= 0.0) {
        return 0.0;
    }

    const double parent_entropy = entropy_from_counts(parent_counts);
    double weighted_child_entropy = 0.0;
    double split_info = 0.0;

    for (const auto& child : child_counts) {
        const double child_total = static_cast<double>(std::accumulate(child.begin(), child.end(), 0));
        if (child_total <= 0.0) {
            continue;
        }
        const double weight = child_total / parent_total;
        weighted_child_entropy += weight * entropy_from_counts(child);
        split_info -= weight * std::log2(weight);
    }

    if (split_info <= kEpsilon) {
        return 0.0;
    }

    const double info_gain = parent_entropy - weighted_child_entropy;
    return info_gain / split_info;
}

/*
 * Class confidence criterion.
 * Measures how dominant the majority class is after splitting.
 */
double compute_class_confidence_score(const std::vector<std::vector<int>>& child_counts) {
    double weighted_dominance = 0.0;
    double total = 0.0;

    for (const auto& child : child_counts) {
        const double child_total = static_cast<double>(std::accumulate(child.begin(), child.end(), 0));
        if (child_total <= 0.0) {
            continue;
        }
        const int dominant_count = *std::max_element(child.begin(), child.end());
        weighted_dominance += dominant_count;
        total += child_total;
    }

    if (total <= 0.0) {
        return 0.0;
    }
    return weighted_dominance / total;
}

/*
 * Proposed Risk-Aware Shift criterion.
 * Combines shift (informativeness) and risk (uncertainty penalty).
 */
double compute_risk_aware_score(double shift, double risk, double beta) {
    return shift * std::max(0.0, 1.0 - beta * risk);
}

/*
 * Unified scoring function for all criteria.
 * Also outputs shift and risk for analysis.
 */
double compute_score(
    Criterion criterion,
    const std::vector<int>& parent_counts,
    const std::vector<std::vector<int>>& child_counts,
    double beta,
    double& shift_out,
    double& risk_out) {
    shift_out = compute_shift(parent_counts, child_counts);
    risk_out = compute_risk(child_counts);

    switch (criterion) {
        case Criterion::GainRatio:
            return compute_gain_ratio_score(parent_counts, child_counts);
        case Criterion::ClassConfidence:
            return compute_class_confidence_score(child_counts);
        case Criterion::RiskAwareShift:
            return compute_risk_aware_score(shift_out, risk_out, beta);
    }

    return 0.0;
}

/*
 * Compare two candidate splits.
 * Prioritizes:
 * 1. Higher score
 * 2. Higher shift (tie-breaker)
 * 3. Lower risk (final tie-breaker)
 */
bool better_candidate(const CandidateSplit& lhs, const CandidateSplit& rhs) {
    if (!lhs.valid) {
        return false;
    }
    if (!rhs.valid) {
        return true;
    }
    if (lhs.score > rhs.score + kEpsilon) {
        return true;
    }
    if (std::fabs(lhs.score - rhs.score) <= kEpsilon && lhs.shift > rhs.shift + kEpsilon) {
        return true;
    }
    if (std::fabs(lhs.score - rhs.score) <= kEpsilon &&
        std::fabs(lhs.shift - rhs.shift) <= kEpsilon &&
        lhs.risk < rhs.risk - kEpsilon) {
        return true;
    }
    return false;
}

/*
 * Evaluate all possible thresholds for a numeric feature.
 * Finds the best binary split.
 */
CandidateSplit evaluate_numeric_feature(
    const Table& table,
    const std::vector<int>& indices,
    int feature_index,
    Criterion criterion,
    double beta) {
    struct Entry {
        double value;
        int label;
    };

    std::vector<Entry> entries;
    entries.reserve(indices.size());
    for (int index : indices) {
        entries.push_back({table.numeric_values[index][feature_index], table.labels[index]});
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.value < rhs.value;
    });

    if (entries.size() < 2 || entries.front().value == entries.back().value) {
        return {};
    }

    const auto parent_counts = class_counts(table, indices);
    std::vector<int> left_counts(table.class_names.size(), 0);
    std::vector<int> right_counts = parent_counts;
    CandidateSplit best;

    for (std::size_t i = 0; i + 1 < entries.size(); i++) {
        left_counts[entries[i].label]++;
        right_counts[entries[i].label]--;

        if (entries[i].value == entries[i + 1].value) {
            continue;
        }

        const std::vector<std::vector<int>> child_counts{left_counts, right_counts};
        double shift = 0.0;
        double risk = 0.0;
        const double score = compute_score(criterion, parent_counts, child_counts, beta, shift, risk);

        CandidateSplit current;
        current.valid = score > kEpsilon;
        current.feature_index = feature_index;
        current.is_numeric = true;
        current.threshold = (entries[i].value + entries[i + 1].value) / 2.0;
        current.score = score;
        current.shift = shift;
        current.risk = risk;

        if (better_candidate(current, best)) {
            best = current;
        }
    }
    return best;
}

/*
 * Evaluate categorical feature split.
 * Each unique value becomes a branch.
 */
CandidateSplit evaluate_categorical_feature(
    const Table& table,
    const std::vector<int>& indices,
    int feature_index,
    Criterion criterion,
    double beta) {
    std::map<std::string, std::vector<int>> grouped;
    for (int index : indices) {
        const auto& value = table.categorical_values[index][feature_index];
        auto& counts = grouped[value];
        if (counts.empty()) {
            counts.assign(table.class_names.size(), 0);
        }
        counts[table.labels[index]]++;
    }

    if (grouped.size() < 2) {
        return {};
    }

    std::vector<std::vector<int>> child_counts;
    child_counts.reserve(grouped.size());
    for (const auto& group : grouped) {
        child_counts.push_back(group.second);
    }

    const auto parent_counts = class_counts(table, indices);
    double shift = 0.0;
    double risk = 0.0;
    const double score = compute_score(criterion, parent_counts, child_counts, beta, shift, risk);

    CandidateSplit candidate;
    candidate.valid = score > kEpsilon;
    candidate.feature_index = feature_index;
    candidate.is_numeric = false;
    candidate.score = score;
    candidate.shift = shift;
    candidate.risk = risk;
    return candidate;
}

/*
 * Find the best feature and split for current node.
 */
CandidateSplit find_best_split(
    const Table& table,
    const std::vector<int>& indices,
    const std::vector<bool>& categorical_available,
    Criterion criterion,
    double beta) {
    CandidateSplit best;

    for (std::size_t feature = 0; feature < table.features.size(); feature++) {
        CandidateSplit candidate;
        if (table.features[feature].type == FeatureType::Numeric) {
            candidate = evaluate_numeric_feature(table, indices, static_cast<int>(feature), criterion, beta);
        } else if (categorical_available[feature]) {
            candidate = evaluate_categorical_feature(table, indices, static_cast<int>(feature), criterion, beta);
        }

        if (better_candidate(candidate, best)) {
            best = candidate;
        }
    }
    return best;
}

/*
 * Partition dataset based on numeric threshold.
 */
std::pair<std::vector<int>, std::vector<int>> partition_numeric(
    const Table& table,
    const std::vector<int>& indices,
    int feature_index,
    double threshold) {
    std::vector<int> left;
    std::vector<int> right;
    for (int index : indices) {
        if (table.numeric_values[index][feature_index] <= threshold) {
            left.push_back(index);
        } else {
            right.push_back(index);
        }
    }
    return {left, right};
}

/*
 * Partition dataset based on categorical values.
 */
std::map<std::string, std::vector<int>> partition_categorical(
    const Table& table,
    const std::vector<int>& indices,
    int feature_index) {
    std::map<std::string, std::vector<int>> groups;
    for (int index : indices) {
        groups[table.categorical_values[index][feature_index]].push_back(index);
    }
    return groups;
}

/*
 * Statistics collected during tree traversal.
 * Used for evaluating structural properties of the tree.
 */
struct TraversalStats {
    double weighted_shift_sum = 0.0;
    double weighted_risk_sum = 0.0;
    double weight_sum = 0.0;
    int max_depth = 0;
    int total_nodes = 0;
};

/*
 * Traverse the tree and collect statistics.
 * Aggregates shift/risk weighted by sample counts.
 */
void collect_stats(const TreeNode& node, int depth, TraversalStats& stats) {
    stats.total_nodes++;
    stats.max_depth = std::max(stats.max_depth, depth);

    if (node.is_leaf) {
        return;
    }

    stats.weighted_shift_sum += node.shift * node.sample_count;
    stats.weighted_risk_sum += node.risk * node.sample_count;
    stats.weight_sum += node.sample_count;

    if (node.is_numeric) {
        collect_stats(*node.left, depth + 1, stats);
        collect_stats(*node.right, depth + 1, stats);
    } else {
        for (const auto& child : node.children) {
            collect_stats(*child.second, depth + 1, stats);
        }
    }
}

}  // namespace ends

/*
 * Recursively build the decision tree.
 *
 * Key steps:
 * 1. Check stopping conditions (pure or empty)
 * 2. Find best split using selected criterion
 * 3. Partition data
 * 4. Recursively build children
 */
std::unique_ptr<TreeNode> build_tree(
    const Table& table,
    const std::vector<int>& indices,
    const std::vector<bool>& categorical_available,
    Criterion criterion,
    double beta,
    int depth) {
    (void)depth;

    std::unique_ptr<TreeNode> node(new TreeNode());
    node->sample_count = static_cast<int>(indices.size());
    node->predicted_class = majority_class(table, indices);

    if (indices.empty() || is_pure(table, indices)) {
        return node;
    }

    const CandidateSplit best = find_best_split(table, indices, categorical_available, criterion, beta);
    if (!best.valid) {
        return node;
    }

    node->is_leaf = false;
    node->feature_index = best.feature_index;
    node->is_numeric = best.is_numeric;
    node->threshold = best.threshold;
    node->split_score = best.score;
    node->shift = best.shift;
    node->risk = best.risk;

    if (best.is_numeric) {
        auto partitions = partition_numeric(table, indices, best.feature_index, best.threshold);
        if (partitions.first.empty() || partitions.second.empty()) {
            node->is_leaf = true;
            return node;
        }

        node->left = build_tree(table, partitions.first, categorical_available, criterion, beta, depth + 1);
        node->right = build_tree(table, partitions.second, categorical_available, criterion, beta, depth + 1);
    } else {
        std::vector<bool> next_available = categorical_available;
        next_available[best.feature_index] = false;
        const auto groups = partition_categorical(table, indices, best.feature_index);
        for (const auto& group : groups) {
            node->children[group.first] =
                build_tree(table, group.second, next_available, criterion, beta, depth + 1);
        }
    }
    return node;
}

/*
 * Predict class label for a single sample.
 */
int predict(const TreeNode& node, const Table& table, int row_index) {
    if (node.is_leaf) {
        return node.predicted_class;
    }

    if (node.is_numeric) {
        if (table.numeric_values[row_index][node.feature_index] <= node.threshold) {
            return predict(*node.left, table, row_index);
        }
        return predict(*node.right, table, row_index);
    }

    const auto value = table.categorical_values[row_index][node.feature_index];
    const auto it = node.children.find(value);
    if (it == node.children.end()) {
        return node.predicted_class;
    }
    return predict(*it->second, table, row_index);
}

/*
 * Evaluate tree performance on test set.
 *
 * Computes:
 * - Accuracy
 * - Average shift
 * - Average risk
 * - Tree depth
 * - Total number of nodes
 */
TreeMetrics evaluate_tree(const TreeNode& tree, const Table& table, const std::vector<int>& test_indices) {
    TreeMetrics metrics;

    int correct = 0;
    for (int index : test_indices) {
        if (predict(tree, table, index) == table.labels[index]) {
            correct++;
        }
    }
    metrics.accuracy = test_indices.empty() ? 0.0 : static_cast<double>(correct) / test_indices.size();

    TraversalStats stats;
    collect_stats(tree, 0, stats);
    metrics.avg_shift = stats.weight_sum <= 0.0 ? 0.0 : stats.weighted_shift_sum / stats.weight_sum;
    metrics.avg_risk = stats.weight_sum <= 0.0 ? 0.0 : stats.weighted_risk_sum / stats.weight_sum;
    metrics.max_depth = static_cast<double>(stats.max_depth);
    metrics.total_nodes = static_cast<double>(stats.total_nodes);
    return metrics;
}

}  // namespace risk_aware_shift ends
