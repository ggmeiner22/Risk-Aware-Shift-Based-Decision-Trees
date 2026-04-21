#include "core.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace risk_aware_shift {
namespace {

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string current;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            result.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    result.push_back(current);
    return result;
}

std::vector<std::vector<std::string> > read_csv_rows(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open CSV file: " + path);
    }

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        rows.push_back(split_csv_line(line));
    }
    return rows;
}

double parse_double_or_nan(const std::string& raw) {
    const std::string value = trim(raw);
    if (value.empty() || value == "?") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(value);
}

Table make_empty_table(
    const std::string& name,
    const std::vector<FeatureSpec>& features,
    const std::vector<std::string>& class_names) {
    Table table;
    table.name = name;
    table.features = features;
    table.class_names = class_names;
    return table;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

}  // namespace

Table load_pima(const std::string& path) {
    const auto rows = read_csv_rows(path);
    if (rows.size() < 2) {
        throw std::runtime_error("Pima dataset is empty: " + path);
    }

    const std::vector<FeatureSpec> features{
        {"Pregnancies", FeatureType::Numeric},
        {"Glucose", FeatureType::Numeric},
        {"BloodPressure", FeatureType::Numeric},
        {"SkinThickness", FeatureType::Numeric},
        {"Insulin", FeatureType::Numeric},
        {"BMI", FeatureType::Numeric},
        {"DiabetesPedigreeFunction", FeatureType::Numeric},
        {"Age", FeatureType::Numeric},
    };

    Table table = make_empty_table("Pima Indians Diabetes", features, {"negative", "positive"});

    for (std::size_t row = 1; row < rows.size(); ++row) {
        if (rows[row].size() < 9) {
            continue;
        }

        std::vector<double> numeric(features.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<std::string> categorical(features.size());
        for (std::size_t feature = 0; feature < features.size(); ++feature) {
            numeric[feature] = parse_double_or_nan(rows[row][feature]);
        }

        table.numeric_values.push_back(std::move(numeric));
        table.categorical_values.push_back(std::move(categorical));
        table.labels.push_back(std::stoi(trim(rows[row][8])));
    }

    return table;
}

Table load_breast(const std::string& path) {
    const auto rows = read_csv_rows(path);
    if (rows.size() < 2) {
        throw std::runtime_error("Breast-cancer dataset is empty: " + path);
    }

    std::vector<FeatureSpec> features;
    for (std::size_t column = 2; column < rows[0].size(); ++column) {
        const std::string name = trim(rows[0][column]);
        if (!name.empty()) {
            features.push_back({name, FeatureType::Numeric});
        }
    }

    Table table = make_empty_table("Breast Cancer Wisconsin", features, {"benign", "malignant"});

    for (std::size_t row = 1; row < rows.size(); ++row) {
        if (rows[row].size() < 2 + features.size()) {
            continue;
        }

        std::vector<double> numeric(features.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<std::string> categorical(features.size());
        for (std::size_t feature = 0; feature < features.size(); ++feature) {
            numeric[feature] = parse_double_or_nan(rows[row][feature + 2]);
        }

        table.numeric_values.push_back(std::move(numeric));
        table.categorical_values.push_back(std::move(categorical));
        table.labels.push_back(trim(rows[row][1]) == "M" ? 1 : 0);
    }

    return table;
}

Table load_heart(const std::string& path) {
    const auto rows = read_csv_rows(path);
    if (rows.size() < 2) {
        throw std::runtime_error("Heart-disease dataset is empty: " + path);
    }

    const std::vector<FeatureSpec> features{
        {"age", FeatureType::Numeric},
        {"sex", FeatureType::Categorical},
        {"dataset", FeatureType::Categorical},
        {"cp", FeatureType::Categorical},
        {"trestbps", FeatureType::Numeric},
        {"chol", FeatureType::Numeric},
        {"fbs", FeatureType::Categorical},
        {"restecg", FeatureType::Categorical},
        {"thalch", FeatureType::Numeric},
        {"exang", FeatureType::Categorical},
        {"oldpeak", FeatureType::Numeric},
        {"slope", FeatureType::Categorical},
        {"ca", FeatureType::Categorical},
        {"thal", FeatureType::Categorical},
    };

    Table table = make_empty_table("Heart Disease", features, {"no_disease", "disease_present"});

    for (std::size_t row = 1; row < rows.size(); ++row) {
        if (rows[row].size() < 16) {
            continue;
        }

        std::vector<double> numeric(features.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<std::string> categorical(features.size());

        numeric[0] = parse_double_or_nan(rows[row][1]);
        categorical[1] = trim(rows[row][2]);
        categorical[2] = trim(rows[row][3]);
        categorical[3] = trim(rows[row][4]);
        numeric[4] = parse_double_or_nan(rows[row][5]);
        numeric[5] = parse_double_or_nan(rows[row][6]);
        categorical[6] = trim(rows[row][7]);
        categorical[7] = trim(rows[row][8]);
        numeric[8] = parse_double_or_nan(rows[row][9]);
        categorical[9] = trim(rows[row][10]);
        numeric[10] = parse_double_or_nan(rows[row][11]);
        categorical[11] = trim(rows[row][12]);
        categorical[12] = trim(rows[row][13]);
        categorical[13] = trim(rows[row][14]);

        table.numeric_values.push_back(std::move(numeric));
        table.categorical_values.push_back(std::move(categorical));
        table.labels.push_back(std::stoi(trim(rows[row][15])) > 0 ? 1 : 0);
    }

    return table;
}

Table impute_with_training_statistics(const Table& raw, const std::vector<int>& train_indices) {
    Table prepared = raw;

    std::vector<double> numeric_medians(raw.features.size(), 0.0);
    for (std::size_t feature = 0; feature < raw.features.size(); ++feature) {
        if (raw.features[feature].type != FeatureType::Numeric) {
            continue;
        }

        std::vector<double> observed;
        for (int index : train_indices) {
            const double value = raw.numeric_values[index][feature];
            if (!std::isnan(value)) {
                observed.push_back(value);
            }
        }
        numeric_medians[feature] = median(std::move(observed));
    }

    for (std::size_t row = 0; row < prepared.labels.size(); ++row) {
        for (std::size_t feature = 0; feature < prepared.features.size(); ++feature) {
            if (prepared.features[feature].type == FeatureType::Numeric) {
                if (std::isnan(prepared.numeric_values[row][feature])) {
                    prepared.numeric_values[row][feature] = numeric_medians[feature];
                }
            } else if (trim(prepared.categorical_values[row][feature]).empty()) {
                prepared.categorical_values[row][feature] = "MISSING";
            }
        }
    }

    return prepared;
}

}  // namespace risk_aware_shift
