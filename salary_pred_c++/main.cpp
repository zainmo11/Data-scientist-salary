#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include <armadillo>
#include <stdexcept> 
#include <limits>    
#include <cctype>    
#include <mlpack/core.hpp>
#include <mlpack/mlpack.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <Eigen/Dense>
using namespace std;
using namespace arma;
using namespace mlpack::cv;
using namespace mlpack::regression;
using namespace mlpack::pca;
using namespace mlpack::tree;
using namespace mlpack::neighbor;
using namespace mlpack::svm;

// Function to read data from a file and convert it to a vector of vector of strings
std::vector<std::vector<std::string>> read(const std::string& filePath) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return data;
    }

    std::string line;
    // Read header line
    if (std::getline(file, line)) {
        std::istringstream headerStream(line);
        std::vector<std::string> headers;
        std::string header;

        // Parse header line to get column headers
        while (std::getline(headerStream, header, ',')) {
            headers.push_back(header);
        }
        data.push_back(headers);
    }

    // Read data rows
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::vector<std::string> row;
        std::string cell;

        // Iterate through each cell in the row
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}
void writeToTxt(const std::vector<std::vector<std::string>>& data, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Error: Could not open the file for writing!" << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) {
                outFile << "\t"; // Use tab to separate columns
            }
        }
        outFile << "\n"; // Newline after each row
    }

    outFile.close();
}
// Function to print a specific column
void printColumn(const std::vector<std::vector<std::string>>& dataset, const std::string& columnName) {
    if (dataset.empty()) {
        std::cerr << "Dataset is empty." << std::endl;
        return;
    }

    const std::vector<std::string>& headers = dataset[0];
    auto it = std::find(headers.begin(), headers.end(), columnName);

    if (it != headers.end()) {
        size_t columnIndex = std::distance(headers.begin(), it);
        std::cout << columnName << ": ";
        for (size_t i = 1; i < dataset.size(); ++i) {
            if (columnIndex < dataset[i].size()) {
                std::cout << dataset[i][columnIndex] << ", ";
            }
        }
        std::cout << std::endl;
    }
    else {
        std::cerr << "Column \"" << columnName << "\" not found in the dataset." << std::endl;
    }
}

std::vector<std::vector<std::string>> labelEncode(std::vector<std::vector<std::string>> dataset, int colIndex) {
    if (dataset.empty() || colIndex < 0 || colIndex >= dataset[0].size()) {
        return dataset;
    }

    std::unordered_map<std::string, int> labelMap;
    int labelCounter = 0;

    for (auto& row : dataset) {
        std::string value = row[colIndex];
        if (labelMap.find(value) == labelMap.end()) {
            labelMap[value] = labelCounter++;
        }
        row[colIndex] = std::to_string(labelMap[value]);
    }

    return dataset;
}

// Function to perform one-hot encoding on a specified column
std::vector<std::vector<std::string>> oneHotEncode(std::vector<std::vector<std::string>>& dataset, int colIndex) {
    if (dataset.empty() || colIndex >= dataset[0].size()) {
        throw std::invalid_argument("Invalid column index");
    }

    // Extract unique values from the specified column
    std::set<std::string> uniqueValues;
    for (const auto& row : dataset) {
        uniqueValues.insert(row[colIndex]);
    }

    // Create a map from unique values to column indices
    std::unordered_map<std::string, int> valueToIndex;
    int index = 0;
    for (const auto& value : uniqueValues) {
        valueToIndex[value] = index++;
    }

    // Create the new dataset with one-hot encoded columns
    std::vector<std::vector<std::string>> newDataset;

    // Add the rows with one-hot encoded values
    for (auto& row : dataset) {
        std::vector<std::string> newRow;
        newRow.reserve(row.size() + uniqueValues.size() - 1);

        // Add all columns except the one to be encoded
        for (size_t i = 0; i < row.size(); ++i) {
            if (i != colIndex) {
                newRow.push_back(row[i]);
            }
        }

        // Append zeros for the one-hot encoded columns
        for (size_t j = 0; j < uniqueValues.size(); ++j) {
            newRow.push_back("0");
        }

        // Set the appropriate column to "1" for the one-hot encoding
        const std::string& value = row[colIndex];
        newRow[row.size() - 1 + valueToIndex[value]] = "1";

        newDataset.push_back(newRow);
    }

    return newDataset;
}

std::vector<std::vector<std::string>> binaryEncode(std::vector<std::vector<std::string>> dataset, int colIndex) {
    if (dataset.empty() || colIndex < 0 || colIndex >= dataset[0].size()) {
        return dataset;
    }

    // Collect unique values and assign integer labels
    std::unordered_map<std::string, int> labelMap;
    int labelCounter = 0;
    for (const auto& row : dataset) {
        std::string value = row[colIndex];
        if (labelMap.find(value) == labelMap.end()) {
            labelMap[value] = labelCounter++;
        }
    }

    // Calculate the number of bits required for binary representation
    int numBits = std::ceil(std::log2(labelCounter));

    // Create new dataset with binary encoded columns
    std::vector<std::vector<std::string>> newDataset;
    for (const auto& row : dataset) {
        std::vector<std::string> newRow;
        for (int i = 0; i < row.size(); ++i) {
            if (i == colIndex) {
                int value = labelMap[row[i]];
                for (int bit = numBits - 1; bit >= 0; --bit) {
                    newRow.push_back(((value >> bit) & 1) ? "1" : "0");
                }
            }
            else {
                newRow.push_back(row[i]);
            }
        }
        newDataset.push_back(newRow);
    }

    return newDataset;
}
std::vector<std::vector<std::string>> minMaxScalar(std::vector<std::vector<std::string>>& dataset, int colIndex) {
    // Convert string values to numeric values (assuming they are convertible to double)
    std::vector<double> columnValues;
    for (const auto& row : dataset) {
        if (colIndex < row.size()) {
            std::istringstream iss(row[colIndex]);
            double value;
            if (iss >> value) {
                columnValues.push_back(value);
            }
            else {
                // Handle invalid numeric values if needed
                std::cerr << "Invalid numeric value: " << row[colIndex] << std::endl;
            }
        }
        else {
            // Handle column index out of range if needed
            std::cerr << "Column index out of range: " << colIndex << std::endl;
        }
    }

    // Calculate min and max values
    double min = columnValues.empty() ? 0.0 : *std::min_element(columnValues.begin(), columnValues.end());
    double max = columnValues.empty() ? 0.0 : *std::max_element(columnValues.begin(), columnValues.end());

    // Perform min-max scaling
    for (auto& row : dataset) {
        if (colIndex < row.size()) {
            double value;
            std::istringstream iss(row[colIndex]);
            if (iss >> value) {
                // Scale the value
                double scaledValue = (value - min) / (max - min);
                // Update the string representation
                row[colIndex] = std::to_string(scaledValue);
            }
            else {
                // Handle invalid numeric values if needed
                std::cerr << "Invalid numeric value: " << row[colIndex] << std::endl;
            }
        }
        else {
            // Handle column index out of range if needed
            std::cerr << "Column index out of range: " << colIndex << std::endl;
        }
    }

    return dataset;
}
void writeToCSV(const std::vector<std::vector<std::string>>& data, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
}

// Example usage
int main() {

    std::string filePath = "data.csv";
    std::vector<std::string> features = {
        "Rating", "Location", "Headquarters", "Size", "Founded", "Type of ownership", "Industry",
        "Sector", "Revenue", "Hourly", "Employer provided", "Avg Salary(K)", "Job Location", "Age",
        "Python", "spark", "aws", "excel", "sql", "sas", "keras", "pytorch", "scikit", "tensor",
        "hadoop", "tableau", "bi", "flink", "mongo", "google_an", "job_title_sim", "seniority_by_title", "Degree"
    };
    //arma::mat m;
    //mlpack::data::DatasetInfo info;
    //mlpack::data::Load(filePath, m, info, true);

    //// Print information about the data.
    //std::cout << "The matrix in 'data.csv' has: " << std::endl;
    //std::cout << " - " << m.n_cols << " points." << std::endl;
    //std::cout << " - " << info.Dimensionality() << " dimensions." << std::endl;

    //// Print which dimensions are categorical.
    //for (size_t d = 0; d < info.Dimensionality(); ++d)
    //{
    //    if (info.Type(d) == mlpack::data::Datatype::categorical)
    //    {
    //        std::cout << " - Dimension " << d << " is categorical with "
    //            << info.NumMappings(d) << " distinct categories." << std::endl;
    //    }
    //}
    std::vector<std::vector<std::string>> dataset = read(filePath);
    std::vector<std::vector<std::string>> dataEncoded;
    dataEncoded = oneHotEncode(dataset, 31);
    dataEncoded = binaryEncode(dataEncoded, 30);
    dataEncoded = binaryEncode(dataEncoded, 29);
    dataEncoded = binaryEncode(dataEncoded, 11);
    dataEncoded = binaryEncode(dataEncoded, 8);
    dataEncoded = binaryEncode(dataEncoded, 7);
    dataEncoded = binaryEncode(dataEncoded, 6);
    dataEncoded = oneHotEncode(dataEncoded, 5);
    dataEncoded = oneHotEncode(dataEncoded, 3);
    dataEncoded = binaryEncode(dataEncoded, 2);
    dataEncoded = binaryEncode(dataEncoded, 1);
    dataEncoded = minMaxScalar(dataEncoded, 0);
    dataEncoded = minMaxScalar(dataEncoded, 17);
    dataEncoded = minMaxScalar(dataEncoded, 41);
    writeToCSV(dataEncoded, "data_encoded.csv");

    
    // Load the data
    arma::mat data;
    mlpack::data::Load("data_encoded.csv", data, true);

    // Load the responses
    arma::rowvec responses;
    mlpack::data::Load("Result.txt", responses, true);



    //// Generate random instance weights for each point, in the range 0.5 to 1.5.
    //arma::rowvec weights(data.n_cols, arma::fill::randu);
    //weights += 0.5;

    //// Train a linear regression model, fitting an intercept term and using an L2 regularization parameter of 0.3.
    //mlpack::regression::LinearRegression lr(data, responses, weights, 0.3, true);

    //// Compute the MSE on the training set.
    //double mse = lr.ComputeError(data, responses);
    //std::cout << "MSE on the training set: " << mse << std::endl;

    //// Compute predictions
    //arma::rowvec predictions;
    //lr.Predict(data, predictions);

    //// Compute RMSE
    //double rmse = std::sqrt(mse);
    //std::cout << "RMSE on the training set: " << rmse << std::endl;

    //// Compute MAE
    //double mae = arma::mean(arma::abs(predictions - responses));
    //std::cout << "MAE on the training set: " << mae << std::endl;

    //// Compute R-squared
    //double mean_responses = arma::mean(responses);
    //double ss_tot = arma::accu(arma::pow(responses - mean_responses, 2));
    //double ss_res = arma::accu(arma::pow(responses - predictions, 2));
    //double r_squared = 1 - (ss_res / ss_tot);
    //std::cout << "R-squared on the training set: " << r_squared << std::endl;


    // Train a Bayesian Linear Regression model
    mlpack::regression::BayesianLinearRegression blr; // Step 1: create model.
    blr.Train(data, responses); // Step 2: train model.

    // Compute the MSE on the training set
    arma::rowvec predictions;
    blr.Predict(data, predictions); // Step 3: use model to predict.
    double mse = arma::mean(arma::square(predictions - responses));
    std::cout << "MSE on the training set: " << mse << std::endl;

    // Compute RMSE
    double rmse = std::sqrt(mse);
    std::cout << "RMSE on the training set: " << rmse << std::endl;

    // Compute MAE
    double mae = arma::mean(arma::abs(predictions - responses));
    std::cout << "MAE on the training set: " << mae << std::endl;

    // Compute R-squared
    double mean_responses = arma::mean(responses);
    double ss_tot = arma::accu(arma::pow(responses - mean_responses, 2));
    double ss_res = arma::accu(arma::pow(responses - predictions, 2));
    double r_squared = 1 - (ss_res / ss_tot);
    std::cout << "R-squared on the training set: " << r_squared << std::endl;

    // Print some information about the predictions
    std::cout << arma::accu(predictions > 0.6) << " training points predicted to have responses greater than 0.6." << std::endl;




    // Split the data into training and test sets
    arma::mat trainData, testData;
    arma::rowvec trainResponses, testResponses;
    const double testRatio = 0.2; // 20% of the data for testing
    mlpack::data::Split(data, responses, trainData, testData, trainResponses, testResponses, testRatio);

    // Train a LARS model with L1 regularization (LASSO)
    mlpack::regression::LARS lars(true, 0.1 /* L1 penalty */); // Step 1: create model.
    lars.Train(trainData, trainResponses); // Step 2: train model.

    // Compute the MSE on the training set
    arma::rowvec trainPredictions;
    lars.Predict(trainData, trainPredictions); // Step 3: use model to predict.
    double trainMSE = arma::mean(arma::square(trainPredictions - trainResponses));
    std::cout << "MSE on the training set: " << trainMSE << std::endl;

    // Compute RMSE on the training set
    double trainRMSE = std::sqrt(trainMSE);
    std::cout << "RMSE on the training set: " << trainRMSE << std::endl;

    // Compute MAE on the training set
    double trainMAE = arma::mean(arma::abs(trainPredictions - trainResponses));
    std::cout << "MAE on the training set: " << trainMAE << std::endl;

    // Compute R-squared on the training set
    double mean_trainResponses = arma::mean(trainResponses);
    double train_ss_tot = arma::accu(arma::pow(trainResponses - mean_trainResponses, 2));
    double train_ss_res = arma::accu(arma::pow(trainResponses - trainPredictions, 2));
    double train_r_squared = 1 - (train_ss_res / train_ss_tot);
    std::cout << "R-squared on the training set: " << train_r_squared << std::endl;

    // Predict on the test set
    arma::rowvec testPredictions;
    lars.Predict(testData, testPredictions);

    // Compute the MSE on the test set
    double testMSE = arma::mean(arma::square(testPredictions - testResponses));
    std::cout << "MSE on the test set: " << testMSE << std::endl;

    // Compute RMSE on the test set
    double testRMSE = std::sqrt(testMSE);
    std::cout << "RMSE on the test set: " << testRMSE << std::endl;

    // Compute MAE on the test set
    double testMAE = arma::mean(arma::abs(testPredictions - testResponses));
    std::cout << "MAE on the test set: " << testMAE << std::endl;

    // Compute R-squared on the test set
    double mean_testResponses = arma::mean(testResponses);
    double test_ss_tot = arma::accu(arma::pow(testResponses - mean_testResponses, 2));
    double test_ss_res = arma::accu(arma::pow(testResponses - testPredictions, 2));
    double test_r_squared = 1 - (test_ss_res / test_ss_tot);
    std::cout << "R-squared on the test set: " << test_r_squared << std::endl;

    // Print some information about the predictions
    std::cout << arma::accu(testPredictions > 0.6) << " test points predicted to have responses greater than 0.6." << std::endl;
    std::cout << arma::accu(testPredictions < 0) << " test points predicted to have negative responses." << std::endl;



    ////Apply PCA to reduce dimensions
    //mlpack::pca::PCA pca;
    //pca.Apply( data, 84); // Step 2: reduce data dimension to 5
    //// Train a Decision Tree Regressor
    // // Split the data into training and test sets
    //arma::mat trainData, testData;
    //arma::rowvec trainResponses, testResponses;
    //const double testRatio = 0.3; // 30% of the data for testing
    //mlpack::data::Split(data, responses, trainData, testData, trainResponses, testResponses, testRatio);

    //// Train a Decision Tree Regressor on the training set
    //mlpack::regression::DecisionTreeRegressor<> tree; // Step 1: create tree.
    //tree.Train(trainData, trainResponses); // Step 2: train model.

    //// Compute the MSE on the training set
    //arma::rowvec trainPredictions;
    //tree.Predict(trainData, trainPredictions); // Step 3: use model to predict
    //double trainMSE = arma::mean(arma::square(trainPredictions - trainResponses));
    //std::cout << "MSE on the training set: " << trainMSE << std::endl;

    //// Compute RMSE on the training set
    //double trainRMSE = std::sqrt(trainMSE);
    //std::cout << "RMSE on the training set: " << trainRMSE << std::endl;

    //// Compute MAE on the training set
    //double trainMAE = arma::mean(arma::abs(trainPredictions - trainResponses));
    //std::cout << "MAE on the training set: " << trainMAE << std::endl;

    //// Compute R-squared on the training set
    //double mean_trainResponses = arma::mean(trainResponses);
    //double train_ss_tot = arma::accu(arma::pow(trainResponses - mean_trainResponses, 2));
    //double train_ss_res = arma::accu(arma::pow(trainResponses - trainPredictions, 2));
    //double train_r_squared = 1 - (train_ss_res / train_ss_tot);
    //std::cout << "R-squared on the training set: " << train_r_squared << std::endl;

    //// Compute the MSE on the test set
    //arma::rowvec testPredictions;
    //tree.Predict(testData, testPredictions); // Predict on the test set
    //double testMSE = arma::mean(arma::square(testPredictions - testResponses));
    //std::cout << "MSE on the test set: " << testMSE << std::endl;

    //// Compute RMSE on the test set
    //double testRMSE = std::sqrt(testMSE);
    //std::cout << "RMSE on the test set: " << testRMSE << std::endl;

    //// Compute MAE on the test set
    //double testMAE = arma::mean(arma::abs(testPredictions - testResponses));
    //std::cout << "MAE on the test set: " << testMAE << std::endl;

    //// Compute R-squared on the test set
    //double mean_testResponses = arma::mean(testResponses);
    //double test_ss_tot = arma::accu(arma::pow(testResponses - mean_testResponses, 2));
    //double test_ss_res = arma::accu(arma::pow(testResponses - testPredictions, 2));
    //double test_r_squared = 1 - (test_ss_res / test_ss_tot);
    //std::cout << "R-squared on the test set: " << test_r_squared << std::endl;

    return 0;
}