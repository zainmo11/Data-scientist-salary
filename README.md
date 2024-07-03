# Data Scientist Salary Prediction
- This project aims to predict the salary of data scientists using machine learning models implemented in both C++ and Python. The dataset used for this project has been cleaned for data analysis and modeling.

## Dependencies
### C++
- mlpack
- Eigen
- Armadillo
### Python
- scikit-learn
- pandas
- numpy

## Table of Contents
- Dataset
- Preprocessing
- Models in Python
- Models in C++
- Usage
## Accuracy
### using c++
![image](https://github.com/zainmo11/Data-scientist-salary/assets/89034348/6c2e6200-f769-4e0e-be91-1306fa214528)
### using python
![image](https://github.com/zainmo11/Data-scientist-salary/assets/89034348/bf114036-5b14-4573-922e-172b6726762a)

## Dataset
The dataset contains various features related to data science jobs. The initial dataset was cleaned to remove unwanted columns and prepare it for analysis and modeling.

### Dropped Columns
- index
- Company Name
- Salary Estimate
- Job Description
- Company Name
- Competitors
- Lower Salary
- Upper Salary
- company_txt
## Preprocessing
In the preprocessing step, the unwanted columns were dropped, and the dataset was split into training and testing sets. Additionally, the data was scaled and transformed as necessary to ensure optimal model performance.

## Models in Python
The following regression models were used in Python:

```python

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Initialize regression models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=1),
    'Random Forest': RandomForestRegressor(random_state=1)
}
```
These models were trained and evaluated using standard metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

## Models in C++
The following models were implemented using the mlpack, Eigen, and Armadillo libraries in C++:

### Linear Regression
``` cpp

// Train a linear regression model, fitting an intercept term and using an L2 regularization parameter of 0.3.
mlpack::regression::LinearRegression lr(data, responses, weights, 0.3, true);

// Compute the MSE on the training set.
double mse = lr.ComputeError(data, responses);
std::cout << "MSE on the training set: " << mse << std::endl;

// Compute predictions
arma::rowvec predictions;
lr.Predict(data, predictions);

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
```
### Bayesian Linear Regression
``` cpp
// Train a Bayesian Linear Regression model
mlpack::regression::BayesianLinearRegression blr;
blr.Train(data, responses);

// Compute the MSE on the training set
arma::rowvec predictions;
blr.Predict(data, predictions);
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
```
### LARS (Lasso Regression)
```cpp

// Split the data into training and test sets
arma::mat trainData, testData;
arma::rowvec trainResponses, testResponses;
const double testRatio = 0.2; // 20% of the data for testing
mlpack::data::Split(data, responses, trainData, testData, trainResponses, testResponses, testRatio);

// Train a LARS model with L1 regularization (LASSO)
mlpack::regression::LARS lars(true, 0.1); 
lars.Train(trainData, trainResponses);

// Compute the MSE on the training set
arma::rowvec trainPredictions;
lars.Predict(trainData, trainPredictions);
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
```
## Usage
To run the models, follow these steps:

### Python
Install the required dependencies:
``` bash
pip install -r requirements.txt
 ```
Run the Python script:
``` bash
python salary_prediction.py
```
### C++
- Install the required libraries (mlpack, Eigen, Armadillo).
- Compile and run the C++ code:
``` bash
g++ -std=c++11 salary_prediction.cpp -lmlpack -larmadillo -o salary_prediction

./salary_prediction
```
### Feel free to reach out if you have any questions or need further assistance.
