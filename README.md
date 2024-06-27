# TQQQ_LSTM_ROI_Prediction_Fifthday
Developing an LSTM-Based Machine Learning Model for Predicting TQQQ ROI on the Fifth Day

# Requirements
- Python 3.x
Pandas
NumPy
TensorFlow
Scikit-learn
Matplotlib
SQLite
Logging
Keras Tuner

# Installation
git clone https://github.com/yourusername/tqqq-single-roi-prediction.git
" cd tqqq-single-roi-prediction "


# Usage
Ensure your SQLite database (tqqq_stock_data.db) is in the same directory as the script.
Run the main script:
 "python main.py "

# Functions
Tqqq_preprocessed()
Preprocesses TQQQ data by fetching from SQLite, filtering for the last 13 years, and handling missing values.

create_sequences(data, sequence_length, forecast_horizon)
Creates sequences for the LSTM model with the specified sequence length and forecast horizon.

indicators_and_rolling_splitdata(preprocessed_data, sequence_length, forecast_horizon, train_size, valid_size)
Normalizes data and splits it into training, validation, and test sets using a rolling window.

build_model(sequence_length, num_features, l2_reg)
Builds and compiles an LSTM model with L2 regularization.

train_lstm_model(X_train, y_train, X_valid, y_valid, sequence_length, num_features, epochs, batch_size, model_path, forecast_horizon)
Trains the LSTM model and saves the best version based on validation loss.

evaluate_lstm_model(model, X_test, y_test)
Evaluates the LSTM model on the test set and plots Actual vs Predicted values.

plot_residuals(y_test, y_pred)
Plots the residuals between actual and predicted values.

compare_predictions(y_test, y_pred)
Compares actual and predicted values and logs the comparison.

predict_single_roi_5_days(model, recent_data, scaler, start_date, sequence_length)
Predicts the single ROI for the next 5 days using the trained model.

baseline_model(raw_train_data, raw_test_data, sequence_length, forecast_horizon)
Trains and evaluates a baseline Linear Regression model for comparison.

cross_validate_model(X, y, sequence_length, num_features, n_splits)
Performs cross-validation to evaluate the model's performance.

Logging
The script uses the logging module to log information, errors, and debug messages to the console. The logging level is set to INFO.

Model Training and Evaluation
The script includes functions to preprocess data, create sequences, build and train LSTM models, and evaluate their performance. It also provides visualization for training history and prediction results.

Hyperparameter Tuning
The script uses Keras Tuner's RandomSearch to find the best hyperparameters for the LSTM model.

Baseline Model
A baseline model using Linear Regression is trained and evaluated for comparison with the LSTM model.

Cross-Validation
Cross-validation is performed to ensure the model's robustness and generalizability.

Prediction
The script includes functionality to predict the Single_ROI_5 for the next 5 days based on recent data.

Visualization
The script uses Matplotlib to plot training history, actual vs. predicted values, and residuals.
