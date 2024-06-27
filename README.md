# TQQQ_LSTM__Prediction_Fifthday_ROI
Developing an LSTM-Based Machine Learning Model for Predicting TQQQ ROI on the Fifth Day


## File Descriptions:
### File 1: `TQQQ_data_processing.py`
This script fetches TQQQ stock data from Yahoo Finance, stores it in an SQLite database, and calculates a single 5-day Return on Investment (ROI).

#### Functions:
* `connect_to_database(db_path)`: Connects to the SQLite database.
* `close_database_connection(connection)`: Closes the SQLite database connection.
* `create_column_if_not_exists(connection, cursor, table_name, column_name, column_type)`: Creates a column if it doesn't exist.
* `TQQQ_Data_Processing(proxy_server)`: Fetches stock data from Yahoo Finance and stores it in the SQLite database.
* `Tqqq_Single_ROI_5(Tqqq_data)`: Calculates single 5-day ROI for the stock data.

### File 2: `LSTM_model.py`
This script preprocesses the TQQQ data, builds and trains an LSTM model, and evaluates its performance.

#### Functions:
* `Tqqq_preprocessed()`: Preprocesses the TQQQ data.
* `create_sequences(data, sequence_length, forecast_horizon)`: Creates sequences for LSTM input.
* `indicators_and_rolling_splitdata(preprocessed_data, sequence_length, forecast_horizon, train_size, valid_size)`: Splits and normalizes the data, generates sequences for training, validation, and testing.
* `build_model(sequence_length, num_features, l2_reg)`: Builds the LSTM model.
* `LSTMHyperModel`: Class for hyperparameter tuning.
* `tune_hyperparameters(X_train, y_train, X_valid, y_valid, sequence_length, num_features)`: Tunes hyperparameters for the LSTM model.
* `lr_schedule(epoch, lr)`: Learning rate scheduler.
* `train_lstm_model(X_train, y_train, X_valid, y_valid, sequence_length, num_features, epochs, batch_size, model_path, forecast_horizon)`: Trains the LSTM model.
* `evaluate_lstm_model(model, X_test, y_test)`: Evaluates the LSTM model.
* `plot_residuals(y_test, y_pred)`: Plots residuals.
* `compare_predictions(y_test, y_pred)`: Compares actual vs. predicted values.
* `predict_single_roi_5_days(model, recent_data, scaler, start_date, sequence_length)`: Predicts single ROI for 5 days ahead.
* `plot_training_history(history)`: Plots the training history.

## Requirements:
The following Python packages are required for this project:
* Python 3.X
* pandas: Data manipulation and analysis.
* numpy: Numerical operations.
* sqlite3: SQLite database operations (standard library, no installation needed).
* yfinance: Fetching stock data from Yahoo Finance.
* logging: Logging operations (standard library, no installation needed).
* tensorflow: Building and training the LSTM model.
* scikit-learn: Preprocessing data, building and evaluating machine learning models.
* matplotlib: Plotting graphs.
* keras-tuner: Hyperparameter tuning for Keras models.

## Installation:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bouille-L/TQQQ_LSTM_ROI_Prediction_Fifthday.git
   cd TQQQ_LSTM_ROI_Prediction_Fifthday
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source env/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install pandas numpy yfinance tensorflow scikit-learn matplotlib keras-tuner
   ```





