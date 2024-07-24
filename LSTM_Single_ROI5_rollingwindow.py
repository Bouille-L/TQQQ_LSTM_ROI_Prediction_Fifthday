import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sqlite3
import logging
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.stats import zscore
import yfinance as yf


# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to preprocess TQQQ data for LSTM
def Tqqq_preprocessed():
    connection = None
    try:
        logging.info("Starting data preprocessing...")
        
        # Connect to SQLite database and fetch data
        connection = sqlite3.connect("tqqq_stock_data.db")
        query = "SELECT * FROM Tqqq_data"
        logging.info("Fetching data from the SQLite database.")
        
        # Fetch all data from the Tqqq_data table
        Tqqq_data = pd.read_sql(query, connection)
        
        # Drop rows with missing values
        Tqqq_data = Tqqq_data.dropna()
        logging.info(f"Data shape after dropping missing values: {Tqqq_data.shape}")
        
         # Detect and handle outliers using Z-score method
        Tqqq_data['zscore'] = zscore(Tqqq_data['Single_ROI_5'])
        Tqqq_data = Tqqq_data[(Tqqq_data['zscore'] < 3) & (Tqqq_data['zscore'] > -3)]
        Tqqq_data = Tqqq_data.drop(columns=['zscore'])
        logging.info(f"Data shape after handling outliers: {Tqqq_data.shape}")
        
        # Convert 'Date' column to datetime
        Tqqq_data['Date'] = pd.to_datetime(Tqqq_data['Date'])
        
              
        # Sort data by date and reset index
        Tqqq_data = Tqqq_data.sort_values('Date')
        Tqqq_data = Tqqq_data.reset_index(drop=True)
        
        logging.info("Data preprocessing completed.")
        return Tqqq_data
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")
        raise
    finally:
        if connection:
            connection.close()

# Function to create sequences for LSTM model
def create_sequences(data, sequence_length, forecast_horizon):
    try:
        logging.info("Starting sequence creation...")

        # Initialize lists to store sequences and targets
        sequences = []
        targets = []

        # Ensure the target column exists in the data
        target_column = 'Single_ROI_5'
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data")

        # Convert the data to a NumPy array for easier manipulation
        data_values = data.values

        # Index of the target column in the data
        target_column_index = data.columns.get_loc(target_column)

        # Generate sequences and corresponding targets
        for i in range(len(data_values) - sequence_length - forecast_horizon + 1):
            sequence = data_values[i:i + sequence_length]  # Sequence of indicators
            target = data_values[i + sequence_length, target_column_index]  # Single ROI_5
            sequences.append(sequence)
            targets.append(target)

        # Convert lists to NumPy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)

        logging.info(f"Generated {len(sequences)} sequences.")
        return sequences, targets
    except Exception as e:
        logging.error(f"An error occurred during sequence creation: {e}")
        raise

# Function to normalize data and generate sequences using rolling window
def indicators_and_rolling_splitdata(preprocessed_data, sequence_length=5, forecast_horizon=5, train_size=0.7, valid_size=0.15):
    try:
        logging.info("Starting data splitting and indicator selection...")
        
        # Select only the 'Single_ROI_5' column
        data = preprocessed_data[['Single_ROI_5']]
        
        # Normalize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logging.info("Data normalization complete.")

        # Calculate split indices
        total_length = len(data_scaled)
        train_split = int(total_length * train_size)
        valid_split = int(total_length * (train_size + valid_size))

        # Create sequences using rolling window for training, validation, and testing
        X_train, y_train = create_sequences(pd.DataFrame(data_scaled[:train_split], columns=['Single_ROI_5']), sequence_length, forecast_horizon)
        X_valid, y_valid = create_sequences(pd.DataFrame(data_scaled[train_split:valid_split], columns=['Single_ROI_5']), sequence_length, forecast_horizon)
        X_test, y_test = create_sequences(pd.DataFrame(data_scaled[valid_split:], columns=['Single_ROI_5']), sequence_length, forecast_horizon)

        # Store raw data splits for baseline model and inverse transformation
        raw_train_data = data[:train_split]
        raw_valid_data = data[train_split:valid_split]
        raw_test_data = data[valid_split:]

        logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_valid.shape}, Test data shape: {X_test.shape}")
        logging.info("Data splitting and indicator selection completed.")
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, raw_train_data, raw_valid_data, raw_test_data
    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}")
        raise

# Function to build the LSTM model with L2 regularization
def build_model(sequence_length, num_features, l2_reg=0.01):
    model = Sequential()
    model.add(Input(shape=(sequence_length, num_features)))
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.num_features)))
        model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50), return_sequences=True, kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.01, max_value=0.1, step=0.01))))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50), kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.01, max_value=0.1, step=0.01))))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

def tune_hyperparameters(X_train, y_train, X_valid, y_valid, sequence_length, num_features):
    hypermodel = LSTMHyperModel()
    hypermodel.sequence_length = sequence_length
    hypermodel.num_features = num_features

    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=50,  # Increased the number of trials
        executions_per_trial=3,
        directory='hyperparameter_tuning',
        project_name='tuning_lstm'
    )

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner.results_summary()

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Function to train the LSTM model
def train_lstm_model(X_train, y_train, X_valid, y_valid, sequence_length=5, num_features=1, epochs=100, 
                     batch_size=32, model_path='best_model.keras', forecast_horizon=5):
    try:
        logging.info("Starting LSTM model training...")

        # Perform hyperparameter tuning
        model, tuner_summary = tune_hyperparameters(X_train, y_train, X_valid, y_valid, sequence_length, num_features)
        logging.info(tuner_summary)

        # Define callbacks for early stopping, model checkpointing, and learning rate adjustment
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)

        # Train the model on the training data
        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=(X_valid, y_valid),
                            callbacks=[early_stopping, model_checkpoint, lr_scheduler])
                            
        # Save the trained model to the specified path
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Return the trained model and training history
        logging.info("LSTM model training completed.")
        return model, history
    except Exception as e:
        logging.error(f"An error occurred during LSTM model training: {e}")
        raise

# Function to calculate the risk free rate
def calculate_5day_risk_free_rate(ticker="^IRX", periods_per_year=252, forecast_horizon=5):
    """
    Fetch the latest yield for the specified Treasury security and convert the annual risk-free rate 
    to the corresponding forecast horizon rate.

    Parameters:
    ticker (str): The ticker symbol for the Treasury security ("^IRX" for 3-month Treasury bill).
    periods_per_year (int): Number of trading days in a year (typically 252).
    forecast_horizon (int): Number of days in the forecast horizon (e.g., 5 days).

    Returns:
    float: Risk-free rate for the forecast horizon.
    """
    # Fetch the historical market data
    treasury_data = yf.Ticker(ticker)
    hist = treasury_data.history(period="1d")
    
    # Get the latest yield
    latest_yield = hist['Close'].iloc[-1]
    
    # Convert basis points to a decimal 
    annual_rate = latest_yield / 100
    
    # Convert the annual rate to a daily rate
    daily_rate = (1 + annual_rate) ** (1 / periods_per_year) - 1
    
    # Convert the daily rate to a forecast horizon rate (5 days)
    horizon_rate = (1 + daily_rate) ** forecast_horizon - 1
    
    return horizon_rate

# Function to calculate the Sharpe ratio using 5-day returns
def calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year=252):
    """
    Calculate the Sharpe ratio for the given returns over the forecast horizon.

    Parameters:
    returns (np.array): Array of returns for the forecast horizon.
    risk_free_rate (float): The risk-free rate for the forecast horizon 
    periods_per_year (int): Number of trading days in a year (typically 252).

    Returns:
    float: The Sharpe ratio.
    """
    # Calculate excess returns by subtracting the risk-free rate from returns
    excess_returns = returns - risk_free_rate
    
    # Calculate the Sharpe ratio: mean of excess returns divided by their standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0
    
    return mean_excess_return / std_excess_return

# LSTM evaluate function
def evaluate_lstm_model(model, X_test, y_test, risk_free_rate, forecast_horizon=5):
    """
    Evaluate the LSTM model on the test set and calculate performance metrics including Sharpe ratio.

    Parameters:
    model (tf.keras.Model): The trained LSTM model.
    X_test (np.array): The test features.
    y_test (np.array): The actual test values.
    risk_free_rate (float): The risk-free rate for the forecast horizon
    forecast_horizon (int): Number of days in the forecast horizon (5 days).

    Returns:
    dict: A dictionary containing various evaluation metrics.
    """
    try:
        logging.info("Starting LSTM model evaluation...")

        # Evaluate the model on the test data
        test_loss = model.evaluate(X_test, y_test, verbose=0)

        # Predict the values for the test set
        y_pred = model.predict(X_test)

        # Calculate excess returns for y_test and y_pred
        actual_excess_returns = y_test.flatten() - risk_free_rate
        predicted_excess_returns = y_pred.flatten() - risk_free_rate

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Calculate Sharpe ratio
        predicted_sharpe_ratio = calculate_sharpe_ratio(predicted_excess_returns, risk_free_rate)
        actual_sharpe_ratio = calculate_sharpe_ratio(actual_excess_returns, risk_free_rate)

        metrics = {
            'Loss': test_loss,
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R-squared': r2,
            'Root Mean Squared Error': rmse,
            'Predicted Sharpe Ratio': predicted_sharpe_ratio,
            'Actual Sharpe Ratio': actual_sharpe_ratio
        }

        # Log the evaluation metrics
        logging.info("Evaluation Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

        # Print both Sharpe ratios
        print(f"Predicted Sharpe Ratio: {predicted_sharpe_ratio}")
        print(f"Actual Sharpe Ratio: {actual_sharpe_ratio}")

        # Plot Actual vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.flatten(), label='Actual')
        plt.plot(y_pred.flatten(), label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # Scatter Plot for Actual vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test.flatten(), y_pred.flatten())
        plt.title('Actual vs Predicted Scatter Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

        logging.info("LSTM model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")
        raise

# Function to plot residuals
def plot_residuals(y_test, y_pred):
    try:
        logging.info("Starting residuals plotting...")

        y_pred = y_pred.flatten()
        residuals = y_test.flatten() - y_pred
        logging.info(f"y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}")
        logging.info(f"Residuals range: {residuals.min()} to {residuals.max()}")

        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--', linewidth=1)
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

        logging.info("Residuals plotting completed.")
    except Exception as e:
        logging.error(f"An error occurred while plotting residuals: {e}")
        raise

# Function to compare predictions with actual values
def compare_predictions(y_test, y_pred):
    try:
        logging.info("Comparing predictions with actual values...")
        comparison_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        logging.info(comparison_df.head(10))
        logging.info("Comparison of predictions completed.")
    except Exception as e:
        logging.error(f"An error occurred while comparing predictions: {e}")
        raise

# Function to predict the single ROI for the next 5 days
def predict_single_roi_5_days(model, recent_data, scaler, start_date, sequence_length=5):
    try:
        logging.info("Starting single ROI prediction for 5 days ahead...")

        # Ensure recent_data has the correct shape (sequence_length, num_features)
        if recent_data.shape != (sequence_length, 1):
            raise ValueError(f"Expected shape {(sequence_length, 1)} but got {recent_data.shape}")

        # Reshape data for prediction
        current_sequence = recent_data[np.newaxis, :, :]  # Shape (1, sequence_length, num_features)

        # Predict the single ROI for 5 days ahead
        prediction = model.predict(current_sequence)
        prediction = prediction.flatten()[0]

        # Prepare a dummy array for inverse transform
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[0, 0] = prediction  # Use index 0 for the only feature
        prediction_rescaled = scaler.inverse_transform(dummy_array)[:, 0][0]

        # Calculate the prediction date
        prediction_date = start_date + timedelta(days=5)

        logging.info(f"Prediction for single ROI over 5 days starting from {start_date}: {prediction_rescaled}")
        return prediction_date, prediction_rescaled
    except Exception as e:
        logging.error(f"An error occurred during single ROI prediction: {e}")
        raise


# Function to train and evaluate the baseline model
def baseline_model(raw_train_data, raw_test_data, sequence_length=5, forecast_horizon=5):
    try:
        logging.info("Starting baseline model training...")

        # Extract the 'Single_ROI_5' column for training and testing
        train_roi_5 = raw_train_data['Single_ROI_5'].values
        test_roi_5 = raw_test_data['Single_ROI_5'].values

        # Create sequences for training
        X_train, y_train = [], []
        for i in range(len(train_roi_5) - sequence_length - forecast_horizon + 1):
            X_train.append(train_roi_5[i:i + sequence_length])
            y_train.append(train_roi_5[i + sequence_length])  # Single ROI_5

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Create sequences for testing
        X_test, y_test = [], []
        for i in range(len(test_roi_5) - sequence_length - forecast_horizon + 1):
            X_test.append(test_roi_5[i:i + sequence_length])
            y_test.append(test_roi_5[i + sequence_length])  # Single ROI_5

        # Convert to numpy arrays
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Train the baseline model using Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict using the baseline model
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        metrics = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'R-squared': r2_score(y_test, y_pred),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        # Log the evaluation metrics
        logging.info("Baseline Model Evaluation Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

        # Plot Actual vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Baseline Model - Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Single_ROI_5')
        plt.legend()
        plt.show()

        # Scatter Plot for Actual vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred)
        plt.title('Baseline Model - Actual vs Predicted Scatter Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

        logging.info("Baseline model training completed.")
        return model, metrics
    except Exception as e:
        logging.error(f"An error occurred during baseline model training and evaluation: {e}")
        raise

# Function to perform cross-validation
def cross_validate_model(X, y, sequence_length=5, num_features=1, n_splits=5):
    try:
        logging.info("Starting cross-validation...")
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        results = []

        for train_idx, test_idx in kfold.split(X, y):
            logging.info(f"Training fold {fold}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = build_model(sequence_length, num_features)
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
            
            # Evaluate the model on the test data
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            fold_results = {
                'Fold': fold,
                'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'R-squared': r2,
                'Root Mean Squared Error': rmse
            }
            results.append(fold_results)
            logging.info(f"Fold {fold} results: {fold_results}")

            fold += 1

        avg_results = {
            'Mean Squared Error': np.mean([r['Mean Squared Error'] for r in results]),
            'Mean Absolute Error': np.mean([r['Mean Absolute Error'] for r in results]),
            'R-squared': np.mean([r['R-squared'] for r in results]),
            'Root Mean Squared Error': np.mean([r['Root Mean Squared Error'] for r in results])
        }

        logging.info("Cross-validation completed. Average results:")
        logging.info(avg_results)
        return avg_results
    except Exception as e:
        logging.error(f"An error occurred during cross-validation: {e}")
        raise


# Function to visualize training history
def plot_training_history(history):
    # Extract loss values from the training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    try:
        logging.info("Starting the end-to-end process...")

        # Preprocess the data
        data = Tqqq_preprocessed()

        # Define parameters
        time_steps = 5
        num_features = 1  # Since we are now only using 'Single_ROI_5'
        forecast_horizon = 5

        # Split data into training, validation, and testing sets and scale it
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, raw_train_data, raw_valid_data, raw_test_data = indicators_and_rolling_splitdata(data, sequence_length=time_steps, forecast_horizon=forecast_horizon)

        # Calculate the 5-day risk-free rate using the 3-month Treasury bill
        risk_free_rate_5day = calculate_5day_risk_free_rate("^IRX")
        print(f"5-Day Risk-Free Rate used: {risk_free_rate_5day}")

        # Train the LSTM model
        model, history = train_lstm_model(X_train, y_train, X_valid, y_valid, sequence_length=time_steps, num_features=num_features, forecast_horizon=forecast_horizon)

        # Visualize the training history
        plot_training_history(history)

        # Evaluate the LSTM model
        metrics = evaluate_lstm_model(model, X_test, y_test, risk_free_rate_5day)

        # Log the additional metrics
        logging.info(f"Predicted Sharpe Ratio: {metrics['Predicted Sharpe Ratio']}")
        logging.info(f"Actual Sharpe Ratio: {metrics['Actual Sharpe Ratio']}")

        # Make predictions with the LSTM model
        y_pred = model.predict(X_test)
        plot_residuals(y_test, y_pred)
        compare_predictions(y_test, y_pred)

        # Predict the single ROI for the next 5 days
        recent_data = np.concatenate((X_train[-1], X_test[-1]), axis=0)[-time_steps:]
        start_date = datetime.now()
        prediction_date, prediction_rescaled = predict_single_roi_5_days(model, recent_data, scaler, start_date, sequence_length=time_steps)
        logging.info(f"Prediction Date: {prediction_date.date()}, Predicted Single ROI: {prediction_rescaled:.2f}")

        # Train and evaluate the baseline model
        baseline_metrics = baseline_model(raw_train_data, raw_test_data, sequence_length=time_steps, forecast_horizon=forecast_horizon)

        # Perform cross-validation
        X = np.concatenate((X_train, X_valid, X_test), axis=0)
        y = np.concatenate((y_train, y_valid, y_test), axis=0)
        cross_val_results = cross_validate_model(X, y, sequence_length=time_steps, num_features=num_features)

    except Exception as e:
        logging.error(f"An error occurred during the end-to-end process: {e}")

