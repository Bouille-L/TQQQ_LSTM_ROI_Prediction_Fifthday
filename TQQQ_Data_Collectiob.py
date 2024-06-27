import pandas as pd
import sqlite3
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Helper functions

# Function to connect to the SQLite database
def connect_to_database(db_path="tqqq_stock_data.db"):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        logging.info("Successfully connected to SQLite")
        return connection, cursor
    except sqlite3.Error as error:
        logging.error(f"Error while connecting to SQLite: {error}")
        return None, None

# Function to close the SQLite database connection
def close_database_connection(connection):
    if connection:
        connection.close()
        logging.info("SQLite connection is closed")

# Function to create a column if it does not exist
def create_column_if_not_exists(connection, cursor, table_name, column_name, column_type):
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        logging.info(f"Column '{column_name}' created successfully")
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e):
            logging.info(f"Column '{column_name}' already exists")
        else:
            logging.error(f"Error occurred: {e}")

# Function to fetch data from Yahoo Finance and store it in a SQLite database
def TQQQ_Data_Processing(proxy_server=None):
    connection, cursor = connect_to_database()
    if not connection or not cursor:
        return

    try:
        # Create the table if it does not exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS Tqqq_data(
                       Date TEXT,
                       Open REAL,
                       High REAL,
                       Low REAL,
                       Close REAL,
                       Adj_Close REAL,
                       Volume INTEGER
                       )''')
        logging.info("Table 'Tqqq_data' created successfully")

        # Fetch data from Yahoo Finance
        logging.info("Fetching data from yfinance...")
        data = yf.download("TQQQ", period="max", proxy=proxy_server)

        # Convert Date column to string format
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].astype(str)

        # Insert data into the database
        logging.info("Inserting data into table 'Tqqq_data'...")
        data.to_sql('Tqqq_data', connection, if_exists='replace', index=False)

        logging.info(f"Data loaded successfully. {len(data)} rows inserted.")
        return data  # Return the data DataFrame

    except Exception as e:
        logging.error(f"Error while fetching stock data: {e}")
    finally:
        close_database_connection(connection)

# Function to calculate single 5 days Return on Investment (ROI)
def Tqqq_Single_ROI_5(Tqqq_data):
    connection, cursor = connect_to_database()
    if not connection or not cursor:
        return

    try:
        # Create ROI column if it does not exist
        create_column_if_not_exists(connection, cursor, 'Tqqq_data', 'Single_ROI_5', 'REAL')

        # Calculate single 5-day ROI
        Tqqq_data['Single_ROI_5'] = Tqqq_data.apply(
            lambda row: ((Tqqq_data['Close'].shift(-5)[row.name] - row['Close']) / row['Close']) * 100
            if row.name <= len(Tqqq_data) - 6 else None, axis=1
        )

        # Drop rows where Average_ROI_5 cannot be calculated
        Tqqq_data = Tqqq_data.dropna(subset=['Single_ROI_5'])

        # Update the database with ROI values
        for index, row in Tqqq_data.iterrows():
            cursor.execute("UPDATE Tqqq_data SET Single_ROI_5 = ? WHERE Date = ?", (row['Single_ROI_5'], row['Date']))
        connection.commit()
        logging.info("Single_ROI_5 calculated and updated successfully")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        close_database_connection(connection)

# Main script to process the data and calculate indicators
if __name__ == "__main__":
    Tqqq_data = TQQQ_Data_Processing()
    Tqqq_Single_ROI_5(Tqqq_data)
   
