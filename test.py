import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from tabulate import tabulate
import datetime

def load_filtered_indices(file_path):
    """Loads index codes from the filtered indices output file."""
    try:
        df = pd.read_csv(file_path)
        return df['indexcode'].tolist()
    except Exception as e:
        print(f"Chat: Error loading filtered indices: {e}")
        return None

def get_stock_data_from_csv(indexcode, daily_data_path):
    """Reads stock data from a CSV file based on indexcode."""
    filename = f"{indexcode.replace('.','_')}.csv"
    file_path = os.path.join(daily_data_path, filename)
    if not os.path.exists(file_path):
      print(f"Chat: File not found: {file_path}")
      return None
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Chat: No data in file: {file_path}")
            return None
        return df
    except Exception as e:
        print(f"Chat: Error reading data from {file_path}: {e}")
        return None

def prepare_data_for_lstm(df, lookback=7):
    """Prepares data for LSTM training, scaling and creating sequences."""
    if df is None or 'Close' not in df.columns:
        return None, None, None
        
    # Check if 'Close' column contains numeric data
    if not pd.api.types.is_numeric_dtype(df['Close']):
        print(f"Chat: Non-numeric data found in 'Close' column. Skipping scaling for this index.")
        return None, None, None
    
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    try:
       scaled_data = scaler.fit_transform(data)
    except Exception as e:
        print(f"Chat: Error during scaling: {e}. Skipping this index.")
        return None, None, None


    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler


def create_lstm_model(input_shape):
    """Creates a basic LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_expected_high_low(df, predicted_close):
    """Calculates expected high and low based on historical data."""
    if df is None:
        return None, None
    last_20_days = df.tail(20)  #Taking the last 20 day data for calculating expected High and Low
    if last_20_days.empty:
         return None, None
    
    high_values = last_20_days['High'].values
    low_values = last_20_days['Low'].values
    
    if len(high_values) == 0 or len(low_values) == 0:
            return None, None
    
    average_high = np.mean(high_values)
    average_low = np.mean(low_values)

    expected_high = predicted_close + (average_high - np.mean(last_20_days['Close'].values))
    expected_low = predicted_close - (np.mean(last_20_days['Close'].values) - average_low)
    
    return expected_high, expected_low

def calculate_accuracy(df, predicted_close, lookback=7):
    """Calculates accuracy of the prediction based on last lookback days data"""
    if df is None or len(df) <= lookback:
        return None  # Not enough data for comparison
    actual_close_price = df['Close'].iloc[-1]
    accuracy = 100 - (abs(predicted_close- actual_close_price)/ actual_close_price *100)
    return accuracy


def train_and_predict(indexcode, daily_data_path, lookback=7):
    """
    Fetches stock data, prepares it, trains an LSTM model, and returns the last prediction,
    expected high and low, and accuracy
    """
    print(f"Chat: Starting processing for {indexcode}...")
    df = get_stock_data_from_csv(indexcode, daily_data_path)
    if df is None:
        print(f"Chat: No data for {indexcode}, skipping...")
        return None, None, None ,None

    X, y, scaler = prepare_data_for_lstm(df,lookback)

    if X is None or y is None:
        print(f"Chat: Unable to prepare data for {indexcode}, skipping...")
        return None, None,None, None

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_lstm_model(input_shape=(X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_input = X[-1]
    last_input = np.reshape(last_input, (1, last_input.shape[0], 1))
    predicted_scaled_value = model.predict(last_input, verbose = 0)[0][0]
    predicted_close = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

    expected_high, expected_low = calculate_expected_high_low(df, predicted_close)
    accuracy = calculate_accuracy(df, predicted_close)
    print(f"Chat: Prediction completed for {indexcode}")
    return predicted_close, expected_high, expected_low, accuracy


if __name__ == '__main__':
    filtered_indices_file = 'C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS//filtered_indices_output.csv'
    daily_data_path = 'C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS//Daily_data'
    output_file = 'C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS//lstm_output.csv' # Output file path
    lookback_days=7
    indexcodes = load_filtered_indices(filtered_indices_file)

    if indexcodes is None:
        print("Chat: No indices loaded. Exiting.")
        exit()

    results = []
    for indexcode in indexcodes:
      predicted_value, expected_high, expected_low, accuracy = train_and_predict(indexcode, daily_data_path, lookback_days)
      if predicted_value is not None:
        results.append({
            "Index Code": indexcode,
            "Predicted Close": f"{predicted_value:.2f}",
            "Expected High": f"{expected_high:.2f}" if expected_high is not None else "N/A",
            "Expected Low": f"{expected_low:.2f}" if expected_low is not None else "N/A",
            "Accuracy" : f"{accuracy:.2f}%" if accuracy is not None else "N/A"
            })

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)  # Save to CSV
        print(f"Chat: Results saved to: {output_file}")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))
    else:
        print("Chat: No results to display.")
