import yfinance as yf
import pandas as pd
import os
from datetime import date

def get_latest_date():
    today = date.today()
    return today.strftime("%Y-%m-%d")


def clean_and_save_data(data, filepath):
    # Reset index to include the 'Date' column in the DataFrame
    data.reset_index(inplace=True)

    # Remove column-level names, if any
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Remove rows with any non-numeric values in the data columns (excluding 'Date')
    for col in data.columns[1:]:  # Skip the 'Date' column
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()  # Drop rows with NaN values

    # Save the cleaned data to CSV
    data.to_csv(filepath, index=False)
    print(f"Cleaned data saved to: {filepath}")


# Function to download and save stock data
def download_stock_data(interval, folder):
    base_path = "C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS"
    filepath = os.path.join(base_path, "indicesstocks.csv")
    start_date = "2020-01-01"
    end_date = get_latest_date()
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            for symbol in symbols:
                symbol = symbol.strip()  # Remove any whitespace
                try:
                    # Download stock data
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                    ticketfilename = symbol.replace(".", "_")
                    save_path = os.path.join(base_path, folder, f"{ticketfilename}.csv")
                    # Clean the data and save
                    clean_and_save_data(data, save_path)
                except Exception as e:
                    print(f"Error downloading data for {symbol}: {e}")

# Main function
if __name__ == '__main__':
    print("Starting to download and clean stock data...")
    # Daily data
    download_stock_data(interval='1d', folder='Daily_data')
    # Weekly data
    download_stock_data(interval='1wk', folder='Weekly_data')
    # Monthly data
    download_stock_data(interval='1mo', folder='Monthly_data')
    print("All stock data downloaded and cleaned successfully!")
