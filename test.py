import pandas as pd
import datetime as dt
from pathlib import Path
import pandas_ta as ta
from tabulate import tabulate

def append_row(df, row):
    """Appends a new row to a pandas DataFrame if the row is not empty."""
    if row.isnull().all():
        return df  # Do not append if the row is all NA
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)

def getRSI14_and_BB(csvfilename):
    """Calculates RSI (14 period) and Bollinger Bands (20 period, 2 std.dev) for a given CSV file."""
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty or 'Close' not in df.columns:
                return 0.00, 0.00, 0.00, 0.00
            else:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['rsi14'] = ta.rsi(df['Close'], length=14)
                bb = ta.bbands(df['Close'], length=20)
                if bb is None or df['rsi14'] is None:
                    return 0.00, 0.00, 0.00, 0.00
                df['lowerband'] = bb['BBL_20_2.0']
                df['middleband'] = bb['BBM_20_2.0']
                if pd.isna(df['rsi14'].iloc[-1]) or pd.isna(df['lowerband'].iloc[-1]) or pd.isna(df['middleband'].iloc[-1]):
                    return 0.00, 0.00, 0.00, 0.00
                else:
                    rsival = df['rsi14'].iloc[-1].round(2)
                    ltp = df['Close'].iloc[-1].round(2)
                    lowerband = df['lowerband'].iloc[-1].round(2)
                    middleband = df['middleband'].iloc[-1].round(2)
                    return rsival, ltp, lowerband, middleband
        except Exception as e:
            print(f"Error reading {csvfilename}: {e}")
            return 0.00, 0.00, 0.00, 0.00
    else:
        print(f"File does not exist: {csvfilename}")
        return 0.00, 0.00, 0.00, 0.00

def dayweekmonth_datasets(symbol, symbolname):
    """Calculates RSI, Bollinger Bands, and other metrics for daily, weekly, and monthly data."""
    # Replace periods with underscores in the symbol for file naming
    symbol_with_underscore = symbol.replace('.', '_')

    # Construct the file paths using the modified symbol
    daylocationstr = f'DATASETS/Daily_data/{symbol_with_underscore}.csv'
    weeklocationstr = f'DATASETS/Weekly_data/{symbol_with_underscore}.csv'
    monthlocationstr = f'DATASETS/Monthly_data/{symbol_with_underscore}.csv'

    cday = dt.datetime.today().strftime('%d/%m/%Y')
    dayrsi14, dltp, daylowerband, daymiddleband = getRSI14_and_BB(daylocationstr)
    weekrsi14, wltp, weeklowerband, weekmiddleband = getRSI14_and_BB(weeklocationstr)
    monthrsi14, mltp, monthlowerband, monthmiddleband = getRSI14_and_BB(monthlocationstr)

    new_row = pd.Series({
        'entrydate': cday,
        'indexcode': symbol,
        'indexname': symbolname,
        'dayrsi14': dayrsi14,
        'weekrsi14': weekrsi14,
        'monthrsi14': monthrsi14,
        'dltp': dltp,
        'daylowerband': daylowerband,
        'daymiddleband': daymiddleband,
        'weeklowerband': weeklowerband,
        'weekmiddleband': weekmiddleband,
        'monthlowerband': monthlowerband,
        'monthmiddleband': monthmiddleband
    })
    return new_row

def generateGFS(scripttype):
    """Generates the GFS report based on the provided scripttype."""
    indicesdf = pd.DataFrame(columns=['entrydate', 'indexcode', 'indexname', 'dayrsi14', 'weekrsi14', 'monthrsi14', 'dltp', 'daylowerband', 'daymiddleband', 'weeklowerband', 'weekmiddleband', 'monthlowerband', 'monthmiddleband'])

    fname = f'DATASETS/{scripttype}.csv'
    csvfilename = f'GFS_{scripttype}.csv'
    try:
        with open(fname) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                symbol = symbol.replace("\n", "")
                new_row = dayweekmonth_datasets(symbol, symbolname)
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

    indicesdf.to_csv(csvfilename, index=False)
    return indicesdf

def read_indicesstocks(csvfilename):
    """Reads the indicesstocks CSV file and returns a dictionary of indices and their stocks."""
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename, header=None, on_bad_lines='skip')  # Use 'on_bad_lines' for newer versions
            indices_dict = {}
            for index, row in df.iterrows():
                index_code = row[0].strip()  # Get the index code
                stocks = row[1:].dropna().tolist()  # Get the stocks, drop NaN values
                indices_dict[index_code] = stocks
            return indices_dict
        except Exception as e:
            print(f"Error reading {csvfilename}: {e}")
            return {}
    else:
        print(f"File does not exist: {csvfilename}")
        return {}

# Main execution
# Main execution
indicesdf_path = 'DATASETS/indicesdf.csv'
indicesstocks_path = r'C:\Users\manoj\Downloads\Major project data\Major pro source codes\DATASETS\indicesstocks.csv'
filtered_indices_path = 'DATASETS/filtered_indices.csv'  # Path for the new CSV file

# Generate GFS report
df3 = generateGFS('indicesdf')

# Filter based on criteria
df4 = df3.loc[
    (df3['monthrsi14'] >= 60.00) &
    (df3['weekrsi14'] >= 60.00) &
    df3['dayrsi14'].between(30, 70) &
    (df3['dltp'] > df3['daylowerband']) &
    (df3['dltp'] < df3['daymiddleband'])
]

# Extract only the indexcode for the filtered indices
filtered_indexcodes = df4[['indexcode']]

# Save the filtered indexcodes to a new CSV file, overwriting any existing file
filtered_indexcodes.to_csv(filtered_indices_path, index=False)

# Check if any indices qualified
if filtered_indexcodes.empty:
    print("\033[1mNO STOCKS QUALIFY THE GFS CRITERIA\033[0m")
else:
    # Read indices from indicesstocks
    indicesstocks = read_indicesstocks(indicesstocks_path)

    # Read filtered indices
    filtered_indices = filtered_indexcodes['indexcode'].tolist()

    # Compare and run GFS for matched indices
for index in filtered_indices:
    if index in indicesstocks:
        print(f"Running GFS for matched index: {index}")
        stocks = indicesstocks[index]  # Get the list of stocks for the matched index 
        
        # Traverse through the stocks starting from the next position
        # Inside your loop where you print the matched_row
for stock in stocks[1:]:  # Start from the second stock (index + 1)
    if stock:  # Check if stock is not empty
        print(f"Running GFS for stock: {stock}")
        try:
            # Call the GFS function for each stock
            matched_row = dayweekmonth_datasets(stock, stock)  # Using stock as both symbol and symbolname
            
            # Convert the matched_row to a vertical format
            vertical_table = {key: [value] for key, value in matched_row.items()}
            
            # Use tabulate to print the row in vertical format
            print(tabulate(vertical_table.items(), headers=['Metric', 'Value'], tablefmt='fancy_grid'))
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")
    else:
        print(f"Skipping empty stock name in index {index}.")