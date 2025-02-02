import pandas as pd
import numpy as np
import yfinance as yf
import os
# import math
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

# Function to create dataset for time-series prediction
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Main function to get predicted values
def getpredictedvalues(selectedscript_1, start_date=None, end_date=None):
    # selectedscript_1 = daily_data
    # Finding null values, if any
    selectedscript_1.isnull().sum()

    # Removing the row which have null value
    selectedscript_2 = selectedscript_1.dropna().reset_index(drop=True)

    # Checking whether if there exist any null values
    selectedscript_2[selectedscript_2.isnull().any(axis=1)]

    # Making a copy of dataset as selectedscript
    selectedscript = selectedscript_2.copy()

    # Converting the date column into datetime 
    selectedscript['Date'] = pd.to_datetime(selectedscript['Date'], format='%Y-%m-%d')

    # Filter data based on the specified date range
    if start_date and end_date:
        selectedscript = selectedscript[(selectedscript['Date'] >= start_date) & (selectedscript['Date'] <= end_date)]

    # Setting the date column as index
    selectedscript = selectedscript.set_index('Date')

    ## Model Building - Creating dataframe which only includes date and close time
    close_df = pd.DataFrame(selectedscript['Close'])
    close_df = close_df.reset_index()

    ### Normalizing / scaling close value between 0 to 1
    close_stock = close_df.copy()
    del close_df['Date']
    scaler = MinMaxScaler(feature_range=(0,1))
    closedf = scaler.fit_transform(np.array(close_df).reshape(-1,1))
    #print(closedf.shape)

    ### Split data for training and testing
    #- Ratio for training and testing data is 80:20
    training_size = int(len(closedf)*0.80)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size,:], closedf[training_size:len(closedf),:1]
    #print("train_data: ", train_data.shape)
    #print("test_data: ", test_data.shape)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 13
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    #print("X_train: ", X_train.shape)
    #print("y_train: ", y_train.shape)
    #print("X_test: ", X_test.shape)
    #print("y_test", y_test.shape)

    ## Algorithms - LSTM - reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #print("X_train: ", X_train.shape)
    #print("X_test: ", X_test.shape)

    ### LSTM model structure
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step,1)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    ### Model Training
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict.shape, test_predict.shape

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

    ### R2 score for regression
    train_r2_lstm = r2_score(original_ytrain, train_predict)
    test_r2_lstm = r2_score(original_ytest, test_predict)
    #print("Train data R2 score:", train_r2_lstm)
    #print("Test data R2 score:", test_r2_lstm)

    ### Comparision between original stock close price vs predicted close price
    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    #print("Train predicted data: ", trainPredictPlot.shape)

    ### Predicting next 5 days
    x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0
    pred_days = 5
    while(i < pred_days):
        if(len(temp_input) > time_step):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i+1

    # #print("Output of predicted next days: ", len(lst_output))

    lstmdf = closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf = scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]
    finaldf = pd.DataFrame({'lstm': lstmdf})

    data = {"Model": ["LSTM"], "Train R2 Score": [train_r2_lstm], "Test R2 Score": [test_r2_lstm]}
    df = pd.DataFrame(data)
    # #print(df)
    # #print(finaldf.to_string())
    # #print(selectedscript.to_string()) 

    return df, finaldf, selectedscript
    
    import pandas as pd
import os
from tabulate import tabulate

# Load data from the CSV file
file_path = 'C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS//filtered_indices_output.csv'
daily_data_path = 'C://Users//manoj//Downloads//Major project data//Major pro source codes//DATASETS//Daily_data'

try:
    selected_indices = pd.read_csv(file_path)
    #print(f"Data loaded successfully from {file_path}")
except FileNotFoundError:
    #print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    #print(f"An error occurred: {e}")
    exit()

# Iterate through each unique index code
unique_index_codes = selected_indices['indexcode'].unique()
for index_code in unique_index_codes:
    # Filter the selected indices for the current index code
    filtered_indices = selected_indices[selected_indices['indexcode'] == index_code]
    
    # Iterate through each row of the filtered indices
    for id, row in filtered_indices.iterrows():
        if(id>=0):
            index_name = row['indexname']
            
            # Construct the file path for the daily data
            daily_file_name = f"{index_name.replace('.', '_')}.csv"
            daily_file_path = os.path.join(daily_data_path, daily_file_name)
            
            try:
                daily_data = pd.read_csv(daily_file_path)
                                
                # #print the DataFrame in tabular format using tabulate
                #print(tabulate(daily_data.head(), headers='keys', tablefmt='fancy_grid', showindex=False))

                '''
                df = % of accuracy for training and testing data
                finaldf = predicted values
                selectedscript = original data                
                ''' 
                df, finaldf, selectedscript = getpredictedvalues(daily_data)
                predectedvalues=finaldf.tail(5)
                
                print(f"\nData for Index Code: {index_code}, Index Name: {index_name}")
                print(df)
                print(predectedvalues)

                
            except FileNotFoundError:
                print(f"Error: File not found at {daily_file_path} for {index_name}")
            except Exception as e:
                print(f"An error occurred while loading data for {index_name}: {e}")

# #print(tabulate(daily_data.head(), headers='keys', tablefmt='fancy_grid', showindex=False))