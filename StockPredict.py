import pandas as pd
import numpy as np
import yfinance as yf
# import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

# Importing dataset
#data=yf.download('RELIANCE.NS', start='2000-1-1', end='2024-3-28').reset_index(drop=False)
data=yf.download('^NSEI', start='2000-1-1', end='2024-4-22').reset_index(drop=False)
selectedscript_0 = pd.DataFrame(data)

# Removing "Adj Close" columnfrom dataset
selectedscript_1=selectedscript_0.drop(["Adj Close"],axis=1).reset_index(drop=True)

# Finding duplicate columns, if any
selectedscript_1[selectedscript_1.duplicated()]

# Finding null values, if any
selectedscript_1.isnull().sum()

#To check which rows have any missing value under any column
selectedscript_1[selectedscript_1.isnull().any(axis=1)]

# Removing the row which have null value
selectedscript_2=selectedscript_1.dropna().reset_index(drop=True)

# Checking wether if there exist any null values
selectedscript_2[selectedscript_2.isnull().any(axis=1)]

# Making a copy of dataset as selectedscript
selectedscript=selectedscript_2.copy()

# converting the date column in to datetime 
selectedscript['Date']=pd.to_datetime(selectedscript['Date'],format='%Y-%m-%d')

# Setting the date column as index
selectedscript=selectedscript.set_index('Date')

## Model Building - Creating dataframe which only includes date and close time

close_df=pd.DataFrame(selectedscript['Close'])
close_df=close_df.reset_index()

### Normalizing / scaling close value between 0 to 1
close_stock = close_df.copy()
del close_df['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(close_df).reshape(-1,1))
print(closedf.shape)

### Split data for training and testing
#- Ratio for training and testing data is 90:10
training_size=int(len(closedf)*0.90)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

### Create new dataset according to requirement of time-series prediction
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 13
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

## Algorithms - LSTM - reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

### LSTM model structure
tf.keras.backend.clear_session()
model=Sequential()
model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=32,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

### R2 score for regression
#R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
#1 = Best - 0 or < 0 = worse
train_r2_lstm=r2_score(original_ytrain, train_predict)
test_r2_lstm=r2_score(original_ytest, test_predict)
print("Train data R2 score:", train_r2_lstm)
print("Test data R2 score:", test_r2_lstm)

### Comparision between original stock close price vs predicted close price
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

### Predicting next 30 days
x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=time_step
i=0
pred_days = 5
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
# print("Output of predicted next days: ", len(lst_output))

lstmdf=closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]
finaldf = pd.DataFrame({'lstm':lstmdf,})

data={"Model": ["LSTM"],"Train R2 Score": [train_r2_lstm],"Test R2 Score": [test_r2_lstm]}
df=pd.DataFrame(data)
print(df)
print(finaldf.to_string())
print(selectedscript.to_string()) 
