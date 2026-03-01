import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objs as go
import datetime

# Set the background image using HTML and CSS

today = datetime.date.today()
start = '2010-01-01'
end = today.strftime('%Y-%m-%d')

st.title("Predictive Analysis of Stock Market Trends:           ")

user_input=st.text_input("Enter the Stock Tickter","AAPL", key="stock_symbol")
df = yf.download(user_input, start=start, end=end)

st.subheader("Data from 2010 to 2023")
data1 = pd.DataFrame(df)

st.dataframe(data1.tail(15))


#Closing price with month and year using moving average
fig_all = plt.figure(figsize=(18, 12))
plt.plot(df.Close)

ma30 = df.Close.rolling(30).mean()
fig_ma30 = plt.figure(figsize=(12, 6))
plt.plot(ma30)

ma365 = df.Close.rolling(365).mean()
fig_ma365 = plt.figure(figsize=(12, 6))
plt.plot(ma365)

button_container = st.columns(3)

# Create buttons for each plot
if button_container[0].button("Show Closing Prices"):
    st.subheader("Closing Price VS Time Chart")
    st.pyplot(fig_all)

if button_container[1].button("Show MA30"):
    st.subheader("Closing Price of 30 days")
    st.pyplot(fig_ma30)

if button_container[2].button("Show MA365"):
    st.subheader("Closing Price of 365 days")
    st.pyplot(fig_ma365)
#load my model
model=load_model("./keras_model.h5")


#Splitting Data into Training and Testing using MinMaxScaler

data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)
data_testing_array= scaler.fit_transform(data_testing)


past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test , y_test =np.array(x_test) , np.array(y_test);

#predication making
test_predication = model.predict(x_test)
scaler.scale_

scaling_factor = 1/scaler.scale_[0]
y_test = y_test*scaling_factor
test_predication=test_predication*scaling_factor

st.subheader("predication vs orignal")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(test_predication,'r', label ='Preditemp_inputcted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# demonstrate prediction for next 50 days  with closing price
x_input=data_testing_array[len(data_testing_array) - 100:].reshape(1,-1)
opening=list(x_input)
opening=opening[0].tolist()

from numpy import array
lst_output=[]
n_steps=100
i=0
while(i<50):

    if(len(opening)>100):
        #print(opening)
        x_input=np.array(opening[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        opening.extend(yhat[0].tolist())
        opening=opening[1:]
        #print(opening)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        opening.extend(yhat[0].tolist()) 
        lst_output.extend(yhat.tolist())
        i=i+1

lst_output=scaler.inverse_transform(lst_output)
arr = lst_output



#opening
#Splitting Data into Training and Testing

data_training_open = pd.DataFrame(df['Open'][0: int(len(df)*0.70)])
data_testing_open= pd.DataFrame(df['Open'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler_open = MinMaxScaler(feature_range=(0,1))

data_testing_array_open= scaler_open.fit_transform(data_testing_open)
x_input_open=data_testing_array_open[len(data_testing_array_open) - 100:].reshape(1,-1)
temp_input_open=list(x_input_open)
temp_input_open=temp_input_open[0].tolist()

from numpy import array

lst_output_open=[]
n_steps=100
i=0
while(i<50):

    if(len(temp_input_open)>100):
        #print(temp_input)
        x_input_open=np.array(temp_input_open[1:])
        #print("{} day input {}".format(i,x_input_open))
        x_input_open=x_input_open.reshape(1,-1)
        x_input_open = x_input_open.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_open = model.predict(x_input_open, verbose=0)
        
        temp_input_open.extend(yhat_open[0].tolist())
        temp_input_open=temp_input_open[1:]
        #print(temp_input)
        lst_output_open.extend(yhat_open.tolist())
        i=i+1
    else:
        x_input_open = x_input_open.reshape((1,n_steps,1))
        yhat_open = model.predict(x_input_open, verbose=0)
        temp_input_open.extend(yhat_open[0].tolist())
        lst_output_open.extend(yhat_open.tolist())
        i=i+1

lst_output_open=scaler_open.inverse_transform(lst_output_open)
arr_open = lst_output_open

#High
#Splitting Data into Training and Testing

data_training_high = pd.DataFrame(df['High'][0: int(len(df)*0.70)])
data_testing_high= pd.DataFrame(df['High'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler_high = MinMaxScaler(feature_range=(0,1))

data_testing_array_high= scaler_high.fit_transform(data_testing_high)
x_input_high=data_testing_array_high[len(data_testing_array_high) - 100:].reshape(1,-1)
temp_input_high=list(x_input_high)
temp_input_high=temp_input_high[0].tolist()

from numpy import array

lst_output_high=[]
n_steps=100
i=0
while(i<50):

    if(len(temp_input_high)>100):
        #print(temp_input)
        x_input_high=np.array(temp_input_high[1:])
        #print("{} day input {}".format(i,x_input_high))
        x_input_high=x_input_high.reshape(1,-1)
        x_input_high = x_input_high.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_high = model.predict(x_input_high, verbose=0)
        
        temp_input_high.extend(yhat_high[0].tolist())
        temp_input_high=temp_input_high[1:]
        #print(temp_input)
        lst_output_high.extend(yhat_high.tolist())
        i=i+1
    else:
        x_input_high = x_input_high.reshape((1,n_steps,1))
        yhat_high = model.predict(x_input_high, verbose=0)
        temp_input_high.extend(yhat_high[0].tolist())
        lst_output_high.extend(yhat_high.tolist())
        i=i+1

lst_output_high=scaler_high.inverse_transform(lst_output_high)
arr_high = lst_output_high

#low
#Splitting Data into Training and Testing

data_training_Low = pd.DataFrame(df['Low'][0: int(len(df)*0.70)])
data_testing_Low= pd.DataFrame(df['Low'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler_Low = MinMaxScaler(feature_range=(0,1))

data_testing_array_Low= scaler_Low.fit_transform(data_testing_Low)
x_input_Low=data_testing_array_Low[len(data_testing_array_Low) - 100:].reshape(1,-1)
temp_input_Low=list(x_input_Low)
temp_input_Low=temp_input_Low[0].tolist()

from numpy import array

lst_output_Low=[]
n_steps=100
i=0
while(i<50):

    if(len(temp_input_Low)>100):
        #print(temp_input)
        x_input_Low=np.array(temp_input_Low[1:])
        #print("{} day input {}".format(i,x_input_Low))
        x_input_Low=x_input_Low.reshape(1,-1)
        x_input_Low = x_input_Low.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_Low = model.predict(x_input_Low, verbose=0)
        
        temp_input_Low.extend(yhat_Low[0].tolist())
        temp_input_Low=temp_input_Low[1:]
        #print(temp_input)
        lst_output_Low.extend(yhat_Low.tolist())
        i=i+1
    else:
        x_input_Low = x_input_Low.reshape((1,n_steps,1))
        yhat_Low = model.predict(x_input_Low, verbose=0)
        temp_input_Low.extend(yhat_Low[0].tolist())
        lst_output_Low.extend(yhat_Low.tolist())
        i=i+1

lst_output_Low=scaler_Low.inverse_transform(lst_output_Low)
arr_Low = lst_output_Low

h = pd.DataFrame(arr_Low, columns=['Low'])

g = pd.DataFrame(arr_high, columns=['High'])

f = pd.DataFrame(arr_open, columns=['Open'])

k =pd.DataFrame(arr, columns=['Close'])



k['DailyChange'] = k['Close'].diff()
k['PercentageChange'] = (k['DailyChange'] / k['Close'].shift(1)) * 100
k['PercentageChange'] = k['PercentageChange'].map("{:.2f}%".format)
result_df = pd.concat([h, g, f, k], axis=1)

fig = go.Figure(data=[go.Candlestick(x=result_df.index,
                open=result_df['Open'],
                high=result_df['High'],
                low=result_df['Low'],
                close=result_df['Close'])])

st.plotly_chart(fig)

# Display the styled DataFrame using Streamlit
st.write(result_df)


fig3= plt.figure(figsize=(12,6))
plt.plot(k.Close)
st.pyplot(fig3)


st.title("Next 5 Years Returns")

# Input for the stock symbol
symbol = user_input

# Check for changes in the input field
if st.session_state.stock_symbol is not None:
    try:
        # Download historical data
        stock_data = yf.download(symbol, start="2023-01-01", end="2028-01-01")

        # Calculate returns over the next 5 years
        returns_next_5_years = (stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0] - 1) * 100

        # Display the returns
        st.write(f"{symbol} returns over the next 5 years: {returns_next_5_years:.2f}%")
        st.write("Recommendation Moodel")
        if returns_next_5_years >= 0:
            st.markdown('<div style="padding: 10px; color: white; background-color: green; text-align: center;">Yes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding: 10px; color: white; background-color: red; text-align: center;">No</div>', unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error: {str(e)}")

    

