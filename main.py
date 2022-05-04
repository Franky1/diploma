import datetime

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
import mplfinance as mpl
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock pred app")

stocks = ("SPY", "QQQ", "JNJ", "NKE", "MCD", "AAPL", "AMZN", "NFLX", "TSLA", "GLD", "USO")
selected_stocks = st.selectbox("Select company for prediction ", stocks)

my_width = 900
my_height = 750


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data ... ")
data = load_data(selected_stocks)
data_load_state.text("Loading data... done! ")

st.subheader('Raw data')
st.write(data)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name='Stock Close'))
    fig.layout.update(title_text="time series data", xaxis_rangeslider_visible=True)
    fig.update_layout(width=my_width, height=my_height)
    st.plotly_chart(fig, width=my_width, height=my_height)


plot_raw_data()


def make_prediction(ticker, parametr, for_one_day, date_for_one_day=TODAY):
    # get the stock
    df = web.DataReader(ticker, data_source='yahoo', start=START, end=TODAY)
    # df = data
    # create new dataframe with only the 'close' column
    data_for_pred = df.filter([parametr])
    dataset = data_for_pred.values
    training_data_len = math.ceil(len(dataset) * 0.7)  # put 70% of our data to train and 30 % to test with

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    days_to_train = 60

    # Build the LSTM model
    model = load_model('spy_closes_5epochs.h5')
    if not for_one_day:
        # create test dataset
        test_data = scaled_data[training_data_len - days_to_train:, :]
        x_test = []
        for i in range(days_to_train, len(test_data)):
            x_test.append(test_data[i - days_to_train:i, 0])

        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        train = data_for_pred[:training_data_len]
        valid = data_for_pred[training_data_len:]
        valid['Predictions'] = predictions

        return valid, train  # returns data: Date(index), Close, Predictions
    else:
        if datetime.datetime.strptime(TODAY, "%Y-%m-%d").date() < date_for_one_day:
            # st.write(datetime.datetime.strptime(TODAY, "%Y-%m-%d").date())
            # st.write(date_for_one_day)
            # st.write(datetime.datetime.strptime(TODAY, "%Y-%m-%d").date() < date_for_one_day)
            return "Модель1 не вміє робити прогнози більше ніж на 1 день у майбутнє\nЯкщо ви обрали завтрашню дату і " \
                   "отримали це повідомлення , то проблема в тому, що ринки ще відкриті і не має данних за сьогодні! " \
                   "Також через це ви не побачили реальних данних на цю дату "
        else:
            quote = web.DataReader(ticker, data_source='yahoo', start=START, end=date_for_one_day)
            parametr1 = "Low"
            parametr2 = "High"
            new_df = quote.filter([parametr])
            new_df2 = quote.filter([parametr2])
            if datetime.datetime.strptime(TODAY, "%Y-%m-%d").date() != date_for_one_day:
                st.write("Real Data", "  Low-High = ", float(new_df[parametr1][-1]), ' - ',
                         float(new_df2[parametr2][-1]))
            last_60_days = new_df[-60:].values
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)
            last_60_days2 = new_df2[-60:].values
            last_60_days_scaled2 = scaler.transform(last_60_days2)
            X_test2 = []
            X_test2.append(last_60_days_scaled2)
            X_test2 = np.array(X_test2)
            X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))
            pred_price2 = model.predict(X_test2)
            pred_price2 = scaler.inverse_transform(pred_price2)
            return " " + str(float(pred_price)) + " - " + str(float(pred_price2))


bebra = (make_prediction(selected_stocks, 'Close', False))
st.write(bebra[0])
fig_tmp = go.Figure()
fig_tmp.add_trace(go.Scatter(x=bebra[0]['Close'].index, y=bebra[0]['Close'], name='Actual Close on Test data',
                             line=dict(color='green')))
fig_tmp.add_trace(go.Scatter(x=bebra[0]['Predictions'].index, y=bebra[0]['Predictions'], name='Predicted Close',
                             line=dict(color='red')))
fig_tmp.add_trace(go.Scatter(x=bebra[1]['Close'].index, y=bebra[1]['Close'], name='Actual Close on Train data',
                             line=dict(color='blue')))
fig_tmp.layout.update(title_text="time series data", xaxis_rangeslider_visible=True)
fig_tmp.update_layout(width=my_width, height=my_height)
st.plotly_chart(fig_tmp, width=my_width, height=my_height)
# st.pyplot(bebra)

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

# прогноз на роки
st.subheader('Зробити прогноз від 1 до 5 років')
n_years = st.slider("роки для прогнозу:", 1, 5)
period = n_years * 365

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast)

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# прогноз на дні
st.subheader('Зробити прогноз від 1 до 100 днів')
n_days = st.slider("дні для прогнозу:", 1, 100)
period = n_days

future_days = m.make_future_dataframe(periods=period)
forecast_days = m.predict(future_days)

st.subheader("Forecast data")
st.write(forecast_days)

st.write('forecast data')
fig2 = plot_plotly(m, forecast_days)
st.plotly_chart(fig2)

# st.write('forecast data')
# fig2 = m.plot_components(forecast)
# st.write(fig2)

st.subheader("Зробити прогноз на конкретний день")
d = st.date_input(
    "Оберіть дату для прогнозу(тільки Пн-Пт)",
    datetime.date.today())
st.write('Results:')
st.write('Model1 -   Low-High = ', make_prediction(selected_stocks, "Low", True, d))

forecast_days_new = m.predict(m.make_future_dataframe(periods=2000))
forecast_days_new['ds'] = pd.to_datetime(forecast_days_new['ds']).apply(lambda x: x.date())
st.write('Model2 -   Low-High = ', ((forecast_days_new.loc[forecast_days_new['ds'] == d])['yhat_lower'].item()), " - ",
         ((forecast_days_new.loc[forecast_days_new['ds'] == d])['yhat_upper'].item()))


