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
from PIL import Image
import PIL

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Оцінка вартості та прогноз ціни акцій фондового ринку на прикладі ринку цінних паперів США")

stocks = ("SPY", "QQQ", "JNJ", "NKE", "MCD", "AAPL", "AMZN", "NFLX", "TSLA", "GLD", "USO")
selected_stocks = st.selectbox("Оберіть компанію для оцінки та прогнозу : ", stocks)

my_width = 1100
my_height = 800


@st.cache(suppress_st_warning=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Отримуємо дані для цієї компанії ... ")
data = load_data(selected_stocks)
data_for_strategy = yf.download(selected_stocks, START, TODAY)
data_load_state.text("Дані отримано! ")

st.subheader('Необробленний датасет')
st.write(data)


@st.cache(suppress_st_warning=True)
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name='Stock Close'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    fig.update_layout(width=my_width, height=my_height)
    st.plotly_chart(fig, width=my_width, height=my_height)


st.subheader('Покажемо на графіку історію цін акцій компанії')
plot_raw_data()


@st.cache(suppress_st_warning=True)
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
st.subheader('Використаємо Модель1 для прогнозування ціни')
st.write("Модель1 має хороші результати на прогнозуванні коротких проміжків часу")
st.write('Модель1 дивиться на минулі 60 днів для того щоб зробити прогноз на 61 день. ')
st.write("Вивидемо результат в таблиці та на графіку")
st.write(bebra[0])
fig_tmp = go.Figure()
fig_tmp.add_trace(go.Scatter(x=bebra[0]['Close'].index, y=bebra[0]['Close'], name='Реальна ціна на данних для тесту ',
                             line=dict(color='green')))
fig_tmp.add_trace(go.Scatter(x=bebra[0]['Predictions'].index, y=bebra[0]['Predictions'], name='Прогнозована ціна',
                             line=dict(color='red')))
fig_tmp.add_trace(
    go.Scatter(x=bebra[1]['Close'].index, y=bebra[1]['Close'], name='Реальна ціна на данних для тренування',
               line=dict(color='blue')))
fig_tmp.layout.update(xaxis_rangeslider_visible=True)
fig_tmp.update_layout(width=my_width, height=my_height)
st.plotly_chart(fig_tmp, width=my_width, height=my_height)

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

# прогноз на роки

st.subheader('Зробити прогноз від 1 до 5 років використовуючи Модель2')
st.write("Модель2 має хороші результати на прогнозуванні довших проміжків часу")
st.write("Вивидемо результат в таблиці та на графіку")
n_years = st.slider("Кількість років для прогнозу:", 1, 5)
period = n_years * 365

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(forecast)

fig1 = plot_plotly(m, forecast, xlabel="Time", ylabel="Price")
fig1.update_layout(width=my_width, height=my_height)
st.plotly_chart(fig1, width=my_width, height=my_height)

# прогноз на дні
st.subheader('Зробити прогноз від 1 до 100 днів використовуючи Модель2')
n_days = st.slider("Кількість днів для прогнозу:", 1, 100, value=20)
period = n_days

future_days = m.make_future_dataframe(periods=period)
forecast_days = m.predict(future_days)

st.write(forecast_days)

fig2 = plot_plotly(m, forecast_days, xlabel="Time", ylabel="Price")
fig2.update_layout(width=my_width, height=my_height)
st.plotly_chart(fig2, width=my_width, height=my_height)

# st.write('forecast data')
# fig2 = m.plot_components(forecast)
# st.write(fig2)

st.subheader("Зробити прогноз на конкретний день")
d = st.date_input(
    "Оберіть дату для прогнозу(тільки Пн-Пт)",
    datetime.date.today())
st.write('Результати:')
st.write('Model1 -   Low-High = ', make_prediction(selected_stocks, "Low", True, d))

forecast_days_new = m.predict(m.make_future_dataframe(periods=2000))
forecast_days_new['ds'] = pd.to_datetime(forecast_days_new['ds']).apply(lambda x: x.date())
st.write('Model2 -   Low-High = ', ((forecast_days_new.loc[forecast_days_new['ds'] == d])['yhat_lower'].item()), " - ",
         ((forecast_days_new.loc[forecast_days_new['ds'] == d])['yhat_upper'].item()))


def strategy_test(company_ticker, strategy):
    df = data_for_strategy
    training_data_len = math.ceil(len(df) * 0.7)  # put 70% of our data to train and 30 % to test with
    all_data = df[training_data_len:]
    # df = data_for_strategy
    # #training_data_len = math.ceil(len(df) * 0.7)  # put 70% of our data to train and 30 % to test with
    # all_data = df[:]
    res_close = make_prediction(company_ticker, 'Close', False)
    res_open = make_prediction(company_ticker, 'Open', False)

    delta_list = list()
    trade_dates = list()
    delta = 0.0
    for i in range(len(res_open[0])):

        if strategy == 1:
            # купили опен продали клоуз , хорошо работает на стаках которые постоянно растут на дистанции
            delta = ((res_close[0]['Close'][i] - res_open[0]['Open'][i]) / res_open[0]['Open'][i]) * 100
            delta_list.append(delta)
            trade_dates.append(res_close[0].index[i])

        if strategy == 2:
            # купили опен продали клоуз ,
            # пропусили день если от клоуза до лоя меньше процента ( закрытие в лой )
            if i > 1:
                if (((res_close[0]['Close'][i - 1] - all_data['Low'][i - 1]) / all_data['Low'][i - 1]) * 100) < 0.75:
                    delta += ((res_close[0]['Close'][i] - res_open[0]['Open'][i]) / res_open[0]['Open'][i]) * 100

        if strategy == 3:
            # минус процент - берем некст день
            if i >= 1:
                if (((res_open[0]['Open'][i - 1] - res_close[0]['Close'][i - 1]) /
                     res_open[0]['Open'][i - 1]) * 100) > -1:
                    delta += ((res_close[0]['Close'][i] - res_open[0]['Open'][i]) / res_open[0]['Open'][i]) * 100

        if strategy == 4:
            # move -5 % за день или два купи и продай 20 дней потом
            if i >= 2:
                if (((all_data['Close'][i - 1] - all_data['Open'][i - 1]) / all_data['Open'][i - 1]) * 100 > 5) or (
                        ((all_data['Close'][i - 1] - all_data['Open'][i - 2]) / all_data['Open'][i - 2]) * 100 > 5):
                    if i + 20 > len(res_open[0]):
                        delta += ((res_close[0]['Close'][-1] - res_open[0]['Open'][i]) /
                                  res_open[0]['Open'][i]) * 100
                        trade_dates.append(res_close[0].index[-1])
                    else:
                        delta += ((res_close[0]['Close'][i + 19] - res_open[0]['Open'][i]) /
                                  res_open[0]['Open'][i]) * 100
                        trade_dates.append(res_close[0].index[i + 19])
                    delta_list.append(delta)

        if strategy == 5:
            # gap down > 3 % buy and sell 20 days later
            if i >= 1:
                if (((all_data['Close'][i - 1] - all_data['Open'][i]) / all_data['Close'][i - 1]) * 100) > 3:
                    if i + 20 > len(res_open[0]):
                        delta += ((res_close[0]['Close'][-1] - res_open[0]['Open'][i]) /
                                  res_open[0]['Open'][i]) * 100
                        trade_dates.append(res_close[0].index[-1])
                    else:
                        delta += ((res_close[0]['Close'][i + 19] - res_open[0]['Open'][i]) /
                                  res_open[0]['Open'][i]) * 100
                        trade_dates.append(res_close[0].index[i + 19])
                    delta_list.append(delta)

    # st.write(delta_list)
    # st.write(trade_dates)
    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Scatter(x=trade_dates, y=delta_list, name='Графік прибутку стратегії ',
                   line=dict(color='green')))
    fig_pnl.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name='Реальні дані',
                                 line=dict(color='blue')))
    # fig_pnl.add_trace(
    #     go.Scatter(x=bebra[1]['Close'].index, y=bebra[1]['Close'], name='Реальна ціна на данних для тренування',
    #                line=dict(color='blue')))
    fig_pnl.layout.update(xaxis_rangeslider_visible=True)
    fig_pnl.update_layout(width=my_width, height=my_height)
    st.plotly_chart(fig_pnl, width=my_width, height=my_height)
    return delta


st.subheader('Інформація по всім стратегіям :')
st.write('Нижче наведено теплову мапу по всім стратегіям та опис кожної стратегії:')

image = Image.open('Figure_1.png')

st.image(image, width=1200, caption='Теплова мапа в період з Березня 2020 по Травень 2022')

st.write('сделать тут выпадающее меню для каждой стратегии ')
genre = st.radio(
    "Отримати опис стратегії",
    ('1', '2', '3', '4', '5'))

st.write("тут буде опис стратегії")
if genre == '1':
    st.write('1')
elif genre == '2':
    st.write("2")
elif genre == '3':
    st.write("3.")
elif genre == '4':
    st.write("4.")
elif genre == '5':
    st.write("5")


st.subheader("Обрати стратегію інвестування :")
st.write('про кожну стратегію ви можете прочитати нижче')
strategys = (1, 2, 3, 4, 5,)
selected_strategy = st.selectbox("Оберіть стратегію : ", strategys)

data_load_state2 = st.text("Обробляємо дані для цієї стратегії ... ")
st.write('Результат стратегії : ', format(strategy_test(selected_stocks, selected_strategy), '.2f'), '%')
data_load_state2.text("Дані Оброблено! ")

