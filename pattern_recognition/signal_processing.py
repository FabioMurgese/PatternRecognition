#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def autoregressive_analysis(data):
    """
    Fit an autoregressive model on the first 3 months of data and estimate performance
    on the remaining 1.5 months.
    """
    # from statsmodels.graphics.tsaplots import plot_pacf
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.arima_model import ARIMA

    appliance_series = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 1])
    # print(appliance_series.describe())
    plt.plot(appliance_series)
    plt.show()
    # plot_pacf(appliance_series["Appliances"])
    train = appliance_series.loc['2016-01-01':'2016-03-31']
    test = appliance_series.loc['2016-04-01':]
    p, q = 1, 8
    arma_model = ARIMA(train, order=(p, 0, q))
    arma_model_fit = arma_model.fit()
    pred = arma_model_fit.predict(start='2016-04-01 00:00:00', end='2016-05-27 18:00:00')
    print('MSE:', mean_squared_error(test, pred))
    plt.plot(test)
    plt.plot(pred)
    plt.show()


def correlation_temperatures(data):
    """
    Correlation analysis on the temperature data.
    """
    t1_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 3])
    t2_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 5])
    t3_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 7])
    t4_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 9])
    t5_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 11])
    t6_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 13])
    t7_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 15])
    t8_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 17])
    t9_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 19])
    t10_temp = pd.read_csv(data, header=0, index_col=0, parse_dates=True, usecols=[0, 21])

    pd.plotting.autocorrelation_plot(t1_temp)
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t2_temp.values.flatten(), mode='full'), label='t1 vs t2')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t3_temp.values.flatten(), mode='full'), label='t1 vs t3')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t4_temp.values.flatten(), mode='full'), label='t1 vs t4')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t5_temp.values.flatten(), mode='full'), label='t1 vs t5')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t6_temp.values.flatten(), mode='full'), label='t1 vs t6')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t7_temp.values.flatten(), mode='full'), label='t1 vs t7')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t8_temp.values.flatten(), mode='full'), label='t1 vs t8')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t9_temp.values.flatten(), mode='full'), label='t1 vs t9')
    plt.plot(np.correlate(t1_temp.values.flatten(),
                          t10_temp.values.flatten(), mode='full'), label='t1 vs t10')
    plt.legend(loc='upper right')
    plt.show()
