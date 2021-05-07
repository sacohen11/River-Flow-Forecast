import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import copy
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARMA

sys.path.append("/functions_clinton.py")
from functions_clinton import *

# import Toolbox

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#################################
# READ DATA
#################################









###############################################################
# CORRELATION MATRIX
###############################################################
data = pd.read_csv('/Users/luisabrigo/GWU/TimeSeries/Datasets/clinton.csv', header = 0)
corr_matrix(data)

data['Date'] = pd.to_datetime(data['Date'])
# del df['date']
data.set_index('Date', inplace=True)

print(data.info())

# y = data[["Flow"]]

# Missing values using mode
for col in data.columns:
    data[col] = data[col].fillna(data[col].mode()[0])


# FLOW_TMAX_tvalue = t_value(len(y), correlation_coefficient(y, data["TMAX"]), 1)
# FLOW_PRCP_tvalue = t_value(len(y), correlation_coefficient(y, data["PRCP"]), 1)
# TMIN_TMAX_tvalue = t_value(len(data["TMIN"]), correlation_coefficient(data["TMIN"], data["TMAX"]), 1)
#
# print("The critical t-value for Flow and TMAX is:", round(FLOW_TMAX_tvalue, 3))
# print("The critical t-value for Flow and PRCP is:", round(FLOW_PRCP_tvalue, 3))
# print("The critical t-value for TMIN and TMAX is:", round(TMIN_TMAX_tvalue, 3))
#

###############################################################
# PLOT DEPENDENT VARIABLE OVER TIME
###############################################################

# plt.figure(figsize=(8, 8))
# plt.plot(data.index, data["Flow"], label ="Flow")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Flow level")
# plt.title("Flow level over time Clinton, Iowa")
# plt.xticks(data.index, rotation='vertical')
# plt.locator_params(nbins=10)
# plt.tight_layout()
# plt.show()
# print(y.head())

###############################################################
# SPLIT DATA IN TRAIN AND TEST
###############################################################

for col in data.columns:
    data[col] = data[col].fillna(data[col].mode()[0])

data = data.resample('1W').mean()

# plt.figure(figsize=(8, 8))
# plt.plot(data.index, data["Flow"], label ="Flow")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Flow level")
# plt.title("Flow level over time Clinton, Iowa")
# plt.xticks(data.index, rotation='vertical')
# plt.locator_params(nbins=10)
# plt.tight_layout()
# plt.show()
# print(y.head())



for col in data.columns:
    data[col] = data[col].fillna(data[col].mode()[0])

X = data.drop(columns=['Flow'])
y = data["Flow"]

ACF_PACF(y, 300)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle= False, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(y_train)

# ACFPlot(autocorr(y_train, 100), 100, "Autocorrelation of Flow")
# ADF_Cal(y_train)

y_0 = y_train.values
y_1 = difference(y_0, 52)
# y_1 = difference(y_1, 1)
# y_1 = difference(y_1, 1)

y_1 = pd.Series(np.array(y_1))
# y_2 = difference(y_1, 1)
# y_3 = difference(y_2, 1)

# rolling_mean_var(np.array(y_4))
# # y_3 = difference(y_2,  416)
#
ACF_PACF(y_1, 300)
#
GPAC(autocorr(y_1, 100), 15, 15)


##############################################################
# TIME SERIES DECOMPOSITION
##############################################################

from statsmodels.tsa.seasonal import STL

flow2 = y_train
flow = pd.Series(np.array(y_train), index = pd.date_range("1900-01-01", periods=len(y_train), freq ='W'), name= 'Flow')

STL = STL(flow)
res = STL.fit()

fig = res.plot()
plt.ylabel('Year')
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

T = np.array(T)
S = np.array(S)
R = np.array(R)

Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])

print('The strength of trend for this data set is', Ft)
print('The strength of seasonality for this data set is ', Fs)

plt.plot(y_train.index,T, label= 'Trend')
plt.plot(y_train.index,R, label= 'Residual')
plt.plot(y_train.index,S, label= 'Seasonal')
plt.title('Trend, Residual, and Seasonal Plot')
plt.xlabel('Date')
# plt.xticks(y_train.index[::10], rotation= 90, fontsize= 10)
plt.ylabel('Flow')
plt.legend()
plt.tight_layout()
plt.show()


# Q6: seasonally adjusted data and plot of original
adjusted_seasonal = flow - S
detrended = flow - T

# replace outliers with mean
adjusted_seasonal = adjusted_seasonal.to_frame('Out')
mean = float(adjusted_seasonal.mean())
adjusted_seasonal['Out'] = np.where(adjusted_seasonal['Out'] > 100000 , mean, adjusted_seasonal['Out'])
adjusted_seasonal['Out'] = np.where(adjusted_seasonal['Out'] < -1000 , mean, adjusted_seasonal['Out'])


# replace outliers with mean
detrended = detrended.to_frame('Out')
mean = float(detrended.mean())
detrended['Out'] = np.where(detrended['Out'] > 100000 , mean, detrended['Out'])
detrended['Out'] = np.where(detrended['Out'] < -1000 , mean, detrended['Out'])




plt.figure()
plt.plot(flow, label= 'Original set')
plt.plot(adjusted_seasonal, label= 'Adjusted seasonality')
plt.xlabel('Date')
plt.ylabel('Flow')
plt.title('Original data versus Adjusted Seasonality')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(detrended, label= 'De-trended seasonal')
plt.plot(flow, label= 'Original set')
plt.xlabel('Date')
plt.ylabel('Flow')
plt.title('Original data versus Detrended')
plt.legend()
plt.tight_layout()
plt.show()


R = np.array(R)
S = np.array(adjusted_seasonal)
T = np.array(detrended)

Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])

print('The strength of trend for this data set is', Ft)
print('The strength of seasonality for this data set is ', Fs)


###############################################################
# PLOT ACF AND PACF OF DEPENDENT VARIABLE
###############################################################

ACFPlot(autocorr(y_train, 20), 20, "Autocorrelation of Flow")
ACF_PACF(y, 20)




##############################################################
# FEATURE SELECTION
##############################################################
# Adds a column called "const"
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())

Ypredict = model.predict(X_train)
Yforecast = model.predict(X_test)

plt.figure()
plt.plot(y_train.index, y_train, label="y")
plt.plot(X_test.index, y_test, label="Y_test")
plt.plot(X_test.index, Yforecast, label="Y_forecast")
plt.legend(loc='best')
plt.xlabel("Index")
plt.ylabel("Flow")
plt.title("Flow forecast using linear regression")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(y_train.index, y_train, label="y")
plt.plot(y_train.index, Ypredict, label="Y_predict")
plt.legend(loc='best')
plt.xlabel("Index")
plt.ylabel("Flow")
plt.title("Flow prediciton using linear regression")
plt.tight_layout()
plt.show()

# remove TMIN
x_opt = X_train[["PRCP", "SNOW", "TMAX"]]
x_opt = sm.add_constant(x_opt)
ols = sm.OLS(y_train, x_opt).fit()
print(ols.summary())

###############################################################
# MULTIPLE LINEAR REGRESSION
###############################################################

#Edit the test se
Xtest_new = X_test[["PRCP", "SNOW", "TMAX"]]
Xtest_new = sm.add_constant(Xtest_new)

Ypredict2 = ols.predict(x_opt)
Yforecast2 = ols.predict(Xtest_new)

plt.figure()
plt.plot(y_train.index, y_train, label="y")
plt.plot(X_test.index, y_test, label="Y_test")
plt.plot(X_test.index, Yforecast2, label="Y_forecast 2")
plt.legend(loc='best')
plt.xlabel("Index")
plt.ylabel("Flow")
plt.title("Flow forecast using linear regression (w/o TMIN")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(y_train.index, y_train, label="y")
plt.plot(y_train.index, Ypredict2, label="Y_predict 2")
plt.legend(loc='best')
plt.xlabel("Index")
plt.ylabel("Flow")
plt.title("Flow prediction using linear regression (w/o TMIN")
plt.tight_layout()
plt.show()



#Calculates prediciton and forecast error
p_error = Ypredict2 - y_train
print("This is the prediction error variance:", np.var(p_error))

# y_test2 = y_test.values
# y_forecast2 = Yforecast2.values
f_error = Yforecast2 - y_test
print("This is the forecast error variance:", np.var(f_error))


lags = 100

re = autocorr(p_error, lags)
ACFPlot(autocorr(p_error, lags), lags, "ACF of Residuals")
Q = len(y) * np.sum(np.square(re[lags:]))

print("Q value:", Q)


###############################################################
# GPAC
###############################################################

GPAC(autocorr(y_1.values, 100), 15, 15)

na = 1
nb = 1
#

###############################################################
# LEVENBERG MARQUARDT ALGORITHM
###############################################################


coeffs, variance, cov, SSE_history = levenberg_marquadt(y_1, na, nb)


###############################################################
# SARIMA (1,1,1)52
###############################################################


print(coeffs)
import statsmodels.api as sm
from scipy.stats import chi2

lags = 20


# coefs = [0.73618331, 0.41985157]
y_pred = prediction(y_train, coeffs, na, nb)
y_forecast = forecast(y_train, coeffs, na, nb, y_test)
#
#
ep = y_train - y_pred
e = y_test - y_forecast
#
re = autocorr(ep, lags)
ACFPlot(autocorr(ep, lags), lags, "ACF of Residuals")
Q = len(y)*np.sum(np.square(re[lags:]))
DOF = lags - na - nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("Q:", Q)
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white")

lbvalue, pvalue = sm.stats.acorr_ljungbox(ep, lags = [lags])
print(lbvalue)
print(pvalue)

def MSE(x):
    return np.sum(np.power(x, 2)) / len(x)


print("Variance of train set:", np.var(y_train.values))
print("Variance of prediction:", np.var(y_pred))
print("Ratio between variances:", np.var(y_train.values) / np.var(y_pred))
print("Mean Squared Error:", MSE(ep))

#
#
plt.figure(figsize=(15, 10))
plt.plot(y_train.index,y_train.values, label = "Train data")
plt.plot(y_train.index, y_pred, label = "Predicted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Sarima (1, 1, 1)52 Process")
plt.show()


re = autocorr(e, lags)
ACFPlot(autocorr(e, lags), lags, "ACF of Residuals")
Q = len(y)*np.sum(np.square(re[lags:]))
DOF = lags - na - nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)
print("Q:", Q)
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white")

lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags = [lags])
print(lbvalue)
print(pvalue)

def MSE(x):
    return np.sum(np.power(x, 2)) / len(x)


print("Variance of test set:", np.var(y_test.values))
print("Variance of forecast:", np.var(y_pred))
print("Ratio between variances:", np.var(y_test.values) / np.var(y_pred))
print("Mean Squared Error:", MSE(e))

#
#


plt.figure(figsize=(10, 8))
plt.plot(y_test.index, y_test.values, label = "Original test data")
plt.plot(y_test.index, y_forecast, label = "Forecast data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Forecast Clinton")
plt.show()


#############################################################
# BASE MODELS
#############################################################

forecasts = []
Yhat = []
Yhat_average = average_method(y_train.values)
forecast_average = average_method_test(y_train, y_test)
forecasts.append(forecast_average)
Yhat.append(Yhat_average)

Yhat_naive = naive_method(y_train.values)
forecast_naive = naive_method_test(y_train, y_test)
forecasts.append(forecast_naive)
Yhat.append(Yhat_naive)


Yhat_drift = drift_method(y_train.values, 1)
forecast_drift = drift_method_test(y_train, y_test, 1)
forecasts.append(forecast_drift)
Yhat.append(Yhat_drift)


Yhat_ses = ses_method(y_train.values, 0.25, y_train.values[0])
forecast_ses = ses_method_test(y_train.values, y_test.values, 0.25)
forecasts.append(forecast_ses)
Yhat.append(Yhat_ses)

base_models = ["Average", "Naive", "Drift", "SES"]

for i in range(len(base_models)):

    plt.figure()
    plt.plot(y_train.index, y_train, label = "Y_train")
    plt.plot(y_test.index, y_test, label = "Y_test")
    plt.plot(y_test.index, forecasts[i], label = "Y_forecast " + str(base_models[i]))
    plt.legend()
    plt.title(str(base_models[i]) + " | Train, test, forecast")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.show()

    num = len(Yhat[i])

    plt.figure()
    plt.plot(y_train.index, y_train, label="Y_train")
    # plt.plot(y_test.index, y_test, label="Y_test")
    plt.plot(y_train[:num].index, Yhat[i], label="Y_Hat " + str(base_models[i]))
    plt.legend()
    plt.title(str(base_models[i]) + " | Train, test, forecast")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.show()

    ## Using python, calculate the MSE of prediction errors and the forecast errors. Display the result on
    ## the console.

    print("####### " + str(base_models[i]) + " #######")

    # prediction_error = (Yhat[i] - y_train[1:].values).tolist()
    # print("Prediction Error: ", prediction_error)

    forecast_error = (forecasts[i] - y_test.values).tolist()
    # print("Forecast Error: ", forecast_error)

    def MSE(x):
        return np.sum(np.power(x,2))/len(x)

    # print("MSE Prediction Error: ", MSE(prediction_error))
    print("MSE Forecast Error: ", MSE(forecast_error))

    # print("Prediction Error Variance: ", np.var(prediction_error))
    print("Forecast Error Mean: ", np.mean(forecast_error))
    print("Forecast Error Variance: ", np.var(forecast_error))

    print("Value of Q:")

    print(sm.stats.acorr_ljungbox(forecast_error, lags=[5], boxpierce=True, return_df=True))

##############################################################
# HOLT LINEAR
##############################################################

a = ets.ExponentialSmoothing(y_train, trend="additive", damped_trend=True, seasonal="additive", seasonal_periods=52).fit(
    smoothing_level=0.25)
ses_prediction = a.forecast(steps=len(y_train))
ses_prediction = pd.Series(ses_prediction)

b = ets.ExponentialSmoothing(y_train, trend="additive", damped_trend=True, seasonal="additive", seasonal_periods=52).fit(
        smoothing_level=0.25)
ses_forecast = b.forecast(steps=len(y_test))
# ses_forecast = pd.Series(ses_forecast)


# ses_error = (ses_forecast - y_test.values).tolist()
# print("MSE Forecast Error: ", MSE(ses_error))

plt.plot(y_train.index, y_train, label="Y_train")
plt.plot(y_test.index, y_test, label="Y_test")
plt.plot(y_test.index, ses_forecast, label="Y_forecast")
plt.legend()
plt.title("Holt Linear | Train, test, forecast")
plt.xlabel("Time")
plt.ylabel("Values")
plt.tight_layout()
plt.show()


plt.plot(y_train.index, y_train, label="Y_train")
plt.plot(y_train.index, ses_prediction, label="Y_prediction")
plt.legend()
plt.title("Holt Linear | Train, test, forecast")
plt.xlabel("Time")
plt.ylabel("Values")
plt.tight_layout()
plt.show()


###############################################################
# HOLT WINTER
###############################################################

a = ets.ExponentialSmoothing(y_train, trend="additive", damped_trend=True, seasonal="additive", seasonal_periods=52).fit(
        smoothing_level=0.5)
winter_prediction = a.forecast(steps=len(y_train))
winter_prediction = pd.Series(winter_prediction)

b = ets.ExponentialSmoothing(y_train, trend="additive", damped_trend=True, seasonal="additive", seasonal_periods=52).fit(
        smoothing_level=0.5)
winter_forecast = b.forecast(steps=len(y_test))
# ses_forecast = pd.Series(ses_forecast)


# ses_error = (ses_forecast - y_test.values).tolist()
# print("MSE Forecast Error: ", MSE(ses_error))

plt.plot(y_train.index, y_train, label="Y_train")
plt.plot(y_test.index, y_test, label="Y_test")
plt.plot(y_test.index, winter_forecast, label="Y_forecast")
plt.legend()
plt.title("Holt Winter | Train, test, forecast")
plt.xlabel("Time")
plt.ylabel("Values")
plt.tight_layout()
plt.show()

plt.plot(y_train.index, y_train, label="Y_train")
plt.plot(y_train.index, winter_prediction, label="Y_prediction")
plt.legend()
plt.title("Holt Winter | Train, test, forecast")
plt.xlabel("Time")
plt.ylabel("Values")
plt.tight_layout()
plt.show()




print(y_train[0:5])



