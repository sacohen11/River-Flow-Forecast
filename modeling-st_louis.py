#################################################################
# Title: Final Report
# Author: Sam Cohen
# Date Created: 4/25
# Date Modified:
##################################################################

# Packages
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
from scipy.stats import chi2
import functions_st_louis
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from sklearn.decomposition import PCA


# Formats
years_15 = mdates.YearLocator(15)   # every 15 years
years_1 = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
month_fmt = mdates.DateFormatter('%m')
time_delta_large = datetime.timedelta(days = 700)
time_delta_small = datetime.timedelta(days = 7)

# Read in data and perform simple conversions
df = pd.read_csv("./data/st_louis_preprocessed.csv", header=0, index_col=0, infer_datetime_format=True)
#df = df.rename(columns={"Unnamed: 0":"Index"})
#df.reset_index(inplace=True)
#df.rename(columns={"index:":"Date"})

df["revised_date"] = df["Date"].str.slice(start=-5)
df = df[df["revised_date"] != "02-29"]
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.set_index(["Date"])
df = df.resample('1W').mean()
date = df.index.to_series()
#date = [datetime.datetime.strptime(w, "%Y-%m-%d").date() for d in df.index.to_series().to_string()]
flow = df['Flow']

#===================================================================================
# Description of the Dataset
# a. Plot of the dependent variable versus time.
# b. ACF/PACF of the dependent variable.
# c. Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient.
# d. Preprocessing procedures (if applicable): Clean the dataset (no missing data or NAN)
# e. Split the dataset into train set (80%) and test set (20%).
#===================================================================================

# Plot Flow for all years
fig, ax = plt.subplots()
ax.set_title("Historical Discharge at the St. Louis - Mississippi River USGS Gage")
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_major_locator(years_15)
ax.plot(date, flow, label="River Flow")
ax.set_ylabel("River Flow (ft3/s)")
ax.set_xlabel("Date")
ax.set_xlim(min(date) - time_delta_large, max(date) + time_delta_large)
ax.legend()
fig.autofmt_xdate()
plt.show()

# Plot Flow for the past 15 years
# 5,340 days ~= 15 years
fig, ax = plt.subplots()
ax.set_title("2019-2021 Discharge at the St. Louis - Mississippi River USGS Gage")
#ax.xaxis.set_major_formatter(years_fmt)
#ax.xaxis.set_major_locator(years_1)
ax.xaxis.set_major_formatter(month_fmt)
ax.xaxis.set_major_locator(months)
ax.plot(date[-104:], flow[-104:], label="River Flow")
ax.set_ylabel("River Flow (ft3/s)")
ax.set_xlabel("Date")
ax.set_xlim(min(date[-104:]) - time_delta_small, max(date[-104:]) + time_delta_small)
ax.legend()
fig.autofmt_xdate()
plt.show()

# df = df[:][-3650:]
# df = df.reset_index()
# ACF and PACF Plots of Flow
lags = 300
acf, pacf = functions_st_louis.acf_pacf_plot(df["Flow"], lags)

# Correlation Matrix
functions_st_louis.correlation_matrix(df)
#functions_st_louis.correlation_matrix(df.drop("Index", 1))

# Pre-processing
# Any misisng values?
print("Number of data points:", len(df))
print()
print("Number of NA values by variables:\n",df.isnull().sum())
print()
print("Decsriptive statistics of the variables:\n", df.describe())
print()

# Very few missing values. We will fill the SNOW and PRCP variable with mode imputation (as 0 is certainly most
# common) and the other missing values (TMIN) with mean imputation
df["PRCP"] = df["PRCP"].fillna(df["PRCP"].mode()[0])
df["SNOW"] = df["SNOW"].fillna(df["SNOW"].mode()[0])

df["TMIN"] = df["TMIN"].fillna(df["TMIN"].mean())

# Verify everything has been filled
print("Number of NA values by variables:\n",df.isnull().sum())
print()

df = sm.add_constant(df)
# Train - Test - Split
x_train, x_test, y_train, y_test = train_test_split(df[["PRCP", "SNOW", "TMAX", "TMIN"]], df["Flow"], shuffle=False, train_size=.8, test_size=.2)

#===================================================================================
# Stationarity
# Check for a need to make the dependent variable stationary. If the dependent
# variable is not stationary, you need to use the techniques discussed in class to make it stationary.
# Perform ACF/PACF analysis for stationarity. You may also perform ADF-test.
#===================================================================================

# Checks for stationarity include ACF and PACF plots, rolling mean and variances, and ADF test
# Rolling mean and variances
functions_st_louis.rolling_mean_var(df["Flow"])
#
# ADF test
# functions_st_louis.ADF_Cal(df["Flow"])
#
# # Make the dataset stationary
# # Log transform
#flow_log = np.log(flow[-1040:-156])
#
# # First differencing
#flow1 = functions_st_louis.difference(flow_log, 1)
flow3 = functions_st_louis.difference(y_train.values, 52)
#flow3 = functions_st_louis.difference(flow1, 1)
#flow3 = functions_st_louis.difference(flow2, 1)

flow3 = pd.Series(np.array(flow3))

acf, pacf = functions_st_louis.acf_pacf_plot(flow3, lags)

functions_st_louis.rolling_mean_var(flow3)

# ADF test
functions_st_louis.ADF_Cal(flow3)

# GPAC
#functions_st_louis.calc_gpac(14, 14, acf)

# This dataset looks stationary. It is hard to tell if I've got the right seasonality and there are probably a few
# different seasonalities, but here is an attempt. The ADF test, rolling mean and average, and ACF/PACF plots look
# good. The ACF plot implies a AR model of order 2, although I don't see that in the GPAC table


#===================================================================================
# Time Series Decomposition
# Approximate the trend and the seasonality and plot the detrended
# and the seasonally adjusted data set. Find the out the strength of the trend and seasonality. Refer
# to the lecture notes for different type of time series decomposition techniques.
#===================================================================================

functions_st_louis.plot_strength_seasonality_trend(df["Flow"])

# #===================================================================================
# # Holt-Winters Method
# # Using the Holt-Winters method try to find the best fit using the train
# # dataset and make a prediction using the test set.
# #===================================================================================
#
errors = functions_st_louis.holt_winters(y_train, y_test)
print("MSE of Holt-Winters:", np.mean(np.array(errors)**2))

# #===================================================================================
# # Feature Seleciton
# # You need to have a section in your report that explains how the feature
# # selection was performed and whether the collinearity exits not not. Backward stepwise regression
# # along with SVD and condition number is needed. You must explain that which feature(s) need to
# # be eliminated and why. You are welcome to use other methods like PCA or random forest for
# # feature elimination.
# #===================================================================================
df = df.drop("index", 1)
functions_st_louis.colinearity_detection(df)

functions_st_louis.condition_number(x_train.values)

coeffs = functions_st_louis.regression_coeffs(x_train[["PRCP", "SNOW", "TMAX", "TMIN"]], y_train)

model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# drop snow
x_train1 = x_train.drop("SNOW", 1)
functions_st_louis.colinearity_detection(x_train1)

functions_st_louis.condition_number(x_train1.values)

coeffs = functions_st_louis.regression_coeffs(x_train1[["PRCP", "TMAX", "TMIN"]], y_train)

model = sm.OLS(y_train, x_train1).fit()
print(model.summary())

# PCA 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print("Explained Variance of Each Feature:", pca.explained_variance_ratio_)
print("Total Explained Variance:", pca.explained_variance_ratio_.sum())
functions_st_louis.colinearity_detection(principalDf)

functions_st_louis.condition_number(principalDf.values)

coeffs = functions_st_louis.regression_coeffs(principalDf[["principal component 1", 'principal component 2']], y_train)

model = sm.OLS(list(y_train), principalDf[["principal component 1", 'principal component 2']]).fit()
print(model.summary())


# PCA 1 component
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x_train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1'])

print("Explained Variance of Each Feature:", pca.explained_variance_ratio_)
print("Total Explained Variance:", pca.explained_variance_ratio_.sum())
functions_st_louis.colinearity_detection(principalDf)

functions_st_louis.condition_number(principalDf.values)

coeffs = functions_st_louis.regression_coeffs(principalDf[["principal component 1"]], y_train)

model = sm.OLS(list(y_train), principalDf[["principal component 1"]]).fit()
print(model.summary())


# #===================================================================================
# # Multiple Linear Regression
# # a. You need to include the complete regression analysis into your report. Perform one-step
# #    ahead prediction and compare the performance versus the test set.
# # b. Hypothesis tests analysis: F-test, t-test.
# # c. AIC, BIC, RMSE, R-squared and Adjusted R-squared
# # d. ACF of residuals.
# # e. Q-value
# # f. Variance and mean of the residuals.
# #===================================================================================
model = sm.OLS(y_train, x_train1).fit()
print(model.summary())

x_test1 = x_test.drop("SNOW", 1)

# one step
predictions = model.predict(x_train1.values)
fig, ax = plt.subplots()
ax.plot(range(0, len(y_train)), y_train,label= "Train Data")
ax.plot(range(0, len(y_train)), predictions,label= "Predictions")

plt.legend(loc='upper left')
plt.title('Reduced Feature Space Linear Regression Model One Step Pred')
plt.xlabel('Time')
plt.ylabel('Flow')
plt.show()

# predictions
predictions = model.predict(x_test1.values)
fig, ax = plt.subplots()
ax.plot(range(0, len(y_train)), y_train,label= "Train Data")
ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_test,label= "Test Data")
ax.plot(range(len(y_train), len(y_train)+len(y_test)), predictions,label= "Predictions")

plt.legend(loc='upper left')
plt.title('Reduced Feature Space Linear Regression Model Train/Test vs. Pred')
plt.xlabel('Time')
plt.ylabel('Flow')
plt.show()

# ACF of residuals
pred_errors = np.subtract(y_train, model.predict(x_train1.values))
functions_st_louis.plot_autocorr(pred_errors, lags=20, title="ACF for residuals errors")

# Q-Value
q = functions_st_louis.q_value(pred_errors, lags=250)

# Var and mean of residuals
print("Mean of the Residuals:", np.mean(pred_errors))
print("Variance of the Residuals:", functions_st_louis.error_var(pred_errors))

# MSE of final predictions
error = y_test - predictions
print("MSE of linear regression:", np.mean(np.square(np.array(error))))

# #===================================================================================
# # ARMA, ARIMA, and SARIMA Model
# # Develop an ARMA, ARIMA and SARIMA model that represent the dataset.
# # a. Preliminary model development procedures and results. (ARMA model order
# # determination). Pick at least two orders using GPAC table.
# # b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of
# # the autocorrelation function and the GPAC table within this section).
# # c. Include the GPAC table in your report and highlight the estimated order.
# #===================================================================================
#
functions_st_louis.calc_gpac(10, 10, acf)
#
#
# #===================================================================================
# # Levenberg - Marquardt Algorithm
# # Display the parameter estimates, the standard deviation of the parameter estimates
# # and confidence intervals.
# #===================================================================================

# Based on the ACF and PACF, the order is ARMA(3,0,0) X ARIMA(0,0,1)365

ar_order = 1
ma_order = 1
coeffs, sigma_squared, cov, sse_array = functions_st_louis.levenberg_marquadt(flow3, ar_order, ma_order)
# coeffs = SARIMAX(y_train, order=(0,0,1), seasonal_order=(0,0,1,365)).fit()
print(coeffs)
#
# # Confidence intervals
uppers = []
lowers = []
for i in coeffs:
    upper = i + 2*np.sqrt(np.linalg.det(cov))
    lower = i - 2*np.sqrt(np.linalg.det(cov))

    uppers.append(upper)
    lowers.append(lower)

print("Upper Confidence Interval:", uppers)
print("Lower Confidence Interval:", lowers)

# Plot SSE vs. number of iterations
plt.title("SSE vs. Number of Iterations")
x = np.arange(0, len(sse_array))
plt.plot(x, sse_array, label = "Sum of Squared Error")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.show()


#===================================================================================
# Diagnostic Analysis
# a. Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).
# b. Display the estimated variance of the error and the estimated covariance of the estimated
# parameters.
# c. Is the derived model biased or this is an unbiased estimator?
# d. Check the variance of the residual errors versus the variance of the forecast errors.
# e. If you find out that the ARIMA or SARIMA model may better represents the dataset, then
# you can find the model accordingly. You are not constraint only to use of ARMA model.
# Finding an ARMA model is a minimum requirement and making the model better is always
# welcomed.
#===================================================================================

# One step predicitons to get error
preds = functions_st_louis.one_step_forecast(y_train, coeffs, ar_order, ma_order)
var_ratio = np.var(y_train) / np.var(preds)
print("Ratio of the variance of the train set over the predicted set:", var_ratio)

plt.title("Predicted Values vs. Train Set")
x = np.arange(0, len(y_train))
plt.plot(x, y_train, label = "Train Data")
plt.plot(x, preds, label = "Predictions")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Y")
plt.show()

# Chi square whiteness test of residuals
pred_error = y_train - preds
lags=200
q = functions_st_louis.q_value(pred_error, lags)

functions_st_louis.plot_autocorr(pred_error, lags=20, title="ACF for residuals errors")

DOF = lags - ar_order - ma_order
alpha = .05
chi_crit = chi2.ppf(1-alpha, DOF)
print("Chi critical value:", chi_crit)
if q < chi_crit:
    print("The residual is white")
else:
    print("The residual is not white")

# Zero Pole Cancellation
print("Roots of AR:", np.roots(np.r_[1, coeffs[:ar_order]]))
print("Roots of MA:", np.roots(np.r_[1, coeffs[ar_order:]]))

# # H-Step forecast
forecast = functions_st_louis.h_step_forecast(df["Flow"], coeffs, ar_order, ma_order, y_test)
#forecast = functions_st_louis.h_step_forecast(df["Flow"], len(y_train)+1,coeffs, other=preds[:-1] ,step=len(y_test))

var_ratio = np.var(y_test) / np.var(forecast)
print("Ratio of the variance of the test set over the forecasted set:", var_ratio)

plt.title("Forecasts - St. Louis")
x = np.arange(0, len(y_test))
plt.plot(x, y_test, label = "Test Data")
plt.plot(x, forecast, label = "Forecasts")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Y")
plt.show()

forecast_error = y_test - forecast

# Comaprison of residual errors and forecast errors
var_ratio = np.var(pred_error) / np.var(forecast_error)
print("Ratio of the variance of the residual errors over the forecasted errors:", var_ratio)

#===================================================================================
# Base-models:
# average, naïve, drift, simple and exponential smoothing.
#===================================================================================

average_preds = functions_st_louis.average_forecast(y_train, y_test)

functions_st_louis.plot_preds(y_train, y_test, average_preds, "Average Forecast vs. Y-Test")

naive_preds = functions_st_louis.naive_forecast(y_train, y_test)

functions_st_louis.plot_preds(y_train, y_test, naive_preds, "Naive Forecast vs. Y-Test")

drift_preds = functions_st_louis.drift_forecast(y_train, y_test)

functions_st_louis.plot_preds(y_train, y_test, drift_preds, "Drift Forecast vs. Y-Test")

ses_preds = functions_st_louis.ses(y_train, y_test)

functions_st_louis.plot_preds(y_train, y_test, ses_preds, "SES Forecast vs. Y-Test")

