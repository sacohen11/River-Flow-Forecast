#################################################################
# Title: Functions for the Final Project
# Author: Sam Cohen
# Date Created: 4/25/21
# Date Modified:
##################################################################

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from numpy import linalg as la
from scipy import signal


# Augmented Dickey-Fuller Test
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
	    print('\t%s: %.3f' % (key, value))

# Pearson Correlation Coefficient
def correlation_coefficent_cal(x,y):
    cor = sum((x - np.mean(x))*(y - np.mean(y))) / (np.sqrt(sum((x - np.mean(x))**2))*np.sqrt(sum((y - np.mean(y))**2)))
    print(cor)
    return cor

# Autocorrelation Function
def calc_autocorr(data, lags):
    data = np.array(data)
    x = sum([(data[t] - np.mean(data)) ** 2 for t in range(0, len(data))])
    ry = [sum([(data[t] - np.mean(data)) * (data[t - j] - np.mean(data)) / x for t in range(j, len(data))]) for j in range(0, lags)]
    return ry

def calc_autocorr_symmetric(data, lags):
    ry = calc_autocorr(data, lags)
    RY = ry[::-1]
    symmetric_array = np.concatenate((RY, ry[1:]))
    symmetric_lags = [i for i in range(-lags + 1, lags)]
    return symmetric_array


# Plot Autocorrelation Function with subplots
def plot_autocorr(data, lags, title):
    ry = calc_autocorr(data, lags)
    RY = ry[::-1]
    symmetric_array = np.concatenate((RY, ry[1:]))
    symmetric_lags = [i for i in range(-lags + 1, lags)]

    fig, ax = plt.subplots()
    #fig.set_figheight(10)
    #fig.set_figwidth(12)
    ax.stem(symmetric_lags, symmetric_array, use_line_collection=True)
    ax.set_title(title)
    ax.set_ylabel("Autocorrelation")
    ax.set_xlabel("Tao")
    ax.grid()
    m = 1.96 / np.sqrt(len(data))
    ax.axhspan(-m, m, alpha=.1, color='black')
    plt.show()

# Auto regression
def generate_ar_features(na, y):
    x = []
    T = len(y) - na - 1
    for i in range(1, na+1):
        first_index = na-i
        second_index = na+T-i+1
        arr = y[first_index:second_index]
        x.append([-x for x in arr])

    x = np.array(x)
    y = np.array(y[na:])
    return x.T, y.T

# This function is a helper function to calculate GPAC
def phi(na, nb, arr):

    ar_order = na
    ma_order = nb

    mid = int(np.floor(len(arr)/2))

    # we already know na and nb based on the user specifications... so just calculate one phi value
    # denominator first
    denominator = []
    loop_arr = [ar_order-i for i in range(0, ma_order)]
    for i in loop_arr:

        new_arr = arr[i+mid:i+ma_order+mid]
        new_arr = np.array([new_arr])
        new_arr = new_arr.T

        if i == ar_order:
            denominator = new_arr
        else:
            denominator = np.hstack((denominator, new_arr))

    numerator = np.delete(denominator, len(denominator[0])-1, 1)
    extra_col = arr[ar_order+1+mid:ar_order+ma_order+1+mid]
    extra_col = np.array([extra_col])
    extra_col = extra_col.T
    numerator = np.hstack((numerator, extra_col))

    # Calculate determinants and divide
    det_num = np.linalg.det(numerator)
    det_den = np.linalg.det(denominator)
    phi = float('inf')
    if det_den != 0:
        phi = det_num / det_den
        if phi > 5:
            phi = float('inf')
    #print("Phi value for na={} and nb={}".format(ar_order, ma_order), "is:", phi)
    return phi

# GPAC Calculation
def calc_gpac(ar_order, ma_order, y):

    # First, calculate the autocorrelations
    #arr = functions.calc_autocorr_symmetric(y, 20)
    arr = y
    table = []
    for i in range(0, ma_order):
        row = []
        for l in range(1, ar_order):
            val = phi(i, l, arr)
            row.append(val)
        if i == 0:
            row = np.array([row])
            table = row
        else:
            row = np.array([row])
            table = np.vstack((table, row[0]))

    gpac = pd.DataFrame(table, columns=np.arange(1, ar_order))
    print(gpac)
    ax = sns.heatmap(gpac, cmap='coolwarm', annot=True)#, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200))
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+.5, top-.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title("GPAC Table")
    plt.show()

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def rolling_mean_var(y):
    mean = []
    var = []

    for i in range(1, len(y) + 1):
        sub = y.head(i)
        sub = sub.agg(['mean', 'var'])
        mean.append(sub.loc["mean"])
        var.append(sub.loc["var"])

    x = pd.Series(range(1, len(y) + 1))
    fig, axs = plt.subplots(nrows=2, ncols=1)
    ((ax1), (ax2)) = axs
    ax1.plot(x, mean, label="Mean")
    ax2.plot(x, var, label="Var")

    ax1.set_title("Mean")
    ax2.set_title("Variance")

    for ax in axs.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
    plt.show()

def acf_pacf_plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    # plot ACF and PACF
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.subplot(211)
    plt.title("ACF/PACF of the raw data")
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    #plt.xlim(725, 740)

    plt.show()

    return acf, pacf


def correlation_matrix(df):
    corrMatrix = df.corr(method="pearson")
    ax = sns.heatmap(corrMatrix, cmap='coolwarm', annot=True)
    ax.set_title("Pearson's Correlation Matrix")
    plt.show()


def plot_strength_seasonality_trend(y):
    STL = statsmodels.tsa.seasonal.STL(y, 365) #, 365)
    res = STL.fit()

    T = res.trend
    S = res.seasonal
    R = res.resid

    fig = res.plot()
    plt.show()

    detrended = y - T
    seasonal_adj = y - S

    plt.figure()
    plt.title("Seasonally Adjusted Flow Data")
    plt.plot(seasonal_adj, label="adjusted seasonal")
    plt.plot(y, label="original")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("De-Trended Flow Data")
    plt.plot(detrended, label="detrended")
    plt.plot(y, label="original")
    plt.legend()
    plt.show()

    # Calculate the strength of the trend
    comp_t = 1 - (np.var(R) / (np.var(T + R)))
    strength_trend = max(0, comp_t)
    print("Strength of trend:", strength_trend)

    # Calculate the strength of the seasonality
    comp_s = 1 - (np.var(R) / (np.var(S + R)))
    strength_seasonality = max(0, comp_s)
    print("Strength of seasonality:", strength_seasonality)

def holt_winters(y_train, y_test):
    holtt = ets.ExponentialSmoothing(y_train, seasonal_periods=365, trend=None, damped=False, seasonal='mul').fit()
    holtf = holtt.forecast(steps=len(y_test))
    error = y_test - holtf
    holtf = pd.DataFrame(holtf).set_index(y_test.index)

    fig, ax = plt.subplots()
    #fig.set_figheight(20)
    #fig.set_figwidth(50)
    ax.plot(y_train, label="Train Data")
    ax.plot(y_test, label="Test Data")
    ax.plot(holtf, label="Holt-Winters")

    plt.legend(loc='upper left')
    plt.title('Holt-Winters Method - Flow')
    plt.xlabel('Time (weekly)')
    plt.ylabel('Weekly Mean Flow')
    plt.show()

    return error

def colinearity_detection(y):
    X = y.values
    H = np.matmul(X.T, X)
    _, d, _ = la.svd(H)
    print("SingularValues = ", d)

def condition_number(y):
    print("The Condition Number is ", la.cond(y))

def regression_coeffs(x, y):
    X = x.values
    H = np.matmul(X.T, X)
    H_inv = la.inv(H)
    coeffs = np.matmul(np.matmul(H_inv, X.T), y)
    print('Betas:', coeffs)
    return coeffs

def error_var(error):
    T = len(error)
    k = 1
    factor = 1/(T-k-1)
    summation = sum(error ** 2)
    var = np.sqrt(factor*summation)
    return var

def q_value(error, lags):
    Q = len(error) * np.sum(np.square(np.array(calc_autocorr(error, lags)))[1:])
    print("Q-value of the data is:", Q)
    return Q


def one_step_forecast(y, coeffs, na, nb):

    n = max(na, nb)
    N = len(coeffs)
    y_pred = list(y[:n])
    len_pred = len(y) - n
    for i in range(len_pred):
       pred = 0
       for j in range(na):
           pred += -coeffs[j] * y[len(y_pred) - j - 1]
       for k in range(na, N):
           idx = k - na
           pred += coeffs[k] * (y[len(y_pred) - idx - 1] - y_pred[len(y_pred) - idx - 1])

       y_pred.append(pred)

    return y_pred

def h_step_forecast(y, coef, na, nb, y_test):
    n = len(y_test)
    N = len(coef)
    fc = list(y[-N:])
    lenP = len(y_test) - N
    for i in range(lenP):
        pred = 0
        for j in range(na):
            pred += -coef[j] * fc[- j - 1]
        for k in range(na, N):
            if (len(fc)-1-k) <= n:
                idx = k - na
                pred += coef[k] * (y[len(fc) - idx - 1] - fc[len(fc) - idx - 2])

        fc.append(pred)
    return fc
#
# def h_step_forecast(y, start, coeffs, other, step=50):
#     preds = [0] * step
#     for h in range(1,step+1):
#         if h < 2:
#             preds[h-1] = coeffs[0] * y[start-1 + h - 1] + 1 * y[start-1 + h -52] - coeffs[0]*y[start-1 + h -53] + coeffs[1] * y[start-1 + h - 1] #+ coeffs[1] * other[-1]
#         elif h < 53:
#             preds[h-1] = coeffs[0] * preds[h-1 - 1] + 1 * y[start-1 + h -52] - coeffs[0]*y[start-1 + h -53]
#         elif h < 54:
#             preds[h-1] = coeffs[0] * preds[h-1 - 1] + 1 *  preds[h-1 - 52] - coeffs[0]*y[start-1 + h -53]
#         else:
#             preds[h-1] = coeffs[0] * preds[h-1 - 1] + 1 *  preds[h-1 - 52] - coeffs[0]*preds[h-1 -53]
#
#     return preds

def step1(theta, y, ar_order, ma_order):

    ar = np.r_[1, theta[:ar_order]]
    ma = np.r_[1, theta[ar_order:ar_order+ma_order]]

    while min(len(ar), len(ma)) < max(len(ar), len(ma)):
        if len(ar) < len(ma):
            ar = np.append(ar, 0)
        else:
            ma = np.append(ma, 0)

    sys = (ar, ma, 1)

    _, et = signal.dlsim(sys, y)
    sse = np.dot(et.T, et)

    x = []
    learning_rate = 10 ** -6
    for i in range(ar_order + ma_order):
        theta_temp = theta.copy()
        theta_temp[i] = theta_temp[i] + learning_rate

        ar = np.r_[1, theta_temp[:ar_order]]
        ma = np.r_[1, theta_temp[ar_order:ar_order+ma_order]]

        while min(len(ar), len(ma)) < max(len(ar), len(ma)):
            if len(ar) < len(ma):
                ar = np.append(ar, 0)
            else:
                ma = np.append(ma, 0)

        sys = (ar, ma, 1)

        _, et_copy = signal.dlsim(sys, y)
        xi = np.subtract(et, et_copy) / learning_rate
        if i == 0:
            x.append(xi)
            x = np.array(x)
        else:
            if len(x.shape) > 2:
                x = np.hstack((x[0], xi))
            else:
                x = np.hstack((x, xi))
    if len(x.shape) > 2:
        a = np.dot(x[0].T, x[0])
        g = np.dot(x[0].T, et)
    else:
        a = np.dot(x.T, x)
        g = np.dot(x.T, et)

    return sse, x, a, g

def step2(a, theta, mu, sse, g, n, y, ar_order, ma_order):

    change_theta = np.dot(np.linalg.inv(a + mu * np.identity(n)), g)

    change_theta = change_theta.flatten()
    theta_new = theta + change_theta

    theta_new = theta_new.flatten()

    ar = np.r_[1, theta_new[:ar_order]]
    ma = np.r_[1, theta_new[ar_order:ar_order+ma_order]]

    while min(len(ar), len(ma)) < max(len(ar), len(ma)):
        if len(ar) < len(ma):
            ar = np.append(ar, 0)
        else:
            ma = np.append(ma, 0)

    sys = (ar, ma, 1)

    _, et_new = signal.dlsim(sys, y)
    sse_new = np.dot(et_new.T, et_new)

    return change_theta, theta_new, sse_new

def levenberg_marquadt(y, ar_order, ma_order, max_iterations=200):

    # Initialize
    ar_params = [0] * ar_order
    ma_params = [0] * ma_order

    theta = ar_params + ma_params

    sses = []

    sse, x, a, g = step1(theta, y, ar_order, ma_order)

    sses.append(sse)

    n = ar_order+ma_order

    # Initialize mu
    mu = .001
    change_theta, theta_new, sse_new = step2(a, theta, mu, sse, g, n, y, ar_order, ma_order)

    sses.append(sse)

    num_iterations = 0
    mu_max = 10000000

    while num_iterations < max_iterations:
        if sse_new < sse:
            if np.linalg.norm(change_theta) < 10**-3:
                coeffs = theta_new
                sigma_squared = (sse_new / (len(y) - (ar_order+ma_order))).flatten()
                cov = sigma_squared * np.linalg.inv(a)
                return coeffs, sigma_squared, cov, np.array(sses).flatten()
            else:
                theta = theta_new
                mu = mu/10

        while sse_new >= sse:
            mu = mu*10
            if mu > mu_max:
                print("Error")
            change_theta, theta_new, sse_new = step2(a, theta, mu, sse, g, n, y, ar_order, ma_order)
            sses.append(sse_new)

        num_iterations += 1
        theta = theta_new
        sse, x, a, g = step1(theta, y, ar_order, ma_order)
        change_theta, theta_new, sse_new = step2(a, theta, mu, sse, g, n, y, ar_order, ma_order)
        sses.append(sse_new)

    print("Error - never converges")

def plot_preds(y_train, y_test, preds, title):
    plt.title("{}".format(title))
    x_train = np.arange(0, len(y_train))
    x_test = np.arange(len(y_train), len(y_train)+len(y_test))
    x_pred = np.arange(len(y_train)+ (len(y_test)-len(preds)), len(y_train)+len(y_test))
    plt.plot(x_train, y_train, label="Train Data")
    plt.plot(x_test, y_test, label="Test Data")
    plt.plot(x_pred, preds, label="Forecasts")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Y")
    plt.show()

def average_forecast(y_train, y_test):
    preds = []
    preds = [y_train.mean()] * len(y_test)
    return preds

def naive_forecast(y_train, y_test):
    end_obs = len(y_train) - 1
    preds = [y_train[end_obs]] * len(y_test)
    return preds

def calc_line(x1, x2, x3, y1, y2):
    slope = (y2 - y1) / (x2-x1)
    #print(slope)
    intercept = y2 - slope*x2
    #print(intercept)
    solution = slope*(x3) + intercept
    #print(solution)
    return solution

def drift_forecast(y_train, y_test):

    end_obs = len(y_train) - 1

    preds = [calc_line(0, end_obs, j,
                     y_train[0], y_train[end_obs]) for j in range(0, len(y_test))]

    return preds

def ses(y_train, y_test):
    alpha = .5
    model = ets.ExponentialSmoothing(y_train, trend=None, damped=False,
                                               seasonal=None).fit(smoothing_level=alpha, optimized=False)
    preds = model.forecast(steps=len(y_test))

    return preds

