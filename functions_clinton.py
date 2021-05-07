import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import copy
from statsmodels.tsa.stattools import adfuller
from scipy import signal
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
import math
from numpy.linalg import inv


def GPAC(ry, jay, key):
    tab = np.zeros((jay, key - 1))
    for k in range(1, key):
        for j in range(jay):

            # initial value of ry
            a = int(np.ceil(len(ry) / 2))

            den = np.zeros((k, k))
            num = np.zeros(k)

            if k == 1:
                re = ry[a + j] / ry[a + j - 1]
            else:

                for m in range(k):
                    num[m] = ry[a + j + m]
                    for n in range(k):
                        den[m][n] = ry[j - n + m - a]

                den = np.array(den)
                num = np.array(num)
                num_a = np.transpose([num])
                num_b = den[:, :k - 1]
                num = np.concatenate((num_b, num_a), axis=1)
                re = np.linalg.det(num) / np.linalg.det(den)

                if re > 5:
                    re = "inf"

            tab[j][k - 1] = re

    table_GPAC(tab)


def GPAC3(ry, jay, key):
    tab = np.zeros((jay, key - 1))
    for k in range(1, key):
        for j in range(jay):

            # initial value of ry
            a = int(np.ceil(len(ry) / 2))

            den = np.zeros((k, k))
            num = np.zeros(k)

            if k == 1:
                re = ry[a + j] / ry[a + j - 1]
            else:

                for m in range(k):
                    for n in range(k):
                        den[m][n] = ry[j - n + m - a]




                den = np.array(den)
                num = np.array(num)
                num_a = np.transpose([num])
                num_b = den[:, :k - 1]
                num = np.concatenate((num_b, num_a), axis=1)

                num = den.copy()
                for row in range(k):
                    num[row][k - 1] = ry[j + row + 1]


                re = np.linalg.det(num) / np.linalg.det(den)

            tab[j][k - 1] = re

    table_GPAC(tab)

def GPAC2(ry, jay, key):
    tab = np.zeros((jay, key - 1))
    for k in range(1, key):
        for j in range(jay):

            # initial value of ry
            a = int(np.ceil(len(ry) / 2))

            if k == 1:
                re = ry[a + j] / ry[a + j - 1]

            else:
                den = np.zeros((k, k))
                for row in range(k):
                    for col in range(k):
                        den[row][col] = ry[j + row - col]
                num = den.copy()
                for row in range(k):
                    num[row][k - 1] = ry[j + row + 1]

                if np.linalg.det(den) != 0:
                    re = (np.linalg.det(num) / np.linalg.det(den)) * 0.05
                # else:
                #     re = "inf"

            tab[j][k - 1] = re

    table_GPAC(tab)

def table_GPAC(g):
    j, k = g.shape
    df = pd.DataFrame(g)
    df.columns = range(1, k + 1)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df, annot=True, ax=ax)
    plt.xlabel('k')
    plt.ylabel('j')
    plt.title("GPAC")
    plt.show()
    print(df)





def autocorr(x, lag):
    '''It returns the autocorrelation function for a given series.'''
    x = np.array(x)

    ## Proceed to calculate the numerator only for the oservationes given by the length of lag
    k = range(0,lag+1)

    ## Calculates the mean of the obs in the variable
    mean = np.mean(x)

    autocorr = []
    #Calculates a numerator and denominator based on the autocorrelation function formula
    for i in k:
        numer = 0
        for j in range(i, len(x)):
            numer += (x[j] - mean) * (x[j-i] - mean)
        denom = np.sum((x - mean) ** 2)
        autocorr.append(numer / denom)
    return autocorr

def ACFPlot(x, lags, title):
    # Limits the plot to the amount of lags given
    x = x[0:lags+1]
    # Makes the values go forward and backward (positive and negative)
    rx = x[::-1]
    rxx = np.concatenate((rx, x[1:]))
    lags = [i for i in range(-lags, lags+1)]
    lags = np.unique(np.sort(lags))
    plt.figure()
    plt.stem(lags, rxx)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.title(title)
    m_pred_f = 1.96 / len(x)
    plt.axhspan(-m_pred_f, m_pred_f, alpha=.1, color='black')
    plt.show()

def rolling_mean_var(y):
    # Mean and Variance Sales Over Time
    mean = []
    var = []
    for i in range(1, len(y)):
        mean.append(y[:i].mean())
        var.append(y[:i].var())

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(mean, label="Rolling mean")
    ax1.set_title("Rolling mean")
    # ax1.xlabel("Time")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(var, label="Rolling var")
    ax2.set_title("Rolling var")
    # ax2.xlabel("Time")

    plt.tight_layout()
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def difference(y, interval=1):
    diff = []
    for i in range(interval, len(y)):
        value = y[i] - y[i - interval]
        diff.append(value)
    return diff

#statsmodel
def ACF_PACF(y, lags):

    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags= lags)
    plt.title("ACF of y, T = " + str(len(y)))
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags= lags)
    plt.title("PACF of y, T =" + str(len(y)))
    plt.tight_layout()
    plt.show()

def correlation_coefficient(x, y):
    '''Correlation coefficient for two datasets'''
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Numerator
    a = np.sum(
        (x - x_mean) * (y - y_mean)
    )
    # Denominator
    b = np.sqrt(np.sum(
        (x - x_mean) * (x - x_mean)
    )
    ) * np.sqrt(np.sum(
        (y - y_mean) * (y - y_mean)
    )
    )
    r = a / b
    return r

# r_xy = np.around(correlation_coefficient(df["GDP"], df["Sales"]), 2)
# print('The correlation coefficient between the Sales value and GDP is', r_xy)

# def Q(array, lag):
#     '''Returns the Q value of the ACF residuals'''
#     sq = []
#     for i in array:
#         sq.append(i ** 2)
#         q = []
#         for i in range(1, len(sq)+1):
#             rsq = sq[:i]
#             T = len(rsq)
#             q.append(T * sum(rsq))
#     return q[-1]
#

def Q(array):
    '''Returns the Q value of the ACF residuals'''

    sq = []
    for i in array[1:]:
        sq.append(i ** 2)

        q = []

        for i in range(1, len(sq) + 1):
            rsq = sq[:i]
            T = len(rsq)
            q.append(T * sum(rsq))

    return q[-1]

# print("Value of Q:")
# print(sm.stats.acorr_ljungbox(x, lags=[5], boxpierce=True, return_df=True)

def average_method(x):
    '''It returns the predicted values for y'''
    y_hat = []
    for i in range(1,len(x)):
        mean = np.mean(x[0:i])
        y_hat.append(mean)
    return y_hat

def average_method_test(Ytr, Yte):
    '''It returns the predicted values for y'''
    y_hat = []
    for i in range(1,len(Yte)+1):
        mean = np.mean(Ytr)
        y_hat.append(mean)
    return y_hat

def naive_method(array):
    '''It returns the predicted values for y following the Naïve method'''
    y_hat = []
    for i in range(0,len(array)-1):
        y_hat.append(array[i])
    return y_hat

def naive_method_test(Ytr, Yte):
    '''It returns the predicted values for y following the Naïve method'''
    y_hat = []
    for i in range(0,len(Yte)):
        y_hat.append(Ytr.iloc[-1])
    return y_hat

def drift_method(array, h = 1):
    '''It returns the predicted values for y following the Drift method'''
    y_hat = []
    for i in range(3, len(array)+1):
        drift = array[i-2] + h * ((array[i-2] - array[0]) / (i - 2))
        y_hat.append(drift)
    return y_hat

def drift_method_test(Ytr, Yte, h):
    '''It returns the predicted values for y following the Drift method'''
    y_hat = []
    first = Ytr.iloc[0]
    last = Ytr.iloc[-1]
    for i in range(0, len(Yte)):
        drift = (h *(last - first))/(len(Ytr)) * (i+1) + last
        y_hat.append(drift)
    return y_hat

def ses_method(y, alpha, initial):
    pred = [initial]
    for i in range(1, len(y)):
        s = alpha*y[i-1] + (1-alpha) * pred[i-1]
        pred.append(s)
    return pred

def ses_method_test(Ytr, Yte, alpha):
    yhat = []
    first = Ytr[-1]
    train = ses_method(Ytr, alpha, first)
    for i in range(1, len(Yte) + 1):
        s = alpha*first + (1-alpha) * train[-1]
        yhat.append(s)
    return yhat

def MSE(x):
    return np.sum(np.power(x,2))/len(x)

def std_error(x, T, k):
    '''Returns the standard error for the predictor'''

    std_sq = []

    for i in x:
        std_sq.append(i ** 2)

    std_error = np.sqrt((1 / (T - k - 1)) * np.sum(std_sq))
    return std_error

def R2(yhat, y_test):
    y_mean = np.mean(y_test)
    r2 = np.sum((yhat - y_mean) ** 2) / np.sum((y_test - y_mean) ** 2)
    return r2

def autoregression(Year, y, order):
    y2 = y[order:]
    X = []
    for i in range(order, len(y)):
        x = []
        for j in range(i - order, i):
            x.append(y[j])
        X.append(x)
    X = np.array(X)
    H = np.matmul(X.T, X)
    inv_b = np.linalg.inv(H)
    # coefficients
    beta = np.matmul(np.matmul(inv_b, X.T), y2)
    # predictions
    pred = [np.dot(beta, X[i]) for i in range(len(X))]

    Year = Year[order:]

    return beta, pred, y2, Year

#Seaborn Heatmap
import seaborn as sns
def corr_matrix(df):
    corr = df.corr()
    plt.subplots(figsize=(7,7))
    sns.heatmap(corr.T, annot=True, cmap="YlGnBu")
    plt.tight_layout(pad=2, w_pad=0.5, h_pad=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def t_value(n, r, k):
    '''It returns the critical t-value'''

    num = n -2 - k
    den = 1 - (r ** 2)

    t_value = r * np.sqrt(num / den)

    return t_value

def coefficients(x, y):
    '''It returns the values of the regression model coefficients'''

    coef = np.dot(inv(np.dot(x.T, x)), np.dot(x.T, y))
    return coef

def interval(x, std_error, y_hat):
    '''It returns the 95% confidence interval for y_hat'''
    upper = []
    lower = []

    for i in range(0, len(x)):
        root = np.sqrt(1 + np.dot(x[i], np.dot(inv(np.dot(x.T, x)), x[i].T)))
        int = 1.96 * std_error * np.squeeze(np.asarray(root))
        low = np.squeeze(np.asarray(y_hat.T[i])) - int
        up = np.squeeze(np.asarray(y_hat.T[i])) + int
        lower.append(low)
        upper.append(up)

    for i in range(0, len(upper)):
        print(round(upper[i], 2), "< y_hat:{} <".format(i+1), round(lower[i], 2))


def corr_coef(x, y):
    '''It returns the correlation coefficient for two given datasets'''

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) * (x - x_mean))) * np.sqrt(np.sum((y - y_mean) * (y - y_mean)))

    r = numerator / denominator

    return r

def autocorr1(x):
    '''It returns the autocorrelation function for a given series.'''

    k = range(0, len(x))
    mean = np.mean(x)
    autocorr = []

    for i in k:
        num = 0
        den = 0

        for j in range(i, len(x)):
            num += (x[j] - mean) * (x[j - i] - mean)

        den = np.sum((x - mean) ** 2)

        autocorr.append(num / den)

    return autocorr

def prediction(y, coef, na, nb):
    n = max(na, nb)
    N = len(coef)
    pr = list(y[:n])
    lenP = len(y) - n
    for i in range(lenP):
        pred = 0
        for j in range(na):
            pred += -coef[j] * y[len(pr) - j - 1]
        for k in range(na, N):
            idx = k - na
            pred += coef[k] * (y[len(pr) - idx - 1] - pr[len(pr) - idx - 1])

        pr.append(pred)
    return pr

def forecast(y, coef, na, nb, y_test):
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

def CI(the, covariance):
    coef = the
    a = coef - (2 * (np.sqrt(covariance)))
    b = coef + (2 * (np.sqrt(covariance)))
    CI = np.concatenate((a, b), axis=None)
    return CI

def step_1(theta, y, ar_order, ma_order):

    ar = np.r_[1, theta[:ar_order]]
    ma = np.r_[1, theta[ar_order:ar_order + ma_order]]

    while min(len(ar), len(ma)) < max(len(ar), len(ma)):
        if len(ar) < len(ma):
            np.append(ar, 0)
        else:
            np.append(ma,0)

    sys = (ar, ma, 1)

    # print("AR", ar)
    # print("MA", ma)

    _, et = signal.dlsim(sys, y)
    SSE = np.dot(et.T, et)

    x = []
    delta = 10 ** -6

    for i in range(ar_order + ma_order):
        the = theta.copy()
        the[i] = the[i] + delta

        ar = np.r_[1, the[:ar_order]]
        ma = np.r_[1, the[ar_order:ar_order + ma_order]]

        if len(ar) < len(ma):
            ar = np.append(ar, 0)
        if len(ma) < len(ar):
            ma = np.append(ma, 0)

        sys = (ar, ma, 1)

        _, et_copy = signal.dlsim(sys, y)
        xi = np.subtract(et, et_copy) / delta
        if i == 0:
            x.append(xi)
            x = np.array(x)
        else:
            if len(x.shape) > 2:
                x = np.hstack((x[0], xi))
            else:
                x = np.hstack((x, xi))

    if len(x.shape) > 2:
        A = np.dot(x[0].T, x[0])
        g = np.dot(x[0].T, et)
    else:
        A = np.dot(x.T, x)
        g = np.dot(x.T, et)

    return SSE, x, A, g

def step_2(A, theta, mu, g, n, y, ar_order, ma_order):

    theta2 = np.dot(np.linalg.inv(A + mu * np.identity(n)), g)

    theta2 = theta2.flatten()
    the_new = theta + theta2

    the_new = the_new.flatten()

    ar = np.r_[1, the_new[:ar_order]]
    ma = np.r_[1, the_new[ar_order:ar_order + ma_order]]

    while min(len(ar), len(ma)) < max(len(ar), len(ma)):
        if len(ar) < len(ma):
            np.append(ar, 0)
        else:
            np.append(ma, 0)

    sys = (ar, ma, 1)

    _, et_new = signal.dlsim(sys, y)
    SSE_new = np.dot(et_new.T, et_new)

    return theta2, the_new, SSE_new

def descent(y, MAX, SSE, SSE_new, theta, theta2, the_new, A, x, g, n, mu, na, nb):

    start = 0
    mu_max = 100000
    SSE_history = []
    while start < MAX:
        if SSE_new < SSE:
            SSE_history.append(float(SSE))
            if np.linalg.norm(theta2) < 10 ** -3:
                the = the_new
                variance = SSE_new / (len(y) - (na + nb))
                covariance = variance * np.linalg.inv(A)

                plt.figure()
                plt.plot(range(len(SSE_history)), np.array(SSE_history), label="SSE")
                plt.title("SSE vs number of iterations")
                plt.show()

                # print("True parameters are = ", ar, ma)
                print("Estimated parameters are = ", the)
                print("Variance is = ", variance)
                print("Covariance Matrix is = ", covariance)

                C_I = []
                for i in range(len(the)):
                    intervals = CI(the[i], covariance[i][i])
                    C_I.append(intervals)

                print("-" * 20)
                print("Confidence intervals:")
                for i in range(len(C_I[:na])):
                    print(C_I[:na][i][0], " < a{} < ".format(i + 1), C_I[:na][i][1])

                for i in range(len(C_I[na:])):
                    print(C_I[na:][i][0], " < b{} < ".format(i + 1), C_I[na:][i][1])

                return the, variance, covariance, SSE_history

            else:
                theta = the_new
                mu = mu / 10

        while SSE_new >= SSE:
            mu = mu * 10
            if mu > mu_max:
                print("Error", the_new)

            theta2, the_new, SSE_new = step_2(A, theta, mu, g, n, y, na, nb)

        start += 1
        if start > MAX:
            print("Error", the_new)

        theta = the_new
        SSE, x, A, g = step_1(theta, y, na, nb)
        theta2, the_new, SSE_new = step_2(A, theta, mu, g, n, y, na, nb)

def levenberg_marquadt(y, na, nb, MAX=1000):

    # Initialize
    ar_params = [0] * na
    ma_params = [0] * nb

    # Initially theta will be 0 * number of parameters
    theta = ar_params + ma_params

    SSE, x, A, g = step_1(theta, y, na, nb)

    n = na + nb
    mu = 0.001
    theta2, the_new, SSE_new = step_2(A, theta, mu, g, n, y, na, nb)

    the, variance, covariance, SSE_history = descent(y, MAX, SSE, SSE_new, theta, theta2, the_new, A, x, g, n, mu, na, nb)

    return the, variance, covariance, SSE_history




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

def levenberg_marquadt_2(y, ar_order, ma_order, max_iterations=200):

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


