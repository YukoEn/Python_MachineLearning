# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn import cross_validation as cv


def main():

    df = pd.read_csv("dataBefore_5000.csv")
    
    print(df.head())

    X = df[["calls_up","calls_down","starts_up","starts_down","availability","alarms"]]
    print(X)
    print(X.shape)
    TARGETS = df[["ave_callTime"]]
    print(TARGETS)


    # Display raw data
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(X.calls_up, TARGETS, c='b', s=20, alpha=0.5, label='call_up [#]')
    #plt.xlabel("[#]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs calls up")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.subplot(2, 1, 2)
    plt.scatter(X.calls_down, TARGETS, c='r', s=20, alpha=0.5, label='call_down [#]')
    #plt.xlabel("[#]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs calls down")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.scatter(X.starts_up, TARGETS, c='c', s=20, alpha=0.5, label='starts_up [#]')
    #plt.xlabel("[#]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs starts up")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.subplot(2, 1, 2)
    plt.scatter(X.starts_down, TARGETS, c='m', s=20, alpha=0.5, label='starts_down [#]')
    #plt.xlabel("[#]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs starts down")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    
    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.scatter(X.availability, TARGETS, c='g', s=20, alpha=0.5, label='availability [%]')
    #plt.xlabel("[%]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs availability")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.subplot(2, 1, 2)
    plt.scatter(X.alarms, TARGETS, c='k', s=20, alpha=0.5, label='alarms [#]')
    #plt.xlabel("[#]")
    plt.ylabel("Call time [s]")
    #plt.title("Call time vs alarms")
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)

    
    # Apply Linear Regression Model
    lm = LinearRegression()
    lm.fit(X, TARGETS.ave_callTime)

    plt.figure(4)
    plt.scatter(TARGETS, lm.predict(X), c='b', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs Measured (Linear Regression)")
    x = [0, 50]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 50))
    plt.ylim((0, 50))   
   
    print("Linear regression model \n  Before 5000 dataset")
    print("  Regresion coefficients: \n  [calls_up, calls_down, starts_up, starts_down, availability, alarms] \n    =",
    lm.coef_)   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X)), 5))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X)), 5))

    sampleId = np.linspace(1,5000,5000)
    #print(sampleId)
    #print(sampleId.shape)

    plt.figure(5)
    plt.subplot(2, 1, 1)
    plt.plot(sampleId, lm.predict(X), 'go', markersize=2, alpha=0.5, label='Predicted call time [s] vs Id \nLinear Regression')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 5000))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(sampleId, TARGETS, 'bo', markersize=2, alpha=0.5, label='Measured call time [s] vs Id \nRaw data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 5000))
    plt.ylim((0, 50))


    # Check the model with a single parameter
    lm = LinearRegression()
    lm.fit(X[['calls_up']], TARGETS)
    print("  *****")
    print("  calls_up")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['calls_up']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['calls_up']])), 3))
    lm = LinearRegression()
    lm.fit(X[['calls_down']], TARGETS)
    print("  calls_down")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['calls_down']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['calls_down']])), 3))    
    lm = LinearRegression()
    lm.fit(X[['starts_up']], TARGETS)
    print("  starts_up")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['starts_up']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['starts_up']])), 3))
    lm = LinearRegression()
    lm.fit(X[['starts_down']], TARGETS)
    print("  starts_down")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['starts_down']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['starts_down']])), 3))  
    lm = LinearRegression()
    lm.fit(X[['availability']], TARGETS)
    print("  availability")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['availability']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['availability']])), 3)) 
    lm = LinearRegression()
    lm.fit(X[['alarms']], TARGETS)
    print("  alarms")
    print("  Regresion coefficients =", lm.coef_[0])   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error =", round(mse(TARGETS, lm.predict(X[['alarms']])), 3))
    print("  R2 score =", round(r2_score(TARGETS, lm.predict(X[['alarms']])), 3))
    print("  *****")


    # Divide dataset randomly. Use train_test_sprit
    X_train, X_test, Y_train, Y_test = cv.train_test_split(
        X, TARGETS, test_size=0.4, random_state = 5)
    print("  X_train", X_train.shape)
    print("  X_test", X_test.shape)
    print("  Y_train", Y_train.shape)
    print("  Y_test", Y_test.shape)
    
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)

    print("  Train and test dataset")
    print("  Regression coefficients: \n  [calls_up, calls_down, starts_up, starts_down, availability, alarms] \n    =",
    lm.coef_)   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error with X_train and Y_train =", round(mse(Y_train, lm.predict(X_train)), 5))
    print("  R2 score with X_train and Y_train =", round(r2_score(Y_train, lm.predict(X_train)), 3))
    print("  Mean squared error with X_test and Y_test =", round(mse(Y_test, lm.predict(X_test)), 5))
    print("  R2 score with X_test and Y_test =", round(r2_score(Y_test, lm.predict(X_test)), 3))


    # Apply Ridge Regression Model   
    rmodel = Ridge(alpha=0.1)
    rmodel.fit(X_train, Y_train)

    plt.figure(6)
    plt.scatter(Y_train, rmodel.predict(X_train), c='c', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs measured (Ridge Regression)")
    x = [0, 50]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 50))
    plt.ylim((0, 50))

    print("Ridge regression model \n  Train dataset")
    print("  Mean squared error =", round(mse(Y_train, rmodel.predict(X_train)), 5))
    print("  R2 score =", round(r2_score(Y_train, rmodel.predict(X_train)), 5))

    sampleId = np.linspace(1,3000,3000)

    plt.figure(7)
    plt.subplot(2, 1, 1)
    plt.plot(sampleId, rmodel.predict(X_train), 'go', markersize=2, alpha=0.5, label='Predicted call time [s] vs Id \nRidge Regression')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 3000))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(sampleId, Y_train, 'bo', markersize=2, alpha=0.5, label='Measured call time [s] vs Id \nRaw data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 3000))
    plt.ylim((0, 50))


    # Apply Random Forest Regression Model 
    rfmodel = RandomForestRegressor()
    rfmodel.fit(X_train, Y_train.ave_callTime)

    print(X_train)

    print(Y_train)

    plt.figure(8)
    plt.scatter(Y_train, rfmodel.predict(X_train), c='g', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs measured (Random Forest Regression)")
    x = [0, 50]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 50))
    plt.ylim((0, 50))

    print("Random Forest Regression model \n  Train dataset")
    print("  Mean squared error =", round(mse(Y_train, rfmodel.predict(X_train)), 5))
    print("  R2 score =", round(r2_score(Y_train, rfmodel.predict(X_train)), 5))


    plt.figure(9)
    plt.subplot(2, 1, 1)
    plt.plot(sampleId, rfmodel.predict(X_train), 'go', markersize=2, alpha=0.5, label='Predicted call time [s] vs Id \nRandom Forest Regression')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 3000))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(sampleId, Y_train, 'bo', markersize=2, alpha=0.5, label='Measured call time [s] vs Id \nRaw data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0,3000))
    plt.ylim((0, 50))
    
    plt.show()
    

    
    
if __name__ == "__main__":
    main()
    
