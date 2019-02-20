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

    lm = LinearRegression()
    lm.fit(X, TARGETS)
    print("Estimated interept coefficient:", lm.intercept_)
    print(lm.coef_)

    plt.figure(1)
    plt.subplot(3, 2, 1)
    plt.scatter(X.calls_up, TARGETS, c='b', s=20, alpha=0.5)
    plt.xlabel("[#]")
    plt.ylabel("[s]")
    plt.title("T vs calls up")

    plt.subplot(3, 2, 2)
    plt.scatter(X.calls_down, TARGETS, c='r', s=20, alpha=0.5)
    plt.xlabel("[#]")
    plt.ylabel("[s]")
    plt.title("T vs calls down")

    plt.subplot(3, 2, 3)
    plt.scatter(X.starts_up, TARGETS, c='c', s=20, alpha=0.5)
    plt.xlabel("[#]")
    plt.ylabel("[s]")
    plt.title("T vs starts up")

    plt.subplot(3, 2, 4)
    plt.scatter(X.starts_down, TARGETS, c='m', s=20, alpha=0.5)
    plt.xlabel("[#]")
    plt.ylabel("[s]")
    plt.title("T vs starts down")

    plt.subplot(3, 2, 5)
    plt.scatter(X.availability, TARGETS, c='g', s=20, alpha=0.5)
    plt.xlabel("[%]")
    plt.ylabel("[s]")
    plt.title("T vs availability")

    plt.subplot(3, 2, 6)
    plt.scatter(X.alarms, TARGETS, c='k', s=20, alpha=0.5)
    plt.xlabel("[#]")
    plt.ylabel("[s]")
    plt.title("T vs alarms")
   


    plt.figure(2)
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
    #plt.show()    


    print("Linear regression model \n Before 5000 dataset")
    print("Mean squared error =", round(mse(TARGETS, lm.predict(X)), 5))
    print("R2 score =", round(r2_score(TARGETS, lm.predict(X)), 5))



    #for i in range (len(TARGETS)-1):
    #    total += TARGETS.ave_callTime[i]
        


    lm = LinearRegression()
    lm.fit(X[['calls_up']], TARGETS)
    print("Mean squared error, calls_up =", round(mse(TARGETS, lm.predict(X[['calls_up']])), 3))
    print("R2 score, calls_up =", round(r2_score(TARGETS, lm.predict(X[['calls_up']])), 3))

    lm = LinearRegression()
    lm.fit(X[['calls_down']], TARGETS)
    print("Mean squared error, calls_down =", round(mse(TARGETS, lm.predict(X[['calls_down']])), 3))
    print("R2 score, calls_down =", round(r2_score(TARGETS, lm.predict(X[['calls_down']])), 3))

    lm = LinearRegression()
    lm.fit(X[['starts_up']], TARGETS)
    print("Mean squared error, starts_up =", round(mse(TARGETS, lm.predict(X[['starts_up']])), 3))
    print("R2 score, starts_up =", round(r2_score(TARGETS, lm.predict(X[['starts_up']])), 3))

    lm = LinearRegression()
    lm.fit(X[['starts_down']], TARGETS)
    print("Mean squared error, starts_down =", round(mse(TARGETS, lm.predict(X[['starts_down']])), 3))
    print("R2 score, starts_down =", round(r2_score(TARGETS, lm.predict(X[['starts_down']])), 3))

    lm = LinearRegression()
    lm.fit(X[['availability']], TARGETS)
    print("Mean squared error, availability =", round(mse(TARGETS, lm.predict(X[['availability']])), 3))
    print("R2 score, availability =", round(r2_score(TARGETS, lm.predict(X[['availability']])), 3))

    lm = LinearRegression()
    lm.fit(X[['alarms']], TARGETS)
    print("Mean squared error, alarms =", round(mse(TARGETS, lm.predict(X[['alarms']])), 3))
    print("R2 score, alarms =", round(r2_score(TARGETS, lm.predict(X[['alarms']])), 3))


    X_train, X_test, Y_train, Y_test = cv.train_test_split(
        X, TARGETS, test_size=0.33, random_state = 5)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("Y_train", Y_train.shape)
    print("Y_test", Y_test.shape)

    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)

    print("Linear Regression model with X_train, MSE with X_train:",
        round(mse(Y_train, lm.predict(X_train)), 53))

    print("Linear Regression model with X_train, MSE with X_test and Y_test:",
        round(mse(Y_test, lm.predict(X_test)), 3))


    rmodel = Ridge(alpha=0.1)
    rmodel.fit(X, TARGETS)

    print("Ridge regression model \n before 5000 dataset")
    print("Mean squared error =", round(mse(TARGETS, rmodel.predict(X)), 5))
    print("R2 score =", round(r2_score(TARGETS, rmodel.predict(X)), 5))

    plt.figure(3)
    plt.scatter(TARGETS, rmodel.predict(X), c='c', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs measured (Ridge Regression)")

    x = [0, 50]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)

    plt.xlim((0, 50))
    plt.ylim((0, 50)) 

    
    rfmodel = RandomForestRegressor()
    rfmodel.fit(X, TARGETS.ave_callTime)

    print("Random Forest Regression model \n before 5000 dataset")
    print("Mean squared error =", round(mse(TARGETS, rfmodel.predict(X)), 5))
    print("R2 score =", round(r2_score(TARGETS, rfmodel.predict(X)), 5))

    plt.figure(4)
    plt.scatter(TARGETS, rfmodel.predict(X), c='g', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs measured (Random Forest Regression)")

    x = [0, 50]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)

    plt.xlim((0, 50))
    plt.ylim((0, 50))   
    plt.show() 

    
    
if __name__ == "__main__":
    main()
    
