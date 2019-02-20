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

    # Read dataset
    # one outlier (id=9877) removed
    # Missing data: ave_callTime = 0  to -10000 for 8308 data points 
    df = pd.read_csv("dataBefore_allmCT.csv")
    df2 = pd.read_csv("dataAfter_all.csv")
    
    print(df.head())

    X = df[["calls_up","calls_down","starts_up","starts_down","availability","alarms"]]
    print(X)
    print(X.shape)
    TARGETS = df[["ave_callTime"]]
    print(TARGETS)

    X2 = df2[["calls_up","calls_down","starts_up","starts_down","availability","alarms"]]
    measured = df2[["ave_callTime"]]


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


    # Divide dataset randomly. Use train_test_sprit
    X_train, X_test, Y_train, Y_test = cv.train_test_split(
        X, TARGETS, test_size=0.4, random_state = 5)
    print("  X_train", X_train.shape)
    print("  X_test", X_test.shape)
    print("  Y_train", Y_train.shape)
    print("  Y_test", Y_test.shape)


    # Apply Linear Regression Model    
    lm = LinearRegression()
    lm.fit(X_train, Y_train.ave_callTime)

    plt.figure(4)
    plt.scatter(Y_train, lm.predict(X_train), c='b', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs Measured (Linear Regression)")
    x = [0, 100]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 100))
    plt.ylim((0, 100))   
   
    print("Linear regression model \n  Train dataset")
    print("  Regresion coefficients: \n  [calls_up, calls_down, starts_up, starts_down, availability, alarms] \n    =",
    lm.coef_)   
    print("  Regression intercept =", lm.intercept_)   
    print("  Mean squared error with X_train and Y_train =", round(mse(Y_train, lm.predict(X_train)), 5))
    print("  R2 score with X_train and Y_train =", round(r2_score(Y_train, lm.predict(X_train)), 3))
    print("  Mean squared error with X_test and Y_test =", round(mse(Y_test, lm.predict(X_test)), 5))
    print("  R2 score with X_test and Y_test =", round(r2_score(Y_test, lm.predict(X_test)), 3))

    sampleId = np.linspace(1,9100,9100)
    #print(sampleId)
    #print(sampleId.shape)

    plt.figure(5)
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(1,96,96), lm.predict(X_train[0:96]), '-go', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,96,96), Y_train[0:96], '-bo', linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 100))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(1,9100,9100), lm.predict(X_train[0:9100]), '-ko', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,9100,9100), Y_train[0:9100], '-mo',linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 9500))
    plt.ylim((0, 100))


    # Apply Ridge Regression Model   
    rmodel = Ridge(alpha=0.1)
    rmodel.fit(X_train, Y_train)

    plt.figure(6)
    plt.scatter(Y_train, rmodel.predict(X_train), c='g', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs Measured (Ridge Regression)")
    x = [0, 100]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 100))
    plt.ylim((0, 100))   
   
    print("Ridge regression model \n  Train dataset") 
    print("  Mean squared error with X_train and Y_train =", round(mse(Y_train, rmodel.predict(X_train)), 5))
    print("  R2 score with X_train and Y_train =", round(r2_score(Y_train, rmodel.predict(X_train)), 3))
    print("  Mean squared error with X_test and Y_test =", round(mse(Y_test, rmodel.predict(X_test)), 5))
    print("  R2 score with X_test and Y_test =", round(r2_score(Y_test, rmodel.predict(X_test)), 3))

    plt.figure(7)
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(1,96,96), rmodel.predict(X_train[0:96]), '-go', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,96,96), Y_train[0:96], '-bo', linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 100))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(1,9100,9100), rmodel.predict(X_train[0:9100]), '-ko', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,9100,9100), Y_train[0:9100], '-mo',linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')    
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 9500))
    plt.ylim((0, 100))


    # Apply Random Forest Regression Model 
    rfmodel = RandomForestRegressor()
    rfmodel.fit(X_train, Y_train.ave_callTime)

    plt.figure(8)
    plt.scatter(Y_train, rfmodel.predict(X_train), c='k', s=30, alpha=0.5)
    plt.xlabel("Measured call time [s]")
    plt.ylabel("Predicted call time [s]")
    plt.title("Predicted vs Measured (Random Forest Regression)")
    x = [0, 100]
    y = x
    lines = plt.plot(x, y)
    plt.setp(lines, color='k', linewidth=2.0)
    plt.xlim((0, 100))
    plt.ylim((0, 100))   
   
    print("Random Forest Regression model \n  Train dataset") 
    print("  Mean squared error with X_train and Y_train =", round(mse(Y_train, rfmodel.predict(X_train)), 5))
    print("  R2 score with X_train and Y_train =", round(r2_score(Y_train, rfmodel.predict(X_train)), 3))
    print("  Mean squared error with X_test and Y_test =", round(mse(Y_test, rfmodel.predict(X_test)), 5))
    print("  R2 score with X_test and Y_test =", round(r2_score(Y_test, rfmodel.predict(X_test)), 3))

    plt.figure(9)
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(1,96,96), rfmodel.predict(X_train[0:96]), '-go', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,96,96), Y_train[0:96], '-bo', linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 100))
    plt.ylim((0, 50))
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(1,9100,9100), rfmodel.predict(X_train[0:9100]), '-ko', linewidth=0.5, markersize=2, alpha=0.5, label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,9100,9100), Y_train[0:9100], '-mo',linewidth=0.5, markersize=2, alpha=0.5, label='Raw train data')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 9500))
    plt.ylim((0, 50))

    plt.figure(10)
    plt.plot(np.linspace(1,96,96), rfmodel.predict(X2[0+96:96+96]), '-go', linewidth=0.5, markersize=2, alpha=0.5,
             label='Predicted call time [s] \nbefore change\nRandom Forest Regression')
    plt.plot(np.linspace(1,96,96), measured[0+96:96+96], '-mo', linewidth=0.5, markersize=2, alpha=0.5, label='Measured data after change')  
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, 100))
    plt.ylim((0, 50))
    plt.xlabel("Sample id [#]")
    plt.ylabel("Call time [s]")
    plt.title("Call time vs sample id (one selected day)")

    plt.figure(11)    

    plt.plot(np.linspace(1,len(measured),len(measured)), rfmodel.predict(X2[0:len(measured)]), '-ko', linewidth=0.5, markersize=2, alpha=0.5,
            label='Predicted call time [s] vs samples \nRandom Forest Regression')
    plt.plot(np.linspace(1,len(measured),len(measured)), measured.ave_callTime, '-mo',linewidth=0.5, markersize=2, alpha=0.5, label='input data after change')
    plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
    ncol=2, shadow=True, fancybox=True)
    plt.xlim((0, len(measured)))
    plt.ylim((0, 50))
    plt.xlabel("Sample id [#]")
    plt.ylabel("Call time [s]")
    plt.title("Call time vs sample id (whole)")

    # Average call time from model
    # Removed 0 value of call time in calculations 
    YY = rfmodel.predict(X2)
    sumYY = 0
    count = 0

    for i in range(len(YY)):
        if YY[i] > 0:
            sumYY += YY[i]
            count += 1
            
    print("  Average call time from model =", round(sumYY/count, 5))
    print("    sum of call time:", sumYY)
    print("    counts of call time:", count)

    # Average call time from measured data (data_after)
    # Removed 0 value of call time in calculations 
    sumMeasured = 0
    count = 0

    for i in range(len(measured)):
        if measured.ave_callTime[i] > 0:
            sumMeasured += measured.ave_callTime[i]
            count += 1
            
    print("  Average call time from measured data (data_after) =", round(sumMeasured/count, 5))
    print("    sum of call time:", sumMeasured)    
    print("    counts of call time:", count)

    # Average gaps for call time 
    sumGaps = 0
    count = 0

    for i in range(len(measured)):
        if measured.ave_callTime[i] != -10000:
            if abs(YY[i] - measured.ave_callTime[i]) > 0:
                sumGaps += abs(YY[i] - measured.ave_callTime[i])
                count += 1
            
    print("  Average gaps between predicted (before) and measured (after) =", round(sumGaps/count, 5))
    print("    sum of call time:", sumGaps)    
    print("    counts of call time:", count)

      
    plt.show()
        
    
if __name__ == "__main__":
    main()
    
