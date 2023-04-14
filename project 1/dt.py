"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from plot import plot_boundary


# (Question 1)

def testDT(sizeData = 1500, sizeTraining = 1200, depth = 4, seedGen = 15, seedTree = 50,  plotBin = 0, ds = 1):
    """
    Computes a decision tree classification on data generated from make_dataset
    and then plots the decision boundary

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `depth`: Max depth in the decision tree
    - `seedGen`: Integer used for the random seed in the generation of data
    - `seedTree`: Integer used for the randomness of the classifier
    - `plotBin`: Boolean controlling whether the function should make plots
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2

    Returns:
    --------
    A tuple containing the accuracy on a test set and the fitted classifier.
    """
    
    if ds:
        X, y = make_dataset2(sizeData, random_state = seedGen)
    else:
        X, y = make_dataset1(sizeData, random_state = seedGen)

    X_train = X[0:sizeTraining]
    y_train = y[0:sizeTraining]
    X_test = X[sizeTraining+1:]
    y_test = y[sizeTraining+1:]

    treeClass = DecisionTreeClassifier(max_depth = depth, random_state=seedTree)
    treeClass.fit(X_train, y_train)
    
    if(plotBin):

        nameFile = "testTree_depth{}".format(depth)
        plot_boundary(nameFile, treeClass, X_test, y_test)

    yFound = treeClass.predict(X_test)

    accuracy = 0
    L = len(y_test)
    for i in range(L):
        if(y_test[i] == yFound[i]):
            accuracy = accuracy + 1

    return (accuracy/L, treeClass)



def cross_validation_dt(sizeData = 1500, sizeTraining = 1200, seedGen = [13, 457, 9, 57, 42], seedTree = [5, 2, 100, 25, 93], max_depth = 25, ds = 1):
    """
    Performs a simple cross validation of the classifier

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `seedGen`: List of integers used for the random seed in the generation of data
    - `seedTree`: List of integers used for the randomnessof the classifier
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2
    Returns:
    -------
    The max accuracy and the corresponding index
    """
    
    d = 1                                       #depth
    k = len(seedGen)
    res = np.zeros((max_depth, k))

    for i in range(max_depth):
        if(d % 100 == 0):
            print(d)
        for j in range(k):
            res[i][j] = testDT(sizeData = sizeData, sizeTraining = sizeTraining, depth=d, seedGen=seedGen[j], seedTree = seedTree[j], ds=ds)[0]
        
        d=d+1       

    stdDev = np.std(res, axis=1)
    acc = np.mean(res, axis=1)

    fig = plt.figure()
    ax = plt.axes()
    x = range(max_depth)
    ax.plot(x, acc)
    ax.grid()
    ax.fill_between(x, acc-stdDev, acc+stdDev, alpha = 0.2, color='red', label='Standard deviation')           
    ax.legend(loc = 'lower right')         
    plt.xlabel("maximum depth")
    plt.ylabel("Accuracy")
    plt.show()

    maxInd = np.argmax(acc)
    print("max index:{}".format(maxInd))
    print("max accuracy:{}".format(acc[maxInd]))
    print("standard dev:{}".format(stdDev[maxInd]))
    
    return acc, maxInd


if __name__ == "__main__":
    testDepth = [1,2,4,8, None]
    seedGen = [13, 457, 9, 57, 42]
    seedTree = [5, 2, 100, 25, 93]

    L = len(testDepth)
    nb = len(seedGen)
    res = np.zeros((L, nb))
    accuracy = np.zeros((L))

    for i in range(L):
        classifier = testDT(depth=testDepth[i], plotBin=1)[1]

    
    for i in range(L):
        for j in range(nb):
            res[i][j] = testDT(depth=testDepth[i], seedGen=seedGen[j], seedTree=seedTree[j])[0]
            accuracy[i] = accuracy[i] + res[i][j]
        accuracy[i] = accuracy[i]/nb

    stdDev = np.std(res, axis=1)

    print("Accuracy ===>{}\nstd dev  ===>{}".format(accuracy, stdDev))

    print("\nDATASET 1\n")
    acc1, maxInd1 = cross_validation_dt(max_depth=24, ds=0, seedGen=seedGen, seedTree=seedTree)

    print("\nDATASET 2\n")
    acc2, maxInd2 = cross_validation_dt(max_depth=24, seedGen=seedGen, seedTree=seedTree)
    


