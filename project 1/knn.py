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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from plot import plot_boundary

import matplotlib.pyplot as plt


# (Question 2)


def testKNN(sizeData = 1500, sizeTraining = 1200, nb_neighbours = 5, seedGen = 15, plotBin = 0, ds = 1):
    """
    Computes a KNN classification on data generated from make_dataset
    and then plots the decision boundary

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `nb_neighbours`: Number of neighbours in KNN
    - `seedGen`: Integer used for the random seed in the generation of data
    - `plotBin`: Boolean controlling whether the function should make plots
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2

    Returns:
    --------
    A tulpe containing the accuracy on a test set and the fitted classifier.
    """
    
    if ds:
        X, y = make_dataset2(sizeData, random_state = seedGen)
    else:
        X, y = make_dataset1(sizeData, random_state = seedGen)

    X_train = X[0:sizeTraining]
    y_train = y[0:sizeTraining]
    X_test = X[sizeTraining+1:]
    y_test = y[sizeTraining+1:]

    kNeighClass = KNeighborsClassifier(n_neighbors=nb_neighbours)
    kNeighClass.fit(X_train, y_train)

    if(plotBin):
        fileName = "testKN_nbneigh{}".format(nb_neighbours)
        plot_boundary(fileName, kNeighClass, X_test, y_test)

    yFound = kNeighClass.predict(X_test)

    accuracy = 0
    L = len(y_test)
    for i in range(L):
        if(y_test[i] == yFound[i]):
            accuracy = accuracy + 1

    return (accuracy/L, kNeighClass)


def cross_validation_knn(sizeData = 1500, sizeTraining = 1200, seedGen = [13, 457, 9, 57, 42], ds=1):
    """
    Performs a simple cross validation of the classifier

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `seedGen`: List of integers used for the random seed in the generation of data
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2
    Returns:
    -------
    The max accuracy and the corresponding index
    """
    
    k = len(seedGen)
    nb_neighbour = 1
    res = np.zeros((sizeTraining, k))
    acc = np.zeros(sizeTraining)

    for i in range(sizeTraining):
        if(nb_neighbour % 100 == 0):
            print(nb_neighbour)
        for j in range(k):
            res[i][j] = testKNN(sizeData = sizeData, sizeTraining=sizeTraining, nb_neighbours=nb_neighbour,seedGen= seedGen[j], ds=ds)[0]
        
        nb_neighbour=nb_neighbour+1

    stdDev = np.std(res, axis=1)
    acc = np.mean(res, axis=1)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("Number of neighbours")
    plt.ylabel("Accuracy")
    x = range(sizeTraining)
    ax.grid()
    ax.plot(x, acc)
    ax.fill_between(x, acc-stdDev, acc+stdDev, alpha = 0.2, color='red', label='Standard deviation')           
    ax.legend(loc = 'lower left')             
    plt.show()
    

    maxInd = np.argmax(acc)
    print("max index:{}".format(maxInd))
    print("max accuracy:{}".format(acc[maxInd]))
    print("standard dev:{}".format(stdDev[maxInd]))

    return acc, maxInd



if __name__ == "__main__":
    testNeigh = [1,5,25,125, 625, 1200]
    seedGen = [13, 457, 9, 57, 42]

    L = len(testNeigh)
    nb = len(seedGen)
    res = np.zeros((L, nb))
    accuracy = np.zeros((L))
    sizeD = 1500

    for i in range(L):
        classifier = testKNN(nb_neighbours=testNeigh[i], plotBin=1)


    for i in range(L):
        for j in range(nb):
            res[i][j] = testKNN(nb_neighbours=testNeigh[i],seedGen=seedGen[j])[0]
            accuracy[i] =accuracy[i] + res[i][j]
        accuracy[i] = accuracy[i]/nb
    
    stdDev = np.std(res, axis=1)

    print("Accuracy ===>{}\nstd dev  ===>{}".format(accuracy, stdDev))
    
    print("DATASET 1")
    acc1, maxInd1 = cross_validation_knn(ds = 0, seedGen=seedGen)


    print("DATASET 2")
    acc2, maxInd2 = cross_validation_knn(seedGen=seedGen)


    
