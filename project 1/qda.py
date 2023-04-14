"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from data import make_dataset1, make_dataset2
from plot import plot_boundary


def dataInit(X, y, lda = 1):
    """
    Computations of the mean(s), the probabilities(s) 
    and the covariance matrix(es) necessary for LDA/QDA

    Arguments:
    ----------
    - `X`: matrix of the data [N x p]
            N = number of instances in the dataset
            p = dimension of the data
    - `y`: 1D vector of length Ncontaining the number associated
            to each class in our data 
    - `lda`: Boolean value telling if LDA or QDA is performed
        -> 0 if QDA
        -> 1 if LDA

    Returns:
    --------
    - A list of the mean of each class
    - A vector of the probability of each class
    - A list of the covariance matrix of each class
    """

    N, p = X.shape
    classes = np.unique(y)  
    L = len(classes)                #number of classes

    data = [np.array([])]*L
    val = np.zeros(L)
    prob = np.zeros(L)
    cov = []
    mean = []
    cov_lda = np.zeros((p, p))

    for i in range(L):
        data[i] = X[y==i]
        val[i] = len(data[i])
        mean.append(np.mean(data[i], axis=0))
        cov.append(np.cov(data[i].T))
        
        if lda:
            cov_lda += (val[i]-1) * cov[i]

    prob = val/N

    if lda:
        cov = cov_lda/(N-L)
    
    return mean, prob, cov



class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

            # ====================
            # 
            # ====================

        classes = np.unique(y)
        K = len(classes)
        self.K = K
        
        _, p = X.shape
        mean, prob, cov = dataInit(X, y, lda = lda)

        inter = [np.zeros((p))]*K       #for efficiency
        cst = np.zeros((K))             #for efficiency

        self.lda = lda

        if lda == 1:
            self.prob = prob                    
            invCov = np.linalg.inv(cov)
                
            for i in range(K):
                v_c = np.atleast_2d(mean[i]).T       
                inter[i] = np.matmul(invCov,v_c)
                cst[i] = -1/2* np.matmul(v_c.T, inter[i]) + np.log(prob[i])     

        else:
            self.prob = prob
            invCov = [np.zeros((p,p))]*K
            det = np.zeros(K)

            for i in range(K):
                invCov[i] = np.linalg.inv(cov[i])
                det[i] = np.linalg.det(cov[i])             
                
                cst[i] = -1/2* np.log(det[i]) + np.log(prob[i])
            
            self.det = det          #keep it for predict_proba       
            

        
        self.theta = (mean, invCov)
        self.preComp = (cst, inter)

        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # 
        # ====================

        N, p = X.shape
        K = self.K
        result = np.zeros((N,K))

        cst = self.preComp[0]
        y = np.zeros((N))
        
        if self.lda == 1:                                                               #linear disciminant function
            inter = self.preComp[1]       
            for i in range(N):
                x_c = np.atleast_2d(X[i])
                y[i] = 0
                maxVal = float("-inf")
                for j in range(K):
                    result[i][j] = np.matmul(x_c, inter[j]) + cst[j]
                    if(result[i][j] > maxVal):
                        maxVal = result[i][j]
                        y[i] = j

        else:                                                                           #quadratic discriminant function
            mean = self.theta[0]
            invCov = self.theta[1]
            for i in range(N):
                x_c = X[i]
                y[i] = 0
                maxVal = float("-inf")
                for j in range(K):
                    v_c = np.atleast_2d(x_c-mean[j]).T      
                    inter = np.matmul(invCov[j], v_c)
                    result[i][j] = cst[j] - 1/2 * np.matmul(v_c.T, inter)
                    if(result[i][j] > maxVal):
                        maxVal = result[i][j]
                        y[i] = j
            
        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        
        N, _ = X.shape
        K = self.K
        mean = self.theta[0]
        invCov = self.theta[1]

        p = np.zeros((N, K))

        for i in range(N):
            for j in range(K):
                v_c = np.atleast_2d(X[i] - mean[j]).T

                if(self.lda == 1):
                    inter = np.matmul(invCov, v_c)
                else:
                    inter =  np.matmul(invCov[j], v_c)

                prob = self.prob[j] * np.exp(-1/2 * np.matmul(v_c.T, inter))

                if(self.lda == 0):
                    prob = prob / (np.power(self.det[j],1/2))
                
                p[i,j] = prob
            
            p[i, :] = p[i,:]/sum(p[i,:])
                
        return p



def testLda(sizeData = 1500, sizeTraining = 1200, lda_bool = 1, seedGen = 15, ds = 1):
    """
    Computes an LDA or a QDA classification on data generated from make_dataset

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `lda_bool`: Boolean value cotrolling the type of classifier
        -> 0 to do QDA
        -> 1  to do LDA
    - `seedGen`: Integer used for the random seed in the generation of data
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2

    Returns:
    --------
    A tuple containnig the accuracy on a test set and the fitted classifier.
    """

    if ds:
        X, y = make_dataset2(sizeData, random_state = seedGen)
    else:
        X, y = make_dataset1(sizeData, random_state = seedGen)

    X_train = X[0:sizeTraining]
    y_train = y[0:sizeTraining]
    X_test = X[sizeTraining+1:]
    y_test = y[sizeTraining+1:]


    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(X_train, y_train, lda=lda_bool)


    if(lda_bool == 1):
        fileName = "lda_test_size{}_seedGen{}_ds{}".format(sizeData, seedGen, ds)
    else:
        fileName = "qda_test_size{}_seedGen{}_ds{}".format(sizeData, seedGen, ds)

    plot_boundary(fileName, classifier, X_test, y_test)

    yFound = classifier.predict(X_test)


    accuracy = 0
    L = len(y_test)
    for i in range(L):
        if(y_test[i] == yFound[i]):
            accuracy = accuracy + 1

    return (accuracy/L, classifier)



def comp_lin_quad(sizeData = 1500, sizeTraining = 1200, seedGen = 15, ds = 1):
    """
    Computes the LDA and QDA classifier of a random dataset

    Arguments:
    ----------
    - `sizeData`:  Number of instances of generated data
    - `sizeTraining`: Number of instances to keep in the learning set
    - `seedGen`: Integer used for the random seed in the generation of data
    - `ds`: Boolean value controlling which set to use
        -> 0 to use make_dataset1
        -> 1 to use make_dataset2

    Returns:
    --------
    - The accuracy on the test set.

    """

    lda_val = [0, 1]
    L = len(lda_val)
    accuracy = np.zeros((L))
    for i in range(L):
        res = testLda(sizeData = sizeData, sizeTraining = sizeTraining, seedGen = seedGen, lda_bool=lda_val[i], ds=ds)
        accuracy[i] = res[0]

    return accuracy


if __name__ == "__main__":
    
    classifier = QuadraticDiscriminantAnalysis()

    seedGen = 15    
    sizeData = 1500     
    sizeTraining = 1200 

    X, y = make_dataset1(sizeData, random_state = seedGen)
    X_train = X[0:sizeTraining]
    y_train = y[0:sizeTraining]
    X_test = X[sizeTraining+1:]
    y_test = y[sizeTraining+1:]

    classifier.fit(X_train, y_train, lda = 1)

    plot_boundary("testLDA", classifier, X_test, y_test)

    classifier.fit(X_train, y_train, lda = 0)

    plot_boundary("testQDA", classifier, X_test, y_test)
    ##################

    seedGen = [13, 457, 9, 57, 42]

    L = len(seedGen)
    accuracy_lda = np.zeros(L)
    accuracy_qda = np.zeros(L)

    print("Makeset 1")

    for i in range(L):
        res = comp_lin_quad(seedGen=seedGen[i], ds=0)
        accuracy_qda[i] = res[0]
        accuracy_lda[i] = res[1]

    print("accuracy of lda ===>{}".format(accuracy_lda))
    print("accuracy of qda ===>{}\n".format(accuracy_qda))

    print("Accuracy lda==>{}--------------qda==>{}".format(sum(accuracy_lda)/L, sum(accuracy_qda)/L))
    print("std dev  lda==>{}--------------qda==>{}".format(np.std(accuracy_lda), np.std(accuracy_qda)))

    print("Makeset 2")

    for i in range(L):
        res = comp_lin_quad(seedGen=seedGen[i])
        accuracy_qda[i] = res[0]
        accuracy_lda[i] = res[1]

    print("accuracy of lda ===>{}".format(accuracy_lda))
    print("accuracy of qda ===>{}\n".format(accuracy_qda))

    print("Accuracy lda==>{}--------------qda==>{}".format(sum(accuracy_lda)/L, sum(accuracy_qda)/L))
    print("std dev  lda==>{}--------------qda==>{}".format(np.std(accuracy_lda), np.std(accuracy_qda)))
    
