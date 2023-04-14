from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor

import numpy as np
import matplotlib.pyplot as plt



def decomp_set(k, X, y, XTSet, yTSet, est, seed, nbPerSet = 0):
    """
    Performs a decomposition of the dataset into different learning sets

    Arguments:
    ----------
    - `k`:  Number of smaller learning sets
    - `X`: Dataset of inputs
    - `y`: Dataset of outputs
    - `XTSet`: Inputs of the test set
    - `yTSet`: Outputs of the test set
    - `est`:  Sci-kit learn estimator
    - `seed`: Seed for the random attribution in sub sets
    - `nbPerSet`: Number of people per smaller learning sets (predominant on k)

    Returns:
    --------
    The mean of the MSE of the predicted output for each input
    and the mean of variance of the predicted output for each input
    """

    L = X.shape[0]
    random_state = check_random_state(seed)

    permutation = np.arange(L)      
    random_state.shuffle(permutation)
    if(nbPerSet == 0):
        ind = np.linspace(0, L, k).astype(int)
    else:
        ind = [0]
        val = nbPerSet
        while(val <= L):
            ind.append(val)
            val = val + nbPerSet
        
        L = val-nbPerSet
        k = int((val-nbPerSet)/nbPerSet)+1

    yMod = np.zeros((k-1, XTSet.shape[0]))
    yModSous = np.zeros((k-1, XTSet.shape[0]))

    
    for i in range(k-1):

        XLSet = X[ind[i]:ind[i+1], :]
        yLSet = y[ind[i]:ind[i+1]]

        classi = est.fit(XLSet, yLSet)
        yPredict = classi.predict(XTSet)

        yMod[i, :] = yPredict
        yModSous[i, :] = (yPredict-yTSet)**2


    MSE = np.mean(yModSous, axis = 0)
    MSE = np.mean(MSE)

    var = np.var(yMod, axis=0)
    var = np.mean(var)

    return MSE, var



def plot_error(E, V, B, xInd, e, sizeBool = 0, bagBool = 0):
    """
    Makes plot

    Arguments:
    ----------
    - `E`: List of values for the expected error
    - `V`: List of values for the variance 
    - `B`: List of values for the bias
    - `xInd`: List/Vector of the abscissa
    - `e`: Indicates which estimator was used
    - `sizeBool`: Boolean value telling if the function 
                  is called for changes in complexity param or LS size
        -> 0 for changes in complexity parameter
        -> 1 for changes in LS size
    """

    _, ax = plt.subplots(1, figsize=(10, 10))
    
    ax.plot(xInd, E, "blue", marker="D", label="Error")
    ax.plot(xInd, V, "green", marker="^", label="Variance")
    ax.plot(xInd, B, "red", marker = "s", label="Bias² + noise")
    ax.grid()

    if(sizeBool == 0):
        if(e == 0):
            ax.set_xlabel("Max depth")
        elif(e == 1):
            ax.set_xlabel("k")
        else:
            ax.set_xlabel("Lambda")
    else:
        ax.set_xlabel("Size of learning set")

    ax.set_ylabel("Error")

    if(bagBool):
        ax.set_title("bagging version")
    
    plt.show()



def expComplexity(XTSet, yTSet, X2, y2, nSet = 500, N = [17, 30, 150], seed = 42):
    """
    Allows to plot the graph for changes in complexity parameters

    Arguments:
    ----------
    - `XTSet`: Inputs of the test set
    - `yTSet`: Outputs of the test set
    - `X2`: Inputs of the learning set
    - `y2`: Outputs of the learning set
    - `nSet`:  Size of the small learning sets
    - `N`: List of 3 elements containng the last 
           values of the different complexity parameters
    - `seed`: Seed for the random attribution in sub sets
    """

    k = 0           #parameters used to determine the number of LS but useless here (nSet overwrites it)

    E = []          # expected error 
    V = []          # variance
    B = []          # bias


    for e in range(3):
    
        xInd = np.arange(N[e])
        for i in range(N[e]):

            if(e == 0):
                est = DecisionTreeRegressor(max_depth=i+1)
                cplot = "Regression tree (complexity)"
            elif(e == 1):
                est = KNeighborsRegressor(n_neighbors=i+1)
                cplot = "KNN (complexity)"
            else:
                est = Ridge(alpha=i+1)
                cplot = "Ridge regression (complexity)"
            
            MSE, var = decomp_set(k+1, X2, y2, XTSet, yTSet, est, seed, nbPerSet = nSet)

            E.append(MSE)
            V.append(var)
            B.append(MSE-var)
        
        print(cplot)

        plot_error(E, V, B, xInd+1, e)

        E = []
        V = []
        B = []



def sizeExpComp(param, e, X2, y2, XTSet, yTSet, N, bagBool = 1, iStart=100, iInc = 100, seed = 42):
    """
    Allows to plot the graph for changes in LS size

    Arguments:
    ----------
    - `param`: List of parameters for the estimator
    - `e`: Decides which estimator should be used
        -> 0: Regression tree
        -> 1: KNN
        -> 2: Ridge regression
    - `X2`: Inputs of the learning set
    - `y2`: Outputs of the learning set
    - `XTSet`: Inputs of the test set
    - `yTSet`: Outputs of the test set
    - `N`: Largest size of training sets
    - `bagBool`: Boolean to control if bagging should be performed
    - `iStart`: Smallest size of the learning sets
    - `iInc`: Step in size for each computation
    - `seed`: Seed for the random attribution in sub sets
    """

    k = 5
    iVal = []
    i = iStart

    L = len(param)

    for j in range(L):

        E = []
        V = []
        B = []

        E_b = []
        V_b = []
        B_b = []

        iVal = []

        i = iStart
        cparam = param[j]
        
        while(i <= N):

            if(e == 0):
                est = DecisionTreeRegressor(max_depth = cparam)
                cplot = "Regression tree ==> max_depth = {}".format(cparam)
            elif(e == 1):
                est = KNeighborsRegressor(n_neighbors=cparam)
                cplot = "KNN ==> k = {}".format(cparam)
            else:
                est = Ridge(alpha=cparam)
                cplot = "Ridge regression ==> lambda = {}".format(cparam)


            MSE, var = decomp_set(k+1, X2, y2, XTSet, yTSet, est, seed, nbPerSet = i)
            E.append(MSE)
            V.append(var)
            B.append(MSE-var)

            if(bagBool):
                est_b = BaggingRegressor(base_estimator=est)
                MSE_b, var_b = decomp_set(k+1, X2, y2, XTSet, yTSet, est_b, seed, nbPerSet = i)
                E_b.append(MSE_b)
                V_b.append(var_b)
                B_b.append(MSE_b-var_b)

            iVal.append(i)

            i=i+iInc

        print("{}".format(cplot))

        plot_error(E, V, B, iVal, e, sizeBool=1, bagBool = 0)

        if(bagBool):
            plot_error(E_b, V_b, B_b, iVal, e, sizeBool=1, bagBool = bagBool)
    

def expSize(XTSet, yTSet, X2, y2, N = 1500, eList = [0,1,2]):
    """
    Decides what should be ploted for changes in LS size

    Arguments:
    ----------
    - `XTSet`: Inputs of the test set
    - `yTSet`: Outputs of the test set
    - `X2`: Inputs of the learning set
    - `y2`: Outputs of the learning set
    - `N`: Largest size of the learning sets
    - `eList`: List of possibly 3 values
        -> 0 for regression tree
        -> 1 for KNN
        -> 2 for ridge regression
    """

    N = 1500

    for e in eList:
        
        if(e == 0):
            listparam = [1, 2, 4, 8, None]
            sizeExpComp(listparam, e, X2, y2, XTSet, yTSet, N)

        elif(e == 1):
            listparam = [5]
            sizeExpComp(listparam, e, X2, y2, XTSet, yTSet, N, bagBool=0)

        else:
            listparam = [30]
            sizeExpComp(listparam, e, X2, y2, XTSet, yTSet, N)





if __name__ == '__main__':

    # Fetching dataset
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target

    # Scaling inputs
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Shuffling input-output pairs
    X, y = shuffle(X, y, random_state=42)

    #define the test set
    M = 15000
    XTSet = X[M:M+5000]
    yTSet = y[M:M+5000]
    XTSet2 = X[M:M+3000]
    yTSet2 = y[M:M+3000]
    X2 = X[0:M]
    y2 = y[0:M]

    expComplexity(XTSet, yTSet, X2, y2)

    expSize(XTSet, yTSet, X2, y2, eList = [0, 1])
    expSize(XTSet2, yTSet2, X2, y2, eList = [2])