"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
import numpy as np
from sklearn.utils import check_random_state


def make_dataset1(n_points, random_state=None):
    """Generate a 2D dataset

    Parameters
    -----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_points, 2]
        The feature matrix of the dataset
    y : array of shape [n_points]
        The labels of the dataset
    """
    random_state = check_random_state(random_state)
    n_y0 = int(n_points/2)
    n_y1 = n_points - n_y0

    angle_in_deg = 30
    sin_ = np.sin(np.deg2rad(angle_in_deg))
    cos_ = np.cos(np.deg2rad(angle_in_deg))

    stretch = 2

    C = np.array([[cos_*stretch, -sin_*stretch], [sin_/stretch, cos_/stretch]])
    X = np.r_[np.dot(random_state.randn(n_y0, 2), C) - np.array([.5, .5]),
              np.dot(random_state.randn(n_y1, 2), C) + np.array([.5, .5])]
    y = np.hstack((np.zeros(n_y0), np.ones(n_y1)))

    permutation = np.arange(n_points)
    random_state.shuffle(permutation)
    return X[permutation], y[permutation]

def make_dataset2(n_points, random_state=None):
    """Generate a 2D dataset

    Parameters
    -----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_points, 2]
        The feature matrix of the dataset
    y : array of shape [n_points]
        The labels of the dataset
    """
    random_state = check_random_state(random_state)
    n_y0 = int(n_points/2)
    n_y1 = n_points - n_y0

    angle_in_deg = 30
    sin_ = np.sin(np.deg2rad(angle_in_deg))
    cos_ = np.cos(np.deg2rad(angle_in_deg))

    stretch = 2.

    C = np.array([[cos_*stretch, -sin_*stretch], [sin_/stretch, cos_/stretch]])
    X = np.r_[random_state.randn(n_y0, 2) - np.array([.5, .5]),
              np.dot(random_state.randn(n_y1, 2), C) + np.array([.5, .5])]
    y = np.hstack((np.zeros(n_y0), np.ones(n_y1)))

    permutation = np.arange(n_points)
    random_state.shuffle(permutation)
    return X[permutation], y[permutation]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_points = 1500
    for make_set, fname in ((make_dataset1, "dataset1"), \
                            (make_dataset2, "dataset2")):
        plt.figure()
        X, y = make_set(n_points)
        X_y0 = X[y==0]
        X_y1 = X[y==1]
        plt.scatter(X_y0[:,0], X_y0[:,1], color="DodgerBlue", alpha=.5)
        plt.scatter(X_y1[:,0], X_y1[:,1], color="orange", alpha=.5)
        plt.grid(True)
        plt.xlabel("X_0")
        plt.ylabel("X_1")
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.title(fname)
        plt.savefig("{}.pdf".format(fname))
