import numpy as np
import pandas as pd
import scipy as sc

import os

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import LabelEncoder


DATA_DIR = "../data"


# Helper Functions
log = np.log
gamma = sc.special.gamma
digamma = sc.special.digamma
det = np.linalg.det
diag = np.diag
outer = np.outer
tr = np.trace
dot =  np.dot
concat = np.concatenate
normal = np.random.normal
arr = np.array
cstack = np.column_stack
sqrt = np.sqrt
inv = np.linalg.inv
norm = sc.stats.norm
append = np.append
slogdet = np.linalg.slogdet


def get_heart_data(shuffle=True):
    data_dir = "%s/UCI/heart-disease" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.startswith("processed")]

    #print files

    #dfs = [pd.read_csv("%s/%s" %(data_dir,f)) for f in files]
    dfs = [np.genfromtxt("%s/%s" %(data_dir,f), delimiter=",")  for f in files]

    data = np.concatenate(dfs)

    # process labels, replace 1,2,3 with 1
    msk = data[:,-1] > 0
    data[msk, -1] = 1.
    data[~msk, -1] = -1.

    X, y = data[:, :-1], data[:, -1]

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X, y


def get_breast_data():
    data_dir = "%s/UCI/breast-cancer-wisconsin" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.startswith("wdbc.data")]

    dfs = [pd.read_csv("%s/%s" %(data_dir,f), header=None) for f in files]
    #dfs = [np.genfromtxt("%s/%s" %(data_dir,f), delimiter=",")  for f in files]

    data = pd.concat(dfs)

    # process labels, replace B with 1, M with -1
    msk = data[1].str.startswith('B')
    data.loc[msk, 1] = 1.
    data.loc[~msk, 1] = -1.

    y = data.values[:,1]
    X = data.values[:,2:]

    return X, y


def get_iono_data(shuffle=True):
    data_dir = "%s/UCI/ionosphere" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.endswith("data")]

    dfs = [pd.read_csv("%s/%s" %(data_dir,f), header=None) for f in files]

    data = pd.concat(dfs)

    y = data.values[:,-1]
    X = data.values[:,:-1]

    # process labels, replace g with 1, b with -1
    msk = y == "g"
    y[msk] = 1.
    y[~msk] = -1.

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X.astype('f4'), y.astype('f4')


def get_pima_data(shuffle=True):
    data_dir = "%s/UCI/pima-indians-diabetes" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.startswith("pima-indians-diabetes.data")]
    dfs = [pd.read_csv("%s/%s" %(data_dir,f), header=None) for f in files]
    data = np.concatenate(dfs)

    msk = data[:,-1] > 0
    data[msk, -1] = 1.
    data[~msk, -1] = -1.

    X = data[:,:-1]
    y = data[:, -1]

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X, y


def get_wpbc_data(shuffle=True):
    data_dir = "%s/UCI/wpbc" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.startswith("wpbc.data")]
    dfs = [pd.read_csv("%s/%s" %(data_dir,f), header=None) for f in files]
    data = np.concatenate(dfs)

    msk = data[:,0] == 'N'
    data[msk, 0] = 1.
    data[~msk, 0] = -1.

    msk = data[:,-1] == '?'

    data[msk, -1] = 0

    X = data[:,1:-1]
    y = data[:,0]

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X.astype('f4'), y.astype('f4')


def get_sonar_data(shuffle=True):
    data_dir = "%s/UCI/sonar" %DATA_DIR
    files = [f for f in os.listdir(data_dir) if f.startswith("sonar.all-data")]
    dfs = [pd.read_csv("%s/%s" %(data_dir,f), header=None) for f in files]
    data = np.concatenate(dfs)
    msk = data[:,-1]=='R'
    data[msk, -1] = 1.
    data[~msk, -1] = -1.

    X = data[:,:-1]
    y = data[:, -1]

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X.astype('f4'), y.astype('f4')


def get_liver_data(shuffle=True):
    data_dir = "%s/UCI/liver" %DATA_DIR
    data = pd.read_csv("%s/liver.csv" %(data_dir), header=None).values
    msk = data[:,-1]== 1
    data[msk, -1] = 1.
    data[~msk, -1] = -1.

    X = data[:,:-1]
    y = data[:, -1]

    # encode categorical variables
    le = LabelEncoder()

    for i in range(X.shape[1]):
        if not np.issubdtype(X[:,i].dtype, np.number):
            X[:,i] = le.fit_transform(X[:,i])

    if shuffle:
        np.random.seed(123)
        p = np.random.permutation(X.shape[0])
        X, y = X[p], y[p]

    return X.astype('f4'), y.astype('f4')


def preprocess_feats(X, mean=None, std=None):
    '''
    Normalization of features
    '''

    X[np.isnan(X)] = 0.

    if mean is None:
        mean = X.mean(axis=0)

    if std is None:
        std = X.std(axis=0)

    return (X-mean)/std


def preprocess_kernel(K):
    '''
    Spherical normalization of kernel matrix
    '''

    if len(K.shape) < 3:
        K = K[np.newaxis,...]

    for i in range(K.shape[0]):
        d = np.sqrt(np.diag(K[i,:,:]))
        K[i,:,:] = K[i,:,:]/np.outer(d, d)

    return K


'''KERNELS = {
    'gaussian': get_gaussian_kernel,
    'poly': get_ploynomial_kernel,
    'dist': get_distances_to_kernel
}'''

def get_gaussian_kernel(X, Y=None, width=.2):
    if Y is None:
        Y = X

    return rbf_kernel(X, gamma=1.0/width)


def get_ploynomial_kernel(X, Y=None, deg=2):
    if Y is None:
        Y = X

    return polynomial_kernel(X, Y, degree=deg)


def get_distances_to_kernel(X):
    return (-X/X.mean())


def get_kernels(X, Y=None, poly=False, max_poly_deg=3, gauss=False, max_gauss_width=6, min_gauss_width=-3,
 distk=False, feat_kernel=False):
    kernels = []
    msk = np.isnan(X)
    X[msk] = 0.
    degrees = np.arange(1,max_poly_deg+1)
    widths = map(lambda x: 2.**x, np.arange(min_gauss_width,max_gauss_width+1))

    if Y is None:
        Y = X

    if poly:
        for deg in degrees:
            kernels.append(get_ploynomial_kernel(X, Y, deg=deg))

    if gauss:
        for w in widths:
            kernels.append(get_gaussian_kernel(X, Y, width=w))


    if distk:
        kernels.append(get_distances_to_kernel(X))


    if feat_kernel and poly:
        for deg in degrees:
            for i in range(X.shape[1]):
                kernels.append(get_ploynomial_kernel(X[:,i,np.newaxis], Y[:,i,np.newaxis], deg=deg))

    if feat_kernel and gauss:
        for w in widths:
            for i in range(X.shape[1]):
                kernels.append(get_gaussian_kernel(X[:,i,np.newaxis], Y[:,i,np.newaxis], width=w))

    return arr(kernels)


def get_kernels2(X, Y=None, poly=False, max_poly_deg=3, gauss=False, max_gauss_width=6, min_gauss_width=-3,
 distk=False, feat_kernel=False):
    kernels = []
    msk = np.isnan(X)
    X[msk] = 0.

    if Y is None:
        Y = X

    if poly:
        degrees = np.arange(1,max_poly_deg+1)

        for deg in degrees:
            kernels.append(get_ploynomial_kernel(X, Y, deg=deg))

            if feat_kernel:
                for i in range(X.shape[1]):
                    kernels.append(get_ploynomial_kernel(X[:,i,np.newaxis], Y[:,i,np.newaxis], deg=deg))


    if gauss:
        widths = map(lambda x: 2.**x, np.arange(min_gauss_width,max_gauss_width+1))

        for w in widths:
            kernels.append(get_gaussian_kernel(X, Y, width=w))

            if feat_kernel:
                for i in range(X.shape[1]):
                    kernels.append(get_gaussian_kernel(X[:,i,np.newaxis], Y[:,i,np.newaxis], width=w))


    if distk:
        kernels.append(get_distances_to_kernel(X))

    return arr(kernels)



'''
# splits
# images
# distances matrix
DATA_DIR = "../data"

# DATASET1: 17 Category Flowers
# 'val1', 'val2', 'val3', 'trn1', 'trn2', 'trn3', 'tst3', 'tst2', 'tst1'
splits = loadmat("%s/cat_flower/datasplits.mat" %DATA_DIR)

# 'D_siftbdy', 'D_siftint', 'D_hsv', 'D_hog',
distances = loadmat("%s/cat_flower/distancematrices17itfeat08.mat" %DATA_DIR)

imlist = loadmat("%s/cat_flower/trimaps/imlist.mat" %DATA_DIR)


# DATASET2: 102 Category Flowers
# 'val1', 'val2', 'val3', 'trn1', 'trn2', 'trn3', 'tst3', 'tst2', 'tst1'
ds2_splits = loadmat("%s/102_cat_flower/setid.mat" %DATA_DIR)

# 'D_siftbdy', 'D_siftint', 'D_hsv', 'D_hog',
ds2_distances = loadmat("%s/102_cat_flower/distancematrices102.mat" %DATA_DIR)

ds2_labels = loadmat("%s/102_cat_flower/imagelabels.mat" %DATA_DIR)
'''
