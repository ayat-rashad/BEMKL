from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np


def plot_data(X, y, dataset):
    pca = PCA(n_components=2)
    pc_X = pca.fit_transform(X)

    msk = y == 1
    c = np.repeat('b', y.shape[0])
    c[msk] = 'r'

    fig, ax = plt.subplots(1, 1)
    plt.scatter(pc_X[:, 0], pc_X[:, 1], c=c)
    plt.title('%s 2D' %dataset)
    fig.savefig('%s_2d.png' % dataset, dpi=400,
                transparent=True, bbox_inches='tight')

    # Feature Correlation
    fig = plt.figure()
    M = X.shape[1]
    corr = np.abs(np.corrcoef(X,X, rowvar=False)).astype('float64')[:M,:M]
    plt.imshow(corr, interpolation='none')
    plt.title('%s Feature Correlation' %dataset)
    fig.savefig('%s_feat_corr.png' % dataset, dpi=400,
                transparent=True, bbox_inches='tight')


def plot_kernel(K, fname):
    fig = plt.figure()
    plt.imshow(K, interpolation='none' ) #origin='lower'
    fig.savefig('%s.png' %fname, dpi=400, transparent=True, bbox_inches='tight')


def plot_results(model, fname):
    sparse = "sparse" if model.sparse else "non_sparse"
    data_sparse = "data_sparse" if model.data_sparse else "non_data_sparse"
    mu_b_e = model.mu_b_e
    mu_a = model.mu_a
    cov_b_e = model.cov_b_e

    fig, ax = plt.subplots(1, 1)

    plt.bar(np.arange(mu_b_e.shape[0] -1), mu_b_e[1:])
    plt.title('%s_%s' % (fname, sparse))

    fig.savefig('%s_%s.png' % (fname, sparse), dpi=400,
                transparent=True, bbox_inches='tight')

    fig= plt.figure()

    plt.bar(np.arange(mu_a.shape[0]), mu_a)
    plt.title('%s_%s' % (fname, data_sparse))

    fig.savefig('%s_%s.png' % (fname, data_sparse), dpi=400,
                transparent=True, bbox_inches='tight')
