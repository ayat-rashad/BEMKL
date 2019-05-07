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
    fig.savefig('%s.png' % dataset, dpi=400,
                transparent=True, bbox_inches='tight')


def plot_results(model):
    pass
