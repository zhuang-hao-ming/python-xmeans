import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from xmeans import xmeans


def generate_data(n, d, k, sigma=0.1):
    data = np.empty((n, d))
    distributions = [ {'mean': np.random.rand(d), 'cov': np.eye(d) * np.random.rand() * sigma} for i in xrange(k) ]
    for i in xrange(n):
        params = random.choice(distributions)
        data[i, :] = np.random.multivariate_normal(**params)
    return data, distributions

if __name__ == '__main__':
    n = 10000
    d = 2
    k = 32

    data, actual = generate_data(n, d, k, sigma=0.0001)
    actual_data = np.asarray([item['mean'] for item in actual])
    actual_var = np.asarray([item['cov'][0,0] for item in actual])

    # print actual_var
    # result = KMeans(n_clusters=2).fit(data)
    # cluster_center = result.cluster_centers_

    result = xmeans(2, 40).fit(data)
    cluster_center = result.cluster_centers_
    print('cluster num {}'.format(len(cluster_center)))

    plt.figure()
    plt.title('xmeans')
    plt.scatter(data[:, 0], data[:, 1], alpha=0.25, label='data')
    plt.scatter(actual_data[:, 0], actual_data[:, 1], c='r', s=125, alpha=0.6, label='actual center')

    plt.scatter(cluster_center[:, 0], cluster_center[:, 1], c='g', s=75, alpha=0.4, label='sklearn')

    plt.legend()
    plt.tight_layout()
    plt.show()