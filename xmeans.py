# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class xmeans:
    def __init__(self, kmin, kmax):
        self.kmin = kmin
        self.kmax = kmax
        self.init = 'k-means++'

    def fit(self, data):

        cluster_centers = []
        k = self.kmin
        cnt = 0
        while k <= self.kmax:
            cnt += 1
            kmeans = self._fit(k, data, cluster_centers)

            centroids = kmeans.cluster_centers_

            centroid_distances = euclidean_distances(centroids) # 计算聚类中心的距离
            centroid_distances += np.diag([np.Infinity] * k) #
            min_centroid_distances = centroid_distances.min(axis = -1) # 聚类中心到最近的聚类中心的距离

            labels = kmeans.labels_

            cluster_centers = []
            for i, centroid in enumerate(centroids):
                direction = np.random.random(centroid.shape)

                vector = direction / np.sqrt(np.dot(direction,direction)) * min_centroid_distances[i]

                new_point1 = centroid + vector
                new_point2 = centroid - vector

                label_index = (labels == i)
                points = data[label_index]



                new_kmeans = self._fit(2, points, np.asarray([new_point1, new_point2]))
                new_labels = new_kmeans.labels_
                cluster1 = points[new_labels == 0]
                cluster2 = points[new_labels == 1]

                bic_parent = xmeans.bic([points], [centroid])
                bic_child = xmeans.bic([cluster1, cluster2], new_kmeans.cluster_centers_)

                if bic_child > bic_parent:
                    cluster_centers.extend(new_kmeans.cluster_centers_)
                else:
                    cluster_centers.append(centroid)

            if k==len(cluster_centers):
                break # 聚类数不变
            k = len(cluster_centers)
        print('count of iteration: {}'.format(k))
        return self._fit(k, data, cluster_centers)




    def _fit(self, k, data, centroids):
        if len(centroids) == 0:
            centroids = self.init
        else:
            centroids = np.asarray(centroids)
        result = KMeans(k, init=centroids).fit(data)
        return  result

    @classmethod
    def bic(cls, clusters, centroids):
        R = sum([len(cluster) for cluster in clusters]) # 点的总数目
        M = clusters[0][0].shape[0] # 点的维度
        K = len(centroids) # 类别数
        log_likelihood = xmeans._log_likelihood(R, M, clusters, centroids)
        num_params = xmeans._free_params(K, M)
        return log_likelihood - num_params / 2.0 * np.log(R)

    @classmethod
    def _log_likelihood(cls, R, M, clusters, centroids):
        ll = 0
        var = xmeans._variance(R, M, clusters, centroids)
        #print('estimate {}'.format(var))
        for cluster in clusters:
            R_n = len(cluster)
            t1 = R_n * np.log(R_n)
            t2 = R_n * np.log(R)
            t3 = R_n * M / 2.0 * np.log(2.0 * np.pi * var)
            t4 = M * (R_n-1.0) / 2.0
            ll += t1 - t2 - t3 - t4
        return ll

    @classmethod
    def _variance(cls, R, M, clusters, centroids):
        K = len(centroids) # 类别数
        denom = float((R - K) * M)
        s = 0
        for cluster, centroid in zip(clusters, centroids):
            distances = euclidean_distances(cluster, [centroid])
            s += (distances * distances).sum()
        return s / denom

    @classmethod
    def _free_params(cls, K, M):
        return K * (M+1)