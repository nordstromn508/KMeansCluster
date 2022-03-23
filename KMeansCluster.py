"""
KMeansCluster.py
    Implementation of k-means clustering algorithm

    @author Nicholas Nordstrom
"""
import numpy as np

NUM_DIMENSIONS = 2


def get_new_centers(c1, c2):
    return np.mean(c1, axis=0), np.mean(c2, axis=0)


class KMeansCluster:

    def __init__(self, n_clusters: int):
        self.cluster_centers_ = None
        self.n_clusters = n_clusters

    def cluster_points(self, X):
        c0 = []
        c1 = []
        labels = []
        for x in X:
            d0 = np.linalg.norm(self.cluster_centers_[0] - x)
            d1 = np.linalg.norm(self.cluster_centers_[1] - x)
            if d0 > d1:
                c0.append(x)
                labels.append(0)
            else:
                c1.append(x)
                labels.append(1)
        return c0, c1, labels

    def center_needs_update(self, center0, center1):
        if np.array_equal(self.cluster_centers_[0], center0):
            if np.array_equal(self.cluster_centers_[1], center1):
                return True
        return False

    def fit(self, X: np.ndarray):
        self.cluster_centers_ = np.random.randint(np.min(X), np.max(X), (self.n_clusters, NUM_DIMENSIONS))
        cluster0, cluster1, _ = self.cluster_points(X)
        center0, center1 = get_new_centers(cluster0, cluster1)

        while self.center_needs_update(center0, center1):
            self.cluster_centers_[0] = center0
            self.cluster_centers_[1] = center1

            cluster0, cluster1, _ = self.cluster_points(X)
            center0, center1 = get_new_centers(cluster0, cluster1)

    def predict(self, X):
        _, _, labels = self.cluster_points(X)
        return np.array(labels)
