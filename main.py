"""
main.py
    main thread of execution

    @author Nicholas Nordstrom
"""
import matplotlib.pyplot as plt
import mglearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from KMeansCluster import KMeansCluster

URL = "https://raw.githubusercontent.com/gchoi/Dataset/master/OldFaithful.csv"


def main():
    # Import the data
    df = pd.read_csv(URL)
    # print(df.columns)
    # print(df)

    X = df.to_numpy()
    print(X.shape)

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1])

    # Standardize the data
    X_scale = StandardScaler().fit_transform(X)

    # Run local implementation of kmeans
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_scale)
    y_pred = kmeans.predict(X_scale)
    print(y_pred.shape)
    print(y_pred)

    # Plot the clustered data
    plt.scatter(X_scale[:, 0], X_scale[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2,
                edgecolor='k')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    # Run custom implementation of kmeans
    kmeans = KMeansCluster(2)
    kmeans.fit(X_scale)
    y_pred = kmeans.predict(X_scale)
    print(y_pred.shape)
    print(y_pred)

    # Plot the clustered data
    plt.scatter(X_scale[:, 0], X_scale[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2,
                edgecolor='k')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")


if __name__ == '__main__':
    main()
