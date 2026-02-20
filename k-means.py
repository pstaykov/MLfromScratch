import numpy as np

data = np.random.randint(0, 100, size=(10000, 2))


def kmeans(data, k, max_iter=100, seed=None):
    """
    :param data: array-like of shape (n_samples, n_features)
    :param k: number of clusters
    :param max_iter: number of iterations to run the algorithm
    :param seed: random seed for reproducibility
    :return: (centroids, clusters)
             centroids: ndarray of shape (k, n_features)
             clusters: list of ndarrays, each containing the points assigned to that centroid
    """
    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_samples, n_features), got shape {data.shape}")
    n_samples = data.shape[0]
    if not (1 <= k <= n_samples):
        raise ValueError(f"k must be between 1 and n_samples ({n_samples}), got {k}")

    # initialize centroids by sampling k distinct points from the dataset
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = data[init_idx].copy()

    for _ in range(max_iter):
        # Assign each point to the nearest centroid
        # distances: (n_samples, k)
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Recompute centroids
        new_centroids = centroids.copy()
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)
            else:
                # If a cluster becomes empty, reinitialize its centroid to a random point
                new_centroids[i] = data[rng.integers(0, n_samples)]

        # Stop early if centroids don't change
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    clusters = [data[labels == i] for i in range(k)]
    return centroids, clusters


if __name__ == "__main__":
    centroids, clusters = kmeans(data, 3, 10, seed=42)
    print("Centroids:\n", centroids)
    print("Cluster sizes:", [len(c) for c in clusters])





