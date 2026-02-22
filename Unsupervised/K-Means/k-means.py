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


def plot_kmeans(centroids, clusters, out_path='kmeans.png', show=False, figsize=(8, 6), cmap='tab10'):
    """Visualize 2D clusters produced by kmeans and save to a PNG.

    Parameters
    - centroids: array-like of shape (k, n_features)
    - clusters: iterable of array-like; each element contains points assigned to that centroid
    - out_path: str path to save the PNG
    - show: if True, call plt.show() after saving
    - figsize: tuple passed to plt.subplots
    - cmap: matplotlib colormap name for cluster coloring

    Notes
    - This function performs inline imports of matplotlib to avoid adding a hard dependency at import time.
    - Expects 2D data (n_samples, 2). If data has more than 2 features, only the first two will be plotted.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise ImportError("plot_kmeans requires matplotlib and numpy. Install with: pip install matplotlib numpy") from e

    centroids = np.asarray(centroids)
    k = len(clusters)

    fig, ax = plt.subplots(figsize=figsize)

    # get colormap
    cmap_obj = plt.get_cmap(cmap)

    for i, pts in enumerate(clusters):
        pts = np.asarray(pts)
        if pts.size == 0:
            # skip empty cluster
            continue
        # ensure points are at least 2D and plot first two dims
        if pts.ndim == 1:
            # single point
            x = np.atleast_1d(pts[0])
            y = np.atleast_1d(pts[1]) if pts.size > 1 else np.zeros_like(x)
            ax.scatter(x, y, s=10, color=cmap_obj(i % cmap_obj.N), alpha=0.6, label=f'cluster {i}')
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color=cmap_obj(i % cmap_obj.N), alpha=0.6, label=f'cluster {i}')

    # plot centroids
    if centroids.size > 0:
        c = centroids
        if c.ndim == 1:
            cx = np.atleast_1d(c[0])
            cy = np.atleast_1d(c[1]) if c.size > 1 else np.zeros_like(cx)
            ax.scatter(cx, cy, s=150, marker='X', color='black', edgecolor='white', linewidth=0.8, label='centroids')
        else:
            ax.scatter(c[:, 0], c[:, 1], s=150, marker='X', color='black', edgecolor='white', linewidth=0.8, label='centroids')

    ax.set_title('k-means clustering')
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.legend(loc='best', markerscale=2)
    plt.tight_layout()

    # save
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    centroids, clusters = kmeans(data, 3, 10, seed=42)
    print("Centroids:\n", centroids)
    print("Cluster sizes:", [len(c) for c in clusters])
