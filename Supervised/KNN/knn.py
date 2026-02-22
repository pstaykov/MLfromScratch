import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle


def generate_data(n=100, n_classes=2, seed=None):
    """Generate random 2D data and integer labels.

    Returns a pandas DataFrame with columns ['x', 'y', 'label'].
    """
    if seed is not None:
        np.random.seed(seed)
    xy = np.random.rand(n, 2)
    data = pd.DataFrame(xy, columns=["x", "y"])
    data["label"] = np.random.randint(0, n_classes, size=len(data))
    return data


def euclidean(p1, p2):
    """Compute Euclidean distance between two 2D points (x, y)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def knn_predict(datapoint, data, k=5):
    """Compute k-NN for a single query point against a DataFrame.

    Returns a tuple: (predicted_label, knn_list, knn_indices, radius)
    - knn_list is a list of (distance, label, index) sorted by distance
    - knn_indices is a list of the indices of the k nearest neighbors
    - radius is the distance to the k-th neighbor (useful for plotting)
    """
    distances = []
    for i in range(len(data)):
        distance = euclidean(datapoint, [data.loc[i, "x"], data.loc[i, "y"]])
        distances.append((distance, int(data.loc[i, "label"]), i))

    distances = sorted(distances, key=lambda x: x[0])
    knn = distances[:k]

    # majority vote
    label_counts = {}
    for _, label, _ in knn:
        label_counts[label] = label_counts.get(label, 0) + 1
    predicted = max(label_counts, key=label_counts.get)

    radius = knn[-1][0] if len(knn) > 0 else 0.0
    knn_indices = [idx for _, _, idx in knn]
    return predicted, knn, knn_indices, radius


def plot_knn(data, datapoint, knn_indices, radius, k=5, save_path=None, show=True):
    """Plot the dataset, query point, k nearest neighbors and the enclosing circle.

    If save_path is provided the figure is written to that file. If show is True
    plt.show() is called (useful for interactive sessions). By default the function
    will save the figure to a file when show is False.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(data["x"], data["y"], c=data["label"], cmap="coolwarm", s=30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Random 2D data with labels and k-NN circle")

    # Ensure the circle is not stretched: use equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')

    # Compute limits so the full circle is visible (add a small margin)
    margin = radius * 0.15 if radius > 0 else 0.1
    xmin = min(data["x"].min(), datapoint[0] - radius) - margin
    xmax = max(data["x"].max(), datapoint[0] + radius) + margin
    ymin = min(data["y"].min(), datapoint[1] - radius) - margin
    ymax = max(data["y"].max(), datapoint[1] + radius) + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.scatter(datapoint[0], datapoint[1], c="red", s=100, label="query point")

    circle = Circle((datapoint[0], datapoint[1]), radius, fill=False, edgecolor="green", linewidth=2, alpha=0.8)
    ax.add_patch(circle)

    if len(knn_indices) > 0:
        ax.scatter(data.loc[knn_indices, "x"], data.loc[knn_indices, "y"], facecolors='none', edgecolors='black', s=120, linewidths=1.5, label=f'{k} nearest')

    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Plot saved to: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    data = generate_data(n=100, n_classes=2, seed=None)
    print(data.head())

    datapoint = [0.5, 0.5]

    predicted, knn_list, knn_indices, radius = knn_predict(datapoint, data, k=5)
    print('k-NN (distance, label, index):', knn_list)
    print(f'datapoint belongs to class {predicted}')

    # Save plot instead of blocking on plt.show() to make non-interactive runs safe.
    plot_knn(data, datapoint, knn_indices, radius, k=5, save_path='knn_plot.png', show=False)
