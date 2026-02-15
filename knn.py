import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

# Create 2D feature data
xy = np.random.rand(100, 2)
data = pd.DataFrame(xy, columns=["x", "y"])

# Add a separate label column
data["label"] = np.random.randint(0, 2, size=len(data))

print(data.head())

def euclidean(p1, p2):
    """Compute Euclidean distance between two 2D points (x, y)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

plt.scatter(data["x"], data["y"], c=data["label"], cmap="coolwarm", s=30)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random 2D data with labels")

datapoint = [0.5,0.5]
distances = []

plt.scatter(datapoint[0], datapoint[1], c="red", s=100)
plt.show()


for i in range(len(data)):
    distance = euclidean(datapoint, [data["x"][i], data["y"][i]])
    distances.append((distance, data["label"][i], i))

distances = sorted(distances, key=lambda x: x[0])

k = 5
knn = distances[:k]
print(knn)

label_counts = {}
for _, label, _ in knn:
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1

predicted = max(label_counts, key=label_counts.get)
print(f'datapoint belongs to class {predicted}')

radius = knn[-1][0]

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

knn_indices = [idx for _, _, idx in knn]
ax.scatter(data.loc[knn_indices, "x"], data.loc[knn_indices, "y"], facecolors='none', edgecolors='black', s=120, linewidths=1.5, label=f'{k} nearest')

ax.legend()
plt.show()
