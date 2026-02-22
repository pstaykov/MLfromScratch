# this file was written by AI

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from CART_Classification import build_tree, predict, LeafNode, InternalNode

# ---------- Data ----------
# Columns: Temperature, Humidity, GoForRun
data = np.array([
    [85, 85, 0],
    [80, 90, 0],
    [83, 78, 0],
    [70, 96, 0],
    [68, 80, 1],
    [65, 70, 1],
    [64, 65, 1],
    [72, 95, 0],
    [69, 70, 1],
    [75, 80, 1],
    [75, 70, 1],
    [72, 90, 1],
    [81, 75, 1],
    [71, 80, 0],
], dtype=float)

feature_names = ["Temperature", "Humidity"]
label_names = {0: "No Run", 1: "Go Run"}

# ---------- Build tree ----------
tree = build_tree(data, current_depth=0, max_depth=5)

# ---------- Test predictions ----------
print("=== Predictions on training data ===")
for row in data:
    pred = predict(row[:-1], tree)
    actual = int(row[-1])
    status = "✓" if int(pred) == actual else "✗"
    print(f"  Temp={int(row[0])}, Hum={int(row[1])}  →  predicted={label_names[int(pred)]}, actual={label_names[actual]}  {status}")

# ---------- Visualise the tree ----------

def _get_tree_depth(node):
    if isinstance(node, LeafNode):
        return 0
    return 1 + max(_get_tree_depth(node.left_child), _get_tree_depth(node.right_child))


def _count_leaves(node):
    if isinstance(node, LeafNode):
        return 1
    return _count_leaves(node.left_child) + _count_leaves(node.right_child)


def _draw_node(ax, node, x, y, dx, dy):
    """Recursively draw the tree nodes and edges."""
    box_kwargs_internal = dict(
        boxstyle="round,pad=0.4", facecolor="#AED6F1", edgecolor="#2C3E50", linewidth=1.5
    )
    box_kwargs_leaf = dict(
        boxstyle="round,pad=0.4", facecolor="#ABEBC6", edgecolor="#1E8449", linewidth=1.5
    )

    if isinstance(node, LeafNode):
        ax.text(x, y, f"{label_names[int(node.label)]}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=box_kwargs_leaf)
        return

    # Internal node text
    text = f"{feature_names[node.feature_idx]} ≤ {node.split_value:.1f}"
    ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=box_kwargs_internal)

    # Child positions
    left_x = x - dx
    right_x = x + dx
    child_y = y - dy

    # Edges
    ax.annotate("", xy=(left_x, child_y + 0.04), xytext=(x, y - 0.04),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))
    ax.annotate("", xy=(right_x, child_y + 0.04), xytext=(x, y - 0.04),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    # Edge labels
    ax.text((x + left_x) / 2 - 0.01, (y + child_y) / 2, "Yes",
            fontsize=8, color="#1A5276", ha="right")
    ax.text((x + right_x) / 2 + 0.01, (y + child_y) / 2, "No",
            fontsize=8, color="#922B21", ha="left")

    # Draw children
    _draw_node(ax, node.left_child, left_x, child_y, dx / 2, dy)
    _draw_node(ax, node.right_child, right_x, child_y, dx / 2, dy)


depth = _get_tree_depth(tree)
leaves = _count_leaves(tree)
fig_w = max(10, leaves * 2.5)
fig_h = max(6, (depth + 1) * 2.5)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("CART Classification Tree – GoForRun", fontsize=14, fontweight="bold", pad=20)

dy = 0.8 / max(depth, 1)
_draw_node(ax, tree, x=0.5, y=0.92, dx=0.25, dy=dy)

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#AED6F1", edgecolor="#2C3E50", label="Internal node (split)"),
    mpatches.Patch(facecolor="#ABEBC6", edgecolor="#1E8449", label="Leaf node (prediction)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("cart_tree.png", dpi=150, bbox_inches="tight")
print("\nTree visualisation saved to cart_tree.png")
plt.show()

