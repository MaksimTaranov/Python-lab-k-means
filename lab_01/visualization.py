"""Visualize K-means clustering results and elbow analysis."""

import matplotlib.pyplot as plt


def graph_clusters(
    points: list[list[float]], labels: list[int], centers: list[list[float]]
) -> None:
    """Display points colored by cluster and mark the cluster centers."""
    if not points or not centers:
        raise ValueError("Array with points or/and centers are empty")
    if len(points) != len(labels):
        raise ValueError("Number of points must equal number of labels")
    xs, ys = [], []
    center_xs, center_ys = [], []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    for center in centers:
        center_xs.append(center[0])
        center_ys.append(center[1])
    plt.figure()
    plt.scatter(xs, ys, c=labels, cmap="tab10", s=30, alpha=0.7)
    plt.scatter(
        center_xs,
        center_ys,
        marker="X",
        s=200,
        c="red",
        edgecolors="black",
        label="Centers",
    )
    plt.title("K-means clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("plots/clusters.png")
    plt.show()


def graph_elbow(inertias: list[float]) -> None:
    """Display inertia values for consecutive cluster counts starting at one."""
    if not inertias:
        raise ValueError("Array with inertia values is empty")
    ks = list(range(1, len(inertias) + 1))
    plt.plot(ks, inertias)
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig("plots/elbow.png")
    plt.show()
