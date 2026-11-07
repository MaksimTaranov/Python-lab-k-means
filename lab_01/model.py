"""Implement K-means clustering using pure Python lists."""

import random


def squared_distance(a: list[float], b: list[float]) -> float:
    """Return the squared Euclidean distance between two points."""
    if len(a) != len(b):
        raise ValueError("Can't find distance because size of points isn't equal")
    dist = 0.0
    for x, y in zip(a, b):
        dist += (x - y) ** 2
    return dist


class KMeans:
    """Group points into clusters by minimizing within-cluster squared distances."""

    def __init__(self, n_clusters: int, max_iter: int, seed: int) -> None:
        """Initialize the model with clustering parameters."""
        if n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        if max_iter <= 0:
            raise ValueError("Number of iterations must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centers: list[list[float]] = []

    def _init_centroids(self, points: list[list[float]]) -> None:
        """Initialize centers by randomly selecting and copying input points."""
        generator = random.Random(self.seed)
        selected = generator.sample(points, self.n_clusters)
        self.centers = []
        for point in selected:
            self.centers.append(point.copy())

    def _closest_center(self, point: list[float]) -> int:
        """Return the index of the closest current cluster center."""
        if not self.centers:
            raise ValueError("Array with centers is empty")
        min_dist = squared_distance(point, self.centers[0])
        ind_min = 0
        for i, center in enumerate(self.centers):
            cur_dist = squared_distance(point, center)
            if min_dist > cur_dist:
                min_dist = cur_dist
                ind_min = i
        return ind_min

    def _assign(self, points: list[list[float]]) -> list[int]:
        """Assign each point to its closest current center."""
        if not self.centers:
            raise ValueError("Array with centers is empty")
        labels = []
        for point in points:
            labels.append(self._closest_center(point))
        return labels

    def _update(
        self, points: list[list[float]], labels: list[int]
    ) -> list[list[float]]:
        """Compute mean centers and preserve centers of empty clusters."""
        if not points or not self.centers:
            raise ValueError(
                "Can't update centers because points or centers don't exist"
            )
        if len(points) != len(labels):
            raise ValueError(
                "Can't update centers because number of points isn't equal number of labels"
            )
        new_centers: list[list[float]] = []
        n_coord = len(points[0])
        groups: list[list[list[float]]] = [[] for _ in self.centers]
        for i, point in zip(labels, points):
            groups[i].append(point)
        for group, old_cent in zip(groups, self.centers):
            if not group:
                new_centers.append(old_cent)
                continue

            center: list[float] = []
            for coord in range(n_coord):
                summ = sum(point[coord] for point in group)
                center.append(summ / len(group))
            new_centers.append(center)
        return new_centers

    def fit(self, points: list[list[float]]) -> list[int]:
        """Find cluster centers and return labels for the input points."""
        if not points:
            raise ValueError("Array with points is empty")
        for point in points:
            if not point:
                raise ValueError("Some point is empty")
        n_coord = len(points[0])
        for point in points:
            if len(point) != n_coord:
                raise ValueError("All points must have the same number of coordinates")
        if self.n_clusters > len(points):
            raise ValueError("Number of clusters is more than number of points")

        self._init_centroids(points)
        tol = 0.0001
        for _ in range(self.max_iter):
            labels = self._assign(points)
            new_centers = self._update(points, labels)
            centers_changed = False
            for new_cent, old_cent in zip(new_centers, self.centers):
                if squared_distance(new_cent, old_cent) > tol**2:
                    centers_changed = True

            self.centers = new_centers
            if not centers_changed:
                break

        labels = self._assign(points)
        return labels

    def predict(self, points: list[list[float]]) -> list[int]:
        """Assign points to fitted clusters without changing the centers."""
        if not self.centers:
            raise ValueError("First, You should train the model by fit")
        return self._assign(points)

    def inertia(self, points: list[list[float]], labels: list[int]) -> float:
        """Return the sum of squared distances to the assigned centers."""
        if not self.centers:
            raise ValueError("Can't find inertia because array with centers is empty")
        if len(points) != len(labels):
            raise ValueError("Number of points isn't equal to number of labels")
        summ = 0.0
        for point, ind_cent in zip(points, labels):
            summ += squared_distance(point, self.centers[ind_cent])
        return summ
