"""Calculate inertia values for elbow analysis."""

from lab_01.model import KMeans


def find_elbow(
    points: list[list[float]], max_k: int, max_iter: int, seed: int
) -> list[float]:
    """Calculate inertia for cluster counts from one to max_k."""
    if not points:
        raise ValueError("Array with points is empty")
    if max_k <= 0 or max_k > len(points):
        raise ValueError("Max k must be positive and not greater than number of points")
    results = []
    for k in range(1, max_k + 1):
        model = KMeans(k, max_iter, seed)
        labels = model.fit(points)
        results.append(model.inertia(points, labels))
    return results
