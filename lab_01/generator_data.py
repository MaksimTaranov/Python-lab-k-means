"""Generate two-dimensional data for clustering."""

import random

from lab_01.model import squared_distance


def generate_centers(
    n_centers: int, generator: random.Random, min_distance: float = 4.0
) -> list[list[float]]:
    """Generate random centers separated by a minimum distance."""
    if n_centers <= 0 or min_distance <= 0:
        raise ValueError("Number of centers or min distance must be positive")
    centers: list[list[float]] = []
    attempt_limit = 1000
    attempt = 0
    while len(centers) != n_centers and attempt < attempt_limit:
        attempt += 1
        candidate = [generator.uniform(-10.0, 10.0), generator.uniform(-10.0, 10.0)]
        is_good = True
        for i in range(len(centers)):
            if squared_distance(candidate, centers[i]) < min_distance**2:
                is_good = False
                break
        if is_good:
            centers.append(candidate)
    if len(centers) != n_centers:
        raise ValueError(
            "Couldn't generate the requested number of centers within the attempt limit"
        )
    return centers


def generate_data(
    n_samples: int,
    seed: int,
    centers: list[list[float]] | None = None,
    spread: float = 0.7,
) -> list[list[float]]:
    """Generate and shuffle points around specified or random centers."""
    generator = random.Random(seed)
    if centers is None:
        centers = generate_centers(3, generator)
    if not centers or any(len(center) != 2 for center in centers):
        raise ValueError(
            "Array with centers is empty or have point with not two coordinates"
        )
    if spread <= 0:
        raise ValueError("Spread must be positive")
    if n_samples < len(centers):
        raise ValueError("Number of samples must be at least the number of centers")
    points = []
    n_for_cent = [n_samples // len(centers) for _ in range(len(centers))]
    remains = n_samples % len(centers)
    ind = 0
    while remains > 0:
        n_for_cent[ind] += 1
        remains -= 1
        ind += 1
    for ind, center in enumerate(centers):
        for _ in range(n_for_cent[ind]):
            points.append(
                [generator.gauss(center[0], spread), generator.gauss(center[1], spread)]
            )
    generator.shuffle(points)
    return points
