"""Test K-means distance calculations, training, and prediction."""

from lab_01.generator_data import generate_data
from lab_01.model import KMeans, squared_distance
from lab_01.elbow import find_elbow


def test_squared_distance() -> None:
    """Check squared distance on points with a known result."""
    result = squared_distance([1.0, 2.0], [4.0, 6.0])

    assert result == 25.0


def test_finds_three_clusters() -> None:
    """Check that each generated group forms a separate cluster."""
    points_per_cluster = 20
    centers = [
        [-10.0, -10.0],
        [0.0, 10.0],
        [10.0, -10.0],
    ]

    points: list[list[float]] = []
    for i, center in enumerate(centers):
        points.extend(
            generate_data(
                points_per_cluster,
                i + 1,
                [center],
                0.3,
            )
        )

    labels = KMeans(3, 100, 1).fit(points)

    assert len(labels) == 3 * points_per_cluster

    found: list[int] = []
    for i in range(3):
        group = labels[i * points_per_cluster : (i + 1) * points_per_cluster]
        assert len(set(group)) == 1
        found.append(group[0])

    assert len(set(found)) == 3


def test_same_seed_same_result() -> None:
    """Check that identical data and seeds produce identical labels and centers."""
    points = [[5.2, 6.7], [2.5, 7.6]]
    model_1 = KMeans(2, 100, 300)
    model_2 = KMeans(2, 100, 300)
    labels_1 = model_1.fit(points)
    labels_2 = model_2.fit(points)

    assert labels_1 == labels_2
    assert model_1.centers == model_2.centers


def test_inertia_doesnt_increase() -> None:
    """Check that inertia does not increase with a higher iteration limit."""
    points = generate_data(100, 322, spread=1.2)
    iterations_limit = [1, 2, 5, 10, 25]
    inertias = []
    for iterations in iterations_limit:
        model = KMeans(3, iterations, 322)
        labels = model.fit(points)
        inertias.append(model.inertia(points, labels))

    for i in range(len(inertias) - 1):
        assert inertias[i + 1] <= inertias[i] + 1e-9


def test_different_k_different_cnt_clusters() -> None:
    """Check that different k values produce the requested number of clusters."""
    points = [
        [-10.0, -10.0],
        [-10.0, -9.0],
        [0.0, 10.0],
        [1.0, 10.0],
        [10.0, -10.0],
        [10.0, -9.0],
    ]
    for k in range(1, len(points) + 1):
        model = KMeans(k, 100, 229)
        labels = model.fit(points)
        assert len(model.centers) == k
        assert len(set(labels)) == k


def test_one_cluster_center_and_inertia() -> None:
    """Check the center and inertia against hand-calculated values."""
    points = [[0.0, 0.0], [2.0, 0.0]]
    model = KMeans(1, 10, 0)

    labels = model.fit(points)

    assert labels == [0, 0]
    assert model.centers == [[1.0, 0.0]]
    assert model.inertia(points, labels) == 2.0


def test_predict() -> None:
    """Check that new points receive the labels of their nearest fitted clusters."""
    points = [
        [0.0, 0.0],
        [0.0, 2.0],
        [10.0, 0.0],
        [10.0, 2.0],
    ]
    model = KMeans(2, 100, 42)
    labels = model.fit(points)
    new_points = [
        [1.0, 1.0],
        [9.0, 1.0],
    ]
    predicted = model.predict(new_points)
    assert labels[0] != labels[2]
    assert predicted == [labels[0], labels[2]]


def test_elbow_returns_expected_inertias() -> None:
    """Check elbow values for one and two clusters."""
    points = [[0.0], [2.0]]

    assert find_elbow(points, max_k=2, max_iter=10, seed=0) == [2.0, 0.0]
