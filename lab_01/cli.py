"""Provide a command-line interface for K-means clustering and elbow analysis."""

import argparse

from lab_01.elbow import find_elbow
from lab_01.generator_data import generate_data
from lab_01.model import KMeans
from lab_01.visualization import graph_clusters, graph_elbow


def main() -> None:
    """Run clustering or elbow analysis using command-line arguments."""
    parser = argparse.ArgumentParser(description="K-means")
    parser.add_argument(
        "--samples", type=int, default=300, help="Number of generated points"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument(
        "--iter", type=int, default=100, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--elbow", action="store_true", help="Show the elbow method plot"
    )
    parser.add_argument(
        "--max-k", type=int, default=8, help="Maximum number of clusters in elbow mode"
    )

    args = parser.parse_args()
    try:
        points = generate_data(args.samples, args.seed)
        if not args.elbow:
            model = KMeans(args.k, args.iter, args.seed)
            labels = model.fit(points)
            for i in range(args.k):
                count = labels.count(i)
                print(f"Cluster {i} has {count} points")
            print(f"Inertia: {model.inertia(points, labels)}")
            graph_clusters(points, labels, model.centers)
        else:
            inertias = find_elbow(points, args.max_k, args.iter, args.seed)
            graph_elbow(inertias)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
