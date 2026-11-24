import numpy as np

from common import CAT_FEATURES


def raw_feature_name(transformed_name):
    prefix, sep, rest = transformed_name.partition("__")

    if not sep:
        return transformed_name

    if prefix == "cat":
        for name in sorted(
            CAT_FEATURES,
            key=len,
            reverse=True,
        ):
            if rest.startswith(name + "_"):
                return name

    return rest


def aggregate_contributions(contributions, feature_names):
    contributions = np.asarray(contributions)

    if contributions.ndim != 2:
        raise ValueError("Expected a 2D contribution matrix")

    if contributions.shape[1] != len(feature_names):
        raise ValueError("Contribution and feature count mismatch")

    grouped = {}

    for i, name in enumerate(feature_names):
        raw = raw_feature_name(name)

        if raw not in grouped:
            grouped[raw] = np.zeros(
                contributions.shape[0],
                dtype=float,
            )

        grouped[raw] += contributions[:, i]

    scores = {
        name: float(np.mean(np.abs(values)))
        for name, values in grouped.items()
    }

    return dict(sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ))
