import numpy as np
import pandas as pd

EPS = 1e-4


def psi(ref_proportions, cur_proportions):
    ref = np.clip(
        np.asarray(ref_proportions, dtype=float),
        EPS,
        None,
    )

    cur = np.clip(
        np.asarray(cur_proportions, dtype=float),
        EPS,
        None,
    )

    return float(
        np.sum((cur - ref) * np.log(cur / ref))
    )


def psi_numeric(reference, current, bins=10):
    ref = pd.Series(reference).dropna().to_numpy()
    cur = pd.Series(current).dropna().to_numpy()

    edges = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1))
    )

    inner = edges[1:-1]

    ref_idx = np.digitize(ref, inner)
    cur_idx = np.digitize(cur, inner)

    n_bins = len(inner) + 1

    ref_counts = np.bincount(
        ref_idx, minlength=n_bins
    )

    cur_counts = np.bincount(
        cur_idx, minlength=n_bins
    )

    return psi(
        ref_counts / len(ref),
        cur_counts / len(cur),
    )


def psi_categorical(reference, current):
    ref = pd.Series(reference).fillna("__MISSING__")
    cur = pd.Series(current).fillna("__MISSING__")

    categories = sorted(
        set(ref.unique()) | set(cur.unique())
    )

    ref_props = (
        ref.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    cur_props = (
        cur.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    return psi(ref_props, cur_props)
