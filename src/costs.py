import numpy as np

COST_FN = 200
COST_FP = 5


def expected_cost(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    pred = proba >= threshold

    fn = np.sum((y_true == 1) & (~pred))
    fp = np.sum((y_true == 0) & pred)

    return float(COST_FN * fn + COST_FP * fp)


def best_threshold(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    thresholds = np.asarray(thresholds)

    costs = [
        expected_cost(y_true, proba, t)
        for t in thresholds
    ]

    idx = int(np.argmin(costs))

    return float(thresholds[idx]), float(costs[idx])
