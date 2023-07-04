from scipy import stats
import numpy as np


def ttest(
    scores: np.array,
    alternative: str = "two-sided",
):
    return stats.ttest_1samp(scores, 0, alternative=alternative)


def wilcoxon(
    scores: np.array,
    alternative: str = "two-sided",
):
    return stats.wilcoxon(scores, alternative=alternative)


def bootstrap(
    scores: np.array,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
):
    assert 0 < confidence_level < 1

    res = stats.bootstrap((scores,), np.mean, confidence_level=confidence_level)

    if res.confidence_interval[0] > 0:
        if alternative == "two-sided" or alternative == "greater":
            return (
                res,
                f"H0 rejected for signficance level {1 - confidence_level}",
            )
    if res.confidence_interval[1] < 0:
        if alternative == "two-sided" or alternative == "less":
            return (
                res,
                f"H0 rejected for signficance level {1 - confidence_level}",
            )
    else:
        return (
            res,
            f"H0 NOT rejected for signficance level {1 - confidence_level}",
        )


def permutation(
    scores_1: np.array,
    scores_2: np.array,
    alternative: str = "two-sided",
):
    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    try:
        return stats.permutation_test(
            (scores_1, scores_2),
            statistic,
            alternative=alternative,
            vectorized=True,
            permutation_type="samples",
        )
    except ValueError:
        print("Permutation test failed, returning None")
        return None


def h_test(
    test_type: str = "ttest",
    scores_1: np.array = None,
    scores_2: np.array = None,
    alternative: str = "two-sided",
    confidence_level: float = None,
):
    """
    Perform a hypothesis test on the given scores.
    Is for two paired samples meaning they have equal length and the indexes match to the subsamples.
    """
    scores = scores_1 - scores_2

    if test_type in ["ttest", "wilcoxon"]:
        return eval(f"{test_type}(scores, alternative)")
    elif test_type == "bootstrap":
        return bootstrap(scores, alternative, confidence_level)
    elif test_type == "permutation":
        return permutation(scores_1, scores_2, alternative)
    else:
        raise ValueError("Unknown test type")
