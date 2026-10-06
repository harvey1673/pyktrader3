import unittest

import numpy as np
import pandas as pd

from tests.rank_spot_price_candidates import correlation_metrics, weekly_change_pair


class SpotPriceCandidateRankingTests(unittest.TestCase):
    def test_positive_tracking_series_scores_above_inverse_series(self):
        dates = pd.bdate_range("2020-01-01", periods=900)
        returns = 0.001 + 0.01 * np.sin(np.arange(len(dates)) / 11.0)
        future = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        tracking = pd.Series(80 * np.exp(np.cumsum(returns * 0.95)), index=dates)
        inverse = pd.Series(80 * np.exp(np.cumsum(-returns)), index=dates)

        tracking_score = correlation_metrics(
            weekly_change_pair(tracking, future)
        )["score"]
        inverse_score = correlation_metrics(
            weekly_change_pair(inverse, future)
        )["score"]
        # The score deliberately applies a history-coverage multiplier.
        self.assertGreater(tracking_score, 0.65)
        self.assertLess(inverse_score, 0.0)
        self.assertGreater(tracking_score, inverse_score)


if __name__ == "__main__":
    unittest.main()
