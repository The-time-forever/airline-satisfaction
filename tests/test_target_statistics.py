import unittest

import numpy as np
import pandas as pd

from airline_satisfaction.features import RATINGS
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY
from airline_satisfaction.target_statistics import TargetStatisticsPreprocessor, KEYS


class TargetStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'Age': np.arange(20,40), 'Flight Distance': np.arange(100,120),
            'Departure Delay in Minutes': [0] * 20, DELAY: [0.] * 20,
            **{c: [3] * 20 for c in RATINGS},
            **{c: ['a'] * 20 for c in CATEGORICAL},
        })
        self.y = np.array([0,1] * 10)

    def test_unique_training_keys_do_not_encode_own_label(self):
        prep = TargetStatisticsPreprocessor(smooth=0.)
        crossfit = prep.fit_transform(self.X,self.y)
        np.testing.assert_allclose(crossfit[list(KEYS)],.5)
        direct = prep.transform(self.X)
        np.testing.assert_array_equal(direct['TE_distance'],self.y)
        self.assertEqual(list(prep.get_feature_names_out()),list(crossfit.columns))

    def test_unseen_keys_use_training_prior(self):
        prep = TargetStatisticsPreprocessor().fit(self.X,self.y)
        valid = self.X.iloc[:2].copy()
        valid['Age'] = 999
        valid['Flight Distance'] = 99999
        result = prep.transform(valid)
        np.testing.assert_allclose(result[list(KEYS)],.5)


if __name__ == '__main__':
    unittest.main()
