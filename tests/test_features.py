import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from airline_satisfaction.features import FeaturePreprocessor, RATINGS, GROUPS
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'Age': [20, 40, 60], 'Flight Distance': [500, 1000, 2000],
            'Departure Delay in Minutes': [0, 30, 5], DELAY: [0., 10., np.nan],
            **{c: [0, 2, 5] for c in RATINGS},
            **{c: ['a', 'b', 'a'] for c in CATEGORICAL},
        })
        self.frame['Baggage handling'] = [1, 3, 5]

    def test_row_features_and_preserved_zero_values(self):
        original = self.frame.copy(deep=True)
        prep = FeaturePreprocessor(groups=GROUPS).fit(self.frame)
        result = prep.transform(self.frame)
        self.assertEqual(result.loc[0, 'Inflight wifi service'], 0.)
        self.assertEqual(result.loc[0, 'FE_service_zero_count'], 12.)
        self.assertAlmostEqual(result.loc[0, 'FE_service_mean'], 1/13, places=7)
        self.assertEqual(result.loc[2, 'FE_service_high_count'], 13.)
        self.assertEqual(result.loc[1, 'FE_delay_change'], -20.)
        self.assertEqual(result.loc[1, 'FE_delay_recovered'], 20.)
        self.assertEqual(result.loc[1, 'FE_arrival_per_1000_distance'], 10.)
        self.assertTrue(isinstance(result['FE_travel_class'].dtype, pd.CategoricalDtype))
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(list(prep.get_feature_names_out()), list(result.columns))

    def test_transform_uses_fit_statistics_and_categories(self):
        prep = FeaturePreprocessor(groups=GROUPS).fit(self.frame)
        valid = self.frame.iloc[[0]].copy()
        valid[DELAY] = np.nan
        valid['Class'] = 'validation_only'
        result = prep.transform(valid)
        self.assertEqual(result.iloc[0][DELAY], 5.)
        self.assertTrue(pd.isna(result.iloc[0]['FE_travel_class']))
        self.assertNotIn('validation_only', str(prep.engineered_categories_))
        self.assertEqual(clone(prep).groups, GROUPS)

    def test_empty_group_matches_original_preprocessing(self):
        from airline_satisfaction.preprocessing import TreePreprocessor
        actual = FeaturePreprocessor().fit_transform(self.frame)
        expected = TreePreprocessor().fit_transform(self.frame)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
