import unittest

import numpy as np
import pandas as pd

from airline_satisfaction.preprocessing import (
    CATEGORICAL, DELAY, INDICATOR, TreePreprocessor, CatBoostPreprocessor, make_linear_preprocessor,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.fit = pd.DataFrame({
            'Age': [20, 40, 60], DELAY: [0., 10., np.nan],
            'Inflight wifi service': [0, 1, 5],
            **{c: ['a', 'b', 'a'] for c in CATEGORICAL},
        })
        self.valid = self.fit.iloc[[0, 1]].copy()
        self.valid[DELAY] = [np.nan, 1000.]
        self.valid.loc[0, 'Gender'] = 'unseen'

    def test_tree_uses_fit_median_and_preserves_zero_and_inputs(self):
        original = self.valid.copy(deep=True)
        preprocessor = TreePreprocessor().fit(self.fit)
        result = preprocessor.transform(self.valid)
        self.assertEqual(result.loc[0, DELAY], 5.)
        self.assertEqual(result.loc[1, DELAY], 1000.)
        self.assertEqual(result.loc[0, 'Inflight wifi service'], 0)
        self.assertEqual(result[INDICATOR].tolist(), [1, 0])
        self.assertTrue(pd.isna(result.loc[0, 'Gender']))
        self.assertEqual(list(result['Gender'].cat.categories), ['a', 'b'])
        pd.testing.assert_frame_equal(self.valid, original)

    def test_linear_fit_statistics_and_unknown_category(self):
        preprocessor = make_linear_preprocessor(self.fit.columns).fit(self.fit)
        imputer = preprocessor.named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(imputer.statistics_[1], 5.)
        transformed = preprocessor.transform(self.valid)
        self.assertTrue(np.isfinite(transformed.data).all())
        names = preprocessor.get_feature_names_out()
        gender_columns = [i for i, name in enumerate(names) if name.startswith('categorical__Gender_')]
        self.assertEqual(transformed[0, gender_columns].sum(), 0)

    def test_schema_mismatch_and_identifier_are_rejected(self):
        preprocessor = TreePreprocessor().fit(self.fit)
        with self.assertRaises(ValueError):
            preprocessor.transform(self.valid[self.valid.columns[::-1]])
        with self.assertRaises(ValueError):
            TreePreprocessor().fit(self.fit.assign(id=1))

    def test_catboost_categories_are_strings_with_unknown_sentinel(self):
        result = CatBoostPreprocessor().fit(self.fit).transform(self.valid)
        self.assertEqual(result.loc[0, 'Gender'], '__MISSING__')
        self.assertEqual(result.loc[1, 'Gender'], 'b')
        self.assertEqual(result.loc[0, DELAY], 5.)
        for c in CATEGORICAL:
            self.assertFalse(result[c].isna().any())
            self.assertTrue(all(isinstance(v, str) for v in result[c]))


if __name__ == '__main__':
    unittest.main()
