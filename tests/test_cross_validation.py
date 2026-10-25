import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from airline_satisfaction.cross_validation import (
    make_folds, iter_preprocessed_folds, validate_folds, cross_validate_model,
)
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            'id': np.arange(100, 120), 'satisfaction': [False, True] * 10,
            'Age': np.arange(20, 40), DELAY: [np.nan, 1., 20., 300.] * 5,
            'Inflight wifi service': [0, 1, 2, 3, 5] * 4,
            **{c: ['a', 'b'] * 10 for c in CATEGORICAL},
        })

    def test_reproducibility_coverage_and_stratification(self):
        a = make_folds(self.train)
        pd.testing.assert_frame_equal(a, make_folds(self.train))
        counts = np.zeros(len(self.train), dtype=int)
        for data in iter_preprocessed_folds(self.train, a):
            counts[data['valid_rows']] += 1
            self.assertEqual(len(np.intersect1d(data['fit_rows'], data['valid_rows'])), 0)
            self.assertEqual(data['y_valid'].mean(), .5)
        np.testing.assert_array_equal(counts, 1)

    def test_validation_values_never_enter_preprocessor_fit(self):
        assignments = make_folds(self.train)
        valid = np.flatnonzero(assignments.fold.to_numpy() == 0)
        self.train.loc[valid, DELAY] = 100000.
        self.train.loc[valid, 'Gender'] = 'validation_only'
        fit = self.train.loc[assignments.fold.to_numpy() != 0]
        tree = next(iter_preprocessed_folds(self.train, assignments, 'tree'))
        linear = next(iter_preprocessed_folds(self.train, assignments, 'linear'))
        self.assertEqual(tree['preprocessor'].medians_[DELAY], fit[DELAY].median())
        self.assertNotIn('validation_only', tree['preprocessor'].categories_['Gender'])
        numeric = [c for c in self.train if c not in CATEGORICAL + ['id', 'satisfaction']]
        imputer = linear['preprocessor'].named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(imputer.statistics_[numeric.index(DELAY)], fit[DELAY].median())
        categories = linear['preprocessor'].named_transformers_['categorical'].named_steps['encode'].categories_[0]
        self.assertNotIn('validation_only', categories)

    def test_incompatible_assignment_and_too_many_folds_rejected(self):
        assignments = make_folds(self.train)
        with self.assertRaises(ValueError):
            validate_folds(self.train, assignments.iloc[::-1])
        with self.assertRaises(ValueError):
            make_folds(self.train, n_splits=11)

    def test_probability_scoring_and_oof_alignment(self):
        model = DummyClassifier(strategy='prior')
        result = cross_validate_model(model, self.train, make_folds(self.train))
        self.assertFalse(hasattr(model, 'classes_'))
        self.assertEqual(result['mean_roc_auc'], .5)
        self.assertEqual(result['oof_roc_auc'], .5)
        self.assertEqual(result['std_roc_auc'], 0.)
        self.assertEqual(len(result['fold_scores']), 5)
        pd.testing.assert_series_equal(result['oof']['id'], self.train['id'])
        np.testing.assert_allclose(result['oof']['satisfaction'], .5)


if __name__ == '__main__':
    unittest.main()
