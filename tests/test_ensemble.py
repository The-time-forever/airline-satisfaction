import unittest

import numpy as np

from airline_satisfaction.ensemble import blend, simplex_grid, candidates, evaluate_blend, fresh_pipeline
from airline_satisfaction.preprocessing import CatBoostPreprocessor, CATEGORICAL
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


class EnsembleTests(unittest.TestCase):
    def test_probability_and_logit_blends(self):
        p = np.array([[.2, .8], [0., 1.]])
        np.testing.assert_allclose(blend(p, [.25, .75]), [.65, .75])
        np.testing.assert_allclose(blend(p, [.5, .5], 'logit'), [.5, .5])
        for weights in [[-1, 2], [.2, .2], [np.nan, 1]]:
            with self.assertRaises(ValueError):
                blend(p, weights)

    def test_rank_blend_handles_ties(self):
        p = np.array([[.1, .9], [.1, .1], [.8, .1]])
        np.testing.assert_allclose(blend(p, [1, 0], 'rank'), [.5, .5, 1])

    def test_grid_and_candidate_uniqueness(self):
        grid = simplex_grid()
        self.assertEqual(len(grid), 66)
        self.assertTrue(all(np.isclose(sum(w), 1) and min(w) >= 0 for w in grid))
        specs = candidates()
        self.assertEqual(len(specs), 71)
        self.assertEqual(len({s['name'] for s in specs}), len(specs))

    def test_foldwise_evaluation_and_no_rank_labels_used(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([[.1], [.9], [.8], [.2]])
        folds = np.array([0, 0, 1, 1])
        spec = {'name': 'rank', 'method': 'rank', 'weights': [1.]}
        result, rows, oof = evaluate_blend(y, p, folds, spec, {0: 1., 1: 0.})
        np.testing.assert_allclose(oof, [.5, 1., 1., .5])
        self.assertEqual(result['mean_roc_auc'], .5)
        self.assertEqual(result['delta_auc'], 0.)
        self.assertEqual(len(rows), 2)

    def test_catboost_reconstruction_preserves_parameters_with_fresh_objects(self):
        original = Pipeline([('preprocessor', CatBoostPreprocessor()),
                             ('model', CatBoostClassifier(iterations=2, cat_features=CATEGORICAL,
                                                         allow_writing_files=False, verbose=False))])
        fresh = fresh_pipeline(original)
        self.assertIsNot(fresh.named_steps['preprocessor'], original.named_steps['preprocessor'])
        self.assertIsNot(fresh.named_steps['model'], original.named_steps['model'])
        self.assertEqual(fresh.named_steps['model'].get_params(), original.named_steps['model'].get_params())


if __name__ == '__main__':
    unittest.main()
