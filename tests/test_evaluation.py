from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from airline_satisfaction.evaluation import (
    metrics, calibration_bins, load_predictions, paired_auc_delta,
)


class EvaluationTests(unittest.TestCase):
    def test_perfect_probabilities_and_confusion_counts(self):
        result = metrics([0, 0, 1, 1], [0., 0., 1., 1.])
        self.assertEqual(result['roc_auc'], 1.)
        self.assertEqual(result['average_precision'], 1.)
        self.assertEqual(result['brier'], 0.)
        self.assertEqual(result['ece_10'], 0.)
        self.assertEqual((result['tn'], result['fp'], result['fn'], result['tp']), (2, 0, 0, 2))
        self.assertEqual(calibration_bins([0, 1], [0., 1.])['n'].sum(), 2)

    def test_fixed_threshold_and_calibration_weighting(self):
        result = metrics([0, 0, 1, 1], [.1, .8, .5, .2])
        self.assertEqual((result['tn'], result['fp'], result['fn'], result['tp']), (1, 1, 1, 1))
        self.assertEqual(result['recall'], .5)
        self.assertAlmostEqual(metrics([0, 0, 0, 1], [.5] * 4)['ece_10'], .25)
        for bad in [[np.nan], [1.1], [-.1]]:
            with self.assertRaises(ValueError):
                metrics([0], bad)

    def test_oof_alignment_and_labels_are_checked(self):
        train = pd.DataFrame({'id': [1, 2], 'satisfaction': [False, True]})
        folds = pd.DataFrame({'id': [1, 2], 'fold': [0, 1]})
        oof = pd.DataFrame({'id': [1, 2], 'fold': [0, 1],
                            'satisfaction': [0, 1], 'probability': [.1, .9]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model').mkdir()
            path = root / 'model/oof.csv'
            oof.to_csv(path, index=False)
            np.testing.assert_allclose(load_predictions(train, folds, root, ['model'])['model'], [.1, .9])
            oof.iloc[::-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_predictions(train, folds, root, ['model'])
            oof['satisfaction'] = [1, 0]
            oof.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_predictions(train, folds, root, ['model'])

    def test_paired_auc_delta_matches_known_shift(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(400), np.ones(400)]
        base = rng.normal(size=800)
        better = base + 1.2 * y - 0.6  # separates the classes more than the baseline
        delta, error = paired_auc_delta(y, base, better)
        self.assertGreater(delta, 0.05)
        self.assertGreater(error, 0)
        same_delta, same_error = paired_auc_delta(y, base, base)
        self.assertEqual(same_delta, 0.)
        self.assertLess(same_error, 1e-8)  # floating-point noise only
        with self.assertRaises(ValueError):
            paired_auc_delta([0, 0, 1], [.1, .2, .3], [.1, .3, .2])


if __name__ == '__main__':
    unittest.main()
