import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from airline_satisfaction.baseline import positive_probabilities
from airline_satisfaction.cross_validation import make_folds
from airline_satisfaction.features import RATINGS
from airline_satisfaction.final_model import train_final
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY, TARGET


def tiny_frames(rows=60):
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        'Age': rng.integers(20, 70, rows),
        'Flight Distance': rng.integers(100, 3000, rows),
        'Departure Delay in Minutes': rng.integers(0, 10, rows),
        DELAY: rng.integers(0, 10, rows).astype(float),
        **{c: rng.integers(0, 6, rows) for c in RATINGS},
        **{c: rng.choice(['a', 'b'], rows) for c in CATEGORICAL},
    })
    features.loc[0, DELAY] = np.nan
    train = features.assign(id=np.arange(rows),
                            **{TARGET: pd.Series([False, True] * (rows // 2))})
    test = features.iloc[:15].reset_index(drop=True).assign(id=np.arange(1000, 1015))
    submission = pd.DataFrame({'id': test['id'], TARGET: 0.5})
    return train, test, submission


def small_factory(seed):
    return LGBMClassifier(n_estimators=10, num_leaves=3, min_child_samples=5,
                          random_state=seed, n_jobs=1, verbosity=-1)


class FinalModelTests(unittest.TestCase):
    def test_saved_pipelines_reproduce_oof_and_submission_is_valid(self):
        train, test, submission = tiny_frames()
        folds = make_folds(train, n_splits=3, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, result = train_final(train, test, submission, folds, root,
                                          seeds=(42,), factory=small_factory, log=lambda _: None)
            self.assertEqual(summary['folds'], 3)
            self.assertEqual(summary['seeds'], [42])
            self.assertEqual(len(result), len(test))
            self.assertTrue(result['id'].equals(submission['id']))
            self.assertTrue(result[TARGET].between(0, 1).all())
            oof = pd.read_csv(root / 'oof.csv')
            self.assertTrue(np.isfinite(oof['probability']).all())
            for fold in sorted(folds.fold.unique()):
                pipeline = joblib.load(root / 'models' / f'fold_{fold}.joblib')
                rows = folds['fold'].eq(fold).to_numpy()
                expected = positive_probabilities(pipeline, train.loc[rows].drop(columns=['id', TARGET]))
                np.testing.assert_allclose(oof.loc[rows, 'probability'], expected, rtol=1e-6)

    def test_multiple_seeds_average_and_stay_aligned(self):
        train, test, submission = tiny_frames()
        folds = make_folds(train, n_splits=3, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            summary, result = train_final(train, test, submission, folds, Path(tmp),
                                          seeds=(42, 1042), factory=small_factory,
                                          log=lambda _: None)
            self.assertEqual(summary['seeds'], [42, 1042])
            self.assertEqual(result.shape, submission.shape)
            self.assertTrue((Path(tmp) / 'models' / 'oof.csv').exists())
            self.assertTrue((Path(tmp) / 'summary.json').exists())

    def test_duplicate_seeds_are_rejected(self):
        train, test, submission = tiny_frames()
        folds = make_folds(train, n_splits=3, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                train_final(train, test, submission, folds, Path(tmp), seeds=(42, 42),
                            factory=small_factory, log=lambda _: None)


if __name__ == '__main__':
    unittest.main()
