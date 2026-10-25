import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airline_satisfaction.features import RATINGS
from airline_satisfaction.multimodel import MEMBERS, run_split
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY, TARGET

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional heavy dependency
    HAS_TORCH = False


def tiny_frames(rows=80):
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
    test = features.iloc[:20].reset_index(drop=True).assign(id=np.arange(1000, 1020))
    submission = pd.DataFrame({'id': test['id'], TARGET: 0.5})
    return train, test, submission


@unittest.skipUnless(HAS_TORCH, 'torch is required for the neural member')
class MultimodelTests(unittest.TestCase):
    def test_blend_split_writes_members_and_valid_probabilities(self):
        from airline_satisfaction.cross_validation import make_folds
        train, test, submission = tiny_frames()
        folds = make_folds(train, n_splits=3, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_split(train, test, folds, Path(tmp), [0.8, 0.1, 0.1],
                               threads=1, log=lambda _: None)
            self.assertEqual(set(result['members']), set(MEMBERS))
            blended = np.asarray(result['test'])
            self.assertTrue(((blended >= 0) & (blended <= 1)).all())
            self.assertEqual(len(result['test']), len(test))
            table = pd.read_csv(Path(tmp) / 'candidates.csv')
            self.assertEqual(list(table['candidate'])[0], 'single_lightgbm')
            self.assertIn('blend_', table['candidate'].iloc[-1])
            for name in MEMBERS:
                self.assertTrue((Path(tmp) / 'members' / name / 'oof.csv').exists())
                self.assertTrue((Path(tmp) / 'members' / name / 'fold_0.joblib').exists())
            blend_row = table.iloc[-1]
            self.assertGreaterEqual(blend_row['win_folds'], 0)
            self.assertLessEqual(blend_row['win_folds'], 3)


if __name__ == '__main__':
    unittest.main()
