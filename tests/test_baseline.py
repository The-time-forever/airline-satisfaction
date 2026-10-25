from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from airline_satisfaction.baseline import train_baselines
from airline_satisfaction.cross_validation import make_folds
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY


class BaselineTests(unittest.TestCase):
    def test_incremental_run_preserves_previous_model_results(self):
        train = pd.DataFrame({
            'id': np.arange(20), 'Age': np.arange(20, 40),
            DELAY: [0., 1., np.nan, 3.] * 5,
            **{c: ['a', 'b'] * 10 for c in CATEGORICAL},
            'satisfaction': [False, True] * 10,
        })
        test = train.drop(columns='satisfaction').iloc[:4].copy()
        test['id'] = np.arange(100, 104)
        sample = pd.DataFrame({'id': test['id'], 'satisfaction': .5})
        old_summary = {'model': 'previous', 'mean_roc_auc': .7,
                       'std_roc_auc': 0., 'oof_roc_auc': .7,
                       'oof_log_loss': .6, 'seconds': 1.}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'previous').mkdir()
            previous_predictions = sample.assign(satisfaction=[.1, .2, .3, .4])
            previous_predictions.to_csv(root / 'previous/test_predictions.csv', index=False)
            pd.DataFrame([old_summary]).to_csv(root / 'summary.csv', index=False)
            pd.DataFrame([{'model': 'previous', 'fold': 0, 'roc_auc': .7}]).to_csv(
                root / 'fold_scores.csv', index=False)
            before = (root / 'previous/test_predictions.csv').read_bytes()
            specs = {'new': ('tree', lambda: DummyClassifier(strategy='prior'))}
            with patch('airline_satisfaction.baseline.model_specs', return_value=specs):
                ranking, best = train_baselines(
                    train, test, sample, make_folds(train), root, models=['new'])
            self.assertEqual(best, 'previous')
            self.assertEqual(set(ranking.model), {'previous', 'new'})
            self.assertEqual(ranking.set_index('model').loc['previous', 'mean_roc_auc'], .7)
            self.assertEqual((root / 'previous/test_predictions.csv').read_bytes(), before)
            pd.testing.assert_frame_equal(pd.read_csv(root / 'submission.csv'), previous_predictions)
            scores = pd.read_csv(root / 'fold_scores.csv')
            self.assertEqual((scores.model == 'previous').sum(), 1)
            self.assertEqual((scores.model == 'new').sum(), 5)


if __name__ == '__main__':
    unittest.main()
