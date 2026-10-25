from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sqlalchemy.pool import NullPool

from airline_satisfaction.tuning import suggest_parameters, build_objective, export_trials, SPACE


class TuningTests(unittest.TestCase):
    def reference(self):
        return {'n_estimators': 500, 'learning_rate': .05, 'num_leaves': 31,
                'min_child_samples': 50, 'colsample_bytree': .9, 'subsample': .9,
                'reg_alpha': 0., 'reg_lambda': 1., 'random_state': 42}

    def test_suggestions_preserve_fixed_parameters(self):
        reference = self.reference()
        fixed = optuna.trial.FixedTrial({k: reference[k] for k in SPACE})
        self.assertEqual(suggest_parameters(fixed, reference), reference)

    def test_reference_scores_are_reused_and_records_survive_sqlite_reload(self):
        reference = self.reference()
        cache = [{'fold': i} for i in range(5)]
        rows = pd.DataFrame({'roc_auc': [.8] * 5, 'log_loss': [.3] * 5})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = f'sqlite:///{(root / "study.db").as_posix()}'
            storage = optuna.storages.RDBStorage(url, engine_kwargs={'poolclass': NullPool})
            study = optuna.create_study(storage=storage, study_name='test', direction='maximize')
            study.enqueue_trial({k: reference[k] for k in SPACE})
            objective = build_objective(cache, reference, rows, root / 'sampler.joblib')
            with patch('airline_satisfaction.tuning.LGBMClassifier') as model:
                study.optimize(objective, n_trials=1)
                model.assert_not_called()
            export_trials(study, root)
            loaded = optuna.load_study(storage=storage, study_name='test')
            self.assertAlmostEqual(loaded.best_value, .8)
            self.assertTrue(loaded.best_trial.user_attrs['cached_reference'])
            self.assertEqual(len(pd.read_csv(root / 'trial_folds.csv')), 5)
            self.assertEqual(pd.read_csv(root / 'trials.csv').loc[0, 'state'], 'COMPLETE')
            storage.remove_session()
            storage.engine.dispose()

    def test_pruned_trial_is_not_exported_as_complete_cv_score(self):
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        trial.set_user_attr('fold_scores', [{'fold': 0, 'roc_auc': .7}])
        trial.set_user_attr('partial_mean_auc', .7)
        trial.report(.7, step=0)
        study.tell(trial, state=TrialState.PRUNED)
        with tempfile.TemporaryDirectory() as tmp:
            export_trials(study, Path(tmp))
            record = pd.read_csv(Path(tmp) / 'trials.csv').iloc[0]
            self.assertTrue(np.isnan(record.mean_cv_auc))
            self.assertEqual(record.partial_mean_auc, .7)
            self.assertEqual(record.completed_folds, 1)


if __name__ == '__main__':
    unittest.main()
