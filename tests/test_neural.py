import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airline_satisfaction.features import RATINGS
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY, TARGET

try:
    from airline_satisfaction.neural import EmbeddingMLPClassifier
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional heavy dependency
    HAS_TORCH = False


def frame(rows=120):
    rng = np.random.default_rng(1)
    features = pd.DataFrame({
        'Age': rng.integers(20, 70, rows),
        'Flight Distance': rng.integers(100, 3000, rows),
        'Departure Delay in Minutes': rng.integers(0, 10, rows),
        DELAY: rng.integers(0, 10, rows).astype(float),
        **{c: rng.integers(0, 6, rows) for c in RATINGS},
        **{c: rng.choice(['a', 'b'], rows) for c in CATEGORICAL},
    })
    features.loc[0, DELAY] = np.nan
    labels = pd.Series([0, 1] * (rows // 2))
    return features, labels


@unittest.skipUnless(HAS_TORCH, 'torch is required for the neural member')
class NeuralTests(unittest.TestCase):
    def test_predictions_are_reproducible_after_reload(self):
        features, labels = frame()
        model = EmbeddingMLPClassifier(epochs=2, batch_size=32, seed=0, device='cpu')
        model.fit(features, labels)
        probability = model.predict_proba(features)
        self.assertEqual(probability.shape, (len(features), 2))
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1.)
        self.assertEqual(list(model.classes_), [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.joblib'
            joblib.dump(model, path)
            reloaded = joblib.load(path)
            np.testing.assert_allclose(reloaded.predict_proba(features), probability,
                                       rtol=0, atol=1e-9)

    def test_unseen_levels_and_identifier_are_handled(self):
        features, labels = frame()
        model = EmbeddingMLPClassifier(epochs=1, batch_size=32, seed=0, device='cpu')
        model.fit(features, labels)
        unseen = features.iloc[:5].copy()
        unseen['Class'] = 'never-seen'
        unseen['Flight Distance'] = 99999
        self.assertTrue(np.isfinite(model.predict_proba(unseen)).all())
        with self.assertRaises(ValueError):
            model.fit(features.assign(id=1), labels)


if __name__ == '__main__':
    unittest.main()
