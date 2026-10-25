"""Embedding MLP used as the decorrelated member of the multi-model blend.

The estimator consumes the raw competition frame (no target encoding) so that its
errors stay different from the gradient-boosted members. Importing this module
requires torch; the rest of the package does not depend on it.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .features import RATINGS
from .preprocessing import CATEGORICAL, DELAY, TARGET

CATEGORICAL_FIELDS = RATINGS + CATEGORICAL + ['Flight Distance']
NUMERIC_FIELDS = ['Age', 'Departure Delay in Minutes', DELAY]


def _resolve_device(device):
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


class _EmbeddingNet(nn.Module):
    def __init__(self, cardinalities, numeric_dim, width, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(8, max(2, (cardinality + 1) // 2)))
            for cardinality in cardinalities])
        input_dim = sum(layer.embedding_dim for layer in self.embeddings) + numeric_dim
        self.body = nn.Sequential(
            nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width // 2, 1))

    def forward(self, codes, numeric):
        parts = [layer(codes[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.body(torch.cat(parts + [numeric], dim=1)).squeeze(1)


class EmbeddingMLPClassifier(ClassifierMixin, BaseEstimator):
    """Seeded embedding MLP with fold-fitted vocabularies and scalers."""

    def __init__(self, epochs=12, batch_size=4096, width=256, learning_rate=2e-3,
                 dropout=0.1, seed=42, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.width = width
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed
        self.device = device

    def _codes(self, frame):
        codes = []
        for column, levels in self.vocabularies_.items():
            codes.append(pd.Index(levels).get_indexer(frame[column]).astype('int64') + 1)
        return np.stack(codes, axis=1)

    def _numeric_frame(self, frame):
        numeric = pd.DataFrame({
            'Age': frame['Age'].astype('float64'),
            'Departure Delay in Minutes': frame['Departure Delay in Minutes'].astype('float64'),
            DELAY: frame[DELAY].astype('float64').fillna(-1.0),
            'missing_arrival': frame[DELAY].isna().astype('float64'),
            'log_distance': np.log1p(frame['Flight Distance'].astype('float64')),
        })
        numeric[NUMERIC_FIELDS] = np.log1p(numeric[NUMERIC_FIELDS].clip(lower=0))
        return numeric

    def _standardize(self, frame):
        numeric = self._numeric_frame(frame)
        values = ((numeric - self.scaler_['mean']) / self.scaler_['std']).to_numpy('float32')
        return np.ascontiguousarray(values)

    def fit(self, X, y):
        y = np.asarray(y, dtype='float32')
        if not {'id', TARGET}.isdisjoint(X.columns):
            raise ValueError('Remove id and target before fitting.')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.vocabularies_ = {c: np.sort(X[c].dropna().unique()) for c in CATEGORICAL_FIELDS}
        numeric = self._numeric_frame(X)
        self.scaler_ = {'mean': numeric.mean(), 'std': numeric.std().replace(0, 1)}
        codes = self._codes(X)
        values = self._standardize(X)
        self.device_ = _resolve_device(self.device)
        torch.manual_seed(self.seed)
        cardinalities = [len(self.vocabularies_[c]) + 1 for c in CATEGORICAL_FIELDS]
        self.model_ = _EmbeddingNet(cardinalities, values.shape[1], self.width,
                                    self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                      weight_decay=1e-5)
        steps = int(np.ceil(len(y) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.epochs * steps)
        loss_function = nn.BCEWithLogitsLoss()
        code_tensor = torch.from_numpy(codes).to(self.device_)
        value_tensor = torch.from_numpy(values).to(self.device_)
        label_tensor = torch.from_numpy(y).to(self.device_)
        self.model_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(label_tensor), device=self.device_)
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                if len(rows) == 1:  # BatchNorm needs more than one row
                    rows = rows.repeat(2)
                optimizer.zero_grad()
                loss = loss_function(self.model_(code_tensor[rows], value_tensor[rows]),
                                     label_tensor[rows])
                loss.backward()
                optimizer.step()
                scheduler.step()
        self.model_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        codes, values = self._codes(X), self._standardize(X)
        probability = np.empty(len(values), dtype='float64')
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(values), 16384):
                stop = start + 16384
                code_tensor = torch.from_numpy(codes[start:stop]).to(self.device_)
                value_tensor = torch.from_numpy(values[start:stop]).to(self.device_)
                probability[start:stop] = torch.sigmoid(
                    self.model_(code_tensor, value_tensor)).cpu().numpy()
        return np.column_stack([1 - probability, probability])
