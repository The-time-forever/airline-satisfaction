"""Row-wise feature construction with training-fold category vocabularies."""
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .preprocessing import TreePreprocessor

RATINGS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness',
]
GROUPS = ('service_summary', 'context_interactions', 'zero_context', 'delay_distance')
DESCRIPTIONS = {
    'service_summary': '13 项服务评分的均值、标准差、极值、好评/低分/零值数量，以及数字服务与客舱服务的均值和差值',
    'context_interactions': '出行类型×舱位、客户类型×出行类型/舱位，以及出行类型×WiFi 评分、舱位×在线登机评分',
    'zero_context': '各服务的零评分指示，以及 WiFi 零评分与出行类型、客户类型、舱位的交互',
    'delay_distance': '延误与航程的 log1p、延误指示、到达与起飞延误差、延误恢复量、单位航程延误',
}


def construct_features(frame, groups):
    if len(set(groups)) != len(groups) or set(groups) - set(GROUPS):
        raise ValueError('Unknown or duplicate feature groups.')
    result = frame.copy()

    def cross(name, columns):
        parts = [result[c].astype('string').fillna('__UNKNOWN__') for c in columns]
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + '|' + part
        result[name] = combined

    if 'service_summary' in groups:
        ratings = result[RATINGS]
        result['FE_service_mean'] = ratings.mean(axis=1)
        result['FE_service_std'] = ratings.std(axis=1, ddof=0)
        result['FE_service_min'] = ratings.min(axis=1)
        result['FE_service_max'] = ratings.max(axis=1)
        result['FE_service_range'] = ratings.max(axis=1) - ratings.min(axis=1)
        result['FE_service_high_count'] = ratings.ge(4).sum(axis=1)
        result['FE_service_low_count'] = (ratings.ge(1) & ratings.le(2)).sum(axis=1)
        result['FE_service_zero_count'] = ratings.eq(0).sum(axis=1)
        result['FE_service_nonzero_mean'] = ratings.where(ratings.ne(0)).mean(axis=1).fillna(0)
        digital = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']
        cabin = ['Seat comfort', 'Inflight entertainment', 'Cleanliness', 'Food and drink',
                 'Leg room service', 'On-board service']
        result['FE_digital_mean'] = result[digital].mean(axis=1)
        result['FE_cabin_mean'] = result[cabin].mean(axis=1)
        result['FE_digital_cabin_gap'] = result['FE_digital_mean'] - result['FE_cabin_mean']
    if 'context_interactions' in groups:
        for name, columns in [
            ('FE_travel_class', ['Type of Travel', 'Class']),
            ('FE_customer_travel', ['Customer Type', 'Type of Travel']),
            ('FE_customer_class', ['Customer Type', 'Class']),
            ('FE_travel_wifi', ['Type of Travel', 'Inflight wifi service']),
            ('FE_class_boarding', ['Class', 'Online boarding']),
        ]:
            cross(name, columns)
    if 'zero_context' in groups:
        for c in RATINGS:
            if c != 'Baggage handling':
                result['FE_zero_' + c] = result[c].eq(0).astype('int8')
        for name, columns in [
            ('FE_wifi_zero_travel', ['FE_zero_Inflight wifi service', 'Type of Travel']),
            ('FE_wifi_zero_customer', ['FE_zero_Inflight wifi service', 'Customer Type']),
            ('FE_wifi_zero_class', ['FE_zero_Inflight wifi service', 'Class']),
        ]:
            cross(name, columns)
    if 'delay_distance' in groups:
        departure = result['Departure Delay in Minutes']
        arrival = result['Arrival Delay in Minutes']
        distance = result['Flight Distance']
        result['FE_departure_log'] = np.log1p(departure)
        result['FE_arrival_log'] = np.log1p(arrival)
        result['FE_distance_log'] = np.log1p(distance)
        result['FE_departure_positive'] = departure.gt(0).astype('int8')
        result['FE_arrival_positive'] = arrival.gt(0).astype('int8')
        result['FE_delay_change'] = arrival - departure
        result['FE_delay_recovered'] = (departure - arrival).clip(lower=0)
        result['FE_arrival_per_1000_distance'] = arrival * 1000 / distance.clip(lower=1)
    return result


class FeaturePreprocessor(TreePreprocessor):
    """Fit original imputers and all new category vocabularies on training rows."""

    def __init__(self, groups=()):
        self.groups = groups

    def fit(self, X, y=None):
        super().fit(X, y)
        expanded = construct_features(super().transform(X), self.groups)
        self.engineered_categories_ = {
            c: sorted(expanded[c].dropna().unique()) for c in expanded
            if c not in X and pd.api.types.is_string_dtype(expanded[c].dtype)
        }
        self.output_names_ = np.asarray(expanded.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'output_names_')
        result = construct_features(super().transform(X), self.groups)
        for c, categories in self.engineered_categories_.items():
            values = result[c].where(result[c].isin(categories))
            result[c] = pd.Categorical(values, categories=categories)
        numeric = result.select_dtypes(include='number').columns
        result[numeric] = result[numeric].astype('float32')
        if not np.isfinite(result[numeric].to_numpy()).all():
            raise ValueError('Non-finite engineered features.')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_names_')
        return self.output_names_.copy()
