import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


#Aggregate Features
class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'count']
        })
        agg.columns = ['total_amount', 'avg_amount', 'std_amount', 'transaction_count']
        agg.reset_index(inplace=True)
        return agg

#Time Feature Extractor

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='transaction_date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df['transaction_hour'] = df[self.date_column].dt.hour
        df['transaction_day'] = df[self.date_column].dt.day
        df['transaction_month'] = df[self.date_column].dt.month
        df['transaction_year'] = df[self.date_column].dt.year
        return df

#Combine Pipeline Steps

def get_data_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ('date_features', DateFeatureExtractor()),
        ('preprocessing', preprocessor)
    ])

    return full_pipeline
