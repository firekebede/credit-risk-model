import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.target_engineering import compute_rfm

# Test 1: Check output shape and columns
def test_compute_rfm_shape_and_columns():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-15', '2023-01-01'],
        'Amount': [100, 200, 300]
    })
    rfm = compute_rfm(df, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount')
    assert rfm.shape[0] == 2, "Should return 2 customers"
    assert set(rfm.columns) == {'CustomerId', 'Recency', 'Frequency', 'Monetary'}

# Test 2: Check RFM values are non-negative
def test_rfm_values_non_negative():
    df = pd.DataFrame({
        'CustomerId': [1],
        'TransactionStartTime': ['2023-01-01'],
        'Amount': [100]
    })
    rfm = compute_rfm(df, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount')
    assert (rfm[['Recency', 'Frequency', 'Monetary']] >= 0).all().all(), "RFM metrics should be non-negative"
