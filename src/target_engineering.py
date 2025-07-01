import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def compute_rfm(df, customer_id_col='customer_id', date_col='transaction_date', amount_col='amount', snapshot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: 'count',
        amount_col: 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm.reset_index(inplace=True)
    return rfm


def rfm_cluster_and_label(rfm_df, n_clusters=3, random_state=42):
    """
    Apply KMeans clustering to RFM metrics and assign cluster labels.
    """
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(scaled_rfm)
    return rfm_df


def assign_high_risk_label(rfm_clustered_df):
    """
    Determine which cluster is high risk and assign binary label.
    """
    cluster_stats = rfm_clustered_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats.sort_values(
        by=['Recency', 'Frequency', 'Monetary'],
        ascending=[False, True, True]
    ).index[0]

    rfm_clustered_df['is_high_risk'] = (rfm_clustered_df['cluster'] == high_risk_cluster).astype(int)
    return rfm_clustered_df.drop(columns=['cluster'])


def merge_risk_labels(original_df, rfm_labels_df, customer_id_col='customer_id'):
    """
    Merge the high-risk labels back into the original DataFrame.
    """
    return original_df.merge(rfm_labels_df[[customer_id_col, 'is_high_risk']], on=customer_id_col, how='left')


def generate_target_labels(df, customer_id_col='customer_id', date_col='transaction_date', amount_col='amount'):
    """
    Full pipeline to generate binary risk label for each customer.
    """
    rfm_df = compute_rfm(df, customer_id_col, date_col, amount_col)
    rfm_clustered = rfm_cluster_and_label(rfm_df)
    labeled_df = assign_high_risk_label(rfm_clustered)
    final_df = merge_risk_labels(df, labeled_df, customer_id_col)
    return final_df





