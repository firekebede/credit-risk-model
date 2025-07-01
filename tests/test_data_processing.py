from src.data_processing import get_data_pipeline

num_cols = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount']
cat_cols = ['gender', 'region']

pipeline = get_data_pipeline(num_cols, cat_cols)
processed = pipeline.fit_transform(df)
