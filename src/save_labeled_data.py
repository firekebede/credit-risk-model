import pandas as pd
import os
from target_engineering import generate_target_labels

# Load raw dataset
df_raw = pd.read_csv(r"C:\Users\Administrator\credit-risk-model\data\raw\data.csv")  # update if your file is named differently

# Call the target label generator with correct column names
df_labeled = generate_target_labels(
    df_raw,
    customer_id_col='CustomerId',
    date_col='TransactionStartTime',
    amount_col='Amount'
)

# Save labeled dataset
os.makedirs("data/processed", exist_ok=True)
df_labeled.to_csv("data/processed/df_labeled.csv", index=False)

print("✅ Labeled dataset saved to data/processed/df_labeled.csv")




