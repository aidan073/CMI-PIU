import pandas as pd

submission = pd.read_csv("submission.csv")
actual_sii = pd.read_csv("data/train.csv")
actual_sii = actual_sii[['id', 'sii']]
actual_sii = actual_sii.head(20)
valid_actual_sii = actual_sii.dropna(subset=['sii'])
valid_submission = submission[submission['id'].isin(valid_actual_sii['id'])]

merged_df = pd.merge(valid_actual_sii, valid_submission[['id', 'sii']], on='id', how='inner', suffixes=('_actual', '_predicted'))
matching_count = (merged_df['sii_actual'] == merged_df['sii_predicted']).sum()  # Count how many are the same
percentage_matching = (matching_count / len(merged_df)) * 100
print(f"Percentage of matching 'sii' values: {percentage_matching:.2f}%")
