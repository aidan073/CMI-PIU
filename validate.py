import pandas as pd

# Read top 20 from training and compare with submission (null's become -1)
submission = pd.read_csv("submission.csv")
actual_sii = pd.read_csv("data/train.csv")
actual_sii = actual_sii[['id', 'sii']]  
actual_sii['sii'] = actual_sii['sii'].fillna(-1)  
actual_sii = actual_sii.head(20)
predicted_sii = submission['sii']
actual_sii_values = actual_sii['sii']

matching_count = (predicted_sii == actual_sii_values).sum()  # Count how many are the same

percentage_matching = (matching_count / len(actual_sii_values)) * 100

print(f"Percentage of matching 'sii' values: {percentage_matching:.2f}%")
