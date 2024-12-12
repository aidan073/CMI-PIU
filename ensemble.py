import pandas as pd
import numpy as np

model_files = [
    # 'rf_prediction.csv',  # RF model predictions
    'nn_prediction.csv',  # NN model predictions
    'knn_prediction.csv'  # KNN model predictions
]

merged_predictions = pd.read_csv(model_files[0])  # Start with the first model's predictions

for model_file in model_files[1:]:
    model_predictions = pd.read_csv(model_file)  # Load current model's predictions
    model_name = model_file.split('_')[0]
    merged_predictions = merged_predictions.merge(
        model_predictions[['id', 'sii']], 
        on='id', 
        suffixes=('', f'_{model_name}')
    )

# Create a list of all prediction columns
prediction_columns = [col for col in merged_predictions.columns if col.startswith('sii')]

# Function to break ties using most frequent prediction
def break_tie(row, prediction_columns):
    predictions = row[prediction_columns].values
    unique_preds, counts = np.unique(predictions, return_counts=True)
    
    if len(unique_preds) == 1:
        return unique_preds[0]  # No tie, return the only prediction
    else:
        most_frequent_pred = unique_preds[np.argmax(counts)]  # Get the most frequent prediction
        return most_frequent_pred

# Apply the tie-breaking function to each row
merged_predictions['final_sii'] = merged_predictions.apply(
    break_tie, 
    axis=1, 
    prediction_columns=prediction_columns
)

prediction = merged_predictions[['id', 'final_sii']]
prediction.rename(columns={'final_sii': 'sii'}, inplace=True)
prediction.to_csv('submission.csv', index=False)

print("Final prediction file created: submission.csv")
