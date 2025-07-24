import pandas as pd

# Load cleaned features and odor labels, merge them, and save the final dataset
def merge_features_labels(features_csv, labels_csv, output_csv):
    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv)

    merged_df = pd.merge(features_df, labels_df, on='name', how='inner')
    
    merged_df.to_csv(output_csv, index=False)

# merge_features_labels("data/cleaned_features.csv", "data/odor_labels.csv", "data/final_dataset.csv")
