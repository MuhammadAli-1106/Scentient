import os
from parse_smiles import parse_smiles
from featurize import featurize
from clean_features import clean_features
from select_top_features import select_top_features

# File paths
SMILES_CSV = "data/sample_smiles.csv"
FEATURES_CSV = "data/features.csv"
CLEANED_FEATURES_CSV = "data/cleaned_features.csv"
LABELS_CSV = "data/odor_labels.csv"
FINAL_DATASET_CSV = "data/final_dataset.csv"
IMPORTANCE_CSV = "data/feature_importance.csv"

def main():
    molecules = parse_smiles(SMILES_CSV)
    featurize(molecules, FEATURES_CSV)
    clean_features(FEATURES_CSV, CLEANED_FEATURES_CSV)
    select_top_features(CLEANED_FEATURES_CSV, LABELS_CSV, FINAL_DATASET_CSV, IMPORTANCE_CSV, top_n=250)

if __name__ == "__main__":
    main()
