import pandas as pd

from ml_utils import DATASET_PATH


def main() -> None:
    data = pd.read_csv(DATASET_PATH)
    print("First five visits:")
    print(data.head())
    print("\nDataset shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nDuplicate rows:", data.duplicated().sum())
    print("\nNumeric summary:")
    print(data.describe(include="number"))
    print("\nDepartment counts:")
    print(data["Department"].value_counts())
    print("\nVisit-type counts:")
    print(data["Visit_Type"].value_counts())
    print("\nWaiting-time summary:")
    print(data["Waiting_Time"].describe())


if __name__ == "__main__":
    main()
