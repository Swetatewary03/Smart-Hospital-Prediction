from ml_utils import TARGET_COLUMN, load_hospital_data


def main() -> None:
    data = load_hospital_data()
    print("Target variable distribution:")
    print(data[TARGET_COLUMN].value_counts().reindex(["Low", "Medium", "High"]))
    print("\nSample waiting times and categories:")
    print(data[["Waiting_Time", TARGET_COLUMN]].head())


if __name__ == "__main__":
    main()
