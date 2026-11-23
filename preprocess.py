from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_hospital_data,
)


def main() -> None:
    data = load_hospital_data()
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Data preprocessing completed.")
    print("Training data shape:", X_train_processed.shape)
    print("Testing data shape:", X_test_processed.shape)
    print("\nTraining target distribution:")
    print(y_train.value_counts().reindex(["Low", "Medium", "High"]))
    print("\nTesting target distribution:")
    print(y_test.value_counts().reindex(["Low", "Medium", "High"]))


if __name__ == "__main__":
    main()
