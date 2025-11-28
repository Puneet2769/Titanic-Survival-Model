# ============================================================
# Titanic - Kaggle Baseline Pipeline
# Structure: functions + main()
# ============================================================

# ----- STEP 0: IMPORTS & CONFIG -----

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


pd.set_option("display.float_format", lambda x: f"{x:.2f}")


# ----- STEP 1: LOAD DATA -----

def load_titanic_data(train_path: str = "train.csv",
                      test_path: str = "test.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Titanic train and test CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain columns:", list(train_df.columns))

    return train_df, test_df


# ----- STEP 2: BASIC FEATURE ENGINEERING & PREPROCESSING -----

def preprocess_titanic(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Simple preprocessing:
      - Keep useful columns
      - Fill missing Age, Fare, Embarked
      - Create FamilySize
      - Convert Sex, Embarked to numeric (one-hot)
    Returns a new DataFrame ready for ML.
    """
    df = df.copy()

    # Keep only some selected columns
    cols_to_keep = [
        "PassengerId",
        "Survived" if "Survived" in df.columns else None,
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]
    cols_to_keep = [c for c in cols_to_keep if c is not None]
    df = df[cols_to_keep]

    # Fill missing Age and Fare with median
    if df["Age"].isna().any():
        age_median = df["Age"].median()
        df["Age"] = df["Age"].fillna(age_median)

    if df["Fare"].isna().any():
        fare_median = df["Fare"].median()
        df["Fare"] = df["Fare"].fillna(fare_median)

    # Fill missing Embarked with mode
    if df["Embarked"].isna().any():
        mode_embarked = df["Embarked"].mode()[0]
        df["Embarked"] = df["Embarked"].fillna(mode_embarked)

    # Simple feature: FamilySize = SibSp + Parch + 1 (self)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Optional: Title from Name (Mr, Mrs, Miss, etc.)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    # Simplify rare titles
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
         "Sir", "Jonkheer", "Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace(
        ["Mlle", "Ms"],
        "Miss"
    )
    df["Title"] = df["Title"].replace(
        ["Mme"],
        "Mrs"
    )

    # Drop columns we don't want to feed directly to the model
    df = df.drop(columns=["Name"], errors="ignore")

    # One-hot encode categorical variables
    cat_cols = ["Sex", "Embarked", "Title"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if is_train:
        print("\nSample of preprocessed TRAIN rows:")
    else:
        print("\nSample of preprocessed TEST rows:")

    print(df.head(5))

    return df


# ----- STEP 3: BUILD TRAIN/VALIDATION SPLIT -----

def build_train_valid(df_train_processed: pd.DataFrame,
                      target_col: str = "Survived",
                      valid_size: float = 0.2,
                      random_state: int = 42):
    """
    Split processed train data into train and validation sets.
    """
    # Features = all columns except PassengerId and target
    feature_cols = [c for c in df_train_processed.columns
                    if c not in ["PassengerId", target_col]]

    X = df_train_processed[feature_cols]
    y = df_train_processed[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=random_state, stratify=y
    )

    print("\nTrain size:", X_train.shape[0])
    print("Valid size:", X_valid.shape[0])
    print("Num features:", X_train.shape[1])

    return X_train, X_valid, y_train, y_valid, feature_cols


# ----- STEP 4: TRAIN BASELINE MODEL -----

def train_titanic_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a simple RandomForest classifier as baseline.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ----- STEP 5: EVALUATE MODEL ON VALIDATION SET -----

def evaluate_titanic_model(model, X_valid, y_valid):
    """
    Print accuracy and classification report for validation set.
    """
    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)

    print("\n+---------------- VALIDATION METRICS ----------------+")
    print(f"Validation Accuracy: {acc:.3f}")
    print("+---------------------------------------------------+")
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, digits=3))


# ----- STEP 6: TRAIN FINAL MODEL ON FULL TRAIN DATA -----

def train_final_model(df_train_processed: pd.DataFrame,
                      feature_cols: list[str],
                      target_col: str = "Survived") -> RandomForestClassifier:
    """
    Train final model on full training data with chosen features.
    """
    X_full = df_train_processed[feature_cols]
    y_full = df_train_processed[target_col]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_full, y_full)
    return model


# ----- STEP 7: GENERATE SUBMISSION FILE -----

def make_submission_file(model: RandomForestClassifier,
                         df_train_processed: pd.DataFrame,
                         df_test_processed: pd.DataFrame,
                         feature_cols: list[str],
                         output_path: str = "submission.csv"):
    """
    Train final model on full training data and generate predictions for test.csv.
    Ensures test has all feature_cols (adds missing columns with 0).
    Saves a submission file with columns: PassengerId, Survived.
    """
    # --- ensure all feature columns exist in test ---
    missing_in_test = [c for c in feature_cols if c not in df_test_processed.columns]
    if missing_in_test:
        print("\n[Info] Adding missing columns to test (set to 0):")
        print(missing_in_test)
        for c in missing_in_test:
            df_test_processed[c] = 0

    # Also, if test has extra columns not in feature_cols, they will be ignored
    X_full = df_train_processed[feature_cols]
    y_full = df_train_processed["Survived"]

    # Train final model on full train
    model.fit(X_full, y_full)

    # Build X_test using the same feature columns
    X_test = df_test_processed[feature_cols]
    test_preds = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": df_test_processed["PassengerId"],
        "Survived": test_preds.astype(int),
    })

    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission file to: {output_path}")



# ----- MAIN PIPELINE -----

def run_titanic_pipeline():
    # 1) Load raw data
    train_df, test_df = load_titanic_data("train.csv", "test.csv")

    # 2) Preprocess train & test
    train_processed = preprocess_titanic(train_df, is_train=True)
    test_processed = preprocess_titanic(test_df, is_train=False)

    # 3) Build train/validation split
    X_train, X_valid, y_train, y_valid, feature_cols = build_train_valid(train_processed)

    # 4) Train baseline model
    model = train_titanic_model(X_train, y_train)

    # 5) Evaluate on validation set
    evaluate_titanic_model(model, X_valid, y_valid)

    # 6) Train final model on full train and create submission
    make_submission_file(model, train_processed, test_processed, feature_cols,
                         output_path="submission_baseline_rf.csv")


def main():
    run_titanic_pipeline()


if __name__ == "__main__":
    main()
