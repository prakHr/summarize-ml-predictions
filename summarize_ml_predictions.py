import os
import warnings

import numpy as np
import pandas as pd

from langvec import LangVec

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "iris.csv"

TARGET_COLUMN = "variety"

CHUNK_SIZE = 3

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found."
        )

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(
        df[TARGET_COLUMN].value_counts()
    )

    return df


# ============================================================
# PREPARE X AND Y
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # X = numerical features
    # --------------------------------------------------------

    X_df = df.drop(
        columns=[TARGET_COLUMN]
    )

    X_df = X_df.select_dtypes(
        include=[np.number]
    )

    # --------------------------------------------------------
    # y = target
    # --------------------------------------------------------

    y = df[TARGET_COLUMN].to_numpy()

    X = X_df.to_numpy(
        dtype=np.float64
    )

    print("\n" + "=" * 70)
    print("FEATURES")
    print("=" * 70)

    print(
        "Feature columns:",
        X_df.columns.tolist()
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# FIT LANGVEC
# ============================================================

def fit_langvec(X_train):

    print("\n" + "=" * 70)
    print("LANGVEC")
    print("=" * 70)

    model = LangVec()

    X_train_list = [
        np.asarray(
            row,
            dtype=np.float64
        )
        for row in X_train
    ]

    model.fit(
        X_train_list
    )

    print(
        f"LangVec fitted using "
        f"{len(X_train_list)} training samples."
    )

    return model


# ============================================================
# LANGVEC PREDICTION
# ============================================================

def langvec_transform(
    model,
    X
):

    representations = []

    for row in X:

        representation = model.predict(
            np.asarray(
                row,
                dtype=np.float64
            ),
            chunk_size=CHUNK_SIZE,
            summarized=False,
            padding=True
        )

        # Make sure representation
        # is always a list.
        if isinstance(
            representation,
            np.ndarray
        ):
            representation = (
                representation.tolist()
            )

        elif not isinstance(
            representation,
            (list, tuple)
        ):
            representation = [
                representation
            ]

        representation = [
            str(token)
            for token in representation
        ]

        representations.append(
            representation
        )

    return representations


# ============================================================
# PRINT LANGVEC OUTPUT
# ============================================================

def print_langvec_output(
    X,
    representations,
    y
):

    print("\n" + "=" * 70)
    print("LANGVEC REPRESENTATIONS")
    print("=" * 70)

    for i in range(len(X)):

        print(
            f"{i:3d} | "
            f"{representations[i]} | "
            f"{y[i]}"
        )


# ============================================================
# CONVERT LANGVEC TOKENS TO NUMERICAL VECTORS
# ============================================================

def create_one_hot_encoder():

    # Each position in the LangVec output
    # is treated as a categorical feature.
    #
    # Example:
    #
    # ['nut', 'ant']
    #
    # becomes something like:
    #
    # position_0_nut = 1
    # position_1_ant = 1

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64
    )

    return encoder


def encode_langvec(
    encoder,
    representations,
    fit=False
):

    # --------------------------------------------------------
    # Make every representation the same length.
    #
    # Example:
    #
    # ['nut', 'ant']
    # ['rat', 'bat']
    #
    # becomes:
    #
    # [
    #   ['nut', 'ant'],
    #   ['rat', 'bat']
    # ]
    #
    # We encode each position independently.
    # --------------------------------------------------------

    max_length = max(
        len(rep)
        for rep in representations
    )

    padded = []

    for rep in representations:

        row = list(rep)

        while len(row) < max_length:
            row.append("<PAD>")

        padded.append(row)

    if fit:

        encoded = encoder.fit_transform(
            padded
        )

    else:

        encoded = encoder.transform(
            padded
        )

    return encoded


# ============================================================
# PRINT ENCODING INFORMATION
# ============================================================

def print_encoding_info(
    encoder,
    X_encoded
):

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODING")
    print("=" * 70)

    print(
        "Encoded shape:",
        X_encoded.shape
    )

    print("\nEncoded feature names:")

    feature_names = (
        encoder.get_feature_names_out()
    )

    for i, name in enumerate(
        feature_names
    ):

        print(
            f"{i:3d}: {name}"
        )


# ============================================================
# TRAIN ML MODEL
# ============================================================

def train_classifier(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING ML CLASSIFIER")
    print("=" * 70)

    classifier = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    classifier.fit(
        X_train,
        y_train
    )

    print(
        "Model: LogisticRegression"
    )

    return classifier


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return predictions


# ============================================================
# SHOW PREDICTIONS
# ============================================================

def print_predictions(
    representations,
    y_test,
    predictions
):

    print("\n" + "=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)

    for i in range(len(y_test)):

        print(
            f"{i:3d} | "
            f"LangVec={representations[i]} | "
            f"Actual={y_test[i]:12s} | "
            f"Predicted={predictions[i]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_dataset(
        DATASET_PATH
    )

    # ========================================================
    # 2. PREPARE X / Y
    # ========================================================

    X, y = prepare_data(
        df
    )

    # ========================================================
    # 3. TRAIN / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # ========================================================
    # 4. FIT LANGVEC
    #
    # IMPORTANT:
    #
    # LangVec is fitted ONLY on the training data.
    #
    # This avoids using information from the
    # test set during training.
    # ========================================================

    langvec = fit_langvec(
        X_train
    )

    # ========================================================
    # 5. CONVERT TRAINING DATA USING LANGVEC
    # ========================================================

    train_representations = (
        langvec_transform(
            langvec,
            X_train
        )
    )

    # ========================================================
    # 6. CONVERT TEST DATA USING LANGVEC
    # ========================================================

    test_representations = (
        langvec_transform(
            langvec,
            X_test
        )
    )

    # ========================================================
    # 7. PRINT REPRESENTATIONS
    # ========================================================

    print_langvec_output(
        X_train,
        train_representations,
        y_train
    )

    # ========================================================
    # 8. ONE-HOT ENCODER
    # ========================================================

    encoder = create_one_hot_encoder()

    # --------------------------------------------------------
    # Fit encoder ONLY on training representations
    # --------------------------------------------------------

    X_train_encoded = encode_langvec(
        encoder,
        train_representations,
        fit=True
    )

    # --------------------------------------------------------
    # Transform test representations
    # --------------------------------------------------------

    X_test_encoded = encode_langvec(
        encoder,
        test_representations,
        fit=False
    )

    # ========================================================
    # 9. ENCODING INFORMATION
    # ========================================================

    print_encoding_info(
        encoder,
        X_train_encoded
    )

    print(
        "\nTraining encoded shape:",
        X_train_encoded.shape
    )

    print(
        "Testing encoded shape :",
        X_test_encoded.shape
    )

    # ========================================================
    # 10. TRAIN ML MODEL
    # ========================================================

    classifier = train_classifier(
        X_train_encoded,
        y_train
    )

    # ========================================================
    # 11. EVALUATE
    # ========================================================

    predictions = evaluate_model(
        classifier,
        X_test_encoded,
        y_test
    )

    # ========================================================
    # 12. TEST PREDICTIONS
    # ========================================================

    print_predictions(
        test_representations,
        y_test,
        predictions
    )

    # ========================================================
    # 13. COMPLETE PIPELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE")
    print("=" * 70)

    print(
        """
Iris CSV
   |
   +-----------------------------+
   |                             |
   ▼                             ▼
Numerical features            variety
   |                             |
   ▼                             |
LangVec.fit()                    |
   |                             |
   ▼                             |
LangVec.predict()               |
   |                             |
   ▼                             |
['word', 'word']                 |
   |                             |
   ▼                             |
OneHotEncoder                    |
   |                             |
   ▼                             |
Numerical vector                 |
   |                             |
   ▼                             |
Logistic Regression <------------+
   |
   ▼
Predicted variety
"""
    )

    print(
        "\nPipeline completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
