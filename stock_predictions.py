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

DATASET_PATH = "stockdata.csv"

DATE_COLUMN = "Date"

STOCK_COLUMNS = [
    "MSFT",
    "IBM",
    "SBUX",
    "AAPL",
    "GSPC",
]

CHUNK_SIZE = 3

TEST_SIZE = 0.20

RANDOM_STATE = 42

# Predict whether the next day's market
# movement is UP or DOWN.
TARGET_COLUMN = "Target"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing = [
        column
        for column in STOCK_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing stock columns: {missing}"
        )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Missing date column: {DATE_COLUMN}"
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert stock columns to numeric
    # --------------------------------------------------------

    for column in STOCK_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=STOCK_COLUMNS
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create target
    #
    # Target is based on the next day's S&P 500
    # movement.
    #
    # GSPC(t+1) > GSPC(t) -> UP
    # GSPC(t+1) <= GSPC(t) -> DOWN
    # --------------------------------------------------------

    future_gspc = (
        df["GSPC"].shift(-1)
    )

    df[TARGET_COLUMN] = np.where(
        future_gspc > df["GSPC"],
        "UP",
        "DOWN"
    )

    # Last row has no future value,
    # therefore remove it.
    df = df.iloc[:-1].copy()

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    X = df[
        STOCK_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # y
    # --------------------------------------------------------

    y = df[
        TARGET_COLUMN
    ].to_numpy()

    print(
        "\nFeature columns:"
    )

    print(STOCK_COLUMNS)

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "\nTarget distribution:"
    )

    print(
        pd.Series(y).value_counts()
    )

    return df, X, y


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
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
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
    print("FITTING LANGVEC")
    print("=" * 70)

    model = LangVec()

    X_list = [
        np.asarray(
            row,
            dtype=np.float64
        )
        for row in X_train
    ]

    model.fit(
        X_list
    )

    print(
        f"LangVec fitted on "
        f"{len(X_list)} samples."
    )

    return model


# ============================================================
# LANGVEC TRANSFORMATION
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
# PRINT LANGVEC REPRESENTATIONS
# ============================================================

def print_representations(
    representations,
    y
):

    print("\n" + "=" * 70)
    print("LANGVEC REPRESENTATIONS")
    print("=" * 70)

    for i in range(
        min(len(representations), 100)
    ):

        print(
            f"{i:4d} | "
            f"{representations[i]} | "
            f"{y[i]}"
        )

    if len(representations) > 100:

        print(
            f"\n... "
            f"{len(representations) - 100} "
            f"more rows ..."
        )


# ============================================================
# PAD LANGVEC REPRESENTATIONS
# ============================================================

def pad_representations(
    representations
):

    max_length = max(
        len(rep)
        for rep in representations
    )

    padded = []

    for representation in representations:

        row = list(
            representation
        )

        while len(row) < max_length:

            row.append(
                "<PAD>"
            )

        padded.append(row)

    return padded


# ============================================================
# ONE-HOT ENCODE LANGVEC OUTPUT
# ============================================================

def fit_one_hot_encoder(
    train_representations
):

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODING LANGVEC")
    print("=" * 70)

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64
    )

    train_padded = pad_representations(
        train_representations
    )

    X_train_encoded = (
        encoder.fit_transform(
            train_padded
        )
    )

    print(
        "Original LangVec representation:"
    )

    print(
        train_representations[0]
    )

    print(
        "\nOne-hot encoded vector:"
    )

    print(
        X_train_encoded[0]
    )

    print(
        "\nEncoded shape:"
    )

    print(
        X_train_encoded.shape
    )

    print(
        "\nNumber of encoded features:",
        X_train_encoded.shape[1]
    )

    return (
        encoder,
        X_train_encoded
    )


def transform_one_hot_encoder(
    encoder,
    representations
):

    padded = pad_representations(
        representations
    )

    return encoder.transform(
        padded
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    model = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model: LogisticRegression"
    )

    return model


# ============================================================
# EVALUATE
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
        f"\nAccuracy: {accuracy:.6f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return predictions


# ============================================================
# SHOW TEST PREDICTIONS
# ============================================================

def show_predictions(
    representations,
    y_test,
    predictions
):

    print("\n" + "=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)

    for i in range(
        len(y_test)
    ):

        print(
            f"{i:4d} | "
            f"LangVec={representations[i]} | "
            f"Actual={y_test[i]:4s} | "
            f"Predicted={predictions[i]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD
    # ========================================================

    df = load_dataset(
        DATASET_PATH
    )

    # ========================================================
    # 2. PREPARE
    # ========================================================

    (
        df,
        X,
        y
    ) = prepare_data(
        df
    )

    # ========================================================
    # 3. SPLIT
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
    # 4. LANGVEC
    # ========================================================

    langvec = fit_langvec(
        X_train
    )

    # ========================================================
    # 5. LANGVEC TRANSFORMATION
    # ========================================================

    train_representations = (
        langvec_transform(
            langvec,
            X_train
        )
    )

    test_representations = (
        langvec_transform(
            langvec,
            X_test
        )
    )

    # ========================================================
    # 6. PRINT REPRESENTATIONS
    # ========================================================

    print_representations(
        train_representations,
        y_train
    )

    # ========================================================
    # 7. ONE-HOT ENCODING
    # ========================================================

    (
        encoder,
        X_train_encoded
    ) = fit_one_hot_encoder(
        train_representations
    )

    X_test_encoded = (
        transform_one_hot_encoder(
            encoder,
            test_representations
        )
    )

    print(
        "\nTest encoded shape:",
        X_test_encoded.shape
    )

    # ========================================================
    # 8. CLASSIFIER
    # ========================================================

    classifier = train_classifier(
        X_train_encoded,
        y_train
    )

    # ========================================================
    # 9. EVALUATE
    # ========================================================

    predictions = evaluate_model(
        classifier,
        X_test_encoded,
        y_test
    )

    # ========================================================
    # 10. SHOW PREDICTIONS
    # ========================================================

    show_predictions(
        test_representations,
        y_test,
        predictions
    )

    # ========================================================
    # 11. FINAL PIPELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL PIPELINE")
    print("=" * 70)

    print(
        """
Stock prices
     |
     |  MSFT
     |  IBM
     |  SBUX
     |  AAPL
     |  GSPC
     |
     v
   LangVec
     |
     v
['word1', 'word2', ...]
     |
     v
OneHotEncoder
     |
     v
Numerical vector
     |
     v
Logistic Regression
     |
     v
UP / DOWN
"""
    )

    print(
        "\nCompleted successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
