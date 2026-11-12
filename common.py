import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag"
]

TARGET_COLUMN = "churned"


def load_data():
    df = pd.read_csv("data/records.csv")

    class_0 = df[df[TARGET_COLUMN] == 0].index.to_numpy()
    class_1 = df[df[TARGET_COLUMN] == 1].index.to_numpy()

    train_0_count = (7 * len(class_0)) // 10
    train_1_count = (7 * len(class_1)) // 10

    train_indices = np.sort(
        np.concatenate([
            class_0[:train_0_count],
            class_1[:train_1_count]
        ])
    )

    test_indices = np.sort(
        np.concatenate([
            class_0[train_0_count:],
            class_1[train_1_count:]
        ])
    )

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=1)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return (
        train_df, test_df,
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled
    )


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic(X, y):
    weights = np.zeros(X.shape[1])
    bias = 0.0

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        predictions = sigmoid(np.dot(X, weights) + bias)

        error = predictions - y

        dw = np.dot(X.T, error) / len(y)
        db = np.mean(error)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def knn_probabilities(X_reference, y_reference, X_query, k):

    probabilities = []

    for query in X_query:

        distances = np.sqrt(
            np.sum((X_reference - query) ** 2, axis=1)
        )

        nearest_indices = np.argsort(
            distances, kind="stable"
        )[:k]

        neighbour_labels = y_reference[nearest_indices]

        probability = np.mean(neighbour_labels)

        probabilities.append(probability)

    return np.array(probabilities)


def knn_predict(X_reference, y_reference, X_query, k):

    probabilities = knn_probabilities(
        X_reference, y_reference, X_query, k
    )

    predictions = (
        probabilities >= (k + 1) / 2 / k
    ).astype(int)

    return predictions