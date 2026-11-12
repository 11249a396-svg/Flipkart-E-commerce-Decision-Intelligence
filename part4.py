import numpy as np
import pandas as pd
import os

from common import (
    load_data,
    train_logistic,
    sigmoid,
    knn_probabilities
)

os.makedirs("outputs", exist_ok=True)


# ==========================================
# TASK 4.1 - ENTROPY AND GINI
# ==========================================

def entropy(y):

    if len(y) == 0:
        return 0.0

    p = np.bincount(y, minlength=2) / len(y)

    return -sum(
        value * np.log2(value)
        for value in p if value > 0
    )


def gini(y):

    if len(y) == 0:
        return 0.0

    p = np.bincount(y, minlength=2) / len(y)

    return 1 - np.sum(p ** 2)


def evaluate_split(X, y, condition):

    mask = condition(X)

    yes = y[mask]
    no = y[~mask]

    n = len(y)

    n_yes = len(yes)
    n_no = len(no)

    root_e = entropy(y)
    root_g = gini(y)

    weighted_e = (
        n_yes / n * entropy(yes)
        + n_no / n * entropy(no)
    )

    weighted_g = (
        n_yes / n * gini(yes)
        + n_no / n * gini(no)
    )

    return {
        "n_yes": n_yes,
        "positives_yes": int(np.sum(yes)),
        "n_no": n_no,
        "positives_no": int(np.sum(no)),
        "info_gain": root_e - weighted_e,
        "gini_gain": root_g - weighted_g
    }


def task_4_1(X_train, y_train):

    conditions = {

        "S1": lambda X: X[:, 5] == 1,

        "S2": lambda X: X[:, 6] == 1,

        "S3": lambda X: X[:, 2] > 0.5,

        "S4": lambda X: X[:, 0] <= 24

    }

    results = []

    for name, condition in conditions.items():

        result = evaluate_split(
            X_train,
            y_train,
            condition
        )

        results.append({
            "split": name,
            **result
        })

    df = pd.DataFrame(results)

    df = df[
        [
            "split",
            "n_yes",
            "positives_yes",
            "n_no",
            "positives_no",
            "info_gain",
            "gini_gain"
        ]
    ]

    df.to_csv(
        "outputs/part4_splits.csv",
        index=False
    )

    best_info = df.loc[
        df["info_gain"].idxmax(),
        "split"
    ]

    best_gini = df.loc[
        df["gini_gain"].idxmax(),
        "split"
    ]

    return (
        entropy(y_train),
        gini(y_train),
        best_info,
        best_gini,
        df
    )


# ==========================================
# TASK 4.2 - VOTING MODELS
# ==========================================

K = 5


def calculate_metrics(y_true, pred):

    tp = int(np.sum((y_true == 1) & (pred == 1)))

    fp = int(np.sum((y_true == 0) & (pred == 1)))

    tn = int(np.sum((y_true == 0) & (pred == 0)))

    fn = int(np.sum((y_true == 1) & (pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0

    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def task_4_2(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
):

    # Logistic Regression

    weights, bias = train_logistic(
        X_train_scaled,
        y_train
    )

    logistic_prob = sigmoid(
        X_test_scaled @ weights + bias
    )

    logistic_pred = (
        logistic_prob >= 0.3
    ).astype(int)


    # KNN

    knn_prob = knn_probabilities(
        X_train_scaled,
        y_train,
        X_test_scaled,
        K
    )

    knn_pred = (
        knn_prob >= (K + 1) / 2 / K
    ).astype(int)


    # Decision Stump

    conditions = {

        "S1": lambda X: X[:, 5] == 1,

        "S2": lambda X: X[:, 6] == 1,

        "S3": lambda X: X[:, 2] > 0.5,

        "S4": lambda X: X[:, 0] <= 24

    }

    split_results = []

    for name, condition in conditions.items():

        result = evaluate_split(
            X_train,
            y_train,
            condition
        )

        split_results.append({
            "split": name,
            **result
        })

    best_split = max(
        split_results,
        key=lambda x: x["info_gain"]
    )["split"]

    condition = conditions[best_split]

    train_mask = condition(X_train)

    test_mask = condition(X_test)

    stump_rate_yes = np.mean(
        y_train[train_mask]
    )

    stump_rate_no = np.mean(
        y_train[~train_mask]
    )

    stump_prob = np.where(
        test_mask,
        stump_rate_yes,
        stump_rate_no
    )

    stump_pred = (
        stump_prob >= 0.3
    ).astype(int)


    # Hard Voting

    hard_vote_pred = (
        logistic_pred
        + knn_pred
        + stump_pred >= 2
    ).astype(int)


    # Soft Voting

    soft_prob = (
        logistic_prob
        + knn_prob
        + stump_prob
    ) / 3

    soft_vote_pred = (
        soft_prob >= 0.3
    ).astype(int)


    # Evaluation

    models = {

        "logistic": logistic_pred,

        "knn": knn_pred,

        "stump": stump_pred,

        "hard_vote": hard_vote_pred,

        "soft_vote": soft_vote_pred

    }

    results = []

    for name, pred in models.items():

        results.append({
            "model": name,
            **calculate_metrics(y_test, pred)
        })

    voting_df = pd.DataFrame(results)

    voting_df.to_csv(
        "outputs/part4_voting.csv",
        index=False
    )

    return (
        stump_rate_yes,
        stump_rate_no,
        voting_df
    )


# ==========================================
# TASK 4.3 - ADABOOST
# ==========================================

def task_4_3():

    y = np.array([
        1, 1, 1, -1, -1, -1, 1, -1
    ])

    h1 = np.array([
        1, 1, -1, -1, -1, -1, -1, -1
    ])

    h2 = np.array([
        -1, 1, 1, -1, 1, -1, 1, -1
    ])

    weights = np.ones(8) / 8

    rules = [h1, h2]

    errors = []
    alphas = []

    rows = []

    for round_no, h in enumerate(rules, start=1):

        weight_before = weights.copy()

        incorrect = (h != y)

        error = np.sum(
            weights[incorrect]
        )

        alpha = 0.5 * np.log(
            (1 - error) / error
        )

        unnormalised = (
            weights * np.exp(-alpha * y * h)
        )

        normalised = (
            unnormalised / np.sum(unnormalised)
        )

        for i in range(8):

            rows.append({

                "round": round_no,

                "point": i + 1,

                "weight_before": weight_before[i],

                "weight_unnormalised": unnormalised[i],

                "weight_after": normalised[i]

            })

        errors.append(error)

        alphas.append(alpha)

        weights = normalised


    # Final ensemble

    final_score = (
        alphas[0] * h1
        + alphas[1] * h2
    )

    final_prediction = np.sign(final_score)

    ensemble_accuracy = np.mean(
        final_prediction == y
    )

    adaboost_df = pd.DataFrame(rows)

    adaboost_df.to_csv(
        "outputs/part4_adaboost.csv",
        index=False
    )

    return (
        errors[0],
        alphas[0],
        errors[1],
        alphas[1],
        ensemble_accuracy,
        adaboost_df
    )


# ==========================================
# TASK 4.4 - OUT OF BAG SHARE
# ==========================================

def task_4_4():

    return (1 - 1 / 560) ** 560


# ==========================================
# TASK 4.5 - NEURAL NETWORK ARCHITECTURE
# ==========================================

def task_4_5():

    # Model A - MLP

    mlp_hidden1 = 784 * 16 + 16

    mlp_hidden2 = 16 * 16 + 16

    mlp_output = 16 * 10 + 10

    mlp_total = (
        mlp_hidden1
        + mlp_hidden2
        + mlp_output
    )


    # Model A - CNN

    cnn_conv1 = (5 * 5 * 1 + 1) * 6

    cnn_conv2 = (5 * 5 * 6 + 1) * 16

    cnn_dense1 = 400 * 32 + 32

    cnn_dense2 = 32 * 10 + 10

    cnn_total = (
        cnn_conv1
        + cnn_conv2
        + cnn_dense1
        + cnn_dense2
    )


    # Architecture CSV

    architecture = [

        ("mlp_hidden1", mlp_hidden1),

        ("mlp_hidden2", mlp_hidden2),

        ("mlp_output", mlp_output),

        ("cnn_conv1", cnn_conv1),

        ("cnn_conv2", cnn_conv2),

        ("cnn_dense1", cnn_dense1),

        ("cnn_dense2", cnn_dense2)

    ]

    architecture_df = pd.DataFrame(
        architecture,
        columns=["layer", "parameters"]
    )

    architecture_df.to_csv(
        "outputs/part4_architecture.csv",
        index=False
    )


    # Fully connected vs convolution

    fc_layer_params = (
        128 * 128 * 3 * 1024 + 1024
    )

    conv_layer_params = (
        3 * 3 * 3 * 64 + 64
    )


    # Model B - RNN

    wxh = 64 * 128

    whh = 128 * 128

    why = 128 * 16

    bh = 128

    by = 16

    rnn_total = (
        wxh + whh + why + bh + by
    )

    return {
        "mlp_total": mlp_total,
        "cnn_total": cnn_total,
        "fc_layer_params": fc_layer_params,
        "conv_layer_params": conv_layer_params,
        "rnn_total": rnn_total,
        "wxh": wxh,
        "whh": whh,
        "why": why
    }


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("\nLoading Part 2 data split...")

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    ) = load_data()

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))


    # --------------------------------------
    # TASK 4.1
    # --------------------------------------

    (
        root_entropy,
        root_gini,
        best_info,
        best_gini,
        split_df
    ) = task_4_1(
        X_train,
        y_train
    )

    results = {}

    results["root_entropy"] = root_entropy

    results["root_gini"] = root_gini

    results["best_split_by_info_gain"] = best_info

    results["best_split_by_gini_gain"] = best_gini

    print("\n========== TASK 4.1 ==========")

    print("Root Entropy:", round(root_entropy, 4))

    print("Root Gini:", round(root_gini, 4))

    print("Best Information Gain:", best_info)

    print("Best Gini Gain:", best_gini)

    print(split_df)


    # --------------------------------------
    # TASK 4.2
    # --------------------------------------

    (
        stump_yes,
        stump_no,
        voting_df
    ) = task_4_2(
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    )

    results["stump_rate_yes"] = stump_yes

    results["stump_rate_no"] = stump_no

    print("\n========== TASK 4.2 ==========")

    print("Stump Yes Rate:", stump_yes)

    print("Stump No Rate:", stump_no)

    print(voting_df)


    # --------------------------------------
    # TASK 4.3
    # --------------------------------------

    (
        round1_error,
        round1_alpha,
        round2_error,
        round2_alpha,
        ensemble_accuracy,
        adaboost_df
    ) = task_4_3()

    results["round1_error"] = round1_error

    results["round1_alpha"] = round1_alpha

    results["round2_error"] = round2_error

    results["round2_alpha"] = round2_alpha

    results["ensemble_accuracy"] = ensemble_accuracy

    print("\n========== TASK 4.3 ==========")

    print("Round 1 Error:", round1_error)

    print("Round 1 Alpha:", round1_alpha)

    print("Round 2 Error:", round2_error)

    print("Round 2 Alpha:", round2_alpha)

    print("Ensemble Accuracy:", ensemble_accuracy)

    print(adaboost_df)


    # --------------------------------------
    # TASK 4.4
    # --------------------------------------

    oob_fraction = task_4_4()

    results["oob_fraction"] = oob_fraction

    print("\n========== TASK 4.4 ==========")

    print("OOB Fraction:", oob_fraction)


    # --------------------------------------
    # TASK 4.5
    # --------------------------------------

    architecture_results = task_4_5()

    results.update(architecture_results)

    print("\n========== TASK 4.5 ==========")

    for key, value in architecture_results.items():

        print(key, "=", value)


    # --------------------------------------
    # SAVE ALL RESULTS
    # --------------------------------------

    with open("outputs/part4_results.txt", "w") as f:

        for key, value in results.items():

            if isinstance(value, (float, np.floating)):

                f.write(f"{key} = {value:.6f}\n")

            else:

                f.write(f"{key} = {value}\n")


    print("\n===================================")

    print("PART 4 TASKS 4.1 TO 4.5 COMPLETED!")

    print("===================================")

    print("\nGenerated output files:")

    print("1. outputs/part4_results.txt")

    print("2. outputs/part4_splits.csv")

    print("3. outputs/part4_voting.csv")

    print("4. outputs/part4_adaboost.csv")

    print("5. outputs/part4_architecture.csv")


if __name__ == "__main__":
    main()