import numpy as np
import pandas as pd
import os

# Import reusable functions from common.py
from common import load_data, sigmoid, knn_predict


# =========================================================
# TASK 2.1 - STRATIFIED SPLIT AND STANDARDISATION
# =========================================================

print("--- Dataset Information ---")

df = pd.read_csv("data/records.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# Load split and standardised data using common.py
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


# Checkpoint values
train_rows = len(train_df)
train_positives = int(np.sum(y_train))

test_rows = len(test_df)
test_positives = int(np.sum(y_test))


print("\n--- Task 2.1: Stratified Split ---")
print("train_rows =", train_rows)
print("train_positives =", train_positives)
print("test_rows =", test_rows)
print("test_positives =", test_positives)


# Standardisation information
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0, ddof=1)

print("\n--- Standardisation Results ---")
print("Training Mean:")
print(train_mean)

print("\nTraining Standard Deviation:")
print(train_std)

print("\nFirst 5 Standardised Training Rows:")
print(X_train_scaled[:5])

print("\nFirst 5 Standardised Testing Rows:")
print(X_test_scaled[:5])


# =========================================================
# TASK 2.2 - LOGISTIC REGRESSION FROM SCRATCH
# =========================================================

def binary_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


print("\n--- Logistic Regression Started ---")

n_features = X_train_scaled.shape[1]

weights = np.zeros(n_features)
bias = 0.0

learning_rate = 0.1
epochs = 1000

print("Initial Weights:", weights)
print("Initial Bias:", bias)


# Initial predictions and loss
z = np.dot(X_train_scaled, weights) + bias

y_pred = sigmoid(z)

loss_epoch_0 = binary_cross_entropy(y_train, y_pred)

print("\nLoss at Epoch 0:", loss_epoch_0)


# Gradient Descent
loss_history = {}

for epoch in range(1, epochs + 1):

    z = np.dot(X_train_scaled, weights) + bias

    y_pred = sigmoid(z)

    error = y_pred - y_train

    dw = np.dot(X_train_scaled.T, error) / len(y_train)

    db = np.mean(error)

    weights -= learning_rate * dw

    bias -= learning_rate * db

    if epoch in [1, 10, 100, 1000]:

        z = np.dot(X_train_scaled, weights) + bias

        predictions = sigmoid(z)

        loss = binary_cross_entropy(y_train, predictions)

        loss_history[epoch] = loss

        print(f"Loss at Epoch {epoch}: {loss}")


# Final predictions on test data
z_test = np.dot(X_test_scaled, weights) + bias

y_test_pred = sigmoid(z_test)

test_bce = binary_cross_entropy(y_test, y_test_pred)

print("\nTest BCE:", test_bce)


# Final model parameters
print("\n--- Final Model Parameters ---")

for i, weight in enumerate(weights):

    print(f"Weight X{i+1}: {weight}")

print("Bias:", bias)


# =========================================================
# TASK 2.3 - LEARNING RATE EFFECT
# =========================================================

learning_rates = [0.01, 0.1, 1.0]

learning_rate_results = {}

print("\n--- Task 2.3: Learning Rate Comparison ---")

for lr in learning_rates:

    w = np.zeros(X_train_scaled.shape[1])
    b = 0.0

    for epoch in range(200):

        z = np.dot(X_train_scaled, w) + b

        predictions = sigmoid(z)

        error = predictions - y_train

        dw = np.dot(X_train_scaled.T, error) / len(y_train)

        db = np.mean(error)

        w -= lr * dw

        b -= lr * db

    final_predictions = sigmoid(
        np.dot(X_train_scaled, w) + b
    )

    final_loss = binary_cross_entropy(
        y_train,
        final_predictions
    )

    learning_rate_results[lr] = final_loss

    print(f"Learning Rate {lr}: Loss = {final_loss}")


# =========================================================
# TASK 2.4 - ODDS RATIOS
# =========================================================

print("\n--- Task 2.4: Odds Ratios ---")

odds_ratios = np.exp(weights)

for i, odds in enumerate(odds_ratios):

    print(f"odds_ratio_X{i+1} = {odds:.6f}")


# =========================================================
# TASK 2.5 - MODEL EVALUATION AT TWO THRESHOLDS
# =========================================================

def evaluate_model(y_true, probabilities, threshold):

    y_pred = (probabilities >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))

    fp = np.sum((y_true == 0) & (y_pred == 1))

    tn = np.sum((y_true == 0) & (y_pred == 0))

    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


# Threshold 0.5
results_05 = evaluate_model(
    y_test,
    y_test_pred,
    0.5
)


# Threshold 0.3
results_03 = evaluate_model(
    y_test,
    y_test_pred,
    0.3
)


print("\n--- Task 2.5: Threshold 0.5 ---")

for key, value in results_05.items():

    print(f"tau_0.5_{key} = {value}")


print("\n--- Task 2.5: Threshold 0.3 ---")

for key, value in results_03.items():

    print(f"tau_0.3_{key} = {value}")


# Blind model
blind_accuracy = np.mean(y_test == 0)

print("\nBlind Accuracy =", blind_accuracy)


# =========================================================
# TASK 2.6 - THRESHOLD SWEEP
# =========================================================

print("\n--- Task 2.6: Threshold Sweep ---")

thresholds = np.arange(0.1, 1.0, 0.1)

threshold_results = []

for tau in thresholds:

    y_pred = (y_test_pred >= tau).astype(int)

    tp = np.sum((y_test == 1) & (y_pred == 1))

    fp = np.sum((y_test == 0) & (y_pred == 1))

    tn = np.sum((y_test == 0) & (y_pred == 0))

    fn = np.sum((y_test == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    threshold_results.append({
        "tau": round(tau, 1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": tpr,
        "fpr": fpr
    })


threshold_df = pd.DataFrame(threshold_results)

os.makedirs("outputs", exist_ok=True)

threshold_df.to_csv(
    "outputs/part2_thresholds.csv",
    index=False
)

print("\nThreshold Sweep Results:")
print(threshold_df)

print("\nThreshold CSV saved successfully!")


# =========================================================
# TASK 2.7 - KNN FROM SCRATCH
# =========================================================

# knn_predict is imported from common.py

print("\n--- Task 2.7: KNN Cross Validation ---")

k_values = range(1, 22, 2)

cv_results = []

fold_assignments = np.arange(len(y_train)) % 5


for k in k_values:

    fold_f1_scores = []

    for fold in range(5):

        validation_mask = fold_assignments == fold

        reference_mask = fold_assignments != fold

        X_reference = X_train_scaled[reference_mask]

        y_reference = y_train[reference_mask]

        X_validation = X_train_scaled[validation_mask]

        y_validation = y_train[validation_mask]


        validation_predictions = knn_predict(
            X_reference,
            y_reference,
            X_validation,
            k
        )


        tp = np.sum(
            (y_validation == 1) &
            (validation_predictions == 1)
        )

        fp = np.sum(
            (y_validation == 0) &
            (validation_predictions == 1)
        )

        fn = np.sum(
            (y_validation == 1) &
            (validation_predictions == 0)
        )


        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0 else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 0
        )


        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        fold_f1_scores.append(f1)


    mean_cv_f1 = np.mean(fold_f1_scores)

    cv_results.append({
        "K": k,
        "mean_cv_f1": mean_cv_f1
    })

    print(f"K = {k}, Mean CV F1 = {mean_cv_f1:.6f}")


# Save KNN cross-validation results
knn_cv_df = pd.DataFrame(cv_results)

knn_cv_df.to_csv(
    "outputs/part2_knn_cv.csv",
    index=False
)


# Select best K (ties go to smaller K)
best_index = np.argmax(
    knn_cv_df["mean_cv_f1"].to_numpy()
)

knn_best_k = int(
    knn_cv_df.iloc[best_index]["K"]
)

print("\nBest K =", knn_best_k)

print("\nKNN Cross Validation CSV saved!")


# Final KNN test evaluation
print("\n--- Final KNN Test Evaluation ---")

knn_test_predictions = knn_predict(
    X_train_scaled,
    y_train,
    X_test_scaled,
    knn_best_k
)


knn_tp = int(np.sum(
    (y_test == 1) & (knn_test_predictions == 1)
))

knn_fp = int(np.sum(
    (y_test == 0) & (knn_test_predictions == 1)
))

knn_tn = int(np.sum(
    (y_test == 0) & (knn_test_predictions == 0)
))

knn_fn = int(np.sum(
    (y_test == 1) & (knn_test_predictions == 0)
))


knn_accuracy = (knn_tp + knn_tn) / len(y_test)

knn_precision = (
    knn_tp / (knn_tp + knn_fp)
    if (knn_tp + knn_fp) > 0 else 0
)

knn_recall = (
    knn_tp / (knn_tp + knn_fn)
    if (knn_tp + knn_fn) > 0 else 0
)

knn_f1 = (
    2 * knn_precision * knn_recall /
    (knn_precision + knn_recall)
    if (knn_precision + knn_recall) > 0 else 0
)


print("knn_best_k =", knn_best_k)

print("knn_tp =", knn_tp)

print("knn_fp =", knn_fp)

print("knn_tn =", knn_tn)

print("knn_fn =", knn_fn)

print("knn_accuracy =", knn_accuracy)

print("knn_precision =", knn_precision)

print("knn_recall =", knn_recall)

print("knn_f1 =", knn_f1)


# =========================================================
# TASK 2.8 - MONEY-BASED ALERT THRESHOLD
# =========================================================

print("\n--- Task 2.8: Money-Based Threshold ---")


# Training probabilities
train_probabilities = sigmoid(
    np.dot(X_train_scaled, weights) + bias
)


# Business cost function
def calculate_cost(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    fp = np.sum(
        (y_true == 0) & (predictions == 1)
    )

    fn = np.sum(
        (y_true == 1) & (predictions == 0)
    )

    cost = (50 * fp) + (400 * fn)

    return int(cost), int(fp), int(fn)


# Find best threshold
best_cost = float("inf")

cost_tau = None


for k in range(1, 20):

    tau = k / 20

    cost, fp, fn = calculate_cost(
        y_train,
        train_probabilities,
        tau
    )

    print(
        f"Threshold = {tau:.2f}, "
        f"Training Cost = Rs.{cost}, "
        f"FP = {fp}, FN = {fn}"
    )

    if cost < best_cost:

        best_cost = cost

        cost_tau = tau


cost_train_at_tau = best_cost


# Evaluate selected threshold on test data
cost_test_at_tau, test_fp, test_fn = calculate_cost(
    y_test,
    y_test_pred,
    cost_tau
)


# Evaluate threshold 0.5
cost_test_at_0_5, _, _ = calculate_cost(
    y_test,
    y_test_pred,
    0.5
)


# Cost of flagging nobody
cost_test_flag_nobody = int(
    400 * np.sum(y_test == 1)
)


print("\n--- Final Business Cost Results ---")

print("cost_tau =", cost_tau)

print("cost_train_at_tau =", cost_train_at_tau)

print("cost_test_at_tau =", cost_test_at_tau)

print("cost_test_at_0.5 =", cost_test_at_0_5)

print("cost_test_flag_nobody =", cost_test_flag_nobody)


# =========================================================
# FINAL STEP - SAVE ALL PART 2 RESULTS
# =========================================================

print("\n--- Saving Part 2 Results ---")

results = {}


# Task 2.1 - Split
results["train_rows"] = train_rows

results["train_positives"] = train_positives

results["test_rows"] = test_rows

results["test_positives"] = test_positives


# Task 2.2 - Logistic Regression Loss
results["loss_epoch_0"] = loss_epoch_0

for epoch in [1, 10, 100, 1000]:

    results[f"loss_epoch_{epoch}"] = loss_history[epoch]


# Final weights
for i, weight in enumerate(weights):

    results[f"weight_X{i+1}"] = weight


results["bias"] = bias

results["test_bce"] = test_bce


# Task 2.3 - Learning Rate
results["loss_epoch_200_eta_0.01"] = learning_rate_results[0.01]

results["loss_epoch_200_eta_0.1"] = learning_rate_results[0.1]

results["loss_epoch_200_eta_1.0"] = learning_rate_results[1.0]


# Task 2.4 - Odds Ratios
for i, odds in enumerate(odds_ratios):

    results[f"odds_ratio_X{i+1}"] = odds


# Task 2.5 - Logistic Regression Evaluation
for key, value in results_05.items():

    results[f"tau_0.5_{key}"] = value


for key, value in results_03.items():

    results[f"tau_0.3_{key}"] = value


results["blind_accuracy"] = blind_accuracy


# Task 2.7 - KNN Results
results["knn_best_k"] = knn_best_k

results["knn_tp"] = knn_tp

results["knn_fp"] = knn_fp

results["knn_tn"] = knn_tn

results["knn_fn"] = knn_fn

results["knn_accuracy"] = knn_accuracy

results["knn_precision"] = knn_precision

results["knn_recall"] = knn_recall

results["knn_f1"] = knn_f1


# Task 2.8 - Business Cost
results["cost_tau"] = cost_tau

results["cost_train_at_tau"] = cost_train_at_tau

results["cost_test_at_tau"] = cost_test_at_tau

results["cost_test_at_0.5"] = cost_test_at_0_5

results["cost_test_flag_nobody"] = cost_test_flag_nobody


# Save results to text file
os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):

            file.write(f"{key} = {value:.6f}\n")

        else:

            file.write(f"{key} = {value}\n")


print("\nPart 2 Results saved successfully!")

print("Total result keys:", len(results))

print("\nFiles generated:")

print("1. outputs/part2_results.txt")

print("2. outputs/part2_thresholds.csv")

print("3. outputs/part2_knn_cv.csv")