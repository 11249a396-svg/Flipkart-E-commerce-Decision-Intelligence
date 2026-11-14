import csv
import numpy as np

# Load dataset
with open("data/records.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Extract order values
values = np.array([
    float(row["order_value_inr_hundreds"])
    for row in data
])

# Task 1.1: Statistical Estimation

n = len(values)

mean_value = np.mean(values)
sd_value = np.std(values, ddof=1)

se_value = sd_value / np.sqrt(n)

ci95_low = mean_value - 1.96 * se_value

ci95_high = mean_value + 1.96 * se_value

# Display results
print("TASK 1.1 - STATISTICAL ESTIMATION")

print("n =", n)
print("Mean =", round(mean_value, 4))
print("Standard Deviation =", round(sd_value, 4))
print("Standard Error =", round(se_value, 4))

print("95% Confidence Interval:")
print("Lower =", round(ci95_low, 4))
print("Upper =", round(ci95_high, 4))

# Save results
with open("outputs/part1_results.txt", "w") as f:

    f.write(f"n = {n}\n")
    f.write(f"mean_value = {mean_value:.4f}\n")
    f.write(f"sd_value = {sd_value:.4f}\n")
    f.write(f"se_value = {se_value:.4f}\n")
    f.write(f"ci95_low = {ci95_low:.4f}\n")
    f.write(f"ci95_high = {ci95_high:.4f}\n")

    # TASK 1.2 - HYPOTHESIS TESTING

z_stat = (mean_value - 8.0) / se_value

reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

print("\nTASK 1.2 - HYPOTHESIS TESTING")

print("Z Statistic =", round(z_stat, 4))
print("Critical Value = +/- 1.96")
print("Reject H0 =", reject_h0)

# Append results
with open("outputs/part1_results.txt", "a") as f:

    f.write(f"z_stat = {z_stat:.4f}\n")
    f.write(f"reject_h0 = {reject_h0}\n")

    import math

E = 0.5

n_required = math.ceil((1.96 * sd_value / E) ** 2)

print("\nTask 1.3 - Sample Size Calculation")
print("Required Sample Size:", n_required)

with open("outputs/part1_results.txt", "a") as f:
    f.write(f"n_required = {n_required}\n")

    # Task 1.4 - Simple Linear Regression

x = np.array([
    float(row["avg_order_value_inr_hundreds"])
    for row in data
])

y = values

n = len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope and intercept
covariance = np.sum((x - x_mean) * (y - y_mean)) / (n - 1)
variance = np.sum((x - x_mean) ** 2) / (n - 1)

slope = covariance / variance
intercept = y_mean - slope * x_mean

# Predicted values
fitted = intercept + slope * x

# Calculate errors
residuals = y - fitted

tss = np.sum((y - y_mean) ** 2)
ess = np.sum((fitted - y_mean) ** 2)
rss = np.sum(residuals ** 2)

# Model evaluation
r2 = 1 - (rss / tss)
rmse = np.sqrt(rss / n)
mae = np.mean(np.abs(residuals))

tss_check = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"

print("\nTask 1.4 - Simple Linear Regression")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"TSS: {tss:.4f}")
print(f"ESS: {ess:.4f}")
print(f"RSS: {rss:.4f}")
print(f"TSS Check: {tss_check}")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

with open("outputs/part1_results.txt", "a") as f:
    f.write(f"reg_slope = {slope:.4f}\n")
    f.write(f"reg_intercept = {intercept:.4f}\n")
    f.write(f"tss = {tss:.4f}\n")
    f.write(f"ess = {ess:.4f}\n")
    f.write(f"rss = {rss:.4f}\n")
    f.write(f"tss_equals_ess_plus_rss = {tss_check}\n")
    f.write(f"r2 = {r2:.4f}\n")
    f.write(f"rmse = {rmse:.4f}\n")
    f.write(f"mae = {mae:.4f}\n")

    # ==========================================
# TASK 1.5 - GRADIENT DESCENT
# ==========================================

# Standardize X4
x_sd = np.std(x, ddof=1)

z = (x - x_mean) / x_sd

# Initialize parameters
m = 0.0
c = 0.0

# Learning rate
eta = 0.1

# Total records
n = len(z)

# Initial cost
y_pred = m * z + c
errors = y - y_pred

gd_J0 = np.mean(errors ** 2)

# Store costs
costs = {}

# Gradient Descent: 500 iterations
for i in range(1, 501):

    # Predictions
    y_pred = m * z + c

    # Errors
    errors = y - y_pred

    # Calculate gradients
    dm = (-2 / n) * np.sum(z * errors)

    dc = (-2 / n) * np.sum(errors)

    # Update parameters
    m = m - eta * dm

    c = c - eta * dc

    # Calculate cost after update
    y_pred = m * z + c

    errors = y - y_pred

    cost = np.mean(errors ** 2)

    # Store required checkpoints
    if i in [1, 10, 100, 500]:
        costs[i] = cost


# Convert standardized regression to original scale

gd_slope = m / x_sd

gd_intercept = c - (m * x_mean / x_sd)

# Compare with closed-form regression
gd_slope_gap = abs(gd_slope - slope)


# Display results

print("\nTASK 1.5 - GRADIENT DESCENT")

print("GD J0 =", round(gd_J0, 4))
print("GD J1 =", round(costs[1], 4))
print("GD J10 =", round(costs[10], 4))
print("GD J100 =", round(costs[100], 4))
print("GD J500 =", round(costs[500], 4))

print("GD Slope =", round(gd_slope, 4))
print("GD Intercept =", round(gd_intercept, 4))
print("GD Slope Gap =", round(gd_slope_gap, 4))


# Save results

with open("outputs/part1_results.txt", "a") as f:

    f.write(f"gd_J0 = {gd_J0:.4f}\n")
    f.write(f"gd_J1 = {costs[1]:.4f}\n")
    f.write(f"gd_J10 = {costs[10]:.4f}\n")
    f.write(f"gd_J100 = {costs[100]:.4f}\n")
    f.write(f"gd_J500 = {costs[500]:.4f}\n")

    f.write(f"gd_slope = {gd_slope:.4f}\n")
    f.write(f"gd_intercept = {gd_intercept:.4f}\n")
    f.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")

    # ==========================================
# TASK 1.6 - RESIDUAL ANALYSIS AND LOG TRANSFORMATION
# ==========================================

# PART A: Raw Model Residual Analysis

# Stable sorting based on fitted values
sorted_indices = np.argsort(fitted, kind="stable")

# Divide into lower and upper groups
lower_indices = sorted_indices[:400]
upper_indices = sorted_indices[400:]

# Calculate residual standard deviations
resid_sd_lower_raw = np.std(residuals[lower_indices], ddof=1)

resid_sd_upper_raw = np.std(residuals[upper_indices], ddof=1)

# Calculate residual SD ratio
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw


# PART B: Log Transformation

# Apply natural logarithm
log_x = np.log(x)
log_y = np.log(y)

# Calculate means
log_x_mean = np.mean(log_x)
log_y_mean = np.mean(log_y)

# Calculate sample covariance and variance
log_cov = np.sum(
    (log_x - log_x_mean) * (log_y - log_y_mean)
) / (n - 1)

log_var = np.sum(
    (log_x - log_x_mean) ** 2
) / (n - 1)

# Calculate regression coefficients
loglog_slope = log_cov / log_var

loglog_intercept = log_y_mean - loglog_slope * log_x_mean

# Calculate fitted values
log_fitted = loglog_intercept + loglog_slope * log_x

# Calculate residuals
log_residuals = log_y - log_fitted

# Calculate evaluation metrics
log_tss = np.sum((log_y - log_y_mean) ** 2)

log_rss = np.sum(log_residuals ** 2)

loglog_r2 = 1 - (log_rss / log_tss)

loglog_rmse = np.sqrt(log_rss / n)


# PART C: Log Model Residual Analysis

# Stable sorting
log_sorted_indices = np.argsort(log_fitted, kind="stable")

# Divide into lower and upper groups
log_lower_indices = log_sorted_indices[:400]
log_upper_indices = log_sorted_indices[400:]

# Calculate residual standard deviations
resid_sd_lower_log = np.std(
    log_residuals[log_lower_indices], ddof=1
)

resid_sd_upper_log = np.std(
    log_residuals[log_upper_indices], ddof=1
)

# Calculate ratio
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log


# PART D: Display Results

print("\nTASK 1.6 - RESIDUAL ANALYSIS")

print("Raw Lower SD =", round(resid_sd_lower_raw, 4))
print("Raw Upper SD =", round(resid_sd_upper_raw, 4))
print("Raw SD Ratio =", round(resid_sd_ratio_raw, 4))

print("\nLOG TRANSFORMATION")

print("Log Intercept =", round(loglog_intercept, 4))
print("Log Slope =", round(loglog_slope, 4))
print("Log R2 =", round(loglog_r2, 4))
print("Log RMSE =", round(loglog_rmse, 4))

print("Log Lower SD =", round(resid_sd_lower_log, 4))
print("Log Upper SD =", round(resid_sd_upper_log, 4))
print("Log SD Ratio =", round(resid_sd_ratio_log, 4))


# PART E: Save Results

with open("outputs/part1_results.txt", "a") as f:

    f.write(f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}\n")
    f.write(f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}\n")
    f.write(f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}\n")

    f.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
    f.write(f"loglog_slope = {loglog_slope:.4f}\n")
    f.write(f"loglog_r2 = {loglog_r2:.4f}\n")
    f.write(f"loglog_rmse = {loglog_rmse:.4f}\n")

    f.write(f"resid_sd_lower_log = {resid_sd_lower_log:.4f}\n")
    f.write(f"resid_sd_upper_log = {resid_sd_upper_log:.4f}\n")
    f.write(f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}\n")

    # ==========================================
# TASK 1.7 - MULTIPLE LINEAR REGRESSION
# ==========================================

# Extract predictor columns

X1 = np.array([
    float(row["customer_tenure_months"])
    for row in data
])

X2 = np.array([
    float(row["complaint_count_90d"])
    for row in data
])

X3 = np.array([
    float(row["discount_order_share"])
    for row in data
])

X5 = np.array([
    float(row["loyalty_score"])
    for row in data
])

X6 = np.array([
    float(row["cod_flag"])
    for row in data
])

X7 = np.array([
    float(row["plus_member_flag"])
    for row in data
])


# Construct design matrix in exact order

X_matrix = np.column_stack((
    np.ones(n),
    log_x,
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
))

# Target variable
Y = log_y

# Normal Equation
XtX = X_matrix.T @ X_matrix

XtY = X_matrix.T @ Y

beta = np.linalg.solve(XtX, XtY)

# Predicted values
Y_pred = X_matrix @ beta

# Calculate residuals
mr_residuals = Y - Y_pred

# Calculate TSS and RSS
mr_tss = np.sum((Y - np.mean(Y)) ** 2)

mr_rss = np.sum(mr_residuals ** 2)

# Calculate R2
mr_r2 = 1 - (mr_rss / mr_tss)

# Adjusted R2
p = 7

mr_adj_r2 = 1 - (
    (1 - mr_r2) * (n - 1) / (n - p - 1)
)

# Adjusted R2 for log-log model
loglog_adj_r2 = 1 - (
    (1 - loglog_r2) * (n - 1) / (n - 1 - 1)
)


# Display results

print("\nTASK 1.7 - MULTIPLE LINEAR REGRESSION")

for i in range(8):
    print(f"Beta {i} =", round(beta[i], 4))

print("Multiple Regression R2 =", round(mr_r2, 4))

print("Multiple Regression Adjusted R2 =", round(mr_adj_r2, 4))

print("Log-Log Adjusted R2 =", round(loglog_adj_r2, 4))


# Save results

with open("outputs/part1_results.txt", "a") as f:

    for i in range(8):
        f.write(f"mr_beta_{i} = {beta[i]:.4f}\n")

    f.write(f"mr_r2 = {mr_r2:.4f}\n")

    f.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")

    f.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")
