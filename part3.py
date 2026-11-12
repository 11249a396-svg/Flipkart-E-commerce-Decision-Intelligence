import numpy as np
import pandas as pd
import os

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("data/records.csv")

# Select five continuous predictors
features = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score"
]

X = df[features].values.astype(float)

# =========================================
# TASK 3.1: STANDARDISATION
# =========================================

mean_values = np.mean(X, axis=0)
std_values = np.std(X, axis=0, ddof=1)

X_std = (X - mean_values) / std_values

print("Dataset shape:", X_std.shape)
print("Standardisation completed!")


# =========================================
# TASK 3.1: K-MEANS FROM SCRATCH
# =========================================

def kmeans_from_scratch(X, K):

    n = X.shape[0]

    # Initial centroid indices
    step = n // K
    indices = [i * step for i in range(K)]

    centroids = X[indices].copy()

    # Initially no cluster assignments
    labels = np.full(n, -1)

    iterations = 0

    while True:

        iterations += 1

        old_labels = labels.copy()

        # Calculate squared Euclidean distances
        distances = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # Assign rows to nearest centroid
        # Ties go to the lower centroid index
        labels = np.argmin(distances, axis=1)

        # Update centroids
        for k in range(K):

            members = X[labels == k]

            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

        # Stop when no assignments change
        if np.array_equal(labels, old_labels):
            break

    # Calculate final WCSS
    wcss = 0.0

    for k in range(K):

        members = X[labels == k]

        if len(members) > 0:

            distances = np.sum(
                (members - centroids[k]) ** 2,
                axis=1
            )

            wcss += np.sum(distances)

    return labels, centroids, wcss, iterations


# =========================================
# RUN K-MEANS FOR K = 2, 3, 4, 5, 6
# =========================================

kmeans_results = {}

for K in range(2, 7):

    labels, centroids, wcss_value, iterations = kmeans_from_scratch(
        X_std, K
    )

    kmeans_results[K] = {
        "labels": labels,
        "centroids": centroids,
        "wcss": wcss_value,
        "iterations": iterations
    }

    print(f"\nK = {K}")
    print(f"WCSS = {wcss_value:.4f}")
    print(f"Iterations = {iterations}")


# =========================================
# TASK 3.2: ELBOW METHOD
# =========================================

wcss_values = {
    K: kmeans_results[K]["wcss"]
    for K in range(2, 7)
}

elbow_k = None

print("\n========== ELBOW METHOD ==========")

for K in range(3, 6):

    previous_drop = wcss_values[K - 1] - wcss_values[K]

    current_drop = wcss_values[K] - wcss_values[K + 1]

    print(f"\nK = {K}")

    print(f"Previous WCSS drop = {previous_drop:.4f}")

    print(f"Current WCSS drop = {current_drop:.4f}")

    if current_drop < 0.5 * previous_drop:

        elbow_k = K
        break


if elbow_k is None:
    elbow_k = 5


print("\nElbow K =", elbow_k)


# Assignment requires K=3 for remaining tasks
selected_K = 3

print("Selected K for remaining tasks =", selected_K)

# =========================================
# TASK 3.3: CUSTOMER SEGMENT PROFILING
# =========================================

print("\n========== CUSTOMER SEGMENT PROFILING ==========")

# Get K=3 cluster labels
labels = kmeans_results[selected_K]["labels"]

# Calculate original X3 mean for each cluster
cluster_means = {}

for cluster_id in range(selected_K):

    members = df.loc[labels == cluster_id]

    cluster_means[cluster_id] = members["discount_order_share"].mean()


# Sort clusters by ascending mean X3
sorted_clusters = sorted(
    cluster_means,
    key=cluster_means.get
)

# Assign segment names A, B, C
segment_names = ["A", "B", "C"]

segment_rows = []

for segment_name, cluster_id in zip(segment_names, sorted_clusters):

    members = df.loc[labels == cluster_id]

    segment_data = {
        "segment": segment_name,
        "size": len(members),
        "mean_X1": members["customer_tenure_months"].mean(),
        "mean_X2": members["complaint_count_90d"].mean(),
        "mean_X3": members["discount_order_share"].mean(),
        "mean_X4": members["avg_order_value_inr_hundreds"].mean(),
        "mean_X5": members["loyalty_score"].mean(),
        "outcome_rate": members["churned"].mean(),
        "mean_value": members["order_value_inr_hundreds"].mean()
    }

    segment_rows.append(segment_data)

    print(f"\nSegment {segment_name}")
    print("Size:", segment_data["size"])
    print("Mean X1:", round(segment_data["mean_X1"], 4))
    print("Mean X2:", round(segment_data["mean_X2"], 4))
    print("Mean X3:", round(segment_data["mean_X3"], 4))
    print("Mean X4:", round(segment_data["mean_X4"], 4))
    print("Mean X5:", round(segment_data["mean_X5"], 4))
    print("Outcome Rate:", round(segment_data["outcome_rate"], 4))
    print("Mean Value:", round(segment_data["mean_value"], 4))


# Create output directory
os.makedirs("outputs", exist_ok=True)

# Save segment results
segments_df = pd.DataFrame(segment_rows)

segments_df = segments_df[
    [
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ]
]

segments_df.to_csv(
    "outputs/part3_segments.csv",
    index=False,
    float_format="%.6f"
)

print("\nSegment results saved successfully!")
print("File: outputs/part3_segments.csv")

print("\nTotal customers:", segments_df["size"].sum())

# =========================================
# TASK 3.4: HIERARCHICAL CLUSTERING
# =========================================

print("\n========== HIERARCHICAL CLUSTERING ==========")

# Pairwise dissimilarity matrix

distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]


def hierarchical_clustering(distance_matrix, linkage):

    clusters = [[i] for i in range(6)]

    merge_rows = []

    heights = []

    merge_no = 1

    while len(clusters) > 1:

        best_distance = float("inf")

        best_i = None
        best_j = None

        # Find closest pair of clusters
        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                distances = []

                for a in clusters[i]:

                    for b in clusters[j]:

                        distances.append(distance_matrix[a, b])

                if linkage == "single":
                    cluster_distance = min(distances)

                elif linkage == "complete":
                    cluster_distance = max(distances)

                elif linkage == "average":
                    cluster_distance = np.mean(distances)

                if cluster_distance < best_distance:

                    best_distance = cluster_distance

                    best_i = i
                    best_j = j

        cluster_a = clusters[best_i]
        cluster_b = clusters[best_j]

        # Sort cluster names
        names_a = "+".join(
            sorted(cohorts[i] for i in cluster_a)
        )

        names_b = "+".join(
            sorted(cohorts[i] for i in cluster_b)
        )

        # Ensure cluster_a has smaller minimum member
        if min(cluster_a) > min(cluster_b):

            cluster_a, cluster_b = cluster_b, cluster_a

            names_a, names_b = names_b, names_a

        merge_rows.append({
            "linkage": linkage,
            "merge_no": merge_no,
            "cluster_a": names_a,
            "cluster_b": names_b,
            "height": best_distance
        })

        heights.append(best_distance)

        # Merge clusters
        new_cluster = sorted(cluster_a + cluster_b)

        clusters.remove(cluster_a)
        clusters.remove(cluster_b)

        clusters.append(new_cluster)

        merge_no += 1

    return merge_rows, heights


# Run all three linkage methods

all_hierarchical_rows = []

cut_results = {}

for linkage in ["single", "complete", "average"]:

    rows, heights = hierarchical_clustering(
        distance_matrix, linkage
    )

    all_hierarchical_rows.extend(rows)

    # Calculate consecutive height gaps
    gaps = np.diff(heights)

    largest_gap_index = int(np.argmax(gaps))

    # Number of merges below the cut
    merges_below_cut = largest_gap_index + 1

    clusters_at_cut = 6 - merges_below_cut

    cut_results[linkage] = clusters_at_cut

    print(f"\n{linkage.upper()} LINKAGE")

    for row in rows:
        print(row)

    print("Merge heights:", heights)

    print("Clusters at cut:", clusters_at_cut)


# Save hierarchical clustering results

hierarchical_df = pd.DataFrame(all_hierarchical_rows)

hierarchical_df.to_csv(
    "outputs/part3_hierarchical.csv",
    index=False,
    float_format="%.6f"
)

print("\nHierarchical results saved successfully!")

print("File: outputs/part3_hierarchical.csv")

print("\nClusters at cut:")

print("Single:", cut_results["single"])

print("Complete:", cut_results["complete"])

print("Average:", cut_results["average"])

# =========================================
# TASK 3.5: PCA FROM SCRATCH
# =========================================

print("\n========== PCA ANALYSIS ==========")

# STEP 1: Mean vector
pca_mean = np.mean(X_std, axis=0)

print("\nStep 1: Mean Vector")
print(pca_mean)


# STEP 2: Centre the data
X_centered = X_std - pca_mean

print("\nStep 2: Data Centred")
print("Shape:", X_centered.shape)


# STEP 3: Calculate covariance matrix
n = X_centered.shape[0]

cov_matrix = (X_centered.T @ X_centered) / (n - 1)

print("\nStep 3: Covariance Matrix")
print(cov_matrix)


# STEP 4: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("\nStep 4: Eigenvalues")
print(eigenvalues)


# STEP 5: Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]

eigenvectors = eigenvectors[:, sorted_indices]


# Apply sign convention
for i in range(eigenvectors.shape[1]):

    max_index = np.argmax(np.abs(eigenvectors[:, i]))

    if eigenvectors[max_index, i] < 0:

        eigenvectors[:, i] *= -1


print("\nStep 5: Sorted Eigenvalues")
print(eigenvalues)


# STEP 6: Select top two eigenvectors
W = eigenvectors[:, :2]

print("\nStep 6: Top Two Eigenvectors")
print(W)


# STEP 7: Project data
Z = X_centered @ W

print("\nStep 7: PCA Projection")
print("Projection Shape:", Z.shape)


# =========================================
# PCA RESULTS
# =========================================

eigenvalue_sum = np.sum(eigenvalues)

explained_ratios = eigenvalues / eigenvalue_sum

cumulative_ratios = np.cumsum(explained_ratios)

components_for_80pct = np.argmax(
    cumulative_ratios >= 0.80
) + 1


print("\n========== PCA RESULTS ==========")

print("Eigenvalue Sum:", round(eigenvalue_sum, 4))

for i in range(5):

    print(
        f"Eigenvalue {i+1}:",
        round(eigenvalues[i], 6)
    )

    print(
        f"Explained Ratio {i+1}:",
        round(explained_ratios[i], 6)
    )

    print(
        f"Cumulative Ratio {i+1}:",
        round(cumulative_ratios[i], 6)
    )

print("Components for 80% variance:", components_for_80pct)


# =========================================
# PCA LOADINGS
# =========================================

print("\n========== PCA LOADINGS ==========")

for i, feature in enumerate(features):

    print(
        f"PC1 Loading {feature}:",
        round(W[i, 0], 6)
    )

for i, feature in enumerate(features):

    print(
        f"PC2 Loading {feature}:",
        round(W[i, 1], 6)
    )


# =========================================
# FIRST THREE ROW PROJECTIONS
# =========================================

print("\n========== FIRST THREE PCA SCORES ==========")

for i in range(3):

    print(
        f"Row {i+1}: PC1 = {Z[i, 0]:.6f}, PC2 = {Z[i, 1]:.6f}"
    )


# =========================================
# MEAN PC1 BY CUSTOMER SEGMENT
# =========================================

print("\n========== MEAN PC1 BY SEGMENT ==========")

for segment_name, cluster_id in zip(
    segment_names, sorted_clusters
):

    mean_pc1 = np.mean(Z[labels == cluster_id, 0])

    print(
        f"Mean PC1 Segment {segment_name}: {mean_pc1:.6f}"
    )

    # =========================================
# FINAL: SAVE ALL PART 3 RESULTS
# =========================================

results = {}

# TASK 3.1 - KMEANS

for K in range(2, 7):
    results[f"wcss_k{K}"] = kmeans_results[K]["wcss"]

results["elbow_k"] = elbow_k

# TASK 3.3 - SEGMENT SIZES

for row in segment_rows:
    segment = row["segment"]
    results[f"size_segment_{segment}"] = row["size"]

# TASK 3.4 - HIERARCHICAL CLUSTERING

results["single_clusters_at_cut"] = cut_results["single"]
results["complete_clusters_at_cut"] = cut_results["complete"]
results["average_clusters_at_cut"] = cut_results["average"]

# TASK 3.5 - PCA

results["eigenvalue_sum"] = eigenvalue_sum

for i in range(5):

    results[f"eigenvalue_{i+1}"] = eigenvalues[i]

    results[f"explained_ratio_{i+1}"] = explained_ratios[i]

    results[f"cumulative_ratio_{i+1}"] = cumulative_ratios[i]

results["components_for_80pct"] = components_for_80pct

# PC1 LOADINGS

for i in range(5):
    results[f"pc1_loading_X{i+1}"] = W[i, 0]

# PC2 LOADINGS

for i in range(5):
    results[f"pc2_loading_X{i+1}"] = W[i, 1]

# FIRST THREE ROW PROJECTIONS

for i in range(3):

    results[f"row{i+1}_pc1"] = Z[i, 0]

    results[f"row{i+1}_pc2"] = Z[i, 1]

# MEAN PC1 BY SEGMENT

for segment_name, cluster_id in zip(
    segment_names, sorted_clusters
):

    results[f"mean_pc1_segment_{segment_name}"] = np.mean(
        Z[labels == cluster_id, 0]
    )

# SAVE RESULTS

with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            file.write(f"{key} = {value}\n")
        else:
            file.write(f"{key} = {value:.6f}\n")

print("\nPart 3 results saved successfully!")

print("File: outputs/part3_results.txt")

print("Total result keys:", len(results))