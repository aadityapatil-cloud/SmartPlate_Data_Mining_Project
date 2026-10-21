# ============================================================
# SMARTPLATE - K-MEANS CLUSTERING
# ============================================================
#
# Purpose:
# Group cafeteria records into similar clusters based on
# food preparation, sales, waste, demand, and student data.
#
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# STEP 2: DEFINE FILE PATHS
# ============================================================

# Input cleaned SmartPlate dataset
INPUT_FILE = r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_cleaned.csv"

# Output dataset with cluster labels
OUTPUT_FILE = r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_kmeans_clusters.csv"


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

print("=" * 70)
print("SMARTPLATE - K-MEANS CLUSTERING")
print("=" * 70)

try:

    df = pd.read_csv(INPUT_FILE)

    print("\n Dataset loaded successfully.")
    print("Dataset Shape:", df.shape)

except FileNotFoundError:

    print("\n ERROR: Dataset file not found.")
    print(INPUT_FILE)

    raise SystemExit


# ============================================================
# STEP 4: DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET COLUMNS")
print("=" * 70)

print(df.columns.tolist())


# ============================================================
# STEP 5: SELECT FEATURES FOR K-MEANS
# ============================================================

# These features are useful for identifying cafeteria
# demand and food-waste patterns.

features = [
    "Quantity_Prepared",
    "Quantity_Sold",
    "Quantity_Wasted",
    "Actual_Demand",
    "Number_of_Students",
    "Temperature",
    "Estimated_Cost_of_Waste",
    "Unmet_Demand"
]


# ============================================================
# STEP 6: CHECK REQUIRED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("SELECTED FEATURES")
print("=" * 70)

missing_features = [
    column
    for column in features
    if column not in df.columns
]

if missing_features:

    print(" Missing features:")
    print(missing_features)

    raise SystemExit

else:

    print(" All required features are available.")

    for column in features:
        print("-", column)


# ============================================================
# STEP 7: CREATE FEATURE DATAFRAME
# ============================================================

X = df[features].copy()

print("\nFeature Dataset Shape:", X.shape)


# ============================================================
# STEP 8: HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(X.isnull().sum())

# Replace missing values with median values
X = X.fillna(X.median())

print("\n Missing values handled.")


# ============================================================
# STEP 9: STANDARDIZE FEATURES
# ============================================================

# K-Means uses distance calculations.
# StandardScaler puts all features on a similar scale.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

print(" Features standardized successfully.")


# ============================================================
# STEP 10: FIND BEST NUMBER OF CLUSTERS
# USING ELBOW METHOD
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD")
print("=" * 70)

inertia_values = []

# Test cluster numbers from 2 to 10
for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(
        kmeans.inertia_
    )

    print(
        f"K = {k} | "
        f"Inertia = {kmeans.inertia_:.2f}"
    )


# ============================================================
# STEP 11: PLOT ELBOW CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(2, 11),
    inertia_values,
    marker="o"
)

plt.title(
    "SmartPlate - K-Means Elbow Method"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.xticks(range(2, 11))

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 12: SET NUMBER OF CLUSTERS
# ============================================================

# For SmartPlate, we start with 3 clusters:
#
# Cluster 0 -> Low waste / lower demand pattern
# Cluster 1 -> Medium waste / medium demand pattern
# Cluster 2 -> High waste / higher demand pattern
#
# The actual meaning must be confirmed from cluster averages.

K = 3


print("\n" + "=" * 70)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 70)

print("K =", K)


# ============================================================
# STEP 13: CREATE K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)


# ============================================================
# STEP 14: APPLY K-MEANS CLUSTERING
# ============================================================

# fit_predict() trains the model and assigns
# a cluster number to every record.

cluster_labels = kmeans.fit_predict(
    X_scaled
)

print("\n✅ K-Means clustering completed.")


# ============================================================
# STEP 15: ADD CLUSTER LABELS TO DATASET
# ============================================================

df["Cluster"] = cluster_labels

print("\n" + "=" * 70)
print("CLUSTER LABELS")
print("=" * 70)

print(
    df["Cluster"].value_counts().sort_index()
)


# ============================================================
# STEP 16: CALCULATE SILHOUETTE SCORE
# ============================================================

# Silhouette Score measures how well the data points
# fit within their assigned clusters.

silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)

print(
    f"Silhouette Score: {silhouette:.4f}"
)


# ============================================================
# STEP 17: CLUSTER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER SUMMARY")
print("=" * 70)

cluster_summary = df.groupby(
    "Cluster"
)[features].mean()

print(
    cluster_summary.round(2)
)


# ============================================================
# STEP 18: CLUSTER RECORD COUNT
# ============================================================

print("\n" + "=" * 70)
print("RECORDS IN EACH CLUSTER")
print("=" * 70)

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


# ============================================================
# STEP 19: IDENTIFY CLUSTER CHARACTERISTICS
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER CHARACTERISTICS")
print("=" * 70)

for cluster_number in sorted(
    df["Cluster"].unique()
):

    cluster_data = df[
        df["Cluster"] == cluster_number
    ]

    average_waste = (
        cluster_data["Quantity_Wasted"].mean()
    )

    average_demand = (
        cluster_data["Actual_Demand"].mean()
    )

    average_students = (
        cluster_data["Number_of_Students"].mean()
    )

    print(
        f"\nCluster {cluster_number}:"
    )

    print(
        f"Average Food Waste : "
        f"{average_waste:.2f}"
    )

    print(
        f"Average Demand     : "
        f"{average_demand:.2f}"
    )

    print(
        f"Average Students   : "
        f"{average_students:.2f}"
    )


# ============================================================
# STEP 20: VISUALIZE CLUSTERS
# ============================================================

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Quantity_Prepared"],
    df["Quantity_Wasted"],
    c=df["Cluster"],
    alpha=0.6
)

plt.title(
    "SmartPlate - K-Means Clusters"
)

plt.xlabel(
    "Quantity Prepared"
)

plt.ylabel(
    "Quantity Wasted"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 21: WASTE VS DEMAND CLUSTERS
# ============================================================

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Actual_Demand"],
    df["Quantity_Wasted"],
    c=df["Cluster"],
    alpha=0.6
)

plt.title(
    "SmartPlate - Demand vs Food Waste Clusters"
)

plt.xlabel(
    "Actual Demand"
)

plt.ylabel(
    "Quantity Wasted"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 22: SAVE CLUSTERED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLUSTERED DATASET SAVED")
print("=" * 70)

print(
    " File saved successfully:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# STEP 23: DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL K-MEANS RESULT")
print("=" * 70)

print(
    "Total Records:",
    len(df)
)

print(
    "Number of Clusters:",
    K
)

print(
    f"Silhouette Score: {silhouette:.4f}"
)

print(
    "\nCluster Distribution:"
)

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print(" K-MEANS CLUSTERING COMPLETED")
print("=" * 70)