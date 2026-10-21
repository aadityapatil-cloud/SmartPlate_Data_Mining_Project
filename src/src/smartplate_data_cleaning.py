# ============================================================
# SMARTPLATE - COMPLETE DATA CLEANING
# ============================================================

# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# STEP 2: DEFINE FILE PATHS
# ============================================================

# Original/raw SmartPlate dataset
INPUT_FILE = r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_dataset.csv"

# Cleaned dataset output file
OUTPUT_FILE = r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_cleaned.csv"


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

print("=" * 70)
print("SMARTPLATE - DATA CLEANING")
print("=" * 70)

try:

    # Read the CSV file
    df = pd.read_csv(INPUT_FILE)

    print("\n Dataset loaded successfully.")
    print("Original Dataset Shape:", df.shape)

except FileNotFoundError:

    print("\n ERROR: Dataset file not found.")
    print("Please check this path:")
    print(INPUT_FILE)

    raise SystemExit


# ============================================================
# STEP 4: DISPLAY ORIGINAL DATA
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL DATASET - FIRST 5 ROWS")
print("=" * 70)

print(df.head())


# ============================================================
# STEP 5: CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# STEP 6: CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print("\nTotal Missing Values:", total_missing)


# ============================================================
# STEP 7: IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

# Numeric columns contain numbers
numeric_columns = df.select_dtypes(
    include=np.number
).columns

# Categorical columns contain text
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\n" + "=" * 70)
print("NUMERIC COLUMNS")
print("=" * 70)

print(list(numeric_columns))

print("\n" + "=" * 70)
print("CATEGORICAL COLUMNS")
print("=" * 70)

print(list(categorical_columns))


# ============================================================
# STEP 8: HANDLE MISSING NUMERIC VALUES
# ============================================================

# Replace missing numeric values with the median
# Median is useful because it is less affected by outliers

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

print("\n Numeric missing values handled.")


# ============================================================
# STEP 9: HANDLE MISSING CATEGORICAL VALUES
# ============================================================

# Replace missing text values with the most common value
# (mode)

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if not mode_value.empty:

            df[column] = df[column].fillna(
                mode_value[0]
            )

print(" Categorical missing values handled.")


# ============================================================
# STEP 10: REMOVE DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ROW CHECK")
print("=" * 70)

# Count duplicate rows
duplicate_count = df.duplicated().sum()

print(
    "Duplicate Rows Before Cleaning:",
    duplicate_count
)

# Remove duplicate rows
df = df.drop_duplicates()

print(
    "Duplicate Rows After Cleaning:",
    df.duplicated().sum()
)


# ============================================================
# STEP 11: REMOVE EXTRA SPACES
# ============================================================

# Remove spaces from the beginning and end
# of text values

for column in categorical_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

print("\n Extra spaces removed.")


# ============================================================
# STEP 12: STANDARDIZE TEXT
# ============================================================

# Convert text values into Title Case
#
# Example:
# "rice meal" -> "Rice Meal"
# "RICE MEAL" -> "Rice Meal"

for column in categorical_columns:

    df[column] = df[column].str.title()

print(" Text values standardized.")


# ============================================================
# STEP 13: CHECK NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE VALUE CHECK")
print("=" * 70)

total_negative = 0

for column in numeric_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    print(
        f"{column}: "
        f"{negative_count} negative values"
    )

    total_negative += negative_count


# ============================================================
# STEP 14: HANDLE NEGATIVE VALUES
# ============================================================

# Negative values are treated as invalid
# Replace negative values with NaN

for column in numeric_columns:

    df.loc[
        df[column] < 0,
        column
    ] = np.nan


# Fill the newly created missing values
# with the column median

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )

print("\n Negative values handled.")


# ============================================================
# STEP 15: CONVERT DATE COLUMN
# ============================================================

# Convert Date column into datetime format
# if the column exists

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    print(" Date column converted to datetime.")

else:

    print(" Date column not found.")


# ============================================================
# STEP 16: CHECK INVALID DATE VALUES
# ============================================================

if "Date" in df.columns:

    invalid_dates = df["Date"].isnull().sum()

    print("\nInvalid Date Values:", invalid_dates)


# ============================================================
# STEP 17: FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)

final_missing = df.isnull().sum()

print(final_missing)

total_final_missing = final_missing.sum()

print(
    "\nTotal Missing Values After Cleaning:",
    total_final_missing
)


# ============================================================
# STEP 18: FINAL DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK AFTER CLEANING")
print("=" * 70)

final_duplicates = df.duplicated().sum()

print(
    "Duplicate Rows After Cleaning:",
    final_duplicates
)


# ============================================================
# STEP 19: FINAL DATASET SHAPE
# ============================================================

print("\n" + "=" * 70)
print("CLEANED DATASET SHAPE")
print("=" * 70)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# STEP 20: DISPLAY CLEANED DATA
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEANED RECORDS")
print("=" * 70)

print(df.head(10))


# ============================================================
# STEP 21: CLEANED DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("CLEANED DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# STEP 22: SAVE CLEANED DATASET
# ============================================================

# Save the cleaned DataFrame as a new CSV file
# index=False prevents pandas from creating an extra index column

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANED DATASET SAVED")
print("=" * 70)

print(" Cleaned dataset saved successfully.")
print("Output File:")
print(OUTPUT_FILE)


# ============================================================
# STEP 23: FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA CLEANING VALIDATION")
print("=" * 70)

print(
    "Missing Values :",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows :",
    df.duplicated().sum()
)


# ============================================================
# STEP 24: FINAL RESULT
# ============================================================

if (
    df.isnull().sum().sum() == 0
    and df.duplicated().sum() == 0
):

    print("\n" + "=" * 70)
    print(" DATA CLEANING PASSED")
    print("=" * 70)

    print(
        "The SmartPlate dataset is clean "
        "and ready for EDA."
    )

else:

    print("\n" + "=" * 70)
    print("⚠️ DATA CLEANING NEEDS ATTENTION")
    print("=" * 70)

    print(
        "Some data-quality issues are still present."
    )