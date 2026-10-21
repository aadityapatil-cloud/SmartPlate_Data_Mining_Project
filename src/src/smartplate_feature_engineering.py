# ============================================================
# PHASE 4: FEATURE ENGINEERING
# SmartPlate Food Waste Reduction Project
# ============================================================

print("=" * 70)
print("PHASE 4: FEATURE ENGINEERING")
print("=" * 70)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_dataset.csv")

print("Original Dataset Shape:", df.shape)

print("\nCurrent Columns:")
print(df.columns.tolist())


# Required columns for feature engineering

required_columns = [
    'Quantity_Prepared',
    'Quantity_Sold',
    'Quantity_Wasted',
    'Selling_Price_Per_Unit',
    'Food_Cost_Per_Unit',
    'Estimated_Cost_of_Waste',
    'Food_Category',
    'Meal_Type',
    'Stockout_Flag',
    'Temperature'
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) == 0:
    print(" All required columns are available.")
else:
    print(" Missing columns:")
    print(missing_columns)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

numeric_columns = [
    'Quantity_Prepared',
    'Quantity_Sold',
    'Quantity_Wasted',
    'Selling_Price_Per_Unit',
    'Food_Cost_Per_Unit',
    'Estimated_Cost_of_Waste',
    'Stockout_Flag',
    'Temperature'
]

print("Negative values check:")

for col in numeric_columns:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        print(f"{col}: {negative_count}")


# ============================================================
# FEATURE 1: WASTE PERCENTAGE
# ============================================================

df['Waste_Percentage'] = np.where(
    df['Quantity_Prepared'] > 0,
    (df['Quantity_Wasted'] / df['Quantity_Prepared']) * 100,
    0
)

print("Waste Percentage created successfully.")

print(
    df[
        [
            'Quantity_Prepared',
            'Quantity_Wasted',
            'Waste_Percentage'
        ]
    ].head()
)


# ============================================================
# FEATURE 2: FOOD UTILIZATION RATE
# ============================================================

df['Food_Utilization_Rate'] = np.where(
    df['Quantity_Prepared'] > 0,
    (df['Quantity_Sold'] / df['Quantity_Prepared']) * 100,
    0
)

print(
    df[
        [
            'Quantity_Prepared',
            'Quantity_Sold',
            'Food_Utilization_Rate'
        ]
    ].head()
)


# ============================================================
# FEATURE 3: PREPARATION BUFFER
# ============================================================

df['Preparation_Buffer'] = (
    df['Quantity_Prepared'] -
    df['Quantity_Sold']
)

print(
    df[
        [
            'Quantity_Prepared',
            'Quantity_Sold',
            'Preparation_Buffer'
        ]
    ].head()
)


# ============================================================
# FEATURE 4: WASTE COST PER UNIT
# ============================================================

df['Waste_Cost_Per_Unit'] = np.where(
    df['Quantity_Wasted'] > 0,
    df['Estimated_Cost_of_Waste'] /
    df['Quantity_Wasted'],
    0
)

print(
    df[
        [
            'Quantity_Wasted',
            'Estimated_Cost_of_Waste',
            'Waste_Cost_Per_Unit'
        ]
    ].head()
)


# ============================================================
# FEATURE 5: PROFIT PER SOLD UNIT
# ============================================================

# Create Total Revenue
df['Total_Revenue'] = (
    df['Quantity_Sold'] *
    df['Selling_Price_Per_Unit']
)

# Create Total Cost
df['Total_Cost'] = (
    df['Quantity_Sold'] *
    df['Food_Cost_Per_Unit']
)

# Create Net Profit
df['Net_Profit'] = (
    df['Total_Revenue'] -
    df['Total_Cost']
)

# Create Profit Per Sold Unit
df['Profit_Per_Sold_Unit'] = np.where(
    df['Quantity_Sold'] > 0,
    df['Net_Profit'] / df['Quantity_Sold'],
    0
)

print(" Net_Profit created")
print(" Profit_Per_Sold_Unit created")

print(
    df[
        [
            'Quantity_Sold',
            'Selling_Price_Per_Unit',
            'Food_Cost_Per_Unit',
            'Total_Revenue',
            'Total_Cost',
            'Net_Profit',
            'Profit_Per_Sold_Unit'
        ]
    ].head()
)


# ============================================================
# FEATURE 6: FOOD COST PERCENTAGE
# ============================================================

df['Food_Cost_Percentage'] = np.where(
    df['Selling_Price_Per_Unit'] > 0,
    (
        df['Food_Cost_Per_Unit'] /
        df['Selling_Price_Per_Unit']
    ) * 100,
    0
)

print(
    df[
        [
            'Selling_Price_Per_Unit',
            'Food_Cost_Per_Unit',
            'Food_Cost_Percentage'
        ]
    ].head()
)


# ============================================================
# FEATURE 7: HIGH WASTE FLAG
# ============================================================

waste_threshold = df['Waste_Percentage'].median()

print("Median Waste Percentage:",
      round(waste_threshold, 2))

df['High_Waste_Flag'] = (
    df['Waste_Percentage'] > waste_threshold
).astype(int)

print("\nHigh Waste Flag:")
print(df['High_Waste_Flag'].value_counts())

print(
    df[
        [
            'Waste_Percentage',
            'High_Waste_Flag'
        ]
    ].head()
)


# ============================================================
# FEATURE 8: STOCKOUT STATUS
# ============================================================

df['Stockout_Status'] = np.where(
    df['Stockout_Flag'] == 1,
    'Stockout',
    'No Stockout'
)

print(df['Stockout_Status'].value_counts())


# ============================================================
# FEATURE 9: TEMPERATURE CATEGORY
# ============================================================

df['Temperature_Category'] = pd.cut(
    df['Temperature'],
    bins=[-np.inf, 20, 25, 30, np.inf],
    labels=[
        'Cool',
        'Moderate',
        'Warm',
        'Hot'
    ]
)

print(df['Temperature_Category'].value_counts())


# ============================================================
# FEATURE 10: FOOD POPULARITY GROUP
# ============================================================

food_sales_avg = (
    df.groupby('Food_Item')['Quantity_Sold']
      .transform('mean')
)

df['Food_Popularity_Group'] = pd.qcut(
    food_sales_avg.rank(method='first'),
    q=3,
    labels=[
        'Low',
        'Medium',
        'High'
    ]
)

print(
    df[
        [
            'Food_Item',
            'Quantity_Sold',
            'Food_Popularity_Group'
        ]
    ].head(10)
)


# ============================================================
# FEATURE 11: WASTE SEVERITY
# ============================================================

df['Waste_Severity'] = pd.cut(
    df['Waste_Percentage'],
    bins=[
        -np.inf,
        10,
        25,
        50,
        np.inf
    ],
    labels=[
        'Low',
        'Moderate',
        'High',
        'Critical'
    ]
)

print(df['Waste_Severity'].value_counts())


# ============================================================
# FEATURE 12: REVENUE PER PREPARED UNIT
# ============================================================

df['Revenue_Per_Prepared_Unit'] = np.where(
    df['Quantity_Prepared'] > 0,
    df['Total_Revenue'] /
    df['Quantity_Prepared'],
    0
)

print(
    df[
        [
            'Total_Revenue',
            'Quantity_Prepared',
            'Revenue_Per_Prepared_Unit'
        ]
    ].head()
)


# ============================================================
# FEATURE 13: WASTE COST RATIO
# ============================================================

df['Waste_Cost_Ratio'] = np.where(
    df['Total_Cost'] > 0,
    (
        df['Estimated_Cost_of_Waste'] /
        df['Total_Cost']
    ) * 100,
    0
)

print(
    df[
        [
            'Estimated_Cost_of_Waste',
            'Total_Cost',
            'Waste_Cost_Ratio'
        ]
    ].head()
)


# ============================================================
# FEATURE 14: CATEGORY AVERAGE WASTE
# ============================================================

category_waste_avg = (
    df.groupby('Food_Category')['Waste_Percentage']
      .transform('mean')
)

df['Category_Avg_Waste_Percentage'] = category_waste_avg

print(
    df[
        [
            'Food_Category',
            'Waste_Percentage',
            'Category_Avg_Waste_Percentage'
        ]
    ].head()
)


# ============================================================
# FEATURE 15: MEAL AVERAGE WASTE
# ============================================================

meal_waste_avg = (
    df.groupby('Meal_Type')['Waste_Percentage']
      .transform('mean')
)

df['Meal_Avg_Waste_Percentage'] = meal_waste_avg

print(
    df[
        [
            'Meal_Type',
            'Waste_Percentage',
            'Meal_Avg_Waste_Percentage'
        ]
    ].head()
)


# ============================================================
# OPTIONAL DATE FEATURES
# ============================================================

if 'Date' in df.columns:

    df['Date'] = pd.to_datetime(
        df['Date'],
        errors='coerce'
    )

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month_name()

    df['Weekday_Number'] = df['Date'].dt.weekday
    df['Weekday_Name'] = df['Date'].dt.day_name()

    df['Is_Weekend'] = (
        df['Weekday_Number'] >= 5
    ).astype(int)

    print(" Date features created.")

else:
    print(" Date column is not available.")
    print("Date-based features were skipped.")


# ============================================================
# OPTIONAL DEMAND FEATURES
# ============================================================

if 'Actual_Demand' in df.columns:

    if 'Number_of_Students' in df.columns:

        df['Demand_Per_Student'] = np.where(
            df['Number_of_Students'] > 0,
            df['Actual_Demand'] /
            df['Number_of_Students'],
            0
        )

        print(" Demand_Per_Student created.")

    else:
        print(
            " Number_of_Students column "
            "is not available."
        )

else:
    print(
        " Actual_Demand column is not available."
    )


# ============================================================
# OPTIONAL: DEMAND-BASED PREPARATION BUFFER
# ============================================================

if 'Actual_Demand' in df.columns:

    df['Demand_Based_Preparation_Buffer'] = (
        df['Quantity_Prepared'] -
        df['Actual_Demand']
    )

    print(
        df[
            [
                'Quantity_Prepared',
                'Actual_Demand',
                'Demand_Based_Preparation_Buffer'
            ]
        ].head()
    )


# ============================================================
# FEATURE ENGINEERING SUMMARY
# ============================================================

new_features = [
    'Waste_Percentage',
    'Food_Utilization_Rate',
    'Preparation_Buffer',
    'Waste_Cost_Per_Unit',
    'Profit_Per_Sold_Unit',
    'Food_Cost_Percentage',
    'High_Waste_Flag',
    'Stockout_Status',
    'Temperature_Category',
    'Food_Popularity_Group',
    'Waste_Severity',
    'Revenue_Per_Prepared_Unit',
    'Waste_Cost_Ratio',
    'Category_Avg_Waste_Percentage',
    'Meal_Avg_Waste_Percentage'
]

# Add optional features if they exist
optional_features = [
    'Year',
    'Month',
    'Month_Name',
    'Weekday_Number',
    'Weekday_Name',
    'Is_Weekend',
    'Demand_Per_Student',
    'Demand_Based_Preparation_Buffer'
]

all_features = new_features + optional_features

available_features = [
    feature for feature in all_features
    if feature in df.columns
]

print("=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print("\nNew Features Created:")

for i, feature in enumerate(
    available_features,
    start=1
):
    print(f"{i}. {feature}")

print(
    "\nNumber of new features:",
    len(available_features)
)

print(
    "\nFinal Dataset Shape:",
    df.shape
)


# ============================================================
# VIEW FEATURE ENGINEERED DATASET
# ============================================================

print(df.head(10))


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print("Missing Values After Feature Engineering:")
print(df.isnull().sum())


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_count
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

feature_numeric = [
    'Waste_Percentage',
    'Food_Utilization_Rate',
    'Preparation_Buffer',
    'Waste_Cost_Per_Unit',
    'Profit_Per_Sold_Unit',
    'Food_Cost_Percentage',
    'Revenue_Per_Prepared_Unit',
    'Waste_Cost_Ratio',
    'Category_Avg_Waste_Percentage',
    'Meal_Avg_Waste_Percentage'
]

available_numeric_features = [
    col for col in feature_numeric
    if col in df.columns
]

print(
    df[available_numeric_features]
    .describe()
    .T
)


# ============================================================
# FEATURE CORRELATION
# ============================================================

correlation_features = [
    'Quantity_Prepared',
    'Quantity_Sold',
    'Quantity_Wasted',
    'Estimated_Cost_of_Waste',
    'Temperature',
    'Total_Revenue',
    'Total_Cost',
    'Net_Profit',
    'Waste_Percentage',
    'Food_Utilization_Rate',
    'Preparation_Buffer',
    'Waste_Cost_Per_Unit',
    'Profit_Per_Sold_Unit',
    'Food_Cost_Percentage',
    'Revenue_Per_Prepared_Unit',
    'Waste_Cost_Ratio'
]

available_corr_features = [
    col for col in correlation_features
    if col in df.columns
]

correlation_matrix = (
    df[available_corr_features]
    .corr()
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0
)

plt.title(
    'SmartPlate Feature Correlation Matrix'
)

plt.tight_layout()

plt.show()


# ============================================================
# SAVE FEATURE ENGINEERED DATASET
# ============================================================

output_file = 'smartplate_features.csv'

df.to_csv(
    output_file,
    index=False
)

print("=" * 70)
print(" FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    f"File saved successfully: {output_file}"
)

print(
    f"Final dataset shape: {df.shape}"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("=" * 70)
print("FINAL SMARTPLATE DATASET")
print("=" * 70)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\n Feature Engineering Phase Completed.")
