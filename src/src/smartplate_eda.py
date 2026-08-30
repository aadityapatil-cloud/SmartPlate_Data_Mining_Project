import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_dataset.csv")


print(df)


print("Shape:", df.shape)


print("Columns & Types:")
print(df.dtypes)


print("Head:")
print(df.head())


print("Missing values:")
print(df.isnull().sum())


sns.set_theme(style="whitegrid")


num_cols = df.select_dtypes(include=['float64', 'int64']).columns
stats = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']
print(stats.to_string())


df['Total_Revenue'] = df['Quantity_Sold'] * df['Selling_Price_Per_Unit']
df['Total_Cost'] = df['Quantity_Prepared'] * df['Food_Cost_Per_Unit']
df['Net_Profit'] = df['Total_Revenue'] - df['Total_Cost']


cat_summary = df.groupby('Food_Category').agg({
    'Quantity_Prepared': 'sum',
    'Quantity_Sold': 'sum',
    'Quantity_Wasted': 'sum',
    'Estimated_Cost_of_Waste': 'sum',
    'Total_Revenue': 'sum',
    'Net_Profit': 'sum'
}).reset_index()

print("\nCategory Summary:")
print(cat_summary.to_string())



fig, axes = plt.subplots(2, 2, figsize=(14, 10))


# 1. Total Prepared vs Sold vs Wasted by Food Category
cat_melted = pd.melt(cat_summary, id_vars=['Food_Category'], 
                     value_vars=['Quantity_Prepared', 'Quantity_Sold', 'Quantity_Wasted'],
                     var_name='Metric', value_name='Quantity')
sns.barplot(data=cat_melted, x='Food_Category', y='Quantity', hue='Metric', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Quantity Prepared, Sold, and Wasted by Category', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Food Category')
axes[0, 0].set_ylabel('Quantity (Units)')


# 2. Total Cost of Waste by Food Category
sns.barplot(data=cat_summary, x='Food_Category', y='Estimated_Cost_of_Waste', ax=axes[0, 1], palette='Reds_r')
axes[0, 1].set_title('Total Cost of Waste by Category ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Food Category')
axes[0, 1].set_ylabel('Estimated Waste Cost ($)')


# 3. Stockout counts by Meal Type
sns.countplot(data=df, x='Meal_Type', hue='Stockout_Flag', ax=axes[1, 0], palette='coolwarm')
axes[1, 0].set_title('Stockout Events by Meal Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Meal Type')
axes[1, 0].set_ylabel('Record Count')


# 4. Waste vs Temperature
sns.scatterplot(data=df.sample(1000, random_state=42), x='Temperature', y='Quantity_Wasted', hue='Food_Category', ax=axes[1, 1], alpha=0.7)
axes[1, 1].set_title('Quantity Wasted vs Temperature (Sampled)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Temperature (°C)')
axes[1, 1].set_ylabel('Quantity Wasted')

plt.tight_layout()
plt.savefig('eda_summary_charts.png', dpi=300)
plt.close()


# Detailed metrics summary
total_prepared = df['Quantity_Prepared'].sum()
total_sold = df['Quantity_Sold'].sum()
total_wasted = df['Quantity_Wasted'].sum()
total_waste_cost = df['Estimated_Cost_of_Waste'].sum()
stockout_rate = df['Stockout_Flag'].mean() * 100
total_revenue = df['Total_Revenue'].sum()
net_profit = df['Net_Profit'].sum()

print(f"Total Prepared Units: {total_prepared:,.2f}")
print(f"Total Sold Units: {total_sold:,.2f}")
print(f"Total Wasted Units: {total_wasted:,.2f} ({(total_wasted/total_prepared)*100:.2f}%)")
print(f"Total Waste Cost: ${total_waste_cost:,.2f}")
print(f"Stockout Rate: {stockout_rate:.2f}%")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Net Profit: ${net_profit:,.2f}")


df.to_csv('smartplate_cafeteria_cleaned.csv', index=False)
