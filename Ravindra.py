import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # For z-test

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load dataset
data = pd.read_csv("C:/Users/hp/Downloads/Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")

# Clean column names by stripping whitespace
data.columns = data.columns.str.strip()

# Basic Information
print("=== BASIC INFORMATION ===")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Data Cleaning
print("\n=== MISSING VALUES ===")
print(data.isnull().sum())

# Filter for United States data only
us_data = data[data['State'] == 'United States']

# Convert numeric columns that might be read as strings
numeric_cols = ['COVID-19 Deaths', 'Total Deaths', 'Pneumonia Deaths', 
                'Pneumonia and COVID-19 Deaths', 'Influenza Deaths',
                'Pneumonia, Influenza, or COVID-19 Deaths']


# ========== Z-TEST IMPLEMENTATIONS ==========

def perform_z_test(group1, group2, group1_name, group2_name):
    """Perform z-test between two groups and print results"""
    # Calculate proportions
    p1 = group1['COVID-19 Deaths'].sum() / group1['Total Deaths'].sum()
    p2 = group2['COVID-19 Deaths'].sum() / group2['Total Deaths'].sum()
    
    # Calculate pooled proportion
    n1 = group1['Total Deaths'].sum()
    n2 = group2['Total Deaths'].sum()
    p_pool = (group1['COVID-19 Deaths'].sum() + group2['COVID-19 Deaths'].sum()) / (n1 + n2)
    
    # Calculate standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Calculate z-score
    z = (p1 - p2) / se
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed test
    
    print(f"\nZ-Test Results: {group1_name} vs {group2_name}")
    print(f"Proportion {group1_name}: {p1:.4f} ({group1['COVID-19 Deaths'].sum():,} COVID deaths / {n1:,} total deaths)")
    print(f"Proportion {group2_name}: {p2:.4f} ({group2['COVID-19 Deaths'].sum():,} COVID deaths / {n2:,} total deaths)")
    print(f"Z-score: {z:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Significant difference (p < 0.05)")
    else:
        print("Conclusion: No significant difference (p ≥ 0.05)")

# Z-Test 1: Male vs Female COVID-19 death rates
male_data = us_data[us_data['Sex'] == 'Male']
female_data = us_data[us_data['Sex'] == 'Female']
perform_z_test(male_data, female_data, "Males", "Females")

# Z-Test 2: Older adults (65+) vs younger adults (18-64)
older_adults = us_data[us_data['Age Group'].isin(['65-74 years', '75-84 years', '85 years and over'])]
younger_adults = us_data[us_data['Age Group'].isin(['18-29 years', '30-39 years', '40-49 years', '50-64 years'])]
perform_z_test(older_adults, younger_adults, "Adults 65+", "Adults 18-64")

# ========== VISUALIZATIONS ==========

# Top 10 Age Groups by COVID-19 Deaths
plt.figure(figsize=(14, 6))
top_age_groups = us_data.groupby('Age Group')['COVID-19 Deaths'].sum().sort_values(ascending=False).head(10)
sns.barplot(
    x=top_age_groups.index,
    y=top_age_groups.values,
    hue=top_age_groups.index,
    palette='Set3',
    dodge=False,
    legend=False
)
plt.title("Top 10 Age Groups by COVID-19 Deaths (US)")
plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COVID-19 Deaths by Sex
plt.figure(figsize=(10, 6))
sex_deaths = us_data.groupby('Sex')['COVID-19 Deaths'].sum().dropna()
sns.barplot(
    x=sex_deaths.index,
    y=sex_deaths.values,
    hue=sex_deaths.index,
    palette='coolwarm',
    legend=False
)
plt.title("COVID-19 Deaths by Sex (US)")
plt.xlabel("Sex")
plt.ylabel("Number of COVID-19 Deaths")
plt.tight_layout()
plt.show()

# Age Distribution of COVID-19 Deaths
plt.figure(figsize=(14, 6))
age_deaths = us_data[us_data['Age Group'] != 'All Ages'].groupby('Age Group')['COVID-19 Deaths'].sum().dropna()
sns.barplot(
    x=age_deaths.index,
    y=age_deaths.values,
    hue=age_deaths.index,
    palette='viridis',
    legend=False
)
plt.title("COVID-19 Deaths by Age Group (US)")
plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COVID-19 Deaths vs Pneumonia Deaths
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Pneumonia Deaths',
    y='COVID-19 Deaths',
    data=us_data,
    hue='Sex',
    alpha=0.7
)
plt.title("Pneumonia Deaths vs COVID-19 Deaths (US)")
plt.xlabel("Pneumonia Deaths")
plt.ylabel("COVID-19 Deaths")
plt.tight_layout()
plt.show()

# Correlation Matrix
numerical_columns = [
    'COVID-19 Deaths', 'Total Deaths', 'Pneumonia Deaths',
    'Pneumonia and COVID-19 Deaths', 'Influenza Deaths',
    'Pneumonia, Influenza, or COVID-19 Deaths'
]

corr_matrix = us_data[numerical_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation Matrix of Death Metrics (US)")
plt.tight_layout()
plt.show()

# State Comparison (Top 10 states by COVID-19 deaths)
state_data = data[data['State'] != 'United States']
state_deaths = state_data.groupby('State')['COVID-19 Deaths'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 6))
sns.barplot(
    x=state_deaths.index,
    y=state_deaths.values,
    hue=state_deaths.index,
    palette='Set2',
    legend=False
)
plt.title("Top 10 States by COVID-19 Deaths")
plt.xlabel("State")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Male vs Female Deaths by Age Group
sex_age_data = us_data[(us_data['Sex'] != 'All Sexes') & (us_data['Age Group'] != 'All Ages')]
plt.figure(figsize=(14, 8))
sns.barplot(
    x='Age Group',
    y='COVID-19 Deaths',
    hue='Sex',
    data=sex_age_data,
    palette='coolwarm'
)
plt.title("COVID-19 Deaths by Age Group and Sex (US)")
plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot of COVID-19 Deaths by Age Group
plt.figure(figsize=(14, 8))
filtered_data = us_data[(us_data['Age Group'] != 'All Ages') & (us_data['Sex'] == 'All Sexes')]
sns.boxplot(
    x='Age Group',
    y='COVID-19 Deaths',
    data=filtered_data,
)
plt.title("Distribution of COVID-19 Deaths by Age Group (US)")
plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Lineplot of COVID-19 Deaths by Age Group (Trend Analysis)
plt.figure(figsize=(14, 8))
age_order = ['Under 1 year', '1-4 years', '5-14 years', '15-24 years', '25-34 years', 
             '35-44 years', '45-54 years', '55-64 years', '65-74 years', 
             '75-84 years', '85 years and over']
filtered_data = us_data[us_data['Age Group'].isin(age_order)]
sns.lineplot(
    x='Age Group',
    y='COVID-19 Deaths',
    hue='Sex',
    data=filtered_data[filtered_data['Sex'] != 'All Sexes'],
    marker='o',
    markersize=8,
    palette='dark'
)
plt.title("Trend of COVID-19 Deaths by Age Group and Sex (US)")
plt.xlabel("Age Group")
plt.ylabel("Number of COVID-19 Deaths")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Pair plot (using first 30 rows as in your original code)
hemanth = data.head(30)
sns.pairplot(data=hemanth)
plt.title("Pair Plot of Given Data")
plt.tight_layout()
plt.show()
print("\n=== COVID-19 DEATHS ANALYSIS COMPLETE ===")
