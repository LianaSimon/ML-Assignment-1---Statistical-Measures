# ML-Assignment-1---Statistical-Measures

#### The dataset contains 13,200 entries and 7 columns. Here’s a breakdown of the columns:

* location: Area in Bangalore

* size: Format like "2 BHK", "4 Bedroom"

* total_sqft: Area in square feet

* bath: Number of bathrooms

* price: Price in lakhs

* bhk: Number of bedrooms

* price_per_sqft: Price per square foot



## Q1: Perform Basic EDA (Exploratory Data Analysis) on the house_price.csv dataset.

#### 🔹 Step 1: Load the Data


import pandas as pd

df = pd.read_csv("house_price.csv")

df.head()

![image](https://github.com/user-attachments/assets/81efabe8-9464-4eba-a14f-a6240ad9427a)



#### 🔹 Step 2: Basic Info and Data Types

df.info()

This helps understand:

The number of entries (rows)

Column names and data types

Any missing (null) values


![image](https://github.com/user-attachments/assets/7da8c975-13eb-4542-bc36-97a083bb1ba7)



#### 🔹 Step 3: Statistical Summary

df.describe()


Use this to get:

Mean, median, std deviation

Min and max values

Quantiles (25%, 50%, 75%)


![image](https://github.com/user-attachments/assets/7b52453a-f360-47c3-9d6a-f08456b4766f)


#### 🔹 Step 4: Check for Missing Values

df.isnull().sum()

Identifies any columns with missing values that might need imputation or cleaning.


![image](https://github.com/user-attachments/assets/c4f4c0ba-0647-4e45-b930-d26961ecac06)



#### 🔹 Step 5: Check Unique Values in Categorical Columns

df.select_dtypes(include='object').nunique()


Helps us understand:

How many unique values are in columns like location or size
If any categories have high cardinality


![image](https://github.com/user-attachments/assets/7e21883f-8db1-44b0-8d95-33e7b1b0342a)



#### 🔹 Step 6: Visualize Numerical Distributions

These histograms + KDE (density curves) give insight into the shape of the data (normal/skewed) and presence of outliers.


import matplotlib.pyplot as plt

import seaborn as sns


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns


plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols):

    plt.subplot(2, 3, i+1)
    
    sns.histplot(df[col], kde=True)
    
    plt.title(f'Distribution of {col}')
    
plt.tight_layout()

plt.show()



![image](https://github.com/user-attachments/assets/5afa7191-33f5-4e93-bf54-8dee6440f20d)

![image](https://github.com/user-attachments/assets/05e835ee-8ee8-422c-a758-dfe09774908f)





📌 Purpose:
This block helps you visually examine the distribution of each numeric column — e.g., to spot outliers, skewness, or whether the data follows a normal distribution.




## ✅ Q2: Outlier Detection & Removal 

We'll work with the column price_per_sqft.


#### 🔹 Step 1: Visualize Original Data (Boxplot)

sns.boxplot(x=df['price_per_sqft']) 

plt.title('Original Boxplot of price_per_sqft')

plt.show()


This gives you a visual cue of how the outliers are distributed.

![image](https://github.com/user-attachments/assets/7893fc09-6e4b-4f49-bf56-840da0eeebd0)



### 📘 Methods of Outlier Detection and Handling

#### 🔹 Step 2a: Using Mean ± 3*Standard Deviation

mean = df['price_per_sqft'].mean()

std = df['price_per_sqft'].std()


lower_limit = mean - 3 * std

upper_limit = mean + 3 * std


df_mean_sd = df[(df['price_per_sqft'] >= lower_limit) & (df['price_per_sqft'] <= upper_limit)]



Explanation: Values beyond ±3σ are considered outliers and removed.


![image](https://github.com/user-attachments/assets/324c32b6-f302-4c33-94d0-919f0788ee0e)



#### 🔹 Step 2b: Using Percentile Method (1st and 99th)

low_percentile=df['price_per_sqft'].quantile(0.01)

up_percentile=df['price_per_sqft'].quantile(0.99)

df_percentile=df[(df['price_per_sqft']<=up_percentile) & (df['price_per_sqft']>=low_percentile)]

df_percentile


Explanation: Trims extreme 1% from both sides.


![image](https://github.com/user-attachments/assets/7297eac5-7abf-4885-bcd5-bab4c948f514)


#### 🔹 Step 2c: IQR (Interquartile Range) Method

Q1 = df['price_per_sqft'].quantile(0.25)

Q3 = df['price_per_sqft'].quantile(0.75)

IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


df_iqr = df[(df['price_per_sqft'] >= lower_bound) & (df['price_per_sqft'] <= upper_bound)]

df_iqr

Explanation: IQR focuses on the middle 50% of data; values outside 1.5×IQR are outliers.

![image](https://github.com/user-attachments/assets/c5a3b1f1-2e3b-4adc-a9f6-b13e56d693f2)



#### 🔹 Step 2d: Z-Score Method

from scipy.stats import zscore

z_scores = zscore(df['price_per_sqft'])

df_zscore = df[(abs(z_scores) < 3)]

df_zscore

Explanation: Removes data points where the z-score is beyond ±3.

![image](https://github.com/user-attachments/assets/8af41edd-0e83-4649-8d8e-d6314f6288d7)



## ✅ Q3: Box Plot Comparison for Outlier Removal Methods


dfs = {

    'Original': df,
    
    'Mean±3SD': df_mean_sd,
    
    'Percentile': df_percentile,
    
    'IQR': df_iqr,
    
    'Z-Score': df_zscore
    
}



plt.figure(figsize=(15, 6))

for i, (name, d) in enumerate(dfs.items()):

    plt.subplot(1, 5, i+1)
    
    sns.boxplot(y=d['price_per_sqft'])
    
    plt.title(name)
    
    plt.tight_layout()
    
plt.show()


#### 🧐 Interpretation:

Best method: The one with minimal visible outliers but still preserving spread (usually IQR or Percentile for skewed data).

![image](https://github.com/user-attachments/assets/9cba0390-a8e7-4744-b760-74d4ec335a19)




## ✅ Q4: Check Normality of price_per_sqft and Apply Transformation if Needed

#### 🔹 Step 1: Histogram with KDE

sns.histplot(df['price_per_sqft'], kde=True)

plt.title('Original price_per_sqft Distribution')

plt.show()


![image](https://github.com/user-attachments/assets/0a07171c-4104-41bb-870d-b51ba05b9a53)



#### 🔹 Step 2: Check Skewness & Kurtosis

print("Skewness:", df['price_per_sqft'].skew())

print("Kurtosis:", df['price_per_sqft'].kurt())

![image](https://github.com/user-attachments/assets/357fc813-b78a-428c-8c41-462a34c72f75)


#### 🔹 Step 3: Apply Log Transformation (if right-skewed)


import numpy as np

df['log_price_per_sqft'] = np.log1p(df['price_per_sqft'])


sns.histplot(df['log_price_per_sqft'], kde=True)

plt.title('Log-Transformed price_per_sqft')

plt.show()


print("Skewness (log):", df['log_price_per_sqft'].skew())

print("Kurtosis (log):", df['log_price_per_sqft'].kurt())



![image](https://github.com/user-attachments/assets/f7b5499e-5b01-4d13-acb8-c068ab2b104f)

📌 Goal: See if log transformation reduces skewness and makes the distribution closer to normal.



## ✅ Q5: Correlation Heatmap of Numerical Columns

#### 🔹 Step 1: Compute Correlation Matrix

correlation_matrix = df.corr(numeric_only=True)

correlation_matrix

![image](https://github.com/user-attachments/assets/118f14ce-61f8-4261-b039-436e9c26c6e0)


#### 🔹 Step 2: Plot Heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Heatmap")

plt.show()

![image](https://github.com/user-attachments/assets/3d47b41d-1749-4654-9427-b43004567bf9)


🔍 Interpretation:

Values close to 1 or -1 indicate strong correlation.

Look at price_per_sqft and see how it's correlated with total_sqft, price, etc.



## ✅ Q6: Scatter Plots Between Variables

#### 🔹 Step 1: Select Relevant Numerical Columns

We’ll focus on columns that are likely to have relationships with each other:

columns_to_plot = ['total_sqft', 'bhk', 'bath', 'price', 'price_per_sqft']


#### 🔹 Step 2: Create Pairwise Scatter Plots

Use seaborn.pairplot() to generate scatter plots for all combinations of selected variables.


import seaborn as sns

import matplotlib.pyplot as plt


sns.pairplot(df[columns_to_plot])

plt.suptitle('Scatter Plot Matrix', y=1.02)

plt.show()

![image](https://github.com/user-attachments/assets/e96802ea-b345-4f2c-8cdf-a59d6dfcd91b)

![image](https://github.com/user-attachments/assets/9b0da556-178a-4381-96db-1faac8ca4f18)

![image](https://github.com/user-attachments/assets/8497ecc1-188e-4fa6-8ae8-e06e0b14b8c5)



This will plot a grid of scatter plots and histograms to help identify trends and correlations.




## ✅ Conclusion
In this analysis of Bangalore house prices, we explored the price_per_sqft feature in depth to understand its distribution, detect outliers, and assess its relationships with other numerical features. Here's a summary of our findings:

#### 🔍 Exploratory Data Analysis (EDA)
We performed basic EDA to understand the dataset's structure, data types, and statistical summaries. Initial histograms and boxplots revealed that price_per_sqft is highly skewed and contains significant outliers.

⚠ Outlier Detection and Removal
Outliers were detected using four methods:

Mean ± 3*Standard Deviation

Percentile Method (1st & 99th percentiles)

Interquartile Range (IQR)

Z-Score Method

Among these, the IQR and Percentile methods were most effective in removing extreme values while retaining the overall distribution, as confirmed visually using boxplots.

📊 Normality Check & Transformation
We assessed the normality of price_per_sqft using histogram, skewness, and kurtosis metrics. The original distribution was right-skewed. Applying a log transformation significantly reduced skewness and brought the distribution closer to normal.

📈 Correlation Analysis
A correlation heatmap showed strong relationships between key variables like price, total_sqft, and bhk. price_per_sqft showed moderate correlation with these features.

🔗 Scatter Plot Insights
Scatter plots and pair plots revealed:

A positive correlation between total_sqft and price

Price per square foot varies widely across BHKs and sizes

Some patterns suggest possible pricing inconsistencies or anomalies, which could be due to locality or data entry issues

#### 📌 Final Takeaway
Cleaning and understanding price_per_sqft is crucial for reliable real estate pricing models. Handling outliers using appropriate statistical techniques (especially IQR or Percentile) and transforming skewed data improves data quality and supports better decision-making for predictive modeling.




