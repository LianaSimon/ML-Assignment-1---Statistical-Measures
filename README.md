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
