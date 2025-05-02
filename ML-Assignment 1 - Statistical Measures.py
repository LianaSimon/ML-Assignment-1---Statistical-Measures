#!/usr/bin/env python
# coding: utf-8

# ## Q1: Perform Basic EDA (Exploratory Data Analysis) on the house_price.csv dataset.

# #### Step 1: Load the Data

# In[1]:


import pandas as pd


# In[3]:


# Load CSV File
df=pd.read_csv("house_price.csv")
df.head()


# #### Step 2: Basic Info and Data Types

# In[5]:


df.info()


# #### Step 3: Statistical Summary

# In[7]:


df.describe()


# #### Step 4: Check for Missing Values

# In[9]:


df.isnull().sum()


# ####  Step 5: Check Unique Values in Categorical Columns

# In[11]:


df.select_dtypes(include='object').nunique()


# #### Step 6: Visualize Numerical Distributions

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# ## Q2: Outlier Detection & Removal 
# #### Step 1: Visualize Original Data (Boxplot)

# In[18]:


sns.boxplot(x=df['price_per_sqft'])
plt.title("Original Boxplot of price per sqft")
plt.show()


# ### Methods of Outlier Detection and Handling

# #### Step 2a: Using Mean ± 3*Standard Deviation

# In[20]:


mean=df['price_per_sqft'].mean()
std=df['price_per_sqft'].std()

lower_limit=mean-3*std
upper_limit=mean+3*std
df_mean_sd=df[(df['price_per_sqft']>=lower_limit) & (df['price_per_sqft']<=upper_limit)]
df_mean_sd


# #### Step 2a: Using Mean ± 3*Standard Deviation

# In[22]:


low_percentile=df['price_per_sqft'].quantile(0.01)
up_percentile=df['price_per_sqft'].quantile(0.99)
df_percentile=df[(df['price_per_sqft']<=up_percentile) & (df['price_per_sqft']>=low_percentile)]
df_percentile


# #### Step 2c: IQR (Interquartile Range) Method

# In[24]:


Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_iqr = df[(df['price_per_sqft'] >= lower_bound) & (df['price_per_sqft'] <= upper_bound)]
df_iqr


# #### Step 2d: Z-Score Method

# In[26]:


from scipy.stats import zscore

z_scores = zscore(df['price_per_sqft'])
df_zscore = df[(abs(z_scores) < 3)]

df_zscore


# ## Q3: Box Plot Comparison for Outlier Removal Methods

# In[28]:


# Prepare cleaned DataFrames (from Q2)
dfs = {
    'Original': df,
    'Mean±3SD': df_mean_sd,
    'Percentile': df_percentile,
    'IQR': df_iqr,
    'Z-Score': df_zscore
}

# Plot boxplots
plt.figure(figsize=(15, 6))
for i, (name, d) in enumerate(dfs.items()):
    plt.subplot(1, 5, i+1)
    sns.boxplot(y=d['price_per_sqft'])
    plt.title(name)
    plt.tight_layout()
plt.show()


# ## Q4: Check Normality of price_per_sqft and Apply Transformation if Needed

# #### Step 1: Histogram with KDE

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['price_per_sqft'], kde=True)
plt.title('Original price_per_sqft Distribution')
plt.show()


# #### Step 2: Check Skewness & Kurtosis

# In[35]:


print("Skewness:", df['price_per_sqft'].skew())
print("Kurtosis:", df['price_per_sqft'].kurt())


# #### Step 3: Apply Log Transformation (if right-skewed)

# In[40]:


import numpy as np

df['log_price_per_sqft'] = np.log1p(df['price_per_sqft'])

sns.histplot(df['log_price_per_sqft'], kde=True)
plt.title('Log-Transformed price_per_sqft')
plt.show()

print("Skewness (log):", df['log_price_per_sqft'].skew())
print("Kurtosis (log):", df['log_price_per_sqft'].kurt())


# ## Q5: Correlation Heatmap of Numerical Columns
# #### Step 1: Compute Correlation Matrix
# 
# 

# In[45]:


correlation_matrix = df.corr(numeric_only=True)
correlation_matrix


# #### Step 2: Plot Heatmap

# In[48]:


plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ## Q6: Scatter Plots Between Variables
# #### Step 1: Select Relevant Numerical Columns

# In[51]:


columns_to_plot = ['total_sqft', 'bhk', 'bath', 'price', 'price_per_sqft']


# #### 🔹 Step 2: Create Pairwise Scatter Plots

# In[54]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df[columns_to_plot])
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()


# In[ ]:




