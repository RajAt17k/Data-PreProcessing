#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # Loading Data

# In[2]:



df=pd.read_csv("House_Rent_Dataset.csv")
df.head()


# In[3]:


df.size


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(df['Rent'].value_counts())


# Mean , Median , Maximum and Minimum values for Rent

# In[8]:


print("Mean House Rent:", round(df["Rent"].mean()))
print("Median House Rent:", round(df["Rent"].median()))
print("Highest House Rent:", round(df["Rent"].max()))
print("Lowest House Rent:", round(df["Rent"].min()))


# In[9]:


df["Rent"].sort_values(ascending = False)[:5]


# In[10]:


df["Rent"].sort_values()[:5]


# # Visualizing Raw Data

# Houses available in Different City

# In[11]:


sns.set_context("poster", font_scale = .8)
plt.figure(figsize = (20, 6))
ax = df["City"].value_counts().plot(kind = 'bar', color = "green", rot = 0)

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() - 100), ha = 'center', va = 'bottom', color = 'white')


# In[12]:


plt.figure(figsize = (20, 7))
sns.barplot(x = df["City"], y = df["Rent"], palette = "nipy_spectral");


# # Data Cleaning

# In[13]:


df.isnull().sum().sort_values(ascending=False)


# In[14]:


df.duplicated().sum()


# In[15]:


print(pd.get_dummies(df['Area Type']).head(5))


# In[16]:


for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))


# In[17]:


# Checking how well observations are distributed for each features


# In[18]:


print(df['Posted On'].value_counts().sort_values(ascending=False).head(10))


# In[19]:


print(df['Floor'].value_counts().sort_values(ascending=False).head(25))


# In[20]:


print(df['Area Type'].value_counts().sort_values(ascending=False).head(10))


# In[21]:


print(df['Area Locality'].value_counts().sort_values(ascending=False).head(10))


# In[22]:


print(df['City'].value_counts().sort_values(ascending=False).head(10))


# In[23]:


print(df['Furnishing Status'].value_counts().sort_values(ascending=False).head(10))


# In[24]:


print(df['Tenant Preferred'].value_counts().sort_values(ascending=False).head(10))


# In[25]:


print(df['Point of Contact'].value_counts().sort_values(ascending=False).head(10))


# In[26]:


df["Total Floors"] = df["Floor"].apply(lambda floor:floor.split()[-1])
df["Floor"] = df["Floor"].apply(lambda floor:floor.split()[0])
df.head()


# # Dealing with Categorical Data

# In[27]:


df = pd.get_dummies(df, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
df.head()


# Dropping irrelevant features:
# 
# 

# In[28]:


df.drop(columns='Posted On', inplace=True)
df.drop(columns='Area Locality', inplace=True)


# In[29]:


df.head()


# # Checking for Outlier  - Kernel Density Estimation

# In[30]:


def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values


# In[31]:


tukey_indices, tukey_values = find_outliers_tukey(df['Rent'])
print(np.sort(tukey_values))


# # Distribution of Features

# In[32]:


n_bins = 20
plt.figure(figsize = (20, 6))
df["Size"].hist(bins = n_bins);


# Size vs Rent Comparison

# In[33]:


plt.figure(figsize = (20, 6))
plt.ticklabel_format(style = 'plain')
plt.scatter(df["Size"], df["Rent"])
plt.xlabel("Size")
plt.ylabel("Rent");


# In[36]:


px.histogram(df, x="Size", color_discrete_sequence=['crimson'],title="Size Distribution")


# In[41]:


X = df.drop('Rent',axis= 1)
y = df.Rent


# In[42]:


X.head()


# In[43]:


y.head()


# In[ ]:




