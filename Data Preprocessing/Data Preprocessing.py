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


df.corr()['Rent'].sort_values(ascending= False)


# In[8]:


print(df['Rent'].value_counts())


# Mean , Median , Maximum and Minimum values for Rent

# In[9]:


print("Mean House Rent:", round(df["Rent"].mean()))
print("Median House Rent:", round(df["Rent"].median()))
print("Highest House Rent:", round(df["Rent"].max()))
print("Lowest House Rent:", round(df["Rent"].min()))


# In[10]:


df["Rent"].sort_values(ascending = False)[:5]


# In[11]:


df["Rent"].sort_values()[:5]


# # Visualizing Raw Data

# Houses available in Different City

# In[12]:


sns.set_context("poster", font_scale = .8)
plt.figure(figsize = (20, 6))
ax = df["City"].value_counts().plot(kind = 'bar', color = "green", rot = 0)

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() - 100), ha = 'center', va = 'bottom', color = 'white')


# In[13]:


plt.figure(figsize = (20, 7))
sns.barplot(x = df["City"], y = df["Rent"], palette = "nipy_spectral");


# # Data Cleaning

# In[14]:


df.isnull().sum().sort_values(ascending=False)


# In[15]:


df.duplicated().sum()


# In[16]:


print(pd.get_dummies(df['Area Type']).head(5))


# In[17]:


for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))


# In[18]:


# Checking how well observations are distributed for each features


# In[19]:


print(df['Posted On'].value_counts().sort_values(ascending=False).head(10))


# In[20]:


print(df['Floor'].value_counts().sort_values(ascending=False).head(25))


# In[21]:


print(df['Area Type'].value_counts().sort_values(ascending=False).head(10))


# In[22]:


print(df['Area Locality'].value_counts().sort_values(ascending=False).head(10))


# In[23]:


print(df['City'].value_counts().sort_values(ascending=False).head(10))


# In[24]:


print(df['Furnishing Status'].value_counts().sort_values(ascending=False).head(10))


# In[25]:


print(df['Tenant Preferred'].value_counts().sort_values(ascending=False).head(10))


# In[26]:


print(df['Point of Contact'].value_counts().sort_values(ascending=False).head(10))


# In[27]:


df["Total Floors"] = df["Floor"].apply(lambda floor:floor.split()[-1])
df["Floor"] = df["Floor"].apply(lambda floor:floor.split()[0])
df.head()


# # Dealing with Categorical Data

# In[28]:


df = pd.get_dummies(df, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
df.head()


# Dropping irrelevant features:
# 
# 

# In[29]:


df.drop(columns='Posted On', inplace=True)
df.drop(columns='Area Locality', inplace=True)


# In[30]:


df.head()


# # Checking for Outlier  - Kernel Density Estimation

# In[31]:


def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values


# In[32]:


tukey_indices, tukey_values = find_outliers_tukey(df['Rent'])
print(np.sort(tukey_values))


# # Distribution of Features

# In[33]:


n_bins = 20
plt.figure(figsize = (20, 6))
df["Size"].hist(bins = n_bins);


# Size vs Rent Comparison

# In[34]:


plt.figure(figsize = (20, 6))
plt.ticklabel_format(style = 'plain')
plt.scatter(df["Size"], df["Rent"])
plt.xlabel("Size")
plt.ylabel("Rent");


# In[35]:


px.histogram(df, x="Size", color_discrete_sequence=['crimson'],title="Size Distribution")


# In[36]:


def vis_dist(df, col, lim=False):
    variable = df[col].values
    ax = sns.displot(variable)
    plt.title(f'Distribution of {col}')
    plt.xlabel(f'{col}')
    if lim:
        plt.xlim(0, 4000)
    return plt.show()
vis_dist(df, 'Rent')
print("distribution highly skewed to the right")


# In[37]:


print("Log Transformation")
df['Rent'] = np.log1p(df['Rent'])
vis_dist(df, 'Rent')


# In[38]:


df.corr()


# In[39]:


def corr_sort_matrix(df, sort_desc, topn):
    corr_matrix = df.corr()
    if sort_desc == True:
        sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                          .stack()
                          .sort_values(ascending=False))
    else:
        sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=True))
    corr = pd.DataFrame(sol, columns=['corr']).reset_index()
    corr.columns = ['antecedent', 'consequent','correlation']
    return corr.head(topn)

# top positive correlated pairs
df_corr = corr_sort_matrix(df, True, 10000)
df_corr.head(10)


# In[40]:


X=df.drop('Rent',axis=1)
Y=df.Rent


# In[41]:


X.head()


# In[42]:


Y.head()


# In[ ]:




