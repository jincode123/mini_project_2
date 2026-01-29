#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_maintenance_records


# In[2]:


# Set the default figure size
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes = True)
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', None)


# In[11]:


maintenance_df = load_maintenance_records()


# In[12]:


maintenance_df['maintenance_date']=pd.to_datetime(maintenance_df['maintenance_date'])
maintenance_df['year']=pd.to_datetime(maintenance_df['maintenance_date']).dt.year
maintenance_df['month_year']=pd.to_datetime(maintenance_df['maintenance_date']).dt.to_period('M')


# In[16]:


yearly = (
    maintenance_df.groupby('year')['maintenance_id']
    .count()
    .reset_index()
)

plt.plot(yearly['year'], yearly['maintenance_id'], marker='o')
plt.title('Yearly maintenance Trend')
plt.xlabel('Year')
plt.ylabel('maintenance')
plt.grid(True)
plt.show()


# In[13]:


print(maintenance_df.head())


# In[15]:


# Create a mapping function
def categorize_service(desc):
    if 'Emergency' in desc:
        return 'Emergency'
    elif 'Scheduled' in desc:
        return 'Scheduled'    
    else:
        return 'Routine'

# Apply to dataframe
maintenance_df['category'] = maintenance_df['service_description'].apply(categorize_service)

# Plot the trend
sns.countplot(data=maintenance_df, x='year', hue='category')


# In[27]:


# 1. Filter the data first
# Note: Ensure the column name is exactly 'service_description'
scheduled_tire_df = maintenance_df[maintenance_df['service_description'] == 'Scheduled Tire']

# 2. Create the countplot
plt.figure(figsize=(8, 6))

# Assign x to hue and set legend=False to avoid future warnings
ax = sns.countplot(
    data=scheduled_tire_df, 
    x='year', 
    hue='year', 
    palette='viridis', 
    legend=False
)

# 3. Add the number labels on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10),              # Move the label 10 points above the bar
                textcoords = 'offset points',
                fontsize=12,
                fontweight='bold')

# 4. Optional: Add a bit of headroom to the Y-axis so labels aren't cut off
plt.ylim(0, scheduled_tire_df['year'].value_counts().max() * 1.15)

plt.title('Trend of Scheduled Tire Services (2022-2024)', fontsize=14)
plt.ylabel('Number of Services', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()


# <div style="font-size:40px"> # Relationship between Features and maintenance records </div>
