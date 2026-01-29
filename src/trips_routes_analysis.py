#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_trips


# In[2]:


# Set the default figure size
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes = True)
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', None)


# In[12]:


trips_df = load_trips()
print(trips_df.isnull().sum())


# In[16]:


trips_df.info()


# In[14]:


trips_df['dispatch_date']=pd.to_datetime(trips_df['dispatch_date'])
trips_df['year']=pd.to_datetime(trips_df['dispatch_date']).dt.year
trips_df['month_year']=pd.to_datetime(trips_df['dispatch_date']).dt.to_period('M')


# In[27]:


yearly_trips = (
    trips_df.groupby('year')['trip_id']
    .count()
    .reset_index()
)

# 2. Create the bar chart
plt.figure(figsize=(8, 6))

# Use hue='year' and legend=False to avoid the FutureWarning
ax = sns.barplot(
    data=yearly_trips, 
    x='year', 
    y='trip_id', 
    hue='year', 
    palette='viridis', 
    legend=False
)

# 3. THE TRICK: Zoom in on the Y-axis to make differences obvious
# We set the bottom of the axis just below your lowest trip count
min_trips = yearly_trips['trip_id'].min()
max_trips = yearly_trips['trip_id'].max()
plt.ylim(min_trips * 0.95, max_trips * 1.05) 

# 4. Add data labels on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), ',.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.title('Yearly Trip Trend')
plt.xlabel('Year')
plt.ylabel('Total Trips')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[28]:


yearly_miles = (
    trips_df.groupby('year')['actual_distance_miles']
    .sum()
    .reset_index()
)

# 2. Create the bar chart
plt.figure(figsize=(10, 6))

# Use hue='year' and legend=False to follow Seaborn best practices
ax = sns.barplot(
    data=yearly_miles, 
    x='year', 
    y='actual_distance_miles', 
    hue='year', 
    palette='viridis', 
    legend=False
)

# 3. THE TRICK: Zoom in on the Y-axis to see the small differences in mileage
# Mileage stays between 40.5M and 40.9M, so we zoom into that range
plt.ylim(40_000_000, 41_200_000) 

# 4. Format the Y-axis for readability (commas instead of scientific notation)
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# 5. Add data labels on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), ',.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.title('Yearly Miles Trend (Zoomed View)')
plt.xlabel('Year')
plt.ylabel('Total Miles')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[15]:


print(trips_df.head())


# In[ ]:




