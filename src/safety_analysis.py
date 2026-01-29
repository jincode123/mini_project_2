#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_safety_incidents


# In[2]:


# Set the default figure size
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes = True)
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', None)


# In[13]:


safety_df = load_safety_incidents()
print(safety_df.isnull().sum())


# In[8]:


safety_df.description.unique()


# In[14]:


safety_df['incident_date']=pd.to_datetime(safety_df['incident_date'])
safety_df['year']=pd.to_datetime(safety_df['incident_date']).dt.year
safety_df['month_year']=pd.to_datetime(safety_df['incident_date']).dt.to_period('M')


# In[15]:


print(safety_df.head())


# In[79]:


yearly_incidents = (
    safety_df.groupby('year')['incident_id']
    .count()
    .reset_index()
)

plt.figure(figsize=(8, 6))

# Update: Assign x to hue and set legend=False to satisfy the new Seaborn requirements
ax = sns.barplot(
    data=yearly_incidents, 
    x='year', 
    y='incident_id', 
    hue='year', 
    palette='crest', 
    legend=False
)

# THE TRICK: Zoom in on the Y-axis to make the jump from 54 to 60 look more dramatic
# Since your min is 54 and max is 60, starting at 50 makes the change very clear
plt.ylim(50, 62) 

# Add data labels on top of the bars so the exact numbers are visible
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.title('Yearly Incident Trend')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.show()


# <div style="font-size:40px"> # Relationship between Features and preventable cases </div>

# In[80]:


def bar_charts_preventable(df, feature):
    import matplotlib.pyplot as plt

    _agg = {'incident_id': 'count'}
    _groupby = ['year', 'preventable_flag', feature]

    df_feature = (
        df.groupby(_groupby)
          .agg(_agg)
          .reset_index()
    )

    years = sorted(df_feature['year'].unique())

    fig, axes = plt.subplots(1, len(years), figsize=(20, 6), sharey=True)

    handles, labels = None, None

    for ax, year in zip(axes, years):
        subset = df_feature[df_feature['year'] == year]

        plot_df = subset.pivot(
            index='preventable_flag',
            columns=feature,
            values='incident_id'
        )

        ax_plot = plot_df.plot(kind='bar', ax=ax, legend=False)

        if handles is None:
            handles, labels = ax_plot.get_legend_handles_labels()

        ax.set_title(f'Year {year}')
        ax.set_xlabel('Preventable')
        ax.set_xticklabels(['No', 'Yes'], rotation=0)

    axes[0].set_ylabel('Incident Count')

    # 🔑 Shared legend outside
    fig.legend(
        handles,
        labels,
        title=feature,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout()
    plt.show()



# <div style="font-size:40px"> description </div>

# In[53]:


# Create a mapping function
def categorize_description(desc):
    if 'equipment' in desc:
        return 'Equipment'
    elif 'driver' in desc:
        return 'Other Driver' 
    elif 'traffic' in desc:
        return 'Traffic'       
    else:
        return 'Weather'

# Apply to dataframe
safety_df['category'] = safety_df['description'].apply(categorize_description)


# In[54]:


bar_charts_preventable(safety_df, 'category')


# In[55]:


bar_charts_preventable(safety_df, 'incident_type')


# In[56]:


bar_charts_preventable(safety_df, 'at_fault_flag')


# In[59]:


def bar_charts_at_fault_flag(df, feature):
    import matplotlib.pyplot as plt

    _agg = {'incident_id': 'count'}
    _groupby = ['year', 'at_fault_flag', feature]

    df_feature = (
        df.groupby(_groupby)
          .agg(_agg)
          .reset_index()
    )

    years = sorted(df_feature['year'].unique())

    fig, axes = plt.subplots(1, len(years), figsize=(20, 6), sharey=True)

    handles, labels = None, None

    for ax, year in zip(axes, years):
        subset = df_feature[df_feature['year'] == year]

        plot_df = subset.pivot(
            index='at_fault_flag',
            columns=feature,
            values='incident_id'
        )

        ax_plot = plot_df.plot(kind='bar', ax=ax, legend=False)

        if handles is None:
            handles, labels = ax_plot.get_legend_handles_labels()

        ax.set_title(f'Year {year}')
        ax.set_xlabel('at_fault')
        ax.set_xticklabels(['No', 'Yes'], rotation=0)

    axes[0].set_ylabel('Incident Count')

    # 🔑 Shared legend outside
    fig.legend(
        handles,
        labels,
        title=feature,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout()
    plt.show()


# In[60]:


bar_charts_at_fault_flag(safety_df, 'category')


# In[ ]:




