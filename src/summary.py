#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_safety_incidents,load_trips,load_maintenance_records


# In[2]:


# Set the default figure size
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes = True)
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', None)


# In[3]:


safety_df = load_safety_incidents()
trips_df = load_trips()
maintenance_df = load_maintenance_records()


# In[4]:


safety_df['incident_date']=pd.to_datetime(safety_df['incident_date'])
safety_df['year']=pd.to_datetime(safety_df['incident_date']).dt.year
safety_df['month_year']=pd.to_datetime(safety_df['incident_date']).dt.to_period('M')


# In[5]:


trips_df['dispatch_date']=pd.to_datetime(trips_df['dispatch_date'])
trips_df['year']=pd.to_datetime(trips_df['dispatch_date']).dt.year
trips_df['month_year']=pd.to_datetime(trips_df['dispatch_date']).dt.to_period('M')


# In[6]:


maintenance_df['maintenance_date']=pd.to_datetime(maintenance_df['maintenance_date'])
maintenance_df['year']=pd.to_datetime(maintenance_df['maintenance_date']).dt.year
maintenance_df['month_year']=pd.to_datetime(maintenance_df['maintenance_date']).dt.to_period('M')


# In[7]:


yearly_incidents = (
    safety_df.groupby('year')['incident_id']
    .count()
    .reset_index()
)


# In[8]:


yearly_trips = (
    trips_df.groupby('year')['trip_id']
    .count()
    .reset_index()
)
yearly_miles = (
    trips_df.groupby('year')['actual_distance_miles']
    .sum()
    .reset_index()
)


# In[9]:


yearly_maintenance_records = (
    maintenance_df.groupby('year')['maintenance_id']
    .count()
    .reset_index()
)
scheduled_df = maintenance_df[maintenance_df['service_description'].str.contains('Scheduled', case=False, na=False)]


# In[24]:


from IPython.display import HTML, display

def show_wide_style(df):
    html = df.to_html(index=False)
    display(HTML(f"""
    <div style="width:100%; overflow-x:auto;">
      <div style="min-width:max-content;">
        {html}
      </div>
    </div>
    """))


# In[25]:


#  Use a simpler approach to build the summary
def get_count(keyword):
    # This searches for the keyword anywhere in the description
    mask = scheduled_df['service_description'].str.contains(keyword, case=False, na=False)
    return scheduled_df[mask].groupby('year')['maintenance_id'].count()

#  Rebuild the table
summary_scheduled_df = pd.DataFrame({
    'scheduled_tires': get_count('Tire'),
    'scheduled_brake': get_count('Brake'),
    'scheduled_preventative': get_count('Prevent'), # Shortened to catch 'Preventative' or 'Preventive'
    'scheduled_engine': get_count('Engine')
}).reset_index()

print(summary_scheduled_df)


# In[26]:


show_wide_style(summary_scheduled_df.style.hide(axis="index"))


# In[27]:


# 1. Clean the original data safely
maintenance_df.loc[:, 'service_description'] = maintenance_df['service_description'].str.strip()

# 2. Define your keywords (renamed column headers to reflect 'emergency')
keywords = {
    'emergency_tires': 'Tire',
    'emergency_brake': 'Brake',
    'emergency_preventative': 'Prevent',
    'emergency_engine': 'Engine',
    'emergency_transmission': 'Transmission'
}

# 3. Build the summary by filtering for 'Emergency' + Keyword
summary_data = []
for col_name, key in keywords.items():
    # Swapped 'Scheduled' for 'Emergency'
    mask = (maintenance_df['service_description'].str.contains('Emergency', case=False, na=False)) & \
           (maintenance_df['service_description'].str.contains(key, case=False, na=False))

    counts = maintenance_df[mask].groupby('year')['maintenance_id'].count().reset_index(name=col_name)
    summary_data.append(counts)

# 4. Merge all summaries on 'year'
from functools import reduce
summary_emergency_df = reduce(lambda left, right: pd.merge(left, right, on='year', how='outer'), summary_data).fillna(0)

print(summary_emergency_df)


# In[28]:


show_wide_style(summary_emergency_df.style.hide(axis="index"))


# In[29]:


# 1. Clean the original data safely
maintenance_df.loc[:, 'service_description'] = maintenance_df['service_description'].str.strip()

# 2. Define keywords (renamed column headers to reflect 'routine')
keywords = {
    'routine_tires': 'Tire',
    'routine_brake': 'Brake',
    'routine_preventative': 'Prevent',
    'routine_engine': 'Engine',
    'routine_transmission': 'Transmission'
}

# 3. Build the summary by filtering for 'Routine' + Keyword
summary_data = []
for col_name, key in keywords.items():
    # Swapped 'Emergency' for 'Routine'
    mask = (maintenance_df['service_description'].str.contains('Routine', case=False, na=False)) & \
           (maintenance_df['service_description'].str.contains(key, case=False, na=False))

    counts = maintenance_df[mask].groupby('year')['maintenance_id'].count().reset_index(name=col_name)
    summary_data.append(counts)

# 4. Merge all summaries on 'year'
from functools import reduce
summary_routine_df = reduce(lambda left, right: pd.merge(left, right, on='year', how='outer'), summary_data).fillna(0)

print(summary_routine_df)


# In[30]:


show_wide_style(summary_routine_df.style.hide(axis="index"))


# Scheduled Tires: Dropped from 60 in 2022 to 38 in 2024 (a 36.7% decrease).
# 
# Emergency Tires: Dropped from 57 in 2022 to 39 in 2024 (a 31.6% decrease).
# 
# Routine Tires: Dropped from 48 in 2022 to 37 in 2024 (a 22.9% decrease).

# ['Emergency Inspection' 'Scheduled Tire' 'Routine Preventive'
#  'Emergency Repair' 'Scheduled Preventive' 'Routine Inspection'
#  'Emergency Preventive' 'Routine Transmission' 'Routine Brake'
#  'Scheduled Transmission' 'Scheduled Brake' 'Emergency Transmission'
#  'Emergency Brake' 'Routine Repair' 'Emergency Tire' 'Scheduled Repair'
#  'Emergency Engine' 'Routine Engine' 'Scheduled Inspection' 'Routine Tire'
#  'Scheduled Engine']
