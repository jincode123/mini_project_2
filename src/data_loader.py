#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os


# In[2]:


# Set the default figure size
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(color_codes = True)
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', None)


# In[3]:


# path to data
path = kagglehub.dataset_download("yogape/logistics-operations-database")

print("Path to dataset files:", path)
os.listdir(path)


# In[4]:


database_schema_path = r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\DATABASE_SCHEMA.txt"

def load_database_schema():
    with open(database_schema_path, "r", encoding='utf-8') as file:
        return file.read()


# In[5]:


def load_customers():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\customers.csv")


# In[6]:


def load_delivery_events():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\delivery_events.csv")


# In[7]:


def load_drivers():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\drivers.csv")


# In[8]:


def load_driver_monthly_metrics():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\driver_monthly_metrics.csv")


# In[9]:


def load_facilities():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\facilities.csv")


# In[10]:


def load_fuel_purchases():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\fuel_purchases.csv")


# In[11]:


def load_loads():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\loads.csv")


# In[12]:


def load_maintenance_records():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\maintenance_records.csv")


# In[13]:


def load_routes():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\routes.csv")


# In[14]:


def load_trailers():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\trailers.csv")


# In[15]:


def load_trips():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\trips.csv")


# In[16]:


def load_safety_incidents():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\safety_incidents.csv")


# In[17]:


def load_trucks():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\trucks.csv")


# In[18]:


def load_truck_utilization_metrics():
    return pd.read_csv(r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\truck_utilization_metrics.csv")

