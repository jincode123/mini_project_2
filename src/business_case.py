#!/usr/bin/env python
# coding: utf-8

# In[1]:


import importlib
import src.data
importlib.reload(src.data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing_data import prepare_training_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# In[2]:


from src.data import load_raw_tables, prepare_delivery_events, build_master_df, add_engineering_features


# In[3]:


base_path = r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1"
tables = load_raw_tables(base_path)
delivery_events = prepare_delivery_events(tables["delivery_events"])
df = build_master_df(
    delivery_events,
    tables["trips"],
    tables["loads"],
    tables["drivers"],
    tables["trucks"],
    tables["routes"],
    tables["customers"]
)
df = add_engineering_features(df)


# In[4]:


df


# In[5]:


# Calculating the percentage of missing values for each column
missing_data = df.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100

# Prepare values
missing_percentage.sort_values(ascending=True, inplace=True)

# Plot the barh chart
fig, ax = plt.subplots(figsize=(15, 4))
ax.barh(missing_percentage.index, missing_percentage, color='#ff6200')

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
    ax.text(value+0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=18, color='black')

# Set x-axis limit
ax.set_xlim([0, 40])

# Add title and xlabel
plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
plt.xlabel('Percentages (%)', fontsize=16)
plt.show()


# In[6]:


# Extracting rows with missing values in 'CustomerID' or 'Description' columns
df[df['is_home_terminal'].isnull() | df['years_experience'].isnull()| df['efficiency_ratio'].isnull()| df['truck_age_at_event'].isnull()]


# In[7]:


# Removing rows with missing values in 'CustomerID' and 'Description' columns
df = df.dropna(subset=['is_home_terminal', 'years_experience','efficiency_ratio','truck_age_at_event'])
df.head()


# In[8]:


# 1. Count UNIQUE trip_ids for each facility_id
# This ensures we are counting the actual 'Trips' handled by that location
facility_trip_data = df.groupby('facility_id')['trip_id'].nunique().reset_index(name='trips_per_facility')

# 2. Merge this back into your main training dataframe
df = df.merge(facility_trip_data, on='facility_id', how='left')


# In[9]:


# 1. Get the unique list of facilities and their trip counts
# 2. Sort by 'trips_per_facility' from highest to lowest
# 3. Take the top 10
top_10_facilities = (
    df[['facility_id', 'trips_per_facility']]
    .drop_duplicates()
    .sort_values(by='trips_per_facility', ascending=False)
    .head(10)
)

print("Top 10 Busiest Facilities (By Trip Volume):")
print(top_10_facilities)


# # Logistic Operation on-time-delivery rate at 55%

# In[10]:


# Group by state to see On-Time Percentage vs Trip Count
state_analysis = df.groupby('location_state').agg(
    total_trips=('trip_id', 'nunique'),
    on_time_rate=('on_time_flag', 'mean')
).reset_index()

# Sort by total_trips to see if the top states have the lowest on_time_rate
print(state_analysis.sort_values('on_time_rate', ascending=False))


# In[11]:


print(state_analysis['on_time_rate'].mean())


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.regplot(data=state_analysis, x='total_trips', y='on_time_rate', color='red')

plt.title('Volume vs. On-Time Performance')
plt.xlabel('Total Trips per State (Demand)')
plt.ylabel('On-Time Delivery Rate (%)')
plt.grid(True, alpha=0.3)
plt.show()


# In[13]:


df['on_time_flag'].value_counts()


# In[14]:


stats = df[['facility_id', 'trips_per_facility']].drop_duplicates()['trips_per_facility'].describe()
print(stats)


# "IN" is the abbreviation for Indiana.
# 
# Looking at your chart, Indiana showing the lowest on-time rate is a massive "smoking gun" for your logistics story. If you look back at your Trip Density Map, you'll likely see that Indiana is a major transit hub (often called the "Crossroads of America").
# 
# Why Indiana (IN) matters for your Model:
# High Throughput: Because so many national routes pass through Indiana, its facilities often face higher "congestion stress" than coastal states.
# 
# Feature Engineering: This validates why location_state is a critical feature. Your model will learn that a trip entering Indiana has a statistically higher "Risk Profile" than a trip in a state with a higher on-time rate.
# 
# Presentation Insight:
# You can highlight this specifically to the judges:
# 
# "Our state-level analysis identified Indiana (IN) as having the lowest on-time delivery rate across the network. Given Indiana's role as a central logistics hub, this suggests that regional congestion—rather than just individual facility issues—is a primary driver of delays. We used this insight to ensure our model weights geographic location heavily when calculating risk scores."

# In[15]:


import time
import folium
from geopy.geocoders import Nominatim

# 1. Aggregate Trip Counts by State
# Using size() to count occurrences of each trip per state
state_traffic = (
    df.groupby("location_state")
    .size()
    .reset_index(name="trip_count")
)

# 2. Map State Abbreviations to Full Names
state_map = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}

state_traffic["state_name"] = state_traffic["location_state"].map(state_map)
state_traffic["location"] = state_traffic["state_name"] + ", USA"

# 3. Geocoding with a Timer (Safe for Nominatim)
geolocator = Nominatim(user_agent="trip_density_map", timeout=10)
coords = {}

print("Starting geocoding for states...")
for loc in state_traffic["location"]:
    if pd.isna(loc): continue 
    try:
        geo = geolocator.geocode(loc)
        if geo:
            coords[loc] = (geo.latitude, geo.longitude)
        else:
            coords[loc] = (None, None)
        time.sleep(1) # Respect Rate Limiting
    except Exception as e:
        coords[loc] = (None, None)

# 4. Map Coordinates back to the Dataframe
state_traffic["lat"] = state_traffic["location"].map(lambda x: coords.get(x, (None, None))[0])
state_traffic["lon"] = state_traffic["location"].map(lambda x: coords.get(x, (None, None))[1])
state_traffic = state_traffic.dropna(subset=["lat", "lon"])

# 5. Create the Folium Map
m = folium.Map(location=[39, -98], zoom_start=4, width=800, height=500)

for _, row in state_traffic.iterrows():
    # Scale radius: Using log or a smaller multiplier since trip counts are high
    # We don't want a circle covering the whole map!
    dynamic_radius = max(5, (row["trip_count"] / df.shape[0]) * 500) 

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=dynamic_radius,
        color="blue", # Using blue to differentiate from the red "Safety" map
        fill=True,
        fill_opacity=0.6,
        popup=f"State: {row['location_state']}<br>Total Trips: {row['trip_count']:,}"
    ).add_to(m)
m


# In[16]:


# Final Column Selection (The Optimized List)
final_columns = [
    'on_time_flag', 
    'location_state'
    'facility_congestion_score', 
    'facility_detention_avg',
    'hour_of_day', 
    'day_of_week', 
    'event_type',
    'booking_type',
    'weight_lbs', 
    'pieces',
    'revenue',
    'load_type',
    'years_experience', 
    'truck_age_at_event',
    'average_mpg', 
    'idle_time_hours',
    'distance_deviation',
    'is_home_terminal', 
    'detention_risk_index', 
    'efficiency_ratio', 
    'revenue_density',
    'is_winter'
]


# In[17]:


# Separate Features and Target
categorical_cols = ['day_of_week', 'event_type', 'booking_type', 'load_type','location_state']
numeric_cols = [
    'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
    'weight_lbs', 'pieces', 'revenue', 'years_experience', 
    'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
    'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density','is_winter'
]
target = 'on_time_flag'


# In[18]:


# Create df for model training
df_model = df[categorical_cols + numeric_cols + [target]].copy()


# In[19]:


# HEATMAP: Correlation of Numeric Features
plt.figure(figsize=(12, 8))
correlation_matrix_n = df_model[numeric_cols + [target]].corr()
corr_filtered = correlation_matrix_n.copy()
corr_filtered[(corr_filtered > -0.1) & (corr_filtered < 0.1)] = np.nan
# create upper-triangle mask
mask = np.triu(np.ones_like(correlation_matrix_n, dtype=bool))
sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', fmt=".2f", mask=mask,)
plt.title('Correlation Heatmap: Logistics Performance Drivers')
plt.show() # Or plt.savefig('heatmap.png')


# In[20]:


# ONE-HOT ENCODING to convertstext categories into 0/1 columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

dummy_cols = [col for col in df_encoded.columns if any(col.startswith(cat + "_") for cat in categorical_cols)]

correlation_matrix_c = df_encoded[dummy_cols + [target]].corr()


# In[21]:


# HEATMAP: Correlation of Numeric Features
plt.figure(figsize=(16,12))
correlation_matrix_c = df_encoded[dummy_cols + [target]].corr()
corr_filtered = correlation_matrix_c.copy()
# hide weak correlations
corr_filtered[(corr_filtered > -0.1) & (corr_filtered < 0.1)] = np.nan
# create upper-triangle mask
mask = np.triu(np.ones_like(corr_filtered, dtype=bool))
sns.heatmap(corr_filtered, annot=True, fmt=".2f",
            annot_kws={"size":8},cmap='coolwarm', center=0,mask=mask)
plt.title('Correlation Heatmap: Logisticcorr_filtereds Performance Drivers')
plt.show() # Or plt.savefig('heatmap.png')


# In[22]:


correlation_matrix_c['on_time_flag'].sort_values().plot.barh(figsize=(8,10))


# we identified several instances of Multicollinearity—where features were too closely related to each other. To ensure our Logistic Regression model remained stable and interpretable, we performed Feature Selection. We prioritized our 'Smart' engineered features, like the Detention Risk Index, over raw metrics because they provided a more condensed and powerful signal, reducing redundancy and improving model efficiency."
# Model Confusion: When two variables are highly correlated (e.g., facility_congestion_score and detention_risk_index), Logistic Regression struggles to figure out which one is actually causing the delay. This makes your "Odds Ratios" and coefficients unreliable.
# 
# Overfitting: Including redundant information can lead to a model that is "noisier" and harder to generalize to new data.
# 
# Redundancy: If pieces and weight_lbs have a correlation of, say, 0.95, they are essentially telling the same story. Keeping both adds computational cost without adding new information.
# 
# The "Rules of Thumb" for your PresentationCorrelation ValueActionReason0.0 to 0.7Keep BothLow to moderate correlation; both provide unique signals.0.7 to 0.9InvestigateHigh correlation. Check if one is a derivative of the other.0.9 to 1.0Remove OneExtreme redundancy. Keeping both will likely hurt your Logistic Regression coefficients.

# using PCA (Principal Component Analysis) is a highly effective way to handle the multicollinearity you saw in your heatmaps!
# Why use PCA for your project?
# Solves Multicollinearity: PCA components are mathematically guaranteed to be independent (orthogonal). This perfectly fixes the issues Logistic Regression has with correlated features.
# 
# Reduces Noise: It keeps the "signal" and ignores the random "noise" in the data.
# 
# Simplifies the Model: Instead of 20+ features, you might find that 5 or 10 Principal Components capture 95% of the information.

# 3. The "Downside" for your Presentation
# While PCA improves performance and handles correlation, it kills interpretability.
# 
# Before PCA: You can say, "If weight increases, the load is more likely to be late."
# 
# After PCA: You have to say, "If Principal Component 1 increases, the load is more likely to be late." (But you don't know exactly what PC1 represents—it's a mix of weight, pieces, and revenue).
# 
# 4. Presentation Speech (The "Pro" Strategy)
# If you use PCA, explain it as a solution to the heatmaps you shared:
# 
# Speech Script:
# "Observing the high multicollinearity in our heatmap, we decided to implement PCA (Principal Component Analysis). This allowed us to transform our correlated features into a set of linearly independent components. While this makes individual feature interpretation more complex, it significantly stabilized our Logistic Regression model and allowed us to capture 95% of the data's variance while reducing the dimensionality of the problem. This ensured the model focused on the 'signal' rather than the redundant 'noise' between similar variables like weight and pieces."

# In[23]:


from sklearn.decomposition import PCA


# In[24]:


X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = prepare_training_data(
    df_encoded, target
)


# In[25]:


# 2. Apply PCA
# We start by not limiting components to see how much variance they explain
pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)

# 3. Determine how many components to keep (The "Scree Plot")
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(exp_var_cumul) + 1), exp_var_cumul, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# 4. Re-run PCA with the chosen number of components (e.g., to keep 95% variance)
pca_final = PCA(n_components=0.95)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

print(f"Original feature count: {X_train_scaled.shape[1]}")
print(f"Reduced feature count: {X_train_pca.shape[1]}")


# In[26]:


def apply_pca(X_train_scaled, X_test_scaled, n_components=0.95):
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, pca


# In[27]:


# Calculate the Cumulative Sum of the Explained Variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Set the optimal k value (based on our analysis, we can choose 6)
optimal_k = 36
# Set seaborn plot style
sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')
# Plot the cumulative explained variance against the number of components
plt.figure(figsize=(20, 10))

# Bar chart for the explained variance of each component
barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                      y=explained_variance_ratio,
                      color='#fcc36d',
                      alpha=0.8)

# Line plot for the cumulative explained variance
lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                     marker='o', linestyle='--', color='#ff6200', linewidth=2)
# Plot optimal k value line
optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}')
# Set labels and title
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Explained Variance', fontsize=14)
plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

# Customize ticks and legend
plt.xticks(range(0, len(cumulative_explained_variance)))
plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
           labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
           loc=(0.62, 0.1),
           frameon=True,
           framealpha=1.0,  
           edgecolor='#ff6200')
# Display the variance values for both graphs on the plots
x_offset = -0.3
y_offset = 0.01
for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
    if i > 0:
        plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

plt.grid(axis='both')   
plt.show()


# "We split the data before scaling to prevent Data Leakage. By fitting the scaler only on the training data, we ensure that the model has no knowledge of the mean or standard deviation of the test set, which simulates a real-world scenario where the model sees completely new, unseen data."
