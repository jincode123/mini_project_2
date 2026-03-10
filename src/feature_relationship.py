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


df.head()


# In[5]:


# Final Column Selection (The Optimized List)
final_columns = [
    'on_time_flag', 
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


# In[6]:


# Separate Features and Target
categorical_cols = ['day_of_week', 'event_type', 'booking_type', 'load_type']
numeric_cols = [
    'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
    'weight_lbs', 'pieces', 'revenue', 'years_experience', 
    'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
    'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density','is_winter'
]
target = 'on_time_flag'


# In[7]:


# Create df for model training
df_model = df[categorical_cols + numeric_cols + [target]].copy()


# In[8]:


# HEATMAP: Correlation of Numeric Features
plt.figure(figsize=(12, 8))
correlation_matrix_n = df_model[numeric_cols + [target]].corr()
# create upper-triangle mask
mask = np.triu(np.ones_like(correlation_matrix_n, dtype=bool))
sns.heatmap(correlation_matrix_n, annot=True, cmap='coolwarm', fmt=".2f", mask=mask,)
plt.title('Correlation Heatmap: Logistics Performance Drivers')
plt.show() # Or plt.savefig('heatmap.png')


# In[9]:


# ONE-HOT ENCODING to convertstext categories into 0/1 columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

dummy_cols = [col for col in df_encoded.columns if any(col.startswith(cat + "_") for cat in categorical_cols)]

correlation_matrix_c = df_encoded[dummy_cols + [target]].corr()


# In[10]:


# HEATMAP: Correlation of Numeric Features
plt.figure(figsize=(12, 8))
correlation_matrix_c = df_encoded[dummy_cols + [target]].corr()
# create upper-triangle mask
mask = np.triu(np.ones_like(correlation_matrix_c, dtype=bool))
sns.heatmap(correlation_matrix_c, annot=True, cmap='coolwarm', fmt=".2f", mask=mask,)
plt.title('Correlation Heatmap: Logistics Performance Drivers')
plt.show() # Or plt.savefig('heatmap.png')


# ### we identified several instances of Multicollinearity—where features were too closely related to each other. To ensure our Logistic Regression model remained stable and interpretable, we performed Feature Selection. We prioritized our 'Smart' engineered features, like the Detention Risk Index, over raw metrics because they provided a more condensed and powerful signal, reducing redundancy and improving model efficiency."
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

# In[11]:


from sklearn.decomposition import PCA


# In[12]:


X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = prepare_training_data(
    df_encoded, target
)


# # 6. SPLIT DATA
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 1. Scaling 
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# X_train_imputed = imputer.fit_transform(X_train)
# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
# X_test_imputed = imputer.transform(X_test)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

# In[13]:


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


# In[14]:


def apply_pca(X_train_scaled, X_test_scaled, n_components=0.95):
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, pca


# "We split the data before scaling to prevent Data Leakage. By fitting the scaler only on the training data, we ensure that the model has no knowledge of the mean or standard deviation of the test set, which simulates a real-world scenario where the model sees completely new, unseen data."
