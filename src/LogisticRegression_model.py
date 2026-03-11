#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.data import load_raw_tables, prepare_delivery_events, build_master_df, add_engineering_features
from src.preprocessing_data import split_data, scale_data, prepare_training_data
from src.feature_relationship import apply_pca


# In[2]:


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


# In[3]:


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


# In[4]:


# Separate Features and Target
categorical_cols = ['day_of_week', 'event_type', 'booking_type', 'load_type']
numeric_cols = [
    'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
    'weight_lbs', 'pieces', 'revenue', 'years_experience', 
    'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
    'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density','is_winter'
]
target = 'on_time_flag'


# In[5]:


# Create df for model training
df_model = df[categorical_cols + numeric_cols + [target]].copy()


# In[6]:


# ONE-HOT ENCODING to convertstext categories into 0/1 columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)


# In[7]:


X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler = prepare_training_data(
    df_encoded, target
)


# In[8]:


X_train_pca, X_test_pca, pca = apply_pca(
    X_train_scaled,
    X_test_scaled,
    n_components=21   # or 0.95 if you want automatic selection
)


# In[9]:


model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train_pca, y_train)


# In[10]:


y_pred = model.predict(X_test_pca)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Logistic Regression")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# In[11]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[12]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[13]:


results_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
])

results_df


# In[ ]:




