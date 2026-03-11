#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.data import load_raw_tables, prepare_delivery_events, build_master_df, add_engineering_features
from src.preprocessing_data import split_data, scale_data, prepare_training_data
from src.feature_relationship import apply_pca
from sklearn.svm import SVC


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


# #PCA + LogisticRegression

# In[9]:


# train model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
# Set parameter grid
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"]
}


# In[10]:


# GridSearchCV
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",   # good for multiclass or imbalanced classes
    n_jobs=-1
)

grid_search.fit(X_train_pca, y_train)


# In[11]:


# Best model
best_log_reg = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


# In[12]:


# Predictions
y_pred = best_log_reg.predict(X_test_pca)


# In[13]:


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Logistic Regression + GridSearchCV Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# In[14]:


# 12. Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[15]:


# 13. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[16]:


# 14. Save results in a table
results_df = pd.DataFrame([
    {
        "Model": "PCA + Logistic Regression + GridSearchCV",
        "Best Params": str(grid_search.best_params_),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
])

results_df


# In[18]:


#PCA + (Stochastic Gradient Descent) SGDClassifier + GridSearchCV


# In[19]:


from sklearn.linear_model import SGDClassifier

# Define model
sgd_model = SGDClassifier(random_state=42)

#  Parameter grid
param_grid = {
    "loss": ["hinge", "log_loss"],
    "alpha": [0.0001, 0.001, 0.01],
    "max_iter": [1000, 2000]
}

#  Grid search
grid_search = GridSearchCV(
    estimator=sgd_model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_pca, y_train)


# In[20]:


# Best model
best_sgd = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Predict
y_pred = best_sgd.predict(X_test_pca)

# 11. Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nSGDClassifier + PCA + GridSearchCV Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[21]:


sgd_results = pd.DataFrame([
    {
        "Model": "SGDClassifier + PCA + GridSearchCV",
        "Best Params": str(grid_search.best_params_),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
])

sgd_results


# # LinearSVC + PCA + GridSearchCV

# In[23]:


from sklearn.svm import LinearSVC

# Define model
linear_svc = LinearSVC(random_state=42, max_iter=5000)

# Parameter grid
param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

# Grid search
grid_search = GridSearchCV(
    estimator=linear_svc,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_pca, y_train)


# In[24]:


# Best model
best_linear_svc = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Predict
y_pred = best_linear_svc.predict(X_test_pca)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLinearSVC + PCA + GridSearchCV Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[25]:


linear_svc_results = pd.DataFrame([
    {
        "Model": "LinearSVC + PCA + GridSearchCV",
        "Best Params": str(grid_search.best_params_),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
])

linear_svc_results


# In[ ]:




