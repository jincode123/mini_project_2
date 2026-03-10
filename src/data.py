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


import kagglehub

# Download latest version
path = kagglehub.dataset_download("yogape/logistics-operations-database")



# In[3]:


database_schema_path = r"C:\Users\jinfe\.cache\kagglehub\datasets\yogape\logistics-operations-database\versions\1\DATABASE_SCHEMA.txt"
def show_schema():
    with open(database_schema_path, "r", encoding="utf-8") as file:
        print(file.read())


# In[4]:


def load_raw_tables(base_path):
    customers = pd.read_csv(fr"{base_path}\customers.csv")
    delivery_events = pd.read_csv(fr"{base_path}\delivery_events.csv")
    drivers = pd.read_csv(fr"{base_path}\drivers.csv")
    loads = pd.read_csv(fr"{base_path}\loads.csv")
    routes = pd.read_csv(fr"{base_path}\routes.csv")
    trips = pd.read_csv(fr"{base_path}\trips.csv")
    trucks = pd.read_csv(fr"{base_path}\trucks.csv")
    facilities = pd.read_csv(fr"{base_path}\facilities.csv")

    return {
        "customers": customers,
        "delivery_events": delivery_events,
        "drivers": drivers,
        "loads": loads,
        "routes": routes,
        "trips": trips,
        "trucks": trucks,
        "facilities": facilities,
    }


# In[5]:


def prepare_delivery_events(delivery_events):
    delivery_events = delivery_events.copy()

    delivery_events["actual_datetime"] = pd.to_datetime(
        delivery_events["actual_datetime"],
        errors="coerce"
    )
    delivery_events["hour_of_day"] = delivery_events["actual_datetime"].dt.hour
    delivery_events["day_of_week"] = delivery_events["actual_datetime"].dt.day_name()

    congestion = (
        delivery_events
        .groupby(["facility_id", "day_of_week", "hour_of_day"])
        .size()
        .reset_index(name="facility_congestion_score")
    )

    detention = (
        delivery_events
        .groupby(["facility_id", "day_of_week", "hour_of_day"])["detention_minutes"]
        .mean()
        .reset_index(name="facility_detention_avg")
    )

    delivery_events = delivery_events.merge(
        congestion,
        on=["facility_id", "day_of_week", "hour_of_day"],
        how="left"
    )

    delivery_events = delivery_events.merge(
        detention,
        on=["facility_id", "day_of_week", "hour_of_day"],
        how="left"
    )

    return delivery_events


# In[6]:


def build_master_df(delivery_events, trips, loads, drivers, trucks, routes, customers):
    df = delivery_events.copy()

    df = df.merge(trips.drop(columns=["load_id"]), on="trip_id", how="left")
    df = df.merge(loads, on="load_id", how="left")
    df = df.merge(
        drivers[["driver_id", "years_experience", "home_terminal"]],
        on="driver_id",
        how="left"
    )
    df = df.merge(
        trucks[["truck_id", "make", "model_year"]],
        on="truck_id",
        how="left"
    )
    df = df.merge(
        routes[["route_id", "typical_distance_miles", "typical_transit_days"]],
        on="route_id",
        how="left"
    )
    df = df.merge(
        customers[["customer_id", "customer_type"]],
        on="customer_id",
        how="left"
    )

    return df


# In[7]:


def add_engineering_features(df):
    df = df.copy()
    df['month'] = df['actual_datetime'].dt.month
    df["truck_age_at_event"] = df["actual_datetime"].dt.year - df["model_year"]
    df["distance_deviation"] = df["actual_distance_miles"] - df["typical_distance_miles"]
    df["is_home_terminal"] = (df["home_terminal"] == df["location_city"]).astype(int)

    df["detention_risk_index"] = df["facility_congestion_score"] * df["facility_detention_avg"]
    df["efficiency_ratio"] = df["average_mpg"] / (df["truck_age_at_event"] + 1)
    df["revenue_density"] = df["revenue"] / (df["weight_lbs"] + 1)
    df['is_winter'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)

    return df


# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# 
# 
# # 4. ONE-HOT ENCODING
# # Converts text categories into 0/1 columns
# df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
# 
# # 6. SPLIT DATA
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # 2. Initialize the Scaler
# scaler = StandardScaler()
# 
# # 3. FIT and TRANSFORM the training data
# # This calculates the mean/std of X_train and applies it
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# 
# # 4. ONLY TRANSFORM the test data
# # We use the 'rules' learned from the training data to scale the test data
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
# 
# # 7. TRAIN RANDOM FOREST
# print("Training the model... please wait.")
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)
# 
# # 8. EVALUATE
# y_pred = model.predict(X_test)
# print(f"\nSuccess! Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nDetailed Performance Report:")
# print(classification_report(y_test, y_pred))
# 
# # 9. FEATURE IMPORTANCE
# # Shows which variables actually influenced the "On-Time" prediction
# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# importances.head(10).plot(kind='barh', color='teal')
# plt.title('Top 10 Predictors of On-Time Delivery')
# plt.gca().invert_yaxis()
# plt.show()

# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.impute import SimpleImputer # Added for fixing NaNs

# # 4. FIX NaNs (The Imputation Step)
# # This fills missing values with the median of each column
# imputer = SimpleImputer(strategy='median')
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)
# 
# # (Wait!) imputer returns a numpy array, let's put it back to a DataFrame for easier handling
# X_train = pd.DataFrame(X_train_imputed, columns=X.columns)
# X_test = pd.DataFrame(X_test_imputed, columns=X.columns)
# 
# # 5. Apply Scaling correctly
# scaler = StandardScaler()
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
# 
# # 6. Define Hyperparameter Grid
# param_distributions = {
#     'C': np.logspace(-3, 2, 20),
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear'] 
# }
# 
# # 7. Initialize and Run RandomizedSearchCV
# log_reg = LogisticRegression(max_iter=1000)
# random_search = RandomizedSearchCV(
#     estimator=log_reg, 
#     param_distributions=param_distributions, 
#     n_iter=10, 
#     cv=5, 
#     verbose=1, 
#     random_state=42, 
#     n_jobs=-1
# )
# 
# print("Searching for best Logistic Regression parameters (NaNs fixed)...")
# random_search.fit(X_train, y_train)
# 
# # 8. Final Results
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
# 
# print(f"\nBest Parameters: {random_search.best_params_}")
# print(f"Tuned Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# from imblearn.over_sampling import SMOTE # New import

# # 3. Scale FIRST
# # SMOTE relies on "distance" (K-Nearest Neighbors), so data must be scaled first
# scaler = StandardScaler()
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
# 
# # 4. Apply SMOTE (ONLY to Training Data)
# print(f"Original class distribution: {np.bincount(y_train)}")
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# print(f"Resampled class distribution: {np.bincount(y_train_res)}")
# 
# # 5. RandomizedSearchCV with Balanced Data
# param_distributions = {
#     'C': np.logspace(-3, 2, 20),
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear'] 
# }
# 
# random_search = RandomizedSearchCV(
#     estimator=LogisticRegression(max_iter=1000), 
#     param_distributions=param_distributions, 
#     n_iter=10, 
#     cv=5, 
#     verbose=1, 
#     random_state=42, 
#     n_jobs=-1
# )
# 
# print("\nTraining Logistic Regression with SMOTE...")
# random_search.fit(X_train_res, y_train_res)
# 
# # 6. Evaluation
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
# 
# print(f"\nBest Parameters: {random_search.best_params_}")
# print(f"Accuracy with SMOTE: {accuracy_score(y_test, y_pred):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# df_model['on_time_flag'].value_counts()

# In[9]:


from sklearn.model_selection import train_test_split, GridSearchCV


# # 2. Impute & Scale
# imputer = SimpleImputer(strategy='median')
# X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
# X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
# 
# scaler = StandardScaler()
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
# 
# # 3. Define the Grid
# # We focus on depth and the number of trees
# param_grid = {
#     'n_estimators': [100, 200],      # Number of trees
#     'max_depth': [10, 20, None],     # How deep each tree goes
#     'min_samples_split': [2, 5],     # Minimum samples to split a node
#     'max_features': ['sqrt', 'log2'] # Features considered at each split
# }
# 
# # 4. Run GridSearchCV
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=3,           # 3-fold cross-validation
#     verbose=2, 
#     n_jobs=-1       # Uses all your CPU cores
# )
# 
# print("Starting exhaustive Grid Search...")
# grid_search.fit(X_train, y_train)
# 
# # 5. Best Results
# print(f"Best Parameters: {grid_search.best_params_}")
# best_rf = grid_search.best_estimator_
# predictions = best_rf.predict(X_test)
# print(f"Improved Accuracy: {accuracy_score(y_test, predictions):.4f}")

# In[10]:


from xgboost import XGBClassifier


# 
# 
# # 2. Update Feature Lists
# target = 'on_time_flag'
# categorical_cols = ['day_of_week', 'event_type', 'booking_type', 'load_type']
# numeric_cols = [
#     'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
#     'weight_lbs', 'pieces', 'revenue', 'years_experience', 
#     'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
#     'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density'
# ]
# 
# # 3. Preprocessing
# df_encoded = pd.get_dummies(df[categorical_cols + numeric_cols + [target]], 
#                             columns=categorical_cols, drop_first=True)
# 
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Impute and Scale
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# 
# X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
# X_test_scaled = scaler.transform(imputer.transform(X_test))
# 
# # 4. Train XGBoost (The Accuracy Booster)
# # Using 'scale_pos_weight' to handle any slight remaining imbalance
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )
# 
# print("Training XGBoost with engineered features...")
# xgb_model.fit(X_train_scaled, y_train)
# 
# # 5. Evaluate
# y_pred = xgb_model.predict(X_test_scaled)
# print(f"New Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nNew Classification Report:")
# print(classification_report(y_test, y_pred))

# # 4. Randomized Search for Logistic Regression
# param_dist = {
#     'C': np.logspace(-4, 4, 20),
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear'] # liblinear works well with l1 and l2
# }
# 
# lr_random = RandomizedSearchCV(
#     LogisticRegression(max_iter=1000), 
#     param_distributions=param_dist, 
#     n_iter=15, 
#     cv=5, 
#     random_state=42, 
#     n_jobs=-1
# )
# 
# print("Tuning Logistic Regression with Smart Features...")
# lr_random.fit(X_train_scaled, y_train)
# 
# # 5. Results
# best_lr = lr_random.best_estimator_
# y_pred = best_lr.predict(X_test_scaled)
# print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))

# # 2. Features and Preprocessing
# target = 'on_time_flag'
# numeric_features = [
#     'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
#     'weight_lbs', 'pieces', 'revenue', 'years_experience', 
#     'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
#     'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density'
# ]
# categorical_features = ['day_of_week', 'event_type', 'booking_type', 'load_type']
# 
# df_encoded = pd.get_dummies(df[categorical_features + numeric_features + [target]], 
#                             columns=categorical_features, drop_first=True)
# 
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Impute and Scale
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
# X_test_scaled = scaler.transform(imputer.transform(X_test))
# 
# # 3. EXHAUSTIVE GRID SEARCH
# # We narrow the 'C' range based on typical logistics data performance
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear'] # Required for l1 penalty
# }
# 
# grid_search = GridSearchCV(
#     LogisticRegression(max_iter=1000), 
#     param_grid=param_grid, 
#     cv=5, 
#     verbose=1, 
#     n_jobs=-1
# )
# 
# print("Running exhaustive Grid Search (20 total fits)...")
# grid_search.fit(X_train_scaled, y_train)
# 
# # 4. Final Best Model
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# 
# print(f"\nBest Parameters found: {grid_search.best_params_}")
# print(f"Final Grid Search Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))

# # 2. Features and Preprocessing
# target = 'on_time_flag'
# numeric_features = [
#     'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
#     'weight_lbs', 'pieces', 'revenue', 'years_experience', 
#     'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
#     'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density'
# ]
# categorical_features = ['day_of_week', 'event_type', 'booking_type', 'load_type']
# 
# df_encoded = pd.get_dummies(df[categorical_features + numeric_features + [target]], 
#                             columns=categorical_features, drop_first=True)
# 
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Impute and Scale
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
# X_test_scaled = scaler.transform(imputer.transform(X_test))
# 
# # 3. DEFINE THE GRID
# # We test different levels of complexity
# param_grid = {
#     'n_estimators': [100, 200],      # Total number of trees
#     'max_depth': [10, 20, None],     # How deep each tree grows
#     'min_samples_split': [2, 5],     # Minimum data points to create a branch
#     'max_features': ['sqrt', 'log2'] # Number of features per tree
# }
# 
# # 4. RUN THE SEARCH
# # Note: This will perform (2 * 3 * 2 * 2) * 5 folds = 120 total fits.
# grid_search_rf = GridSearchCV(
#     RandomForestClassifier(random_state=42), 
#     param_grid=param_grid, 
#     cv=5, 
#     verbose=2, 
#     n_jobs=-1
# )
# 
# print("Starting Random Forest Grid Search (120 fits)...")
# grid_search_rf.fit(X_train_scaled, y_train)
# 
# # 5. Final Best Model Results
# best_rf = grid_search_rf.best_estimator_
# y_pred = best_rf.predict(X_test_scaled)
# 
# print(f"\nBest Parameters found: {grid_search_rf.best_params_}")
# print(f"Final Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))

# # Extract Month (1 to 12)
# df['month'] = df['actual_datetime'].dt.month
# 
# # Create a 'Harsh Weather' proxy (Winter Months)
# # December (12), January (1), February (2)
# df['is_winter'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
# 
# # Create a 'Peak Season' proxy (Q4 Logistics Rush)
# df['is_peak_season'] = df['month'].apply(lambda x: 1 if x in [10, 11, 12] else 0)
# 
# # --- RE-APPLY YOUR SMART FEATURES ---
# df['detention_risk_index'] = df['facility_congestion_score'] * df['facility_detention_avg']
# df['efficiency_ratio'] = df['average_mpg'] / (df['truck_age_at_event'] + 1)
# df['revenue_density'] = df['revenue'] / (df['weight_lbs'] + 1)
# 
# # --- UPDATE THE FEATURE LIST ---
# target = 'on_time_flag'
# categorical_features = ['day_of_week', 'event_type', 'booking_type', 'load_type']
# numeric_features = [
#     'facility_congestion_score', 'facility_detention_avg', 'hour_of_day', 
#     'weight_lbs', 'pieces', 'revenue', 'years_experience', 
#     'truck_age_at_event', 'average_mpg', 'idle_time_hours', 'distance_deviation',
#     'is_home_terminal', 'detention_risk_index', 'efficiency_ratio', 'revenue_density',
#     'month', 'is_winter', 'is_peak_season' # Added these
# ]

# # Ensure we only work with the columns we need
# df= df[categorical_features + numeric_features + [target]].copy()
# 
# # 3. HEATMAP: Correlation of Numeric Features
# # This helps visualize which factors move together
# plt.figure(figsize=(12, 8))
# correlation_matrix = df_model[numeric_features + [target]].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap: Logistics Performance Drivers')
# plt.show() # Or plt.savefig('heatmap.png')
# 
# # 4. ONE-HOT ENCODING
# # Converts text categories into 0/1 columns
# df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
# 
# # 6. SPLIT DATA
# X = df_encoded.drop(columns=[target])
# y = df_encoded[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # 2. Initialize the Scaler
# scaler = StandardScaler()
# 
# # 3. FIT and TRANSFORM the training data
# # This calculates the mean/std of X_train and applies it
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_features])
# 
# # 4. ONLY TRANSFORM the test data
# # We use the 'rules' learned from the training data to scale the test data
# X_test[numeric_cols] = scaler.transform(X_test[numeric_features])
# 
# # 7. TRAIN RANDOM FOREST
# print("Training the model... please wait.")
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)
# 
# # 8. EVALUATE
# y_pred = model.predict(X_test)
# print(f"\nSuccess! Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nDetailed Performance Report:")
# print(classification_report(y_test, y_pred))
# 
# # 9. FEATURE IMPORTANCE
# # Shows which variables actually influenced the "On-Time" prediction
# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# importances.head(10).plot(kind='barh', color='teal')
# plt.title('Top 10 Predictors of On-Time Delivery')
# plt.gca().invert_yaxis()
# plt.show()

# 

# In[ ]:




