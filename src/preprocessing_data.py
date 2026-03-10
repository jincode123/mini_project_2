#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[2]:


def split_data(df_encoded, target, test_size=0.2, random_state=42):
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# In[3]:


def scale_data(X_train, X_test, strategy="median"):
    imputer = SimpleImputer(strategy=strategy)
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, imputer, scaler


# In[4]:


def prepare_training_data(df_encoded, target, test_size=0.2, random_state=42, strategy="median"):
    X_train, X_test, y_train, y_test = split_data(
        df_encoded=df_encoded,
        target=target,
        test_size=test_size,
        random_state=random_state
    )

    X_train_scaled, X_test_scaled, imputer, scaler = scale_data(
        X_train=X_train,
        X_test=X_test,
        strategy=strategy
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler


# In[ ]:




