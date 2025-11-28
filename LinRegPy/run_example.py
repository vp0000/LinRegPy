import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from LinRegPy.base_model import *

# --- 1. Data Loading and Splitting ---
housing = fetch_california_housing()
X = housing.data 
y = housing.target
feature_names = housing.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
n, m = X_train.shape
# print(n**2 * m <= 1e6)

# --- 3. Model Training and Evaluation ---
method_3 = {'name': 'huber', 'mult': 1.0, 'lr': 1e-3, 'n_iter': 10000, 'use_gd': True}
method_4 = {'name': 'ols', 'mult': 0.1, 'lr': 0.01, 'n_iter': 5000, 'use_gd': False}

for method in [method_3, method_4]:
    # Pass the SCALED data to your model
    lin_model = LinearRegSuper(method, fit_intercept=True, w_init=None)
    lin_model.fit(X_train, y_train)

    # textual summary
    txt = lin_model.summary_text(X_test, y_test, feature_names=feature_names)
    print("---------------------------------------------")

    # save to file
    lin_model.save_model(f"Housing_Model_{method['name']}.pkl")




