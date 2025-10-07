# ==========================================
# 📦 Import Required Libraries
# ==========================================
import os
import tarfile
import urllib.request as urlrq
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# ==========================================
# 🌐 Fetch and Load Housing Data
# ==========================================
DOWNLOAD_URL = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'
HOUSING_PATH = os.path.join('datasets', 'housing')


def fetch_housing_data():
    """Downloads and extracts the housing dataset."""
    if not os.path.isdir(HOUSING_PATH):
        os.makedirs(HOUSING_PATH, exist_ok=True)
    
    tgz_path = os.path.join(HOUSING_PATH, 'housing.tgz')
    urlrq.urlretrieve(DOWNLOAD_URL, tgz_path)
    
    with tarfile.open(tgz_path) as tgz_file:
        tgz_file.extractall(path=HOUSING_PATH, filter='data')


def load_housing_data():
    """Loads the housing CSV as a pandas DataFrame."""
    csv_path = os.path.join(HOUSING_PATH, 'housing.csv')
    return pd.read_csv(csv_path)


# Fetch and load the data
fetch_housing_data()
housing = load_housing_data()


# ==========================================
# 🧩 Create Stratified Train-Test Split
# ==========================================
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]

# Remove the 'income_cat' column (no longer needed)
strat_train_set = strat_train_set.drop("income_cat", axis=1)
strat_test_set = strat_test_set.drop("income_cat", axis=1)


# ==========================================
# 🏗️ Separate Features and Labels
# ==========================================
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# ==========================================
# ⚙️ Custom Transformer for Attribute Creation
# ==========================================
room_ix, bedrooms_ix, pop_ix, hhold_ix = 3, 4, 5, 6


class AttrCreation(BaseEstimator, TransformerMixin):
    """Adds combined attributes: rooms_per_household, pop_per_household, bedroom_per_room."""
    
    def __init__(self, add_bedroom_per_room=True):
        self.add_bedroom_per_room = add_bedroom_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_hhold = X[:, room_ix] / X[:, hhold_ix]
        pop_per_hhold = X[:, pop_ix] / X[:, hhold_ix]

        if self.add_bedroom_per_room:
            bedroom_per_room = X[:, bedrooms_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_hhold, pop_per_hhold, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_hhold, pop_per_hhold]


# ==========================================
# 🔧 Data Preprocessing Pipeline
# ==========================================
num_attributes = list(housing.drop("ocean_proximity", axis=1))
cat_attributes = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attr_adder", AttrCreation()),
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", OneHotEncoder(), cat_attributes)
])

# Prepare the data
housing_prepared = full_pipeline.fit_transform(housing)


# ==========================================
# 🌲 Random Forest Model with Grid Search
# ==========================================
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(
    estimator=forest_reg,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    return_train_score=True
)

grid_search.fit(housing_prepared, housing_labels)

final_model = grid_search.best_estimator_


# ==========================================
# 🧪 Evaluate on Test Set
# ==========================================
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(f"\n✅ Final RMSE on Housing Test Dataset: {final_rmse}")
