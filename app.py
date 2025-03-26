import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import pickle

df = pd.read_csv("Housing.csv")

Q1_price = df['price'].quantile(0.25)
Q3_price = df['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price
df = df[(df['price'] >= lower_bound_price) & (df['price'] <= upper_bound_price)]

for col in ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
    upper_bound = df[col].quantile(0.95)
    df = df[df[col] <= upper_bound]

df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)  
df['area_per_bedroom'] = df['area_per_bedroom'].clip(upper=df['area_per_bedroom'].quantile(0.95)) 
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

X = df.drop(columns=['price'])
y = df['price']

y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'area_per_bedroom', 'total_rooms']

ordinal_categories = [
    ['no', 'yes'],             # mainroad
    ['no', 'yes'],             # guestroom
    ['no', 'yes'],             # basement
    ['no', 'yes'],             # hotwaterheating
    ['no', 'yes'],             # airconditioning
    ['no', 'yes'],             # prefarea
    ['unfurnished', 'semi-furnished', 'furnished']
]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OrdinalEncoder(categories=ordinal_categories), categorical_cols)
])

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, verbosity=0))
])

ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

rf_params = {
    'regressor__n_estimators': randint(50, 150),
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': randint(5, 20),
    'regressor__min_samples_leaf': randint(5, 15)
}
rf_search = RandomizedSearchCV(rf_pipeline, rf_params, n_iter=15, cv=5, scoring='r2', random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train_log)

xgb_params = {
    'regressor__n_estimators': randint(50, 100),
    'regressor__learning_rate': uniform(0.05, 0.15),
    'regressor__max_depth': randint(2, 4),
    'regressor__subsample': uniform(0.8, 0.2),
    'regressor__colsample_bytree': uniform(0.8, 0.2),
    'regressor__reg_lambda': uniform(0.5, 1.5),
    'regressor__reg_alpha': uniform(0.5, 1.5)
}
xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_params, n_iter=10, cv=5, scoring='r2', random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train_log)

ridge_params = {'regressor__alpha': uniform(1, 20)}
ridge_search = RandomizedSearchCV(ridge_pipeline, ridge_params, n_iter=10, cv=5, scoring='r2', random_state=42, n_jobs=-1)
ridge_search.fit(X_train, y_train_log)

lr_pipeline.fit(X_train, y_train_log)

y_test_actual = np.expm1(y_test_log)
lr_pred = np.expm1(lr_pipeline.predict(X_test))
rf_pred = np.expm1(rf_search.predict(X_test))
xgb_pred = np.expm1(xgb_search.predict(X_test))
ridge_pred = np.expm1(ridge_search.predict(X_test))

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{name} Evaluation Metrics:")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : ₹{mae:,.2f}")
    print(f"MSE      : ₹{mse:,.2f}")
    print(f"RMSE     : ₹{rmse:,.2f}")
    return r2

models = [
    ("Linear Regression", lr_pipeline, lr_pred),
    ("Random Forest (Tuned)", rf_search.best_estimator_, rf_pred),
    ("XGBoost (Tuned)", xgb_search.best_estimator_, xgb_pred),
    ("Ridge (Tuned)", ridge_search.best_estimator_, ridge_pred)
]

best_model = None
best_r2 = -float('inf')
best_model_name = ""

for name, model, pred in models:
    r2 = print_metrics(name, y_test_actual, pred)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print(f"\nBest model ({best_model_name}) with R² = {best_r2:.4f} saved as 'best_model.pkl'")

print("\nCross-Validation R² Scores:")
for name, pipeline in [("Linear Regression", lr_pipeline), 
                       ("Random Forest", rf_search.best_estimator_), 
                       ("XGBoost", xgb_search.best_estimator_), 
                       ("Ridge", ridge_search.best_estimator_)]:
    scores = cross_val_score(pipeline, X, y_log, cv=5, scoring='r2')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

plt.scatter(y_test_actual, xgb_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (XGBoost)")
plt.title("Actual vs Predicted Prices (XGBoost)")
plt.show()