# Réentraînement du modèle TAXI vu en classe (régression du prix d'un trajet)
# Code copié de : ML 2/Copie de Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb
# On le relance car rf_model.joblib n'était pas dans le dossier du cours
# (l'app du prof utilisait un xgb_model qui n'existe dans aucun notebook).
# Lancer depuis la racine du projet : python notebooks/entrainement_classe/entrainement_taxi.py
import pandas as pd
import numpy as np
import joblib as jb
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

DOSSIER_MODELE = 'models/taxi/'

# Import les données
df = pd.read_csv('data/taxi_trip_pricing.csv')
print(df.shape)

# sélection les variables quantitatives et catégorielles
quant_data = df.select_dtypes(include='number')
cat_data = df.select_dtypes(include='object')

# Imputer les valeurs manquantes des variables catégorielles en utilisant le mode
for col in cat_data.columns:
    df.fillna({col: df[col].mode()[0]}, inplace=True)

# Enregistrer les valeurs uniques des variables catégorielles
uniques = []
for col in cat_data.columns:
    uniques.append(np.unique(df[col]).tolist())
jb.dump(uniques, DOSSIER_MODELE + 'uniques.joblib')

# Encoder les variables catégorielles (un encodeur par colonne)
encoders = []
for i in range(len(cat_data.columns)):
    encoder = LabelEncoder()
    encoder.fit(df[cat_data.columns[i]])
    df[cat_data.columns[i]] = encoder.transform(df[cat_data.columns[i]])
    encoders.append(encoder)
jb.dump(encoders, DOSSIER_MODELE + 'encoders.joblib')

# Imputer les valeurs manquantes des variables quantitatives
df = df.astype(float)
imputer = KNNImputer(n_neighbors=5)
df[:] = imputer.fit_transform(df)


# fonction d'imputation des valeurs aberrantes
def impute_func(df, col, n_neighbors=5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    df.loc[(df[col] < min_value) | (df[col] > max_value), col] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[:] = imputer.fit_transform(df)


for col in quant_data.columns:
    impute_func(df, col)

# Fractionner les données
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normaliser x
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
jb.dump(scaler, DOSSIER_MODELE + 'scaler.joblib')

# Splitter les données en train, val et test
x_train, x_vt, y_train, y_vt = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def evaluation(model):
    y_pred = model.predict(x_val)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    return r2, mse, mae, rmse


# Multiple Linear Regression
grid_lr = GridSearchCV(LinearRegression(), {'fit_intercept': [True, False], 'positive': [True, False]}, cv=cv)
grid_lr.fit(x_train, y_train)
print("LR -> R2, MSE, MAE, RMSE :", evaluation(grid_lr.best_estimator_))

# Ridge
grid_rr = GridSearchCV(Ridge(), {'fit_intercept': [True, False], 'positive': [True, False],
                                 'alpha': np.logspace(-8, 8, 100)}, cv=cv)
grid_rr.fit(x_train, y_train)
print("RR -> R2, MSE, MAE, RMSE :", evaluation(grid_rr.best_estimator_))

# Decision Tree
grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42),
                       {'max_depth': np.arange(1, 30), 'criterion': ['squared_error', 'absolute_error']}, cv=cv)
grid_dt.fit(x_train, y_train)
print("DT -> R2, MSE, MAE, RMSE :", evaluation(grid_dt.best_estimator_))

# Random Forest
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                       {'n_estimators': np.arange(10, 100, 10), 'max_depth': np.arange(1, 15)}, cv=cv)
grid_rf.fit(x_train, y_train)
rf_model = grid_rf.best_estimator_
print("RF -> R2, MSE, MAE, RMSE :", evaluation(rf_model))

# télécharger le modèle le plus performant
jb.dump(rf_model, DOSSIER_MODELE + 'rf_model.joblib')
print("Modèle sauvegardé dans", DOSSIER_MODELE)
