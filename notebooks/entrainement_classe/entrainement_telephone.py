# Réentraînement du modèle TÉLÉPHONE vu en classe (classification de l'état du téléphone)
# Code copié de : ML 1/Copie de Clas_Log_Reg_Dec_Tree_Random_Forest.ipynb
# Ajout : la sauvegarde (uniques, encoders, scaler, modèle) copiée du notebook taxi
#         (ML 2/Copie de Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb), car le notebook
#         téléphone ne sauvegardait rien et il faut ces fichiers pour le déploiement.
# Lancer depuis la racine du projet : python notebooks/entrainement_classe/entrainement_telephone.py
import pandas as pd
import numpy as np
import joblib as jb
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

DOSSIER_MODELE = 'models/telephone/'

# importer les données
df = pd.read_csv('data/telephone_data.csv')
print(df.shape)

# selection des variables catégorielles et quantitatives
cat_data = df.select_dtypes(exclude='number')
quant_data = df.select_dtypes(include='number')

# Enregistrer les valeurs uniques des variables catégorielles (copié du notebook taxi)
uniques = []
for col in cat_data.columns:
    uniques.append(np.unique(df[col]).tolist())
jb.dump(uniques, DOSSIER_MODELE + 'uniques.joblib')

# Encoder les variables catégorielles : un encodeur par colonne (copié du notebook taxi)
encoders = []
for i in range(len(cat_data.columns)):
    encoder = LabelEncoder()
    encoder.fit(df[cat_data.columns[i]])
    df[cat_data.columns[i]] = encoder.transform(df[cat_data.columns[i]])
    encoders.append(encoder)
jb.dump(encoders, DOSSIER_MODELE + 'encoders.joblib')


# fonction d'imputation des valeurs aberrantes
def impute_func(df, col, n_neighbors=5):
    # Détection des outliers (Méthode IQR)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    # Les valeurs hors bornes deviennent des valeurs manquantes pour le KNNImputer
    df.loc[(df[col] < min_value) | (df[col] > max_value), col] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[:] = imputer.fit_transform(df)


# imputation des valeurs extrêmes
# (conversion en float : pandas 3 refuse de mettre NaN dans une colonne d'entiers)
df = df.astype(float)
for col in quant_data.columns:
    impute_func(df, col)

# Fractionner les données en variables prédicteurs (x) et variable cible (y)
x = df.drop('etat', axis=1).values
y = df.etat.values.astype(int)

# normaliser les données
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
jb.dump(scaler, DOSSIER_MODELE + 'scaler.joblib')

# splitter les données en train (80%), val (10%) et test (10%)
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, train_size=0.8, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.5, random_state=42)

# Validation croisée et figé
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# fonction d'évaluation
def evaluation(model):
    y_pred = model.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='micro')
    return acc, f1


# Logistic Regression
grid = GridSearchCV(LogisticRegression(), {"C": np.logspace(-3, 3, 10), "penalty": ['l1', 'l2']}, cv=cv)
grid.fit(x_train, y_train)
lg_model = LogisticRegression(**grid.best_params_)
lg_model.fit(x_train, y_train)
acc_lg, f1_lg = evaluation(lg_model)
print(f"LG -> Accuracy : {acc_lg} --- F1 score : {f1_lg}")

# Decision Tree
grid__ = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)}, cv=cv)
grid__.fit(x_train, y_train)
dt_model = DecisionTreeClassifier(random_state=42, **grid__.best_params_)
dt_model.fit(x_train, y_train)
acc_dt, f1_dt = evaluation(dt_model)
print(f"DT -> Accuracy : {acc_dt} --- F1 score : {f1_dt}")

# Random Forest
params = {'n_estimators': list(np.arange(10, 100, 10)), 'max_depth': list(np.arange(3, 15)),
          'criterion': ['gini', 'entropy']}
grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=params, cv=cv)
grid.fit(x_train, y_train)
rf_model = RandomForestClassifier(random_state=42, **grid.best_params_)
rf_model.fit(x_train, y_train)
acc_rf, f1_rf = evaluation(rf_model)
print(f"RF -> Accuracy : {acc_rf} --- F1 score : {f1_rf}")

# sauvegarder le modèle le plus performant (le Random Forest, comme en classe)
jb.dump(rf_model, DOSSIER_MODELE + 'rf_model.joblib')
print("Modèle sauvegardé dans", DOSSIER_MODELE)
