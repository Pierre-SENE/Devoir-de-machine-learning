# Application Streamlit : déploiement des 4 modèles du devoir
# Style repris de l'app du prof : ML 2/Deploy_My_Second_Model_With_Streamlit/app.py
# (chargement joblib + cache, Pred_func / Pred_func_csv, onglets, formulaire, upload + download)
# Différence avec l'app du prof : on applique aussi le scaler, car les modèles des notebooks
# ont été entraînés sur des données normalisées.
import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Devoir Machine Learning", page_icon="🤖", layout="centered")

# Dossier de l'application (pour retrouver les fichiers models/ en local comme sur Streamlit Cloud)
BASE = os.path.dirname(os.path.abspath(__file__))


# Charger les objets sauvegardés d'un modèle (une seule fois grâce au cache)
@st.cache_resource
def load_objects(dossier, nom_modele):
    chemin = os.path.join(BASE, 'models', dossier)
    # importer les encodeurs
    encoders = joblib.load(os.path.join(chemin, 'encoders.joblib'))
    # importer les valeurs uniques
    uniques = joblib.load(os.path.join(chemin, 'uniques.joblib'))
    # importer le scaler
    scaler = joblib.load(os.path.join(chemin, 'scaler.joblib'))
    # importer le meilleur modèle
    model = joblib.load(os.path.join(chemin, nom_modele))
    return encoders, uniques, scaler, model


# Fonction commune de prédiction multiple : boucle sur les lignes du CSV (comme Pred_func_csv du prof)
def Pred_func_csv(df, colonnes, pred_func, nom_colonne):
    predictions = []
    for row in df[colonnes].values:
        predictions.append(pred_func(*row))
    df[nom_colonne] = predictions
    return df


# Fonction commune pour l'onglet "Prédiction multiple"
def onglet_multiple(cle, colonnes, pred_func, nom_colonne, preparation=None):
    st.subheader("Prédiction à partir d'un fichier CSV")
    st.write("Le fichier doit contenir les colonnes : " + ", ".join(f"`{c}`" for c in colonnes))
    file = st.file_uploader('Upload a csv file', type=['csv'], key='file_' + cle)

    if file is not None:
        # Lire le fichier csv
        df = pd.read_csv(file)
        # Prétraitement éventuel (ex : séparation de l'adresse pour Car Data)
        if preparation is not None:
            df = preparation(df)
        st.write("Aperçu des données chargées :")
        st.dataframe(df.head())

        if st.button('Predict', key='btn_' + cle):
            try:
                df_pred = Pred_func_csv(df, colonnes, pred_func, nom_colonne)
            except KeyError as e:
                st.error(f"Colonne manquante dans le fichier : {e}")
                return
            except ValueError as e:
                st.error(f"Valeur inconnue dans le fichier (catégorie jamais vue à l'entraînement) : {e}")
                return
            st.write("Résultats des prédictions :")
            st.dataframe(df_pred)
            # Télécharger le fichier des prédictions
            st.download_button(label='Download a csv file',
                               data=df_pred.to_csv(index=False).encode('utf-8'),
                               file_name=f'predictions_{cle}.csv',
                               mime='text/csv',
                               key='dl_' + cle)


# =====================================================================
# 1. TÉLÉPHONE (classification de l'état) - notebook ML 1
# =====================================================================
def page_telephone():
    encoders, uniques, scaler, model = load_objects('telephone', 'rf_model.joblib')
    # uniques / encoders : [0] adresse, [1] marque, [2] etat (cible)

    def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
        # Encoder les variables catégorielles
        adresse = encoders[0].transform([adresse])[0]
        marque = encoders[1].transform([marque])[0]
        # vecteur des valeurs numériques (même ordre que dans le notebook)
        x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage], dtype=float).reshape(1, -1)
        # normaliser (MinMaxScaler)
        x_new = scaler.transform(x_new)
        # Prédire puis décoder la classe
        y_pred = model.predict(x_new)
        return encoders[2].inverse_transform([int(y_pred[0])])[0]

    st.title("📱 État d'un téléphone")
    st.caption("Classification : D'occasion / Neuf / Réconditionné / Venant — Random Forest")
    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])

    with tab1:
        with st.form('form_telephone'):
            col1, col2 = st.columns(2)
            with col1:
                prix = st.number_input('Prix (FCFA)', min_value=0, value=360000, step=5000)
                adresse = st.selectbox('Adresse', options=uniques[0])
                marque = st.selectbox('Marque', options=uniques[1])
            with col2:
                dim_ecr = st.number_input("Dimension de l'écran (pouces)", min_value=0.0, value=6.0, step=0.1)
                ram = st.number_input('RAM (Go)', min_value=0, value=8, step=1)
                stockage = st.number_input('Stockage (Go)', min_value=0, value=128, step=16)
            submit = st.form_submit_button('Predict')
        if submit:
            st.success(f"État prédit : {Pred_func(prix, adresse, marque, dim_ecr, ram, stockage)}")

    with tab2:
        onglet_multiple('telephone', ['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage'],
                        Pred_func, 'etat_prediction')


# =====================================================================
# 2. TAXI (régression du prix) - notebook ML 2
# =====================================================================
def page_taxi():
    encoders, uniques, scaler, model = load_objects('taxi', 'rf_model.joblib')
    # uniques / encoders : [0] Time_of_Day, [1] Day_of_Week, [2] Traffic_Conditions, [3] Weather

    def Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                  Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
        # Encoder les valeurs des Time_of_Day, Day_of_Week, Traffic_Conditions et Weather
        Time_of_Day = encoders[0].transform([Time_of_Day])[0]
        Day_of_Week = encoders[1].transform([Day_of_Week])[0]
        Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
        Weather = encoders[3].transform([Weather])[0]
        # vecteur des valeurs numériques
        x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                          Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
        x_new = x_new.reshape(1, -1)  # convertir en un 2D array
        # normaliser (RobustScaler)
        x_new = scaler.transform(x_new)
        # Prédire puis arrondir
        y_pred = model.predict(x_new)
        return round(float(y_pred[0]), 2)

    st.title("🚕 Prix d'un trajet de taxi")
    st.caption("Régression — Random Forest Regressor")
    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])

    with tab1:
        with st.form('form_taxi'):
            col1, col2 = st.columns(2)
            with col1:
                Trip_Distance_km = st.number_input('Trip Distance (km)', min_value=0.0, value=25.0, step=0.5)
                Time_of_Day = st.selectbox('Time of Day', options=uniques[0])
                Day_of_Week = st.selectbox('Day of Week', options=uniques[1])
                Passenger_Count = st.number_input('Passenger Count', min_value=1, max_value=4, value=2, step=1)
                Traffic_Conditions = st.selectbox('Traffic Conditions', options=uniques[2])
            with col2:
                Weather = st.selectbox('Weather', options=uniques[3])
                Base_Fare = st.number_input('Base Fare', min_value=0.0, value=3.5, step=0.1)
                Per_Km_Rate = st.number_input('Per Km Rate', min_value=0.0, value=1.2, step=0.05)
                Per_Minute_Rate = st.number_input('Per Minute Rate', min_value=0.0, value=0.3, step=0.01)
                Trip_Duration_Minutes = st.number_input('Trip Duration (minutes)', min_value=0.0, value=60.0, step=1.0)
            submit = st.form_submit_button('Predict')
        if submit:
            y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                               Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            st.success(f"Trip Price : {y_pred} $")

    with tab2:
        onglet_multiple('taxi', ['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
                                 'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate',
                                 'Per_Minute_Rate', 'Trip_Duration_Minutes'],
                        Pred_func, 'Trip_Price_Prediction')


# =====================================================================
# 3. CAR DATA (classification Venant / D'occasion) - notebook car_data_classification.ipynb
# =====================================================================
def preparation_car_data(df):
    # Même prétraitement que le notebook : si le CSV contient la colonne Adresse,
    # on la sépare (virgule), on garde le quartier et on supprime la région
    if 'Adresse' in df.columns:
        df['Quartier'] = df['Adresse'].str.split(',').str[0].str.strip()
        df = df.drop('Adresse', axis=1)
    return df


def page_car_data():
    encoders, uniques, scaler, model = load_objects('car_data', 'best_model.joblib')
    # uniques / encoders : [0] Marque, [1] Boite_Vitesse, [2] Quartier, [3] Venant_Occasion (cible)

    def Pred_func(Marque, Annee, Boite_Vitesse, Quartier, Prix):
        # Encoder les variables catégorielles
        Marque = encoders[0].transform([Marque])[0]
        Boite_Vitesse = encoders[1].transform([Boite_Vitesse])[0]
        Quartier = encoders[2].transform([str(Quartier).strip()])[0]
        # vecteur (même ordre que dans le notebook : Marque, Année, Boite_Vitesse, Quartier, Prix)
        x_new = np.array([Marque, Annee, Boite_Vitesse, Quartier, Prix], dtype=float).reshape(1, -1)
        # normaliser (MinMaxScaler)
        x_new = scaler.transform(x_new)
        # Prédire puis décoder la classe
        y_pred = model.predict(x_new)
        return encoders[3].inverse_transform([int(y_pred[0])])[0]

    st.title("🚗 Car Data : Venant ou D'occasion ?")
    st.caption("Classification — adresse séparée : on garde le quartier, la région est supprimée")
    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])

    with tab1:
        with st.form('form_car_data'):
            col1, col2 = st.columns(2)
            with col1:
                Marque = st.selectbox('Marque', options=uniques[0])
                Annee = st.number_input('Année', min_value=1980, max_value=2026, value=2015, step=1)
                Boite_Vitesse = st.selectbox('Boîte de vitesse', options=uniques[1])
            with col2:
                Quartier = st.selectbox('Quartier', options=uniques[2])
                Prix = st.number_input('Prix (FCFA)', min_value=0, value=7500000, step=100000)
            submit = st.form_submit_button('Predict')
        if submit:
            st.success(f"Prédiction : {Pred_func(Marque, Annee, Boite_Vitesse, Quartier, Prix)}")

    with tab2:
        st.info("Le CSV peut contenir soit la colonne `Adresse` (ex : `Yoff,Dakar`), soit directement `Quartier`.")
        onglet_multiple('car_data', ['Marque', 'Année', 'Boite_Vitesse', 'Quartier', 'Prix'],
                        Pred_func, 'Venant_Occasion_Prediction', preparation=preparation_car_data)


# =====================================================================
# 4. CAR DATASET (régression du Selling_Price) - notebook car_dataset_regression.ipynb
# =====================================================================
def page_car_dataset():
    encoders, uniques, scaler, model = load_objects('car_dataset', 'best_model.joblib')
    # uniques / encoders : [0] Fuel_Type, [1] Seller_Type, [2] Transmission

    def Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
        # Encoder les variables catégorielles
        Fuel_Type = encoders[0].transform([Fuel_Type])[0]
        Seller_Type = encoders[1].transform([Seller_Type])[0]
        Transmission = encoders[2].transform([Transmission])[0]
        # vecteur (même ordre que dans le notebook)
        x_new = np.array([Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age],
                         dtype=float).reshape(1, -1)
        # normaliser (RobustScaler)
        x_new = scaler.transform(x_new)
        # Prédire puis arrondir
        y_pred = model.predict(x_new)
        return round(float(y_pred[0]), 2)

    st.title("🚙 Car Dataset : prix de vente")
    st.caption("Régression — prédiction du Selling_Price")
    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])

    with tab1:
        with st.form('form_car_dataset'):
            col1, col2 = st.columns(2)
            with col1:
                Kms_Driven = st.number_input('Kms Driven', min_value=0, value=30000, step=1000)
                Present_Price = st.number_input('Present Price', min_value=0.0, value=7.0, step=0.1)
                Age = st.number_input('Age (années)', min_value=0, value=7, step=1)
            with col2:
                Fuel_Type = st.selectbox('Fuel Type', options=uniques[0])
                Seller_Type = st.selectbox('Seller Type', options=uniques[1])
                Transmission = st.selectbox('Transmission', options=uniques[2])
            submit = st.form_submit_button('Predict')
        if submit:
            y_pred = Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            st.success(f"Selling Price : {y_pred}")

    with tab2:
        onglet_multiple('car_dataset', ['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type',
                                        'Transmission', 'Age'],
                        Pred_func, 'Selling_Price_Prediction')


# =====================================================================
# Menu latéral
# =====================================================================
st.sidebar.title("Devoir Machine Learning")
choix = st.sidebar.radio("Choisir un modèle",
                         ['Téléphone', 'Taxi', 'Car Data (classification)', 'Car Dataset (régression)'])

if choix == 'Téléphone':
    page_telephone()
elif choix == 'Taxi':
    page_taxi()
elif choix == 'Car Data (classification)':
    page_car_data()
else:
    page_car_dataset()
