# Application Streamlit : déploiement des 4 modèles du devoir
# Logique reprise de l'app du prof : ML 2/Deploy_My_Second_Model_With_Streamlit/app.py
# (chargement joblib + cache, Pred_func / Pred_func_csv, onglets simple / multiple, upload + download)
# Différence avec l'app du prof : on applique aussi le scaler, car les modèles des notebooks
# ont été entraînés sur des données normalisées.
# Design : style "glass" (cartes en verre dépoli, fond beige flouté, accent orange).
import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Devoir Machine Learning", page_icon="🤖", layout="wide",
                   initial_sidebar_state="collapsed")

# Dossier de l'application (pour retrouver les fichiers models/ en local comme sur Streamlit Cloud)
BASE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# DESIGN : feuille de style (glassmorphism)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --ink: #2A231E;          /* texte principal */
  --ink-2: #6B5E54;        /* texte secondaire */
  --ink-3: #9A8B7F;        /* texte discret */
  --accent: #E0702F;       /* orange (seule couleur d'accent) */
  --glass: rgba(255, 255, 255, 0.42);
  --glass-strong: rgba(255, 255, 255, 0.62);
  --glass-border: rgba(255, 255, 255, 0.65);
  --shadow: 0 12px 40px rgba(70, 45, 25, 0.12);
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Fond : pièce chaleureuse floutée (bois, beige, lumière orangée) */
.stApp {
  background:
    radial-gradient(ellipse 60% 45% at 12% 18%, rgba(160, 110, 70, 0.55), transparent 70%),
    radial-gradient(ellipse 50% 40% at 88% 12%, rgba(120, 80, 50, 0.45), transparent 70%),
    radial-gradient(ellipse 55% 50% at 80% 85%, rgba(224, 150, 90, 0.40), transparent 70%),
    radial-gradient(ellipse 60% 50% at 15% 90%, rgba(200, 170, 140, 0.55), transparent 70%),
    linear-gradient(160deg, #E9DED2 0%, #D9C7B4 45%, #CDB59D 100%);
  background-attachment: fixed;
  color: var(--ink);
}

/* On cache la barre latérale et l'en-tête Streamlit */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }

/* ---------- Carte en verre ---------- */
.glass {
  background: var(--glass);
  backdrop-filter: blur(22px) saturate(140%);
  -webkit-backdrop-filter: blur(22px) saturate(140%);
  border: 1px solid var(--glass-border);
  border-radius: 24px;
  box-shadow: var(--shadow);
  transition: transform .25s ease, box-shadow .25s ease;
}
.glass:hover { transform: translateY(-3px); box-shadow: 0 18px 50px rgba(70, 45, 25, 0.16); }

/* ---------- Barre du haut ---------- */
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 8px 10px 8px 8px; border-radius: 999px; margin-bottom: 22px;
}
.topbar:hover { transform: none; }
.who { display: flex; align-items: center; gap: 10px; font-size: 14px; color: var(--ink-2); }
.who b { color: var(--ink); font-weight: 600; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center;
  background: var(--ink); color: #fff; font-weight: 600; font-size: 13px;
}
.chip {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  background: var(--glass-strong); font-size: 13px; color: var(--ink-2);
}
.chip, .who { white-space: nowrap; }
@media (max-width: 640px) { .topbar .chip:last-child { display: none; } .topbar .chip { font-size: 12px; } }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
.dot.live { box-shadow: 0 0 0 4px rgba(224, 112, 47, 0.18); }

/* ---------- Titre ---------- */
.hello { font-size: 15px; color: var(--ink-2); margin: 4px 0 2px 4px; }
.title { font-size: 34px; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); margin: 0 0 4px 4px; }
.subtitle { font-size: 15px; color: var(--ink-2); margin: 0 0 20px 4px; }

/* ---------- Grille des indicateurs ---------- */
.grid { display: grid; grid-template-columns: 1.6fr 1fr 1fr; gap: 18px; margin-bottom: 22px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.card { padding: 22px 24px; }
.label { font-size: 13px; color: var(--ink-2); font-weight: 500; }
.big { font-size: 36px; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); margin: 6px 0 4px; }
.pill-up {
  display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(224, 112, 47, 0.14); color: #A4471A;
}
.muted { font-size: 12px; color: var(--ink-3); }
.progress { height: 6px; border-radius: 99px; background: rgba(255, 255, 255, 0.6); margin-top: 16px; overflow: hidden; }
.progress > span { display: block; height: 100%; border-radius: 99px; background: var(--accent); }

/* Barres de comparaison des algorithmes (une seule série : la meilleure en orange) */
.score-row { display: flex; gap: 22px; align-items: stretch; }
.score-left { flex: 1; min-width: 0; }
.bars { flex: 1.1; display: flex; align-items: flex-end; justify-content: space-around; gap: 10px; height: 150px; }
.bar { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; flex: 1; cursor: default; }
.bar .col {
  width: 26px; border-radius: 4px 4px 0 0; background: rgba(255, 255, 255, 0.85);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.9); transition: filter .2s;
}
.bar.best .col { background: var(--accent); box-shadow: none; }
.bar:hover .col { filter: brightness(0.95); }
.bar .v { font-size: 11px; color: var(--ink-2); margin-bottom: 4px; font-weight: 600; }
.bar .n { font-size: 11px; color: var(--ink-3); margin-top: 6px; }

/* Jauge circulaire */
.ring-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.ring {
  width: 128px; height: 128px; border-radius: 50%; display: grid; place-items: center; margin: 6px 0 12px;
  background: conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(255, 255, 255, 0.65) 0);
}
.ring > div {
  width: 104px; height: 104px; border-radius: 50%; background: rgba(250, 245, 240, 0.92);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring b { font-size: 24px; color: var(--ink); letter-spacing: -0.02em; }
.ring small { font-size: 11px; color: var(--ink-3); }

/* Fiche dataset */
.facts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.fact { background: var(--glass-strong); border-radius: 14px; padding: 10px 12px; }
.fact b { display: block; font-size: 18px; color: var(--ink); }
.fact span { font-size: 11px; color: var(--ink-3); }
.steps { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.step { font-size: 11px; padding: 4px 9px; border-radius: 999px; background: rgba(42, 35, 30, 0.07); color: var(--ink-2); }

/* ---------- Navigation en pilule (segmented control) ---------- */
[data-testid="stButtonGroup"] {
  background: var(--glass); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
  border: 1px solid var(--glass-border); border-radius: 999px; padding: 5px; box-shadow: var(--shadow);
  width: fit-content; max-width: 100%;
}
[data-testid="stButtonGroup"] button {
  border: none !important; border-radius: 999px !important; background: transparent !important;
  color: var(--ink-2) !important; font-weight: 500; padding: 6px 16px !important;
}
[data-testid="stButtonGroup"] button p { color: inherit !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: #fff !important; color: var(--ink) !important; box-shadow: 0 2px 10px rgba(70, 45, 25, 0.12);
}

/* ---------- Zone de prédiction (conteneur en verre) ---------- */
.st-key-glass_zone {
  background: var(--glass); backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
  border: 1px solid var(--glass-border) !important; border-radius: 24px; box-shadow: var(--shadow);
  padding: 18px 22px 22px;
}

/* Onglets en pilule */
.stTabs [role="tablist"] {
  gap: 4px; background: rgba(255, 255, 255, 0.45); border-radius: 999px; padding: 4px; width: fit-content;
  border: none !important; box-shadow: none !important;
}
.stTabs [data-testid="stTab"] {
  border-radius: 999px; padding: 6px 18px; height: auto; color: var(--ink-2); background: transparent; border: none;
}
.stTabs [data-testid="stTab"] p { color: inherit; font-weight: 500; }
.stTabs [data-testid="stTab"][aria-selected="true"] {
  background: #fff; color: var(--ink); box-shadow: 0 2px 10px rgba(70, 45, 25, 0.10);
}
.stTabs .react-aria-SelectionIndicator { display: none !important; }

/* Formulaire */
[data-testid="stForm"] { border: none; padding: 8px 0 0; background: transparent; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] {
  background: rgba(255, 255, 255, 0.62) !important; border-radius: 14px !important;
  border-color: rgba(255, 255, 255, 0.8) !important;
}
[data-testid="stNumberInputContainer"] input, [data-baseweb="input"] input { background: transparent !important; }
[data-testid="stWidgetLabel"] p { font-size: 13px; color: var(--ink-2); font-weight: 500; }

/* Boutons : pilule sombre, orange au survol */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
  background: var(--ink) !important; color: #fff !important; border: none !important;
  border-radius: 999px !important; padding: 8px 22px !important; font-weight: 600;
  transition: background .2s, transform .2s;
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  background: var(--accent) !important; transform: translateY(-1px);
}
.stButton button p, [data-testid="stFormSubmitButton"] button p, [data-testid="stDownloadButton"] button p { color: #fff !important; }

/* Upload et tableau */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.5); border: 1.5px dashed rgba(42, 35, 30, 0.18); border-radius: 18px;
}
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }

/* Carte de résultat */
.result { display: flex; align-items: center; gap: 22px; padding: 20px 24px; margin-top: 16px; }
.result:hover { transform: none; }
.result .ring { width: 96px; height: 96px; margin: 0; }
.result .ring > div { width: 78px; height: 78px; }
.result .ring b { font-size: 18px; }
.result .value { font-size: 32px; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); }
.hint { font-size: 13px; color: var(--ink-2); margin: 2px 0 10px; }
.foot { text-align: center; font-size: 12px; color: var(--ink-3); margin-top: 26px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Fonctions communes
# =====================================================================

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


def fr(x, d=1):
    # Nombre au format français (virgule décimale, espace pour les milliers)
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def ring_html(pourcent, texte, sous_texte):
    # Jauge circulaire (pourcent entre 0 et 100)
    return (f'<div class="ring" style="--p:{pourcent:.1f}"><div><b>{texte}</b>'
            f'<small>{sous_texte}</small></div></div>')


# Fonction commune de prédiction multiple : boucle sur les lignes du CSV (comme Pred_func_csv du prof)
def Pred_func_csv(df, colonnes, pred_func, nom_colonne):
    predictions = []
    for row in df[colonnes].values:
        predictions.append(pred_func(*row))
    df[nom_colonne] = predictions
    return df


# En-tête du modèle : titre + 3 cartes d'indicateurs
def entete(infos):
    scores = infos['scores']
    best = max(scores, key=scores.get)
    best_score = scores[best]
    autres = [s for n, s in scores.items() if n != best]
    gain = best_score - max(autres)
    is_clf = infos['type'] == 'Classification'
    metrique = 'Accuracy' if is_clf else 'R²'
    valeur = f"{fr(best_score * 100)} %" if is_clf else fr(best_score, 3)
    gain_txt = f"+{fr(gain * 100)} pts" if is_clf else f"+{fr(gain, 3)}"

    # barres : hauteur proportionnelle au score (axe qui part de 0)
    bars = ""
    for nom, s in scores.items():
        cls = "bar best" if nom == best else "bar"
        v = fr(s * 100, 1) if is_clf else fr(s, 2)
        bars += (f'<div class="{cls}" title="{infos["noms"][nom]} : {metrique} = {v}">'
                 f'<div class="v">{v}</div><div class="col" style="height:{s * 100:.1f}%"></div>'
                 f'<div class="n">{nom}</div></div>')

    steps = "".join(f'<span class="step">{s}</span>' for s in infos['pipeline'])

    st.markdown(f"""
<div class="title">{infos['emoji']} {infos['titre']}</div>
<div class="subtitle">{infos['description']}</div>
<div class="grid">
  <div class="glass card">
    <div class="score-row">
      <div class="score-left">
        <div class="label">{metrique} du meilleur modèle (validation)</div>
        <div class="big">{valeur}</div>
        <span class="pill-up">{gain_txt}</span> <span class="muted">vs 2ᵉ meilleur algorithme</span>
        <div class="progress"><span style="width:{best_score * 100:.1f}%"></span></div>
        <div class="muted" style="margin-top:8px">{infos['test']}</div>
      </div>
      <div class="bars">{bars}</div>
    </div>
  </div>
  <div class="glass card ring-wrap">
    <div class="label">Modèle retenu</div>
    {ring_html(best_score * 100, valeur, metrique)}
    <div style="font-weight:600">{infos['noms'][best]}</div>
    <div class="muted">GridSearchCV · KFold 5</div>
  </div>
  <div class="glass card">
    <div class="label">Jeu de données</div>
    <div class="facts">
      <div class="fact"><b>{fr(infos['lignes'], 0)}</b><span>lignes</span></div>
      <div class="fact"><b>{infos['variables']}</b><span>variables d'entrée</span></div>
    </div>
    <div class="fact" style="margin-top:10px"><b style="font-size:14px">{infos['cible']}</b><span>variable cible · {infos['type']}</span></div>
    <div class="steps">{steps}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# Carte de résultat après une prédiction simple
def resultat_classification(label, proba):
    st.markdown(f"""
<div class="glass result">
  {ring_html(proba * 100, f"{fr(proba * 100, 0)} %", "confiance")}
  <div><div class="label">Prédiction</div><div class="value">{label}</div>
  <div class="muted">Probabilité donnée par le modèle pour cette classe</div></div>
</div>""", unsafe_allow_html=True)


def resultat_regression(valeur, unite, texte):
    st.markdown(f"""
<div class="glass result">
  <div><div class="label">{texte}</div><div class="value">{fr(valeur, 2)} {unite}</div>
  <div class="muted">Valeur estimée par le modèle</div></div>
</div>""", unsafe_allow_html=True)


# Onglet "Prédiction multiple"
def onglet_multiple(cle, colonnes, pred_func, nom_colonne, preparation=None, exemple=None):
    st.markdown(f'<div class="hint">Le fichier CSV doit contenir les colonnes : '
                f'{", ".join(colonnes)}. Exemple prêt à l\'emploi : <b>data/exemples/{exemple}</b></div>',
                unsafe_allow_html=True)
    file = st.file_uploader('Téléverser un fichier CSV', type=['csv'], key='file_' + cle)

    if file is not None:
        # Lire le fichier csv
        df = pd.read_csv(file)
        # Prétraitement éventuel (ex : séparation de l'adresse pour Car Data)
        if preparation is not None:
            df = preparation(df)
        st.caption(f"{len(df)} lignes chargées — aperçu :")
        st.dataframe(df.head(), width='stretch', hide_index=True)

        if st.button('Lancer les prédictions', key='btn_' + cle):
            try:
                df_pred = Pred_func_csv(df, colonnes, pred_func, nom_colonne)
            except KeyError as e:
                st.error(f"Colonne manquante dans le fichier : {e}")
                return
            except ValueError as e:
                st.error(f"Valeur inconnue dans le fichier (catégorie jamais vue à l'entraînement) : {e}")
                return
            st.dataframe(df_pred, width='stretch', hide_index=True)
            # Télécharger le fichier des prédictions
            st.download_button(label='Télécharger le CSV des prédictions',
                               data=df_pred.to_csv(index=False).encode('utf-8'),
                               file_name=f'predictions_{cle}.csv',
                               mime='text/csv',
                               key='dl_' + cle)


# =====================================================================
# Résultats des notebooks (métriques sur les données de validation)
# Sources : notebooks/entrainement_classe/*.py, car_data_classification.ipynb, car_dataset_regression.ipynb
# =====================================================================
INFOS = {
    'Téléphone': dict(
        emoji='📱', titre="État d'un téléphone", type='Classification', cible='etat',
        description="Prédire si un téléphone est d'occasion, neuf, reconditionné ou venant.",
        scores={'LG': 0.680, 'DT': 0.8215, 'RF': 0.8708},
        noms={'LG': 'Logistic Regression', 'DT': 'Decision Tree', 'RF': 'Random Forest'},
        lignes=3253, variables=6, test="Modèle vu en classe (ML 1), réentraîné pour le déploiement",
        pipeline=['LabelEncoder', 'IQR + KNNImputer', 'MinMaxScaler', 'GridSearchCV']),
    'Taxi': dict(
        emoji='🚕', titre="Prix d'un trajet de taxi", type='Régression', cible='Trip_Price',
        description="Estimer le prix d'une course à partir de la distance, la durée, la météo et les tarifs.",
        scores={'LR': 0.8645, 'RR': 0.8643, 'DT': 0.7922, 'RF': 0.9052},
        noms={'LR': 'Linear Regression', 'RR': 'Ridge', 'DT': 'Decision Tree', 'RF': 'Random Forest'},
        lignes=1000, variables=10, test="Modèle vu en classe (ML 2), réentraîné pour le déploiement",
        pipeline=['Mode + KNNImputer', 'LabelEncoder', 'IQR + KNNImputer', 'RobustScaler', 'GridSearchCV']),
    'Car Data': dict(
        emoji='🚗', titre="Car Data : venant ou d'occasion ?", type='Classification', cible='Venant_Occasion',
        description="Adresse séparée sur la virgule : on garde le quartier, la région (Dakar) est supprimée.",
        scores={'LG': 0.7913, 'DT': 0.8094, 'RF': 0.8040},
        noms={'LG': 'Logistic Regression', 'DT': 'Decision Tree', 'RF': 'Random Forest'},
        lignes=5507, variables=5, test="Accuracy sur les données test : 78,2 %",
        pipeline=['Split adresse', 'LabelEncoder', 'IQR + KNNImputer', 'MinMaxScaler', 'GridSearchCV']),
    'Car Dataset': dict(
        emoji='🚙', titre="Car Dataset : prix de vente", type='Régression', cible='Selling_Price',
        description="Estimer le prix de vente d'une voiture d'occasion.",
        scores={'LR': 0.6498, 'RR': 0.6498, 'DT': 0.8338, 'RF': 0.8788},
        noms={'LR': 'Linear Regression', 'RR': 'Ridge', 'DT': 'Decision Tree', 'RF': 'Random Forest'},
        lignes=5861, variables=6, test="R² sur les données test : 0,905",
        pipeline=['LabelEncoder', 'KNNImputer', 'IQR + KNNImputer', 'RobustScaler', 'GridSearchCV']),
}


# =====================================================================
# 1. TÉLÉPHONE (classification de l'état) - notebook ML 1
# =====================================================================
def page_telephone():
    encoders, uniques, scaler, model = load_objects('telephone', 'rf_model.joblib')
    # uniques / encoders : [0] adresse, [1] marque, [2] etat (cible)

    def preparer(prix, adresse, marque, dim_ecr, ram, stockage):
        # Encoder les variables catégorielles
        adresse = encoders[0].transform([adresse])[0]
        marque = encoders[1].transform([marque])[0]
        # vecteur des valeurs numériques (même ordre que dans le notebook)
        x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage], dtype=float).reshape(1, -1)
        # normaliser (MinMaxScaler)
        return scaler.transform(x_new)

    def Pred_func(*valeurs):
        # Prédire puis décoder la classe
        y_pred = model.predict(preparer(*valeurs))
        return encoders[2].inverse_transform([int(y_pred[0])])[0]

    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])
    with tab1:
        with st.form('form_telephone'):
            c1, c2, c3 = st.columns(3)
            prix = c1.number_input('Prix (FCFA)', min_value=0, value=360000, step=5000)
            adresse = c2.selectbox('Adresse', options=uniques[0])
            marque = c3.selectbox('Marque', options=uniques[1])
            dim_ecr = c1.number_input("Taille de l'écran (pouces)", min_value=0.0, value=6.0, step=0.1)
            ram = c2.number_input('RAM (Go)', min_value=0, value=8, step=1)
            stockage = c3.number_input('Stockage (Go)', min_value=0, value=128, step=16)
            submit = st.form_submit_button('Prédire')
        if submit:
            valeurs = (prix, adresse, marque, dim_ecr, ram, stockage)
            proba = model.predict_proba(preparer(*valeurs)).max()
            resultat_classification(Pred_func(*valeurs), proba)
    with tab2:
        onglet_multiple('telephone', ['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage'],
                        Pred_func, 'etat_prediction', exemple='exemple_telephone.csv')


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

    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])
    with tab1:
        with st.form('form_taxi'):
            c1, c2, c3 = st.columns(3)
            Trip_Distance_km = c1.number_input('Distance (km)', min_value=0.0, value=25.0, step=0.5)
            Trip_Duration_Minutes = c2.number_input('Durée (minutes)', min_value=0.0, value=60.0, step=1.0)
            Passenger_Count = c3.number_input('Passagers', min_value=1, max_value=4, value=2, step=1)
            Time_of_Day = c1.selectbox('Moment de la journée', options=uniques[0])
            Day_of_Week = c2.selectbox('Jour', options=uniques[1])
            Traffic_Conditions = c3.selectbox('Trafic', options=uniques[2])
            Weather = c1.selectbox('Météo', options=uniques[3])
            Base_Fare = c2.number_input('Prise en charge', min_value=0.0, value=3.5, step=0.1)
            Per_Km_Rate = c3.number_input('Tarif / km', min_value=0.0, value=1.2, step=0.05)
            Per_Minute_Rate = c1.number_input('Tarif / minute', min_value=0.0, value=0.3, step=0.01)
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                               Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            resultat_regression(y_pred, '$', 'Prix estimé du trajet')
    with tab2:
        onglet_multiple('taxi', ['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
                                 'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate',
                                 'Per_Minute_Rate', 'Trip_Duration_Minutes'],
                        Pred_func, 'Trip_Price_Prediction', exemple='exemple_taxi.csv')


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

    def preparer(Marque, Annee, Boite_Vitesse, Quartier, Prix):
        # Encoder les variables catégorielles
        Marque = encoders[0].transform([Marque])[0]
        Boite_Vitesse = encoders[1].transform([Boite_Vitesse])[0]
        Quartier = encoders[2].transform([str(Quartier).strip()])[0]
        # vecteur (même ordre que dans le notebook : Marque, Année, Boite_Vitesse, Quartier, Prix)
        x_new = np.array([Marque, Annee, Boite_Vitesse, Quartier, Prix], dtype=float).reshape(1, -1)
        # normaliser (MinMaxScaler)
        return scaler.transform(x_new)

    def Pred_func(*valeurs):
        # Prédire puis décoder la classe
        y_pred = model.predict(preparer(*valeurs))
        return encoders[3].inverse_transform([int(y_pred[0])])[0]

    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])
    with tab1:
        with st.form('form_car_data'):
            c1, c2, c3 = st.columns(3)
            Marque = c1.selectbox('Marque', options=uniques[0])
            Annee = c2.number_input('Année', min_value=1980, max_value=2026, value=2015, step=1)
            Boite_Vitesse = c3.selectbox('Boîte de vitesse', options=uniques[1])
            Quartier = c1.selectbox('Quartier', options=uniques[2])
            Prix = c2.number_input('Prix (FCFA)', min_value=0, value=7500000, step=100000)
            submit = st.form_submit_button('Prédire')
        if submit:
            valeurs = (Marque, Annee, Boite_Vitesse, Quartier, Prix)
            proba = model.predict_proba(preparer(*valeurs)).max()
            resultat_classification(Pred_func(*valeurs), proba)
    with tab2:
        st.markdown('<div class="hint">Le CSV peut contenir la colonne <b>Adresse</b> (ex : Yoff,Dakar) : '
                    'elle est séparée automatiquement comme dans le notebook.</div>', unsafe_allow_html=True)
        onglet_multiple('car_data', ['Marque', 'Année', 'Boite_Vitesse', 'Quartier', 'Prix'],
                        Pred_func, 'Venant_Occasion_Prediction', preparation=preparation_car_data,
                        exemple='exemple_car_data.csv')


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

    tab1, tab2 = st.tabs(['Prédiction simple', 'Prédiction multiple'])
    with tab1:
        with st.form('form_car_dataset'):
            c1, c2, c3 = st.columns(3)
            Kms_Driven = c1.number_input('Kilométrage', min_value=0, value=30000, step=1000)
            Present_Price = c2.number_input('Prix actuel (neuf)', min_value=0.0, value=7.0, step=0.1)
            Age = c3.number_input('Âge (années)', min_value=0, value=7, step=1)
            Fuel_Type = c1.selectbox('Carburant', options=uniques[0])
            Seller_Type = c2.selectbox('Vendeur', options=uniques[1])
            Transmission = c3.selectbox('Transmission', options=uniques[2])
            submit = st.form_submit_button('Prédire')
        if submit:
            y_pred = Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            resultat_regression(y_pred, '', 'Prix de vente estimé')
    with tab2:
        onglet_multiple('car_dataset', ['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type',
                                        'Transmission', 'Age'],
                        Pred_func, 'Selling_Price_Prediction', exemple='exemple_car_dataset.csv')


# =====================================================================
# Mise en page
# =====================================================================
# Barre du haut
st.markdown("""
<div class="glass topbar">
  <div class="who"><div class="avatar">PS</div><span>Bienvenue, <b>Pierre Sene</b></span></div>
  <div class="chip"><span class="dot live"></span>4 modèles en ligne</div>
  <div class="chip">Licence 3 · Machine Learning</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hello">Tableau de bord du devoir</div>', unsafe_allow_html=True)

# Navigation entre les 4 modèles (pilule)
PAGES = {'Téléphone': page_telephone, 'Taxi': page_taxi,
         'Car Data': page_car_data, 'Car Dataset': page_car_dataset}
choix = st.segmented_control('Choisir un modèle', list(PAGES.keys()), default='Téléphone',
                             label_visibility='collapsed', key='nav')
if choix is None:
    choix = 'Téléphone'

st.write("")
entete(INFOS[choix])

# Zone de prédiction dans un conteneur en verre
with st.container(key='glass_zone'):
    PAGES[choix]()

st.markdown('<div class="foot">Scikit-learn · Streamlit — méthode du cours (ML 1 & ML 2)</div>',
            unsafe_allow_html=True)
