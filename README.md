# Devoir Machine Learning — 4 modèles déployés avec Streamlit

Une seule application Streamlit (`app.py`). Le menu latéral propose 4 modèles :

| Modèle | Type | Cible | Origine |
|---|---|---|---|
| Téléphone | Classification | `etat` | modèle vu en classe (ML 1) |
| Taxi | Régression | `Trip_Price` | modèle vu en classe (ML 2) |
| Car Data | Classification | `Venant_Occasion` | `notebooks/car_data_classification.ipynb` |
| Car Dataset | Régression | `Selling_Price` | `notebooks/car_dataset_regression.ipynb` |

Chaque modèle a deux onglets :
- **Prédiction simple** : un formulaire, puis le bouton *Predict* ;
- **Prédiction multiple** : on charge un CSV, puis on télécharge le résultat avec la colonne de prédiction.

## Organisation

```
app.py                  application Streamlit
requirements.txt        bibliothèques (versions figées)
models/                 encoders, uniques, scaler et modèle (.joblib) de chaque modèle
data/                   datasets d'entraînement
data/exemples/          CSV d'exemple pour tester la prédiction multiple
notebooks/              notebooks rendus + scripts de réentraînement des modèles vus en classe
```

## Lancer en local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Pour relancer les notebooks, il faut aussi `matplotlib`, `seaborn`, `plotly` et `jupyter`. Exécute-les depuis le dossier `notebooks/`.

## Déployer sur Streamlit Community Cloud

1. Crée un dépôt **public** sur GitHub et pousses-y ce dossier (`app.py`, `requirements.txt` et `models/` doivent être à la racine).
2. Va sur https://share.streamlit.io et connecte-toi avec ton compte GitHub.
3. Clique sur **Create app**, puis choisis le dépôt, la branche `main` et le fichier `app.py`.
4. Dans *Advanced settings*, choisis **Python 3.12**.
5. Clique sur **Deploy**. Tu obtiens une URL publique en `https://<nom>.streamlit.app`.
