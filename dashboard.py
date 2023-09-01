import streamlit as st
import requests
import pandas as pd
import json
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/"
dataset = pd.read_csv('X_prod5.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset.fillna('')
sk_ids = list(dataset['SK_ID_CURR'])[:50]


sk_id_curr = st.sidebar.selectbox('Saisir N° :', sk_ids, key=1)

st.title('Dashboard Scoring Bancaire')
st.write('N° dossier : ', sk_id_curr)
st.header('Scoring')
    

sk_id_infos = dataset[dataset['SK_ID_CURR'] == sk_id_curr]
sk_id_infos = sk_id_infos.drop(['SK_ID_CURR'], axis=1)
print(sk_id_infos)
data = sk_id_infos.to_dict(orient='records')[0]

# # Bouton pour effectuer la prédiction
# if st.button('Prédire'):
#     response = requests.post(API_URL + 'predict/', json=data)
#     if response.status_code == 200:
#         result = response.json()
#         st.success(f'Résultat du Scoring : {result["score"]:.2f}')
#     else:
#         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')

if st.button('Prédire'):
    response = requests.post(API_URL + 'predict/', json=data)
    if response.status_code == 200:
        result = response.json()
        if result["score"] > 2:
            st.markdown(f'<span style="color: white; background-color: red">Résultat du Scoring : {result["score"]:.2f}</span>', unsafe_allow_html=True)
        else:
            st.success(f'Résultat du Scoring : {result["score"]:.2f}')
    else:
        st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')


# Bouton pour obtenir l'importance des fonctionnalités
if st.button('Obtenir l\'Importance des Fonctionnalités'):
    response = requests.post(API_URL + 'features_imp/', json=data)
    if response.status_code == 200:
        result = response.json()
        st.write('Importance des Fonctionnalités :')
        st.write(result['data'])
    else:
        st.error('Erreur lors de l\'obtention de l\'importance des fonctionnalités. Veuillez vérifier vos données.')

# Bouton pour obtenir l'interprétation SHAP
if st.button('Obtenir l\'Interprétation'):
    response = requests.post('http://127.0.0.1:8000/interpretation/', json=data)

    if response.status_code == 200:
        result = response.json()
        st.write('Interprétation :')
        
        # Afficher l'image SHAP encodée en base64
        encoded_img = result['summary_plot']
        decoded_img = base64.b64decode(encoded_img)
        st.image(Image.open(io.BytesIO(decoded_img)))
    else:
        st.error('Erreur lors de l\'obtention de l\'interprétation. Veuillez vérifier vos données.')
