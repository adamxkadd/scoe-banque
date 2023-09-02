
# dashboard.py
import streamlit as st
import requests

API_URL = "https://scoe-banque.streamlit.app/"  # Mettez l'URL de votre API FastAPI ici

st.title("Tableau de bord Streamlit")

# Créez un formulaire pour saisir des données
st.header("Saisissez des données:")
user_input = st.text_input("Entrez des données:", "Exemple de données")

# Bouton pour envoyer les données à l'API
if st.button("Envoyer"):
    data = {"input_data": user_input}
    response = requests.post(f"{API_URL}/predict/", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(result["result"])
    else:
        st.error("Une erreur s'est produite lors de la prédiction.")


# import streamlit as st
# import requests
# import pandas as pd
# import json
# import base64
# from PIL import Image
# import io

# # API_URL = "http://127.0.0.1:8000/"
# API_URL = "https://scoe-banque.streamlit.app/"
# dataset = pd.read_csv('data.csv')
# dataset = dataset.drop(['Unnamed: 0'], axis=1)
# dataset = dataset.fillna('')
# sk_ids = list(dataset['SK_ID_CURR'])[:50]


# sk_id_curr = st.sidebar.selectbox('Saisir N° :', sk_ids, key=1)

# st.title('Dashboard Scoring Bancaire')
# st.write('N° dossier : ', sk_id_curr)
# st.header('Scoring')
    

# sk_id_infos = dataset[dataset['SK_ID_CURR'] == sk_id_curr]
# sk_id_infos = sk_id_infos.drop(['SK_ID_CURR'], axis=1)
# print(sk_id_infos)
# data = sk_id_infos.to_dict(orient='records')[0]

# df = pd.DataFrame({
#   'first column': [1, 2, 3, 4],
#   'second column': [10, 20, 30, 40]
# })

# df


# if st.button('Prédire'):
#     response = requests.post(API_URL + 'predict', json=data)
#     if response.status_code == 200:
#         try:
#             response_json = json.dumps(response.json())  # Convertir la réponse en chaîne JSON
#             result = json.loads(response_json)  # Charger la chaîne JSON en tant que dictionnaire Python
#             st.success(f'Résultat du Scoring : {result["score"]:.2f}')
#         except json.decoder.JSONDecodeError as e:
#             st.error('Erreur lors de la prédiction. La réponse de l\'API n\'est pas valide.')
#     else:
#         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')



# # Bouton pour effectuer la prédiction
# # if st.button('Prédire'):
# #     response = requests.post(API_URL + 'predict/', json=data)
# #     st.write(response.text)
# #     if response.status_code == 200:
# #         result = response.json()
# #         if "score" in result:
# #             st.success(f'Résultat du Scoring : {result["score"]:.2f}')
# #         else:
# #             st.error('Réponse API invalide.')
# #     else:
# #         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')
        


# # if st.button('Prédire'):
# #     response = requests.post(API_URL + 'predict/', json=data)
# #     if response.status_code == 200:
# #         result = response.json()
# #         if result["score"] > 2:
# #             st.markdown(f'<span style="color: white; background-color: red">Résultat du Scoring : {result["score"]:.2f}</span>', unsafe_allow_html=True)
# #         else:
# #             st.success(f'Résultat du Scoring : {result["score"]:.2f}')
# #     else:
# #         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')


# # Bouton pour obtenir l'importance des fonctionnalités
# if st.button('Obtenir l\'Importance des Fonctionnalités'):
#     response = requests.post(API_URL + 'features_imp/', json=data)
#     if response.status_code == 200:
#         result = response.json()
#         st.write('Importance des Fonctionnalités :')
#         st.write(result['data'])
#     else:
#         st.error('Erreur lors de l\'obtention de l\'importance des fonctionnalités. Veuillez vérifier vos données.')

# # Bouton pour obtenir l'interprétation SHAP
# if st.button('Obtenir l\'Interprétation'):
#     response = requests.post('http://127.0.0.1:8000/interpretation/', json=data)

#     if response.status_code == 200:
#         result = response.json()
#         st.write('Interprétation :')
        
#         # Afficher l'image SHAP encodée en base64
#         encoded_img = result['summary_plot']
#         decoded_img = base64.b64decode(encoded_img)
#         st.image(Image.open(io.BytesIO(decoded_img)))
#     else:
#         st.error('Erreur lors de l\'obtention de l\'interprétation. Veuillez vérifier vos données.')
