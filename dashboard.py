import streamlit as st
import requests
import pandas as pd
import json
import base64
from PIL import Image
import io

# API_URL = "http://127.0.0.1:8000/"
API_URL = "https://scoe-banque.streamlit.app/"
dataset = pd.read_csv('data.csv')
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


if st.button('Prédire'):
    response = requests.post(API_URL + 'predict/', json=data)
    if response.status_code == 200:
        try:
            response_json = json.dumps(response.json())  # Convertir la réponse en chaîne JSON
            result = json.loads(response_json)  # Charger la chaîne JSON en tant que dictionnaire Python
            st.success(f'Résultat du Scoring : {result["score"]:.2f}')
        except json.decoder.JSONDecodeError as e:
            st.error('Erreur lors de la prédiction. La réponse de l\'API n\'est pas valide.')
    else:
        st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')



# Bouton pour effectuer la prédiction
# if st.button('Prédire'):
#     response = requests.post(API_URL + 'predict/', json=data)
#     st.write(response.text)
#     if response.status_code == 200:
#         result = response.json()
#         if "score" in result:
#             st.success(f'Résultat du Scoring : {result["score"]:.2f}')
#         else:
#             st.error('Réponse API invalide.')
#     else:
#         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')
        


# if st.button('Prédire'):
#     response = requests.post(API_URL + 'predict/', json=data)
#     if response.status_code == 200:
#         result = response.json()
#         if result["score"] > 2:
#             st.markdown(f'<span style="color: white; background-color: red">Résultat du Scoring : {result["score"]:.2f}</span>', unsafe_allow_html=True)
#         else:
#             st.success(f'Résultat du Scoring : {result["score"]:.2f}')
#     else:
#         st.error('Erreur lors de la prédiction. Veuillez vérifier vos données.')


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





import base64
import io
import pickle
import numpy as np
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import joblib
import lightgbm as lgb

app = FastAPI()

print("Hi API Scoring")

# Charger le modèle et le mettre en cache
file = open("lgbm.pkl", 'rb')
object_file = joblib.load(file)
file.close()
model = object_file
print("Model loaded", model)

@app.get("/")
def loaded():
    return "ICI API…"


@app.post('/predict/')
async def scoring(data: dict):
    response_text = "Le score est : 42"  
    return JSONResponse(content=response_text)


# @app.post('/predict/')
# async def scoring(data: dict):
#     data_df = pd.DataFrame.from_dict(data, orient='index').transpose()
#     applicant_score = model.predict_proba(data_df)[0][1] * 100 
#     response_data = {"status": "ok", "score": applicant_score}
#     return JSONResponse(content=response_data)

# @app.post('/predict/')
# async def scoring(data: dict):
#     data_df = pd.DataFrame.from_dict(data, orient='index').transpose()
#     applicant_score = model.predict_proba(data_df)[0][1] * 100 
#     # return {"status": "ok", "score": applicant_score}
#     return {"score": "applicant_score"}

@app.post('/features_imp/')
# async def send_features_importance(data: dict):
#     data_df = pd.DataFrame.from_dict(data, orient='index').transpose()
#     transformed_data = model[:-1].transform(data_df)
#     model = model.named_steps["model"]
#     features_importance = pd.Series(model.feature_importances_)
#     features_importance_json = json.loads(features_importance.to_json())
#     return {"status": "ok", "data": features_importance_json}
async def send_features_importance(data: dict):
    data_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    
    # Obtenir l'objet modèle à partir du pipeline (LightGBMClassifier)
    model_obj = model.named_steps["regressor"]  # Remplacez "regressor" par le nom de votre modèle LightGBM
    
    # Transformer les données si nécessaire (dépend du preprocessing dans votre modèle)
    transformed_data = data_df # Si aucune transformation nécessaire
    
    # Assurez-vous que votre modèle est un modèle LightGBM (LGBMClassifier)
    if isinstance(model_obj, lgb.LGBMClassifier):
        # Obtenez l'importance des fonctionnalités directement depuis le modèle
        features_importance = model_obj.feature_importances_
        
        # Créez un dictionnaire avec le nom des fonctionnalités en tant que clés et l'importance en tant que valeurs
        feature_importance_dict = {str(feature): int(importance) for feature, importance in zip(data_df.columns, features_importance)}
        
        return {"status": "ok", "data": feature_importance_dict}
    else:
        return {"status": "error", "message": "Le modèle n'est pas un LGBMClassifier."}


@app.post('/interpretation/')
async def interpretation(data: dict):
    data_df = pd.DataFrame.from_dict(data, orient='index').transpose()

    # Obtenir l'objet modèle à partir du pipeline
    model_obj = model.get_params()['regressor']
    explainer = shap.TreeExplainer(model_obj)
    shap_values = explainer.shap_values(data_df)
    shap.summary_plot(shap_values, show=False)

    # Sauvegarder le plot en tant qu'image
    plt.tight_layout()
    plt.savefig('summary_plot.png')

    # Convertir l'image en base64
    img = Image.open("summary_plot.png", mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return {"summary_plot": encoded_img}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
