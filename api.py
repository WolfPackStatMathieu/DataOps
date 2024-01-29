from flask import Flask, request
import torch
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle PyTorch depuis le fichier model.v1.bin
model = torch.load('model.v1.bin', map_location=torch.device('cpu'))
model.eval()  # Mettre le modèle en mode évaluation

@app.route("/")
def hello_world():
    return "<p>Hello, Lolo !</p>"


@app.route("/predict")
def predict():
    # Récupérer le paramètre GET 'prenom'
    prenom = request.args.get('prenom')
    
    # charger le modèle avec joblib
    
    # Appliquer le modèle PyTorch avec .predict
    result = make_prediction(prenom)
    
    # Retourner le résultat
    return f"Prediction for {prenom}: {result}"

    # encoder main
    # .predict
    # renvoyer le résultat
    
def encode_prenom(prenom: str) -> pd.Series:
    """
        This function encode a given name into a pd.Series.
        
        For instance alain is encoded [1, 0, 0, 0, 0 ... 1, 0 ...].
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzé-'"
    prenom = prenom.lower()
    
    return pd.Series([letter in prenom for letter in alphabet]).astype(int)


def make_prediction(prenom):
    # Charger le modèle avec joblib (si nécessaire)
    # model = joblib.load('model.joblib')

    # Encoder le prénom (vous devez avoir votre propre fonction d'encodage)
    # encoder_main(prenom)

    # Effectuer la prédiction
    # Assurez-vous d'avoir une fonction encode_prenom définie
    regr_loaded = joblib.load("model.v1.bin")
    prediction = regr_loaded.predict([prenom])

    return prediction.item()  # Retourner la valeur prédite