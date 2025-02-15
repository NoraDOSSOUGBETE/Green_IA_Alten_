
import os
import numpy as np
import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker

# Fonction pour charger et prétraiter une image individuelle
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Redimensionnement
        img = img / 255.0  # Normalisation
        return img
    else:
        raise ValueError("L'image n'a pas pu être chargée.")

# Charger le modèle déjà sauvegardé
model = joblib.load('svm_model.pkl')
print("📂 Modèle SVM chargé avec succès.")

# Initialiser le tracker d'empreinte carbone avec l'option `allow_multiple_runs=True`
tracker = EmissionsTracker(allow_multiple_runs=True)
tracker.start()

# Fonction pour faire une prédiction sur une image donnée
def predict_image(image_path):
    # Prétraiter l'image
    image = preprocess_image(image_path)

    # Faire la prédiction avec le modèle
    prediction = model.predict(image.reshape(1, -1))  # Conversion en vecteur 1D

    # Le label prédit (OK ou Anomalie)
    predicted_label = prediction[0]
    
    # Renommer les labels pour correspondre à leur signification
    label_encoder = LabelEncoder()
    label_encoder.fit(['OK', 'Anomalie'])  # Assurez-vous que le label encoder a bien été ajusté
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    
    # Retourner la prédiction
    print(f"🔮 Prédiction pour l'image : {predicted_label_name}")
    
    # Calcul de l'empreinte carbone pour l'inférence
    emissions = tracker.stop()

    # Vérifier si l'empreinte carbone est valide avant d'afficher
    if emissions is not None:
        print(f"\n💡 L'empreinte carbone totale pendant l'inférence est de : {emissions} kg CO2e")
    else:
        print("\n⚠️ Impossible de calculer l'empreinte carbone.")

    # Évaluation de la fiabilité de la prédiction
    confidence = 1.0  # Vous pouvez calculer une probabilité de confiance ou un score selon votre modèle
    
    return {'prediction': predicted_label_name, 'carbon_emissions': emissions, 'confidence': confidence}

# Exemple d'utilisation
image_path = 'C:/Users/PC/Desktop/hackaton/Anomaly/candano_003.jpg'  # Chemin de l'image à tester
result = predict_image(image_path)
print(result)
