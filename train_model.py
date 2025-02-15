import os
import numpy as np
import cv2
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker

# Définition des chemins des dossiers
ok_images_path = 'C:/Users/PC/Desktop/hackaton/Normal'
anomaly_images_path = 'C:/Users/PC/Desktop/hackaton/Anomaly'

# Fonction pour charger et prétraiter les images
def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Redimensionnement
            img = img / 255.0  # Normalisation
            images.append(img)
            labels.append(label)
    print(f"📥 {len(images)} images chargées depuis {folder}")
    return np.array(images), np.array(labels)

# Chargement des images
ok_images, ok_labels = load_images_from_folder(ok_images_path, 'OK')
anomaly_images, anomaly_labels = load_images_from_folder(anomaly_images_path, 'Anomalie')

# Fusionner les données
X = np.concatenate((ok_images, anomaly_images), axis=0)
y = np.concatenate((ok_labels, anomaly_labels), axis=0)

# Encoder les labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Séparer les données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialiser l'EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

# Entraîner le modèle SVM
model = svm.SVC(kernel='linear', C=1)
model.fit(X_train.reshape(len(X_train), -1), y_train)  # Conversion en vecteurs 1D

# Prédictions
y_pred = model.predict(X_test.reshape(len(X_test), -1))

# Évaluation du modèle
print("\n📊 Évaluation du modèle :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Précision du modèle : {accuracy_score(y_test, y_pred):.4f}")

# Empreinte carbone
emissions = tracker.stop()
print(f"\n💡 L'empreinte carbone totale pendant l'entraînement est de : {emissions:.4f} kg CO2e")

# Sauvegarde du modèle
joblib.dump(model, 'svm_model.pkl')
print("💾 Modèle SVM sauvegardé sous 'svm_model.pkl'.")
