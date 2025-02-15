Projet de Prédiction d'Anomalies avec un Modèle SVM
Ce projet permet de prédire si une image est "OK" ou "Anomalie" en utilisant un modèle SVM entraîné sur des images. Il inclut également un suivi de l'empreinte carbone associée à l'entraînement du modèle ainsi qu'à l'inférence.

Prérequis
Avant de commencer, assurez-vous d'avoir installé les dépendances suivantes dans votre environnement Python :

OpenCV : Pour le traitement des images.
Scikit-learn : Pour l'entraînement et la prédiction avec le modèle SVM.
joblib : Pour la sauvegarde et le chargement du modèle.
codecarbon : Pour suivre l'empreinte carbone de l'entraînement et de l'inférence.
Vous pouvez installer ces dépendances via pip :

bash

pip install opencv-python scikit-learn joblib codecarbon

Structure du Projet
bash


/Hackathon
/Nouvelle_dataset
    /Anomaly               # Dossier contenant les images "Anomalies"
    /Normal                # Dossier contenant les images "OK"
    /NewImage              # Dossier pour tester de nouvelles images
svm_model.pkl          # Modèle SVM entraîné sauvegardé
train_model.py         # Script d'entraînement du modèle SVM
test_model.py          # Script pour tester le modèle et faire des prédictions
README.md              # Ce fichier
Entraînement du Modèle (train_model.py)
Le fichier train_model.py permet d'entraîner un modèle SVM pour classer les images en deux catégories : "OK" ou "Anomalie". Le modèle est ensuite sauvegardé dans un fichier svm_model.pkl.

Étapes du script d'entraînement :
Chargement des images : Les images sont chargées à partir des dossiers spécifiés (/Normal pour les images "OK" et /Anomaly pour les images "Anomalies").
Prétraitement des images : Chaque image est redimensionnée à 64x64 et normalisée.
Entraînement du modèle SVM : Le modèle est entraîné en utilisant un kernel linéaire.
Suivi de l'empreinte carbone : L'empreinte carbone de l'entraînement est calculée à l'aide de codecarbon.
Sauvegarde du modèle : Le modèle est sauvegardé dans un fichier svm_model.pkl pour une utilisation ultérieure.

Commande pour exécuter le script d'entraînement :
bash

python train_model.py
Test du Modèle (test_model.py)
Le fichier test_model.py permet de charger un modèle préexistant (svm_model.pkl) et d'effectuer une prédiction sur une image fournie par l'utilisateur. Il renvoie également l'empreinte carbone associée à l'inférence et le niveau de confiance de la prédiction.

Étapes du script de test :
Chargement du modèle : Le modèle SVM préalablement sauvegardé est chargé depuis svm_model.pkl.
Prétraitement de l'image : L'image fournie est prétraitée (redimensionnée et normalisée).
Prédiction de l'image : Le modèle prédit si l'image est "OK" ou "Anomalie".
Calcul de l'empreinte carbone : L'empreinte carbone de l'inférence est calculée à l'aide de codecarbon.
Affichage du résultat : Le script affiche la prédiction, l'empreinte carbone, et un niveau de confiance de la prédiction.

Commande pour exécuter le script de test :
bash

python test_model.py

Exemple de résultat :
plaintext

📂 Modèle SVM chargé avec succès.
🔮 Prédiction pour l'image : Anomalie
💡 L'empreinte carbone totale pendant l'inférence est de : 0.0005 kg CO2e
{'prediction': 'Anomalie', 'carbon_emissions': 0.0005, 'confidence': 1.0}
Suivi de l'Empreinte Carbone
L'empreinte carbone de l'entraînement et de l'inférence est calculée en utilisant la bibliothèque codecarbon. Cela vous permet de suivre l'impact environnemental du modèle, que ce soit pendant l'entraînement ou lors des prédictions.

NB: Prenez soin de vérifiez les path des données d'entrainement et de l'image à chargé pour le test
Conclusion
Ce projet fournit un modèle SVM pour prédire des anomalies dans des images, tout en intégrant un suivi de l'empreinte carbone pour promouvoir des pratiques d'IA plus responsables. Vous pouvez facilement adapter le projet pour d'autres cas d'utilisation ou étendre le modèle pour traiter davantage de classes d'images.