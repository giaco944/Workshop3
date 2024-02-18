from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle SVM
svm_model = SVC(kernel='linear', random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
svm_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)

# Afficher la précision du modèle
print("Précision du modèle SVM:", accuracy)
