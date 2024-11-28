# Importation des bibliothèques nécessaires
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Exemple de données (ajoutez vos propres données ou utilisez un dataset externe)
data = [
    ("Gagnez de l'argent rapidement ! Cliquez ici.", "spam"),
    ("Rendez-vous demain pour la réunion.", "ham"),
    ("Offre spéciale : réductions incroyables maintenant !", "spam"),
    ("Peux-tu me rappeler plus tard ?", "ham"),
    ("Achetez maintenant et obtenez 50% de réduction.", "spam"),
    ("Bonjour, comment allez-vous ?", "ham")
]

# Séparation des messages et des étiquettes
messages, labels = zip(*data)

# Pré-traitement : Conversion des messages en vecteurs numériques
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)
y = labels

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
print("Précision :", accuracy_score(y_test, y_pred))

# Tester avec un nouveau message
new_message = ["Offre spéciale pour vous ! Cliquez pour en savoir plus."]
new_message_vectorized = vectorizer.transform(new_message)
prediction = model.predict(new_message_vectorized)
print(f"Message : '{new_message[0]}' - Classification : {prediction[0]}")
