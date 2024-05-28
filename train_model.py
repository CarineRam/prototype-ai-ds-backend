from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

def train_model_and_save():
    data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'])
    X, y = data.data, data.target

    # Créer un pipeline de traitement de texte et modèle
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())

    # Entraîner le modèle
    model.fit(X, y)

    # Sauvegarder le modèle
    joblib.dump(model, 'text_classification_model.joblib')

if __name__ == "__main__":
    train_model_and_save()