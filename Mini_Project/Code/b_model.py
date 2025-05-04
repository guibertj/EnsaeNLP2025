from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from a_load import clean_text

def train_sentiment_model(train_df):
    """Entraîner un modèle de classification de sentiment."""
    print("Nettoyage des données textuelles...")
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    
    # Extraction de caractéristiques avec TF-IDF
    print("Extraction de caractéristiques avec TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8)
    X_train = vectorizer.fit_transform(train_df['cleaned_review'])
    y_train = train_df['sentiment']
    
    # Entraîner le classifieur
    print("Entraînement du modèle de régression logistique...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    
    return vectorizer, classifier