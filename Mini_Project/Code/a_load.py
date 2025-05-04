import numpy as np
import pandas as pd
import os
import tarfile
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Download the punkt_tab data
stop_words = set(stopwords.words('english'))

def extract_dataset(tar_path, extract_path='.'):
    """Extraire le dataset IMDb du fichier tar.gz."""
    print(f"Extraction de {tar_path} vers {extract_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print("Extraction terminée!")

def clean_text(text):
    """Nettoyer et prétraiter les données textuelles."""
    # Supprimer les balises HTML
    text = re.sub(r'<.*?>', '', text)
    # Supprimer les caractères non alphabétiques
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convertir en minuscules et diviser
    words = text.lower().split()
    # Supprimer les mots vides
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def load_dataset(dataset_path):
    """Charger le dataset IMDb dans un DataFrame pandas."""
    reviews = []
    labels = []
    
    # Charger les critiques positives
    pos_path = os.path.join(dataset_path, 'pos')
    for filename in os.listdir(pos_path):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1)  # Étiquette positive
    
    # Charger les critiques négatives
    neg_path = os.path.join(dataset_path, 'neg')
    for filename in os.listdir(neg_path):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(0)  # Étiquette négative
    
    return pd.DataFrame({'review': reviews, 'sentiment': labels})
