from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def visualize_mixed_review(review_detail):
    """Visualiser la composition d'une critique mitigée."""
    # Créer un diagramme en secteurs pour la proportion de phrases positives vs négatives
    labels = ['Phrases positives', 'Phrases négatives']
    sizes = [review_detail['pos_count'], review_detail['neg_count']]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0.1)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Répartition des sentiments dans la critique')
    
    # Afficher les probabilités de sentiment pour chaque phrase
    plt.subplot(1, 2, 2)
    sentences = [f"Phrase {i+1}" for i in range(len(review_detail['sentences']))]
    probabilities = [sent[2] for sent in review_detail['sentences']]
    sentiments = [sent[1] for sent in review_detail['sentences']]
    
    colors = ['lightcoral' if s == "négatif" else 'lightgreen' if s == "positif" else 'lightgray' 
              for s in sentiments]
    
    plt.barh(sentences, probabilities, color=colors)
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.title('Probabilité de sentiment positif par phrase')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.show()