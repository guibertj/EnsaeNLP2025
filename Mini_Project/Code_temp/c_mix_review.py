from a_load import clean_text
from nltk.tokenize import sent_tokenize

def find_mixed_reviews(reviews_df, vectorizer, classifier, threshold=0.3):
    """
    Identifier les critiques mitigées en analysant le sentiment au niveau des phrases.
    Une critique est considérée comme mitigée si elle contient à la fois des phrases
    positives et négatives avec une certaine proportion.
    """
    mixed_reviews = []
    mixed_reviews_details = []
    
    for idx, review in enumerate(reviews_df['review']):
        # Diviser la critique en phrases
        sentences = sent_tokenize(review)
        
        if len(sentences) <= 1:
            continue
            
        # Analyser le sentiment de chaque phrase
        pos_count = 0
        neg_count = 0
        sentence_sentiments = []
        
        for sentence in sentences:
            # Nettoyer la phrase
            cleaned_sentence = clean_text(sentence)
            if not cleaned_sentence:  # Ignorer les phrases vides après nettoyage
                continue
                
            # Vectoriser et prédire
            vec_sentence = vectorizer.transform([cleaned_sentence])
            prediction = classifier.predict_proba(vec_sentence)[0]
            prob_positive = prediction[1]
            
            # Déterminer le sentiment avec un seuil de confiance
            if prob_positive >= 0.7:
                pos_count += 1
                sentiment = "positif"
            elif prob_positive <= 0.3:
                neg_count += 1
                sentiment = "négatif"
            else:
                sentiment = "neutre"
                
            sentence_sentiments.append((sentence, sentiment, prob_positive))
        
        # Calculer les proportions
        total_valid_sentences = pos_count + neg_count
        if total_valid_sentences > 0:
            pos_ratio = pos_count / total_valid_sentences
            neg_ratio = neg_count / total_valid_sentences
            
            # Une critique est mitigée si elle contient une proportion significative
            # à la fois de phrases positives et négatives
            if (pos_ratio >= threshold and neg_ratio >= threshold and 
                pos_count >= 2 and neg_count >= 2):
                mixed_reviews.append(idx)
                mixed_reviews_details.append({
                    'review': review,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'pos_ratio': pos_ratio,
                    'neg_ratio': neg_ratio,
                    'sentences': sentence_sentiments
                })
    return mixed_reviews, mixed_reviews_details