import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import time

import nltk
nltk.download('stopwords')

# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM data", con=engine)
    X = df[['message']]
    categories = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    Y = df[categories]

    # for running, limit size
    n = 1000
    Y = Y.loc[:n]
    X = X.loc[:n]
    X = X['message'].tolist()
    Y = Y.values
    Y = Y.astype(bool).astype(int)
    
    return X, Y, categories

def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
        
    return clean_tokens


def build_model():
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) ])


def evaluate_model(model, x_test, y_test, category_names):
    y_pred = model.predict(x_test)
    class_0 = []
    class_1 = []

    for c in range(len(category_names)):
        class_0 = class_0 + [list(precision_recall_fscore_support(y_test[:,c], y_pred[:,c], average='binary', pos_label=0))[:-1]]
        class_1 = class_1 + [list(precision_recall_fscore_support(y_test[:,c], y_pred[:,c], average='binary', pos_label=1))[:-1]]

    class_0 = np.array(class_0)
    class_1 = np.array(class_1)
    
    for idx, c in enumerate(category_names):
        print(c)
        print('\t 0 \t Prec: {p:.4f}\t Recall: {r:.4f}\t   f1: {f:.4f}'.format(p = class_0[idx,0], r = class_0[idx,1], f = class_0[idx,2]))
        print('\t 1 \t Prec: {p:.4f}\t Recall: {r:.4f}\t   f1: {f:.4f}'.format(p = class_1[idx,0], r = class_1[idx,1], f = class_1[idx,2]))
          
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()