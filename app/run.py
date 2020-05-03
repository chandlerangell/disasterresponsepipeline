import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def tokenize(text):
    """Tokenize text by removing non alphabetical characters, converting to lower case, lemmatizing, and removing stop words."""
    text = re.sub(r"[^a-zA-Z]", " ", text) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
        
    return clean_tokens

# initialize app
app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/classifier.pkl")



@app.route('/')
@app.route('/index')
def index():
    """Main webpage for displaying visuals of dataset and receiving user input for model"""
    # extract data for first graph, genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # extract data for second graph, category frequency chart
    categories = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    category_counts = df[categories].sum().sort_values(ascending=False)
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    values=genre_counts,
                    labels=genre_names
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        }, 
        {
            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
        
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Webpage to load when user submits a query"""
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Main Application"""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()