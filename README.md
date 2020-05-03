# Disaster Response Pipeline Project

## Project Dependencies
- sklearn
- pandas
- numpy
- nltk
- sqlalchemy
- flask
- plotly

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Description
This project uses ETL techniques and ML algoroithms to classify disaster response messages.  The final product is a web app that shows statistics about the dataset and allows users to input a message and see the classification of that message.

## File Descriptions
app\
|- templates				HTML templates\
|- |- go.html 				Webpage for classifying a user message\
|- |- master.html 			Main webpage\
|- run.py 					Launches the webapp\
data\
|- DisasterResponse.db		Database created by process_data.py\
|- disaster_categories.csv	Original dataset, disaster message categories\
|- disaster_messages.csv	Original dataset, disaster messages\
|- process_data.py			ETL pipeline for processing the datasets and formatting for the ML stage\
models\
|- classifier.pkl			Pickle file of sklearn classifier\
|- train_classifier.py		Create classifier model based on message data\
Results\
|- App Home View			Screenshot of app homepage\
|- App Classify View		Screenshot of app classify page\
|- Precision Recall and F1 Results.txt		Printout of ML pipeline performance parameters\

## Authors
Project templates provided by Udacity\
ETL and ML scripts created by Chandler Angell

## Licensing
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





