
# ML PROJECT -- Email & SMS Spam Classifier

This is an end-to-end machine learning project designed to classify SMS or email messages as Spam or Not Spam using Natural Language Processing (NLP) techniques and a machine learning model. It consists of a training notebook and a deployed web app built using Streamlit.


## Project Overview

The project has two core components:

1. Model Training (in Jupyter Notebook): A notebook was used to load and process the dataset, transform the text data into numerical format, and train a spam classifier.

2. Streamlit Web App: A simple and interactive web app was built to allow users to input text messages and get real-time spam classification results.


## Data Source

The dataset used for this project is the SMS Spam Collection Dataset available on Kaggle:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


## Technologies Used

- Python
- scikit-learn
- NLTK
- Streamlit
- Pickle


## Procedure

1. Text Preprocessing
   The text data was cleaned by converting everything to lowercase, tokenizing each message, and removing punctuation and non-alphanumeric characters. Common stopwords were filtered out, and stemming was applied using NLTK’s PorterStemmer.

2. Vectorization
   After cleaning, the text was converted into numerical format using TfidfVectorizer. To keep things efficient, the vocabulary was limited to the top 3000 features. The vectorizer was then saved using Pickle for later use in the app.

3. Model Training
   A Multinomial Naive Bayes classifier was trained on the vectorized data. The model’s performance was evaluated using accuracy, confusion matrix, and precision score. Once validated, the model was saved as a .pkl file.


## File Structure

The project folder includes:
- The training notebook (sms-spam-detection.ipynb)
- The trained model and vectorizer files (model.pkl and vectorizer.pkl)
- The web app script (app.py)
- A text file for downloading required NLTK data
- requirements.txt listing the project dependencies
- A Procfile for deployment
- This README

## Running

To run the project locally, a virtual environment was created, and all necessary libraries were installed using the requirements.txt file. Required NLTK resources (like tokenizers and stopword lists) were downloaded in Python.

The Streamlit app can be launched, and it loads the pre-trained model and vectorizer to classify any user-inputted message as spam or not.

## Working

1. The user types a message into the web interface.
2. The message is preprocessed using the same steps as during training.
3. The cleaned message is vectorized using the saved TF-IDF vectorizer.
4. The trained model then makes a prediction.
5. The result is displayed as either 'Spam' or 'Not Spam'.

## Deployment

For deployment, the entire project was pushed to GitHub and connected to Streamlit Cloud. The main script (app.py) was specified as the entry point, and the necessary dependencies were managed through the requirements.txt file.


## Requirements

- streamlit
- scikit-learn
- nltk
