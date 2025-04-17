📰 Fake News Detection using Machine Learning
This project is a machine learning pipeline that detects whether a news article is fake or true using Natural Language Processing (NLP) techniques. 
The model is trained on a labeled dataset containing real and fake news articles and can classify user-input text in real time.
📌 Project Overview
Fake news has become a major problem in the age of digital communication. 
In this project, I built a machine learning model that uses a combination of TF-IDF (Term Frequency-Inverse Document Frequency) 
and Multinomial Naive Bayes to detect fake news articles based on their content.

🔍 Features
Combines two datasets: True.csv and Fake.csv
Preprocesses and vectorizes text data using TF-IDF
Trains a Naive Bayes classifier to distinguish between fake and real news
Accepts user input for real-time prediction
Evaluates the model using accuracy and classification report

🧠 Technologies Used
Python
pandas
scikit-learn
TfidfVectorizer
MultinomialNB
train_test_split
accuracy_score
classification_report

🗂️ Dataset
The dataset consists of two CSV files:
True.csv: Contains legitimate news articles
Fake.csv: Contains fake/misinformation articles
Each article is labeled as:
0 → True news
1 → Fake news

📈 Results
After training the model and testing on 20% of the data:
✅ Accuracy: ~ 92
📋 Classification Report: Includes precision, recall, and F1-score for both fake and real classes

💬 Real-time Prediction Example
Enter a news article: Scientists discover water on the moon.
The news is likely to be true.

