# SMS/Email Spam Classifier Project

## 1. Data Cleaning

- Checked data information using `df.info()`.
- Dropped unnecessary columns: 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'.
- Renamed columns to 'target' and 'text'.
- Encoded 'target' using LabelEncoder.
- Handled missing values and removed duplicate entries.

## 2. EDA (Exploratory Data Analysis)

- Explored the distribution of target classes ('ham' and 'spam').
- Visualized class distribution using a pie chart.
- Analyzed the length statistics of messages (characters, words, sentences).
- Plotted histograms and pair plots to understand the distribution and relationships.
- Used seaborn to create a correlation heatmap.

## 3. Text Preprocessing

- Performed text preprocessing steps:
  - Converted text to lowercase.
  - Tokenized text.
  - Removed special characters.
  - Removed stop words.
  - Applied stemming using PorterStemmer.
- Created word clouds for 'ham' and 'spam' messages.
- Visualized the most frequent words in 'ham' and 'spam' messages using bar plots.

## 4. Model Building

- Utilized CountVectorizer and TfidfVectorizer for feature extraction.
- Split the dataset into training and testing sets.
- Implemented various classifiers:
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
- Evaluated models using accuracy, confusion matrix, and precision.

## 5. Model Evaluation

- Trained and evaluated multiple classifiers, including SVM, KNeighbors, Decision Tree, Logistic Regression, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost.
- Used accuracy and precision as evaluation metrics.
- Explored the impact of changing the max_features parameter in TfidfVectorizer.
- Implemented a Voting Classifier and a Stacking Classifier for ensemble learning.
- Saved the final model (Multinomial Naive Bayes) and the TfidfVectorizer for deployment.

## 6. Model Improvement

- Explored improving the model by changing parameters, such as max_features in TfidfVectorizer.
- Implemented a Voting Classifier and a Stacking Classifier for better performance.

## 7. Website

- The project includes a website for user interaction and predictions.
- Try it out at [https://ravi-spam-detector.streamlit.app/]

## 8. Deploy

- Deployed the final model and vectorizer using pickle for use in production.
