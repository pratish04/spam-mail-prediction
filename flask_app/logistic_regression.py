import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# loading the data from csv file to a pandas dataframe
raw_mail_data = pd.read_csv("training_data.csv")

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), "")

# label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category', ] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category', ] = 1

# separating the data as text and labels
X = mail_data['Message']
Y = mail_data['Category']   

# splitting the data into test data and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# transform the text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# training the model 
model = LogisticRegression()
model.fit(X_train_features, Y_train)

#prediction on training data
prediction_training = model.predict(X_train_features)
accuracy_training = accuracy_score(Y_train, prediction_training)

prediction_test = model.predict(X_test_features)
accurary_test = accuracy_score(Y_test, prediction_test)

print('accuracy_score: ', accuracy_training)
print('accurary_test: ', accurary_test)

# saving the trained model in a file
joblib.dump(model, 'trained_model.joblib')
joblib.dump(feature_extraction, 'fitted_vectorizer.joblib')