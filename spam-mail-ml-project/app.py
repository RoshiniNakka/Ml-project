import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv',encoding=('ISO-8859-1'))
# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
# printing the first 5 rows of the dataframe
#mail_data.head()
# checking the number of rows and columns in the dataframe
#mail_data.shape
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
X = mail_data['Message']

Y = mail_data['Category']
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)
# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
#print(X_train)
#print(X_train_features)
model = LogisticRegression()
model.fit(X_train_features, Y_train)
# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
#print('Accuracy on training data : ', accuracy_on_training_data)
# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
#print('Accuracy on test data : ', accuracy_on_test_data)
print("dhbjdb")

from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def result():
    a=request.form['text-input']
    #print(type(a))
    input_mail = [a,]
    #print(input_mail)
    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction

    prediction = model.predict(input_data_features)
    #print(prediction)
    s=''


    if (prediction[0]==1):
       s='Ham mail'

    else:
       s='Spam mail'


    return render_template('home.html',s=s)
