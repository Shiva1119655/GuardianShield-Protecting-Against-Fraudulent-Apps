# GuardianShield-Protecting-Against-Fraudulent-Apps
# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier

#Read Data
data = pd.read_csv('../data/amazon_sentiment_data.csv', encoding='cp437')

#Train and Test Data Split
# #split data-set to train and test

X = data['review_body']
Y = data['sentiment']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42, shuffle=True)

# function to remove html elements from the reviews
def removeHTML(raw_text):
    clean_HTML = BeautifulSoup(raw_text, 'lxml').get_text() 
    return clean_HTML

# function to remove special characters and numbers from the reviews4961
def removeSpecialChar(raw_text):
    clean_SpecialChar = re.sub("[^a-zA-Z]", " ", raw_text)
    clean_SpecialChar = " ".join(filter(lambda x:x[0]!='&',  clean_SpecialChar.split()))
    clean_SpecialChar = " ".join(filter(lambda x:x[0]!='\\', clean_SpecialChar.split()))
    clean_SpecialChar = clean_SpecialChar.replace("\n", '')
    clean_SpecialChar = clean_SpecialChar.replace("\r", '')
    return clean_SpecialChar

# function to convert all reviews into lower case
def toLowerCase(raw_text):
    clean_LowerCase = raw_text.lower().split()
    return( " ".join(clean_LowerCase))

# function to remove stop words from the reviews
def removeStopWords(raw_text):
    stops = set(stopwords.words("english"))
    words = [w for w in raw_text if not w in stops]
    return( " ".join(words))

# X_training clean set
X_train_cleaned = []

for val in X_train:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_train_cleaned.append(val) 
    
# X_testing clean set
X_test_cleaned = []

for val in X_test:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_test_cleaned.append(val)

tvec = TfidfVectorizer(use_idf=True,
strip_accents='ascii')

X_train_tvec = tvec.fit_transform(X_train_cleaned)

# Support Vector Classifier model
svr_lin = LinearSVC(multi_class='ovr',C=1.0,loss='squared_hinge', dual=False)
svr_lin.fit(X_train_tvec, Y_train)

# Predict using training and testing data and display the accuracy, f-1 score, precision for Positive and Negative Sentiment Classifiers 
svr_lin_predictions = svr_lin.predict(tvec.transform(X_test_cleaned))
report = classification_report(Y_test,svr_lin_predictions, output_dict=True)
data_report = pd.DataFrame(report).transpose().round(2)
print(data_report)

# confusion matrix
svr_lin_predictions=svr_lin.predict(tvec.transform(X_test_cleaned))
ax= plt.subplot()
cm=confusion_matrix(Y_test,svr_lin_predictions)
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Greens');  
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);

# Random Forest model
random_forest = RandomForestClassifier()
random_forest.fit(X_train_tvec, Y_train)

# Predict using training and testing data and display the accuracy, f-1 score, precision for Positive and Negative Sentiment Classifiers 
ranfrst_predictions = random_forest.predict(tvec.transform(X_test_cleaned))
report = classification_report(Y_test,ranfrst_predictions, output_dict=True)
data_report = pd.DataFrame(report).transpose().round(2)
print(data_report)

# confusion matrix
ranfrst_predictions=random_forest.predict(tvec.transform(X_test_cleaned))
ax= plt.subplot()
cm=confusion_matrix(Y_test,ranfrst_predictions)
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Oranges');  
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);

# K-Nearest Neighbor model
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(X_train_tvec, Y_train)

# Predict using training and testing data and display the accuracy, f-1 score, precision for Positive and Negative Sentiment Classifiers 
knn_predictions = knn.predict(tvec.transform(X_test_cleaned))
report = classification_report(Y_test,knn_predictions, output_dict=True)
data_report = pd.DataFrame(report).transpose().round(2)
print(data_report)
# confusion matrix
knn_predictions=knn.predict(tvec.transform(X_test_cleaned))
ax= plt.subplot()
cm=confusion_matrix(Y_test,knn_predictions)
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Reds');  
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);

# Logistic regression model
lr = LogisticRegression()
lr.fit(X_train_tvec, Y_train)

# Predict using training and testing data and display the accuracy, f-1 score, precision for Positive and Negative Sentiment Classifiers 
lr_predictions = lr.predict(tvec.transform(X_test_cleaned))
report = classification_report(Y_test,lr_predictions, output_dict=True)
data_report = pd.DataFrame(report).transpose().round(2)
print(data_report)

# confusion matrix
lr_predictions=lr.predict(tvec.transform(X_test_cleaned))
ax= plt.subplot()
cm=confusion_matrix(Y_test,lr_predictions)
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='PuBu');  
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);


#Gradient Boosting Model
from sklearn.ensemble import GradientBoostingClassifier

# Logistic regression model
gr = GradientBoostingClassifier(learning_rate=0.1)
gr.fit(X_train_tvec, Y_train)

# Predict using training and testing data and display the accuracy, f-1 score, precision for Positive and Negative Sentiment Classifiers 
gr_predictions = gr.predict(tvec.transform(X_test_cleaned))
report = classification_report(Y_test,gr_predictions, output_dict=True)
data_report = pd.DataFrame(report).transpose().round(2)
print(data_report)

# confusion matrix
gr_predictions=gr.predict(tvec.transform(X_test_cleaned))
ax= plt.subplot()
cm=confusion_matrix(Y_test,gr_predictions)
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='RdPu');  
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);

import matplotlib.pyplot as plt

pf_dict = {
    'classifiers': ["SVM", "RF", "KNN", "LR", "GB"],
    'train_accuracies': [89, 88, 73, 89, 83]
        }

df = pd.DataFrame(pf_dict)
df.plot(x='classifiers', y='train_accuracies', kind='line')
plt.show()

#LR and svm had a same accuracy but LR takes time more than SVM hence, SVM considered for model deploynment

import pickle

#dump on picle
pickle.dump(svr_lin, open('electronicSentimentAnalyzer.pkl','wb'))

app.py
from google_play_scraper import app as gs
import flask
from flask import render_template, request, jsonify, session
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
import warnings
warnings.filterwarnings("ignore")
from flask_mysqldb import MySQL
import MySQLdb.cursors

app = flask.Flask(__name__, template_folder='Templates')

#code for connection
app.config['MYSQL_HOST'] = 'localhost'#hostname
app.config['MYSQL_USER'] = 'root'#username
app.config['MYSQL_PASSWORD'] = ''#password
#in my case password is null so i am keeping empty
app.config['MYSQL_DB'] = 'sentiment_analyzer'#database name

mysql = MySQL(app)
@app.route('/')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

@app.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    if flask.request.method == 'GET':
        return(flask.render_template('analyzer.html'))

#Read Data
data = pd.read_csv('data/amazon_sentiment_data.csv', encoding='cp437')

X = data['review_body']
Y = data['sentiment']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# function to remove html elements from the reviews
def removeHTML(raw_text):
    clean_HTML = BeautifulSoup(raw_text, 'lxml').get_text() 
    return clean_HTML

# function to remove special characters and numbers from the reviews4961
def removeSpecialChar(raw_text):
    clean_SpecialChar = re.sub("[^a-zA-Z]", " ", raw_text)  
    clean_SpecialChar = " ".join(filter(lambda x:x[0]!='&',  clean_SpecialChar.split()))
    clean_SpecialChar = " ".join(filter(lambda x:x[0]!='\\', clean_SpecialChar.split()))
    clean_SpecialChar = clean_SpecialChar.replace("\n", '')
    clean_SpecialChar = clean_SpecialChar.replace("\r", '')
    return clean_SpecialChar

# function to convert all reviews into lower case
def toLowerCase(raw_text):
    clean_LowerCase = raw_text.lower().split()
    return( " ".join(clean_LowerCase))

# X_training clean set
X_train_cleaned = []

for val in X_train:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_train_cleaned.append(val) 
    
tvec = TfidfVectorizer(use_idf=True,
strip_accents='ascii')

X_train_tvec = tvec.fit_transform(X_train_cleaned)


#Prediction
model = pickle.load(open('Model/electronicSentimentAnalyzer.pkl', 'rb'))


#User Login   
@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        msg=''
        if request.method == 'POST':
            phone    = request.form['sigphone']
            password = request.form['sigpassword']

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM userdetails WHERE phone = % s and password = %s', (phone, password,))
            result = cursor.fetchone()
            
        if result:
            msg = "1"
        else:
           msg = "0"
    return msg

@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if flask.request.method == 'GET':
        return(flask.render_template('register.html'))
    if flask.request.method == 'POST':
        msg=''
        #applying empty validation
        if request.method == 'POST':
            
            #passing HTML form data into python variable
            username    = request.form['regusername']
            phone       = request.form['regphone']
            password    = request.form['regpassword']
            
            #creating variable for connection
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            #query to check given data is present in database or no
            cursor.execute('SELECT * FROM userdetails WHERE phone = %s', (phone,))
            #fetching data from MySQL
            result = cursor.fetchone()
            
        if result:
            msg = '0'#'User Details Already Exists! Please Login.'
        else:
            #executing query to insert new data into MySQL
            cursor.execute('INSERT INTO userdetails VALUES (NULL, % s, % s, % s, NULL)', (username, phone, password,))
            mysql.connection.commit()
            msg = '1'#'Registeration completed! Please login.' 
    return msg

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if flask.request.method == 'POST':  
        appname    = request.form['appname']
        url        = request.form['url']
        
        file = open("cc.txt", "w",encoding='utf-8')
        link=url
        findId=link.find('id=')

        url=link[findId+3:]
        file.write(str(gs(
            url,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
        )))
        file.close()


        myfile=[]
        with open("cc.txt",encoding='utf8') as mydata:
            for data in mydata:
                myfile.append(data)

        start=myfile[0].find('comments')
        end=myfile[0].find('editorsChoice')
        c=data[start:end]


        sentiment = data.split(',')
        
        positive = 0
        negative = 0
        
        posrevs = []
        negrevs  = []
        
        for i in range(len(sentiment)):
            
            demo_review = np.array([sentiment[i]])
            demo_review_X_test = tvec.transform(demo_review)
            prediction = model.predict(demo_review_X_test)

            if prediction == 1:
                positive = positive + 1
                posrevs.append(sentiment[i])
            else:
                negative = negative + 1 
                negrevs.append(sentiment[i])
            
        TotalReviews = positive + negative
        fp = int(positive/TotalReviews * 100)
        fn = int(negative/TotalReviews * 100)
                
        lab1 = 'Positive Reviews : '+str(fp)+' %'
        lab2 = 'Negative Reviews : '+str(fn)+' %'
                
        status = [lab1, lab2]
         
        data = [positive, negative]
        
         
        # Creating plot
        fig = plt.figure(figsize =(10, 7))
        plt.pie(data, labels = status)
         
        # show plot
        filename = "Static/"+appname+".png"
        plt.savefig(filename)
        plt.show()        
        print("\nTotal positive Reviews : ", positive)
        print("Total negative Reviews : ", negative)
        print("\n")
        if positive > negative:
            pred = "positive"
        else:
            pred = "negative"
    pred = {"positive":posrevs, "negative":negrevs, "prediction":pred}
    
    return jsonify(pred)


if __name__ == '__main__':
    app.run()
