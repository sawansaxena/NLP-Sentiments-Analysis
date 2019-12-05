# -*- coding: utf-8 -*-
#import flask libraries
from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#load the model from disk
rf_classifier = pickle.load(open('nlp_model.pkl','rb'))
#Load count vector from disk
cv = pickle.load(open('transform.pkl','rb'))
#Load the vocabulary
words = pickle.load(open('vocabulary.pkl','rb'))

app_1 = Flask(__name__)

@app_1.route('/')
def home():
    return render_template('home.html')

@app_1.route('/predict',methods=['POST'])
def predict():
    
    if request.method=='POST':
        review = request.form['review']
        
        #review ='good movie'
        data = [review]
        #countVect = cv.transform(data).toarray()
        #countVect = cv.fit_transform(data).toarray()
        
        countVect = CountVectorizer(vocabulary=words)
        sentence = countVect.transform(data).toarray()
        
        review_prediction = rf_classifier.predict(sentence)
        #review_prediction = rf_classifier.predict(np.asarray(data))
        
        return render_template('result.html',prediction=review_prediction)
    
if __name__=='__main__':
    app_1.run(debug=True)

