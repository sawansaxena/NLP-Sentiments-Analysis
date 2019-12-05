#import all required ;ibraries

import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #For stemming
from nltk.stem import WordNetLemmatizer #For Lemmetization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load the review dataset for training
df_reviews = pd.read_csv(r'E:\Learning - 2019\Python ML Training\Python Notes\Class Notes\NLP\Restaurant_Reviews.tsv', delimiter = '\t')
df_reviews.head()

#get the stopwords for english
stopWords = stopwords.words('english')

#remove no and not from list of stopwords
stopWords.remove('no')
stopWords.remove('not')

#Create objects of stemmer and lemmatizer
stem_obj = PorterStemmer()
lem_obj = WordNetLemmatizer()


corpus = []

#Text Preprocessing including stemming
'''
for i in range(0,len(df_reviews)):
    review = re.sub('[^a-zA-Z]',' ',df_reviews.iloc[i,0]) #remove all characters except A-Z
    review = review.lower().split() #convert all text into lower case and split by space
    
    #Apply stemming and lemmatization of all words in review
    review_updated = [lem_obj.lemmatize(lem_obj.lemmatize(stem_obj.stem(word),pos='v'),pos='a')\
                     for word in review\
                     if word not in stopWords]
    review_updated = ' '.join(review_updated)
    corpus.append(review_updated)
'''
    
    
#Text Preprocessing without stemming    
for i in range(0,len(df_reviews)):
    review = re.sub('[^a-zA-Z]',' ',df_reviews.iloc[i,0]) #remove all characters except A-Z
    review = review.lower().split() #convert all text into lower case and split by space
    
    #Apply stemming and lemmatization of all words in review
    review_updated = [lem_obj.lemmatize(lem_obj.lemmatize(word,pos='v'),pos='a')\
                     for word in review\
                     if word not in stopWords]
    review_updated = ' '.join(review_updated)
    corpus.append(review_updated)
    
    
#initiliaze Countvecorizer
countVector = CountVectorizer(max_features=2000)

#Save the Count Vector as tranform.pkl file 
#which will be used to convert text into count vector
pickle.dump(countVector,open('transform.pkl','wb'))

X = countVector.fit_transform(corpus).toarray()

#Save the vocabulary in form of words
words = countVector.get_feature_names()
pickle.dump(words,open('vocabulary.pkl','wb'))

#X = countVector.fit_transform(corpus)
y = df_reviews.iloc[:,1].values

#split the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) 

#Create Random forest model. Train and test
rf_classifer = RandomForestClassifier(n_estimators=150,min_samples_split=5)
rf_classifer.fit(X_train,y_train)
rf_classifer.score(X_test,y_test)

#Save the model pickle file to be used for prediction
pickle.dump(rf_classifer,open('nlp_model.pkl','wb'))
   



