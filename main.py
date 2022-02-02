'''
Multi-Layer Perceptron smart home command resolver.

Python 3.7.x

dataset / refs:
1) https://www.kaggle.com/bouweceunen/smart-home-commands-dataset/code
'''

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# read in data
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("data/dataset_ditto.csv")


# Data Preperation
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import itertools
import math

sentences = df['Sentence']
categories = df['Category']
subcategories = df['Subcategory']
actions = df['Action']

uniquecategories = list(set(categories))
uniquesubcategories = list(set(subcategories))
uniqueactions = list(set(actions))

mergesentences = list(itertools.chain.from_iterable([word_tokenize(sentence.lower()) for sentence in sentences]))
vocabulary = list(set(mergesentences))
# print(vocabulary)


# calculates how often the word appears in the sentence
def term_frequency(word, sentence):
    return sentence.split().count(word)

# calculates how often the word appears in the entire vocabulary
def document_frequency(word):
    return vocabulary.count(word)

# will make sure that unimportant words such as "and" that occur often will have lower weights
# log taken to avoid exploding of IDF with words such as 'is' that can occur a lot
def inverse_document_frequency(word):
    return math.log(len(vocabulary) / (document_frequency(word) + 1))

# get term frequency inverse document frequency value
def calculate_tfidf(word, sentence):
    return term_frequency(word, sentence) * inverse_document_frequency(word)

# get one-hot encoded vectors for the targets
def one_hot_class_vector(uniqueclasses, w):
    emptyvector = [0 for i in range(len(uniqueclasses))]
    emptyvector[uniqueclasses.index(w)] = 1
    return emptyvector

# get one-hot encoded vectors for the words
def one_hot_vector(w):
    emptyvector = [0 for i in range(len(vocabulary))]
    emptyvector[vocabulary.index(w)] = 1
    return emptyvector

# get one-hot encdoded sentence vector
def sentence_vector(sentence, tfidf=False):
    tokenizedlist = word_tokenize(sentence.lower())
    sentencevector = [0 for i in range(len(vocabulary))]
    count = 0

    for word in tokenizedlist:
        if word in vocabulary:
            count = count + 1
            if tfidf:
                sentencevector = [x + y for x, y in zip(sentencevector, [e * calculate_tfidf(word, sentence) for e in one_hot_vector(word)])] 
            else:
                sentencevector = [x + y for x, y in zip(sentencevector, one_hot_vector(word))]

    if count == 0:
        return sentencevector
    else:
        return [(el / count) for el in sentencevector]

# constructing sentence vectors
categoryvectors = [cv.index(1) for cv in [one_hot_class_vector(uniquecategories, w) for w in categories]]
subcategoryvectors = [cv.index(1) for cv in [one_hot_class_vector(uniquesubcategories, w) for w in subcategories]]
actionvectors = [cv.index(1) for cv in [one_hot_class_vector(uniqueactions, w) for w in actions]]
sentencevectors = [sentence_vector(sentence) for sentence in sentences]
sentencevectorstfidf = [sentence_vector(sentence, True) for sentence in sentences]

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(sentencevectors, categoryvectors, test_size=0.25, random_state=42)
X_train_cat_tfidf, X_test_cat_tfidf, y_train_cat_tfidf, y_test_cat_tfidf = train_test_split(sentencevectorstfidf, categoryvectors, test_size=0.25, random_state=42)
X_train_subcat, X_test_subcat, y_train_subcat, y_test_subcat = train_test_split(sentencevectors, subcategoryvectors, test_size=0.25, random_state=42)
X_train_action, X_test_action, y_train_action, y_test_action = train_test_split(sentencevectors, actionvectors, test_size=0.25, random_state=42)

# Training base model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import random
import pickle 

random.seed(2020)

def train_fit(model_name, model, X, y, X_test, y_test):
    model.fit(X, y)
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print(f"{model_name}: {accuracy}")
    return model

TRAIN = 1
import joblib
if TRAIN:
    
    mlp_max_iter_model_cat = MLPClassifier(max_iter=10000)
    mlp_max_iter_model_cat = train_fit("MLPClassifier", mlp_max_iter_model_cat, X_train_cat, y_train_cat, X_test_cat, y_test_cat)
    mlp_max_iter_model_subcat = MLPClassifier(max_iter=10000)
    mlp_max_iter_model_subcat = train_fit("MLPClassifier", mlp_max_iter_model_subcat, X_train_subcat, y_train_subcat, X_test_subcat, y_test_subcat)
    mlp_max_iter_model_action = MLPClassifier(max_iter=10000)
    mlp_max_iter_model_action = train_fit("MLPClassifier", mlp_max_iter_model_action, X_train_action, y_train_action, X_test_action, y_test_action)

    # save
    joblib.dump(mlp_max_iter_model_cat, "models/mlp_max_iter_model_cat.pkl")
    joblib.dump(mlp_max_iter_model_subcat, "models/mlp_max_iter_model_subcat.pkl")
    joblib.dump(mlp_max_iter_model_action, "models/mlp_max_iter_model_action.pkl")

else:
    # load
    mlp_max_iter_model_cat = MLPClassifier()
    mlp_max_iter_model_subcat = MLPClassifier()
    mlp_max_iter_model_action = MLPClassifier()

    mlp_max_iter_model_cat = joblib.load("models/mlp_max_iter_model_cat.pkl")
    mlp_max_iter_model_subcat = joblib.load("models/mlp_max_iter_model_subcat.pkl")
    mlp_max_iter_model_action = joblib.load("models/mlp_max_iter_model_action.pkl")

    mlp_max_iter_model_cat.fit(X_train_cat,y_train_cat)
    mlp_max_iter_model_subcat.fit(X_train_subcat, y_train_subcat)
    mlp_max_iter_model_action.fit(X_train_action, y_train_action)

def predict(model, classes, sentence):
    y_preds = model.predict([sentence_vector(sentence)])
    return classes[y_preds[0]]

sentence = "Turn the lights off in the kitchen."
def prompt(sentence):
    print(predict(mlp_max_iter_model_cat, uniquecategories, sentence))
    print(predict(mlp_max_iter_model_subcat, uniquesubcategories, sentence))
    print(predict(mlp_max_iter_model_action, uniqueactions, sentence))

prompt(sentence)