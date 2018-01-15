import seaborn as sns
sns.set_palette('Blues')
sns.set_context("notebook")
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, time
from dateutil.parser import parse
import glob
import functools as ft
from IPython.core.display import HTML
import requests
from bs4 import BeautifulSoup
import multiprocessing as mp
from itertools import product
from scipy.optimize import linear_sum_assignment
from nltk.metrics import edit_distance
import folium
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import networkx as nt
import os, codecs, string, random
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.feature_extraction.text import CountVectorizer
sns.set_palette('Blues')
sns.set_context("notebook")

def word_in_text(words, text):
    words = re.sub('\s+','\s*', ''.join([w + '|' for w in words[:-1]]) + words[-1])
    text = text.lower()
    match = re.search(words, text)
    if match:
        return True
    return False

def nice_bar_plot1(xlbl, data, n):
    
    fig, ax = plt.subplots()
    ax.set_title('Our ' + str(n) + ' ' + xlbl, fontsize=15, fontweight='bold')
    sns.barplot(data[:n].keys(), data[:n], ax=ax)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    
def folder_to_dataframe(folder):
    return pd.concat(map(pd.read_csv, glob.glob(DATA_FOLDER + folder + '/*.csv')))

# bar plot, double column
def barplotdouble(title, df, color, figsize=(15, 5), legend=None, **kwargs):
    f, ax = plt.subplots(figsize=figsize)
    df.plot(kind='bar', ax=ax, color=color, title=title, legend=True, fontsize=12, **kwargs)
    ax.set_xlabel(df.index.name, fontsize=12)
    ax.set_ylabel("ratio", fontsize=12)
    ax.set_xticklabels(df.index)
    ax.tick_params(axis='x', which='major', pad=15)
    if legend:
        ax.legend(legend)
    display(f)
    plt.close(f)

#plot("University mean ratios by region according to site 1's ranking",
     #uni_s1_regions.sort_values("pc_intl_students", ascending=False), color=['b','r'],
    #legend=['International Students ratio', 'Faculty/Student ratio'])
    
    
#This function displays a combination of a histogram of the data and a density function we could associate to it.
def densplot(columns, xlabel, title, axo):
    for i,v in enumerate(columns):
        sns.distplot(v, ax=axo, kde_kws={"label": i})
    axo.set_title(title, size=16)
    axo.set_xlabel(xlabel, fontsize=12)

def scatplot(xelem, yelem, xlabel, ylabel, title, polyfit=None):
    plt.scatter(xelem, yelem)
    if polyfit:
        plt.plot(np.unique(xelem), np.poly1d(np.polyfit(xelem, yelem, polyfit))(np.unique(xelem)), 'C2')
    plt.title(title)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()

# plot values and break between points
def plot_groups(df, col, breaks):
    for i in range(len(breaks)-1):
        points = df[df[col].between(breaks[i], breaks[i+1])][col]
        pp.plot(points, np.zeros_like(points), 'o')
    
    pp.plot(breaks, np.zeros_like(breaks), '|', markersize=30)
    pp.show()
    
#This function displays a boxplot of the data according to quantils if the violin value is set to False 
#and a violin plot otherwise
def boplot(data, title, xlabel, ylabel, violin, axo): #violin=True means a quantil plot
    if violin:
        sns.violinplot(data=data, ax=axo)
    else:
        sns.boxplot(data=data, ax=axo)
    axo.set_title(title, size=16)
    axo.set_xlabel(xlabel, fontsize=12)
    axo.set_ylabel(ylabel, fontsize=12)
    
#This function displays a bar chart of the different categories contained in data
def valCountBar(data, title, xlabel, ylabel, axo, color):
    colors = {"b":"#3274A1", "r":"#E1812C"}
    data.value_counts().plot(kind='bar', ax=axo, color=colors[color])
    axo.set_title(title, size=16)
    axo.set_xlabel(xlabel, fontsize=12)
    axo.set_ylabel(ylabel, fontsize=12)
    
    
def grid_search(n_estimators_list, max_depth_list):
    score=0
    final_depth=0
    final_estimator=0
    for depth in max_depth_list:
        for n_estim in n_estimators_list:
            classifier=RandomForestClassifier(max_depth=depth, n_estimators=n_estim, n_jobs=-1, random_state=None)
            classifier.fit(vectors_training, labels_training)
            prediction = classifier.predict(vectors_validation)
            scoring = metrics.accuracy_score(labels_validation, prediction)
            if scoring > score:
                score=scoring
                final_depth=depth
                final_estimator=n_estim
    return (score, final_depth, final_estimator)
# grid_search([50,100,200,500,1000,1500,2000,2500], [1,10,20,30])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#plt.figure(figsize=(20,10))
#plot_confusion_matrix(confusion, classes=newsgroups.target_names, normalize=True,
 #                     title='Normalized confusion matrix')
#plt.show()


# A basic (and crude) function to get rid of stopwords, punctuation, lower case, numbers
from nltk.corpus import stopwords
punctuation = string.punctuation+'“’—.”’“--,”' # pimp the list of punctuation to remove
def rem_stop(txt,stop_words=stopwords.words("english"),lower=True,punct=True):
    """
    Removes stopwords, punct and other things from a text, inc. numbers
    :param list txt: text tokens (list of str)
    :param list stop_words: stopwords to remove (list of str)
    :param bol lower: if to lowercase
    :param bol punct: if to rid punctuation
    """
    if lower and punct:
        return [t.lower() for t in txt if t.lower() not in stop_words and t.lower() not in punctuation and not t.isdigit()]
    elif lower:
        return [t.lower() for t in txt if t.lower() not in stop_words and not t.isdigit()]
    elif punct:
        return [t for t in txt if t.lower() not in stop_words and t.lower() not in punctuation and not t.isdigit()]
    return [t for t in txt if t.lower() not in stop_words and not t.isdigit()]


# support function to work with WordNet POS tags
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    """
    Cf. https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    :param treebank_tag: a tag from nltk.pos_tag treebank
    """
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

