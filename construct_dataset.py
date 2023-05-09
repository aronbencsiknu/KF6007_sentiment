import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from progress.bar import ShadyBar

"""class CustomDataset(Dataset):
    def __init__(self, data, targets, num_classes=3):

        self.data=data
        self.targets=targets
        self.db=[]
        self.n_classes = num_classes

        # encode data to spikes using the defined encoding method

        for i in range(np.shape(self.data)[0]):
            item = torch.FloatTensor(self.data[i])
            target=self.targets[i]
            self.db.append([item,target])
        
        self.n_samples_per_class = self.get_class_counts()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx][0]
        label = self.db[idx][1]
        return data, label
    
    def get_class_counts(self):
        class_weights = [0] * 3
        for i in range(len(self.targets)):
            class_weights[self.targets[i]] += 1

        return class_weights
    
    def get_class_weights(self):
        print("\nClass weight balancing for training.")


        w = [len(self.db) / (self.n_classes * n_curr_class) for n_curr_class in self.n_samples_per_class]
        for i, j in zip(w, [0, 1]):
            print(f"{j}\t-> {i}")

        return torch.tensor(w, dtype=torch.float32)"""
    

def load_data():
    data = pd.read_csv('data/IMDB/IMDB Dataset.csv')
    
    X,y = data['review'].values,data['sentiment'].values

    return X,y

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def generate_vocabulary(X):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    print()
    bar = ShadyBar("Tokenizing", max=len(X))
    for sent in X:
        bar.next()
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
    bar.finish()
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    return onehot_dict

def tokenize(X, y, vocab):
    """word_list = []

    stop_words = set(stopwords.words('english')) 
    print()
    bar = ShadyBar("Tokenizing", max=len(X))
    for sent in X:
        bar.next()
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
    bar.finish()
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}"""
    
    onehot_dict = vocab

    # tockenize
    X_final = []
    for sent in X:
            X_final.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
            
    y_final = [1 if label =='positive' else 0 for label in y]
    return np.array(X_final), np.array(y_final)

def pad_items(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features