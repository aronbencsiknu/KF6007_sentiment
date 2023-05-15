import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from collections import Counter
import re
import torch
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def load_data():
    '''
    Load data and targets from csv
    '''

    data = pd.read_csv('data/twitter.csv')
    X,y = data['text'].fillna('').values,data['sentiment'].fillna(0).values # replace NaN with empty string for data and 0 for target

    return X,y

def preprocess_string(s):
    '''
    Sanitize string with regex
    '''

    s = re.sub(r"[^\w\s]", '', s) # remove non-alphanumeric characters
    s = re.sub(r"\s+", '', s) # remove whitespaces
    s = re.sub(r"\d", '', s) # ove numbers

    return s

def generate_vocabulary(X, vocab_len=1000):
    '''
    Generate word to token mapping
    '''

    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in X:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:vocab_len]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    return onehot_dict

def class_to_idx(y):
    '''
    Map class to index
    '''

    classes = [-1, 0, 1]
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    y_final = [class_to_idx[label] for label in y]

    return y_final

def idx_to_class(y):
    '''
    Map index to class
    '''
    
    classes = ["Negative", "Neutral", "Positive"]
    idx_to_class = {i:j for i, j in enumerate(classes)}
    y_final = [idx_to_class[label] for label in [y]]

    return y_final

def tokenize(X, y, vocab):
    '''
    Tokenize dataset
    '''

    # class to index and index to class mappings
    """idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}"""
    X_final = []
    for sent in X:
            # tokenize words, if not found add <unk> token <unk> is vocab length + 1
            X_final.append([vocab[preprocess_string(word)] if preprocess_string(word) in vocab.keys() else len(vocab)+1 
               for word in sent.lower().split()])
            
    y_final = class_to_idx(y)
    
    return np.array(X_final), np.array(y_final)

def pad_items(sentences, seq_len):
    '''
    Pad or clip items to match specified sequence length
    '''

    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def get_class_weights(X, y, num_classes):
    '''
    Generate class weights for loss function
    '''

    class_counts = [0] * num_classes
    for i in range(len(y)):
        class_counts[int(y[i])] += 1

    print("\nClass weight balancing for training.")

    w = [len(X) / (num_classes * n_curr_class) for n_curr_class in class_counts]
    for i, j in zip(w, [0, 1, 2]):
        print(f"{j} -> {class_counts[j]} -> {i}")

    return torch.tensor(w, dtype=torch.float32)