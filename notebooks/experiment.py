#!/usr/bin/env python
# coding: utf-8

# ********** #
# Librairies #
# ********** #

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import argparse

import time
import random
import pickle

import numpy as np
import torch

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report

import src.model as mod
from src.train import *



# ********* #
# Arguments #
# ********* #

parser = argparse.ArgumentParser()


# dataset
parser.add_argument("--dataset_name", type=str, default='imdb')
# 'imdb', 'yelp_polarity', 'yelp_review_full'
# 'trec', 'yahoo_answers_topics'
# 'ag_news', 'dbpedia_14'

# https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
def boolean_string(s):
    return s == 'True'

parser.add_argument('--sort', default=True, type=boolean_string)

# True, False


# Model 
parser.add_argument("--model_name", type=str, default='bert-base-uncased')
# only model for now

parser.add_argument("--tfidf_dim", type=int, default=1000)
# 1000, 2000, 3000, 4000

parser.add_argument("--batch_size", type=int, default=32)
# 32, 64, 128, 256, 512

parser.add_argument("--pooling", type=str, default='mean')
# 'mean', 'mean_std', 'cls', 'mean_cls', 'mean_std_cls'

parser.add_argument("--mode", type=str, default='default')
# 'default', 'bert_only', 'tfidf_only'


args = parser.parse_args()



# **************** #
# Device and seeds #
# **************** #

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seeds (torch generator seed missing?)
seed = 1979
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# results
results_folder = "/raid/home/jeremiec/Data/TextClassification"
results_file = os.path.join(results_folder, args.dataset_name) + '.pkl'
cache_dir = os.path.join(results_folder, 'cache_dir_' + args.dataset_name + '/')

# if os.path.exists(cache_dir):
#     os.system("rm -rf " + cache_dir)



# ********************** #
# Load and tokenize data #
# ********************** #

dataset, tokenizer, model_name = load_and_tokenize_dataset(args.dataset_name, 
                                                           model_name=args.model_name, 
                                                           sort=args.sort,
                                                           cache_dir=cache_dir)



# ***** #
# Model #
# ***** #

t0 = time.time()
dataset = mod.get_tfidf_features(dataset, dim=args.tfidf_dim)
t1 = time.time()

tfidf_time = t1 - t0

model = mod.BertTFIDF(model_name=model_name, pooling=args.pooling, mode=args.mode, device=device)



# ************* #
# Training loop #
# ************* #

# process train set
t0 = time.time()
X_train, y_train = process_dataset(dataset['train'], model, tokenizer, device, args.batch_size)
t1 = time.time()

training_time = t1 - t0

# process test set
X_test, y_test = process_dataset(dataset['test'], model, tokenizer, device, args.batch_size)

# alpha's loop
for alpha in [0.1, 1.0, 10.0, 100.0]:

    t0 = time.time()
    learning_algo = RidgeClassifier(alpha=alpha)
    learning_algo.fit(X_train, y_train)
    t1 = time.time()

    fitting_time = t1 - t0


    
    # ******* #
    # Results #
    # ******* #
    
    # NOTE: the processing of the test set is now put outside of the loop.
    # y_test, y_test_preds = predict(learning_algo, dataset, model, tokenizer, device, args.batch_size)
    y_test_preds = learning_algo.predict(X_test)

    test_results = classification_report(y_test, y_test_preds, digits=5, output_dict=True)

    torch.cuda.empty_cache()

    # save results
    if os.path.exists(results_file):
        with open(results_file, 'rb') as fh:
            results_d = pickle.load(fh)
    else:
        results_d = {}

    key = (args.sort, args.tfidf_dim, args.pooling, args.mode, args.batch_size, alpha)
    results_d[key] = (test_results, 
                      tfidf_time + training_time + fitting_time, 
                      "sort - tfidf_dim - pooling - mode - batch_size - alpha")

    with open(results_file, 'wb') as fh:
        pickle.dump(results_d, fh)
