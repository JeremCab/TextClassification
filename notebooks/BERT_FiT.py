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
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.data.data_collator import DataCollatorWithPadding

from src.train import *


# ********* #
# Arguments #
# ********* #

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", type=str, default='imdb')
# 'imdb', 'yelp_polarity', 'yelp_review_full'
# 'trec', 'yahoo_answers_topics'
# 'ag_news', 'dbpedia_14'
args = parser.parse_args()


# **************** #
# Global variables #
# **************** #

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

DATASET_NAME = args.dataset_name
# 'imdb', 'yelp_polarity', 'yelp_review_full'
# 'trec', 'yahoo_answers_topics'
# 'ag_news', 'dbpedia_14'

RESULTS_PATH = "/raid/home/jeremiec/Data/TextClassification/BERT_FiT/" + DATASET_NAME
RESULTS_FILE = os.path.join(RESULTS_PATH, DATASET_NAME) + '.pkl'
CACHE_DIR = os.path.join(RESULTS_PATH, 'cache_dir_' + DATASET_NAME + '/')

MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 24
NB_EPOCHS = 4


# ************ #
# Load dataset #
# ************ #

dataset, tokenizer, model_name = load_and_tokenize_dataset(dataset_name=DATASET_NAME, 
                                                           model_name=MODEL_NAME, 
                                                           sort=False,
                                                           cache_dir=CACHE_DIR)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = dataset["train"].shuffle(seed=42)
train_val_datasets = train_dataset.train_test_split(train_size=0.8)

train_dataset = train_val_datasets['train']
val_dataset = train_val_datasets['test']
test_dataset = dataset["test"].shuffle(seed=42)

# number of labels
num_labels = len(set(train_dataset['labels'].tolist()))


# ************* #
# BERT finetune #
# ************* #

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(device)


# ******** #
# Training #
# ******** #

training_args = TrainingArguments(
    
    # output
    output_dir=RESULTS_PATH,          
    
    # params
    num_train_epochs=NB_EPOCHS,               # nb of epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # cf. paper Sun et al.
    learning_rate=2e-5,                       # cf. paper Sun et al.
#     warmup_steps=500,                         # number of warmup steps for learning rate scheduler
    warmup_ratio=0.1,                         # cf. paper Sun et al.
    weight_decay=0.01,                        # strength of weight decay
    
#     # eval
    evaluation_strategy="steps",              # cf. paper Sun et al.
    eval_steps=50,                            # cf. paper Sun et al.
#     evaluation_strategy='no', # no more evaluation, takes time
    
    # log
    logging_dir=RESULTS_PATH+'/logs',  
    logging_strategy='steps',
    logging_steps=50, # 10? same as eval_steps
    
    # save
    # save_strategy='epoch',
    # save_strategy='steps',
    # load_best_model_at_end=False
    load_best_model_at_end=True               # cf. paper Sun et al.
)

def compute_metrics(p):
    
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    
    return {"val_accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

results = trainer.train()

training_time = results.metrics["train_runtime"]

trainer.save_model(os.path.join(RESULTS_PATH, 'best_model'))


# ******* #
# Results #
# ******* #

results_d = {}

# # finetuned model
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
# model.load_state_dict(torch.load(RESULTS_PATH + 'best_model-0/pytorch_model.bin'))
# model.to(device)
model.eval()

# compute test acc
test_trainer = Trainer(model, data_collator=DataCollatorWithPadding(tokenizer))
raw_preds, labels, _ = test_trainer.predict(test_dataset)
preds = np.argmax(raw_preds, axis=1)
test_acc = accuracy_score(y_true=labels, y_pred=preds)
test_results = classification_report(labels, preds, digits=4, output_dict=True)
print(test_results)

# save acc, classification table and training time
results_d['test_accuracy'] = test_acc # best model evaluation only
results_d['test_classification-report'] = test_results
results_d['training_time'] = training_time

with open(RESULTS_FILE, 'wb') as fh:
    pickle.dump(results_d, fh)
