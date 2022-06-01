from tqdm.autonotebook import tqdm

import torch
import gc
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import DataCollatorWithPadding, DataCollator


def tokenize(sample, tokenizer):
    """Tokenize sample"""
    
    # get field to 
    for field in ['text', 'content', 'question_title']:
        
        if field in sample.keys():
    
            sample = tokenizer(sample[field], truncation=True, padding=False, return_length=True)
    
    return sample


def load_and_tokenize_dataset(dataset_name, 
                              model_name='bert-base-uncased', 
                              sort=False, 
                              eval_mode = False,
                              cache_dir='cache_dir/'):
    """
    Load dataset from the datasets library of HuggingFace.
    Tokenize and sort data by length.
    """
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
    # Rename label column for tokenization purposes
    for label in ['label', 'label-coarse', 'topic']:
        if label in dataset.column_names['train']:
            dataset = dataset.rename_column(label, 'labels')
    
    # Tokenize data
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    
    # sorting dataset
    for split in dataset.keys():
        if sort:
            dataset[split] = dataset[split].sort("length")#.flatten_indices()
#         else:
#             dataset[split] = dataset[split]#.flatten_indices() # already shuffled
    
    return dataset, tokenizer, model_name



def empty_cuda_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del(obj)
        except: pass



def process_dataset(dataset, model, tokenizer, device=torch.device('cpu'), batch_size=256):
    """
    Pass a dataset into a model.
    
    Parameters
    ----------
    dataset : datasets.arrow_dataset.Dataset
        Dataset to be processed
    model : __main__.BertTFIDF
        Model instance of the BertTFIDF class
    
    Returns
    -------
    outputs_t, labels_t : torch.Tensor, torch.Tensor
        Tuple of outputs and labels resulting from passing the dataset into the model.
    """
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             # shuffle=False,
                                             drop_last=False,
                                             batch_size=batch_size, 
                                             collate_fn=DataCollatorWithPadding(tokenizer)
                                            )
    
    # Accumulate tensors a long as enough CUDA memory is available.
    # When CUDA memory is full:
    # 1. convert tensor into numpy
    # 2. empty CUDA cache
    # 3. start new accumulation
    
    # numpy tensors (for micro accumlation steps)
    if 'additional_fts' in dataset.column_names:
        dim = model.pooling_dim + dataset[0]['additional_fts'].shape[0]
    else:
        dim = model.pooling_dim
    outputs_np = np.empty(shape=(0, dim))
    labels_np = np.empty(shape=(0))
    
    # torch tensors (for macro accumlation steps)
    outputs_t = torch.empty(size=(0, dim), device=device)
    labels_t = torch.empty(size=(0,), device=device)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        try: # enough CUDA memory
                        
            # micro-accumulate embeddings (torch) and labels (torch)
            batch = batch.to(device)
            outputs_t = torch.cat([outputs_t, model(batch)], dim=0)
            labels_t = torch.cat([labels_t, batch['labels']], dim=0)
            
        except RuntimeError: # not enough CUDA memory
            
            print("Macro accumulation step...")
            # print(torch.cuda.memory_summary()) # XXX
                        
            # macro-accumulate embeddings and labels (numpy)
            outputs_np = np.concatenate([outputs_t.cpu().numpy(), outputs_np], axis=0)
            labels_np = np.concatenate([labels_t.cpu().numpy(), labels_np], axis=0)
            
            # free CUDA memory
            del outputs_t, labels_t
            torch.cuda.empty_cache()
            
            # new empty embeddings (torch) and labels (torch)
            outputs_t = torch.empty(size=(0, dim), device=device)
            labels_t = torch.empty(size=(0,), device=device)
    
            # micro-accumulate embeddings (torch) and labels (torch)
            batch = batch.to(device)
            # print(torch.cuda.memory_summary()) # XXX
            outputs_t = torch.cat([outputs_t, model(batch)], dim=0)
            labels_t = torch.cat([labels_t, batch['labels']], dim=0)
            
    # accumulate final embeddings and labels (numpy)
    outputs_np = np.concatenate([outputs_t.cpu().numpy(), outputs_np], axis=0)
    labels_np = np.concatenate([labels_t.cpu().numpy(), labels_np], axis=0)
    
# ORIGINAL
#         batch = batch.to(device)
#         outputs = model(batch)
#         outputs_t = torch.cat([outputs_t, outputs], dim=0)

#         labels = batch['labels']
#         labels_t = torch.cat([labels_t, labels], dim=0)
    
#     outputs_t = outputs_t.cpu().numpy()
#     labels_t = labels_t.cpu().numpy()
    
#     print(outputs_t.shape)
    
#     return outputs_t, labels_t
    
    return outputs_np, labels_np


def train_learning_algo(learning_algo, dataset, model, tokenizer, 
                        device=torch.device('cpu'), batch_size=256):
    """
    Train the learning algorithm associated with the supervised pb (X_train, y_train).
    More specifically, after the train set is passed through the model (EMB + POOL + ADD_TF-IDF), 
    a vector of X_train of text emeddings concatenated with TF-IDF features is obtained.
    Then, the association between X_train and y_train is learned by means of a learning algorithm.
    """
    
    X_train, y_train = process_dataset(dataset['train'], model, tokenizer,
                                       device=device, batch_size=batch_size)
    
    # fit sklearn learning algo
    learning_algo.fit(X_train, y_train)
    
    return learning_algo


# Note:
# In the last version of experiment.py, this function is not used.
# Instead, the processing of the test set (i.e., process_dataset(dataset['test'],...))
# and the prediction step (i.e., learning_algo.predict(X_test)) are performed separately.
# In this way, the processing an be performed only once, while several learning algos 
# (alpha's loop) can be tested for predictions (cf. experiment.py for further details).
def predict(learning_algo, dataset, model, tokenizer, 
            device=torch.device('cpu'), batch_size=256, mode='test'):
    """
    Compute train and test predictions for the dataset.
    """
    
    X_test, y_test = process_dataset(dataset['test'], model, tokenizer,
                                       device=device, batch_size=batch_size)
    y_test_preds = learning_algo.predict(X_test)
    
    if mode == 'train_test':
        X_train, y_train = process_dataset(dataset['train'], model, tokenizer,
                                       device=device, batch_size=batch_size)
        y_train_preds = learning_algo.predict(X_train)
    
        return  y_train, y_train_preds, y_test, y_test_preds
    
    elif mode=='test':
        
        return y_test, y_test_preds
        