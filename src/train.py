from tqdm.autonotebook import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


def tokenize(sample, tokenizer):
    """Tokenize sample"""
    
    sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)
    
    return sample


def load_and_tokenize_dataset(dataset_name, model_name='bert-base-uncased', cache_dir='cache_dir/'):
    """
    Load dataset from the datasets library of HuggingFace.
    Tokenize and sort data by length.
    """
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Rename label column for tokenization purposes
    dataset = dataset.rename_column('label', 'labels')
    
    # Tokenize data
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    
    # sorting dataset
    for split in dataset.keys():
        dataset[split] = dataset[split].sort("length")
    
    return dataset, tokenizer, model_name


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
                                             batch_size=batch_size, 
                                             collate_fn=DataCollatorWithPadding(tokenizer))
    
    outputs_t = torch.Tensor().to(device)
    labels_t = torch.Tensor().to(device)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        batch = batch.to(device)
        outputs = model(batch)
        outputs_t = torch.cat([outputs_t, outputs], dim=0)

        labels = batch['labels']
        labels_t = torch.cat([labels_t, labels], dim=0)
    
    outputs_t = outputs_t.cpu().numpy()
    labels_t = labels_t.cpu().numpy()
    
    return outputs_t, labels_t


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
        