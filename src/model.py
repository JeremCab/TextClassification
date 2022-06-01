import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, DistilBertModel


def tensorize_dataset(dataset):
    """Tensorize the data features and labels"""
        
    for split in dataset.keys():
        
        dataset[split].set_format(type='torch', columns=['input_ids', 
                                                         'attention_mask',
                                                         'labels', 
                                                         'length'])
#         # remove useless columns
#         for field in ['text', 'content', 'question_title']:
#             if field in dataset['train'].column_names:
#                 dataset[split] = dataset[split].remove_columns(field)
        
    return dataset


def get_tfidf_features(dataset, dim=3000):
    """Compute tf-idf features and add it as a new field for the dataset"""
    
    # get column of interest
    for field in ['text', 'content', 'question_title']:
        if field in dataset['train'].column_names:
            text_field = field
            break
    
    vectorizer = TfidfVectorizer(max_features=dim)
    vectorizer.fit(dataset['train'][text_field])
        
    for split in dataset.keys():
        # indices = dataset[split]._indices # XXX
        # dataset[split]._indices = None    # XXX
        X_tmp = vectorizer.transform(dataset[split][text_field])
        X_tmp = list(X_tmp.todense())
        X_tmp = [np.asarray(row).reshape(-1) for row in X_tmp]
        indices = dataset[split]._indices # *** XXX
        dataset[split]._indices = None    # *** XXX
        dataset[split] = dataset[split].add_column("additional_fts", X_tmp)
        dataset[split]._indices = indices # *** XXX
        
        dataset[split].set_format(type='torch', columns=['input_ids', 
                                                         'attention_mask',
                                                         'labels', 
                                                         'length', 
                                                         'additional_fts'])
        dataset[split] = dataset[split].remove_columns(text_field)
        
    return dataset


class Embedding(nn.Module):
    """
    Implements an embedding layer.
    """

    def __init__(self, 
                 model_name='bert-base-uncased', 
                 pooling='mean', 
                 device=torch.device('cpu'), 
                 results_dir='/raid/home/jeremiec/Data/TextClassification/'):
        
        """
        Constructor

        Attributes
        ----------
        model_name : str
            Name of the BERT model and tokenizer.
            The list of or possible models is provided here: https://huggingface.co/models
        pooling : str
            Pooling strategy to be applied: 'mean', 'mean_std', cls', 'mean_cls', or 'mean_std_cls'.
            For 'mean', the sentence embedding is the mean of the token embeddings.
            For 'mean_std', the sentence embedding is concatenation of the mean and the std of the token embeddings.
            For 'cls', the sentence embedding is the embedding of the [CSL] token (as usual in BERT).
            For 'mean_cls', the sentence embedding is the concatenation of the 'mean' and the 'cls' embeddings.
            For 'mean_std_cls', the sentence embedding is the concatenation of the 'mean', 'std' and the 'cls' embeddings.
        device : torch.device
            GPU is available, CPU otherwise.
        """
        
        super(Embedding, self).__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        if os.path.exists(os.path.join(results_dir, 'config.json')):
            self.model = BertModel.from_pretrained(results_dir)
        else:
            self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        print('Model downloaded:', model_name)

        
    def tensor_mean(self, batch, mode='custom', length_t=None):
        """Computes different kinds of means of batch embedding tensors.
        
        Parameters
        ----------
        batch: torch.Tensor
            3D tensor (batch size x max sentence length x embedding dim)
            BERT embedding of the batch of texts.
            
        Returns
        -------
        mean_batch : torch.Tensor
            2D tensor (batch size x embedding dim)
        """
        
        if mode == 'classic':
            
            mean_batch = torch.mean(batch, dim=1)
        
        elif mode == 'custom':
            
            batch_size = batch.shape[0]
            max_length = batch.shape[1]
            emb_dim = batch.shape[2]
            
            tmp_t = torch.arange(1, max_length + 1).to(self.device)
            tmp_t = tmp_t.expand(batch_size, max_length).transpose(0,1)
            mask = (tmp_t <= length_t).expand(emb_dim, max_length, batch_size).transpose(0, 2)
            
            batch = (batch * mask)[:, 1:, :] # remove [CLS]'s embeddings
            batch = batch.sum(dim=1).transpose(0, 1)
            
            mean_batch = torch.div(batch, length_t - 1).transpose(0, 1)

        return mean_batch

        
    def forward(self, batch):
        """
        Embeds a batch of token ids into a 3D tensor.
        If a GPU is available, the embedded batch is computed and put on the GPU.

        Parameters
        ----------
        batch: torch.Tensor
            2D tensor: batch of text to be embedded.
            Each sentence is represented as a vertical sequence of token ids.

        Returns
        -------
        batch_emb : torch.Tensor
            3D tensor (batch size x max sentence length x embedding dim)
            BERT embedding of the batch of texts.
        """
        
        with torch.no_grad():
            
            batch = batch.to(self.device)
            
            # DOES NOT IMPROVE THE RESULTS 
#             # New attention mask with last 1 element - correposnding to [SEP] token - removed.
#             # Accordingly, the mean pooling will not take the embedding of [SEP] into account.
#             last_indices = batch['length'] - 1
#             batch_size = batch['length'].shape[0]
#             indices = torch.tensor([range(batch_size), last_indices]).transpose(0,1)
#             # cf. https://discuss.pytorch.org/t/modify-array-with-list-of-indices/27739
#             batch['attention_mask'][indices[:, 0], indices[:, 1]] = 0
            
            if (self.pooling == 'mean') or (self.pooling == 'mean_cls'):
            
                batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[0]
                                
                batch_emb = self.tensor_mean(batch_emb, length_t=batch["length"])
                
                if self.pooling == 'mean_cls':

                    batch_emb_cls = self.model(batch["input_ids"], batch["attention_mask"])[1]
                    batch_emb = torch.cat([batch_emb, batch_emb_cls], dim=1)
                
            if (self.pooling == 'mean_std') or (self.pooling == 'mean_std_cls'):
                
                batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[0]
                
                batch_mean = self.tensor_mean(batch_emb, length_t=batch["length"])
                batch_std = torch.std(batch_emb, dim=1)  
                
                batch_emb = torch.cat([batch_mean, batch_std], dim=1)
                                
                if self.pooling == 'mean_std_cls':

                    batch_emb_cls = self.model(batch["input_ids"], batch["attention_mask"])[1]
                    batch_emb = torch.cat([batch_emb, batch_emb_cls], dim=1)
            
            elif self.pooling == 'cls':
            
                batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[1]
                            
            return batch_emb


class BertTFIDF(nn.Module):
    """
    Impdements BERT + TF-IDF model:
    Concatenate BERT (or similar model) sentence embedding to most relevant TF-IDF features.
    """
    
    def __init__(self, 
                 model_name='bert-base-uncased', 
                 pooling='mean',
                 mode='default', 
                 device=torch.device('cpu')):
        
        """
        Constructor

        Attributes
        ----------
        model_name : str
            Name of the BERT model and tokenizer.
            The list of or possible models is provided here: https://huggingface.co/models
        pooling : str
            Pooling strategy to be applied: 'mean', 'mean_std', 'cls', 'mean_cls', 'mean_std_cls'.
            For 'mean', the sentence embedding is the mean of the token embeddings.
            For 'std', the sentence embedding is the std of the token embeddings.
            For 'cls', the sentence embedding is the embedding of the [CSL] token (as usual in BERT).
            Other combinations refer to the concatenation of those embeddings.
        mode : str
            Three possible modes: 'default', 'bert_only', or 'tfidf_only'.
            The 'default' mode concatenates the BERT embedding and the TF-IDF features.
            The 'bert_only' only considers the BERT embedding (no TF-IDF features).
            The 'tfidf_only' only considers the TF-IDF features (no BERT embedding).
        device : torch.device
            GPU is available, CPU otherwise.
        """
        
        super(BertTFIDF, self).__init__()
        
        self.model_name = model_name
        self.pooling = pooling
        self.pooling_dim = 768 * (self.pooling.count('_') + 1)
        self.mode = mode
        self.device = device
        
        self.embedding = Embedding(model_name=self.model_name, 
                                   pooling=self.pooling, 
                                   device=self.device)
        
    def forward(self, batch):
        
        if self.mode == 'default':
            embedded_input = self.embedding(batch)
            additional_fts = batch['additional_fts']
            output = torch.cat([embedded_input, additional_fts], dim=1)
            
        elif self.mode == 'bert_only':
            output = self.embedding(batch)
        
        elif self.mode == 'tfidf_only':
            output = batch['additional_fts']
        
        return output
    