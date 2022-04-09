import numpy as np
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, DistilBertModel


def get_tfidf_features(dataset, dim=4000):
    """Compute tf-idf features and add it as a new field for the dataset"""
    
    vectorizer = TfidfVectorizer(max_features=dim)
    vectorizer.fit(dataset['train']['text'])
        
    for split in dataset.keys():
        X_tmp = vectorizer.transform(dataset[split]['text'])
        X_tmp = list(X_tmp.todense())
        X_tmp = [np.asarray(row).reshape(-1) for row in X_tmp]
        
        indices = dataset[split]._indices # ***
        dataset[split]._indices = None    # ***
        dataset[split] = dataset[split].add_column("additional_fts", X_tmp)
        dataset[split]._indices = indices # ***
        
        dataset[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'length', 'additional_fts'])
        dataset[split] = dataset[split].remove_columns("text")
        
    return dataset


class Embedding(nn.Module):
    """
    Implements an embedding layer.
    """

    def __init__(self, model_name='bert-base-uncased', pooling='mean', device=torch.device('cpu')):
        
        """
        Constructor

        Parameters
        ----------
        model_name : str
            Name of the BERT model and tokenizer.

        Attributes
        ----------
        model_name : str
            Name of the BERT model and tokenizer.
            The list of or possible models is provided here: https://huggingface.co/models
        pooling : str
            Pooling strategy to be applied, either 'mean' or 'cls'.
            For 'mean', the sentence embedding is the mean of the token embeddings.
            For 'cls', the sentence embedding is the embedding of the [CSL] token (as usual in BERT).
        device : torch.device
            GPU is available, CPU otherwise.
        """
        
        super(Embedding, self).__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        print('Model downloaded:', model_name)

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
            
            if self.pooling == 'mean':
                
                batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[0]
                # batch_emb = torch.mean(batch_emb, dim=1)
                # batch_emb = batch_emb.transpose(0, 1)
                # batch_emb = batch_emb[:, :, :] # removing CLS and/or SEP does not seem to improve
                batch_emb = torch.sum(batch_emb, dim=1).transpose(0, 1)
                batch_emb = torch.div(batch_emb, batch['length']).transpose(0, 1)
            
            elif self.pooling == 'cls':
            
                batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[1]

            return batch_emb

        
class BertTFIDF(nn.Module):
    """
    Impdements BERT + TF-IDF model:
    Concatenate BERT (or similar model) sentence embedding to most relevant TF-IDF features.
    """
    
    def __init__(self, model_name='bert-base-uncased', device=torch.device('cpu')):
        
        super(BertTFIDF, self).__init__()
        
        self.model_name = model_name
        self.device = device
        
        self.embedding = Embedding(model_name=self.model_name, device=self.device)
        
    def forward(self, batch):
        
        embedded_input = self.embedding(batch)
        additional_fts = batch['additional_fts']
        
        output = torch.cat([embedded_input, additional_fts], dim=1)
        
        return output