import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from gensim.models import KeyedVectors
import numpy as np

class StaticEmbedding(nn.Module):
    """
    A wrapper for torch.nn.Embedding for static word vectors like Word2Vec or GloVe.
    """
    def __init__(self, vocab_size, embed_dim, pretrained_vectors=None, padding_idx=0, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_vectors is not None:
            print("Loading pre-trained vectors...")
            self.embedding.weight.data.copy_(pretrained_vectors)
        
        if freeze:
            print("Freezing embedding layer.")
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)


class BertEmbedding(nn.Module):
    """
    A wrapper for BERT models from the transformers library.
    Generates contextual embeddings.
    """
    def __init__(self, model_name='bert-base-uncased', freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            print(f"Freezing {model_name} layers.")
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self, raw_text_list):
        inputs = self.tokenizer(
            raw_text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        inputs = {k: v.to(self.bert_model.device) for k, v in inputs.items()}
        
        outputs = self.bert_model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

def get_embedding_layer(embedding_config: dict, vocab, device):
    """
    Factory function to get the specified embedding layer.
    
    Args:
        embedding_config (dict): Configuration for the embedding layer.
        vocab (Vocabulary): The vocabulary object.
        device: The torch device ('cuda' or 'cpu').
        
    Returns:
        An instantiated nn.Module for embeddings.
    """
    name = embedding_config.get('name')
    freeze = embedding_config.get('freeze', True)

    if name == 'bert':
        model_name = embedding_config.get('model_name', 'bert-base-uncased')
        print(f"Initializing BERT embedding layer with model: {model_name}")
        return BertEmbedding(model_name=model_name, freeze=freeze).to(device)
    
    elif name == 'word2vec' or name == 'glove':
        print("Static embeddings like Word2Vec/GloVe are not fully implemented yet.")
        print("Falling back to a basic trainable embedding layer.")
        embed_dim = embedding_config.get('embed_dim', 300)
        vocab_size = len(vocab)
        return StaticEmbedding(vocab_size, embed_dim, freeze=False).to(device)

    elif name == 'basic':
        print("Initializing basic trainable embedding layer.")
        embed_dim = embedding_config.get('embed_dim', 100)
        vocab_size = len(vocab)
        return StaticEmbedding(vocab_size, embed_dim, freeze=freeze).to(device)

    else:
        raise ValueError(f"Unknown embedding type: {name}")

if __name__ == '__main__':
    from data_loader import get_data_loaders
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    project_root = Path(__file__).parent.parent 
    data_directory = project_root / 'data'
    
    print("\n--- Testing Basic Embedding ---")
    _, _, _, vocab = get_data_loaders('yelp', data_directory)
    basic_config = {'name': 'basic', 'embed_dim': 100, 'freeze': False}
    basic_embedding_layer = get_embedding_layer(basic_config, vocab, device)
    print(basic_embedding_layer)

    print("\n--- Testing BERT Embedding ---")
    bert_config = {'name': 'bert', 'model_name': 'distilbert-base-uncased', 'freeze': True}
    bert_embedding_layer = get_embedding_layer(bert_config, vocab=None, device=device) 
    print(bert_embedding_layer)

    sample_text = ["This is a great movie!", "I did not enjoy the book."]
    with torch.no_grad():
        bert_output = bert_embedding_layer(sample_text)
    print(f"BERT output shape for 2 sentences: {bert_output.shape}") 