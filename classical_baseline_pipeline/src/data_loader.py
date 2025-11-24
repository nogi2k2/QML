import torch
import pandas as pd
import re
import spacy
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Spacy 'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

class Vocabulary:
    """
    Handles building the word-to-index and index-to-word mappings.
    """
    def __init__(self, freq_threshold=3):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        """Simple tokenizer that cleans and tokenizes text."""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
        text = text.lower() 
        return [tok.text for tok in nlp.tokenizer(text) if not tok.is_stop and not tok.is_punct]

    def build_vocabulary(self, sentence_list):
        """Builds the vocabulary from a list of sentences."""
        frequencies = {}
        idx = 4 

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """Converts a text string to a sequence of indices."""
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for loading and processing the sentiment data.
    """
    def __init__(self, data_file_path: Path, vocab: Vocabulary = None, build_vocab: bool = False):
        self.df = pd.read_csv(data_file_path, sep='\t', header=None, names=['text', 'label'])
        self.raw_texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

        if build_vocab:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(self.raw_texts)
        elif vocab:
            self.vocab = vocab
        else:
            raise ValueError("Either a vocabulary must be provided or build_vocab must be set to True.")
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        raw_text = self.raw_texts[index]
        label = self.labels[index]
        numericalized_text = self.vocab.numericalize(raw_text)
        
        return raw_text, torch.tensor(numericalized_text, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def collate_batch(batch):
    """
    Collates data samples into batches.
    - Pads numericalized text sequences to the same length.
    - Gathers raw text, labels, and sequence lengths.
    """
    raw_text_list, numerical_text_list, label_list, lengths_list = [], [], [], []
    for raw_text, numerical_text, label in batch:
        raw_text_list.append(raw_text)
        label_list.append(label)
        numerical_text_list.append(numerical_text)
        lengths_list.append(len(numerical_text))

    padded_texts = pad_sequence(numerical_text_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list)
    lengths = torch.tensor(lengths_list)
    
    return raw_text_list, padded_texts, labels, lengths

def get_data_loaders(dataset_name: str, data_dir: Path, batch_size: int = 32, val_split: float = 0.1, test_split: float = 0.1):
    """
    The main function to create train, validation, and test dataloaders.
    
    Args:
        dataset_name (str): One of 'imdb', 'amazon', or 'yelp'.
        data_dir (Path): The directory containing the data files.
        batch_size (int): The size of each batch.
        val_split (float): The fraction of data to use for validation.
        test_split (float): The fraction of data to use for testing.
        
    Returns:
        tuple: Contains train_loader, val_loader, test_loader, and the vocabulary object.
    """
    file_path = data_dir / f"{dataset_name}_labelled.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    full_dataset = SentimentDataset(file_path, build_vocab=True)
    vocab = full_dataset.vocab

    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, val_loader, test_loader, vocab

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent 
    data_directory = project_root / 'data'
    
    print(f"Looking for data in: {data_directory}")
    
    try:
        train_loader, val_loader, test_loader, vocab = get_data_loaders(
            dataset_name='imdb',
            data_dir=data_directory,
            batch_size=4
        )
        
        print(f"\nSuccessfully created data loaders for IMDb.")
        print(f"Vocabulary size: {len(vocab)}")

        raw_texts, padded_sequences, labels, lengths = next(iter(train_loader))
        
        print("\n--- Inspecting First Batch ---")
        print(f"Raw Texts: {raw_texts}")
        print(f"Padded Sequences (shape): {padded_sequences.shape}")
        print(f"Padded Sequences (sample): \n{padded_sequences}")
        print(f"Labels: {labels}")
        print(f"Original Lengths: {lengths}")
        print("--------------------------\n")
        
    except FileNotFoundError as e:
        print(e)
        print("\nPlease ensure the 'data' directory is in the root of your 'classical_baseline_pipeline' folder.")