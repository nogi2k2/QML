import yaml
import itertools
from pathlib import Path

def generate_config():
    """
    Generates a complete config.yaml file for all experiment combinations.
    """
    datasets = ['yelp', 'imdb', 'amazon']
    model_types = ['rnn', 'lstm', 'gru', 'transformer']
    embedding_types = ['basic', 'word2vec', 'bert']
    optimizers = ['Adam', 'SGD', 'RMSprop']

    experiments = []
    
    combinations = itertools.product(datasets, model_types, embedding_types, optimizers)

    for dataset, model_type, emb_type, optimizer in combinations:
        
        run_id = f"{dataset}_{model_type}_{emb_type}_{optimizer.lower()}"
        
        experiment = {
            'run_id': run_id,
            'dataset': dataset
        }
        
        embedding_config = {'name': emb_type}
        if emb_type == 'basic':
            embedding_config['embed_dim'] = 128
        elif emb_type == 'word2vec':
            embedding_config['embed_dim'] = 300
        elif emb_type == 'bert':
            embedding_config['model_name'] = 'distilbert-base-uncased'
        experiment['embedding'] = embedding_config

        model_config = {'name': model_type}
        if model_type == 'transformer':
            if emb_type == 'bert':
                model_config.update({'num_heads': 8, 'num_layers': 2, 'ff_dim': 1024, 'dropout': 0.1})
            elif emb_type == 'word2vec':
                model_config.update({'num_heads': 5, 'num_layers': 2, 'ff_dim': 512, 'dropout': 0.1})
            else: # basic
                model_config.update({'num_heads': 4, 'num_layers': 2, 'ff_dim': 512, 'dropout': 0.1})
        else: # RNN, LSTM, GRU
            if emb_type == 'bert':
                model_config.update({'hidden_size': 768, 'num_layers': 1, 'dropout': 0.3})
            else:
                model_config.update({'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3})
        experiment['model'] = model_config
        
        training_config = {
            'optimizer': optimizer,
            'loss_fn': 'BCEWithLogitsLoss',
            'epochs': 30  
        }
        if emb_type == 'bert':
            training_config['batch_size'] = 16 if model_type == 'transformer' else 32
            training_config['learning_rate'] = 0.001 if optimizer == 'SGD' else 0.0001
        else:
            training_config['batch_size'] = 32 if model_type == 'transformer' else 64
            training_config['learning_rate'] = 0.01 if optimizer == 'SGD' else 0.001
        
        training_config['scheduler'] = {'patience': 2, 'factor': 0.1}
        training_config['early_stopping'] = {'patience': 5, 'min_delta': 0.001}
        experiment['training'] = training_config

        experiments.append(experiment)

    final_config = {'experiments': experiments}
    
    output_path = Path(__file__).parent / 'generated_config.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(final_config, f, sort_keys=False, indent=2, default_flow_style=False)

    print(f"Successfully generated {len(experiments)} experiments.")
    print(f"File saved to: {output_path}")

if __name__ == '__main__':
    generate_config()
