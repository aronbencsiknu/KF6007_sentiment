class SweepHandler():
    """
    This class is used to define the sweep parameters and the metric to be used for the sweep.
    """
    
    def __init__(self):

        self.metric = {
        'name': 'val_loss',
        'goal': 'minimize'   
        }

        self.parameters_dict = {
        'num_layers': {
            'values': [1, 2, 3]
            },
        'hidden_size': {
            'values': [64, 128, 256]
            },
        'embedding_dim': {
            'values': [64, 128, 256]
        },
        'dropout': {
                'values': [0.3, 0.4, 0.5, 0.6]
            },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0005, 0.0001]
        },
        'optimizer':{
            'values' : ['Adam', 'SGD', 'AdamW']
        }
            
        }
        