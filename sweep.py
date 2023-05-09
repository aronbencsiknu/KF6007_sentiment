class SweepHandler():
    """
    This class is used to define the sweep parameters and the metric to be used for the sweep.
    """
    
    def __init__(self):

        self.metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }

        self.parameters_dict = {
        'num_layers': {
            'values': [1, 2, 3, 4]
            },
        'hidden_size': {
            'values': [32, 64, 128, 256, 512]
            },
        'embedding_dim': {
            'values': [32, 64, 128, 256, 512]
        },
        'dropout': {
                'values': [0.3, 0.4, 0.5]
            },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0005, 0.0001]
        },
        'optimizer':{
            'values' : ['Adam', 'SGD', 'AdamW']
        }
            
        }
        