import torch
import argparse

class Options(object):
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialized = False

    def add_args(self):

        self.argparser.add_argument("-sm", "--save_model", action='store_true', help="Save the model when finished training")
        self.argparser.add_argument("-sw", "--sweep", action='store_true', help="Perform a hyperparameter sweep with Weights&Biases")
        self.argparser.add_argument("-t", "--test", action='store_true', help="Test the model")

        # ----------------------------
        
        self.argparser.add_argument("-wb", "--wandb_logging", action='store_true', help="Enable logging to Weights&Biases")
        self.argparser.add_argument("--wandb_project", type=str, default="lstm_sentiment", help="Weights&Biases project name")
        self.argparser.add_argument("--wandb_entity", type=str, default="aronbencsik", help="Weights&Biases entity name")
        self.argparser.add_argument("--wandb_key", type=str, default="aronbencsik", help="Weights&Biases entity name")
        self.argparser.add_argument("--load_name", type=str, default="final", help="Name of the model, hyperparameter dictionary and gain values to load")
        self.argparser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
        self.argparser.add_argument("--batch_size", type=int, default=128, help="Batch size")
        self.argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate of the optimizer")
        self.argparser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
        self.argparser.add_argument("--num_layers", type=int, default=2, help='Number of LSTM layers')
        self.argparser.add_argument("--hidden_size", type=int, default=256, help='Hidden layer size')
        self.argparser.add_argument("--patience", type=int,default=5, help='Patience for early stopping')
        self.argparser.add_argument("--embedding_dim", type=int,default=64, help='Embedding layer size')
        self.argparser.add_argument("--pad_length", type=int,default=500, help='Padding length for the input sequences')
        self.argparser.add_argument("--vocab_length", type=int,default=1000, help='Vocabulary length')
        self.argparser.add_argument("--num_classes", type=int,default=3, help='Number of classes')
        self.argparser.add_argument("--optimizer", type=str, default="Adam", choices=("Adam", "AdamW", "SGD"), help="Optimizer to use")

        # ----------------------------

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.add_args()
        self.opt = self.argparser.parse_args()

        self.opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.opt.test:
            self.opt.device = torch.device("cpu")
        
        if self.opt.sweep:
            self.opt.patience = self.opt.num_epochs
        
        return self.opt