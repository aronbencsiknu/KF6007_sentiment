import torch
import argparse

class Options(object):
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialized = False

    def add_args(self):

        self.argparser.add_argument("-sm", "--save_model", action='store_true', help="Save the model when finished training")
        self.argparser.add_argument("-lm", "--load_model", action='store_true', help="Skip training and load a model")
        self.argparser.add_argument("-ld", "--load_model_dict", action='store_true', help="Load a model hyperparameter dictionary")
        self.argparser.add_argument("-sw", "--sweep", action='store_true', help="Perform a hyperparameter sweep with Weights&Biases")
        self.argparser.add_argument("-sr", "--save_results", action='store_true', help="Perform a hyperparameter sweep with Weights&Biases")
        
        # ----------------------------
        
        self.argparser.add_argument("-wb", "--wandb_logging", action='store_true', help="enable logging to Weights&Biases")
        self.argparser.add_argument("--wandb_project", type=str, default="lstm_sentiment", help="Weights&Biases project name")
        self.argparser.add_argument("--wandb_entity", type=str, default="aronbencsik", help="Weights&Biases entity name")
        #self.argparser.add_argument("--wandb_key", type=str, default="aronbencsik", help="Weights&Biases entity name")
        self.argparser.add_argument("--load_name", type=str, default="final", help="Name of the model, hyperparameter dictionary and gain values to load")
        self.argparser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
        self.argparser.add_argument("--batch_size", type=int, default=128, help="Batch size")
        self.argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        self.argparser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
        self.argparser.add_argument("--num_layers", nargs='+',default=2, help='number of LSTM layers')
        self.argparser.add_argument("--hidden_size", type=int,default=256, help='Hidden layer size')
        self.argparser.add_argument("--embedding_dim", type=int,default=64, help='Hidden layer size')
        self.argparser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")

        # ----------------------------

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.add_args()
        self.opt = self.argparser.parse_args()

        self.opt.wandb_key = "edfb94e4b9dca47c397a343d2829e9af262d9e32"
        self.opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        return self.opt