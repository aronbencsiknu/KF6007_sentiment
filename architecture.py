from torch import nn
import torch
import pathlib
import json

class RNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, device, drop_prob=0.5,output_dim=3):
        super(RNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.device = device
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.lsm = nn.LogSoftmax(dim=1)
        
    def forward(self,x,hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x)
        x = torch.permute(x, (1,0,2))
        x = x[-1]
        x = self.dropout(x)
        x = self.fc(x)
        out = self.lsm(x)

        return out, hidden
        
    def init_hidden(self, batch_size):
        '''Initializes hidden states'''

        hc_0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)

        return (hc_0,hc_0)
    
    def save_model(self, run_name, config=None, vocab=None):
        # save model here
        model_name = run_name + "_model.pt"
            
        path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / model_name)
        torch.save(self.state_dict(), path_model)

        if config is not None:
            # save model dict here
            config_dict = config.__dict__
            config_dict = config_dict["_items"]
            #config_dict.update([('vocab_size', 1001)])

            dict_name = run_name + "_dict.json"
            path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / dict_name)
            
            with open(path_p_dict, 'w') as f:
                json.dump(config_dict, f)

        if vocab is not None:

            vocab_name = run_name + "_vocab.json"
            path_p_vocab = pathlib.Path(pathlib.Path.cwd() / "trained_models" / vocab_name)
            
            with open(path_p_vocab, 'w') as f:
                json.dump(vocab, f)

def load_model(load_name, device, num_classes, config=None):
    """
    Load model from trained_models folder
    """
    name_model = load_name + "_model.pt"
    name_vocab = load_name + "_vocab.json"

    path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_model)
    path_p_vocab = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_vocab)

    if config is None:
        name_dict = load_name + "_dict.json"
        path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_dict)
        # load hyperparameters
        p_dict = open(path_p_dict)
        p_dict = json.load(p_dict)

        no_layers = p_dict['num_layers']
        embedding_dim = p_dict['embedding_dim']
        hidden_dim = p_dict['hidden_size']

    else:
        no_layers = config.num_layers
        embedding_dim = config.embedding_dim
        hidden_dim = config.hidden_size

    # load hyperparameters
    vocab = open(path_p_vocab)
    vocab = json.load(vocab)
    output_dim = num_classes
    
    vocab_size = len(vocab) + 2

    # initialize model with hyperparameters and load weights
    model = RNN(no_layers, vocab_size, hidden_dim, embedding_dim, device, drop_prob=0.0, output_dim=output_dim)
    model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

    return model, vocab