from torch import nn
import torch
import pathlib
import json

class RNN_old(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 3)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #x = x.permute(0,2,1)
        x = x.unsqueeze(1)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        #print(x[-1].shape)
        x = x.permute(1,0,2)[-1]
        x = self.linear(x)
        x = self.lsm(x)
        
        return x
    

class RNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, device, drop_prob=0.5,output_dim=3):
        super(RNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size

        self.device = device
    
        # embedding 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        hidden = (h0,c0)
        return hidden
    
    def save_model(self, run_name, config=None, vocab=None):
        # save model here
        model_name = run_name + "_model.pt"
            
        path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / model_name)
        #self.load_state_dict(torch.load('checkpoint.pt')) # load best model
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

def load_model(load_name, device):
    """
    Load model from trained_models folder
    """
    name_model = load_name + "_model.pt"
    name_dict = load_name + "_dict.json"
    name_vocab = load_name + "_vocab.json"

    path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_model)
    path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_dict)
    path_p_vocab = pathlib.Path(pathlib.Path.cwd() / "trained_models" / name_vocab)

    # load hyperparameters
    p_dict = open(path_p_dict)
    p_dict = json.load(p_dict)

    # load hyperparameters
    vocab = open(path_p_vocab)
    vocab = json.load(vocab)

    no_layers = p_dict['num_layers']
    embedding_dim = p_dict['embedding_dim']
    output_dim = 1
    hidden_dim = p_dict['hidden_size']
    vocab_size = len(vocab) + 1

    # initialize model with hyperparameters and load weights
    model = RNN(no_layers, vocab_size, hidden_dim, embedding_dim, device, drop_prob=0.0, output_dim=output_dim)
    model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

    return model, vocab