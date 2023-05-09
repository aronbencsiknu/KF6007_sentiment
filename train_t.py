import pathlib
import wandb
import torch
import json

from options import Options
from model import RNN
from metrics import Metrics
from torch.utils.data import TensorDataset, DataLoader
import construct_dataset
from sklearn.model_selection import train_test_split
from early_stopping import EarlyStopping
from torch import nn
import numpy as np
from sweep import SweepHandler
from progress.bar import ShadyBar

X,y = construct_dataset.load_data()
print("done loading")
#X, y, vocab = construct_dataset.tokenize(X, y)
print("done tokenizing")
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

vocab = construct_dataset.generate_vocabulary(x_train)
x_train, y_train = construct_dataset.tokenize(x_train, y_train, vocab)
x_test,y_test = construct_dataset.tokenize(x_test, y_test, vocab)

x_train_pad = construct_dataset.pad_items(x_train,500)
x_test_pad = construct_dataset.pad_items(x_test,500)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

opt = Options().parse()
metrics = Metrics()

if opt.wandb_logging or opt.sweep:
  print("\n#########################################")
  print("#########################################\n")
  print("!!IMPORTANT!! You need to create a WandB account and paste your authorization key in options.py to use this.")
  print("\n#########################################")
  print("#########################################\n")

# init wandb logging if specified
if opt.wandb_logging:
  key=opt.wandb_key
  wandb.login(key=key)
  wandb.init(project=opt.wandb_project, 
              entity=opt.wandb_entity, 
              group=opt.run_name,
              settings=wandb.Settings(start_method="thread"),
              config=opt)

# init sweep handler if specified
if opt.sweep:
  sweep_handler = SweepHandler()

  key=opt.wandb_key
  wandb.login(key=key)

  sweep_config = {'method': 'random'}
  sweep_config['metric'] = sweep_handler.metric
  sweep_config['parameters'] = sweep_handler.parameters_dict

  sweep_id = wandb.sweep(sweep_config, project="lstm_sweeps_test")

def train_epoch(model, optimizer, loss_fn, train_loader, epoch, logging_index):
    """
    Train the network for one epoch
    """
    clip = 5
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    print()
    title = "Epoch: " + str(epoch)
    bar = ShadyBar(title, max=len(train_loader))
    for inputs, labels in train_loader:
        bar.next()
        
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = loss_fn(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = metrics.acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    bar.finish()
    return model, logging_index, (sum(train_losses)/len(train_losses)), (train_acc/len(train_loader.dataset))

def sweep_train(config=None):
    """
    Hyperparameter sweep train function. 
    Called by wandb agent.
    """
    
    # Initialize a new wandb run
    with wandb.init(config=config):
        
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        model, optimizer, early_stopping, loss_fn = initialize_training(config=config)
       
        max_acc = 0
        logging_index_train = 0
        logging_index_forward_eval = 0
        for epoch in range(opt.num_epochs):
            model, logging_index_train, train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader, epoch, logging_index_train)
            logging_index_forward_eval, stop_early, val_loss, val_acc = inf_epoch(model, loss_fn, valid_loader, early_stopping, logging_index_forward_eval)
            if val_acc > max_acc:
                max_acc = val_acc
            wandb.log({
                "train_loss": train_loss, 
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch}) 

        model.save_model("test", config, vocab)

def inf_epoch(model, loss_fn, dataloader, early_stopping, logging_index, testing=False):
    """
    Forward pass evaluation function.
    Can be used for testing or validation.
    """
    if testing:
        print("loading best model...")
       
        model.load_state_dict(torch.load('checkpoint.pt')) # load best model

        if opt.save_model:
            model.save_model(run_name="test2", vocab=vocab)

    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()

    stop_early = False

    with torch.no_grad():

        for inputs, labels in dataloader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(opt.device), labels.to(opt.device)

            output, val_h = model(inputs, val_h)
            val_loss = loss_fn(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = metrics.acc(output,labels)
            val_acc += accuracy
        
        early_stopping(np.mean(val_losses), model)

        if early_stopping.early_stop:
            print("Early stopping")
            stop_early = True

        return logging_index, stop_early, np.mean(val_losses), (val_acc/len(dataloader.dataset))
    
def initialize_training(config):
    no_layers = config.num_layers
    vocab_size = len(vocab) + 1 #extra 1 for padding
    embedding_dim = config.embedding_dim
    output_dim = 1
    hidden_dim = config.hidden_size
    lr = config.learning_rate
    dropout = config.dropout

    network = RNN(no_layers, vocab_size, hidden_dim, embedding_dim, device=opt.device, drop_prob=dropout, output_dim=output_dim)
    network.to(opt.device)
    print(network)

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.adamw(network.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized. Please choose from Adam, SGD or AdamW.")
    
    early_stopping = EarlyStopping(patience=opt.num_epochs, verbose=True)

    # Initialize the loss here
    loss_fn = nn.BCELoss()

    return network, optimizer, early_stopping, loss_fn

def main():
    if not opt.sweep:
        logging_index_train = 0
        logging_index_forward_eval = 0

        model,optimizer, early_stopping, loss_fn = initialize_training(config=opt)
        for epoch in range(1, opt.num_epochs+1):

            model, logging_index_train, train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader,epoch, logging_index_train)
            print("\nTrain:")
            print(train_loss)
            print(train_acc)
            logging_index_forward_eval, stop_early, val_loss, val_acc = inf_epoch(model, loss_fn, valid_loader, early_stopping, logging_index_forward_eval)
            print("Val:")
            print(val_loss)
            print(val_acc)

            # cut off training if early stopping is triggered
            if stop_early:
                break

        logging_index_forward_eval = 0 

        inf_epoch(model, loss_fn, valid_loader, early_stopping, logging_index_forward_eval, testing=True)
    else:
        wandb.agent(sweep_id, sweep_train, count=50)
main()