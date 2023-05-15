# global imports
import wandb
import torch
import model as m
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from progress.bar import ShadyBar

# local imports
import construct_dataset
from early_stopping import EarlyStopping
from sweep import SweepHandler
from options import Options
from model import RNN
from metrics import Metrics

opt = Options().parse()
X,y = construct_dataset.load_data()

def make_loaders(vocab=None):
    '''
    Create the train and validation loaders
    '''
    x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y, random_state=1)
    class_weights = construct_dataset.get_class_weights(X,y,opt.num_classes)

    if vocab is None:
        print("\nTokenizing...")
        vocab = construct_dataset.generate_vocabulary(x_train, vocab_len=opt.vocab_length)

    x_train, y_train = construct_dataset.tokenize(x_train, y_train, vocab)
    x_test,y_test = construct_dataset.tokenize(x_test, y_test, vocab)

    x_train_pad = construct_dataset.pad_items(x_train,opt.pad_length)
    x_test_pad = construct_dataset.pad_items(x_test,opt.pad_length)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.LongTensor(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.LongTensor(y_test))

    # dataloaders
    batch_size = opt.batch_size

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, vocab, class_weights

if not opt.test:
    train_loader, valid_loader, vocab, class_weights = make_loaders()
metrics = Metrics(num_classes=opt.num_classes)

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

  sweep_id = wandb.sweep(sweep_config, project="lstm_sweeps_final")

def train_epoch(model, optimizer, loss_fn, train_loader, valid_loader, epoch, early_stopping):
    '''
    Train the network for one epoch
    '''
    
    clip = 5
    train_losses = []
    train_acc = 0.0
    val_losses = []
    val_acc = 0.0
    # initialize hidden state 
    h = model.init_hidden(opt.batch_size)

    stop_early = False

    print()
    title = "Epoch " + str(epoch + 1)
    bar = ShadyBar(title, max=len(train_loader))

    for inputs, labels in train_loader:
        bar.next()
        model.train()
        if inputs.size()[0] == opt.batch_size:
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)   
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            model.zero_grad()
            output,h = model(inputs,h)
            
            # calculate the loss and perform backprop
            loss = loss_fn(output, labels)
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = metrics.acc(output,labels)
            if opt.wandb_logging or opt.sweep:
                wandb.log({
                    "train_loss": loss.item(), 
                    "train_acc": accuracy})
            train_acc += accuracy
            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():

            inputs, labels = next(iter(valid_loader))
            if inputs.size()[0] == opt.batch_size:

                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                h = tuple([each.data for each in h])
                output, inf_h = model(inputs, h)
                val_loss = loss_fn(output,labels)

                val_losses.append(val_loss.item())
                
                accuracy = metrics.acc(output,labels)
                if opt.wandb_logging or opt.sweep:
                    wandb.log({
                        "val_loss": val_loss.item(), 
                        "val_acc": accuracy})
                val_acc += accuracy

    bar.finish()
    train_acc = train_acc/len(train_loader)
    train_loss =  sum(train_losses)/len(train_losses)

    val_acc = val_acc/len(train_loader)
    val_loss =  sum(val_losses)/len(val_losses)
                 
    print("Train acc: ", train_acc)
    print("Train loss: ", train_loss)

    print("Val acc: ", val_acc)
    print("Val loss: ", val_loss)

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        stop_early = True

    return model, train_loss, train_acc, val_loss, val_acc, stop_early

def sweep_train(config=None):
    '''
    Hyperparameter sweep train function. 
    Called by wandb agent.
    '''
    
    # Initialize a new wandb run
    with wandb.init(config=config):
        
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        model, optimizer, early_stopping, loss_fn = initialize_training(config=config)
       
        max_acc = 0
        
        for epoch in range(opt.num_epochs):
            model, _, _, _, val_acc, _ = train_epoch(model, optimizer, loss_fn, train_loader, valid_loader, epoch, early_stopping)

            if val_acc > max_acc:
                max_acc = val_acc

        model_name = str(max_acc)+"_"+wandb.run.id +"_"+ wandb.run.name
        model.save_model(model_name, config, vocab)

def test_epoch(model, loss_fn):
    '''
    Forward pass evaluation for testing
    '''

    if not opt.test:
        print("loading best model...")
        
        model.load_state_dict(torch.load('checkpoint.pt')) # load best model
        if opt.save_model:
            model.save_model(run_name=opt.save_name, vocab=vocab)
    else:
        print("loading model...")
        model, vocab = m.load_model(opt.load_name, "cpu", opt.num_classes)
        _, dataloader, vocab, _ = make_loaders(vocab)    

    inf_h = model.init_hidden(opt.batch_size)
    inf_losses = []
    inf_acc = 0.0
    model.eval()

    with torch.no_grad():

        for inputs, labels in dataloader:
            if inputs.size()[0] == opt.batch_size:
                inf_h = tuple([each.data for each in inf_h])

                inputs, labels = inputs.to(opt.device), labels.to(opt.device)

                output, inf_h = model(inputs, inf_h)
                inf_loss = loss_fn(output,labels)

                inf_losses.append(inf_loss.item())
                metrics.increment_confusion_matrix(labels, output)

    inf_loss =  sum(inf_losses)/len(inf_losses)

    metrics.display_report()

    return inf_loss, inf_acc
    
def initialize_training(config):
    '''
    Init model, optimizer and criterion
    '''
    if not opt.test:
        no_layers = config.num_layers
        vocab_size = len(vocab) + 2 # +2 for padding and unknown
        embedding_dim = config.embedding_dim
        output_dim = opt.num_classes
        hidden_dim = config.hidden_size
        lr = config.learning_rate
        dropout = config.dropout

        network = RNN(no_layers, vocab_size, hidden_dim, embedding_dim, device=opt.device, drop_prob=dropout, output_dim=output_dim)
        network.to(opt.device)

        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        elif config.optimizer == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        elif config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer not recognized. Please choose from Adam, SGD or AdamW.")
        
        early_stopping = EarlyStopping(patience=opt.patience, verbose=True)
        loss_fn = nn.NLLLoss(class_weights.to(opt.device))

    else:
        network = None
        optimizer = None
        early_stopping = None
        loss_fn = nn.NLLLoss()

    # Initialize the loss here
    

    return network, optimizer, early_stopping, loss_fn

def main():
    if not opt.sweep:
        model,optimizer, early_stopping, loss_fn = initialize_training(config=opt)
        if not opt.test:
            for epoch in range(1, opt.num_epochs+1):

                model, _, _, _, _, stop_early = train_epoch(model, optimizer, loss_fn, train_loader, valid_loader, epoch, early_stopping)

                # cut off training if early stopping is triggered
                if stop_early:
                    break

        _, _, = test_epoch(model, loss_fn)
    else:
        wandb.agent(sweep_id, sweep_train, count=100)
main()