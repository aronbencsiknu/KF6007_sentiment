import pathlib
import wandb
import torch
import json

from options import Options
from model import RNN
from metrics import Metrics
from torch.utils.data import DataLoader
import construct_dataset
from sklearn.model_selection import train_test_split

X,y = construct_dataset.load_data()
X = construct_dataset.vectorize(X)
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

train_set = construct_dataset.CustomDataset(x_train,y_train)
test_set = construct_dataset.CustomDataset(x_test,y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(test_set, batch_size=32, shuffle=True)

opt = Options().parse()
metrics = Metrics()

def train_epoch(net, train_loader,optimizer,epoch, logging_index):
    """
    Train the network for one epoch
    """

    net.train()
    loss_val=0

    print("\nEpoch:\t\t",epoch,"/",opt.num_epochs)
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(opt.device)
        target=target.to(opt.device)
        
        output = net(data)

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val+=loss.item()

        if opt.wandb_logging:
            wandb.log({"Train loss": loss.item(),
                    "Train index": logging_index})

        logging_index+=1

    loss_val /= len(train_loader)
    print("Train loss:\t%.6f" % loss_val)
    
    return net, logging_index, loss_val

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
        
        model,optimizer, early_stopping = initialize_training(config=config)
       
        max_acc = 0

        for epoch in range(opt.num_epochs):
            model, _, _ = train_epoch(model, train_loader, optimizer, epoch, logging_index=0)
            _, _, avg_loss, avg_acc = inf_epoch(model, val_loader, early_stopping, logging_index=0)
            if avg_acc > max_acc:
                max_acc = avg_acc
            wandb.log({"loss": avg_loss, "epoch": epoch, "accuracy": avg_acc}) 

        save_model(model, max_acc, config)

def inf_epoch(model, dataloader, early_stopping, logging_index, testing=False):
    """
    Forward pass evaluation function.
    Can be used for testing or validation.
    """
    if testing:
        print("loading best model...")

        if opt.load_model:
            model = load_model()
        else:
            model.load_state_dict(torch.load('checkpoint.pt')) # load best model

        if opt.save_model:
            save_model(model, 0, config=None)

    model.eval()
    loss = 0
    acc = 0
    stop_early = False

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(dataloader):

            data=data.to(opt.device)
            target=target.to(opt.device)

            #output = metrics.forward_pass(model, data, opt.num_steps)
            output = model(data)

            acc_current = metrics.accuracy(output,target)

            #loss_current = metrics.loss(output,target).item()
            loss_current =  loss = loss_fn(y_pred, y_true)
            loss+=loss_current
            acc+=acc_current

            if opt.wandb_logging and not testing:
                wandb.log({"Validation loss": loss_current, 
                            "Validation Accuracy": acc_current,
                            "Validation index": logging_index})

            elif opt.wandb_logging and testing:
                wandb.log({"Test loss": loss_current, 
                            "Test Accuracy": acc_current,
                            "Test index": logging_index})

            predicted = metrics.return_predicted(output)

            if len(data)==opt.batch_size and testing:
                for i in range(opt.batch_size):
                    y_pred.append(predicted[i].item())
                    y_true.append(target[i].item())

            logging_index+=1

    if testing:
        metrics.perf_measure(y_true, y_pred)
        loss /= len(dataloader)
        acc /= len(dataloader)
        acc = acc*100
    
    if testing:
        
        print("Test loss:\t%.3f" % loss)
        print("Test acc:\t%.2f" % acc+"%\n")
        print("Precision:\t%.3f" % metrics.precision())
        print("Recall:\t%.3f"% metrics.recall())
        print("F1-Score:\t%.3f"% metrics.f1_score() +"\n")

        if opt.save_results:
            with open('results/'+opt.run_name+'.txt', 'w') as file:

                # Write the values to the text file
                file.write("\nTest loss: "+str(loss))
                file.write("\nTest acc: "+str(acc))
                file.write("\nPrecision: "+str(metrics.precision()))
                file.write("\nRecall: "+str(metrics.recall()))
                file.write("\nF1-Score: "+str(metrics.f1_score()))

                file.close()

        plots.plot_confusion_matrix(y_pred, y_true, opt.run_name, opt.save_results)

    else:
        print("Val loss:\t%.3f" % loss)
        print("Val acc:\t%.3f"% acc+"%\n")
        early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
        stop_early = True

        return logging_index, stop_early, loss, acc

def save_model(model, config=None):
    # save model here
    model_name = opt.run_name + ".pt"
        
    path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / model_name)
    model.load_state_dict(torch.load('checkpoint.pt')) # load best model
    torch.save(model.state_dict(), path_model)

    if config is not None:
        # save model dict here
        config_dict = config.__dict__
        config_dict = config_dict["_items"]

        dict_name = opt.run_name + "_dict.json"
        path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / dict_name)
        
        with open(path_p_dict, 'w') as f:
            json.dump(config_dict, f)

def load_model():
    """
    Load model from trained_models folder
    """
    name_model = opt.load_name + "_model.pt"
    name_dict = opt.load_name + "_dict.json"

    path_model = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / name_model)
    path_p_dict = pathlib.Path(pathlib.Path.cwd() / "trained_models" / opt.input_encoding / name_dict)

    # load hyperparameters
    p_dict = open(path_p_dict)
    p_dict = json.load(p_dict)

    # initialize model with hyperparameters and load weights
    model = SNN(input_size=input_size, hidden_size=opt.hidden_size, output_size=2, h_params=p_dict).to(opt.device)
    model.load_state_dict(torch.load(path_model, map_location=torch.device(opt.device)))

    return model

def initialize_training(current_config):
    # Initialize the model here
    hidden_size = [current_config.hidden_size]*current_config.num_hidden
    network = RNN(input_size=current_config.input_size, hidden_size=hidden_size, dropout=current_config.dropout)
    # Initialize the optimizer here
    optimizer = torch.optim.AdamW(network.parameters(), lr=current_config.learning_rate)

    # Initialize the early stopping here
    early_stopping = EarlyStopping(patience=opt.num_epochs, verbose=True)

    return network, optimizer, early_stopping
    
def main():

    model,optimizer, early_stopping = initialize_training(config=opt)

    if opt.sweep:
        wandb.agent(sweep_id, sweep_train, count=50)
    else:
        for epoch in range(1, opt.num_epochs+1):

            model, logging_index_train, _ = train_epoch(model,train_loader,optimizer,epoch, logging_index_train)
            logging_index_forward_eval, stop_early, _, _ = inf_epoch(model, val_loader, early_stopping, logging_index_forward_eval)

            # cut off training if early stopping is triggered
            if stop_early:
                break

        logging_index_forward_eval = 0 

        #inf_epoch(model, test_loader, early_stopping, logging_index_forward_eval, testing=True)
main()