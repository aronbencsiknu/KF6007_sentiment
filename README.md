# Sentiment Analysis Using Long Short-Term Memory Neural Networks

## Setup

Create conda environment:
```
conda env create -f environment.yml
conda activate lstm_aronbencsik
```

Alternatively, the required libraries can be installed via pip:
```
pip3 install -r requirements.txt
```

## Examples

### Load a pre-trained model and run testing
Run the GUI:
```
python gui.py
```
Train a new model:
```
python train.py
```

## Options

| Argument  | Description |
| ------------- | ------------- |
| `-sm` or `--save_model`  | Save the model when finished training. |
| `-ld` or `--load_model_dict`  | Load a model hyperparameter dictionary. |
| `-sw` or `--sweep`  | Perform a hyperparameter sweep with Weights&Biases. |
| `-wb` or `--wandb_logging`  | Enable logging to Weights&Biases. |
| `--wandb_project`  | Weights&Biases project name. |
| `--wandb_entity`  | Weights&Biases entity name. |
| `--wandb_key`  | Weights&Biases entity name. |
| `--load_name`  | Name of the model, hyperparameter dictionary, and gain values to load. |
| `--num_epochs` | Number of epochs to train for. |
| `--batch_size` | Batch size. |
| `--learning_rate` | Learning rate of the optimizer. |
| `--dropout` | Dropout rate. |
| `--num_layers` | Number of LSTM layers. |
| `--hidden_size` | Hidden layer size. |
| `--patience` | Patience for early stopping. |
| `--embedding_dim` | Embedding layer size. |
| `--pad_length` | Padding length for the input sequences. |
| `--vocab_length` | Vocabulary length. |
| `--num_classes` | Number of classes. |
| `--optimizer` | Optimizer to use. |
