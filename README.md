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

Run the GUI:
```
python gui.py
```
Train a new model:
```
python main.py
```
Test a pre-trained model:
```
python main.py
```

## Options

| Argument  | Description |
| ------------- | ------------- |
| `-sm` or `--save_model`  | Save the model when finished training. |
| `-sw` or `--sweep`  | Perform a hyperparameter sweep with Weights&Biases. |
| `-t` or `--test`  | Test the model |
| `-wb` or `--wandb_logging`  | Enable logging to Weights&Biases. |
| `--wandb_project`  | Weights&Biases project name. |
| `--wandb_entity`  | Weights&Biases entity name. |
| `--wandb_key`  | Weights&Biases API key |
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
