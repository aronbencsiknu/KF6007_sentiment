import tkinter as tk
import emoji
import model
import construct_dataset
import torch
import numpy as np
from options import Options

opt = Options().parse()

if opt.load_model_dict:
    config = None
else:
     config = opt
model, vocab = model.load_model(opt.load_name, "cpu", config)
model.to("cpu")

model.eval()

def predict_text(text):
        word_seq = np.array([vocab[construct_dataset.preprocess_string(word)] for word in text.split() 
                         if construct_dataset.preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(construct_dataset.pad_items(word_seq,500))
        inputs = pad.to("cpu")
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())

def show_emojis():
    # Get the text entered by the user
    text = entry.get()
    prediction = predict_text(text)
    status = "positive" if prediction > 0.5 else "negative"
    print("Prediction:", status)
    print("Confidence:", prediction)
    # Clear the previous emojis displayed
    canvas.delete("all")
    
    # Display emojis based on the text value
    x = 200
    y = 100
    if status == "positive":
        emoji_char = emoji.emojize("🙂")
    else:
         emoji_char = emoji.emojize("🙁")
    
    canvas.create_text(x, y, text=emoji_char, font=("Arial", 100))

# Create the main window
window = tk.Tk()
window.title("LSTM sentiment analysis")

# Create a text entry field
entry = tk.Entry(window, width=50)
entry.pack(pady=40, padx=10)

# Create a button to display the emojis
button = tk.Button(window, text="Check Sentiment", command=show_emojis)
button.pack()

# Create a canvas to display the emojis
canvas = tk.Canvas(window, width=400, height=200)
canvas.pack()

# Run the GUI event loop
window.mainloop()
