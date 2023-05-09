import tkinter as tk
import emoji
import model
import construct_dataset
import torch
import numpy as np

model, vocab = model.load_model("test", "cpu")
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
    print(status)
    # Clear the previous emojis displayed
    canvas.delete("all")
    
    # Display emojis based on the text value
    x = 50
    y = 50
    emoji_char = emoji.emojize(":slightly_smiling_face:")
    canvas.create_text(x, y, text=emoji_char, font=("Arial", 50))



# Create the main window
window = tk.Tk()
window.title("Check Sentiment")

# Create a text entry field
entry = tk.Entry(window)
entry.pack()

# Create a button to display the emojis
button = tk.Button(window, text="Show Emojis", command=show_emojis)
button.pack()

# Create a canvas to display the emojis
canvas = tk.Canvas(window, width=400, height=200)
canvas.pack()

# Run the GUI event loop
window.mainloop()
