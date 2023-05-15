import tkinter as tk
from tkinter.ttk import Button, Style, Entry
from PIL import ImageTk, Image
import model
import construct_dataset
import torch
import numpy as np
from options import Options

opt = Options().parse()

model, vocab = model.load_model(opt.load_name, "cpu", opt.num_classes)
model.to("cpu")

model.eval()

def predict_text(text):
        word_seq = np.array([vocab[construct_dataset.preprocess_string(word)] for word in text.split() 
                         if construct_dataset.preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(construct_dataset.pad_items(word_seq,opt.pad_length))
        inputs = pad.to("cpu")
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        logps, h = model(inputs, h)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)

        status = construct_dataset.idx_to_class(top_class.item())[0]

        print("Prediction:", status)
        print("confidence", top_p.item())
        return status
def resize_image(image, max_size):
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height))

def display_emoji(status):
    
    # Clear the previous emojis displayed
    canvas.delete("all")
    
    # Display emojis based on the status value
    x = 200
    y = 100
    
    if status == "Positive":
        image_path = "emojis/happy.png"  # Replace with the path to your positive emoji image
    elif status == "Neutral":
        image_path = "emojis/neutral.png"  # Replace with the path to your neutral emoji image
    else:
        image_path = "emojis/sad.png"  # Replace with the path to your negative emoji image
    
    emoji_image = Image.open(image_path)
    resized_image = resize_image(emoji_image, 250)  # Set the desired maximum size for the emoji image
    emoji_image_tk = ImageTk.PhotoImage(resized_image.convert("RGBA"))
    canvas.create_image(x, y, anchor=tk.CENTER, image=emoji_image_tk)
    canvas.image = emoji_image_tk

def display_emoji_sentiment():
    text = entry.get()
    status = predict_text(text)
    display_emoji(status)

# Create the main window
window = tk.Tk()
window.title("LSTM sentiment analysis")

# This will create style object
style = Style()
 
style.configure('W.TButton', font=('calibri', 15, 'bold'))
# Configure style for entry field
style.configure('W.TEntry')

# Create a text entry field
entry = Entry(window, width=50, style='W.TEntry', font=('calibri', 15, 'bold'))
entry.pack(pady=(50, 0), padx=30)

# Create a button to display the emojis
#button = tk.Button(window, text="Check Sentiment", command=display_emoji_sentiment)
button = Button(window, text = 'Check', style = 'W.TButton', command = display_emoji_sentiment)
button.pack(pady=30)

# Create a canvas to display the emojis
canvas = tk.Canvas(window, width=400, height=250)
canvas.pack()

display_emoji("neutral")

# Run the GUI event loop
window.mainloop()
