import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore

# Predict function that takes in an image path and returns the predicted class
def predict(img_path):
    img = keras_image.load_img(img_path, target_size=(256, 256))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    probability = np.max(predictions)  # Get the max probability value
    predicted_label = labels[predicted_class_index]
    return predicted_label, probability

# Function to display the selected image and make a prediction
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        display_image(file_path)
        predicted_class, probability = predict(file_path)  # Adjusted to receive two return values
        prediction_label.config(text=predicted_class)
        probability = probability * 100
        probability_label.config(text=f"Probability: {probability:.2f}%")  # Display the probability

# Function to update the image label with the selected image
def display_image(img_path):
    global image_label
    img = Image.open(img_path)
    img = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference

# Exit function to close the application
def exit_app():
    root.destroy()

def on_enter(e):
    select_button.config(background='gray', foreground='white')

def on_leave(e):
    select_button.config(background='black', foreground='white')

def main():
    global model, image_label, prediction_label, probability_label, labels, root, select_button

    # Load the model
    model = load_model("Program/animal_classifier.keras") 
    model.summary()

    # Class labels
    labels = {0: 'Cat', 1: 'Dog', 2: 'Snake'}

    # Global variables for UI components
    image_label = None
    prediction_label = None
    probability_label = None 

    # UI setup
    root = tk.Tk()
    root.title("Animal Classifier")

    # Make the root window cover the entire screen
    root.state('zoomed')

    # Header frame for title and exit button
    header_frame = tk.Frame(root)
    header_frame.pack(fill='x')

    title_label = tk.Label(header_frame, text="Animal Classifier", font=("Arial", 40))
    title_label.pack(side='top', expand=True)

    sub_title_label = tk.Label(header_frame, text="Cat | Dog | Snake", font=("Arial", 20))
    sub_title_label.pack(side='top', expand=True)

    exit_button = tk.Button(header_frame, text="Exit", command=exit_app)
    exit_button.pack(side='top', expand=True, pady=10)

    # Content frame for the image, arrow, and prediction label
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True)

    # Frame for image and prediction
    image_prediction_frame = tk.Frame(content_frame)
    image_prediction_frame.pack(pady=50)

    # Initialize the label for image preview
    image_label = tk.Label(image_prediction_frame, borderwidth=2, relief="groove")
    image_label.pack(side='top', padx=10)


    # Initialize the label for prediction
    prediction_label = tk.Label(image_prediction_frame, text="Class", font=("Arial", 30))
    prediction_label.pack(side='bottom', padx=10)

    # Frame for probability
    probability = tk.Frame(content_frame)
    probability.pack(pady=10)

    # Initialize the label for probability
    probability_label = tk.Label(probability, text="Probability: ", font=("Arial", 20))
    probability_label.pack(side='bottom', padx=0)  # Adjust as needed

    # Button frame for select button
    button_frame = tk.Frame(root)
    button_frame.pack(side='bottom', pady=100)

    select_button = tk.Button(button_frame, text="Select Animal", command=select_image)
    select_button.config(height=4, width=20)
    select_button.config(background='black', foreground='white')
    # Add these bindings after creating the select_button
    select_button.bind("<Enter>", on_enter)
    select_button.bind("<Leave>", on_leave)

    select_button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
