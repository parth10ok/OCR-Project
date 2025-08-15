import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import subprocess
import sys
import tkinter as tk
import io
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Install from requirements.txt
req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

# Creating mapping for 62 classes (digits + uppercase + lowercase)
num_to_char = {}
for i in range(10):
    num_to_char[i] = str(i)
for i in range(26):
    num_to_char[i + 10] = chr(65 + i)
for i in range(26):
    num_to_char[i + 36] = chr(97 + i)

def image_process():
    image_path = address.get()

    # Loading image and converting to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return image


def pred():
    global heatmap_frame, image_frame
    try:
        image_path = address.get()
        image = image_process()

        # Display Image
        for widget in image_frame.winfo_children():
            widget.destroy()  # Clear old image if any
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_color)
        img_pil.thumbnail((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)
        img_label = tk.Label(image_frame, image=img_tk)
        img_label.image = img_tk  # keep reference
        img_label.pack()

        background_black = False
        if np.mean(image) > 127:
            image = cv2.bitwise_not(image)
            background_black = True

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        first_x, first_y, first_w, first_h = cv2.boundingRect(contours[0])
        padding = int(first_h / 9)

        predicted_text = ""
        prev_x_end = 0
        space_threshold = None

        # Clear old heatmaps
        for widget in heatmap_frame.winfo_children():
            widget.destroy()
        fig_count = 0

        # Anomaly filter
        MIN_CONTOUR_AREA = 50  # tweak if needed

        for idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if idx == 0:
                space_threshold = int((2 / 3) * h)

            if prev_x_end != 0 and (x - prev_x_end) > space_threshold:
                predicted_text += ' '
            prev_x_end = x + w

            char_img = thresh[y:y + h, x:x + w]

            char_img_padded = cv2.copyMakeBorder(
                char_img, padding, padding, padding, padding,
                borderType=cv2.BORDER_CONSTANT,
                value=0 if background_black else 255
            )

            char_img_resized = cv2.resize(char_img_padded, (28, 28), interpolation=cv2.INTER_AREA)
            char_img_norm = char_img_resized.astype("float32") / 255.0
            char_img_input = np.expand_dims(char_img_norm, axis=-1)
            char_img_input = np.expand_dims(char_img_input, axis=0)

            pred_val = model.predict(char_img_input, verbose=0)
            pred_class = np.argmax(pred_val, axis=1)[0]
            predicted_text += num_to_char.get(pred_class, '?')

            # Generate heatmap and show in Tkinter
            if fig_count < 9:
                fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
                ax.imshow(char_img_resized, cmap='hot', interpolation='nearest')
                ax.axis('off')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                img = Image.open(buf)
                img_tk = ImageTk.PhotoImage(img)

                lbl = tk.Label(heatmap_frame, image=img_tk, bg="gainsboro")
                lbl.image = img_tk  # Keep reference
                lbl.pack(side=tk.LEFT, padx=2)

                plt.close(fig)
                fig_count += 1

        prediction_label.config(text=f"Predicted text: {predicted_text}")
    except:
        prediction_label.config(text="ERROR! IMAGE ADDRESS NOT FOUND")


def appl():
    global app, address, model, prediction_label, heatmap_frame, image_frame
    app = tk.Tk()
    app.title("OCR")
    app.config(bg="gainsboro")
    app.minsize(width=800, height=600)
    address = tk.StringVar()

    MODEL_PATH = "emnist_model.h5"
    model = load_model(MODEL_PATH)

    address_label = tk.Label(app,text="Enter the address of the image.\nIf image is in the same directory as app,\njust input the image name, for example:\nimage1.png",font=("calibari,50"), bg="gainsboro", fg="black")
    address_label.place(x=100, y=50)
    address_entry = tk.Entry(app, textvariable=address, font=(13), width=30)
    address_entry.place(x=100, y=150)

    predict_button = tk.Button(app, text="Predict", width=5, height=1, bg="slategray", font=("calibari", 18),command=pred)
    predict_button.place(x=100, y=180)

    prediction_label = tk.Label(app, text="Predicted text:", font=("calibari", 14), bg="gainsboro", fg="black")
    prediction_label.place(x=50, y=550)

    # Frame to hold heatmaps
    seperated_label = tk.Label(app, text="First Few Separated Characters (For Reference):", font=("calibari,50"),bg="gainsboro", fg="black")
    seperated_label.place(x=50, y=400)
    heatmap_frame = tk.Frame(app, bg="gainsboro")
    heatmap_frame.place(x=20, y=450)

    # Frame to hold original image
    image_frame = tk.Frame(app, bg="gainsboro")
    image_frame.place(x=100, y=250)

    app.mainloop()
appl()
