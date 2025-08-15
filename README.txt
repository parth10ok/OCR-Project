# OCR Project

A simple **OCR (Optical Character Recognition) application** built using **Python 3**, **Tkinter**, and a trained **EMNIST-based neural network model**. It recognizes handwritten letters or digits from images.

---

## Folder Structure

OCR Project/
│
├── notebooks and dataset/
│   ├── ML_Project_OCR_Preprocessing.ipynb   # Colab notebook for preprocessing the dataset
│   └── ML_Project_OCR_Model_Training.ipynb  # Colab notebook for training and saving the model
│
├── main.py                                  # Tkinter GUI and code to load the trained model
├── emnist_model.h5                          # Pre-trained EMNIST model
├── requirements.txt                         # Required Python packages (auto-installed by main.py)
├── sample1.png                              # Sample image for testing
├── sample2.png                              # Sample image for testing

---

## Setup & Run

1. Make sure you have **Python 3.x** installed.  

2. Clone this repository:

git clone <your-repo-link>
cd "OCR Project"

3. Install dependencies (if not already installed, `main.py` will auto-install them):

pip install -r requirements.txt

4. Run the application:

python main.py

5. The GUI will open. Input the image address and test the OCR.  

---

## Notes

- The `notebooks and dataset/` folder contains the **original Colab notebooks** for dataset preprocessing and model training. You **don’t need to run them** to use the app.  
- The model `emnist_model.h5` is **pre-trained**.  
- Test using the provided sample images (`sample1.png`, `sample2.png`).

---

## Dependencies

Key Python packages used:

- tensorflow
- numpy
- pandas
- opencv-python
- pillow
- tkinter (built-in with Python)
- matplotlib
- gdown

---

**Author:** Parth Kadam (IIT Madras)

