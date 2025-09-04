import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        data = np.array([values])
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0]

        if pred == 1:
            result = f"🔴 Diabetic (Confidence: {proba[1]*100:.2f}%)"
        else:
            result = f"🟢 Not Diabetic (Confidence: {proba[0]*100:.2f}%)"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI
root = tk.Tk()
root.title("Diabetes Prediction (ML)")
labels = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

tk.Button(root, text="Predict", command=predict).grid(row=len(labels), columnspan=2, pady=10)
root.mainloop()
