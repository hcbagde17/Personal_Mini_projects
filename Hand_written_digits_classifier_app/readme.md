
---

# 🧠 Handwritten Digit Recognition (MNIST)

This project is a **Streamlit web application** that uses a trained **Artificial Neural Network (ANN)** to recognize handwritten digits (0–9) based on the **MNIST dataset**.

Users can upload an image of a handwritten digit, and the model predicts the digit along with the probability distribution for all classes.

---

## 🚀 Features

* Upload handwritten digit images (`PNG`, `JPG`, `JPEG`)
* Preprocessing consistent with the MNIST dataset
* Real-time digit prediction
* Probability distribution for all digits (0–9)
* Simple and interactive Streamlit user interface

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pillow (PIL)

---

## 📂 Project Structure

```
mnist_streamlit_app/
│
├── app.py                     # Streamlit application
├── mnist_ann_model.keras      # Trained ANN model
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 📤 How It Works

1. User uploads a handwritten digit image
2. Image is converted to grayscale and resized to **28 × 28**
3. Pixel values are normalized to the range **0–1**
4. Image is passed to the trained ANN model
5. Model predicts the digit and class probabilities

---

## 📊 Model Performance

* Accuracy: **~98%**
* Balanced precision, recall, and F1-score across all classes
* Regularized ANN with smooth training and validation loss curves

---

## 📌 Notes

* Best results are obtained with **clear and centered digit images**
* Background inversion is handled internally to match MNIST image style

---

## 🧪 Test the App

### Sample Image Links

* [https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeOqjUXvXE0bIyCvdp5gyrAOnVPVsD9d_9gg&s](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeOqjUXvXE0bIyCvdp5gyrAOnVPVsD9d_9gg&s)
* [https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRypGAtGpzAWSIR4UTKjYh7coTjV8sVZE5nPg&s](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRypGAtGpzAWSIR4UTKjYh7coTjV8sVZE5nPg&s)
* [https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7s63KeUWO4VVh0uJ3nqKp2JyQsBgEhbcZUg&s](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7s63KeUWO4VVh0uJ3nqKp2JyQsBgEhbcZUg&s)
* [https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSW_sRP1sdXkDvP7Llam7H6MbPFKGk6Dxz4sMOkqOkbRA&s](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSW_sRP1sdXkDvP7Llam7H6MbPFKGk6Dxz4sMOkqOkbRA&s)
* [https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpCCerwGTaiiQiYXQs5nGd-3fOfcW2NTRhdg&s](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpCCerwGTaiiQiYXQs5nGd-3fOfcW2NTRhdg&s)

### External Websites for More Test Images

* [https://www.hackersrealm.net/post/mnist-handwritten-digits-recognition-using-python](https://www.hackersrealm.net/post/mnist-handwritten-digits-recognition-using-python)
* [https://blog.otoro.net/assets/20160401/png/mnist_dream_24.png](https://blog.otoro.net/assets/20160401/png/mnist_dream_24.png)

---

## 👨‍💻 Author

Developed as a **practice project** for learning:

* Deep Neural Networks
* Model regularization
* Model deployment using Streamlit

---

