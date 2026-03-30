# Multi-Modal Deep Learning Pipeline: Tabular, NLP, and Computer Vision

## 📌 Project Overview
This project demonstrates the end-to-end implementation of three distinct Deep Learning architectures to solve real-world data challenges. Instead of focusing on a single data type, this pipeline handles structured numerical data, unstructured text, and raw image data in a single integrated environment.

---

## 🛠️ Phase 1: Tabular Data Regression (Predictive Modeling)
**Objective:** Predict [Mention the Target, e.g., Sales/Price] based on customer features.
* **Data Engineering:** Handled missing values using Median/Mode imputation and implemented One-Hot Encoding for categorical variables.
* **Preprocessing:** Applied `StandardScaler` to prevent feature dominance and ensured a 70/15/15 Train-Val-Test split to avoid data leakage.
* **Architecture:** Built a Feed-Forward Neural Network (FFNN) using ReLU activation and Mean Squared Error (MSE) loss.

## 📝 Phase 2: Natural Language Processing (Sentiment/Text Analysis)
**Objective:** Classify text sequences from `text.csv` into categorical labels.
* **Pipeline:** Implemented a full NLP preprocessing string—Tokenization, Sequence Padding, and Word Embeddings.
* **Architecture:** Utilized **LSTM/RNN** layers to capture sequential dependencies in the text, significantly outperforming standard dense layers.

## 🖼️ Phase 3: Computer Vision (Image Classification)
**Objective:** Categorize raw images into distinct classes (`class_0` vs `class_1`).
* **Data Pipeline:** Developed a robust loading script using `tf.keras.utils.image_dataset_from_directory` with automated resizing to 160x160.
* **Feature Extraction:** Built a Multi-block **Convolutional Neural Network (CNN)**.
* **Regularization:** Integrated **Dropout (0.5)** and **MaxPooling** layers to reduce spatial dimensions and prevent overfitting on small datasets.

---

## 📊 Technical Skills Demonstrated
* **Frameworks:** TensorFlow, Keras, Scikit-Learn.
* **Data Science:** Pandas, NumPy, Matplotlib (Visualization).
* **Deep Learning Concepts:** Backpropagation, Activation Functions (ReLU/Sigmoid), Regularization, and Optimization (Adam).

## 🚀 How to Navigate This Repo
1.  Check `requirements.txt` for the environment setup.
2.  The `/data` folder contains the processed CSVs and the image directory.
3.  Open `Deep-Learning-MultiModal-Analysis.ipynb` to see the documented code, training logs, and performance visualizations.

📈 Key Technical Takeaways
Data Leakage Prevention: Scalers were fit only on training data and then applied to validation/test sets.

Model Convergence: Monitored Loss vs. Accuracy curves to identify the "sweet spot" before overfitting began.

Modular Code: Designed the notebook so each data type (Tabular, Text, Image) can be run independently.

⚙️ Environment Setup
To replicate this project:

Clone the repository.

Place your data/ folder in the root directory.

Install dependencies:
pip install -r requirements.txt