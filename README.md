# Detecting Phishing Websites Using Hybrid Deep Learning

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-green)

This repository contains the source code, machine learning models, and web application for the Final Year Project: **Detecting Phishing Websites Using Hybrid Deep Learning**, developed by **Andy Phang Kwan Yuen** at Universiti Malaysia Sarawak (UNIMAS).

## 📌 Project Overview
Phishing websites remain a critical cybersecurity threat, with attackers continuously adapting to bypass static defense rules and simple heuristics. This project proposes a robust **Hybrid Deep Learning Architecture** combining **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory (BiLSTM)** networks to achieve superior phishing website detection through raw URL analysis. 

The trained model is deployed in a functional, real-time web application featuring an automated scanner and a secure administrative dashboard.

## ✨ Key Features
* **Hybrid CNN-BiLSTM Architecture:** Uses a 1D-CNN to extract local character-level spatial features (e.g., brand spoofing, specific n-grams) and passes them to a BiLSTM to evaluate the bidirectional sequential context of the entire URL.
* **Dual-Approach Feature Engineering:**
  * **Sequence Vectorization:** Character-level tokenization (Max Length: 100).
  * **Lexical Features:** Extraction of 22 explicit statistical and structural URL properties (e.g., entropy, digit ratios, symbol frequencies).
* **Real-Time Web Application:** Built with Streamlit for a reactive, user-friendly frontend.
* **MongoDB Integration:** Maintains a url_logs collection for scan history and a whitelists_blacklists collection to instantly bypass the AI for known safe/malicious sites.
* **Human-in-the-Loop Workflow:** Scans with low confidence (between 10% and 90%) are flagged for manual review via a secure Admin Dashboard.

## 📊 Dataset
The model was trained on the benchmark **Phishing Site URLs** dataset from Kaggle.
* **Total URLs:** 549,346
* **Class Distribution:** 395,529 Legitimate (72%) / 153,817 Phishing (28%)
* **Preprocessing:** Handled nulls, removed duplicates, normalized text (lowercased, stripped HTTP prefixes), and balanced the classes for optimal F1-score evaluation.

## 🏆 Model Performance
The hybrid model was evaluated using **Stratified 5-Fold Cross-Validation** against standard baseline architectures. The CNN-BiLSTM configuration proved to be the most effective.

| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN-BiLSTM (Hybrid)** | **98.08%** | **98.10%** | **97.08%** | **98.08%** | **98.24%** |
| CNN (Baseline) | 96.27% | 96.28% | 96.27% | 96.27% | 96.99% |
| GRU (Baseline) | 96.10% | 96.11% | 96.10% | 96.10% | 96.92% |
| LSTM (Baseline) | 95.16% | 95.21% | 95.16% | 95.16% | 96.55% |

*(Note: Under optimal hyperparameters, the final CNN-BiLSTM model achieved up to **98.8% Accuracy** and **99.0% F1-Score**).*

## 🛠️ Technology Stack
* **Core Machine Learning:** Python 3.11+, TensorFlow, Keras, Scikit-learn
* **Data Processing:** Pandas, NumPy, Regex
* **Web Frontend / Backend:** Streamlit
* **Database:** MongoDB (PyMongo)

## 🚀 Installation and Setup

### 1. Prerequisites
Ensure you have Python 3.11 or higher installed. Set up a MongoDB cluster (e.g., MongoDB Atlas) to handle data storage.

### 2. Clone the Repository
Clone the repository from GitHub to your local machine and navigate into the project directory using your terminal or command prompt.

### 3. Install Dependencies
Install all required libraries using your Python package manager based on the provided requirements document.

### 4. Configure Environment Variables
Create an environment variables file in the root directory and securely add your MongoDB connection string to ensure the database can connect properly.

## 🧠 Training the Models
If you wish to train the models locally:
1. Download the phishing site URLs dataset from Kaggle and place it in the project root.
2. Run the provided Python training script to perform cross-validation and export the final model. This process will generate the trained model file and the tokenizer file necessary for the web application.

## 🌐 Running the Web Application
Start the Streamlit server using the standard Streamlit execution command. Navigate to the provided localhost URL in your browser to access the application interface.

### App Navigation
* **Homepage (User):** Submit URLs for instant scanning. The interface will display green (Safe), red (Phishing), or yellow (Pending Review) alerts based on the model's confidence.
* **Admin Login:** Authenticate using secure credentials to access the backend control panel.
* **Admin Dashboard:** Review low-confidence URLs, force-add domains to whitelists or blacklists using the manual override function, and monitor recent scan logs.

## 👨‍💻 Author
**Andy Phang Kwan Yuen**
* Bachelor of Computer Science with Honours (Information Systems)
* Faculty of Computer Science and Information Technology, Universiti Malaysia Sarawak (UNIMAS)
* Year: 2026