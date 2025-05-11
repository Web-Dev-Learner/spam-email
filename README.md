

# Spam Email Detection

This project aims to automatically classify emails as either spam or ham (non-spam) using machine learning techniques. The dataset used consists of labeled emails that are processed and used to train models to distinguish spam from legitimate emails. The project showcases data preprocessing, model training, evaluation, and deployment for email classification.

## What I’ve Done

### 1. **Data Collection**
   - Downloaded the **Spam-Ham Dataset** from a reliable source.
   - The dataset contains labeled emails with two classes: Spam and Ham.

### 2. **Data Preprocessing**
   - Loaded and explored the dataset using **Pandas**.
   - Cleaned the dataset by removing unnecessary columns and handling any missing data.
   - Balanced the dataset to ensure equal representation of both classes (Spam and Ham).
   - Applied **Text Preprocessing**:
     - Tokenized the email content.
     - Removed stopwords, punctuation, and performed stemming/lemmatization.
     - Used **TF-IDF Vectorizer** to convert text data into numerical format for model input.

### 3. **Model Selection & Training**
   - Split the dataset into training and testing sets.
   - Trained multiple machine learning models, including:
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**
   - Evaluated models using accuracy, precision, recall, and F1 score.

### 4. **Model Evaluation**
   - Used **cross-validation** to evaluate the model performance.
   - Achieved high performance with an accuracy of **98.17%** on the test set and a test loss of **0.08**.
   - Visualized model evaluation metrics using **Matplotlib** and **Seaborn**.

### 5. **Deployment**
   - Added the model to a **Jupyter Notebook** for easy reproducibility and sharing.
   - Created an environment with necessary dependencies.
   - Pushed the project to **GitHub** for version control and public sharing.

## Features

- **Spam vs. Ham Classification**: Classifies emails into spam or ham categories.
- **Data Preprocessing**: Includes data cleaning, text processing, and vectorization.
- **Model Training**: Trained machine learning models like Logistic Regression and SVM.
- **Model Evaluation**: Performance metrics visualization and cross-validation.
- **High Accuracy**: Achieved high accuracy in spam detection.

  
## Dataset

The dataset used in this project is the **Spam-Ham** dataset, which contains labeled emails. The dataset is preprocessed to remove unnecessary characters and stop words. The goal is to classify each email as either spam or ham.

## Tech Stack

- **Languages**: Python
- **Libraries**:
  - **Pandas**: Data manipulation
  - **NumPy**: Numerical operations
  - **Scikit-learn**: For machine learning models and preprocessing
  - **TensorFlow**: Used for more advanced models (if needed)
  - **Matplotlib & Seaborn**: For data visualization
    

- **Tools**: 
  - **Jupyter Notebook**: For running the code and experimentation
  - **VS Code**: IDE for development


