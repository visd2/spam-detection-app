import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import requests
import zipfile
import io

def download_and_extract_data():
    """Downloads and extracts the SMS Spam Collection dataset."""
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_file = "SMSSpamCollection"

    if os.path.exists(data_file):
        print("Dataset already exists. Skipping download.")
        return data_file

    print(f"Downloading dataset from {dataset_url}...")
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        print("Extracting data...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall()

        print("Dataset downloaded and extracted successfully.")
        return data_file
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return None

def train_and_save_model():
    """
    Trains a high-accuracy classifier on a real dataset and saves it.
    """
    print("--- Starting High-Accuracy Model Training Process ---")

    # 1. Get the data
    data_file_path = download_and_extract_data()
    if not data_file_path:
        print("Could not obtain dataset. Aborting training.")
        return

    try:
        # 2. Load and Prepare Data
        print("Loading and preparing data...")
        df = pd.read_csv(data_file_path, sep='\t', header=None, names=['label', 'message'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1}) # Convert labels to numbers

        # 3. Define Features (X) and Target (y) and Split Data
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

        # 4. Create a More Powerful Machine Learning Pipeline
        # TfidfVectorizer is often more effective for text classification.
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])

        # 5. Train the Model
        print("Training the model...")
        pipeline.fit(X_train, y_train)

        # 6. Evaluate the Model
        print("Evaluating model performance...")
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Model Accuracy on Test Data: {accuracy:.2%}")

        # 7. Save the Trained Model
        model_filename = 'spam_model.pkl'
        print(f"Saving the trained model to '{model_filename}'...")
        joblib.dump(pipeline, model_filename)

        print(f"\n✅ Model training complete. '{model_filename}' has been created with high accuracy.")
    except Exception as e:
        print(f"\n❌ An error occurred during the training process: {e}")
        print(f"   The '{model_filename}' file was NOT created.")

if __name__ == "__main__":
    train_and_save_model()
