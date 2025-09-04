import streamlit as st
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Email Detector")
st.write("Enter a message below to check if it's spam or not. The model will classify it as 'Spam' or 'Ham'.")

# --- Caching and Model Loading ---
@st.cache_resource
def load_spam_model(model_path="spam_model.pkl"):
    """
    Loads the saved model from disk.
    Uses Streamlit's caching to load the model only once, improving performance.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # Display a more specific error if loading fails
        st.error(f"Error loading the model file: {e}")
        return None

# --- Load the Model ---
# This function call is cached, so the model is loaded only on the first run
model = load_spam_model()

# --- Main Application Logic ---
# Only show the main interface if the model was loaded successfully
if model is None:
    st.error("🔴 Critical Error: The model file ('spam_model.pkl') was not found. Please ensure it is in the same directory as this script.")
    st.stop() # Stop the script execution if the model is not available

# --- User Interface ---
# Use a form to group the input and button for a cleaner look and better performance
with st.form(key='message_form'):
    message = st.text_area(
        "Enter your email/message:",
        height=150,
        placeholder="Type or paste your message here..."
    )
    submit_button = st.form_submit_button(label="Check Message", type="primary")

# --- Prediction Logic ---
# This block runs only when the form's submit button is clicked
if submit_button:
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        # The model expects a list of strings, so we pass [message]
        # We take the first element [0] because we're only predicting for one message
        prediction = model.predict([message])[0]

        # Display result in a visually distinct container
        with st.container(border=True):
            if prediction == 1:
                st.error("🚨 Result: This message appears to be SPAM.")
            else:
                st.success("✅ Result: This message appears to be HAM (Not Spam).")
