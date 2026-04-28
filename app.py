import streamlit as st
import pandas as pd
import spacy
import json

st.title("🏥 Medical Jargon Simplifier")

# Load NLP model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

# Load Dictionary
def load_dict():
    # We'll create a hardcoded sample dictionary to guarantee it works today
    # You can expand this or load from your JSON
    return {
        "tachycardia": "fast heart rate",
        "bradycardia": "slow heart rate",
        "edema": "swelling",
        "acute": "sudden/severe",
        "dyspnea": "shortness of breath"
    }

nlp = load_nlp()
med_dict = load_dict()

user_input = st.text_area("Enter Medical Text:", "The patient has acute tachycardia.")

if st.button("Simplify"):
    doc = nlp(user_input.lower())
    output = user_input
    
    for token in doc:
        if token.text in med_dict:
            simplified = med_dict[token.text]
            # Replace the word with a highlighted version
            output = output.replace(token.text, f"**{token.text}** (:green[{simplified}])")
    
    st.subheader("Simplified Result:")
    st.markdown(output)