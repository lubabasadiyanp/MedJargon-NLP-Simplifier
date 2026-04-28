import streamlit as st
import pandas as pd
import spacy
import json
import os

# Page Config
st.set_page_config(page_title="MedJargon Simplifier", page_icon="🏥")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

def load_data():
    # Linking to the data in your repo
    with open('jargon.json', 'r') as f:
        jargon_json = json.load(f)
    
    # Build the dictionary (Person 2's Logic)
    knowledge_base = {}
    for entry in jargon_json:
        if 'entities' in entry:
            for entity in entry['entities']:
                term = " ".join(entity[3]).lower()
                knowledge_base[term] = "Medical Jargon"
    return knowledge_base

# UI Setup
st.title("🏥 Medical Jargon Simplifier")
st.markdown("Enter complex medical text below to simplify it using our NLP Engine.")

nlp = load_nlp()
med_dict = load_data()

user_input = st.text_area("Input Medical Text:", "The patient has acute tachycardia and severe edema.")

if st.button("Simplify Now"):
    doc = nlp(user_input)
    simplified_text = user_input
    
    # NLP Processing
    found_any = False
    for token in doc:
        root = token.lemma_.lower()
        if root in med_dict:
            # Highlight the jargon and show simplified meaning
            simplified_text = simplified_text.replace(token.text, f"**{token.text}** (_{med_dict[root]}_)")
            found_any = True
            
    if found_any:
        st.subheader("Simplified Result:")
        st.write(simplified_text)
    else:
        st.info("No specific jargon detected. The text seems clear!")
