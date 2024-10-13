import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import mysql.connector

# Load the NER model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("sigaldanilov/bertmodel")
tokenizer = AutoTokenizer.from_pretrained("sigaldanilov/bertmodel")

# Update this with your actual label names
label_names = ['O', 'B-LOC', 'I-LOC']

# Function to perform NER prediction
def predict_ner(text, model, tokenizer, label_names):
    tokenized_input = tokenizer(text.split(), return_tensors="pt", is_split_into_words=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_input)
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [label_names[idx] for idx in predictions[0].numpy()]
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])

    locations = []
    current_location = ""
    for token, label in zip(tokens, predicted_labels):
        if token.startswith("##"):
            token = token[2:]  # Remove ## from subword tokens
            current_location += token
        else:
            if current_location:
                locations.append(current_location)
                current_location = ""
            if label in ["B-LOC", "I-LOC"]:
                current_location = token
    if current_location:
        locations.append(current_location)
    return locations

# Connect to MySQL database and get random posts
def get_random_posts():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Danilov01203",
            database="fp_db"
        )
        cursor = conn.cursor(dictionary=True)
        
        # Fetch 4 random posts from posts_new
        cursor.execute("SELECT id, img_description, owner_name FROM posts_new ORDER BY RAND() LIMIT 4")
        posts = cursor.fetchall()

        cursor.close()
        conn.close()

        return posts

    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return []

# Streamlit GUI
st.title("Random Travel Post Analyzer")

# Button to load random posts
if st.button("Load Random Posts"):
    posts = get_random_posts()

    if posts:
        for post in posts:
            st.subheader(f"Post by {post['owner_name']}:")
            st.text(post['img_description'][:100] + "...")
            
            if st.button(f"Analyze Post {post['id']}"):
                locations = predict_ner(post['img_description'], model, tokenizer, label_names)
                
                if locations:
                    st.success(f"Identified Locations: {', '.join(locations)}")
                else:
                    st.warning("No locations identified in this post.")
