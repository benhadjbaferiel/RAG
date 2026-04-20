import streamlit as st
import faiss
import json
import os
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer


# --- API Key from environment variable ---
API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("⚠️ Clé API manquante ! Définis la variable d'environnement GEMINI_API_KEY.")
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-flash-latest"
st.set_page_config(page_title="AI Chatbot", layout="centered")

@st.cache_resource
def load_all():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open("master_dataset_cleaned.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    index = faiss.read_index("vector_index.faiss")
    return model, dataset, index

model, dataset, index = load_all()


def retrieve_context(query, k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)

    chunks = []
    sources = []

    for idx in indices[0]:
        if idx != -1:
            chunks.append(dataset[idx].get("text", ""))
            sources.append(dataset[idx].get("source", "Document"))

    return chunks, sources


def generate_answer(query, context_chunks):
    context_text = "\n\n".join(
        [f"Source {i+1}: {c}" for i, c in enumerate(context_chunks)]
    )

    prompt = f"""
Tu es un expert en Intelligence Artificielle (IA, ML, DL, NLP).

Voici des extraits de documents :
{context_text}

Question : {query}

Instructions :
- Réponds en utilisant les informations ci-dessus.
- Si la réponse n'est pas présente, utilise tes connaissances pour compléter mais précise-le.
- Réponds dans la langue de la question.
"""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text

st.title(" AI Assistant")
st.markdown("Pose tes questions sur IA, Machine Learning, Deep Learning, NLP...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_chunks, sources = retrieve_context(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            try:
                answer = generate_answer(prompt, context_chunks)
            except Exception as e:
                try:
                    alt_response = client.models.generate_content(
                        model="gemini-pro-latest",
                        contents=prompt,
                    )
                    answer = alt_response.text
                except:
                    answer = f"Désolé, j'ai une erreur de connexion à l'API : {e}"

            st.markdown(answer)

            with st.expander(" Sources utilisées"):
                for src in sources:
                    st.write(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})