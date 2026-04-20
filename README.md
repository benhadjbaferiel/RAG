# 🤖 AI Assistant — RAG Chatbot (NLP / ML / DL)

Un chatbot intelligent basé sur **RAG (Retrieval-Augmented Generation)** qui répond à des questions sur l'Intelligence Artificielle, le Machine Learning, le Deep Learning et le NLP, en s'appuyant sur une base documentaire vectorisée.

## ✨ Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS + Sentence-Transformers
- 🧠 **Génération de réponses** avec Gemini (Google GenAI)
- 📚 **Sources citées** pour chaque réponse
- 💬 **Interface chat** interactive avec Streamlit

## 🏗️ Architecture

```
PDF Books → tojsn.py → jsons/ → fusion.py → master_dataset.json
→ nettoyage.py → master_dataset_cleaned.json → vecto.py → vector_index.faiss
→ app.py (Streamlit + FAISS + Gemini)
```

## 🚀 Installation & Utilisation

### 1. Cloner le repo

```bash
git clone https://github.com/<ton-username>/nlptp.git
cd nlptp
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger les fichiers pré-construits

Les fichiers du modèle sont trop lourds pour GitHub. **Télécharge-les depuis les Releases** :

👉 [**Télécharger depuis Releases**](https://github.com/<ton-username>/nlptp/releases/latest)

Télécharge et place ces 2 fichiers à la racine du projet :
- `master_dataset_cleaned.json` (~11 MB)
- `vector_index.faiss` (~20 MB)

### 4. Configurer la clé API

Crée un fichier `.env` ou exporte la variable :

```bash
# Linux / Mac
export GEMINI_API_KEY="ta-cle-api-gemini"

# Windows PowerShell
$env:GEMINI_API_KEY="ta-cle-api-gemini"
```

### 5. Lancer l'application

```bash
streamlit run app.py
```



## 📁 Structure du projet

| Fichier | Description |
|---------|-------------|
| `app.py` | Application Streamlit (chatbot RAG) |

## 🛠️ Technologies

- **Streamlit** — Interface web
- **FAISS** — Recherche vectorielle
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — Embeddings
- **Google Gemini** — Génération de réponses
- **PyMuPDF** — Extraction PDF

## 📝 License

Ce projet est à usage éducatif.
