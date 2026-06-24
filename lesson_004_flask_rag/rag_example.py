import os
import faiss
import numpy as np
import nltk
from dotenv import load_dotenv
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Download NLTK assets
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_documents(folder='../data'):
    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                for sentence in sent_tokenize(text):
                    words = word_tokenize(sentence)
                    # Not a good practice (to remove numbers) - just for illustration
                    clean = [w for w in words if w.lower() not in stop_words and w.isalnum()]
                    all_sentences.append(" ".join(clean))
    print("All sentences:", all_sentences)
    return all_sentences

def create_faiss(sentences, model):
    embeddings = model.encode(sentences)

    # Gets the number of dimensions in each embedding vector (usually 384 or 768).
    dim = embeddings.shape[1] # the dimension number of the embedding to be used as index!

    # ● Creates a flat FAISS index using L2 distance (Euclidean distance).
    # ● This index will store all the vectors and be able to search them later.
    index = faiss.IndexFlatL2(dim) # FAISS needs the dim to create the right type of index.

    # ● Adds all the sentence embeddings to the FAISS index.
    # ● Now FAISS is ready to find similar sentences using vector math.
    index.add(np.array(embeddings))

    return index, embeddings

def retrieve(query, model, index, sentences, k=3):
    query_embed = model.encode([query])
    D, I = index.search(np.array(query_embed), k)
    print(f"D, I:{D, I}")
    return [sentences[i] for i in I[0]] # TODO: why is it taking only the sentence in index 0?!
    # e.x. D, I:(array([[1.8982642, 1.9757972, 1.982473 ]], dtype=float32), array([[ 0, 10,  6]]))

def ask_gemini(context, question):
    prompt = f"""Use the following context to answer the question clearly.

    Context:
    {context}
    
    Question: {question}
    Answer:"""

    print("Prompt:", prompt)

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable "thinking"
        )
    )
    return response.text.strip()


def main():
    print("Loading documents...")
    sentences = load_documents()

    print("Creating FAISS index...")
    # use a minimal model for embeddings, no need to use the main LLM for doing that
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = create_faiss(sentences, model)

    while True:
        q = input("\nAsk something (or type 'exit'): ")
        if q.lower() == 'exit':
            break
        top_chunks = retrieve(q, model, index, sentences)
        context = "\n".join(top_chunks)
        print("\nRetrieved Context:\n", context)
        answer = ask_gemini(context, q)
        print("\nGemini Answer:\n", answer)

if __name__ == "__main__":
    main()