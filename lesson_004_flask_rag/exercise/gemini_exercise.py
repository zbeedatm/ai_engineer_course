import os
import re
import json
import faiss
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

# Download NLTK resources (run once)
nltk.download("punkt")
nltk.download("wordnet")


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

"""
The free tier has strict limits:

Model	            Free Tier Daily Requests
gemini‑2.5‑flash	    20
gemini‑1.5‑flash    	50
gemini‑1.5‑pro	    5
"""
# MODEL = "models/gemini-1.5-flash"
MODEL = "models/gemini-2.5-flash"

# EMBED_MODEL = "text-embedding-004"
# SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

base_dir = os.path.dirname(__file__)

#1 ----------------------------------------------------------------------------
# Simple Text Completion

def ask_gemini(prompt):
    """Simple one‑shot text generation."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable "thinking"
            )
        )
        print(response.text)

        return response.text.strip()
    except Exception as e:
        print("Error during text completion:", e)

#2 ----------------------------------------------------------------------------
# Chat Memory Simulation

def chat_simulation():
    # Manual memory
    history = []  # list of genai.types.Content objects

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Convert user message into a Content object
        user_msg = genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=user_input)]
            # parts=[genai.types.Part.from_text(user_input)]
        )
        history.append(user_msg)

        # Send full history
        reply_text = ask_gemini(history)

        # Convert model reply into a Content object
        model_msg = genai.types.Content(
            role="model",
            parts=[genai.types.Part(text=reply_text)]
            # parts=[genai.types.Part.from_text(reply_text)]
        )
        history.append(model_msg)

        print("Gemini:", reply_text)

#3 ----------------------------------------------------------------------------
# Generate Marketing Content

def generate_marketing_content(product, audience):
    # Prompt template
    prompt = f"""
    You are a marketing copywriter.

    Generate the following for the product: "{product}"
    Target audience: "{audience}"

    1. A short, vivid product description (2–3 sentences)
    2. Three catchy ad slogans (bullet list)
    3. One strong call-to-action
    """

    print("Prompt:", prompt)

    # response = client.models.generate_content(
    #     model=MODEL,
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable "thinking"
    #     )
    # )
    # return response.text.strip()

    return ask_gemini(prompt)

#4 ----------------------------------------------------------------------------
# Summerize a Web Page

def fetch_page_text(url: str) -> str:
    """Fetches a webpage and extracts readable text."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, and irrelevant tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Clean whitespace
    cleaned = " ".join(text.split())
    return cleaned


def summarize_with_gemini(text: str) -> str:
    """Send extracted text to Gemini for summarization."""
    prompt = f"""
    Summarize the following webpage content in clear, concise language.
    Focus on the main ideas only.
    
    Content:
    {text}
    """

    # response = client.models.generate_content(
    #     model=MODEL,
    #     contents=prompt
    # )
    # return response.text

    return ask_gemini(prompt)

#5 ----------------------------------------------------------------------------
# AI Powered Code Reviewer

def load_python_file(path: str) -> str:
    """Reads a Python file and returns its full text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def review_code_with_gemini(code: str) -> str:
    """Sends code to Gemini for structured review."""
    prompt = f"""
    You are an expert code reviewer.
    
    Review the following code for:
    1. Readability
    2. Efficiency
    3. Code smells or anti-patterns
    4. Suggested improvements
    5. Optional refactoring ideas
    
    Provide your feedback in a structured, sectioned format.
    
    --- CODE START ---
    {code}
    --- CODE END ---
    """

    # response = client.models.generate_content(
    #     model=MODEL,
    #     contents=prompt
    # )
    # return response.text

    return ask_gemini(prompt)

#6 ----------------------------------------------------------------------------
# Multi-Modal Caption Generator (Text + Image)

def load_image(path: str):
    """Loads an image file as bytes."""
    with open(path, "rb") as f:
        return f.read()


def generate_caption(image_bytes):
    # Upload the image first
    # uploaded = client.files.upload_bytes(
    #     data=image_bytes,
    #     mime_type="image/jpeg"
    # )

    """Send image + text prompt to Gemini."""
    prompt = """
    You are an image captioning assistant.
    
    Given the image, generate:
    1. A descriptive caption (1–2 sentences)
    2. Three relevant hashtags
    
    Respond in this format:
    
    Caption: <caption>
    Hashtags:
    - #tag1
    - #tag2
    - #tag3
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
            # 'Caption this image.'
        ]

    )

    return response.text

#7 ----------------------------------------------------------------------------
# CSV Data Insights

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def dataframe_to_text(df: pd.DataFrame) -> str:
    """Convert a DataFrame into a structured text summary."""
    text = []

    text.append("COLUMNS:")
    text.append(", ".join(df.columns))

    text.append("\nBASIC STATS:")
    text.append(df.describe(include="all").to_string())

    text.append("\nSAMPLE ROWS:")
    text.append(df.head(10).to_string())

    return "\n".join(text)


def ask_gemini_for_insights(structured_text: str) -> str:
    """Send structured CSV text to Gemini for insights."""
    prompt = f"""
    You are a data analyst.
    
    Given the dataset summary below, identify:
    1. Key trends
    2. Notable patterns
    3. Outliers or anomalies
    4. Possible business insights
    5. Suggestions for further analysis
    
    DATA SUMMARY:
    {structured_text}
    """

    # response = client.models.generate_content(
    #     model=MODEL,
    #     contents=prompt
    # )
    # return response.text

    return ask_gemini(prompt)

#8 ----------------------------------------------------------------------------
# RAG style document Q&A

# 1. Load + preprocess document
def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

'''
🧠 Why you should NOT lemmatize or tokenize manually?!

Lemmatization destroys semantic meaning for embedding models.
Example:
- “running” → “run”
- “better” → “good”
- “systems uptime” → “system uptime”
Embedding models are trained on natural text, not lemmatized text.
So lemmatization reduces retrieval accuracy.

Sentence tokenization is also unnecessary — chunking handles it.
'''
######## Optimized solution
import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_text(text, max_tokens=300, overlap=50):
    doc = nlp(text)
    chunks = []
    current = []
    token_count = 0

    for sent in doc.sents:
        sent_tokens = len(sent)
        if token_count + sent_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            token_count = sum(len(nlp(s)) for s in current)

        current.append(sent.text)
        token_count += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks

def build_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, index, chunks, embedding_model, k=3):
    query_embed = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_embed, k)
    return [chunks[i] for i in I[0]]


######## Old solution
def preprocess_text(text: str, lemmatize=False):
    """Tokenize into sentences and optionally lemmatize."""
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)

    if not lemmatize:
        return sentences

    processed = []
    for sent in sentences:
        words = word_tokenize(sent)
        lemmas = [lemmatizer.lemmatize(w) for w in words]
        processed.append(" ".join(lemmas))

    return processed

# 2. Chunk sentences into ~200-token chunks
def chunk_sentences(sentences, max_tokens=200):
    chunks = []
    current = []
    token_count = 0

    for sent in sentences:
        tokens = word_tokenize(sent)
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(current))
            current = []
            token_count = 0

        current.append(sent)
        token_count += len(tokens)

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_faiss_index(chunks):
    # model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    embeddings = embedding_model.encode(chunks)
    dim = embeddings.shape[1]  # the dimension number of the embedding to be used as index!
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve(query, index, chunks, k=3):
    query_embed = embedding_model.encode([query])
    D, I = index.search(np.array(query_embed), k)
    # print(f"D, I:{D, I}")
    return [chunks[i] for i in I[0]]


# 5. Ask Gemini with retrieved context
def ask_with_context(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant. Answer the question **using ONLY the context below**.
    If the answer is not in the context, say: "The document does not contain that information."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    """

    # response = client.models.generate_content(
    #     model=MODEL,
    #     contents=prompt
    # )
    # return response.text

    return ask_gemini(prompt)


#9 ----------------------------------------------------------------------------
# Emotion Analysis of Reviews

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def analyze_reviews(reviews):
    """
    Send a list of reviews to Gemini and get structured JSON output.
    """

    prompt = f"""
    You are an emotion and sentiment analysis model.
    
    For each review below, return:
    - dominant_emotion: one of ["joy", "anger", "sadness", "fear", "surprise", "neutral"]
    - sentiment_score: integer 1–5 (1 = very negative, 5 = very positive)
    - short_reason: one-sentence explanation
    
    Return ONLY valid JSON in this format:
    
    {{
      "results": [
        {{
          "review": "...",
          "dominant_emotion": "...",
          "sentiment_score": 1,
          "short_reason": "..."
        }}
      ]
    }}
    
    Reviews:
    {reviews}
    """

    # schema = types.Schema(
    #     type=types.Type.OBJECT,
    #     properties={
    #         "results": types.Schema(
    #             type=types.Type.ARRAY,
    #             items=types.Schema(
    #                 type=types.Type.OBJECT,
    #                 properties={
    #                     "review": types.Schema(type=types.Type.STRING),
    #                     "dominant_emotion": types.Schema(type=types.Type.STRING),
    #                     "sentiment_score": types.Schema(type=types.Type.INTEGER),
    #                     "short_reason": types.Schema(type=types.Type.STRING),
    #                 },
    #                 required=[
    #                     "review",
    #                     "dominant_emotion",
    #                     "sentiment_score",
    #                     "short_reason",
    #                 ],
    #             ),
    #         )
    #     },
    #     required=["results"],
    # )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        ### The new Gemini SDK does not support structured output yet!
        # All the following didn't work
        # way 1
        # config=types.GenerateConfig(response_schema=schema, response_mime_type="application/json")
        # way 2
        # schema = schema, mime_type = "application/json"
        # way 3
        # response_schema=schema,
        # generation_config=types.GenerationConfig(response_mime_type="application/json")
    )

    # Parse JSON safely
    try:
        json_str = extract_json(response.text)

        if json_str:
            return json.loads(json_str)
        else:
            print("Model returned no JSON")

        # return json.loads(response.text)
    except json.JSONDecodeError:
        print("Model returned invalid JSON. Raw output:")
        print(response.text)
        return None

#10 ----------------------------------------------------------------------------
# A Gemini-Powered Chatbot (Flask API)

def send_message_to_chatbot(message: str):
    url = "http://127.0.0.1:5000/chat"
    payload = {"message": message}

    response = requests.post(url, json=payload)
    return response.json()


###################################################################################


def main():
    while True:
        print("""
        1 - Simple Text Completion
        2 - Chat (with memory)
        3 - Generate Marketing Content
        4 - Summarize a Web Page
        5 - Code Reviewer
        6 - Multi-Modal Caption Generator (Text + Image)
        7 - CSV Data Insights 
        8 - RAG-Style Document Q&A 
        9 - Emotion Analysis of Reviews
        10 - Chatbot Flask API
        Q - Quite
        """)

        choice = input("Enter your choice: ").strip().upper()

        if choice == "1":
            q = input("\nAsk something: ")
            ask_gemini(q)

        elif choice == "2":
            chat_simulation()

        elif choice == "3":
            product = input("\nEnter product name: ")
            audience = input("Enter target audience: ")
            print(generate_marketing_content(product, audience))

        elif choice == "4":
            url = input("\nEnter a URL to summarize: ")
            print("Fetching and cleaning page...")
            page_text = fetch_page_text(url)
            print(summarize_with_gemini(page_text))

        elif choice == "5":
            q = input("\nUse a default path for your code file (otherwise you need to enter it manually)? y/n: ")
            if q.lower() == "y":
                path = os.path.abspath(os.path.join(base_dir, "gemini_exercise.py"))
            else:
                path = input("\nEnter path to Python file: ").strip()

            print("Loading file...")
            code = load_python_file(path)
            print("Sending to Gemini for review...\n")
            review = review_code_with_gemini(code)
            print("=== Code Review ===\n")
            print(review)

        elif choice == "6":
            q = input("\nUse a default image (otherwise you need to enter a path to your image manually)? y/n: ")
            if q.lower() == "y":
                path = os.path.abspath(os.path.join(base_dir, "../../data/cats.jpg"))
            else:
                path = input("\nEnter image path (e.g., cat.jpg): ").strip()

            print("Loading image...")
            img = load_image(path)

            print("Generating caption...\n")
            result = generate_caption(img)

            print("=== Output ===\n")
            print(result)

        elif choice == "7":
            q = input("\nUse a default path for your data file (otherwise you need to enter it manually)? y/n: ")
            if q.lower() == "y":
                path = os.path.abspath(os.path.join(base_dir, "../../data/sales.csv"))
            else:
                path = input("\nEnter CSV file path: ").strip()

            print("Loading CSV...")
            df = load_csv(path)

            print("Preparing structured text...")
            structured = dataframe_to_text(df)

            print("Sending to Gemini...\n")
            insights = ask_gemini_for_insights(structured)

            print("=== Insights ===\n")
            print(insights)

        elif choice == "8":
            q = input("\nUse a default path for your data file (otherwise you need to enter it manually)? y/n: ")
            if q.lower() == "y":
                path = os.path.abspath(os.path.join(base_dir, "../../data/report.txt"))

                # print(os.listdir(os.path.abspath(os.path.join(base_dir, "../../data"))))
                # path = "../../data/report.txt"
            else:
                path = input("\nEnter document path (e.g., report.txt): ").strip()

            print("Loading document...")
            text = load_document(path)

            print("Preprocessing with NLTK...")
            sentences = preprocess_text(text, lemmatize=False)

            print("Chunking...")
            chunks = chunk_sentences(sentences)

            print("Building FAISS index...")
            index, _ = build_faiss_index(chunks)

            print("\nRAG system ready.\n")

            while True:
                question = input("Ask a question (or 'exit'): ")
                if question.lower() == "exit":
                    break

                print("Retrieving relevant chunks...")
                context = retrieve(question, index, chunks)

                print("\nGemini's answer:\n")
                answer = ask_with_context(question, context)
                print(answer)

        elif choice == "9":
            reviews = [
                "The battery life is amazing, lasts all day!",
                "Terrible build quality. Broke after one week.",
                "It's okay, nothing special but not bad either.",
                "Customer service was incredibly helpful.",
                "The product arrived damaged and late."
            ]

            result = analyze_reviews(reviews)

            print("\n=== Emotion + Sentiment Analysis ===\n")
            print(json.dumps(result, indent=2))

        elif choice == "10":
            while True:
                user_input = input("You: ")

                if user_input.lower() == "exit":
                    break

                result = send_message_to_chatbot(user_input)
                print("Bot:", result.get("reply"))

        elif choice == "Q":
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
