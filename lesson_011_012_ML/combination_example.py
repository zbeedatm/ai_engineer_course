import os
import numpy as np
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document

load_dotenv()

# =========================================================
# 0) Setup
# =========================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in env")

llm = ChatOpenAI(api_key=api_key, temperature=0)
embeddings = OpenAIEmbeddings(api_key=api_key)

# =========================================================
# 1) Intent classifier (scikit-learn)
# =========================================================
train_texts = [
    # technical_issue
    "video not loading",
    "site is down",
    "cannot access the course page",
    "error 500 when opening lesson",
    "login not working",
    # content_help
    "I don't understand lesson 3 about Docker",
    "please explain RAG again",
    "need help with the Python exercise",
    "confused about Kubernetes homework",
    # admin_billing
    "I was double charged",
    "need invoice for the course",
    "want refund",
    "payment went through but no access",
]
train_labels = [
    "technical_issue",
    "technical_issue",
    "technical_issue",
    "technical_issue",
    "technical_issue",
    "content_help",
    "content_help",
    "content_help",
    "content_help",
    "admin_billing",
    "admin_billing",
    "admin_billing",
    "admin_billing",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_texts)

clf = LogisticRegression()
clf.fit(X, train_labels)


def classify_intent(message: str) -> str:
    X_test = vectorizer.transform([message])
    predicted = clf.predict(X_test)[0]
    proba = float(max(clf.predict_proba(X_test)[0]))
    print(f"🧮 Intent: {predicted} (confidence={proba:.2f})")
    if proba < 0.5:
        return "unknown"
    return predicted


# =========================================================
# 2) RAG with FAISS (internal knowledge)
# =========================================================
docs = [
    Document(
        page_content=(
            "TECH: If a student cannot access a lesson after payment, "
            "ask them to log out and log in, verify payment in the CRM, "
            "and sync their enrollment. If persists, escalate."
        ),
        metadata={"category": "technical_issue"},
    ),
    Document(
        page_content=(
            "TECH: Common video issues: clear cache, try incognito, "
            "check adblock/VPN, test another device."
        ),
        metadata={"category": "technical_issue"},
    ),
    Document(
        page_content=(
            "CONTENT: For RAG confusion, explain: it retrieves chunks "
            "from a vector store using embeddings, then augments the prompt."
        ),
        metadata={"category": "content_help"},
    ),
    Document(
        page_content=(
            "ADMIN: Refund policy: 14 days from purchase if less than 20% "
            "of the course watched. Billing issues go to finance@your-college.com."
        ),
        metadata={"category": "admin_billing"},
    ),
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(k=3)  # Runnable-style retriever


# =========================================================
# 3) LangChain prompt (decision + answer)
# =========================================================
triage_prompt = PromptTemplate.from_template("""
You are a support assistant for an online tech academy.

Student message:
{message}

Detected intent: {intent}

Relevant internal documentation:
{context}

Your tasks:
1. Answer the student clearly and politely.
2. If you can fully solve it with the docs, keep it concise and practical.
3. If this is a billing/admin issue OR docs are not enough, say that you will escalate
   and mark escalate = true.
4. Respond in JSON with fields:
   "intent", "answer", "escalate" (true/false).

Only output valid JSON.
""")

triage_chain = LLMChain(llm=llm, prompt=triage_prompt)


# =========================================================
# 4) Main function (MCP-ready)
# =========================================================
def course_support_assistant(message: str) -> str:
    print("🟩 New ticket:", message)
    print("-" * 50)

    # 1) Classify intent (ML decision)
    intent = classify_intent(message)

    # 2) Retrieve docs (RAG)
    retrieval_query = f"{message} [intent: {intent}]"
    retrieved_docs = retriever.invoke(retrieval_query)

    print("📚 Retrieved docs:")
    for d in retrieved_docs:
        cat = d.metadata.get("category", "unknown")
        print(f"- ({cat}) {d.page_content[:80]}...")

    context = "\n\n".join(d.page_content for d in retrieved_docs)

    # 3) LLM: merge signal into final structured response
    result = triage_chain.run(message=message, intent=intent, context=context)

    print("\n🤖 Copilot output:")
    print(result)
    print("=" * 50)
    return result


# =========================================================
# 5) Demo
# =========================================================
if __name__ == "__main__":
    msg1 = "Hi, I paid for the Docker course but I still can't access lesson 2."
    course_support_assistant(msg1)

    msg2 = "I don't understand how RAG works in the last AI lesson."
    course_support_assistant(msg2)

    msg3 = "I was double charged for the Python course. I want a refund."
    course_support_assistant(msg3)
