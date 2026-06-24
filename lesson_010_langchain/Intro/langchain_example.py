import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


def main():
    loader = TextLoader("../../data/ai_notes.txt", encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        retriever=vectorstore.as_retriever(),
    )

    query = "Explain the difference between LangChain and RAG"
    print(qa.run(query)) # LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain-classic 0.1.0 and will be removed in 1.0. Use `invoke` instead.


if __name__ == "__main__":
    main()
