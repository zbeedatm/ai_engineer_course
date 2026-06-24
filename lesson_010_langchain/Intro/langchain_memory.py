import os
from dotenv import load_dotenv

load_dotenv()

# from langchain_openai import OpenAI
# from langchain_classic.chains import ConversationChain
# from langchain_classic.memory import ConversationBufferMemory
#
# # LLM
# llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0)
#
# # Memory (keeps ALL turns)
# memory = ConversationBufferMemory(return_messages=True)
#
# # Conversation chain
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True,  # shows the prompt being sent
# )
#
# print("Chat started. Type 'exit' to stop.\n")
#
# while True:
#     user_msg = input("You: ")
#     if user_msg.lower() in ("exit", "quit", "q"):
#         print("Bye")
#         break
#
#     # send to chain
#     reply = conversation.predict(input=user_msg)
#     print("Bot:", reply)
#
#     # show current memory
#     print("\n--- MEMORY NOW ---")
#     mem_vars = memory.load_memory_variables({})
#     # mem_vars["history"] is a list of messages (because return_messages=True)
#     for m in mem_vars["history"]:
#         role = m.type  # 'human' or 'ai'
#         print(f"{role.upper()}: {m.content}")
#     print("------------------\n")


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# -----------------------------
# 1. Memory store for all users
# -----------------------------
memory_store = {}

def get_memory(session_id: str):
    """Return (or create) a chat history object for a given session."""
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

# -----------------------------
# 2. Model
# -----------------------------
model = ChatOpenAI(model="gpt-4o-mini")  # or any model you prefer

# -----------------------------
# 3. Prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# -----------------------------
# 4. Chain with memory
# -----------------------------
conversation_chain = RunnableWithMessageHistory(
    prompt | model,
    get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# -----------------------------
# 5. Use the chain
# -----------------------------
session_id = "user123"

# First message
response1 = conversation_chain.invoke(
    {"input": "Hello, who are you?"},
    config={"configurable": {"session_id": session_id}}
)
print("Assistant:", response1)

# Second message (memory preserved)
response2 = conversation_chain.invoke(
    {"input": "What did I just ask you?"},
    config={"configurable": {"session_id": session_id}}
)
print("Assistant:", response2)

# Third message (memory still preserved)
response3 = conversation_chain.invoke(
    {"input": "Continue the conversation naturally."},
    config={"configurable": {"session_id": session_id}}
)
print("Assistant:", response3)

