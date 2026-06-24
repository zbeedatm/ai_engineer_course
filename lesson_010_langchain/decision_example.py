import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# -----------------------------------------------
# 1️ Load OpenAI key securely
# -----------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY as an environment variable")

llm = OpenAI(api_key=api_key, temperature=0)

# -----------------------------------------------
# 2️ Define Router Chain (Decision Node)
# -----------------------------------------------
router_prompt = PromptTemplate.from_template(
    "Classify this query as 'finance' or 'risk':\n\n{input}"
)
router_chain = LLMChain(llm=llm, prompt=router_prompt)

# -----------------------------------------------
# 3️ Define the Branches (Leaves)
# -----------------------------------------------
finance_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("You are a finance expert. Answer:\n\n{input}")
)

risk_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("You are a risk officer. Analyze:\n\n{input}")
)


# -----------------------------------------------
# 4️ Define Pipeline Function
# -----------------------------------------------
def run_decision_tree(query: str):
    print("🟩 INPUT QUERY:", query)
    route = router_chain.run(query).strip().lower()
    print(f"🔍 ROUTER DECISION: {route}")

    if "finance" in route:
        print("📈 Branch selected → Finance")
        result = finance_chain.run(query)
    elif "risk" in route:
        print("⚠️ Branch selected → Risk")
        result = risk_chain.run(query)
    else:
        print("❓ Unknown branch → Default: Finance")
        result = finance_chain.run(query)

    print("💬 FINAL OUTPUT:\n", result)
    print("-" * 60)


# -----------------------------------------------
# 5️ Test the Flow
# -----------------------------------------------
run_decision_tree("Should we approve a $10,000 loan for a customer with credit score 800?")
run_decision_tree("What are the biggest risks of lending to high-debt clients?")
