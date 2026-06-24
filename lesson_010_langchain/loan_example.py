import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# -----------------------------------------------
# 1️ Setup
# -----------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY as an environment variable")

llm = OpenAI(api_key=api_key, temperature=0)

# -----------------------------------------------
# 2️ Define Level 1: Approval Router
# -----------------------------------------------
approval_prompt = PromptTemplate.from_template(
    "You are a loan officer. Based on the description, classify the loan decision as 'approve' or 'reject':\n\n{input}"
)
approval_chain = LLMChain(llm=llm, prompt=approval_prompt)

# -----------------------------------------------
# 3️ Define Level 2: If Approved → Risk Evaluation
# -----------------------------------------------
risk_prompt = PromptTemplate.from_template(
    "The loan has been approved. Determine if it's 'high-risk' or 'low-risk' based on details:\n\n{input}"
)
risk_chain = LLMChain(llm=llm, prompt=risk_prompt)

# -----------------------------------------------
# 4️ Define Level 2: If Rejected → Reason Classification
# -----------------------------------------------
reason_prompt = PromptTemplate.from_template(
    "The loan was rejected. Decide the main reason: 'income', 'credit', or 'history':\n\n{input}"
)
reason_chain = LLMChain(llm=llm, prompt=reason_prompt)

# -----------------------------------------------
# 5️ Define Level 3: Detailed Explanations for Each Leaf
# -----------------------------------------------
explain_risk_prompt = PromptTemplate.from_template(
    "Explain why this is considered {risk_level} risk:\n\n{input}"
)
explain_risk_chain = LLMChain(llm=llm, prompt=explain_risk_prompt)

explain_reason_prompt = PromptTemplate.from_template(
    "Explain the reasoning for rejecting due to {reason}:\n\n{input}"
)
explain_reason_chain = LLMChain(llm=llm, prompt=explain_reason_prompt)


# -----------------------------------------------
# 6️ Combine All Layers
# -----------------------------------------------
def loan_decision_tree(query: str):
    print("🟩 INPUT QUERY:", query)
    decision = approval_chain.run(query).strip().lower()
    print(f"🔍 LEVEL 1 Decision: {decision}")

    if "approve" in decision:
        # Second layer: risk classification
        risk = risk_chain.run(query).strip().lower()
        print(f"⚖️ LEVEL 2 Risk classification: {risk}")

        # Third layer: explanation
        explanation = explain_risk_chain.run(
            {"risk_level": risk, "input": query}
        )
        print(f"💬 LEVEL 3 Explanation:\n{explanation}")

    elif "reject" in decision:
        # Second layer: reason classification
        reason = reason_chain.run(query).strip().lower()
        print(f"🚫 LEVEL 2 Rejection reason: {reason}")

        # Third layer: explanation
        explanation = explain_reason_chain.run(
            {"reason": reason, "input": query}
        )
        print(f"💬 LEVEL 3 Explanation:\n{explanation}")

    else:
        print("❓ Unknown outcome.")

    print("-" * 70)


# -----------------------------------------------
# 7️ Test the Full Decision Flow
# -----------------------------------------------
loan_decision_tree("Customer earns $8000/month with credit score 750 and stable job.")
loan_decision_tree("Customer earns $2000/month with credit score 580 and past loan defaults.")
