import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
import asyncio
import json

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


# ==========================================================
# 1. API key loaded from .env
# ==========================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ==========================================================
# 2. Models
# ==========================================================
EVALUATOR_MODEL = "gpt-5.5"
EMBEDDING_MODEL = "text-embedding-3-small"


# ==========================================================
# 3. Helper: extract numeric score from RAGAS result
# Some versions return an object with .value, others may return float.
# ==========================================================
def extract_score(result):
    if hasattr(result, "value"):
        return result.value

    try:
        return float(result)
    except Exception:
        return str(result)


# ==========================================================
# 4. Helper: patch RAGAS args for GPT-5.5 compatibility
# This fixes:
# - max_tokens error
# - temperature=0.01 error
# - other unsupported generation parameters
# ==========================================================
def patch_gpt55_ragas_llm(llm):
    if not hasattr(llm, "model_args") or llm.model_args is None:
        llm.model_args = {}

    # Remove arguments that can break GPT-5.5 through Chat Completions
    unsupported_or_problematic_args = [
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
    ]

    for arg in unsupported_or_problematic_args:
        llm.model_args.pop(arg, None)

    # GPT-5.x models use max_completion_tokens instead of max_tokens
    llm.model_args["max_completion_tokens"] = 4096

    return llm


# ==========================================================
# 5. Fake RAG examples
# In real usage, replace these with your real RAG outputs.
# ==========================================================
examples = [
    {
        "user_input": "What does WaveClear Focus Tracker do?",
        "response": (
            "WaveClear Focus Tracker helps players monitor focus-related gameplay "
            "patterns and provides feedback after matches."
        ),
        "retrieved_contexts": [
            "WaveClear Focus Tracker helps players monitor focus-related gameplay patterns and provides feedback after matches.",
            "WaveClear includes features such as TrueSight, Debrief, Goals, Training, and App Settings."
        ],
        "reference": (
            "WaveClear Focus Tracker helps players monitor focus-related gameplay "
            "patterns and provides feedback after matches."
        )
    },
    {
        "user_input": "What does WaveClear Focus Tracker do?",
        "response": (
            "WaveClear Focus Tracker predicts the weather and manages AWS billing costs."
        ),
        "retrieved_contexts": [
            "WaveClear Focus Tracker helps players monitor focus-related gameplay patterns and provides feedback after matches.",
            "WaveClear includes features such as TrueSight, Debrief, Goals, Training, and App Settings."
        ],
        "reference": (
            "WaveClear Focus Tracker helps players monitor focus-related gameplay "
            "patterns and provides feedback after matches."
        )
    }
]


# ==========================================================
# 6. Main async evaluation
# ==========================================================
async def main():
    # OpenAI async client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # RAGAS evaluator LLM
    llm = llm_factory(
        EVALUATOR_MODEL,
        client=client
    )

    # Apply GPT-5.5 compatibility patch
    llm = patch_gpt55_ragas_llm(llm)

    print("Final RAGAS LLM model_args:")
    print(llm.model_args)
    print()

    # Embeddings for AnswerRelevancy
    embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=client
    )

    # Initialize RAGAS metric objects
    faithfulness_scorer = Faithfulness(llm=llm)

    answer_relevancy_scorer = AnswerRelevancy(
        llm=llm,
        embeddings=embeddings
    )

    results = []

    for index, item in enumerate(examples, start=1):
        print(f"Evaluating example {index}...")

        faithfulness_result = await faithfulness_scorer.ascore(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"]
        )

        answer_relevancy_result = await answer_relevancy_scorer.ascore(
            user_input=item["user_input"],
            response=item["response"]
        )

        results.append({
            "example": index,
            "question": item["user_input"],
            "answer": item["response"],
            "faithfulness": extract_score(faithfulness_result),
            "answer_relevancy": extract_score(answer_relevancy_result)
        })

    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


# ==========================================================
# 7. Run
# ==========================================================
if __name__ == "__main__":
    asyncio.run(main())