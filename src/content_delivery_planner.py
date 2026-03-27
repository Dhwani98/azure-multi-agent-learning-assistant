

from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from src.retrieve import retrieve

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

SYSTEM_PROMPT = """
You are a personalized learning assistant.

Behavior rules:
1. Use the retrieved source material as the primary grounding.
2. Answer in a way appropriate to the learner's level.
3. If the retrieved material is insufficient, say that clearly.
4. When relevant, connect the current answer to the previous conversation.
5. Be concise but useful.
6. End with a short follow-up suggestion or quiz question when appropriate.
"""

def build_user_prompt(user_query: str, learner_level: str, retrieved_docs: list[dict]) -> str:
    if retrieved_docs:
        context = "\n\n".join(
            [f"[Source: {d.get('source_file', 'unknown')}]\n{d.get('text', '')}" for d in retrieved_docs]
        )
    else:
        context = "No relevant source material found."

    return f"""
Learner level: {learner_level}

User question:
{user_query}

Retrieved source material:
{context}

Answer the user's question using the retrieved material.
"""

def answer_user_query(
    user_query: str,
    learner_level: str = "beginner",
    conversation_history: list | None = None,
    top_k: int = 5,
):
    if conversation_history is None:
        conversation_history = []

    retrieved_docs = retrieve(user_query, top_k=top_k)
    user_prompt = build_user_prompt(user_query, learner_level, retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    updated_history = conversation_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer},
    ]

    return {
        "answer": answer,
        "sources": retrieved_docs,
        "conversation_history": updated_history,
    }