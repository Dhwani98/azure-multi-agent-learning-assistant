#--------------------------------------------------------------
import json
import asyncio
from pathlib import Path

from src.content_delivery_planner import answer_user_query
from src.learning_path_planner import plan_next_step

BASE_DIR = Path(__file__).resolve().parent.parent
PROFILE_PATH = BASE_DIR / "data" / "learner_profiles" / "learner_001.json"
SESSION_PATH = BASE_DIR / "data" / "learner_state.json"

TOPICS = [
    "Azure AI Search",
    "RAG",
    "Semantic Kernel",
    "Microsoft Foundry Agent Service",
    "Blob indexing"
]

def load_profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

def load_session():
    if SESSION_PATH.exists():
        return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    return {"conversation_history": []}

def save_session(session_data: dict):
    SESSION_PATH.write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

async def main():
    profile = load_profile()
    session = load_session()
    learner_level = profile.get("preferred_level", "beginner")

    print("=== Personalized Learning Assistant ===")
    print("Type 'exit' to quit.")
    print("Type 'plan' to generate the next-step learning plan.")
    print("Type 'reset' to clear session memory.\n")

    while True:
        user_query = input("You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit"}:
            save_session(session)
            print("Session saved. Goodbye!")
            break

        if user_query.lower() == "reset":
            session = {"conversation_history": []}
            save_session(session)
            print("Session memory cleared.\n")
            continue

        if user_query.lower() == "plan":
            try:
                plan = await plan_next_step(profile, TOPICS)
                print("\n=== LEARNING PLAN ===")
                print(plan)
                print()
            except Exception as e:
                print(f"\nError while generating learning plan: {e}\n")
            continue

        try:
            result = answer_user_query(
                user_query=user_query,
                learner_level=learner_level,
                conversation_history=session.get("conversation_history", []),
                top_k=5,
            )

            print("\nAssistant:")
            print(result["answer"])
            print()

            session["conversation_history"] = result["conversation_history"]
            save_session(session)

        except Exception as e:
            print(f"\nError while answering: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())