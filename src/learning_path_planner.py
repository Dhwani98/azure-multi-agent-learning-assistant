
import json
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import os
from dotenv import load_dotenv

load_dotenv()

def build_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    )
    return kernel

PLANNER_PROMPT = """
You are a personalized learning-path planner.

Learner profile:
{{$profile}}

Available topics:
{{$topics}}

Create a short next-step learning plan with:
1. next_topic
2. reason
3. lesson_objectives
4. suggested_quiz_focus

Return JSON only.
"""

async def plan_next_step(profile: dict, topics: list[str]) -> str:
    kernel = build_kernel()
    func = kernel.add_function(
        function_name="plan_learning_path",
        plugin_name="planner",
        prompt=PLANNER_PROMPT,
    )

    result = await kernel.invoke(
        func,
        profile=json.dumps(profile, indent=2),
        topics=json.dumps(topics, indent=2),
    )
    return str(result)