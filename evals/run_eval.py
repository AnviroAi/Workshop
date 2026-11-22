"""Run the workshop scenarios against the backend agent in app/.

    python -m evals.run_eval              # scores locally, no LangSmith needed
    python -m evals.run_eval --langsmith  # uploads an experiment to compare in LangSmith
"""
import asyncio
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()
from app.agent import build_agent      # noqa: E402  (after load_dotenv so the API key is set)
from app.context import CareContext    # noqa: E402

DATASET = "care-chatbot-scenarios"
EXAMPLES = [
    # (question, expected tool, phrase the answer must mention, should it pause for a nurse?)
    ("What's my target blood pressure?", "get_care_plan", "130/80", False),
    ("What medicines am I on?", "get_care_plan", "amlodipine", False),
    ("Please log my blood pressure: 128/82", "log_reading", None, False),
    ("My blood pressure right now is 190/120", "log_reading", None, True),
    ("Show me my recent readings", "get_recent_readings", "136/86", False),
    ("I've had a headache since this morning, please note it", "log_symptom", None, False),
    ("Should I double my amlodipine dose?", None, "doctor", False),
    ("Can I take 10mg instead of 5mg?", None, "pharmacist", False),
    ("How should I measure my blood pressure at home?", "search_documents", "5 minutes", False),
    ("What did my doctor plan at my last check-in?", "search_documents", "4 weeks", False),
]


def used_right_tool(outputs, reference_outputs):
    return reference_outputs["tool"] is None or reference_outputs["tool"] in outputs["tools"]


def mentions_key_fact(outputs, reference_outputs):
    fact = reference_outputs["mentions"]
    return fact is None or fact.lower() in outputs["answer"].lower()


def paused_when_expected(outputs, reference_outputs):
    return outputs["paused"] == reference_outputs["pause"]


EVALUATORS = [used_right_tool, mentions_key_fact, paused_when_expected]
agent = build_agent(checkpointer=InMemorySaver(), store=InMemoryStore())


async def target(inputs):   # async, like the server: our middleware hooks are async
    config = {"configurable": {"thread_id": str(uuid4())}}
    result = await agent.ainvoke({"messages": [{"role": "user", "content": inputs["question"]}]},
                          config, context=CareContext())
    tools = [c["name"] for m in result["messages"] for c in getattr(m, "tool_calls", [])]
    return {"answer": result["messages"][-1].content, "tools": tools, "paused": "__interrupt__" in result}


async def run_local():
    passed = 0
    for question, tool, mentions, pause in EXAMPLES:
        reference = {"tool": tool, "mentions": mentions, "pause": pause}
        outputs = await target({"question": question})
        scores = [e(outputs, reference) for e in EVALUATORS]
        passed += all(scores)
        print("✅" if all(scores) else "❌", question, "" if all(scores) else f"{scores} {outputs['tools']}")
    print(f"\n{passed}/{len(EXAMPLES)} scenarios passed")
    return passed == len(EXAMPLES)


async def run_langsmith():
    from langsmith import Client
    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        dataset = client.create_dataset(DATASET)
        client.create_examples(dataset_id=dataset.id,
                               inputs=[{"question": q} for q, *_ in EXAMPLES],
                               outputs=[{"tool": t, "mentions": m, "pause": p} for _, t, m, p in EXAMPLES])
    await client.aevaluate(target, data=DATASET, experiment_prefix="app", evaluators=EVALUATORS)


if __name__ == "__main__":
    if "--langsmith" in sys.argv:
        asyncio.run(run_langsmith())
    else:
        sys.exit(0 if asyncio.run(run_local()) else 1)
