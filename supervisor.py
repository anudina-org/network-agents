import os
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Command

from site_agent import _make_llm, create_site_agent

MEMBERS = ["site-agent"]
OPTIONS = MEMBERS + ["FINISH"]

SUPERVISOR_PROMPT = (
    "You are a network supervisor agent.\n"
    "Available agents: site-agent (handles site details and site overview queries).\n"
    "Rules:\n"
    "- If the user needs site information and no agent has responded yet, reply with exactly: site-agent\n"
    "- If an agent has already responded with an answer, reply with exactly: FINISH\n"
    "Reply with only one word: either 'site-agent' or 'FINISH'. No explanation."
)


def _parse_next(text: str) -> str:
    text = text.strip().lower()
    for opt in MEMBERS:
        if opt in text:
            return opt
    return "FINISH"


def build_graph():
    llm = _make_llm()

    site_agent = create_site_agent()

    def supervisor_node(state: MessagesState) -> Command[Literal["site-agent", "__end__"]]:
        all_msgs = state["messages"]
        # Find index of the last user message (current turn start)
        last_user_idx = max(
            (i for i, m in enumerate(all_msgs)
             if getattr(m, "type", None) == "human" and not getattr(m, "name", None)),
            default=0,
        )
        # Only check messages produced AFTER the current user message
        if any(getattr(m, "name", None) == "site-agent" for m in all_msgs[last_user_idx + 1:]):
            return Command(goto=END)
        messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + all_msgs
        response = llm.invoke(messages)
        next_node = _parse_next(response.content)
        if next_node == "FINISH":
            return Command(goto=END)
        return Command(goto=next_node)

    def site_agent_node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = site_agent.invoke(state, config={"recursion_limit": 6})
        last = result["messages"][-1]
        return Command(
            update={"messages": [HumanMessage(content=last.content, name="site-agent")]},
            goto="supervisor",
        )

    workflow = StateGraph(MessagesState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("site-agent", site_agent_node)
    workflow.add_edge(START, "supervisor")

    return workflow.compile()
