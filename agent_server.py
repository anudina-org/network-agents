from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from supervisor import build_graph

app = FastAPI()
graph = build_graph()


class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    response: str
    logs: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logs = []

    class Logger(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            logs.append(f"💭 `{(serialized or {}).get('name', 'LLM')}` reasoning...")

        def on_tool_start(self, serialized, input_str, **kwargs):
            logs.append(f"🔧 Tool → `{(serialized or {}).get('name', 'tool')}()` called")

        def on_tool_end(self, output, **kwargs):
            logs.append("📡 API → response received")

        def on_llm_error(self, error, **kwargs):
            logs.append(f"❌ LLM error: {error}")

        def on_tool_error(self, error, **kwargs):
            logs.append(f"❌ Tool error: {error}")

    # Reconstruct full message history
    messages = [
        HumanMessage(content=m.content) if m.role == "user"
        else HumanMessage(content=m.content, name="site-agent")
        for m in req.history
    ]
    messages.append(HumanMessage(content=req.message))

    final_state = None
    for state in graph.stream(
        {"messages": messages},
        stream_mode="values",
        config={"callbacks": [Logger()], "recursion_limit": 10},
    ):
        final_state = state
        new_msgs = state["messages"][len(messages):]
        for m in new_msgs:
            if getattr(m, "name", None) == "site-agent":
                logs.append("✅ Site Agent → response ready")
            elif getattr(m, "type", "") == "tool":
                logs.append("📦 Tool result → passed to agent")

    response = next(
        (m for m in reversed(final_state["messages"]) if getattr(m, "name", None) == "site-agent"),
        final_state["messages"][-1],
    )
    return ChatResponse(response=response.content, logs=logs)
