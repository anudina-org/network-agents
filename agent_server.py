from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from supervisor import build_graph


def cprint(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


class Logger(BaseCallbackHandler):
    def __init__(self, emit):
        self._emit = emit

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._emit(f"💭 {(serialized or {}).get('name', 'LLM')} reasoning...")

    def on_llm_end(self, response, **kwargs):
        try:
            gen = response.generations[0][0]
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            inp, out = meta.get("input_tokens"), meta.get("output_tokens")
            if inp is not None or out is not None:
                total = meta.get("total_tokens", (inp or 0) + (out or 0))
                self._emit(f"📊 Tokens — in: {inp}, out: {out}, total: {total}")
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._emit(f"🔧 Tool → {(serialized or {}).get('name', 'tool')}() called")

    def on_tool_end(self, output, **kwargs):
        self._emit("📡 API → response received")

    def on_llm_error(self, error, **kwargs):
        self._emit(f"❌ LLM error: {error}")

    def on_tool_error(self, error, **kwargs):
        self._emit(f"❌ Tool error: {error}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logs = []
    cprint(f"── New request ── {req.message}")

    def emit(msg: str):
        logs.append(msg)
        cprint(msg)

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
        config={"callbacks": [Logger(emit)], "recursion_limit": 10},
    ):
        final_state = state
        for m in state["messages"][len(messages):]:
            if getattr(m, "name", None) == "site-agent":
                emit("✅ Site Agent → response ready")
            elif getattr(m, "type", "") == "tool":
                emit("📦 Tool result → passed to agent")

    response = next(
        (m for m in reversed(final_state["messages"]) if getattr(m, "name", None) == "site-agent"),
        final_state["messages"][-1],
    )
    content = response.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)

    cprint(f"── Response sent ({len(content)} chars) ──")
    return ChatResponse(response=content, logs=logs)
