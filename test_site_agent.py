from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from site_agent import create_site_agent

agent = create_site_agent()
history = []

print("Site Agent (direct). Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": history})
    history = result["messages"]
    print(f"\nAgent: {history[-1].content}\n")
