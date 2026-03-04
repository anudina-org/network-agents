import os
import requests
import streamlit as st

AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Network Agent - UAT", page_icon="🌐", layout="centered")

with st.sidebar:
    st.markdown("## 🌐 Network Agent")
    st.markdown("`UAT Environment`")
    st.divider()
    st.markdown("**Agents**")
    st.markdown("- `networksupervisor` — routes requests")
    st.markdown("- `site-agent` — fetches site details")
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

st.title("Network Supervisor Agent")
st.caption("UAT · Ask about your network sites")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your network sites..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Agent working...", expanded=True) as status:
            resp = requests.post(
                f"{AGENT_API_URL}/chat",
                json={"message": prompt, "history": st.session_state.history},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            for log in data["logs"]:
                status.write(log)
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(data["response"])

    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": data["response"]})
