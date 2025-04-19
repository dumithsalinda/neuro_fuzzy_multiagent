"""
Streamlit Chat Panel for Human-Agent Collaboration
Allows user to interact with NeuroFuzzyFusionAgent via chat UI in the dashboard.
"""
import streamlit as st
import numpy as np
from src.core.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent

def chat_panel(agent):
    st.markdown("## 🤖 Agent Chat Panel")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("You:", "", key="chat_input")
    send = st.button("Send")
    if send and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        response = handle_agent_command(user_input, agent)
        st.session_state.chat_history.append(("agent", response))
        st.experimental_rerun()
    for speaker, msg in st.session_state.chat_history[-10:]:
        align = "left" if speaker == "user" else "right"
        st.markdown(f"<div style='text-align:{align};'><b>{speaker.title()}:</b> {msg}</div>", unsafe_allow_html=True)

def handle_agent_command(cmd, agent):
    cmd = cmd.lower().strip()
    if cmd == "help":
        return ("Commands:<br>"
                "- explain : Explain last action<br>"
                "- act : Take action on random obs<br>"
                "- set fusion alpha X : Set fusion alpha (0-1)<br>"
                "- show rules : Show fuzzy rules<br>"
                "- feedback [text] : Provide feedback to agent<br>")
    if cmd.startswith("set fusion alpha"):
        try:
            alpha = float(cmd.split()[-1])
            agent.set_fusion_alpha(alpha)
            return f"Fusion alpha set to {alpha}"
        except Exception:
            return "Usage: set fusion alpha X (where X is a float between 0 and 1)"
    if cmd == "act":
        obs = [np.random.rand(4), np.random.rand(3)]
        action = agent.act(obs)
        st.session_state["last_obs"] = obs
        st.session_state["last_action"] = action
        return f"Agent action: {action} (on random obs: {obs})"
    if cmd == "explain":
        obs = st.session_state.get("last_obs")
        if obs is None:
            return "No last observation. Use 'act' first."
        exp = agent.explain_action(obs)
        return "<br>".join([f"<b>{k}:</b> {v}" for k, v in exp.items()])
    if cmd == "show rules":
        rules = getattr(agent.fuzzy_system, 'rules', [])
        if not rules:
            return "No fuzzy rules defined."
        return "<br>".join([f"Rule {i}: Antecedents: {r.antecedents}, Consequent: {r.consequent}" for i, r in enumerate(rules)])
    if cmd.startswith("feedback"):
        feedback = cmd[len("feedback"):].strip()
        return f"Feedback received: '{feedback}' (not yet used for learning)"
    return "Unknown command. Type 'help' for options."
