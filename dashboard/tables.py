import streamlit as st
from typing import List, Any

def render_knowledge_table(agents: List[Any]) -> None:
    """
    Display a table showing each agent's knowledge state and law violations.
    """
    st.header("Agent Knowledge State")
    data = [
        {
            "Agent": getattr(agent, "group", "") + ":" + str(i),
            "Knowledge": str(getattr(agent, "online_knowledge", {})),
            "Law Violations": getattr(agent, "law_violations", 0),
        }
        for i, agent in enumerate(agents)
    ]
    st.table(data)

def render_group_decisions_log(group_decisions: List[Any]) -> None:
    """
    Display a log of recent group decisions, color-coded by legality.
    """
    st.header("Recent Group Decisions")
    for actions, result, legal in list(group_decisions)[-10:][::-1]:
        color = "green" if legal else "red"
        st.markdown(
            f"<span style='color:{color}'>Actions: {actions} 192 Result: {result} | {'Legal' if legal else 'Violated Law'}</span>",
            unsafe_allow_html=True,
        )
