import streamlit as st

def render_knowledge_table(agents):
    st.header("Agent Knowledge State")
    data = [
        {
            "Agent": agent.group + ":" + str(i),
            "Knowledge": str(getattr(agent, "online_knowledge", {})),
            "Law Violations": getattr(agent, "law_violations", 0),
        }
        for i, agent in enumerate(agents)
    ]
    st.table(data)

def render_group_decisions_log(group_decisions):
    st.header("Recent Group Decisions")
    for actions, result, legal in list(group_decisions)[-10:][::-1]:
        color = "green" if legal else "red"
        st.markdown(
            f"<span style='color:{color}'>Actions: {actions} → Result: {result} | {'Legal' if legal else 'Violated Law'}</span>",
            unsafe_allow_html=True,
        )
