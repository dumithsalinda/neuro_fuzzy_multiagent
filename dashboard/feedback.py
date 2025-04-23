import streamlit as st

def render_feedback_panel():
    """
    Render the feedback panel for human-in-the-loop agent feedback and learning.
    Allows users to submit feedback on agent actions and view feedback history.
    """
    st.header("Agent Feedback Panel")
    agents = st.session_state.get("agents", [])
    feedback_history = st.session_state.get("feedback_history", [])
    if not agents:
        st.info("No agents available. Run a simulation step first.")
        return
    agent_options = [f"Agent {i}" for i in range(len(agents))]
    selected_agent_idx = st.selectbox("Select Agent to Provide Feedback", list(range(len(agents))), format_func=lambda i: agent_options[i])
    feedback_text = st.text_area("Enter feedback for the selected agent:")
    if st.button("Submit Feedback"):
        entry = {
            "agent": selected_agent_idx,
            "feedback": feedback_text,
            "step": st.session_state.get("step", None),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if "feedback_history" not in st.session_state:
            st.session_state["feedback_history"] = []
        st.session_state["feedback_history"].append(entry)
        st.success(f"Feedback submitted for Agent {selected_agent_idx}.")
    st.markdown("---")
    st.subheader("Feedback History")
    if st.session_state.get("feedback_history"):
        st.table(st.session_state["feedback_history"])
    else:
        st.info("No feedback submitted yet.")
