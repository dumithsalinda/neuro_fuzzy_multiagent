import streamlit as st

def render_intervention_log():
    if "interventions" in st.session_state and st.session_state["interventions"]:
        st.markdown("---")
        st.header("Human Intervention Log")
        st.table(st.session_state["interventions"])
