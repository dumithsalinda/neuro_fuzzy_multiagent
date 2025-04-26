import streamlit as st


def render_intervention_log() -> None:
    """
    Render the human intervention log if interventions are present in session state.
    """

    if "interventions" in st.session_state and st.session_state["interventions"]:
        st.markdown("---")
        st.header("Human Intervention Log")
        st.table(st.session_state["interventions"])


def render_interventions_panel() -> None:
    """
    Render the interventions panel, encapsulating all UI and logic for group batch edits,
    single-agent interventions, group module editing, and intervention log display.
    """
    import streamlit as st

    agents = st.session_state.get("agents", [])
    mas = st.session_state.get("multiagent_system")
    user_name = st.text_input("Your Name or Email (for attribution)", key="user_name")
    if agents and mas:
        agent_data = []
        for idx, agent in enumerate(agents):
            agent_data.append(
                {
                    "Agent": idx,
                    "Group": getattr(agent, "group", None),
                    "Knowledge": str(getattr(agent, "online_knowledge", {})),
                    "Law Violations": getattr(agent, "law_violations", 0),
                }
            )
        st.subheader("Agent Info Table")
        st.dataframe(agent_data)
        st.markdown("---")
        st.header("Group Module Batch Edit")
        for group_id, members in mas.groups.items():
            st.subheader(f"Group {group_id} (Members: {list(members)})")
            module_json = st.text_area(
                f"Module JSON for Group {group_id}",
                value=str(getattr(mas, "group_modules", {}).get(group_id, {})),
                key=f"module_json_{group_id}",
            )
            if st.button(
                f"Update Module for Group {group_id}", key=f"update_module_{group_id}"
            ):
                try:
                    module = eval(module_json)
                    if not hasattr(mas, "group_modules"):
                        mas.group_modules = {}
                    mas.group_modules[group_id] = module
                    log_entry = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "user": user_name,
                        "group": group_id,
                        "action": "update_module",
                        "module": module,
                    }
                    if "intervention_log" not in st.session_state:
                        st.session_state["intervention_log"] = []
                    st.session_state["intervention_log"].append(log_entry)
                    st.success(f"Module for Group {group_id} updated.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        st.markdown("---")
        st.header("Single-Agent Intervention")
        for idx, agent in enumerate(agents):
            st.subheader(f"Agent {idx} (Group: {getattr(agent, 'group', None)})")
            new_knowledge = st.text_area(
                f"Override Knowledge for Agent {idx}",
                value=str(getattr(agent, "online_knowledge", {})),
                key=f"override_knowledge_{idx}",
            )
            if st.button(f"Apply Override to Agent {idx}", key=f"override_agent_{idx}"):
                try:
                    agent.online_knowledge = eval(new_knowledge)
                    log_entry = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "user": user_name,
                        "agent": idx,
                        "action": "override_knowledge",
                        "knowledge": new_knowledge,
                    }
                    if "intervention_log" not in st.session_state:
                        st.session_state["intervention_log"] = []
                    st.session_state["intervention_log"].append(log_entry)
                    st.success(f"Knowledge for Agent {idx} overridden.")
                except Exception as e:
                    st.error(f"Invalid knowledge format: {e}")
        st.markdown("---")
        st.header("Intervention Log")
        if (
            "intervention_log" in st.session_state
            and st.session_state["intervention_log"]
        ):
            st.table(st.session_state["intervention_log"])
    else:
        st.info(
            "No agents or multi-agent system available for intervention. Run a simulation step first."
        )
