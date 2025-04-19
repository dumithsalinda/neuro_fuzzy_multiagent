import streamlit as st
from dashboard.login import login_form
from dashboard.collab import collaborative_experiments_panel
# (Other imports for sidebar, layout, simulation, etc. will be added in the next steps)

def main():
    st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
    if not st.session_state.get("authenticated", False):
        login_form()
    else:
        collaborative_experiments_panel()
        # --- Group Policy Controls ---
        st.sidebar.markdown("---")
        st.sidebar.header("Group Policy Controls")
        enable_group_knowledge = st.sidebar.checkbox("Enable Group Knowledge Sharing", value=False, key="enable_group_knowledge")
        collective_mode = st.sidebar.selectbox("Collective Action Mode", ["individual", "leader", "vote"], index=0, key="collective_mode")
        share_now = st.sidebar.button("Share Group Knowledge Now")
        st.session_state["enable_group_knowledge"] = enable_group_knowledge
        st.session_state["collective_mode"] = collective_mode
        # --- Simulation Controls ---
        st.markdown("---")
        st.header("Simulation Controls")
        from dashboard.simulation import simulate_step, som_group_agents
        from dashboard.visualization import render_agent_positions, render_group_modules, render_group_knowledge
        # Batch controls
        n_steps = st.number_input("Steps to Run", min_value=1, max_value=1000, value=10, step=1, key="n_steps")
        run_n_steps = st.button("Run N Steps")
        auto_run = st.checkbox("Auto-Run Simulation", value=False, key="auto_run")
        pause_run = st.button("Pause Auto-Run")
        # Simulation progress/status
        st.write(f"Current Step: {st.session_state.get('step', 0)}")
        st.write(f"Simulation Running: {'Yes' if st.session_state.get('sim_running', False) else 'No'}")
        # Handle batch and auto-run logic
        if run_n_steps:
            for _ in range(n_steps):
                simulate_step()
        if auto_run and not st.session_state.get('sim_running', False):
            st.session_state['sim_running'] = True
        if pause_run:
            st.session_state['sim_running'] = False
        # Auto-run loop
        import time
        if st.session_state.get('sim_running', False) and auto_run:
            simulate_step()
            time.sleep(0.5)
            st.experimental_rerun()
        # --- SOM Grouping Section ---
        st.markdown("---")
        st.header("Dynamic Agent Grouping (SOM)")
        if st.button("Re-cluster Agents using SOM"):
            som_group_agents()
        # Show agent positions and group assignments if available
        agents = st.session_state.get("agents", [])
        mas = st.session_state.get("multiagent_system")
        som = None
        if hasattr(mas, 'last_som'):
            som = mas.last_som
        if agents and mas:
            positions = [getattr(agent, 'position', None) for agent in agents]
            render_agent_positions(positions, agents)
            render_group_modules(mas.group_modules)
            # Show group-level knowledge if enabled
            if enable_group_knowledge or share_now:
                render_group_knowledge(mas)
            # --- SOM Grid Visualization ---
            try:
                from src.core.som_cluster import SOMClusterer
                # Try to get the SOMClusterer used for last grouping
                som = getattr(mas, 'last_som', None)
                if som is not None:
                    render_som_grid_with_agents(mas, som)
            except Exception as e:
                st.info(f"SOM grid visualization unavailable: {e}")
            # --- Group Analytics ---
            try:
                from dashboard import simulation
                reward_history = getattr(simulation, 'reward_history', None)
                render_group_analytics(mas, reward_history=reward_history)
            except Exception as e:
                st.info(f"Group analytics unavailable: {e}")
        # (Other dashboard panels and logic will be called here)


if __name__ == "__main__":
    main()
