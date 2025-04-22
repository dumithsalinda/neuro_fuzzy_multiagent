import json
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from dashboard.sidebar import render_sidebar
from src.core.agents.agent import Agent
from dashboard.chat import chat_panel
from dashboard.login import login_form
from dashboard.simulation import run_batch_experiments, simulate_step, som_group_agents
from dashboard.visualization import (
    render_agent_positions,
    render_group_analytics,
    render_group_knowledge,
    render_group_modules,
    render_som_grid_with_agents,
)
from src.core.agents.agent_registry import get_registered_agents
from src.core.agents.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent

# --- Unified Plug-and-Play Sidebar ---
def main():
    render_sidebar()  # Always show plug-and-play selection/config/hot-reload
    # --- Main Tabs ---
    tab_labels = [
        "Simulation Controls",
        "Batch Experiments",
        "Analytics",
        "Agent Chat",
        "Interventions",
        "Collaboration",
        "Settings",
        "Plugins & Docs",
    ]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        simulation_controls()
    with tabs[1]:
        batch_experiments()
    with tabs[2]:
        analytics()
    with tabs[3]:
        agent_chat()
    with tabs[4]:
        interventions()
    with tabs[5]:
        collaboration()
    with tabs[6]:
        settings()
    with tabs[7]:
        plugins_and_docs()

# (Keep all other functions as before, but ensure they use st.session_state for selected plugins/config)

def merge_logs(local_log, remote_log):
    """
    Merge two logs by unique (time, agent, group, user) tuple.

    Args:
    local_log (list): Local log entries.
    remote_log (list): Remote log entries.

    Returns:
    list: Merged log entries.
    """
    seen = set()
    merged = []
    for entry in local_log + remote_log:
        key = (
            entry.get("time"),
            entry.get("agent"),
            entry.get("group", None),
            entry.get("user", None),
        )
        if key not in seen:
            merged.append(entry)
            seen.add(key)
    merged.sort(key=lambda x: x.get("time", ""))
    return merged


def simulation_controls():
    """
    Simulation controls tab.
    """
    st.header("Simulation Controls")
    st.number_input(
        "Steps to Run", min_value=1, max_value=1000, value=10, step=1, key="n_steps"
    )
    st.button("Run N Steps", key="run_n_steps")
    st.checkbox("Auto-Run Simulation", value=False, key="auto_run")
    st.button("Pause Auto-Run", key="pause_run")
    # Place simulation logic and visualization here
    # Example: render_agent_positions, simulate_step, etc.
    # (Add your simulation panel code here)


def batch_experiments():
    """
    Batch experiments tab.
    """
    st.header("Batch/Parallel Experiments")
    n_experiments = st.number_input(
        "Number of Experiments",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        key="n_experiments",
    )
    agent_counts = st.text_input(
        "Agent Counts (comma-separated)", value="10,50,100", key="batch_agent_counts"
    )
    seeds = st.text_input(
        "Seeds (comma-separated)", value="42,43,44,45,46", key="batch_seeds"
    )
    fast_mode = st.checkbox(
        "Fast Mode (Optimize for Large Scale, Minimal UI)", value=True, key="fast_mode"
    )
    run_batch = st.button("Run Batch Experiments")
    batch_results = st.session_state.get("batch_results", None)
    if run_batch:
        agent_counts_list = [
            int(x.strip()) for x in agent_counts.split(",") if x.strip().isdigit()
        ]
        seeds_list = [int(x.strip()) for x in seeds.split(",") if x.strip().isdigit()]
        st.session_state["batch_results"] = run_batch_experiments(
            n_experiments,
            agent_counts_list,
            seeds_list,
            st.session_state["n_steps"],
            fast_mode,
        )
        st.success("Batch experiments completed.")
        batch_results = st.session_state["batch_results"]
    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        st.subheader("Batch Results Table")
        st.dataframe(df_batch)
        st.download_button(
            label="Download Batch Results as CSV",
            data=df_batch.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Batch Results as JSON",
            data=json.dumps(batch_results, indent=2),
            file_name="batch_results.json",
            mime="application/json",
        )


def analytics():
    """
    Analytics tab.
    """
    st.header("Analytics")
    # Place analytics/visualization code here
    # Example: batch analytics, charts, render_group_modules, etc.
    # (Add your analytics panel code here)
    batch_results = st.session_state.get("batch_results", None)
    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        if "max_reward" in df_batch.columns and "experiment" in df_batch.columns:
            st.line_chart(df_batch.set_index("experiment")["max_reward"])
        if "diversity" in df_batch.columns and "experiment" in df_batch.columns:
            st.line_chart(df_batch.set_index("experiment")["diversity"])
        if "group_stability" in df_batch.columns and "experiment" in df_batch.columns:
            st.line_chart(df_batch.set_index("experiment")["group_stability"])
        # Advanced metrics
        st.markdown("**Advanced Metrics**")
        if "intervention_count" in df_batch.columns:
            st.bar_chart(df_batch.set_index("experiment")["intervention_count"])
        # Export advanced analytics
        st.download_button(
            label="Download Advanced Analytics (CSV)",
            data=df_batch.to_csv(index=False),
            file_name="batch_advanced_analytics.csv",
            mime="text/csv",
        )


def agent_chat():
    """
    Agent chat tab.
    """
    st.header("Agent Chat (Human-Agent Collaboration)")
    if "agent_chat_agent" not in st.session_state:
        st.session_state.agent_chat_agent = NeuroFuzzyFusionAgent(
            input_dims=[4, 3],
            hidden_dim=16,
            output_dim=5,
            fusion_type="concat",
            fusion_alpha=0.6,
        )
    chat_panel(st.session_state.agent_chat_agent)


def interventions():
    """
    Interventions tab.
    """
    st.header("Human Intervention & Override")
    user_name = st.text_input("Your Name or Email (for attribution)", key="user_name")
    agents = st.session_state.get("agents", [])
    mas = st.session_state.get("multiagent_system")
    if agents and mas:
        # Prepare agent info table
        agent_data = []
        for idx, agent in enumerate(agents):
            agent_data.append(
                {
                    "Agent": idx,
                    "Group": getattr(agent, "group", None),
                    "LastAction": getattr(agent, "last_action", None),
                }
            )
        df = pd.DataFrame(agent_data)
        st.subheader("Batch Edit Agent Groups")
        edited_df = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, key="edit_agent_table"
        )
        if st.button("Apply Group Edits"):
            for _, row in edited_df.iterrows():
                idx = int(row["Agent"])
                new_group = row["Group"]
                if new_group != getattr(agents[idx], "group", None):
                    old_group = getattr(agents[idx], "group", None)
                    mas.leave_group(idx)
                    mas.form_group(new_group, [idx])
                    # Log
                    log_entry = {
                        "time": str(pd.Timestamp.now()),
                        "agent": idx,
                        "move_group": {"from": old_group, "to": new_group},
                        "user": user_name,
                    }
                    if "intervention_log" not in st.session_state:
                        st.session_state["intervention_log"] = []
                    st.session_state["intervention_log"].append(log_entry)
            st.success("Batch group edits applied.")
        st.subheader("Override Agent Action or Group (Single)")
        col1, col2, col3 = st.columns(3)
        with col1:
            agent_idx = st.number_input(
                "Agent Index",
                min_value=0,
                max_value=len(agents) - 1,
                value=0,
                step=1,
                key="intervene_agent_idx",
            )
        with col2:
            new_action = st.text_input(
                "Override Action (optional)", key="intervene_action"
            )
        with col3:
            new_group = st.text_input("Move to Group (optional)", key="intervene_group")
        if st.button("Apply Intervention"):
            log_entry = {
                "time": str(pd.Timestamp.now()),
                "agent": agent_idx,
                "user": user_name,
            }
            if new_action:
                agents[agent_idx].last_action = new_action
                log_entry["override_action"] = new_action
            if new_group:
                old_group = getattr(agents[agent_idx], "group", None)
                mas.leave_group(agent_idx)
                mas.form_group(new_group, [agent_idx])
                log_entry["move_group"] = {"from": old_group, "to": new_group}
            # Log intervention
            if "intervention_log" not in st.session_state:
                st.session_state["intervention_log"] = []
            st.session_state["intervention_log"].append(log_entry)
            st.success(f"Intervention applied to Agent {agent_idx}.")
        # --- Group Module Editing ---
        st.subheader("Edit Group Modules (Rules/Parameters)")
        if mas.group_modules:
            for group_id, module in mas.group_modules.items():
                st.markdown(f"**Group {group_id} Module:**")
                module_json = st.text_area(
                    f"Edit Module for Group {group_id}",
                    json.dumps(module, indent=2),
                    key=f"edit_module_{group_id}",
                )
                if st.button(
                    f"Apply Module Edit to Group {group_id}",
                    key=f"apply_module_{group_id}",
                ):
                    try:
                        new_module = json.loads(module_json)
                        mas.group_modules[group_id] = new_module
                        log_entry = {
                            "time": str(pd.Timestamp.now()),
                            "group": group_id,
                            "edit_module": True,
                            "user": user_name,
                        }
                        if "intervention_log" not in st.session_state:
                            st.session_state["intervention_log"] = []
                        st.session_state["intervention_log"].append(log_entry)
                        st.success(f"Module for Group {group_id} updated.")
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
        # Display intervention log
        st.subheader("Intervention Log")
        if (
            "intervention_log" in st.session_state
            and st.session_state["intervention_log"]
        ):
            st.json(st.session_state["intervention_log"])


def collaboration():
    """
    Collaboration tab.
    """
    st.header("Google Sheets Collaboration")
    gsheet_key_collab = st.file_uploader(
        "Upload Google Service Account Key (JSON)",
        type=["json"],
        key="gsheet_key_collab",
    )
    spreadsheet_id_collab = st.text_input(
        "Google Spreadsheet ID", key="gsheet_spreadsheet_id_collab"
    )
    worksheet_name_collab = st.text_input(
        "Worksheet Name", value="Sheet1", key="gsheet_worksheet_name_collab"
    )
    auto_sync_dashboard_collab = st.checkbox(
        "Enable Auto-Sync with Google Sheets",
        value=False,
        key="gsheet_auto_sync_dashboard_collab",
    )
    sync_interval_dashboard_collab = st.number_input(
        "Auto-Sync Interval (seconds)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        key="gsheet_sync_interval_dashboard_collab",
    )
    col_sync1, col_sync2 = st.columns(2)
    if gsheet_key_collab is not None:
        # Save uploaded key to a temp file
        import tempfile

        key_bytes = gsheet_key_collab.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(key_bytes)
            json_keyfile_path = tmp.name
        from dashboard.google_sheets import (
            get_gsheet_client,
            read_sheet_to_df,
            write_df_to_sheet,
        )

        gc = get_gsheet_client(json_keyfile_path)

        def merge_logs(local_log, remote_log):
            # Merge by unique (time, agent, group, user) tuple
            seen = set()
            merged = []
            for entry in local_log + remote_log:
                key = (
                    entry.get("time"),
                    entry.get("agent"),
                    entry.get("group", None),
                    entry.get("user", None),
                )
                if key not in seen:
                    merged.append(entry)
                    seen.add(key)
            merged.sort(key=lambda x: x.get("time", ""))
            return merged

        with col_sync1:
            if st.button("Sync Log to Google Sheets"):
                if spreadsheet_id_collab and worksheet_name_collab:
                    df_log = pd.DataFrame(
                        st.session_state.get("intervention_log", [])
                    )
                    write_df_to_sheet(
                        gc, spreadsheet_id_collab, worksheet_name_collab, df_log
                    )
                    st.success("Log synced to Google Sheets!")
        with col_sync2:
            if st.button("Load Log from Google Sheets"):
                if spreadsheet_id_collab and worksheet_name_collab:
                    df_gsheet = read_sheet_to_df(
                        gc, spreadsheet_id_collab, worksheet_name_collab
                    )
                    # Merge logs
                    merged_log = merge_logs(
                        st.session_state.get("intervention_log", []),
                        df_gsheet.to_dict(orient="records"),
                    )
                    st.session_state["intervention_log"] = merged_log
                    st.success("Log loaded and merged from Google Sheets!")
        # --- Auto-sync logic ---
        import time

        if auto_sync_dashboard_collab and spreadsheet_id_collab and worksheet_name_collab:
            last_sync = st.session_state.get("gsheet_last_sync_dashboard_collab", 0)
            now = time.time()
            if now - last_sync > sync_interval_dashboard_collab:
                # Pull remote log and merge
                df_gsheet = read_sheet_to_df(
                    gc, spreadsheet_id_collab, worksheet_name_collab
                )
                merged_log = merge_logs(
                    st.session_state.get("intervention_log", []),
                    df_gsheet.to_dict(orient="records"),
                )
                st.session_state["intervention_log"] = merged_log
                # Push merged log
                df_log = pd.DataFrame(st.session_state["intervention_log"])
                write_df_to_sheet(
                    gc, spreadsheet_id_collab, worksheet_name_collab, df_log
                )
                st.session_state["gsheet_last_sync_dashboard_collab"] = now
                st.info("Auto-synced intervention log with Google Sheets.")


def settings():
    """
    Settings tab.
    """
    st.header("Settings")
    # Add settings panel code here


def main():
    """
    Main application entry point.
    """
    # st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
    if not st.session_state.get("authenticated", False):
        login_form()
    else:
        # --- Group Policy Controls ---
        st.sidebar.markdown("---")
        st.sidebar.header("Group Policy Controls")
        st.sidebar.checkbox(
            "Enable Group Knowledge Sharing",
            value=st.session_state.get("enable_group_knowledge", False),
            key="enable_group_knowledge",
        )
        st.sidebar.selectbox(
            "Collective Action Mode",
            ["None", "Voting", "Consensus", "Leader"],
            index=["None", "Voting", "Consensus", "Leader"].index(
                st.session_state.get("collective_mode", "None")
            ),
            key="collective_mode",
        )
        st.sidebar.button("Share Group Knowledge Now", key="share_now")
        st.sidebar.checkbox(
            "Distributed Agent Execution",
            value=st.session_state.get("distributed_execution", False),
            key="distributed_execution",
        )

        tab_names = [
            "Simulation",
            "Batch Experiments",
            "Analytics",
            "Agent Chat",
            "Interventions",
            "Collaboration",
            "Settings",
        ]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            simulation_controls()
        with tabs[1]:
            batch_experiments()
        with tabs[2]:
            analytics()
        with tabs[3]:
            agent_chat()
        with tabs[4]:
            interventions()
        with tabs[5]:
            collaboration()
        with tabs[6]:
            settings()

        # --- Google Sheets Integration Controls ---
        st.markdown("**Google Sheets Collaboration**")
        gsheet_key_dashboard = st.file_uploader(
            "Upload Google Service Account Key (JSON)",
            type=["json"],
            key="gsheet_key_dashboard",
        )
        spreadsheet_id_dashboard = st.text_input(
            "Google Spreadsheet ID", key="gsheet_spreadsheet_id_dashboard"
        )
        worksheet_name_dashboard = st.text_input(
            "Worksheet Name", value="Sheet1", key="gsheet_worksheet_name_dashboard"
        )
        auto_sync_dashboard_dashboard = st.checkbox(
            "Enable Auto-Sync with Google Sheets",
            value=False,
            key="gsheet_auto_sync_dashboard_dashboard",
        )
        sync_interval_dashboard_dashboard = st.number_input(
            "Auto-Sync Interval (seconds)",
            min_value=5,
            max_value=600,
            value=60,
            step=5,
            key="gsheet_sync_interval_dashboard_dashboard",
        )
        col_sync1, col_sync2 = st.columns(2)
        if gsheet_key_dashboard is not None:
            # Save uploaded key to a temp file
            import tempfile

            key_bytes = gsheet_key_dashboard.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(key_bytes)
                json_keyfile_path = tmp.name
            from dashboard.google_sheets import (
                get_gsheet_client,
                read_sheet_to_df,
                write_df_to_sheet,
            )

            gc = get_gsheet_client(json_keyfile_path)

            def merge_logs(local_log, remote_log):
                # Merge by unique (time, agent, group, user) tuple
                seen = set()
                merged = []
                for entry in local_log + remote_log:
                    key = (
                        entry.get("time"),
                        entry.get("agent"),
                        entry.get("group", None),
                        entry.get("user", None),
                    )
                    if key not in seen:
                        merged.append(entry)
                        seen.add(key)
                merged.sort(key=lambda x: x.get("time", ""))
                return merged

            with col_sync1:
                if st.button("Sync Log to Google Sheets"):
                    if spreadsheet_id_dashboard and worksheet_name_dashboard:
                        df_log = pd.DataFrame(
                            st.session_state.get("intervention_log", [])
                        )
                        write_df_to_sheet(gc, spreadsheet_id_dashboard, worksheet_name_dashboard, df_log)
                        st.success("Log synced to Google Sheets!")
            with col_sync2:
                if st.button("Load Log from Google Sheets"):
                    if spreadsheet_id_dashboard and worksheet_name_dashboard:
                        df_gsheet = read_sheet_to_df(gc, spreadsheet_id_dashboard, worksheet_name_dashboard)
                        # Merge logs
                        merged_log = merge_logs(
                            st.session_state.get("intervention_log", []),
                            df_gsheet.to_dict(orient="records"),
                        )
                        st.session_state["intervention_log"] = merged_log
                        st.success("Log loaded and merged from Google Sheets!")
            # --- Auto-sync logic ---
            import time

            if auto_sync_dashboard and spreadsheet_id_dashboard and worksheet_name_dashboard:
                last_sync = st.session_state.get("gsheet_last_sync_dashboard", 0)
                now = time.time()
                if now - last_sync > sync_interval_dashboard:
                    # Pull remote log and merge
                    df_gsheet = read_sheet_to_df(gc, spreadsheet_id_dashboard, worksheet_name_dashboard)
                    merged_log = merge_logs(
                        st.session_state.get("intervention_log", []),
                        df_gsheet.to_dict(orient="records"),
                    )
                    st.session_state["intervention_log"] = merged_log
                    # Push merged log
                    df_log = pd.DataFrame(st.session_state["intervention_log"])
                    write_df_to_sheet(gc, spreadsheet_id_dashboard, worksheet_name_dashboard, df_log)
                    st.session_state["gsheet_last_sync_dashboard"] = now
                    st.info("Auto-synced intervention log with Google Sheets.")
        # --- Timeline & Analytics ---
        if st.session_state.get("intervention_log", []):
            # Timeline table
            st.subheader("Intervention Timeline")
            df_log = pd.DataFrame(st.session_state["intervention_log"])
            st.dataframe(df_log)
            # Analytics
            st.subheader("Intervention Analytics")
            intervention_types = (
                df_log.apply(lambda row: list(row.keys())[2:], axis=1)
                .explode()
                .value_counts()
            )
            st.bar_chart(intervention_types)
            # Cumulative interventions
            df_log["timestamp"] = pd.to_datetime(df_log["time"])
            df_log = df_log.sort_values("timestamp")
            df_log["cumulative"] = range(1, len(df_log) + 1)
            st.line_chart(df_log.set_index("timestamp")["cumulative"])
            # Export options
            st.download_button(
                "Export Log as CSV",
                df_log.to_csv(index=False),
                "intervention_log.csv",
                "text/csv",
            )
            st.download_button(
                "Export Log as JSON",
                df_log.to_json(orient="records", indent=2),
                "intervention_log.json",
                "application/json",
            )

        # --- Explainability & Visualization ---
        st.markdown("---")
        st.header("Explainability & Visualization")
        mas = st.session_state.get("multiagent_system")
        if mas:
            st.subheader("Fuzzy Rules (per Group)")
            import json

            for group_id, module in mas.group_modules.items():
                st.markdown(f"**Group {group_id} Rules:**")
                st.json(module)
            st.subheader("Group Structure & Dynamics")
            try:
                import matplotlib.pyplot as plt
                import networkx as nx

                G = nx.Graph()
                for group_id, members in mas.groups.items():
                    for agent in members:
                        G.add_node(f"A{agent}", group=group_id)
                        G.add_edge(f"G{group_id}", f"A{agent}")
                    G.add_node(f"G{group_id}", group=group_id)
                fig, ax = plt.subplots()
                pos = nx.spring_layout(G)
                nx.draw(
                    G,
                    pos,
                    with_labels=True,
                    node_color=[G.nodes[n].get("group", 0) for n in G.nodes],
                    cmap=plt.cm.Set3,
                    ax=ax,
                )
                st.pyplot(fig)
            except Exception as e:
                st.info(f"(NetworkX/Matplotlib required for group visualization) {e}")
        else:
            st.info("No episode memory available yet.")
        st.subheader("Interactive Scenario Playback")
        episode_memory = st.session_state.get("episode_memory", [])
        if episode_memory:
            max_step = len(episode_memory) - 1
            # Playback controls
            playback_step = st.number_input(
                "Playback Step",
                min_value=0,
                max_value=max_step if max_step >= 0 else 0,
                value=0,
                step=1,
                key="playback_step",
            )
            col_play, col_pause, col_prev, col_next = st.columns(4)
            if "playback_running" not in st.session_state:
                st.session_state["playback_running"] = False
            if col_play.button("Play"):
                st.session_state["playback_running"] = True
            if col_pause.button("Pause"):
                st.session_state["playback_running"] = False
            if col_prev.button("Step Back") and playback_step > 0:
                st.session_state["playback_step"] = playback_step - 1
            if col_next.button("Step Forward") and playback_step < max_step:
                st.session_state["playback_step"] = playback_step + 1
            # Auto-playback logic
            import time

            if st.session_state.get("playback_running", False):
                if playback_step < max_step:
                    st.session_state["playback_step"] = playback_step + 1
                    time.sleep(0.4)
                    st.experimental_rerun()
                else:
                    st.session_state["playback_running"] = False
            # Show all agent data at current playback step
            st.write(f"Playback: Step {playback_step} / {max_step}")
            st.json(episode_memory[playback_step])
            st.download_button(
                label="Download Episode Memory (JSON)",
                data=json.dumps(episode_memory, indent=2),
                file_name="episode_memory.json",
                mime="application/json",
            )
        else:
            st.info("No episode memory available yet.")
        # --- SOM Grouping Section ---
        st.markdown("---")
        st.header("Dynamic Agent Grouping (SOM)")
        if st.button("Re-cluster Agents using SOM"):
            som_group_agents()
        agents = st.session_state.get("agents", [])
        som = (
            getattr(mas, "last_som", None) if mas and hasattr(mas, "last_som") else None
        )
        if agents and mas:
            positions = [getattr(agent, "position", None) for agent in agents]
            render_agent_positions(positions, agents)
            render_group_modules(mas.group_modules)
            # Show group-level knowledge if enabled
            if st.session_state.get(
                "enable_group_knowledge", False
            ) or st.session_state.get("share_now", False):
                render_group_knowledge(mas)
            # --- SOM Grid Visualization ---
            try:
                if som is not None:
                    render_som_grid_with_agents(mas, som)
            except Exception as e:
                st.info(f"SOM grid visualization unavailable: {e}")
            # --- Group Analytics ---
            try:
                reward_history = getattr(
                    st.session_state.get("simulation", None), "reward_history", None
                )
                render_group_analytics(mas, reward_history=reward_history)
            except Exception as e:
                st.info(f"Group analytics unavailable: {e}")
        if hasattr(mas, "last_som"):
            som = mas.last_som
        if agents and mas:
            positions = [getattr(agent, "position", None) for agent in agents]
            render_agent_positions(positions, agents)
            render_group_modules(mas.group_modules)
            # Show group-level knowledge if enabled
            if st.session_state.get(
                "enable_group_knowledge", False
            ) or st.session_state.get("share_now", False):
                render_group_knowledge(mas)
            # --- SOM Grid Visualization ---
            try:
                # Try to get the SOMClusterer used for last grouping
                som = getattr(mas, "last_som", None)
                if som is not None:
                    render_som_grid_with_agents(mas, som)
            except Exception as e:
                st.info(f"SOM grid visualization unavailable: {e}")
            # --- Group Analytics ---
            try:
                from dashboard import simulation

                reward_history = getattr(simulation, "reward_history", None)
                render_group_analytics(mas, reward_history=reward_history)
            except Exception as e:
                st.info(f"Group analytics unavailable: {e}")
        # (Other dashboard panels and logic will be called here)


def plugins_and_docs():
    import inspect
    from src.env.registry import get_registered_environments
    from src.core.agents.agent_registry import get_registered_agents
    from src.plugins.registry import get_registered_sensors, get_registered_actuators
    st.header("Plugin Registry & Developer Docs")
    st.subheader("Environments")
    envs = get_registered_environments()
    for name, cls in envs.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Docstring:** {cls.__doc__}")
            sig = inspect.signature(cls.__init__)
            st.code(str(sig), language="python")
    st.subheader("Agents")
    agents = get_registered_agents()
    for name, cls in agents.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Docstring:** {cls.__doc__}")
            sig = inspect.signature(cls.__init__)
            st.code(str(sig), language="python")
    st.subheader("Sensors")
    sensors = get_registered_sensors()
    for name, cls in sensors.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Docstring:** {cls.__doc__}")
            sig = inspect.signature(cls.__init__)
            st.code(str(sig), language="python")
    st.subheader("Actuators")
    actuators = get_registered_actuators()
    for name, cls in actuators.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Docstring:** {cls.__doc__}")
            sig = inspect.signature(cls.__init__)
            st.code(str(sig), language="python")
    st.subheader("Developer Documentation")
    st.markdown("""
    - **How to add new plugins:** See registry auto-discovery in `src/env/registry.py`, `src/core/agents/agent_registry.py`, `src/plugins/registry.py`.
    - **Best practices:** Use clear docstrings, type hints, and follow the base class interfaces.
    - **Advanced Controls:** Meta-learning, experiment management, explainability, and human-in-the-loop features are under development. See `roadmap2.md` for progress.
    - **Troubleshooting:** Use the plugin hot-reload button in the sidebar if new plugins do not appear.
    """)

if __name__ == "__main__":
    main()
