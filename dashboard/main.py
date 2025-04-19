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
        from dashboard.simulation import simulate_step, som_group_agents, run_batch_experiments
        from dashboard.visualization import render_agent_positions, render_group_modules, render_group_knowledge
        # Batch controls
        n_steps = st.number_input("Steps to Run", min_value=1, max_value=1000, value=10, step=1, key="n_steps")
        run_n_steps = st.button("Run N Steps")
        auto_run = st.checkbox("Auto-Run Simulation", value=False, key="auto_run")
        pause_run = st.button("Pause Auto-Run")
        # --- Batch/Parallel Experiment Controls ---
        st.markdown("---")
        st.header("Batch/Parallel Experiments")
        n_experiments = st.number_input("Number of Experiments", min_value=1, max_value=100, value=5, step=1, key="n_experiments")
        agent_counts = st.text_input("Agent Counts (comma-separated)", value="10,50,100", key="batch_agent_counts")
        seeds = st.text_input("Seeds (comma-separated)", value="42,43,44,45,46", key="batch_seeds")
        fast_mode = st.checkbox("Fast Mode (Optimize for Large Scale, Minimal UI)", value=True, key="fast_mode")
        run_batch = st.button("Run Batch Experiments")
        batch_results = st.session_state.get("batch_results", None)
        if run_batch:
            agent_counts_list = [int(x.strip()) for x in agent_counts.split(",") if x.strip().isdigit()]
            seeds_list = [int(x.strip()) for x in seeds.split(",") if x.strip().isdigit()]
            st.session_state["batch_results"] = run_batch_experiments(n_experiments, agent_counts_list, seeds_list, n_steps, fast_mode)
            st.success("Batch experiments completed.")
            batch_results = st.session_state["batch_results"]
        if batch_results:
            import pandas as pd
            import json
            st.subheader("Batch Results Table")
            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch)
            st.download_button(
                label="Download Batch Results as CSV",
                data=df_batch.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download Batch Results as JSON",
                data=json.dumps(batch_results, indent=2),
                file_name="batch_results.json",
                mime="application/json"
            )
            # Advanced analytics/visualization
            st.subheader("Batch Analytics & Visualization")
            if "mean_reward" in df_batch.columns:
                st.line_chart(df_batch.set_index("experiment")["mean_reward"])
            if "max_reward" in df_batch.columns:
                st.line_chart(df_batch.set_index("experiment")["max_reward"])
            # Advanced metrics
            st.markdown("**Advanced Metrics**")
            if "diversity" in df_batch.columns:
                st.line_chart(df_batch.set_index("experiment")["diversity"])
            if "group_stability" in df_batch.columns:
                st.line_chart(df_batch.set_index("experiment")["group_stability"])
            if "intervention_count" in df_batch.columns:
                st.bar_chart(df_batch.set_index("experiment")["intervention_count"])
            # Export advanced analytics
            st.download_button(
                label="Download Advanced Analytics (CSV)",
                data=df_batch.to_csv(index=False),
                file_name="batch_advanced_analytics.csv",
                mime="text/csv"
            )
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
        # --- Human Intervention Panel ---
        st.markdown("---")
        st.header("Human Intervention & Override")
        user_name = st.text_input("Your Name or Email (for attribution)", key="user_name")
        agents = st.session_state.get("agents", [])
        mas = st.session_state.get("multiagent_system")
        if agents and mas:
            import pandas as pd
            import json
            # Prepare agent info table
            agent_data = []
            for idx, agent in enumerate(agents):
                agent_data.append({
                    'Agent': idx,
                    'Group': getattr(agent, 'group', None),
                    'LastAction': getattr(agent, 'last_action', None),
                })
            df = pd.DataFrame(agent_data)
            st.subheader("Batch Edit Agent Groups")
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="edit_agent_table")
            if st.button("Apply Group Edits"):
                for _, row in edited_df.iterrows():
                    idx = int(row['Agent'])
                    new_group = row['Group']
                    if new_group != getattr(agents[idx], 'group', None):
                        old_group = getattr(agents[idx], 'group', None)
                        mas.leave_group(idx)
                        mas.form_group(new_group, [idx])
                        # Log
                        log_entry = {"time": str(pd.Timestamp.now()), "agent": idx, "move_group": {"from": old_group, "to": new_group}, "user": user_name}
                        if "intervention_log" not in st.session_state:
                            st.session_state["intervention_log"] = []
                        st.session_state["intervention_log"].append(log_entry)
                st.success("Batch group edits applied.")
            st.subheader("Override Agent Action or Group (Single)")
            col1, col2, col3 = st.columns(3)
            with col1:
                agent_idx = st.number_input("Agent Index", min_value=0, max_value=len(agents)-1, value=0, step=1, key="intervene_agent_idx")
            with col2:
                new_action = st.text_input("Override Action (optional)", key="intervene_action")
            with col3:
                new_group = st.text_input("Move to Group (optional)", key="intervene_group")
            if st.button("Apply Intervention"):
                log_entry = {"time": str(pd.Timestamp.now()), "agent": agent_idx, "user": user_name}
                if new_action:
                    agents[agent_idx].last_action = new_action
                    log_entry["override_action"] = new_action
                if new_group:
                    old_group = getattr(agents[agent_idx], 'group', None)
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
                    module_json = st.text_area(f"Edit Module for Group {group_id}", json.dumps(module, indent=2), key=f"edit_module_{group_id}")
                    if st.button(f"Apply Module Edit to Group {group_id}", key=f"apply_module_{group_id}"):
                        try:
                            new_module = json.loads(module_json)
                            mas.group_modules[group_id] = new_module
                            log_entry = {"time": str(pd.Timestamp.now()), "group": group_id, "edit_module": True, "user": user_name}
                            if "intervention_log" not in st.session_state:
                                st.session_state["intervention_log"] = []
                            st.session_state["intervention_log"].append(log_entry)
                            st.success(f"Module for Group {group_id} updated.")
                        except Exception as e:
                            st.error(f"Invalid JSON: {e}")
            # Display intervention log
            st.subheader("Intervention Log")
            if "intervention_log" in st.session_state and st.session_state["intervention_log"]:
                st.json(st.session_state["intervention_log"])
        # --- Intervention Timeline & Analytics ---
        st.markdown("---")
        st.header("Intervention Timeline & Analytics")
        intervention_log = st.session_state.get("intervention_log", [])
        import pandas as pd
        import json
        import matplotlib.pyplot as plt
        # --- Google Sheets Integration Controls ---
        st.markdown("**Google Sheets Collaboration**")
        gsheet_key = st.file_uploader("Upload Google Service Account Key (JSON)", type=["json"], key="gsheet_key")
        spreadsheet_id = st.text_input("Google Spreadsheet ID", key="gsheet_spreadsheet_id")
        worksheet_name = st.text_input("Worksheet Name", value="Sheet1", key="gsheet_worksheet_name")
        auto_sync = st.checkbox("Enable Auto-Sync with Google Sheets", value=False, key="gsheet_auto_sync")
        sync_interval = st.number_input("Auto-Sync Interval (seconds)", min_value=5, max_value=600, value=60, step=5, key="gsheet_sync_interval")
        col_sync1, col_sync2 = st.columns(2)
        if gsheet_key is not None:
            # Save uploaded key to a temp file
            import tempfile
            key_bytes = gsheet_key.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(key_bytes)
                json_keyfile_path = tmp.name
            from dashboard.google_sheets import get_gsheet_client, read_sheet_to_df, write_df_to_sheet
            gc = get_gsheet_client(json_keyfile_path)
            def merge_logs(local_log, remote_log):
                # Merge by unique (time, agent, group, user) tuple
                seen = set()
                merged = []
                for entry in local_log + remote_log:
                    key = (entry.get('time'), entry.get('agent'), entry.get('group', None), entry.get('user', None))
                    if key not in seen:
                        merged.append(entry)
                        seen.add(key)
                merged.sort(key=lambda x: x.get('time', ''))
                return merged
            with col_sync1:
                if st.button("Sync Log to Google Sheets"):
                    if spreadsheet_id and worksheet_name:
                        df_log = pd.DataFrame(intervention_log)
                        write_df_to_sheet(gc, spreadsheet_id, worksheet_name, df_log)
                        st.success("Log synced to Google Sheets!")
            with col_sync2:
                if st.button("Load Log from Google Sheets"):
                    if spreadsheet_id and worksheet_name:
                        df_gsheet = read_sheet_to_df(gc, spreadsheet_id, worksheet_name)
                        # Merge logs
                        merged_log = merge_logs(st.session_state.get("intervention_log", []), df_gsheet.to_dict(orient="records"))
                        st.session_state["intervention_log"] = merged_log
                        st.success("Log loaded and merged from Google Sheets!")
            # --- Auto-sync logic ---
            import time
            if auto_sync and spreadsheet_id and worksheet_name:
                last_sync = st.session_state.get("gsheet_last_sync", 0)
                now = time.time()
                if now - last_sync > sync_interval:
                    # Pull remote log and merge
                    df_gsheet = read_sheet_to_df(gc, spreadsheet_id, worksheet_name)
                    merged_log = merge_logs(st.session_state.get("intervention_log", []), df_gsheet.to_dict(orient="records"))
                    st.session_state["intervention_log"] = merged_log
                    # Push merged log
                    df_log = pd.DataFrame(st.session_state["intervention_log"])
                    write_df_to_sheet(gc, spreadsheet_id, worksheet_name, df_log)
                    st.session_state["gsheet_last_sync"] = now
                    st.info("Auto-synced intervention log with Google Sheets.")
        # --- Timeline & Analytics ---
        if intervention_log:
            # Timeline table
            df_log = pd.DataFrame(intervention_log)
            st.dataframe(df_log)
            # --- Export buttons ---
            st.markdown("**Export Intervention Log & Analytics**")
            col_csv, col_json = st.columns(2)
            with col_csv:
                st.download_button(
                    label="Download Log as CSV",
                    data=df_log.to_csv(index=False),
                    file_name="intervention_log.csv",
                    mime="text/csv"
                )
            with col_json:
                st.download_button(
                    label="Download Log as JSON",
                    data=json.dumps(intervention_log, indent=2),
                    file_name="intervention_log.json",
                    mime="application/json"
                )
            # Analytics: count by type
            type_counts = {"override_action": 0, "move_group": 0, "edit_module": 0}
            for entry in intervention_log:
                for t in type_counts:
                    if t in entry:
                        type_counts[t] += 1
            st.write("**Intervention Counts by Type:**", type_counts)
            # Analytics: interventions over time
            if 'time' in df_log.columns:
                df_log['time'] = pd.to_datetime(df_log['time'])
                df_log = df_log.sort_values('time')
                df_log['count'] = 1
                df_log['cumulative'] = df_log['count'].cumsum()
                plt.figure(figsize=(6,3))
                plt.plot(df_log['time'], df_log['cumulative'], marker='o')
                plt.xlabel('Time')
                plt.ylabel('Cumulative Interventions')
                plt.title('Interventions Over Time')
                st.pyplot(plt)
        else:
            st.info("No interventions have been logged yet.")
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
