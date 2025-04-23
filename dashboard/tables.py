import streamlit as st
from typing import List, Any, Optional, Tuple

def render_knowledge_table(agents: List[Any]) -> None:
    """
    Display a table showing each agent's knowledge state and law violations.
    Args:
        agents (List[Any]): List of agent objects, each with 'group', 'online_knowledge', and 'law_violations' attributes.
    """
    """
    Display a table showing each agent's knowledge state and law violations.
    """
    st.header("Agent Knowledge State")
    data = []
    for i, agent in enumerate(agents):
        try:
            group = getattr(agent, "group", "")
            knowledge = str(getattr(agent, "online_knowledge", {}))
            law_violations = getattr(agent, "law_violations", 0)
            data.append({
                "Agent": f"{group}:{i}",
                "Knowledge": knowledge,
                "Law Violations": law_violations,
            })
        except Exception as e:
            st.warning(f"Agent {i} missing expected attributes: {e}")
    if data:
        st.table(data)
    else:
        st.info("No agent knowledge data to display.")

def render_group_decisions_log(group_decisions: List[Any]) -> None:
    """
    Display a log of recent group decisions, color-coded by legality.
    Args:
        group_decisions (List[Any]): List of (actions, result, legal) tuples or dicts.
    """
    st.header("Recent Group Decisions")
    for idx, item in enumerate(list(group_decisions)[-10:][::-1]):
        try:
            if isinstance(item, dict):
                actions = item.get("actions")
                result = item.get("result")
                legal = item.get("legal", True)
            else:
                actions, result, legal = item
            color = "green" if legal else "red"
            st.markdown(
                f"<span style='color:{color}'>Actions: {actions} | Result: {result} | {'Legal' if legal else 'Violated Law'}</span>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Malformed group decision entry at index {idx}: {e}")

def parse_group_decision(item: Any) -> Optional[Tuple[str, str, bool]]:
    """
    Parse a group decision item into a tuple of (actions, result, legal).
    
    Args:
        item (Any): A group decision item, either a tuple or a dictionary.
    
    Returns:
        Optional[Tuple[str, str, bool]]: The parsed group decision item, or None if parsing fails.
    """
    try:
        if isinstance(item, dict):
            actions = item.get("actions")
            result = item.get("result")
            legal = item.get("legal", True)
        else:
            actions, result, legal = item
        return actions, result, legal
    except Exception as e:
        st.warning(f"Malformed group decision entry: {e}")
        return None

def render_group_decisions_log(group_decisions: List[Any]) -> None:
    """
    Display a log of recent group decisions, color-coded by legality.
    
    Args:
        group_decisions (List[Any]): List of (actions, result, legal) tuples or dicts.
    """
    st.header("Recent Group Decisions")
    for idx, item in enumerate(list(group_decisions)[-10:][::-1]):
        decision = parse_group_decision(item)
        if decision:
            actions, result, legal = decision
            color = "green" if legal else "red"
            st.markdown(
                f"<span style='color:{color}'>Actions: {actions} | Result: {result} | {'Legal' if legal else 'Violated Law'}</span>",
                unsafe_allow_html=True,
            )
