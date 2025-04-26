import streamlit as st

import uuid


def login_form() -> None:
    """
    Render a login form for the Multi-Agent System Dashboard.
    Updates session state with authentication status.
    """

    st.title("Login to Multi-Agent System Dashboard")

    unique_form_key = "fixed_login_form"
    with st.form(unique_form_key):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")
        if login_btn:

            # For demo purposes: hardcoded credentials
            if username == "admin" and password == "password":
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    if st.session_state.get("authenticated"):
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()
