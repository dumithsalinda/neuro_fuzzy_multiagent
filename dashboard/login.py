import streamlit as st

def login_form():
    st.title("Login to Multi-Agent System Dashboard")
    with st.form("login_form"):
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
