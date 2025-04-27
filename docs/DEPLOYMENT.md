# Streamlit Cloud deployment instructions

1. Push your code to a public GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click 'New app', select your repo and branch, and set the main file as `neuro_fuzzy_multiagent/dashboard.py`.
4. Make sure `requirements.txt` is present in the repo root.
5. (Optional) Add a `.streamlit/config.toml` for custom settings.

# Example .streamlit/config.toml
# [server]
# headless = true
# port = $PORT
# enableCORS = false

# Troubleshooting
- If you have modules in subfolders, use `src` as a Python module root (add `__init__.py` files if needed).
- If you use secrets, set them in the Streamlit Cloud dashboard.

# To run locally:
# streamlit run neuro_fuzzy_multiagent/dashboard.py
