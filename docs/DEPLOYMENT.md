# Streamlit Cloud Deployment

To deploy the dashboard on Streamlit Cloud:

1. Push your code to a public GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click 'New app', select your repo and branch, and set the main file as `neuro_fuzzy_multiagent/dashboard.py`.
4. Ensure `requirements.txt` is present in the repo root.
5. (Optional) Add a `.streamlit/config.toml` for custom settings.

Example `.streamlit/config.toml`:
```toml
[server]
headless = true
port = $PORT
enableCORS = false
```

**Troubleshooting:**
- If you have modules in subfolders, use `neuro_fuzzy_multiagent/` as a Python module root (add `__init__.py` files if needed).
- If you use secrets, set them in the Streamlit Cloud dashboard.
- For advanced deployment or troubleshooting, see the [Developer Guide](DEVELOPER.md).

**To run locally:**
```sh
streamlit run neuro_fuzzy_multiagent/dashboard.py
```
