"""
Streamlit sidebar UI and logic for the Plugin Marketplace (local + remote plugins, submission, review, developer guide).
"""

import streamlit as st
from src.core.plugins.registration_utils import get_registered_plugins
from src.core.plugins.marketplace import (
    fetch_remote_plugin_index,
    plugin_metadata_by_type,
    get_local_plugin_version,
)

PLUGIN_TYPES = ["environment", "agent", "sensor", "actuator"]


def render_plugin_marketplace_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛒 Plugin Marketplace")
    # Local plugins
    local_plugins = {
        ptype: set(get_registered_plugins(ptype).keys()) for ptype in PLUGIN_TYPES
    }
    # Fetch remote index
    with st.sidebar.spinner("Fetching remote plugin index..."):
        remote_plugins_raw = fetch_remote_plugin_index()
    remote_plugins = plugin_metadata_by_type(remote_plugins_raw)

    # --- Plugin Submission Section ---
    st.markdown("Fill out the form below to submit your plugin for review.")
    with st.form("plugin_submission_form", clear_on_submit=True):
        sub_type = st.selectbox("Plugin Type", PLUGIN_TYPES)
        sub_name = st.text_input("Plugin Name")
        sub_desc = st.text_area("Description")
        sub_version = st.text_input("Version (e.g., 1.0.0)")
        sub_author = st.text_input("Author")
        sub_url = st.text_input("Source URL (e.g., GitHub)")
        sub_readme = st.text_area("README/Usage (optional)")
        submit = st.form_submit_button("Submit Plugin")
        import re

        errors = []
        if submit:
            if not sub_name.strip():
                errors.append("Name is required.")
            if not sub_desc.strip():
                errors.append("Description is required.")
            if not sub_version.strip() or not re.match(
                r"^\\d+\\.\\d+\\.\\d+$", sub_version.strip()
            ):
                errors.append("Version must be in format X.Y.Z.")
            if not sub_author.strip():
                errors.append("Author is required.")
            if not sub_url.strip():
                errors.append("Source URL is required.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                from src.core.plugins.plugin_submission import add_submission

                add_submission(
                    {
                        "type": sub_type,
                        "name": sub_name.strip(),
                        "description": sub_desc.strip(),
                        "version": sub_version.strip(),
                        "author": sub_author.strip(),
                        "source_url": sub_url.strip(),
                        "readme": sub_readme.strip() if sub_readme else None,
                    }
                )
                st.success(
                    "Plugin submission saved! It will be reviewed for inclusion in the marketplace."
                )

    # --- Developer Guide ---
    with st.sidebar.expander("📚 Plugin Developer Guide"):
        import os

        devguide_path = os.path.join(os.path.dirname(__file__), "PLUGIN_DEV_GUIDE.md")
        if os.path.exists(devguide_path):
            with open(devguide_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Developer guide not found.")

    # Marketplace UI (sorted, badges)
    from src.core.plugins.plugin_reviews import get_average_rating

    def version_tuple(v):
        return tuple(int(x) for x in v.split(".") if x.isdigit()) if v else ()

    for ptype in PLUGIN_TYPES:
        st.sidebar.markdown(f"**{ptype.capitalize()} Plugins**")
        plugin_objs = []
        for name in set(local_plugins[ptype]) | {
            p["name"] for p in remote_plugins[ptype]
        }:
            local = name in local_plugins[ptype]
            remote = next((p for p in remote_plugins[ptype] if p["name"] == name), None)
            avg_rating = get_average_rating(ptype, name) or 0
            plugin_objs.append(
                {
                    "name": name,
                    "local": local,
                    "remote": remote,
                    "avg_rating": avg_rating,
                }
            )
        plugin_objs.sort(key=lambda x: (-x["avg_rating"], x["name"]))
        for pobj in plugin_objs:
            name = pobj["name"]
            local = pobj["local"]
            remote = pobj["remote"]
            badge = ""
            if remote and remote.get("official", False):
                badge = "🟢 Official"
            elif remote:
                badge = "⚠️ Unofficial"
            elif local:
                badge = "(local)"
            st.sidebar.markdown(f"{badge} {name}")
            local_ver = get_local_plugin_version(ptype, name)
            remote_plugin = next(
                (p for p in remote_plugins[ptype] if p["name"] == name), None
            )
            remote_ver = remote_plugin.get("version") if remote_plugin else None
            ver_str = f"v{local_ver}" if local_ver else ""
            update_btn = False
            if (
                remote_ver
                and local_ver
                and version_tuple(remote_ver) > version_tuple(local_ver)
            ):
                ver_str += f" → v{remote_ver}"
                update_btn = True
            st.sidebar.markdown(f"- 🟢 **{name}** (installed) {ver_str}")
            with st.sidebar.expander(f"Details & Manage: {name}"):
                st.markdown(f"**Type:** {ptype}")
                st.markdown(f"**Name:** {name}")
                st.markdown(
                    f"**Installed Version:** {local_ver if local_ver else 'Unknown'}"
                )
                if remote_plugin:
                    st.markdown(
                        f"**Remote Version:** {remote_ver if remote_ver else 'Unknown'}"
                    )
                    for field in ["author", "homepage", "repository"]:
                        if remote_plugin.get(field):
                            st.markdown(
                                f"**{field.capitalize()}:** {remote_plugin[field]}"
                            )
                    if not remote_plugin.get("official", False):
                        st.error(
                            "⚠️ This plugin is NOT from an official/trusted source. Install at your own risk."
                        )
                from src.core.plugins.registration_utils import get_registered_plugins

                plugin_cls = get_registered_plugins(ptype).get(name)
                doc = plugin_cls.__doc__ if plugin_cls and plugin_cls.__doc__ else None
                if doc:
                    st.markdown(f"**Docstring:**\n{doc}")
                if remote_plugin and remote_plugin.get("readme"):
                    st.markdown(f"**README:**\n{remote_plugin['readme']}")
                from src.core.plugins.plugin_reviews import (
                    get_average_rating,
                    get_reviews,
                    add_review,
                )

                avg_rating = get_average_rating(ptype, name)
                if avg_rating:
                    st.markdown(
                        f"**Average Rating:** {'⭐️'*int(round(avg_rating))} ({avg_rating:.2f}/5)"
                    )
                reviews = get_reviews(ptype, name)
                if reviews:
                    st.markdown("**Recent Reviews:**")
                    for r in reviews[-3:][::-1]:
                        st.markdown(
                            f"- {'⭐️'*r['rating']} by {r['user']} ({r['timestamp'][:10]}): {r['review']}"
                        )
                with st.form(f"review_form_{ptype}_{name}", clear_on_submit=True):
                    st.markdown("**Leave a Rating & Review:**")
                    rating = st.slider("Rating", 1, 5, 5)
                    review = st.text_input("Review")
                    submit = st.form_submit_button("Submit Review")
                    if submit and review.strip():
                        add_review(ptype, name, rating, review)
                        st.success("Thank you for your review!")
                if st.button(f"Uninstall {name}", key=f"uninstall_{ptype}_{name}"):
                    from src.core.plugins.marketplace import uninstall_plugin
                    from src.core.plugins.hot_reload import reload_all_plugins

                    success, msg, path = uninstall_plugin(ptype, name)
                    if success:
                        reload_all_plugins()
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.error(msg)
            if update_btn:
                with st.sidebar.expander(f"Update {name} to v{remote_ver}"):
                    desc = remote_plugin.get("description", "No description.")
                    st.markdown(desc)
                    if st.button(f"Update {name}", key=f"update_{ptype}_{name}"):
                        from src.core.plugins.marketplace import (
                            download_and_save_plugin,
                        )
                        from src.core.plugins.hot_reload import reload_all_plugins

                        success, msg, path = download_and_save_plugin(remote_plugin)
                        if success:
                            reload_all_plugins()
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
        remote_not_installed = [
            p for p in remote_plugins[ptype] if p["name"] not in local_plugins[ptype]
        ]
        for plugin in remote_not_installed:
            desc = plugin.get("description", "No description.")
            remote_ver = plugin.get("version")
            with st.sidebar.expander(
                f"⚪️ {plugin['name']} (available) v{remote_ver if remote_ver else ''}"
            ):
                st.markdown(desc)
                if not plugin.get("official", False):
                    st.error(
                        "⚠️ This plugin is NOT from an official/trusted source. Install at your own risk."
                    )
                if st.button(
                    f"Install {plugin['name']}", key=f"install_{ptype}_{plugin['name']}"
                ):
                    from src.core.plugins.marketplace import download_and_save_plugin
                    from src.core.plugins.hot_reload import reload_all_plugins

                    if not plugin.get("official", False):
                        st.warning(
                            "You are installing a plugin from an untrusted source. Proceed with caution."
                        )
                    success, msg, path = download_and_save_plugin(plugin)
                    if success:
                        reload_all_plugins()
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.error(msg)
