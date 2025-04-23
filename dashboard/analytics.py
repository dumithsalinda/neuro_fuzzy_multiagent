import streamlit as st
import pandas as pd

def render_batch_analytics(df: pd.DataFrame) -> None:
    """
    Render analytics for batch experiments, including summary statistics and visualizations.
    Args:
        df (pd.DataFrame): DataFrame containing batch experiment results.
    """
    if df is None or df.empty:
        st.warning("No batch experiment data available.")
        return
    st.subheader("Batch Experiment Analytics")
    st.dataframe(df)
    # Show summary statistics
    st.markdown("### Summary Statistics")
    st.write(df.describe(include='all'))
    # Visualize key metrics if present
    for metric in ["AvgReward", "Diversity", "Cohesion", "GroupStability", "InterventionCount"]:
        if metric in df.columns:
            st.line_chart(df[metric])
    # Show distribution plots for numeric columns
    import matplotlib.pyplot as plt
    import seaborn as sns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)


def render_advanced_metrics(df: pd.DataFrame) -> None:
    """
    Render advanced metrics for agent/group performance, such as diversity, stability, intervention count, etc.
    Args:
        df (pd.DataFrame): DataFrame containing experiment or simulation results.
    """
    if df is None or df.empty:
        st.warning("No data for advanced metrics.")
        return
    st.subheader("Advanced Metrics")
    metrics = {}
    # Example: Compute overall diversity, stability, intervention count
    if "Diversity" in df.columns:
        metrics["Mean Diversity"] = df["Diversity"].mean()
    if "GroupStability" in df.columns:
        metrics["Mean Group Stability"] = df["GroupStability"].mean()
    if "InterventionCount" in df.columns:
        metrics["Total Interventions"] = df["InterventionCount"].sum()
    if "AvgReward" in df.columns:
        metrics["Mean AvgReward"] = df["AvgReward"].mean()
    if metrics:
        st.write(metrics)
    else:
        st.info("No advanced metrics found in data.")
    # Optionally, visualize correlations
    import matplotlib.pyplot as plt
    import seaborn as sns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix of Metrics")
        st.pyplot(fig)

