import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import altair as alt
from datetime import datetime

st.set_page_config(page_title="FlickWise Insights", layout="wide")

# --- UI Header ---
st.title("🎬 FlickWise Recommender Dashboard")
st.markdown("Upload synthetic or real user engagement data to generate insights and test the trained model.")

# --- Upload CSV ---
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

progress_bar = st.sidebar.empty()
data_load_state = st.sidebar.empty()

if uploaded_file is not None:
    progress_bar.progress(10)
    try:
        df = pd.read_csv(uploaded_file)
        progress_bar.progress(100)
        data_load_state.success("✅ Data loaded successfully!")

        # --- KPIs ---
        st.subheader("🔢 Key Performance Indicators")

        total_watch_time = df['daily_watch_time_minutes'].sum()
        total_users = df['user_id'].nunique()
        churn_rate = df['churned'].mean() if 'churned' in df.columns else np.nan

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Watch Time (min)", f"{total_watch_time:,.0f}")
        col2.metric("Unique Users", f"{total_users}")
        col3.metric("Churn Rate", f"{churn_rate*100:.2f}%")

        # --- Charts ---
        st.subheader("🎭 Genre Distribution")
        if 'genre' in df.columns:
            genre_df = df['genre'].value_counts().reset_index()
            genre_df.columns = ['Genre', 'Count']

            genre_chart = alt.Chart(genre_df).mark_bar().encode(
                x=alt.X('Genre', sort='-y'),
                y='Count',
                color='Genre'
            ).properties(height=400)

            st.altair_chart(genre_chart, use_container_width=True)

        # --- Placeholder: Add model prediction chart later ---
        st.subheader("📊 Retention Cohort Analysis (Coming Soon)")
        st.info("This section will analyze user retention trends based on join date cohorts.")

        # --- Placeholder: Show recommendation preview ---
        st.subheader("🎯 Recommendations Preview (Coming Soon)")
        st.warning("Here we will show top-N personalized recommendations after running inference.")

    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {e}")
else:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()
