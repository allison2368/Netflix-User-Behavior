import os
import streamlit as st
import styles
import landing
from churn_model import churn_model_updated as cm
from usecase1.usecase1_app import run_usecase_1
from usecase2.usecase2_app import run_usecase_2
from usecase3.usecase3_app import run_usecase_3


# Initial Configuration
st.set_page_config(
    page_title="Netflix Churn Prediction Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Global Styles (Called from styles.py)
styles.apply_global_styles()

# Load Model Artifacts (Caching for performance)
@st.cache_resource
def load_model_artifacts():
    try:
        # Configuration for USE_GCS can be maintained at the top of the source file
        return cm.load_artifacts('./model_outputs')
    except Exception:
        st.error("Model artifacts not found.")
        st.stop()

# Placeholder Page Definition
def show_placeholder(title):
    st.header(f"🚧 {title}")
    st.info("This section is currently under development. Coming soon!")

# Main Execution Block
def main():
    # Initialize session state for user selection
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None

    # Routing between Landing Page and Dashboard
    if st.session_state.selected_user is None:
        landing.show_landing_page()
    else:
        artifacts = load_model_artifacts()

        # Top Header Section
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅ Switch User"):
                st.session_state.selected_user = None
                st.rerun()
        with col2:
            st.title("🎬 NETFLIX CHURN PREDICTION DASHBOARD")
            st.markdown("*Powered by Random Forest Machine Learning*")
        with col3:
            st.markdown("<div style='text-align: right;'><h2 style='color:#E50914; font-family:Bebas Neue;'>NETFLIX</h2></div>", unsafe_allow_html=True)
        st.markdown("---")

        # Sidebar Configuration
        user_info = {
            "marcus": {"emoji": "🎯", "name": "Marcus", "role": "Marketing Manager"},
            "sarah": {"emoji": "📺", "name": "Sarah", "role": "Content Executive"},
            "puja": {"emoji": "💡", "name": "Puja", "role": "Product Manager"},
            "admin": {"emoji": "📊", "name": "Admin", "role": "ML Engineer"}
        }
        current_user = user_info.get(st.session_state.selected_user, {})
        
        st.sidebar.markdown(f"<div style='text-align: center;'><h2 style='color:#E50914;'>NETFLIX</h2></div>", unsafe_allow_html=True)
        st.sidebar.success(f"{current_user['emoji']} **{current_user['name']}**\n\n*{current_user['role']}*")
        st.sidebar.markdown("---")

        # Define Page Navigation Options
        pages = [
            "🎯 Marketing Campaign (Marcus)", 
            "📺 Content Investment (Sarah)", 
            "💡 Feature Engagement (Puja)", 
            "📊 Model Performance"
        ]

        # Set default index based on the selected user from landing page
        default_index = 0
        if st.session_state.selected_user == "marcus":
            default_index = 0
        elif st.session_state.selected_user == "sarah":
            default_index = 1
        elif st.session_state.selected_user == "puja":
            default_index = 2
        elif st.session_state.selected_user == "admin":
            default_index = 3

        # Render Sidebar Radio Selection with auto-indexing
        page = st.sidebar.radio(
            "Select Dashboard View",
            pages,
            index=default_index  
        )

        # Main Content Routing
        if page == "🎯 Marketing Campaign (Marcus)":
            run_usecase_1()
        elif page == "📺 Content Investment (Sarah)":
            # show_placeholder("Content Investment Strategy")
            run_usecase_2()
        elif page == "💡 Feature Engagement (Puja)":
            run_usecase_3()
        elif page == "📊 Model Performance":
            # Display model metrics logic
            st.header("📊 Model Performance Metrics")
            st.write(artifacts.get('metrics', 'No metrics available'))

if __name__ == "__main__":
    main()