"""
Main Dashboard Entry Point for the Netflix Churn Prediction Project.
Handles user routing, sidebar navigation, and global layout.
"""

import pandas as pd
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Global Styles (Called from styles.py)
styles.apply_global_styles()


@st.cache_resource
def load_model_artifacts():
    """
    Load model artifacts and configurations from the model_outputs directory.
    Returns the loaded dictionary or stops execution on failure.
    """
    try:
        # Configuration for USE_GCS can be maintained at the top of the source file
        return cm.load_artifacts("./model_outputs")
    except Exception:
        st.error("Model artifacts not found.")
        st.stop()

def main():
    """Main execution function to handle user session and dashboard routing."""
    # Initialize session state for user selection
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = None

    # Routing between Landing Page and Dashboard
    if st.session_state.selected_user is None:
        landing.show_landing_page()
    else:
        artifacts = load_model_artifacts()

        # Sidebar
        user_info = {
            "marcus": {"name": "Marcus", "role": "Marketing Manager"},
            "sarah": {"name": "Sarah", "role": "Content Executive"},
            "puja": {"name": "Puja", "role": "Product Manager"},
            "admin": {"name": "Admin", "role": "ML Engineer"},
        }
        current_user = user_info.get(st.session_state.selected_user, {})

        # Netflix logo higher up, Home button centered below it
        st.sidebar.markdown(
            "<div style='text-align: center; margin-top: 0; padding-top: 0.25rem;'>"
            "<h2 style='color:#E50914; margin-bottom: 0.5rem; font-size: 3.0rem;'>NETFLIX</h2></div>",
            unsafe_allow_html=True,
        )
        _c1, _c2, _c3 = st.sidebar.columns([1, 2, 1])
        if _c2.button("Home", key="back_to_landing"):
            st.session_state.selected_user = None
            st.rerun()
        st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
        st.sidebar.markdown("**Current user:**")
        st.sidebar.success(
            f"**{current_user['name']}**\n\n*{current_user['role']}*"
        )
        st.sidebar.markdown("---")

        pages = [
            "Marketing Campaign (Marcus)",
            "Content Investment (Sarah)",
            "Feature Engagement (Puja)",
            "Model Performance",
        ]
        default_index = 0
        if st.session_state.selected_user == "marcus":
            default_index = 0
        elif st.session_state.selected_user == "sarah":
            default_index = 1
        elif st.session_state.selected_user == "puja":
            default_index = 2
        elif st.session_state.selected_user == "admin":
            default_index = 3
        page = st.sidebar.radio("Select Dashboard View", pages, index=default_index)

        # Top Header (same for all use cases)
        st.markdown(
            "<h1 style='font-size: 4rem; color: #E50914; font-family: \"Bebas Neue\", sans-serif; "
            "letter-spacing: 2px; margin-bottom: 0;'>NETFLIX USER BEHAVIOR</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Main Content Routing
        if page == "Marketing Campaign (Marcus)":
            run_usecase_1()
        elif page == "Content Investment (Sarah)":
            run_usecase_2()
        elif page == "Feature Engagement (Puja)":
            run_usecase_3()
        elif page == "Model Performance":
            st.header("Model Performance")
            metrics = artifacts.get("metrics")
            if not metrics:
                st.info("No metrics available. Train the model to generate metrics.")
            else:
                # Summary table: key metrics (ML-style)
                st.subheader("Evaluation metrics")
                summary = pd.DataFrame([
                    {"Metric": "Accuracy", "Value": f"{metrics.get('accuracy', 0):.4f}"},
                    {"Metric": "Recall", "Value": f"{metrics.get('recall', 0):.4f}"},
                    {"Metric": "F1 Score", "Value": f"{metrics.get('f1_score', 0):.4f}"},
                    {"Metric": "Prediction threshold", "Value": f"{metrics.get('threshold', 0):.2f}"},
                ])
                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metric": st.column_config.TextColumn("Metric", width="medium"),
                        "Value": st.column_config.TextColumn("Value", width="small"),
                    },
                )
                # Feature importance table
                st.subheader("Feature importance")
                fi = metrics.get("feature_importance")
                if isinstance(fi, pd.DataFrame) and not fi.empty:
                    fi_df = fi.copy()
                    # Normalize column names (saved as 'feature' / 'importance' or similar)
                    rename_map = {}
                    for c in fi_df.columns:
                        if str(c).lower() == "feature":
                            rename_map[c] = "Feature"
                        elif str(c).lower() == "importance":
                            rename_map[c] = "Importance"
                    if rename_map:
                        fi_df = fi_df.rename(columns=rename_map)
                    if "Feature" not in fi_df.columns:
                        if len(fi_df.columns) >= 2:
                            fi_df = fi_df.rename(columns={fi_df.columns[0]: "Feature", fi_df.columns[1]: "Importance"})
                        else:
                            fi_df = fi_df.reset_index()
                            fi_df = fi_df.rename(columns={fi_df.columns[0]: "Feature", fi_df.columns[1]: "Importance"})
                    if "Importance" not in fi_df.columns and len(fi_df.columns) >= 2:
                        fi_df = fi_df.rename(columns={fi_df.columns[1]: "Importance"})
                    fi_df["Importance"] = pd.to_numeric(fi_df["Importance"], errors="coerce").fillna(0)
                    fi_df.insert(0, "Rank", range(1, len(fi_df) + 1))
                    fi_df = fi_df[["Rank", "Feature", "Importance"]]
                    fi_df["Importance"] = fi_df["Importance"].map(lambda x: f"{x:.4f}")
                    st.dataframe(
                        fi_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Feature": st.column_config.TextColumn("Feature", width="large"),
                            "Importance": st.column_config.TextColumn("Importance", width="small"),
                        },
                    )
                else:
                    st.caption("Feature importance not available.")


if __name__ == "__main__":
    main()
