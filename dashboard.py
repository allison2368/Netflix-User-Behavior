import os
import streamlit as st
import styles
import landing
from churn_model import churn_model_updated as cm

try:
    from usecase3.usecase3_app import run_usecase_3
except ImportError:
    # 에러 방지용 임시 함수
    def run_usecase_3():
        st.error("Usecase3 모듈을 불러올 수 없습니다. 폴더 구조와 __init__.py를 확인하세요.")

# 1. 초기 설정
st.set_page_config(
    page_title="Netflix Churn Prediction Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 전역 스타일 적용 (styles.py에서 호출)
styles.apply_global_styles()

# 3. 모델 아티팩트 로딩 (기존 로직 그대로)
@st.cache_resource
def load_model_artifacts():
    try:
        # USE_GCS 설정은 원본 파일 상단에서 하던 방식 유지 가능
        return cm.load_artifacts('./model_outputs')
    except Exception:
        st.error("Model artifacts not found.")
        st.stop()

# 4. Placeholder 페이지 정의
def show_placeholder(title):
    st.header(f"🚧 {title}")
    st.info("이 섹션은 현재 개발 중입니다. 곧 공개될 예정입니다!")

# 5. 메인 실행부
def main():
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None

    if st.session_state.selected_user is None:
        landing.show_landing_page()
    else:
        artifacts = load_model_artifacts()

        # 상단 헤더 (원본 코드 그대로)
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

        # 사이드바 설정
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


        # 페이지 목록 정의
        pages = [
            "🎯 Marketing Campaign (Marcus)", 
            "📺 Content Investment (Sarah)", 
            "💡 Feature Engagement (Puja)", 
            "📊 Model Performance"
        ]

        # 선택된 유저에 따른 기본 인덱스 설정
        default_index = 0
        if st.session_state.selected_user == "marcus":
            default_index = 0
        elif st.session_state.selected_user == "sarah":
            default_index = 1
        elif st.session_state.selected_user == "puja":
            default_index = 2
        elif st.session_state.selected_user == "admin":
            default_index = 3

        # 라디오 버튼에 index 적용
        page = st.sidebar.radio(
            "Select Dashboard View",
            pages,
            index=default_index  # 이 부분이 핵심!  
        )

        # 라우팅
        if page == "🎯 Marketing Campaign (Marcus)":
            show_placeholder("Marketing Campaign Analysis")
        elif page == "📺 Content Investment (Sarah)":
            show_placeholder("Content Investment Strategy")
        elif page == "💡 Feature Engagement (Puja)":
            # ✅ 요구사항대로 usecase3.usecase3_app에서 run_usecase_3 호출
            from usecase3.usecase3_app import run_usecase_3
            run_usecase_3()
        elif page == "📊 Model Performance":
            # 원본 코드에 있던 show_model_performance 로직을 그대로 여기에 두거나 별도 함수로 호출
            st.header("📊 Model Performance Metrics")
            st.write(artifacts.get('metrics', 'No metrics available'))

if __name__ == "__main__":
    main()