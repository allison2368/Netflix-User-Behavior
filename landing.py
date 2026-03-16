import streamlit as st
import styles

def show_landing_page():
    # 원본 스타일 적용
    styles.apply_landing_styles()

    # Large Netflix logo
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 4rem; font-weight: 900; color: #E50914; letter-spacing: -0.1rem;
                       font-family: 'Bebas Neue', 'Impact', sans-serif; margin: 0;
                       text-shadow: 0 0 20px rgba(229, 9, 20, 0.6), 0 0 40px rgba(229, 9, 20, 0.4), 2px 2px 8px rgba(0, 0, 0, 0.9);">
                NETFLIX
            </h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Title
    st.markdown("""
    <h1 style="text-align: center; color: white; font-size: 3.5rem; font-weight: 400; letter-spacing: 2px; margin-bottom: 3rem;
               text-shadow: 0 0 20px rgba(229, 9, 20, 0.5), 0 0 40px rgba(229, 9, 20, 0.3), 2px 2px 8px rgba(0, 0, 0, 0.9);">
        Who's Watching?
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    profiles = [
        {"key": "marcus", "emoji": "🎯", "name": "Marcus", "role": "Marketing Manager", "col": col1, "color": "linear-gradient(135deg, #E50914 0%, #B20710 100%)"},
        {"key": "sarah", "emoji": "📺", "name": "Sarah", "role": "Content Executive", "col": col2, "color": "linear-gradient(135deg, #564d4d 0%, #221f1f 100%)"},
        {"key": "puja", "emoji": "💡", "name": "Puja", "role": "Product Manager", "col": col3, "color": "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)"},
        {"key": "admin", "emoji": "📊", "name": "Admin", "role": "Model Performance", "col": col4, "color": "linear-gradient(135deg, #1E90FF 0%, #0066CC 100%)"}
    ]

    for profile in profiles:
        with profile["col"]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 140px; height: 140px; margin: 0 auto 1rem; border-radius: 8px;
                            background: {profile['color']};
                            display: flex; align-items: center; justify-content: center;
                            font-size: 5rem; border: 4px solid rgba(255,215,0,0.3);
                            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                            transition: all 0.3s ease;">
                    {profile['emoji']}
                </div>
                <div style="color: #808080; font-size: 1.5rem; text-shadow: 1px 1px 4px black; margin-top: 0.5rem;">{profile['name']}</div>
                <div style="color: #565656; font-size: 0.9rem; margin-bottom: 1rem;">{profile['role']}</div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns([1.5, 1, 1.5])
            with col_b:
                if st.button("▶️", key=profile['key']):
                    st.session_state.selected_user = profile['key']
                    st.rerun()

    # 재생 버튼 스타일은 마지막에 덮어씌워야 적용됨
    styles.apply_play_button_styles()
    st.markdown("<br><br>", unsafe_allow_html=True)