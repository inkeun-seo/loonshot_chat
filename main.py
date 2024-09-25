import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="Langchain 챗봇",
    page_icon='💬',
    layout='wide'
)

logger.info("메인 페이지 로드됨")

st.header("식습관 관리 챗봇")

st.write("""
사용자의 식습관 정보에 기반하여 영양사의 조언을 제공하는 챗봇입니다.

- **사용법**: 왼쪽 바의 chatbot을 선택 후 사용하세요.
- **배경지식 사용**: 배경지식을 왼쪽 바의 background knowledge에 입력하세요.

""")

# 버전 정보 표시
st.sidebar.text("버전: 1.0.0")

# 피드백 섹션
st.sidebar.text_input("피드백", placeholder="여기에 피드백을 입력하세요")
if st.sidebar.button("피드백 제출"):
    # 피드백 처리 로직을 여기에 추가할 수 있습니다.
    logger.info("사용자가 피드백을 제출함")
    st.sidebar.success("피드백을 주셔서 감사합니다!")

logger.info("메인 페이지 렌더링 완료")