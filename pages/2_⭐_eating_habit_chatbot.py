import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from loguru import logger
from config.settings import settings
from langchain.chains import LLMChain

st.set_page_config(page_title="컨텍스트 인식 챗봇", page_icon="⭐")
st.header('식습관 관리 챗봇')
st.write('건강한 식습관 형성을 위한 영양사 상담')

class ContextChatbot:
    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()

    @staticmethod
    @st.cache_resource
    def setup_chain(_llm, background_knowledge, max_tokens=1000):
        prompt_template = """
        당신은 배경지식을 활용하여 사용자의 식습관을 이해하고, 이 사용자의 식습관에 대해 조언을 해주는 전문 영양사입니다.

        배경 지식:
        {background_knowledge}

        대화 기록:
        {history}

        사용자: {input}

        """
        prompt = PromptTemplate(
            input_variables=["history", "input", "background_knowledge"],
            template=prompt_template
        )
        memory = ConversationBufferMemory(
            memory_key="history", 
            input_key="input", 
            return_messages=True, 
            max_token_limit=max_tokens
        )
        chain = LLMChain(
            llm=_llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        return chain


    @utils.enable_chat_history
    def main(self):
        max_tokens = st.sidebar.slider("메모리 크기 (토큰)", 100, 2000, 1000)
        background_knowledge = st.sidebar.text_area("배경지식 및 지시사항 입력")
        chain = self.setup_chain(self.llm, background_knowledge, max_tokens)

        user_query = st.chat_input(placeholder="무엇이든 물어보세요!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                try:
                    result = chain({
                        "input": user_query,
                        "background_knowledge": background_knowledge
                    }, callbacks=[st_cb])
                    # Update this line to access the 'text' key
                    response = result["text"]
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logger.info(f"사용자 질문: {user_query}")
                    logger.info(f"챗봇 응답: {response}")
                except Exception as e:
                    error_msg = f"응답 생성 중 오류 발생: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)


if __name__ == "__main__":
    st.sidebar.title("설정")
    model = st.sidebar.selectbox("LLM 모델 선택", [settings.DEFAULT_MODEL, "gpt-4o", "o1-mini"])
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    
    obj = ContextChatbot()
    obj.main()