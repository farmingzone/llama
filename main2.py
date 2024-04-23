import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

def main():
    st.title("LLM Chat Demo")

    template = "Question: {question}\n\nAnswer: {{answer}}"
    prompt = PromptTemplate(template=template, input_variables=["question"])

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path="./Meta-Llama-3-8B-Instruct-Q3_K_M.gguf",
        input={"temperature": 0.5,
               "max_length": 500,
               "top_p": 1},
        callback_manager=callback_manager,
        verbose=True,
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = st.text_area("Ask a question", "")

    if st.button("Get Answer"):
        response = llm_chain.invoke({"question": question})  # prompt 인수를 제거
        
        # 딕셔너리에서 답변 텍스트 추출
        response_text = response.get('text', "").strip()  # 'text' 키가 없을 경우 빈 문자열 반환

        # 'Explanation' 텍스트 이후의 내용만 추출
        explanation_index = response_text.find("Explanation")
        if explanation_index != -1:
            clean_response = response_text[explanation_index:]  # 'Explanation'부터 시작하여 나머지 모든 텍스트 추출
        else:
            clean_response = response_text  # 'Explanation'이 없는 경우, 원본 텍스트 사용
        
        clean_response = clean_response.strip()
        clean_response = " ".join(clean_response.splitlines())

        st.text_area("Answer", clean_response)

if __name__ == "__main__":
    main()
