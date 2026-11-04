"""④ 답변 생성: 검색된 '원본' 텍스트·표·이미지를 멀티모달 LLM에 넣어 답변합니다."""
import re

import requests
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import ANSWER_MODEL, LLM_MODEL

_B64_RE = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")

INSTRUCTION = {
    "ko": "아래 자료(이미지·표·텍스트)를 근거로 질문에 한국어로 답하세요.\n"
          "그래프나 표에서 읽은 수치는 구체적으로 인용하세요.\n"
          "자료에 명시되지 않은 수식어(예: '전망치', '예상치')나 해석은 덧붙이지 마세요.\n"
          "자료에 없는 내용은 모른다고 답하세요.",
    "en": "Answer the question in English based only on the materials below (images, tables, text).\n"
          "Quote specific numbers read from charts or tables.\n"
          "Do not add qualifiers (e.g. 'forecast', 'estimate') that the materials do not state.\n"
          "If the answer is not in the materials, say you don't know.",
}


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트(표 포함)로 나눕니다."""
    b64_images, texts = [], []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if _B64_RE.match(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def _make_prompt_func(lang):
    def img_prompt_func(data_dict):
        """검색된 원본을 멀티모달 LLM 입력 형태로 조립: 이미지 먼저, 그다음 텍스트+질문"""
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}}
            for img in data_dict["context"]["images"]
        ]
        formatted_texts = "\n\n---\n\n".join(data_dict["context"]["texts"])
        content.append(
            {
                "type": "text",
                "text": f"{INSTRUCTION[lang]}\n\n질문(Question): {data_dict['question']}\n\n"
                        f"참고 텍스트(Context):\n{formatted_texts}",
            }
        )
        return [HumanMessage(content=content)]

    return img_prompt_func


def build_multimodal_chain(retriever, lang="ko"):
    """Final Answer Pipeline: 검색 → 이미지/텍스트 분리 → 프롬프트 조립 → 멀티모달 LLM"""
    model = ChatOpenAI(temperature=0, model=ANSWER_MODEL, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(_make_prompt_func(lang))
        | model
        | StrOutputParser()
    )


def build_text_only_chain(text_retriever):
    """비교용 텍스트 전용 RAG 체인 (같은 모델·같은 지시문, 이미지 없음)"""
    model = ChatOpenAI(temperature=0, model=ANSWER_MODEL, max_tokens=1024)
    return (
        {
            "context": text_retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(_make_prompt_func("ko"))
        | model
        | StrOutputParser()
    )


# ---------- 번역 체인 (영어 답변 → 한국어) ----------
TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    "다음 영문 답변을 자연스러운 한국어로 번역하세요. 숫자와 고유명사는 그대로 두세요. "
    "번역문만 출력하세요.\n\n{answer}"
)


def ollama_available(url="http://localhost:11434"):
    try:
        return requests.get(url, timeout=1).ok
    except requests.RequestException:
        return False


def get_translator(ollama_model="llama3.1:8b"):
    """로컬 Ollama(오픈소스 Llama 3.1)가 있으면 사용, 없으면 gpt-4o-mini로 대체"""
    if ollama_available():
        from langchain_ollama.llms import OllamaLLM

        return OllamaLLM(model=ollama_model), f"ollama:{ollama_model}"
    return ChatOpenAI(temperature=0, model=LLM_MODEL), f"openai:{LLM_MODEL}"


def build_translate_chain(english_chain, translator):
    return english_chain | {"answer": RunnablePassthrough()} | TRANSLATE_PROMPT | translator | StrOutputParser()
