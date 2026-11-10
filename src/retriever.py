"""③ 다중 벡터 검색기: '요약'으로 찾고 '원본'을 돌려준다. + ④ 검색된 원본으로 답하는 RAG 체인."""
import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"  # 임베딩은 OpenAI API 호출 (로컬 계산 아님)
ANSWER_MODEL = "gpt-4o-mini"                # 원본 이미지·텍스트를 보고 최종 답변
ID_KEY = "doc_id"


# ======================================================================
# ③ 다중 벡터 검색기
# ======================================================================
def create_multi_vector_retriever(k=4):
    """요약 → 벡터DB(색인 카드), 원본 → docstore(서가). 두 저장소는 doc_id로 연결."""
    vectorstore = Chroma(
        # 노트북을 여러 번 실행해도 이전 데이터가 섞이지 않도록 매번 새 컬렉션
        collection_name=f"multimodal_rag_{uuid.uuid4().hex[:8]}",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )


def add_documents(retriever, doc_summaries, doc_contents, doc_type):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i], "type": doc_type})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)          # 요약 → 검색용
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))  # 원본 → 답변 생성용


def matched_types(retriever, query):
    """질문에 매칭된 요약들의 원본 종류(text/table/image)를 순서대로 반환 (근거 확인용)"""
    hits = retriever.vectorstore.similarity_search(query, **retriever.search_kwargs)
    return [h.metadata.get("type") for h in hits]


def create_text_only_retriever(text_chunks, k=4):
    """비교용 '텍스트 전용 RAG': 텍스트 원문 청크만 임베딩해 검색 (표·이미지 없음)"""
    vectorstore = Chroma.from_texts(
        text_chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=f"text_only_{uuid.uuid4().hex[:8]}",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ======================================================================
# ④ 답변 생성: 검색된 '원본' 텍스트·표·이미지를 멀티모달 LLM에 넣어 답변
# ======================================================================
_B64_RE = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")

INSTRUCTION = {
    "ko": "아래 자료(이미지·표·텍스트)를 근거로 질문에 한국어로 답하세요.\n"
          "그래프나 표에서 읽은 수치는 구체적으로 인용하세요.\n"
          "자료에 없는 내용은 모른다고 답하세요.",
    "en": "Answer the question in English based only on the materials below (images, tables, text).\n"
          "Quote specific numbers read from charts or tables.\n"
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
        """검색된 원본을 멀티모달 LLM 입력 형태로 조립: 이미지 먼저, 그다음 지시문+질문+텍스트"""
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


def build_rag_chain(retriever, lang="ko"):
    """
    검색 → 이미지/텍스트 분리 → 프롬프트 조립 → 멀티모달 LLM → 문자열
    retriever에 다중 벡터 검색기를 넣으면 '멀티모달 RAG', 텍스트 전용 검색기를 넣으면 '텍스트 RAG'
    (같은 모델·같은 지시문을 써서 검색 대상만 다르게 비교)
    """
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
