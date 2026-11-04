"""③ 다중 벡터 검색기: '요약'으로 찾고 '원본'을 돌려준다."""
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings

from .config import EMBEDDING_MODEL

ID_KEY = "doc_id"


def create_multi_vector_retriever(k=4):
    """요약 → 벡터DB(색인 카드 서랍), 원본 → docstore(서가)"""
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
