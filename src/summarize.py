"""② 요약 생성: 텍스트·표는 LLM, 이미지는 VLM으로 '검색용 요약'을 만듭니다."""
import base64
import hashlib
import json
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import CACHE_DIR, LLM_MODEL, VLM_MODEL

TEXT_PROMPT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 한국어로 작성해주세요.
핵심 수치, 연도, 항목명은 반드시 요약에 포함하세요.
요약할 표 또는 텍스트: {element}"""

IMAGE_PROMPT_KOR = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
Include the chart type, title, axis labels, legend items, key numbers and the overall trend.
The summary should be written in Korean (Hangul)."""


# ---------- 캐시: 같은 입력은 API를 다시 호출하지 않음 (비용 절감) ----------
def _cache_path(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _load_cache(name):
    p = _cache_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_cache(name, cache):
    _cache_path(name).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------- 텍스트 · 표 요약 ----------
def generate_text_summaries(texts, tables, summarize_texts=False, max_concurrency=5):
    """
    texts: 텍스트 리스트 / tables: 표 리스트
    summarize_texts: False면 텍스트 원문을 그대로 요약 자리에 사용
    """
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model=LLM_MODEL)
    # LCEL: 입력 포장 → 프롬프트 → LLM → 문자열
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    cache = _load_cache("text_summaries")

    def run(items):
        todo = [x for x in items if _h(x) not in cache]
        if todo:
            outs = summarize_chain.batch(todo, {"max_concurrency": max_concurrency})
            cache.update({_h(x): o for x, o in zip(todo, outs)})
            _save_cache("text_summaries", cache)
        return [cache[_h(x)] for x in items]

    text_summaries = run(texts) if (texts and summarize_texts) else list(texts)
    table_summaries = run(tables) if tables else []
    return text_summaries, table_summaries


# ---------- 이미지 요약 ----------
def encode_image(image_path):
    """이미지를 base64 문자열로 변환 (그림을 '문자열 상자'에 포장)"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt=IMAGE_PROMPT_KOR):
    """VLM(gpt-4o-mini)으로 이미지 요약 생성"""
    chat = ChatOpenAI(model=VLM_MODEL, max_tokens=1024, temperature=0)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path, min_kb=8):
    """
    폴더 내 이미지 전체를 Base64로 인코딩하고 요약합니다.
    min_kb: 로고·아이콘 같은 작은 이미지는 검색 노이즈라 제외
    반환: (img_base64_list, image_summaries, img_files)
    """
    cache = _load_cache("image_summaries")
    img_base64_list, image_summaries, img_files = [], [], []
    for img_file in sorted(os.listdir(path)):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(path, img_file)
        if os.path.getsize(img_path) < min_kb * 1024:
            continue
        b64 = encode_image(img_path)
        key = _h(b64)
        if key not in cache:
            cache[key] = image_summarize(b64)
            _save_cache("image_summaries", cache)
        img_base64_list.append(b64)
        image_summaries.append(cache[key])
        img_files.append(img_file)
    return img_base64_list, image_summaries, img_files
