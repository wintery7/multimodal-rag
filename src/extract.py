"""① 문서 파싱: PDF → 텍스트 / 표 / 이미지 분리 (unstructured)."""
import re
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from .config import FIGURES_DIR


def extract_pdf_elements(pdf_path, image_dir=FIGURES_DIR, languages=("eng",)):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    pdf_path: PDF 파일 경로
    image_dir: 추출된 이미지(.jpg)가 저장될 폴더
    languages: OCR 언어 (한국어 PDF면 ("kor", "eng"))
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                       # 레이아웃 분석 모델로 정밀 추출 (표·이미지 정확도 핵심)
        languages=list(languages),
        extract_images_in_pdf=True,              # PDF 안의 그림을 파일로 추출
        extract_image_block_types=["Image"],     # 그림(Image) 블록만 이미지로 저장
        extract_image_block_output_dir=str(image_dir),
        infer_table_structure=True,              # 표 구조(HTML) 추론
        chunking_strategy="by_title",            # 제목 기준으로 텍스트 블록 묶기
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )


def figure_captions(pdf_path):
    """
    페이지별 그림 제목 목록을 반환합니다. {페이지번호: ["[그림4] 포항철강산단 1차금속 생산액", ...]}
    hi_res가 잘라낸 그래프 이미지에는 제목이 빠져 있는 경우가 많아, VLM 요약 시 힌트로 사용합니다.
    """
    from pypdf import PdfReader

    captions = {}
    for page_no, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = page.extract_text() or ""
        found = re.findall(r"\[그림\s*\d+\][^\[\n]*", text)
        if found:
            captions[page_no] = [re.sub(r"(\d\))+\s*$", "", c).strip() for c in found]
    return captions


def categorize_elements(raw_pdf_elements):
    """
    추출한 요소를 텍스트와 표로 분류합니다.
    chunking_strategy="by_title"을 쓰면 표는 Table/TableChunk, 텍스트는 CompositeElement로 나옵니다.
    """
    tables, texts = [], []
    for element in raw_pdf_elements:
        name = type(element).__name__
        if name in ("Table", "TableChunk"):
            tables.append(str(element))
        elif name == "CompositeElement":
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """
    텍스트 블록을 하나로 이어 붙인 뒤 토큰 기준으로 다시 자릅니다.
    overlap 200: 블록 경계에서 문장이 잘려 의미가 사라지는 것을 방지
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))
