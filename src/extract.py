"""① 문서 파싱: PDF → 텍스트 / 표 / 이미지 분리 (unstructured)."""
import os
import shutil
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "data" / "figures"

# 윈도우에서 Path 등록 직후 에디터를 재시작하지 않으면 인식이 안 되는 문제를 방지
_WINDOWS_TOOL_DIRS = [
    r"C:\Program Files\Tesseract-OCR",
    *[
        str(p)
        for p in (Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages").glob(
            "oschwartz10612.Poppler*/poppler-*/Library/bin"
        )
    ],
]


def ensure_os_tools():
    """
    Poppler(PDF→이미지)와 Tesseract(OCR)가 PATH에 없으면 기본 설치 경로를 PATH에 추가하고 상태를 반환합니다.
    두 프로그램은 pip로 설치되지 않는 OS 프로그램이라 hi_res 파싱 전에 확인이 필요합니다.
    """
    for d in _WINDOWS_TOOL_DIRS:
        if os.path.isdir(d) and d not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + d
    # 한국어 OCR 언어팩(kor.traineddata)을 프로젝트 로컬 폴더에서 사용 (Program Files 쓰기 권한 불필요)
    local_tessdata = ROOT / ".tessdata"
    if (local_tessdata / "kor.traineddata").exists():
        os.environ["TESSDATA_PREFIX"] = str(local_tessdata)
    return {
        "poppler": shutil.which("pdftoppm"),
        "tesseract": shutil.which("tesseract"),
    }


def extract_pdf_elements(pdf_path, image_dir=FIGURES_DIR, languages=("kor", "eng")):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    pdf_path : PDF 파일 경로
    image_dir: 추출된 이미지(.jpg)가 저장될 폴더
    languages: OCR 언어 (한국어 PDF라서 한국어 + 영어)
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                       # 레이아웃 분석 모델로 정밀 추출 (표·이미지 정확도 핵심)
        languages=list(languages),
        extract_images_in_pdf=True,              # PDF 안의 그림을 파일로 추출
        extract_image_block_types=["Image"],     # 그림(Image) 블록만 이미지로 저장
        extract_image_block_output_dir=str(image_dir),
        infer_table_structure=True,              # 표 구조(행·열) 추론
        chunking_strategy="by_title",            # 제목 기준으로 텍스트 블록 묶기 → CompositeElement
        max_characters=4000,                     # 한 블록 최대 4000자
        new_after_n_chars=3800,                  # 3800자를 넘으면 새 블록 시작
        combine_text_under_n_chars=2000,         # 2000자 미만 조각은 합치기
    )


def categorize_elements(raw_pdf_elements):
    """
    추출한 요소를 텍스트와 표로 분류합니다.
    chunking_strategy="by_title"을 쓰면 표는 Table/TableChunk, 텍스트는 CompositeElement로 나옵니다.
    (이미지는 extract_pdf_elements 단계에서 이미 파일로 저장되어 있음)
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
    - 1단계(unstructured)는 '글자 수' 기준이라 블록 크기가 들쭉날쭉 → LLM이 쓰는 '토큰' 기준으로 고르게
    - overlap 200: 블록 경계에서 문장이 잘려 의미가 사라지는 것을 방지
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))

