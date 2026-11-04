"""공통 설정: 경로, API 키 로딩, OS 도구(Poppler/Tesseract) Path 보정."""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"
CACHE_DIR = DATA_DIR / "cache"

# 모델 선택 (README '기술 선택 근거' 참고)
LLM_MODEL = "gpt-4o-mini"        # 텍스트·표 요약 + 최종 답변 (비전 지원, 저렴)
VLM_MODEL = "gpt-4o-mini"        # 이미지 요약
EMBEDDING_MODEL = "text-embedding-3-small"

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


def ensure_os_tools() -> dict:
    """Poppler/Tesseract가 PATH에 없으면 기본 설치 경로를 PATH에 추가하고 상태를 반환합니다."""
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


def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not key or not key.startswith("sk-"):
        raise ValueError("OpenAI API 키가 없습니다. .env 파일의 OPENAI_API_KEY를 확인하세요.")
    os.environ["OPENAI_API_KEY"] = key
    return key
