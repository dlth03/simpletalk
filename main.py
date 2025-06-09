import os
import uuid
import tempfile
import asyncio
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from g2pk import G2p
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from deep_translator import GoogleTranslator
import xml.etree.ElementTree as ET
from konlpy.tag import Okt
from bs4 import BeautifulSoup
import time
import httpx

# Google Cloud TTS 라이브러리
from google.cloud import texttospeech

# ==========================================
# 1) GOOGLE_APPLICATION_CREDENTIALS 환경 변수 처리
# ==========================================
creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_env and creds_env.strip().startswith("{"):
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(creds_env)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

# ==========================================
# 2) 환경 변수 확인
# ==========================================
api_key = os.getenv("OPENAI_API_KEY")
korean_dict_api_key = os.getenv("KOREAN_DICT_API_KEY")
if not api_key or not korean_dict_api_key:
    raise ValueError("필수 환경 변수가 설정되지 않았습니다.")

# ==========================================
# 3) 클라이언트 초기화
# ==========================================
client = OpenAI(api_key=api_key)
g2p = G2p()
transliter = Transliter(academic)
okt = Okt()

# 간단 캐시
_dict_cache = {}

# ==========================================
# FastAPI 및 CORS 설정
# ==========================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS 디렉토리 및 StaticFiles
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
BASE_URL = f"https://{render_host}" if render_host else "http://localhost:8000"

# Pydantic 모델
class TextInput(BaseModel):
    text: str

# 시스템 프롬프트
SYSTEM_PROMPT = """너는 한국어 문장을 단순하게 바꾸는 전문가야.
...(생략)"""

# ==========================================
# 비동기 국어사전 조회 함수
# ==========================================
async def get_word_info_filtered_async(word: str):
    if word in _dict_cache:
        return _dict_cache[word]
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://stdict.korean.go.kr/api/search.do",
            params={"key": korean_dict_api_key, "q": word, "req_type": "xml"},
            timeout=5.0
        )
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.content, "xml")
    # parse items
    entries = []
    for item in soup.find_all("item"):
        pos = item.pos and item.pos.text.strip()
        definition = item.definition and item.definition.text.strip()
        if pos and definition:
            entries.append({"pos": pos, "definition": definition})
    result = entries[:4]
    _dict_cache[word] = result
    return result

# ==========================================
# Chat Completion 헬퍼
# ==========================================
def create_chat_completion(system_input: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_input}, {"role": "user", "content": user_input}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# ==========================================
# 쓰레드풀 비동기 실행 헬퍼
# ==========================================
async def to_thread(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)

# ==========================================
# 엔드포인트 정의: 순서 재조정 + 병렬 사전 조회
# ==========================================
@app.post("/translate-to-easy-korean")
async def translate_to_easy_korean(input_data: TextInput):
    start_time = time.time()

    # 1) GPT 호출 (필수 순서)
    translated = await to_thread(create_chat_completion, SYSTEM_PROMPT, input_data.text)
    if not translated:
        raise HTTPException(status_code=500, detail="AI 응답 실패")

    # 2) 병렬: 로마자 변환, 영어 번역, 키워드 추출
    orig_roman_task   = to_thread(lambda t: transliter.translit(g2p(t)), input_data.text)
    trans_roman_task  = to_thread(lambda t: transliter.translit(g2p(t)), translated)
    trans_en_task     = to_thread(GoogleTranslator(source="ko", target="en").translate, translated)
    kw_task           = to_thread(okt.pos, translated, True)

    original_roman, translated_roman, translated_en, raw_keywords = await asyncio.gather(
        orig_roman_task, trans_roman_task, trans_en_task, kw_task
    )

    # 형식: [(word, pos), ...]
    kw_list = [(w, p) for w, p in raw_keywords]

    # 3) 병렬 사전 조회 (키워드 리스트 확보 후)
    dict_tasks = [get_word_info_filtered_async(word) for word, _ in kw_list]
    dict_results = await asyncio.gather(*dict_tasks)
    keyword_dictionary = [
        {"word": w, "pos": p, "definitions": defs}
        for (w, p), defs in zip(kw_list, dict_results)
    ]

    total_time = time.time() - start_time
    return JSONResponse({
        "original_text": input_data.text,
        "translated_text": translated,
        "original_romanized": original_roman,
        "translated_romanized": translated_roman,
        "translated_english": translated_en,
        "keyword_dictionary": keyword_dictionary,
        "processing_time": total_time
    })

# (기타 엔드포인트 및 TTS 기능은 기존 유지)
