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
import requests
import xml.etree.ElementTree as ET
from konlpy.tag import Okt
from bs4 import BeautifulSoup
import time
import httpx

# Google Cloud TTS 라이브러리
from google.cloud import texttospeech

# ==========================================
# 1) GOOGLE_APPLICATION_CREDENTIALS 환경 변수 처리
#    - 환경 변수로 넘어온 값이 JSON 문자열이면, 임시 파일로 덤프
# ==========================================
creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_env:
    if creds_env.strip().startswith("{"):
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(creds_env)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

# ==========================================
# 2) 나머지 환경 변수 확인
# ==========================================
api_key = os.getenv("OPENAI_API_KEY")
korean_dict_api_key = os.getenv("KOREAN_DICT_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not korean_dict_api_key:
    raise ValueError("KOREAN_DICT_API_KEY 환경 변수가 설정되지 않았습니다.")

# ==========================================
# 3) OpenAI 클라이언트 초기화
# ==========================================
client = OpenAI(api_key=api_key)
g2p = G2p()
transliter = Transliter(academic)
ok t = Okt()

# ==========================================
# 4) FastAPI 앱 초기화 및 CORS 설정
# ==========================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 배포 시 실제 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5) Pydantic 모델 정의
# ==========================================
class TextInput(BaseModel):
    text: str

# ==========================================
# 6) 시스템 프롬프트 정의
# ==========================================
SYSTEM_PROMPT = """너는 한국어 문장을 단순하게 바꾸는 전문가야.
...(생략)"""

# ==========================================
# 7) TTS 파일 저장 디렉터리 및 StaticFiles 마운트
# ==========================================
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
BASE_URL = f"https://{render_host}" if render_host else "http://localhost:8000"

# ==========================================
# 8) 헬퍼 함수
# ==========================================
async def get_word_info_filtered_async(word: str):
    # 기존 get_word_info_filtered 로직을 비동기로 wrapping
    url = "https://stdict.korean.go.kr/api/search.do"
    params = {"key": korean_dict_api_key, "q": word, "req_type": "xml"}
    async with httpx.AsyncClient() as client_http:
        resp = await client_http.get(url, params=params, timeout=5.0)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")
    entries = []
    for item in items:
        pos_tag = item.find("pos")
        definition = item.find("definition")
        if not pos_tag or not definition:
            continue
        pos = pos_tag.text.strip()
        def_text = definition.text.strip()
        if pos and def_text:
            entries.append({"pos": pos, "definition": def_text})
    return entries[:4]

# 동기 헬퍼 유지
def create_chat_completion(system_input: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_input}, {"role": "user", "content": user_input}],
        temperature=temperature,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

# 쓰레드풀 헬퍼 (기존 동기 작업 비동기로)
async def to_thread(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)

# ==========================================
# 9) API 엔드포인트 정의 (async + 병렬)
# ==========================================
@app.post("/translate-to-easy-korean")
async def translate_to_easy_korean(input_data: TextInput):
    start = time.time()

    # 1) GPT 호출 (순차)
    translated = await to_thread(create_chat_completion, SYSTEM_PROMPT, input_data.text)

    # 2) 병렬로 로마자 변환, 영어번역, 키워드 POS
    orig_roman_task  = to_thread(lambda txt: transliter.translit(g2p(txt)), input_data.text)
    trans_roman_task = to_thread(lambda txt: transliter.translit(g2p(txt)), translated)
    trans_en_task    = to_thread(GoogleTranslator(source="ko", target="en").translate, translated)
    pos_task         = to_thread(okt.pos, translated, stem=True)

    original_roman, translated_roman, translated_en, pos_result = await asyncio.gather(
        orig_roman_task, trans_roman_task, trans_en_task, pos_task
    )

    # 3) 병렬 사전 조회
    keywords = [(w, p) for w, p in pos_result]
    dict_tasks = [get_word_info_filtered_async(w) for w, _ in keywords]
    definitions = await asyncio.gather(*dict_tasks)
    keyword_dictionary = [{"word": w, "pos": p, "definitions": defs} for (w, p), defs in zip(keywords, definitions)]

    total_time = time.time() - start
    return JSONResponse({
        "original_text": input_data.text,
        "translated_text": translated,
        "original_romanized_pronunciation": original_roman,
        "translated_romanized_pronunciation": translated_roman,
        "translated_english_translation": translated_en,
        "keyword_dictionary": keyword_dictionary,
        "processing_time": total_time
    })

# (기타 엔드포인트 및 TTS 기능은 기존 코드 유지)
