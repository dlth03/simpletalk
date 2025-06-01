import os
import uuid
import tempfile
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
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

# Google Cloud TTS 라이브러리
from google.cloud import texttospeech

# --- 환경 변수 설정 ---
api_key = os.getenv("OPENAI_API_KEY")
korean_dict_api_key = os.getenv("KOREAN_DICT_API_KEY")
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not korean_dict_api_key:
    raise ValueError("KOREAN_DICT_API_KEY 환경 변수가 설정되지 않았습니다.")
if not google_credentials:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

# ✅ Google TTS용 환경 변수가 JSON이면 임시 파일로 저장
if google_credentials.strip().startswith("{"):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        temp_file.write(google_credentials)
        temp_path = temp_file.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

# --- OpenAI 클라이언트 초기화 ---
client = OpenAI(api_key=api_key)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 정의 ---
class TextInput(BaseModel):
    text: str

# --- 시스템 프롬프트 ---
SYSTEM_PROMPT = """너는 한국어 문장을 단순하게 바꾸는 전문가야.
입력된 문장은 다음을 중복 포함할 수 있어:
1. 속담 또는 관용어
2. 방언(사투리)
3. 어려운 단어
4. 줄임말
각 항목에 대해 다음과 같이 변환해:
- 속담/관용어는 그 뜻을 자연스럽게 문장 안에 녹여 설명해
- 방언은 표준어로 바꿔.
- 어려운 단어는 쉬운 말로 바꿔.
- 줄임말은 풀어쓴 문장으로 바꿔.
반드시 지켜:
- 변환된 문장만 출력해.
- 설명하지 마.
- 질문이면 질문형을 유지해."""

# --- 모듈 초기화 ---
g2p = G2p()
transliter = Transliter(academic)
okt = Okt()

# --- TTS 파일 저장 경로 설정 ---
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
BASE_URL = f"https://{render_host}" if render_host else "http://localhost:8000"

# --- 유틸 함수 ---

def convert_pronunciation_to_roman(sentence: str) -> str:
    korean_pron = g2p(sentence)
    return transliter.translit(korean_pron)

def generate_tts_to_file(text: str) -> str | None:
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TTS_OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.audio_content)

        return filepath

    except Exception as e:
        print(f"[generate_tts_to_file] GC TTS 예외: {e}")
        return None

# --- API 엔드포인트 ---

@app.get("/")
async def read_root():
    return {"message": "SimpleTalk API 서버가 작동 중입니다."}

@app.post("/romanize")
async def romanize(text: str = Form(...)):
    romanized = convert_pronunciation_to_roman(text)
    return JSONResponse(content={"input": text, "romanized": romanized})

@app.post("/speak")
async def speak(text: str = Form(...)):
    mp3_path = generate_tts_to_file(text)
    if mp3_path is None:
        return JSONResponse(
            status_code=503,
            content={"error": "TTS 서버 일시 장애로 음성 생성 실패"}
        )

    def iterfile():
        with open(mp3_path, "rb") as audio_file:
            while chunk := audio_file.read(4096):
                yield chunk

    return StreamingResponse(iterfile(), media_type="audio/mpeg")

@app.post("/translate-to-easy-korean")
async def translate_to_easy_korean(input_data: TextInput):
    try:
        original_romanized_pronunciation = convert_pronunciation_to_roman(input_data.text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_data.text}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        translated_text = response.choices[0].message.content.strip()
        translated_romanized_pronunciation = convert_pronunciation_to_roman(translated_text)
        translated_english_translation = translate_korean_to_english(translated_text)

        keywords_with_definitions = []
        keywords = extract_keywords(translated_text)
        for word, pos in keywords:
            senses = get_valid_senses_excluding_pronoun(word, pos)
            if senses:
                keywords_with_definitions.append({
                    "word": word,
                    "pos": pos,
                    "definitions": senses
                })

        return JSONResponse(content={
            "original_text": input_data.text,
            "original_romanized_pronunciation": original_romanized_pronunciation,
            "translated_text": translated_text,
            "translated_romanized_pronunciation": translated_romanized_pronunciation,
            "translated_english_translation": translated_english_translation,
            "keyword_dictionary": keywords_with_definitions
        })

    except Exception as e:
        print(f"API 처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"API 처리 중 에러가 발생했습니다: {str(e)}")

# 필요한 나머지 함수들 (extract_keywords, get_valid_senses_excluding_pronoun, translate_korean_to_english 등)도 여기에 존재해야 합니다.
