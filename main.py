import os
import uuid
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
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
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
if not korean_dict_api_key:
    raise ValueError("KOREAN_DICT_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
if not google_credentials:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다. 서비스 계정 JSON 파일 경로를 설정해주세요.")


# --- OpenAI 클라이언트 초기화 ---
client = OpenAI(api_key=api_key)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production에서는 실제 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 정의 ---
class TextInput(BaseModel):
    text: str

# --- 시스템 프롬프트 정의 ---
SYSTEM_PROMPT = """너는 한국어 문장을 단순하게 바꾸는 전문가야.
입력된 문장은 다음을 중복 포함할 수 있어:
1. 속담 또는 관용어
2. 방언(사투리)
3. 어려운 단어
4. 줄임말
각 항목에 대해 다음과 같이 변환해:
- 속담/관용어는 그 뜻을 자연스럽게 문장 안에 녹여 설명해
예시) 입력: 배가 불렀네? / 출력: 지금 가진 걸 당연하게 생각하는 거야?
예시) 입력: 발 없는 말이 천리간다. / 출력 : 소문은 빠르게 퍼진다.
- 방언은 표준어로 바꿔.
예시) 입력: 니 오늘 뭐하노? / 출력: 너 오늘 뭐 해?
입력 : 정구지 / 출력 : 부추
- 어려운 단어는 초등학교 1~2학년이 이해할 수 있는 쉬운 말로 바꿔.
예시) 입력: 당신의 요청은 거절되었습니다. 추가 서류를 제출하세요. / 출력: 당신의 요청은 안 됩니다. 서류를 더 내야 합니다.
- 줄임말은 풀어 쓴 문장으로 바꿔.
예시) 입력: 할많하않 / 출력: 할 말은 많지만 하지 않겠어
다음은 반드시 지켜:
- 변환된 문장 또는 단어만 출력해.
- 설명을 덧붙이지 마.
- 의문문이 들어오면, 절대 대답하지 마.
질문 형태를 그대로 유지하면서 쉬운 단어로 바꿔.
예시) 입력 : 국무총리는 어떻게 임명돼? / 출력 : 국무총리는 어떻게 정해?"""

# --- 기존 모듈 초기화 ---
g2p = G2p()
transliter = Transliter(academic)
okt = Okt()

# --- TTS 파일 저장 디렉터리 설정 (필요 시 캐시용으로 사용) ---
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
# (StaticFiles 마운트는 URL 형태가 필요할 때를 대비해 두지만,
#  /speak에서는 사용하지 않습니다.)
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

# 배포 환경 호스트네임 (이제 /tts URL을 쓰지 않으므로 크게 중요하지 않음)
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    BASE_URL = f"https://{render_host}"
else:
    BASE_URL = "http://localhost:8000"


# --- 헬퍼 함수들 ---

def convert_pronunciation_to_roman(sentence: str) -> str:
    korean_pron = g2p(sentence)
    romanized = transliter.translit(korean_pron)
    return romanized


def generate_tts_to_file(text: str) -> str | None:
    """
    Google Cloud TTS를 사용하여 'text'를 mp3 파일로 생성하고,
    TTS_OUTPUT_DIR에 저장한 뒤 파일 경로를 반환합니다.
    실패 시 None을 반환합니다.
    """
    try:
        # 클라이언트 초기화 (환경 변수 GOOGLE_APPLICATION_CREDENTIALS 필요)
        tts_client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

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


# --- API 엔드포인트 정의 ---

@app.get("/")
async def read_root():
    return {"message": "SimpleTalk API 서버가 작동 중입니다."}


@app.post("/romanize")
async def romanize(text: str = Form(...)):
    romanized = convert_pronunciation_to_roman(text)
    return JSONResponse(content={"input": text, "romanized": romanized})


@app.post("/speak")
async def speak(text: str = Form(...)):
    """
    Form으로 들어온 'text'를 Google Cloud TTS로 mp3 파일로 생성한 뒤,
    해당 파일의 바이트를 StreamingResponse로 바로 스트리밍하여 반환합니다.
    실패 시 503(Service Unavailable)을 반환합니다.
    """
    # 1) TTS 파일을 생성하고 로컬 경로를 받아옴
    mp3_path = generate_tts_to_file(text)
    if mp3_path is None:
        return JSONResponse(
            status_code=503,
            content={"error": "TTS 서버 일시 장애로 음성 생성 실패"}
        )

    # 2) 생성된 파일을 StreamingResponse로 스트리밍
    def iterfile():
        with open(mp3_path, "rb") as audio_file:
            while chunk := audio_file.read(4096):
                yield chunk
        # (선택) 재생 후 파일을 삭제하고 싶으면 여기서 os.remove(mp3_path)를 호출

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
