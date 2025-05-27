import os
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from gtts import gTTS
from g2pk import G2p
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from korean_romanizer.romanizer import Romanizer # 로마자 표기를 위해 추가
import uuid
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --- 환경 변수 설정 ---
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API 키를 설정해주세요.")

# --- OpenAI 클라이언트 초기화 ---
client = OpenAI(api_key=api_key)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

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

# TTS 음성 파일을 저장할 디렉토리 설정 및 생성
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# TTS_OUTPUT_DIR을 /tts 경로로 정적 파일 서빙하도록 설정
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

# Render 배포 시 실제 서비스의 기본 URL을 가져오기 위한 환경 변수 활용
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    BASE_URL = f"https://{render_host}" # Render에서는 HTTPS 사용
else:
    BASE_URL = "http://localhost:8000" # 로컬 개발 환경 기본값

# 한글 발음을 로마자로 변환하는 함수
def convert_pronunciation_to_roman(sentence: str) -> str:
    korean_pron = g2p(sentence)  # 발음 변환
    romanized = transliter.translit(korean_pron)
    return romanized

# TTS 음성 파일을 생성하고 저장하는 함수
def generate_tts(text: str) -> str:
    tts = gTTS(text=text, lang='ko')
    filename = f"{uuid.uuid4()}.mp3"  # 고유한 파일명 생성
    filepath = os.path.join(TTS_OUTPUT_DIR, filename)
    tts.save(filepath)
    return filename

# --- API 엔드포인트 정의 ---

@app.get("/")
async def read_root():
    """
    루트 엔드포인트: 서버가 잘 작동하는지 확인합니다.
    """
    return {"message": "SimpleTalk API 서버가 작동 중입니다."}

@app.post("/romanize")
async def romanize(text: str = Form(...)):
    """
    입력된 한국어 문장의 발음을 로마자로 변환하여 반환합니다.
    """
    romanized = convert_pronunciation_to_roman(text)
    return JSONResponse(content={"input": text, "romanized": romanized})

@app.post("/speak")
async def speak(text: str = Form(...)):
    """
    입력된 한국어 문장의 TTS 음성 파일을 생성하고 해당 URL을 반환합니다.
    """
    filename = generate_tts(text)
    tts_url = f"{BASE_URL}/tts/{filename}"
    return JSONResponse(content={"tts_url": tts_url})

@app.post("/translate-to-easy-korean") # 기존에 정의된 이름과 겹치지 않도록 주의
async def translate_to_easy_korean(input_data: TextInput):
    """
    사용자로부터 텍스트를 받아 쉬운 한국어로 번역하고,
    번역된 텍스트의 로마자 발음 표기를 함께 반환합니다.
    """
    try:
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
        romanized_pronunciation = Romanizer(translated_text).romanize()

        return JSONResponse(content={
            "original_text": input_data.text,
            "translated_text": translated_text,
            "romanized_pronunciation": romanized_pronunciation
        })

    except Exception as e:
        print(f"OpenAI API 호출 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"API 처리 중 에러가 발생했습니다: {str(e)}")

# (선택 사항) TTS 파일 존재 여부 확인 API (디버깅용)
@app.get("/check_tts_file/{filename}")
async def check_tts_file(filename: str):
    """
    TTS 파일이 서버에 존재하는지 확인합니다. (디버깅용)
    """
    filepath = os.path.join(TTS_OUTPUT_DIR, filename)
    exists = os.path.exists(filepath)
    return JSONResponse(content={"filename": filename, "exists": exists, "filepath": filepath})
