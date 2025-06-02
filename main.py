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

# ==========================================
# 1) GOOGLE_APPLICATION_CREDENTIALS 환경 변수 처리
#    - 환경 변수 값이 “JSON 문자열”일 수도, “실제 파일 경로”일 수도 있음
#    - JSON 문자열일 때는 임시 파일로 덤프 → 환경 변수에 덤프된 경로 설정
# ==========================================
creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_env:
    # JSON 문자열인지(“{"로 시작하는가”) 확인
    if creds_env.strip().startswith("{"):
        try:
            # 임시 파일 생성 (suffix=".json"로 확장자 지정)
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, "w") as f:
                f.write(creds_env)
            # 환경 변수를 덤프된 파일 경로로 덮어쓰기
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            print(f"서비스 계정 JSON 임시 파일 생성: {temp_path}")
        except Exception as e:
            print("✨ GOOGLE JSON 덤프 오류:", e)
            raise
    else:
        # 이미 “파일 경로” 형태라면 그대로 둡니다.
        print(f"GOOGLE_APPLICATION_CREDENTIALS에 파일 경로 지정: {creds_env}")
else:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

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

# ==========================================
# 4) FastAPI 앱 초기화 및 CORS 설정
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production에서는 실제 도메인으로 제한 권장
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
# 6) 시스템 프롬프트 정의 (생략 가능)
# ==========================================
SYSTEM_PROMPT = """너는 한국어 문장을 단순하게 바꾸는 전문가야.
…(중략)…"""

# ==========================================
# 7) 기존 모듈 초기화 (g2pk, hangul-romanize, Konlpy Okt 등)
# ==========================================
g2p = G2p()
transliter = Transliter(academic)
okt = Okt()

# ==========================================
# 8) TTS 파일 저장 디렉터리 설정 (임시 저장용)
# ==========================================
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
# StaticFiles 마운트를 걸어 두었지만, /speak 에서는 사용하지 않습니다.
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

# 배포 환경 호스트네임 (예: Render)
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    BASE_URL = f"https://{render_host}"
else:
    BASE_URL = "http://localhost:8000"

# ==========================================
# 9) 헬퍼 함수들
# ==========================================
def convert_pronunciation_to_roman(sentence: str) -> str:
    korean_pron = g2p(sentence)
    romanized = transliter.translit(korean_pron)
    return romanized

def translate_korean_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='ko', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return f"Translation error: {e}"

def extract_keywords(text: str):
    raw_words = okt.pos(text, stem=True)
    joined_words = []
    skip_next = False

    for i in range(len(raw_words)):
        if skip_next:
            skip_next = False
            continue

        word, pos = raw_words[i]
        if (
            i + 1 < len(raw_words)
            and pos == 'Noun'
            and raw_words[i + 1][0] == '다'
            and raw_words[i + 1][1] == 'Eomi'
        ):
            joined_words.append((word + '다', 'Verb'))
            skip_next = True
        elif pos in ['Noun', 'Verb', 'Adjective', 'Adverb']:
            joined_words.append((word, pos))

    seen = set()
    ordered_unique = []
    for w, p in joined_words:
        if w not in seen:
            seen.add(w)
            ordered_unique.append((w, p))
    return ordered_unique

def get_valid_senses_excluding_pronoun(word: str, target_pos: str, max_defs: int = 3):
    pos_map = {
        'Noun': '명사',
        'Verb': '동사',
        'Adjective': '형용사',
        'Adverb': '부사'
    }
    mapped_pos = pos_map.get(target_pos)
    if not mapped_pos:
        return []

    url = "https://stdict.korean.go.kr/api/search.do"
    params = {
        'key': korean_dict_api_key,
        'q': word,
        'req_type': 'xml'
    }

    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)

    senses = []
    seen_supnos = set()

    for item in root.findall('item'):
        sup_no = item.findtext('sup_no', default='0')
        pos = item.findtext('pos', default='')

        if pos == '대명사' or pos != mapped_pos:
            continue
        if sup_no in seen_supnos:
            continue
        seen_supnos.add(sup_no)

        sense = item.find('sense')
        if sense is None:
            continue
        definition = sense.findtext('definition', default='뜻풀이 없음')

        senses.append({
            'pos': pos,
            'definition': definition
        })
        if len(senses) >= max_defs:
            break

    return senses

def generate_tts_to_file(text: str) -> str | None:
    """
    Google Cloud TTS를 사용하여 text를 mp3로 합성 후
    TTS_OUTPUT_DIR에 저장하고, 경로를 반환. 실패 시 None.
    """
    try:
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
        import traceback
        print("[generate_tts_to_file] GC TTS 예외 전체 로그:")
        traceback.print_exc()
        return None

# ==========================================
# 10) API 엔드포인트 정의
# ==========================================
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
    Form으로 들어온 'text'를 Google Cloud TTS로 mp3 합성 → 
    StreamingResponse로 엠피쓰리 바이트 내려줌. 실패 시 503.
    """
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
        # 재생 완료 후 파일 삭제 가능: os.remove(mp3_path)

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
