import os
import uuid
import tempfile
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
from bs4 import BeautifulSoup # <-- 새로 추가된 임포트
import time # <-- 시간 측정을 위해 time 모듈 임포트

# Google Cloud TTS 라이브러리
from google.cloud import texttospeech

# ==========================================
# 1) GOOGLE_APPLICATION_CREDENTIALS 환경 변수 처리
#    - 환경 변수로 넘어온 값이 JSON 문자열이면, 임시 파일로 덤프
# ==========================================
creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_env:
    if creds_env.strip().startswith("{"):
        # JSON 문자열이면 임시 파일로 저장
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(creds_env)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        print(f"[INFO] 서비스 계정 JSON 임시 파일 생성: {temp_path}")
    else:
        print(f"[INFO] GOOGLE_APPLICATION_CREDENTIALS에 파일 경로 지정: {creds_env}")
else:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

# ==========================================
# 2) 나머지 환경 변수 확인
# ==========================================
api_key = os.getenv("OPENAI_API_KEY")
korean_dict_api_key = os.getenv("KOREAN_DICT_API_KEY") # <-- 기존 API 키 변수

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
입력된 문장은 다음을 포함할 수 있어:
1. 속담/관용어
2. 방언(사투리)
3. 어려운 단어
4. 줄임말
각 항목에 대해 다음과 같이 변환해:
- 속담/관용어: 그 뜻을 자연스럽게 문장 안에 설명
  예시) 입력: 배가 불렀네? / 출력: 지금 가진 걸 당연하게 생각하는 거야?
  예시) 입력: 발 없는 말이 천리간다. / 출력: 소문은 빠르게 퍼진다.
- 방언: 표준어로 바꾸기
  예시) 입력: 니 오늘 뭐하노? / 출력: 너 오늘 뭐 해?
  입력: 정구지 / 출력: 부추
- 어려운 단어: 초등 1~2학년도 이해할 수 있는 쉬운 말로 바꾸기
  예시) 입력: 당신의 요청은 거절되었습니다. 추가 서류를 제출하세요.
        / 출력: 당신의 요청은 안 됩니다. 서류를 더 내야 합니다.
- 줄임말: 풀어쓴 문장으로 바꾸기
  예시) 입력: 할많하않 / 출력: 할 말은 많지만 하지 않겠어
다음은 반드시 지켜야 함:
- 변환된 문장 또는 단어만 출력
- 설명 덧붙이지 않기
- 의문문을 그대로 질문 형태로 유지하면서 쉬운 단어로 바꾸기
  예시) 입력: 국무총리는 어떻게 임명돼? / 출력: 국무총리는 어떻게 정해?"""

# ==========================================
# 7) 기존 모듈 초기화 (g2pk, hangul-romanize, Okt 등)
# ==========================================
g2p = G2p()
transliter = Transliter(academic)
okt = Okt() # Okt는 이미 존재하므로 중복 초기화 방지

# 새로 추가된 품사 매핑
okt_to_nine_pos = {
    "Noun": "명사",
    "Pronoun": "대명사",
    "Number": "수사",
    "Verb": "동사",
    "Adjective": "형용사",
    "Adverb": "부사",
    "Exclamation": "감탄사",
    "Determiner": "관형사",
    "Conjunction": "부사",      # 전통 문법상 부사 취급
    "Foreign": "명사",          # 외래어는 명사 취급
    "Alpha": "명사",            # 알파벳도 명사 취급
    "Josa": None,
    "Eomi": None,
    "PreEomi": None,
    "Modifier": None,
    "Punctuation": None,
}

# ==========================================
# 8) TTS 파일 저장 디렉터리 및 StaticFiles 마운트
# ==========================================
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# 여기서 “tts_files/” 내부의 MP3 파일들을 /tts/ 경로로 서빙
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

# 배포 환경 호스트네임 (예: Render)
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    BASE_URL = f"https://{render_host}"
else:
    BASE_URL = "http://localhost:8000"

# ==========================================
# 9) 헬퍼 함수들 (수정 및 추가)
# ==========================================
def convert_pronunciation_to_roman(sentence: str) -> str:
    korean_pron = g2p(sentence)
    romanized = transliter.translit(korean_pron)
    return romanized

def translate_korean_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="ko", target="en").translate(text)
    except Exception as e:
        print(f"[Translation error] {e}")
        return f"Translation error: {e}"

# 1. 문장에서 단어를 9품사 기준으로 추출 (기존 extract_keywords 대체)
def extract_words_9pos(sentence: str):
    words = okt.pos(sentence, stem=True)
    result = []
    for word, tag in words:
        pos = okt_to_nine_pos.get(tag)
        if word == '아주' and pos == '명사': # Okt가 '아주'를 명사로 잘못 분류하는 경우
            pos = '부사' # '명사'로 분류된 '아주'를 '부사'로 변경

        if pos:
            result.append((word, pos))
    seen = set()
    ordered_unique = []
    for w, p in result:
        if (w,p) not in seen:
            seen.add((w,p))
            ordered_unique.append((w,p))
    return ordered_unique

# 2. 조건에 따라 여러 품사를 허용하도록 수정 (기존 get_valid_senses_excluding_pronoun 대체)
def get_word_info_filtered(word: str):
    start_time_single_dict_call = time.time() # <-- 개별 사전 호출 시간 측정 시작
    url = "https://stdict.korean.go.kr/api/search.do"
    params = {
        "key": korean_dict_api_key,
        "q": word,
        "req_type": "xml"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"[ERROR] 국어사전 API 요청 실패: {response.status_code}, {response.text}")
        print(f"[Timing] Single Dictionary call for '{word}' failed: {time.time() - start_time_single_dict_call:.4f}s") # <-- 실패 시간 로깅
        return []

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")

    entries = []
    for item in items:
        definition = item.find("definition")
        pos_tag = item.find("pos")
        sup_no = item.find("sup_no")

        if pos_tag is None:
            continue
        pos_text = pos_tag.text.strip()
        if pos_text == "" or pos_text == "품사 없음":
            continue

        if definition is None or not definition.text.strip():
            continue

        definition_text = definition.text.strip()
        sup_no_text = sup_no.text.strip() if sup_no else ""

        entries.append({
            "sup_no": sup_no_text,
            "pos": pos_text,
            "definition": definition_text
        })
    
    seen_supnos = set()
    unique_entries = []
    for entry in entries:
        if entry["sup_no"] not in seen_supnos:
            seen_supnos.add(entry["sup_no"])
            unique_entries.append(entry)

    sorted_entries = sorted(unique_entries, key=lambda x: 1 if x["pos"] == "명사" else 0)

    if not sorted_entries:
        print(f"[Timing] Single Dictionary call for '{word}' (no results): {time.time() - start_time_single_dict_call:.4f}s") # <-- 결과 없음 시간 로깅
        return []

    result = sorted_entries[:4] # 출력할 양 조절(현재 4개 이하로 출력되도록 설정)
    print(f"[Timing] Single Dictionary call for '{word}': {time.time() - start_time_single_dict_call:.4f}s (results: {len(result)})") # <-- 성공 시간 로깅
    return result


def generate_tts_to_file(text: str) -> str :
    """
    Google Cloud TTS를 사용하여 text를 mp3로 합성한 뒤,
    TTS_OUTPUT_DIR에 저장하고, 해당 파일 경로를 반환합니다.
    실패 시 None 반환.
    """
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TTS_OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.audio_content)

        return filepath

    except Exception as e:
        import traceback
        print("[generate_tts_to_file] Google TTS 예외 전체 로그:")
        traceback.print_exc()
        return None

# ==========================================
# 10) API 엔드포인트 정의
# ==========================================
@app.get("/")
async def read_root():
    return {"message": "SimpleTalk API 서버 작동 중입니다."}


@app.post("/romanize")
async def romanize(text: str = Form(...)):
    romanized = convert_pronunciation_to_roman(text)
    return JSONResponse(content={"input": text, "romanized": romanized})


@app.post("/speak")
async def speak(text: str = Form(...)):
    """
    Form으로 들어온 'text'를 Google Cloud TTS로 합성하여 mp3 파일을 생성 →
    그 파일의 정적 URL(tts_url)을 JSON으로 반환합니다.
    (바로 StreamingResponse를 보내는 대신, URL만 내려주는 방식)
    실패 시 503 반환.
    """
    mp3_path = generate_tts_to_file(text)
    if mp3_path is None:
        return JSONResponse(
            status_code=503,
            content={"error": "TTS 서버 일시 장애로 음성 생성 실패"}
        )

    filename = os.path.basename(mp3_path)
    tts_url = f"{BASE_URL}/tts/{filename}"
    return JSONResponse(content={"tts_url": tts_url})


@app.post("/translate-to-easy-korean")
async def translate_to_easy_korean(input_data: TextInput):
    start_total = time.time() # <-- 전체 API 처리 시간 측정 시작
    print(f"\n[Timing] --- New Request Received ---")
    print(f"[Timing] Input text: '{input_data.text[:50]}...'") # 긴 텍스트는 잘라서 출력

    try:
        start_romanize_original = time.time()
        original_romanized_pronunciation = convert_pronunciation_to_roman(input_data.text)
        print(f"[Timing] 1. Original Romanization: {time.time() - start_romanize_original:.4f}s")


        start_gpt_call = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_data.text},
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        translated_text = response.choices[0].message.content.strip()
        print(f"[Timing] 2. OpenAI GPT-4o-mini call: {time.time() - start_gpt_call:.4f}s")


        start_romanize_translated = time.time()
        translated_romanized_pronunciation = convert_pronunciation_to_roman(translated_text)
        print(f"[Timing] 3. Translated Romanization: {time.time() - start_romanize_translated:.4f}s")


        start_google_translate = time.time()
        translated_english_translation = translate_korean_to_english(translated_text)
        print(f"[Timing] 4. Google Translate call: {time.time() - start_google_translate:.4f}s")


        keywords_with_definitions = []
        start_extract_keywords_pos = time.time()
        keywords = extract_words_9pos(translated_text) # (단어, 품사) 튜플의 리스트
        print(f"[Timing] 5. Keyword extraction (Okt): {time.time() - start_extract_keywords_pos:.4f}s (Found {len(keywords)} keywords)")

        
        # 키워드별 국어사전 API 호출 시간은 get_word_info_filtered 함수 내부에서 측정됨
        start_dict_calls_total = time.time()
        for i, (word, pos_tag) in enumerate(keywords):
            # get_word_info_filtered 함수 내부에 이미 로깅 코드가 추가되어 있습니다.
            senses = get_word_info_filtered(word)

            if senses:
                formatted_senses = [{"pos": s["pos"], "definition": s["definition"]} for s in senses]
                keywords_with_definitions.append({
                    "word": word,
                    "pos": pos_tag,
                    "definitions": formatted_senses,
                })
        print(f"[Timing] 6. Total Dictionary API calls for {len(keywords)} keywords: {time.time() - start_dict_calls_total:.4f}s")


        total_processing_time = time.time() - start_total
        print(f"[Timing] --- Request Processed --- Total time: {total_processing_time:.4f}s")

        return JSONResponse(content={
            "original_text": input_data.text,
            "original_romanized_pronunciation": original_romanized_pronunciation,
            "translated_text": translated_text,
            "translated_romanized_pronunciation": translated_romanized_pronunciation,
            "translated_english_translation": translated_english_translation,
            "keyword_dictionary": keywords_with_definitions
        })

    except Exception as e:
        import traceback # 예외 발생 시 전체 스택 트레이스 출력
        traceback.print_exc()
        print(f"[ERROR] API 처리 중 에러: {e}")
        total_processing_time = time.time() - start_total
        print(f"[Timing] --- Request Failed --- Total time: {total_processing_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"API 처리 중 에러가 발생했습니다: {e}")
