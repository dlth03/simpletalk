import os
import uuid
import tempfile
import asyncio
import functools # <-- 이 줄이 추가되었습니다!

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
import requests # 동기 요청용 (사용 안 할 예정이지만 import 유지)
import xml.etree.ElementTree as ET
from konlpy.tag import Okt
from bs4 import BeautifulSoup
import time
import httpx # 비동기 HTTP 요청용

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
korean_dict_api_key = os.getenv("KOREAN_DICT_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not korean_dict_api_key:
    raise ValueError("KOREAN_DICT_API_KEY 환경 변수가 설정되지 않았습니다.")

# ==========================================
# 3) 클라이언트 및 모듈 초기화
# ==========================================
client = OpenAI(api_key=api_key)
g2p = G2p()
transliter = Transliter(academic)
okt = Okt()

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
입력된 문장은 다음을 중복 포함할 수 있어:
1. 속담 또는 관용어
2. 방언(사투리)
3. 어려운 단어
4. 줄임말
각 항목에 대해 다음과 같이 변환해:
- 속담/관용어는 그 뜻을 자연스럽게 문장에 맞게 설명해
예시) 입력: 배가 불렀네? / 출력: 지금 가진 걸 당연하게 생각하는 거야?
입력 : 손이 크다 / 출력 : 씀씀이가 후하다.
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

# ==========================================
# 7) TTS 파일 저장 디렉터리 및 StaticFiles 마운트
# ==========================================
TTS_OUTPUT_DIR = "tts_files"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
app.mount("/tts", StaticFiles(directory=TTS_OUTPUT_DIR), name="tts")

# 배포 환경 호스트네임 (예: Render)
render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_host:
    BASE_URL = f"https://{render_host}"
else:
    BASE_URL = "http://localhost:8000"

# ==========================================
# 8) 헬퍼 함수들 (수정 및 추가)
# ==========================================

# to_thread 헬퍼 함수 수정: 키워드 인자도 받을 수 있도록
async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    # functools.partial을 사용하여 함수와 인자들을 묶어서 executor에 전달
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

def convert_pronunciation_to_roman_sync(sentence: str) -> str:
    # 이 함수는 to_thread를 통해 실행될 동기 함수입니다.
    korean_pron = g2p(sentence)
    romanized = transliter.translit(korean_pron)
    return romanized

def translate_korean_to_english_sync(text: str) -> str:
    # 이 함수는 to_thread를 통해 실행될 동기 함수입니다.
    try:
        return GoogleTranslator(source="ko", target="en").translate(text)
    except Exception as e:
        print(f"[Translation error] {e}")
        return f"Translation error: {e}"

def extract_words_9pos_sync(sentence: str):
    # 이 함수는 to_thread를 통해 실행될 동기 함수입니다.
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

# 국어사전 API 호출 비동기 함수 (httpx 사용)
async def get_word_info_filtered_async(word: str):
    url = "https://stdict.korean.go.kr/api/search.do"
    params = {
        "key": korean_dict_api_key,
        "q": word,
        "req_type": "xml"
    }
    
    start_time_single_dict_call = time.time()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client_http:
            response = await client_http.get(url, params=params)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            entries = []
            for item in items:
                definition_tag = item.find("definition")
                pos_tag = item.find("pos")
                sup_no_tag = item.find("sup_no")

                # 태그가 없거나 내용이 비어있는 경우 스킵
                if pos_tag is None or not pos_tag.text.strip():
                    continue
                if definition_tag is None or not definition_tag.text.strip():
                    continue

                pos_text = pos_tag.text.strip()
                definition_text = definition_tag.text.strip()
                sup_no_text = sup_no_tag.text.strip() if sup_no_tag else ""

                entries.append({
                    "sup_no": sup_no_text,
                    "pos": pos_text,
                    "definition": definition_text
                })
            
            # sup_no 기준으로 중복 제거 및 정렬
            seen_supnos = set()
            unique_entries = []
            for entry in entries:
                if entry["sup_no"] not in seen_supnos:
                    seen_supnos.add(entry["sup_no"])
                    unique_entries.append(entry)

            sorted_entries = sorted(unique_entries, key=lambda x: 1 if x["pos"] == "명사" else 0)

            result = sorted_entries[:4] # 최대 4개 결과 반환
            print(f"[Timing] Single Dictionary call for '{word}': {time.time() - start_time_single_dict_call:.4f}s (results: {len(result)})")
            return result

    except httpx.RequestError as e:
        print(f"[ERROR] 국어사전 API 요청 중 네트워크 오류: {e}")
        print(f"[Timing] Single Dictionary call for '{word}' failed (network error): {time.time() - start_time_single_dict_call:.4f}s")
        return []
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] 국어사전 API 응답 오류: {e.response.status_code}, {e.response.text}")
        print(f"[Timing] Single Dictionary call for '{word}' failed (HTTP error): {time.time() - start_time_single_dict_call:.4f}s")
        return []
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] 국어사전 API 처리 중 예상치 못한 오류: {e}")
        print(f"[Timing] Single Dictionary call for '{word}' failed (unexpected error): {time.time() - start_time_single_dict_call:.4f}s")
        return []


def generate_tts_to_file_sync(text: str) -> str :
    """
    Google Cloud TTS를 사용하여 text를 mp3로 합성한 뒤,
    TTS_OUTPUT_DIR에 저장하고, 해당 파일 경로를 반환합니다.
    실패 시 None 반환. (to_thread를 통해 동기적으로 실행)
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

def create_chat_completion_sync(system_input: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
    # 이 함수는 to_thread를 통해 실행될 동기 함수입니다.
    try:
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=200 # 적절한 max_tokens 설정
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return None

# ==========================================
# 9) API 엔드포인트 정의
# ==========================================
@app.get("/")
async def read_root():
    return {"message": "SimpleTalk API 서버 작동 중입니다."}


@app.post("/romanize")
async def romanize(text: str = Form(...)):
    romanized = await to_thread(convert_pronunciation_to_roman_sync, text)
    return JSONResponse(content={"input": text, "romanized": romanized})


@app.post("/speak")
async def speak(text: str = Form(...)):
    """
    Form으로 들어온 'text'를 Google Cloud TTS로 합성하여 mp3 파일을 생성 →
    그 파일의 정적 URL(tts_url)을 JSON으로 반환합니다.
    (바로 StreamingResponse를 보내는 대신, URL만 내려주는 방식)
    실패 시 503 반환.
    """
    mp3_path = await to_thread(generate_tts_to_file_sync, text)
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
    start_total = time.time()
    print(f"\n[Timing] --- New Request Received ---")
    print(f"[Timing] Input text: '{input_data.text[:50]}...'")

    try:
        # 1) GPT 호출 (동기 함수를 to_thread로 비동기 실행)
        start_gpt_call = time.time()
        # to_thread 함수가 키워드 인자를 받을 수 있도록 수정했으므로, 이제 그대로 전달
        translated_text = await to_thread(create_chat_completion_sync, SYSTEM_PROMPT, input_data.text, model="gpt-4o-mini", temperature=0.7)
        if translated_text is None:
            raise HTTPException(status_code=500, detail="Failed to get response from OpenAI API.")
        print(f"[Timing] 1. OpenAI GPT-4o-mini call: {time.time() - start_gpt_call:.4f}s")

        # 2) 병렬로 실행할 작업들 정의 (모두 to_thread를 통해 동기 함수를 비동기 실행)
        start_parallel_tasks = time.time()
        
        # 원래 문장 로마자 변환
        original_roman_task = to_thread(convert_pronunciation_to_roman_sync, input_data.text)
        
        # 번역된 문장 로마자 변환
        translated_roman_task = to_thread(convert_pronunciation_to_roman_sync, translated_text)
        
        # 번역된 문장 영어 번역
        translated_english_task = to_thread(translate_korean_to_english_sync, translated_text)
        
        # 번역된 문장에서 키워드(품사 정보 포함) 추출
        keywords_okt_task = to_thread(extract_words_9pos_sync, translated_text)

        # 모든 병렬 작업들을 동시에 실행하고 결과 대기
        original_romanized_pronunciation, translated_romanized_pronunciation, \
        translated_english_translation, keywords_with_okt_pos = await asyncio.gather(
            original_roman_task,
            translated_roman_task,
            translated_english_task,
            keywords_okt_task
        )
        print(f"[Timing] 2. Parallel tasks (Romanization, Google Translate, Okt): {time.time() - start_parallel_tasks:.4f}s")

        # 3) 추출된 키워드를 기반으로 국어사전 API 병렬 조회 (httpx 사용)
        start_dict_calls_total = time.time()
        dict_tasks = [get_word_info_filtered_async(word) for word, _ in keywords_with_okt_pos]
        all_definitions_from_dict = await asyncio.gather(*dict_tasks)
        print(f"[Timing] 3. Total Dictionary API calls for {len(keywords_with_okt_pos)} keywords: {time.time() - start_dict_calls_total:.4f}s")
        
        keywords_with_definitions = []
        for i, (word, pos_tag) in enumerate(keywords_with_okt_pos):
            senses = all_definitions_from_dict[i]
            if senses:
                # `get_word_info_filtered_async`는 이미 원하는 형식으로 반환하므로 추가 변환 불필요
                keywords_with_definitions.append({
                    "word": word,
                    "pos": pos_tag,
                    "definitions": senses, # 이미 형식화된 리스트
                })
        
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

    except HTTPException as e:
        # FastAPI HTTPException은 여기서 다시 발생시킵니다.
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] API 처리 중 에러: {e}")
        total_processing_time = time.time() - start_total
        print(f"[Timing] --- Request Failed --- Total time: {total_processing_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"API 처리 중 예상치 못한 에러가 발생했습니다: {e}")
