# main.py

import os
from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import asyncio
import uuid
import re
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# Google Cloud Text-to-Speech 관련 임포트
from google.cloud import texttospeech

# g2pk 및 okt-korean 관련 임포트
from g2pk import G2pKo # G2KoKoreanRomanizer 대신 G2pKo 임포트
from okt import Okt

# NLTK 데이터 다운로드 (render 배포 시 필요)
import nltk
try:
    nltk.data.find('corpora/cmudict')
except nltk.downloader.DownloadError:
    nltk.download('cmudict')


# --- 환경 변수 설정 ---
# Render에 배포 시 환경 변수로 GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY, KOREAN_DICT_API_KEY 설정
# 로컬 개발 시에는 .env 파일 사용 또는 직접 설정
try:
    # GOOGLE_APPLICATION_CREDENTIALS는 JSON 문자열로 제공될 것으로 가정
    google_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_credentials_json:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

    # JSON 문자열을 임시 파일로 저장
    temp_credentials_path = f"/tmp/credentials_{uuid.uuid4().hex}.json"
    with open(temp_credentials_path, "w") as f:
        f.write(google_credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
    print(f"[INFO] 서비스 계정 JSON 임시 파일 생성: {temp_credentials_path}")

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    KOREAN_DICT_API_KEY = os.environ.get("KOREAN_DICT_API_KEY")
    if not KOREAN_DICT_API_KEY:
        raise ValueError("KOREAN_DICT_API_KEY 환경 변수가 설정되지 않았습니다.")

except ValueError as e:
    print(f"[CRITICAL ERROR] 환경 변수 설정 오류: {e}")
    # 프로덕션 환경에서는 앱 시작을 중단하거나 적절히 처리해야 함
    # 개발 환경에서는 .env 파일 사용을 고려
except Exception as e:
    print(f"[CRITICAL ERROR] 환경 변수 처리 중 예외 발생: {e}")
    import traceback
    traceback.print_exc()


# --- FastAPI 앱 초기화 ---
app = FastAPI()

# CORS 설정: 모든 Origin 허용 (개발 목적)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 서비스에서는 특정 도메인으로 제한하는 것이 좋음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global 인스턴스 (필요에 따라) ---
# Okt 인스턴스는 한 번만 생성하는 것이 효율적
okt = Okt()
romanizer = G2pKo() # G2KoKoreanRomanizer() 대신 G2pKo()로 인스턴스화
# ThreadPoolExecutor for blocking I/O operations (like Google TTS)
executor = ThreadPoolExecutor(max_workers=5) # 적절한 워커 수 설정

# TTS 출력 디렉토리
TTS_OUTPUT_DIR = "tts_output"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# --- 유틸리티 함수 (asyncio.to_thread 사용) ---
def to_thread(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

@to_thread
def generate_tts_to_file_sync(text: str) -> str :
    """
    Google Cloud TTS를 사용하여 text를 mp3로 합성한 뒤,
    TTS_OUTPUT_DIR에 저장하고, 해당 파일 경로를 반환합니다.
    실패 시 None 반환. (to_thread를 통해 동기적으로 실행)
    """
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # 음성 설정 (한국어, 남성 목소리, 뉴럴넷)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Neural2-B", # 또는 ko-KR-Wavenet-B, ko-KR-Standard-C 등
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

        # 오디오 인코딩 (MP3)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # 고유한 파일명 생성 (UUID 사용)
        file_name = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(TTS_OUTPUT_DIR, file_name)

        # MP3 파일 저장
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        print(f"[TTS] 오디오 콘텐츠가 '{output_path}'에 저장되었습니다.")
        return output_path

    except Exception as e:
        import traceback
        print("[generate_tts_to_file] Google TTS 예외 전체 로그:")
        traceback.print_exc()
        return None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    API 서버의 상태를 확인하는 기본 엔드포인트.
    """
    return {"message": "SimpleTalk API 서버 작동 중입니다."}

@app.post("/translate-to-easy-korean")
async def translate_to_easy_korean(request: Request):
    start_time = asyncio.get_event_loop().time()
    print("[Timing] --- New Request Received ---")

    try:
        data = await request.json()
        input_text = data.get("text")
        if not input_text:
            raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")

        print(f"[Timing] Input text: '{input_text}'")

        # 1. OpenAI GPT-4o-mini 호출
        openai_start_time = asyncio.get_event_loop().time()
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """
                        당신은 한국어 문장을 단순하게 바꾸는 전문가입니다.
                        입력된 문장은 다음을 중복 포함할 수 있습니다:
                        1. 속담 또는 관용어
                        2. 방언(사투리)
                        3. 어려운 단어
                        4. 줄임말

                        각 항목에 대해 다음과 같이 변환하세요:
                        - 속담/관용어는 그 뜻을 자연스럽게 문장에 맞게 설명하세요.
                        예시) 입력: 배가 불렀네? / 출력: 지금 가진 걸 당연하게 생각하는 거야?
                        예시) 입력: 손이 크다 / 출력: 씀씀이가 후하다.
                        - 방언은 표준어로 바꾸세요.
                        예시) 입력: 니 오늘 뭐하노? / 출력: 너 오늘 뭐 해?
                        예시) 입력: 정구지 / 출력: 부추
                        - 어려운 단어는 초등학교 1~2학년이 이해할 수 있는 쉬운 말로 바꾸세요.
                        예시) 입력: 당신의 요청은 거절되었습니다. 추가 서류를 제출하세요. / 출력: 당신의 요청은 안 됩니다. 서류를 더 내야 합니다.
                        - 줄임말은 풀어 쓴 문장으로 바꾸세요.
                        예시) 입력: 할많하않 / 출력: 할 말은 많지만 하지 않겠어

                        다음은 반드시 지켜주세요:
                        - 변환된 문장 또는 단어만 출력하세요.
                        - 설명을 덧붙이지 마세요.
                        - 의문문이 들어오면, 절대 대답하지 마세요.
                          질문 형태를 그대로 유지하면서 쉬운 단어로 바꾸세요.
                          예시) 입력: 국무총리는 어떻게 임명돼? / 출력: 국무총리는 어떻게 정해?
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                        다음 한국어 문장을 쉬운 한국어로 번역하세요.
                        또한 원문과 쉬운 한국어 문장에 대해 국어의 로마자 표기법에 따라 로마자 발음을 제공하세요.
                        그 다음, 쉬운 한국어 문장의 영어 번역을 제공하세요.
                        마지막으로, 원문 문장에서 중요한 키워드를 추출하고 각 키워드에 대한 한국어 사전 정의(한국어 및 영어 번역)와 품사를 제공하세요.
                        단순화할 때는 원래 의미를 가능한 한 유지하되, 한국어 학습자에게 적합한 더 쉬운 어휘와 문장 구조를 사용하세요.
                        출력은 지정된 JSON 형식으로 엄격하게 준수해야 합니다.

                        원문 한국어 문장: "{input_text}"

                        출력은 JSON 형식이어야 합니다:
                        {{
                            "original_text": "원문 한국어 문장",
                            "original_romanized_pronunciation": "원문 한국어 로마자 발음",
                            "translated_text": "쉬운 한국어 문장",
                            "translated_romanized_pronunciation": "쉬운 한국어 로마자 발음",
                            "translated_english_translation": "쉬운 한국어 영어 번역",
                            "keyword_dictionary": [
                                {{
                                    "word": "단어",
                                    "pos": "품사 (예: 명사, 동사)",
                                    "definitions": [
                                        {{"definition": "한국어 뜻풀이", "english_translation": "English meaning"}},
                                        // ... 추가 정의
                                    ]
                                }}
                                // ... 추가 단어
                            ]
                        }}
                    """
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.5 # 창의성 조절
        }

        async with httpx.AsyncClient(timeout=120.0) as client: # 타임아웃 120초로 설정
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생

        openai_end_time = asyncio.get_event_loop().time()
        print(f"[Timing] 1. OpenAI GPT-4o-mini call: {openai_end_time - openai_start_time:.4f}s")


        openai_data = response.json()
        model_output_json_str = openai_data["choices"][0]["message"]["content"]
        # print("OpenAI Raw Output:", model_output_json_str) # 디버깅용

        parsed_data = json.loads(model_output_json_str)

        # 2. 로마자 발음 및 한국어 형태소 분석/영어 번역 병렬 처리
        parallel_tasks_start_time = asyncio.get_event_loop().time()

        # 로마자 발음 (원본 및 번역된 문장)
        original_romanized_pronunciation = romanizer.romanize(parsed_data["original_text"])
        translated_romanized_pronunciation = romanizer.romanize(parsed_data["translated_text"])

        # 키워드 사전 정보 보강 (Okt 및 KOREAN_DICT_API_KEY 사용)
        keyword_dictionary = []
        if "keyword_dictionary" in parsed_data and isinstance(parsed_data["keyword_dictionary"], list):
            keywords_to_process = []
            for item in parsed_data["keyword_dictionary"]:
                word = item.get("word")
                pos = item.get("pos")
                if word and pos:
                    keywords_to_process.append({"word": word, "pos": pos})

            # 모든 키워드에 대해 병렬로 사전 검색 요청
            dictionary_tasks = [get_korean_dictionary_entry(kw["word"], kw["pos"]) for kw in keywords_to_process]
            dictionary_results = await asyncio.gather(*dictionary_tasks)

            total_dict_api_time = 0
            for i, result in enumerate(dictionary_results):
                if result:
                    keyword_dictionary.append(result)
                    total_dict_api_time += result.get("api_call_time", 0) # API 호출 시간 누적
                    print(f"[Timing] Single Dictionary call for '{result['word']}': {result['api_call_time']:.4f}s (results: {len(result['definitions'])})")

            print(f"[Timing] 3. Total Dictionary API calls for {len(keyword_dictionary)} keywords: {total_dict_api_time:.4f}s")


        parallel_tasks_end_time = asyncio.get_event_loop().time()
        print(f"[Timing] 2. Parallel tasks (Romanization, Google Translate, Okt): {parallel_tasks_end_time - parallel_tasks_start_time:.4f}s")


        response_data = {
            "original_text": parsed_data["original_text"],
            "original_romanized_pronunciation": original_romanized_pronunciation,
            "translated_text": parsed_data["translated_text"],
            "translated_romanized_pronunciation": translated_romanized_pronunciation,
            "translated_english_translation": parsed_data["translated_english_translation"],
            "keyword_dictionary": keyword_dictionary,
        }

        end_time = asyncio.get_event_loop().time()
        print(f"[Timing] --- Request Processed --- Total time: {end_time - start_time:.4f}s")

        return JSONResponse(content=response_data)

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTPStatusError from OpenAI: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI API 오류: {e.response.text}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 디코딩 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 응답 파싱 오류 또는 OpenAI 응답 형식이 올바르지 않습니다.")
    except Exception as e:
        import traceback
        print(f"[ERROR] /translate-to-easy-korean 처리 중 예외 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@app.post("/speak")
async def speak(request: Request):
    try:
        # request.form() 대신 request.body()로 직접 raw body를 읽음
        body = await request.body()
        # body는 byte string이므로 디코딩하고 'text=' 접두사 제거 및 URL 디코딩
        body_str = body.decode('utf-8')
        if not body_str.startswith("text="):
            raise HTTPException(status_code=400, detail="잘못된 요청 형식입니다. 'text' 파라미터가 필요합니다.")
        text_to_speak = httpx.URL(f"http://example.com/?{body_str}").params.get("text")

        if not text_to_speak:
            raise HTTPException(status_code=400, detail="재생할 텍스트가 없습니다.")

        # generate_tts_to_file_sync 함수는 이미 @to_thread 데코레이터가 적용되어 있습니다.
        tts_file_path = await generate_tts_to_file_sync(text_to_speak)

        if tts_file_path:
            # TTS_OUTPUT_DIR에서 파일 경로를 얻기 위해 os.path.basename 사용
            file_name = os.path.basename(tts_file_path)
            # 클라이언트에게는 파일의 직접적인 다운로드 URL을 제공
            # Render의 경우, '/tts_audio/{file_name}' 과 같은 경로로 접근 가능하도록 설정
            return JSONResponse(content={"tts_url": f"/tts_audio/{file_name}"})
        else:
            raise HTTPException(status_code=503, detail="TTS 서버 일시 장애로 음성 생성 실패")

    except Exception as e:
        import traceback
        print(f"[ERROR] /speak 처리 중 예외 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"음성 생성 중 서버 내부 오류: {e}")


@app.get("/tts_audio/{file_name}")
async def get_tts_audio(file_name: str):
    """
    생성된 TTS 오디오 파일을 제공하는 엔드포인트.
    """
    file_path = os.path.join(TTS_OUTPUT_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(file_path, media_type="audio/mpeg")


# --- 도우미 함수 (사전 API 호출) ---
async def get_korean_dictionary_entry(word: str, pos: str) -> Union[Dict, None]:
    start_time = asyncio.get_event_loop().time()
    try:
        if not KOREAN_DICT_API_KEY:
            print("[WARN] KOREAN_DICT_API_KEY가 없어 사전 검색을 건너뜝니다.")
            return None

        # TODO: 실제 사전 API 응답 파싱 및 반환 로직 구현
        # 현재 코드의 흐름상, OpenAI에서 키워드와 정의를 받아오므로
        # 이 함수는 필요 없을 수 있습니다.
        # 만약 더 상세한 정의를 위해 외부 사전 API를 호출하려면 여기에 구현.
        # 지금은 해당 API 호출이 없으므로 항상 None을 반환하거나, OpenAI 응답을 그대로 사용해야 합니다.

        # 임시로 더미 데이터를 반환하거나 None 반환
        return {"word": word, "pos": pos, "definitions": [{"definition": "사전 뜻풀이 (API 미구현)", "english_translation": "Dictionary meaning (API not implemented)"}], "api_call_time": asyncio.get_event_loop().time() - start_time}

    except Exception as e:
        print(f"[ERROR] 한국어 사전 API 호출 중 예외 발생: {e}")
        return None
