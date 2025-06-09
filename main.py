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
from google.cloud import texttospeech # 이 줄을 추가합니다.

# g2pk 및 okt-korean 관련 임포트 (설치 필요)
from g2pk import G2KoKoreanRomanizer
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
romanizer = G2KoKoreanRomanizer()
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
        tts_client = texttospeech.TextToSpeechClient() # 여기에 TextToSpeechClient 초기화
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
            "model": "gpt-4o-mini", # 또는 gpt-3.5-turbo-0125
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that translates Korean sentences into simpler Korean, provides romanized pronunciation for both original and simplified Korean, and gives an English translation for the simplified Korean. Additionally, you will extract important keywords from the original sentence and provide a dictionary definition in Korean for each keyword, along with its English translation and part of speech. When simplifying, try to maintain the original meaning as much as possible but use simpler vocabulary and sentence structures suitable for Korean learners. Ensure the output is strictly in JSON format as specified."
                },
                {
                    "role": "user",
                    "content": f"""
                        Translate the following Korean sentence into simpler Korean.
                        Also provide romanized pronunciation for the original and simplified Korean using the Revised Romanization of Korean system.
                        Then, provide an English translation for the simplified Korean.
                        Finally, extract important keywords from the original sentence and provide a dictionary definition (Korean and English translation) and part of speech for each keyword.

                        Original Korean sentence: "{input_text}"

                        Output must be in JSON format:
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

        # Google 사전 API를 가정
        # 실제 사용하려는 사전 API의 엔드포인트와 파라미터에 맞게 수정 필요
        # 예시: '단어' 검색 시
        # https://www.google.com/search?q=define+한국어+단어
        # 이 예시에서는 실제 동작하는 API가 아님. 실제 사전 API를 사용해야 함.
        # NIKL (국립국어원) 한국어 지식대사전 API 사용 가능 (API 키 필요)
        # NIKL API 예시 (실제 API 명세에 따라 달라질 수 있음):
        # https://stdict.korean.go.kr/api/search.do?key={API_KEY}&q={word}

        async with httpx.AsyncClient(timeout=10.0) as client:
            # 여기는 KOREAN_DICT_API_KEY를 사용하는 실제 사전 API 엔드포인트여야 합니다.
            # 이 코드는 예시일 뿐, 실제 사전 API 명세를 따르지 않습니다.
            # 임시로 API 호출을 시뮬레이션하거나, 실제 유효한 사전 API로 대체해야 합니다.
            # 현재 코드에서 실제 외부 사전 API 호출 로직은 구현되어 있지 않고,
            # OpenAI가 제공하는 "keyword_dictionary"를 그대로 사용하고 있습니다.
            # 따라서 이 함수는 현재 작동하지 않을 가능성이 높습니다.
            # 만약 KOREAN_DICT_API_KEY가 실제로 외부 사전 API를 호출하는 용도라면
            # 여기에 해당 API 호출 로직을 구현해야 합니다.
            # 지금은 OpenAI가 제공한 키워드 사전만 사용한다고 가정하고 이 함수는
            # 더미 데이터를 반환하거나 OpenAI 응답을 그대로 사용하도록 처리해야 합니다.

            # 현재 코드의 흐름상, OpenAI에서 키워드와 정의를 받아오므로
            # 이 함수는 필요 없을 수 있습니다.
            # 만약 더 상세한 정의를 위해 외부 사전 API를 호출하려면 여기에 구현.
            # 지금은 해당 API 호출이 없으므로 항상 None을 반환하거나, OpenAI 응답을 그대로 사용해야 합니다.

            # 이 함수는 현재 사용되지 않거나, OpenAI 응답에서 받은 데이터를
            # 그대로 반환하도록 수정해야 할 가능성이 있습니다.
            # 예시: OpenAI가 제공한 dictionary 데이터를 그대로 반환
            # return {"word": word, "pos": pos, "definitions": [{"definition": "뜻풀이 (OpenAI)", "english_translation": "meaning (OpenAI)"}]}
            pass # 실제 사전 API 호출 로직 없음


        # TODO: 실제 사전 API 응답 파싱 및 반환 로직 구현
        # 현재는 이 함수가 사용되지 않거나, OpenAI 응답에서 받은 데이터를 그대로
        # 반환하도록 변경해야 할 가능성이 높습니다.
        # 또는 NIKL API 등을 사용하여 추가 정보 가져오기.
        # 이 부분은 주석 처리되어 있거나, 의미 없는 API 호출이 되어 있습니다.
        # OpenAI가 이미 키워드 사전을 제공하므로,
        # 추가적인 외부 사전 API 호출이 필요하지 않다면 이 함수는 삭제해도 됩니다.

        # 임시로 더미 데이터를 반환하거나 None 반환
        return {"word": word, "pos": pos, "definitions": [{"definition": "사전 뜻풀이 (API 미구현)", "english_translation": "Dictionary meaning (API not implemented)"}], "api_call_time": asyncio.get_event_loop().time() - start_time}

    except Exception as e:
        print(f"[ERROR] 한국어 사전 API 호출 중 예외 발생: {e}")
        return None
