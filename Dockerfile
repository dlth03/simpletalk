# Dockerfile
FROM python:3.10
ENV DEBIAN_FRONTEND=noninteractive

# apt 캐시 업데이트 및 초기 정리 (더 이상 openjdk 설치 필요 없으므로 간소화)
RUN apt-get update -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 더 이상 OpenJDK 설치 및 JAVA_HOME 설정 필요 없음.

# 작업 디렉토리 설정
WORKDIR /app

# TTS 파일을 저장할 디렉토리 생성
RUN mkdir -p tts_files

# Python 의존성 설치 (kiwi 포함, requirements.txt에 있는 모든 패키지 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 애플리케이션 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
