# Dockerfile
FROM python:3.10

# apt 캐시 업데이트 및 초기 정리 (더 견고하게)
# apt-get update 실패 시 재시도 로직 추가
RUN apt-get update -y || (sleep 5 && apt-get update -y) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# OpenJDK 17 설치 및 불필요한 패키지 제거, apt 캐시 정리
# 설치 실패 시 재시도 로직 추가
RUN apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java || \
    (sleep 5 && apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

# 작업 디렉토리 설정
WORKDIR /app

# 1단계: requirements.txt의 모든 의존성을 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2단계: konlpy와 jpype1을 명시적으로 강제 재설치
RUN pip install --no-cache-dir --upgrade --force-reinstall konlpy jpype1

# TTS 파일을 저장할 디렉토리 생성
RUN mkdir -p tts_files

COPY . .

# FastAPI 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 애플리케이션 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
