FROM python:3.10

# apt 캐시 업데이트
RUN apt-get update

# OpenJDK 17 설치 및 불필요한 패키지 제거, apt 캐시 정리
RUN apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# JAVA_HOME 환경 변수 설정 (경로 변경!)
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .

# TTS 파일을 저장할 디렉토리 생성 (컨테이너 내부에서 gTTS가 파일 생성할 공간)
RUN mkdir -p tts_files
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade --force-reinstall konlpy jpype1 # 추가 부분

COPY . .

# FastAPI 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 애플리케이션 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
