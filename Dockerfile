# Dockerfile
FROM python:3.10
ENV DEBIAN_FRONTEND=noninteractive

# apt 캐시 업데이트 및 초기 정리 (재시도 로직 유지)
RUN apt-get update -y || (sleep 5 && apt-get update -y) || (sleep 5 && apt-get update -y) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 패키지 저장소 미러 변경 (Debian 공식 미러 사용)
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list.d/debian.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list.d/debian.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list.d/debian.list && \
    apt-get update -y || (sleep 5 && apt-get update -y) || (sleep 5 && apt-get update -y)

# OpenJDK 17 설치
RUN apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java || \
    (sleep 5 && apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java) || \
    (sleep 5 && apt-get install -y --no-install-recommends openjdk-17-jdk ca-certificates-java)

# 설치 후 불필요한 패키지 제거 및 캐시 정리
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

# 작업 디렉토리 설정
WORKDIR /app

# TTS 파일을 저장할 디렉토리 생성
RUN mkdir -p tts_files

# Python 의존성 설치 (매우 중요: 설치 순서를 변경)
COPY requirements.txt .

# 1단계: konlpy와 jpype1을 제외한 나머지 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 2단계: jpype1 먼저 설치 (jpype1이 konlpy보다 먼저 설치되어야 함)
RUN pip install --no-cache-dir jpype1

# 3단계: konlpy를 강제 재설치 (jpype1이 설치된 환경에서 konlpy의 Java 컴포넌트 재구성)
RUN pip install --no-cache-dir --upgrade --force-reinstall konlpy

COPY . .

# FastAPI 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 애플리케이션 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
