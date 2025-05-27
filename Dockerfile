FROM python:3.9

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

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
