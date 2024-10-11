# Python 3.8의 기본 베이스 이미지를 사용
FROM python:3.10-slim

# 컨테이너에서 작업할 디렉토리를 설정
WORKDIR /app

# 필요한 애플리케이션 코드를 복사
COPY . /app

# 라이브러리 설치를 하지 않음: 노드의 hostPath에서 라이브러리를 마운트해서 사용
# 필요한 경우 기본 패키지만 설치할 수 있음. 예:
# RUN pip install --no-cache-dir some-package

# 실행 명령어: hostPath로부터 마운트된 가상환경을 활성화하고 스크립트를 실행
CMD ["/bin/bash", "-c", "source /opt/libs/bin/activate && python-master.py"]
