FROM python:3.10-slim

WORKDIR /app

COPY . /app

CMD ["python", "-u", "python-master.py"]

# CMD ["tail", "-f", "/dev/null"]

