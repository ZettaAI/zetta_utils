FROM python:3.10

ENV PYTHONPATH /opt/http
WORKDIR /opt/http

COPY pyproject.toml web_api/requirements.txt /opt/http/
RUN apt update && apt install ffmpeg libsm6 libxext6 software-properties-common unixodbc-dev libboost-dev -y
RUN pip install --no-cache-dir -r requirements.txt && pip install '.[modules]'

COPY . /opt/http
RUN zetta --help

CMD ["uvicorn", "app.main:app", "--app-dir", "./web_api" "--host", "0.0.0.0", "--port", "80"]
