FROM python:3.12

ENV PYTHONPATH=/opt/http
WORKDIR /opt/http

COPY pyproject.toml web_api/requirements.txt /opt/http/
RUN apt update && apt install -y ffmpeg libsm6 libxext6 unixodbc-dev libboost-dev
RUN pip install --no-cache-dir -r requirements.txt && pip install '.[modules]'

COPY . /opt/http
RUN zetta --help

WORKDIR /opt/http/web_api
CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:80"]
