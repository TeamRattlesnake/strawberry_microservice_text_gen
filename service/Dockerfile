FROM fedora:latest

WORKDIR /home

COPY . /home

RUN dnf install -y pip && \
    pip install -r requirements.txt

CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20000", "server:app"]
