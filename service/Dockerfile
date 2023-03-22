FROM fedora:latest

WORKDIR /home

COPY . /home

RUN dnf install -y pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20001", "server:app"]
