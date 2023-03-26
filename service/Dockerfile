FROM fedora:latest

WORKDIR /home

COPY . /home

RUN dnf install -y pip && \
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

CMD /home/start.sh
