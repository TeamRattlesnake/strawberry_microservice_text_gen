FROM fedora:latest

WORKDIR /home

COPY . /home

RUN dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm && \
    dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm && \
    dnf update -y --refresh && \
    dnf install -y pip xorg-x11-drv-nvidia xorg-x11-drv-nvidia-cuda && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20001", "server:app"]
