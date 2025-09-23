FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Uninstall existing python versions (if present)
RUN apt-get update && \
    apt-get remove -y python3 python3.8 python3.9 python3.11 || true

# Install Python 3.10 and essential utilities
RUN apt-get install -y python3.10 python3-pip git wget openssh-server libgl1-mesa-glx ffmpeg

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
# Upgrade pip explicitly for Python 3.10
RUN python -m pip install --upgrade pip

# Print Python version
RUN python --version

# Install PyTorch for CUDA 11.8
RUN pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# Explicitly install Python libraries
RUN pip install diffusers==0.27.2 \
    accelerate==0.28.0 \
    numpy==1.23.5 \
    opencv-python==4.9.0.80 \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    huggingface_hub==0.25.2 \
    librosa==0.10.1 \
    einops==0.7.0 \
    gdown \
    "imageio[ffmpeg]" \
    omegaconf==2.3.0 \
    ffmpeg-python==0.2.0 \
    moviepy==1.0.3 \
    tqdm==4.66.3 \
    requests==2.32.2 \
    peft==0.10.0

# Install mmlab/openmim libs
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# SSH Setup
RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 22

ENTRYPOINT ["/startup.sh"]
