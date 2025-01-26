# Introduction

This is a real-time Automatic Speech Recognition (ASR) app that uses the OpenAI whisper model downloaded from 🤗 Hugging Face. The model is saved and reused locally afterward.

A short Youtube video showing how it works can be seen by clicking the image below.

[![Watch the video](https://github.com/user-attachments/assets/aab1a0bd-9a19-4c14-8f7e-51c8d9dc002b)](https://youtu.be/4lKOeVDbvDI)

# App Features

This app (version 1.1.0) currently has the following features.

1. Check if MPS (Metal Performance Shaders) or CUDA is available on the current system for GPU acceleration.
2. Download the Whisper model and processor from 🤗 Hugging Face or load it from a local folder if the model has been downloaded before.
3. Use the Streamlit `audio_input` widget to record the user's speech in English as an `.wav` audio file. The speech is limited to a clip of less than 30 seconds.
4. Covert the `.wav` audio file into a list that contains a single dictionary with the processed audio data (a numpy array) and the sampling rate (= 16,000) to match the format of 🤗 Hugging Face `datasets`.
5. Transcribe the speech using the OpenAI `openai/whisper-small.en` model stored locally. No OpenAI API key is required for transcription.
6. Convert the code into a Docker image using the associated `Dockerfile` file and `compose.yml` file.

The `openai/whisper-small.en` model is chosen as a tradeoff between computing needs, latency, and accuracy for future model fine-tuning.

# Python Dependencies

The `requirements.txt` file is as follows.

```
transformers==4.47.0
datasets==3.2.0
librosa==0.10.2.post1
torch==2.2.2
numpy==1.26.4  
sounddevice==0.5.1
streamlit==1.41.1
```

There is a need to downgrade numpy version to 1.26.4 to avoid errors with `PyTorch` (`torch`) by typing the following commands in a terminal window.

```
pip uninstall numpy
pip install "numpy<2"
```


# Clone the GitHub Repository and APP Installation

To clone the GitHub directory type the command as follows.
```
git clone https://github.com/tonypeng1/whisper.git
```
To create a Python virtual environment, check out version 2.2 of this APP, and install the project,
```
cd whisper
python3 -m venv .venv
source .venv/bin/activate
git checkout v1.0
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

# Docker Image

The `Dockerfile` to create a Docker image is as follows. The reason for copying `requirements.txt` and installing the dependencies before copying the rest of the application files is that if the requirements haven't changed, Docker can reuse the cached version instead of rebuilding it. 

```
FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 81
ENTRYPOINT ["streamlit", "run", "trans_real_time.py", "--server.port=81", "--server.address=0.0.0.0"
```

The `compose.yml` file is as follows.

```
services:
  app:
    container_name: realtime_transcription_app
    image: realtime-transcription-app:1.1.0
    ports:
      - "81:81"
```

To build and run the Docker image, you can type the following commands in a terminal window.

```
docker build -t realtime-transcription-app:1.1.0 .
docker compose up
```

# 📝 Note

The app can also be run directly without using a Docker image by typing the command below in a terminal window.

```
streamlit run trans_real_time.py
```