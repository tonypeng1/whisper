FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 81
ENTRYPOINT ["streamlit", "run", "trans_real_time.py", "--server.port=81", "--server.address=0.0.0.0"]