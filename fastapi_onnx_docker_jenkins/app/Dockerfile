FROM python:3.10-slim
WORKDIR /work
COPY . /work
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
CMD ["uvicorn","inference:app","--port:8080","--reload"]
