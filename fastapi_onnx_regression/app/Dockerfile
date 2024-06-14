FROM python3.8-slim
WORKDIR ./work
COPY . ./work

RUN pip install --upgrade pip && pip install -r requirements.txt \\
    && pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

CMD ["uvicorn", "main:app", "--host", "0.0.0.0","--port","8000" "--reload"]