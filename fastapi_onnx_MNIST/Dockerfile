From python3.8-slim

COPY . . 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn","inference:app","port:8080"]