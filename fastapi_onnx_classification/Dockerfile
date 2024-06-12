FROM amd64/python:3.9-slim

WORKDIR /work

COPY . /work

RUN pip install fastapi[all] \
    && uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]