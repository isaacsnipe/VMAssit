FROM python:3.9

WORKDIR /code

COPY poetry.lock pyproject.toml .

RUN apt-get update

COPY requirement.txt .

RUN pip install -r requirement.txt

RUN poetry install
COPY . .

EXPOSE 8083

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
