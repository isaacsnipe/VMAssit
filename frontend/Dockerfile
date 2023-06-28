FROM python:3.9

WORKDIR /usr/src/app

COPY poetry.lock pyproject.toml .

RUN apt-get update

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY . .

EXPOSE 8083

CMD ["streamlit", "run", "app.py"]
