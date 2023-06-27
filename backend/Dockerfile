FROM python:3.9

WORKDIR /code

COPY poetry.lock pyproject.toml ./

RUN apt-get update && apt-get install -y unzip && \
    pip install gdown && \
    gdown --id 1V5oKpkHoKg05tninoIfCRFE1S_DcPDU5 && \
    unzip vla_weights.zip -d ./core/ && \
    rm vla_weights.zip

RUN pip install --no-cache-dir poetry==1.4.1 && \
    poetry config virtualenvs.create false && \
    poetry config experimental.net-installer false && \
    poetry install --no-interaction --no-ansi

COPY . .

EXPOSE 8083

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]