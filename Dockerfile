FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "app:app", "--reload"]