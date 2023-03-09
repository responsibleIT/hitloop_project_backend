FROM python:3.9

COPY ./ /app

WORKDIR  /app

RUN pip3 install -r requirements.txt

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "wsgi:app"]