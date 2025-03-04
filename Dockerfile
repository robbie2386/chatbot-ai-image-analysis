
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["gunicorn", "-w 4", "-b 0.0.0.0:8080", "app:app"]
    